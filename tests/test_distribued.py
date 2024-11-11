import asyncio
import json

import pytest
from dranspose.mapping import MappingSequence
from dranspose.helpers.utils import parameters_hash

from dranspose.protocol import (
    GENERIC_WORKER,
    ControllerUpdate,
    RedisKeys,
    ParameterName,
    WorkParameter,
    WorkerName,
)
from dranspose.helpers.utils import done_callback

import logging
import redis.asyncio as redis
from redis import Redis

from dranspose.distributed import DistributedService, DistributedSettings
from dranspose.worker import WorkerState


class DummyWorker(DistributedService):
    def __init__(self, name: WorkerName):
        self._worker_settings = DistributedSettings()

        state = WorkerState(
            name=name,
            tags={GENERIC_WORKER},
        )
        super().__init__(state, self._worker_settings)
        self._logger.info("created worker with state %s", state)

    async def run(self) -> None:
        await self.register()


async def cleanup_redis(rds: Redis) -> None:
    await rds.delete(RedisKeys.updates())
    queues = await rds.keys(RedisKeys.ready("*"))
    if len(queues) > 0:
        await rds.delete(*queues)
    assigned = await rds.keys(RedisKeys.assigned("*"))
    if len(assigned) > 0:
        await rds.delete(*assigned)
    params = await rds.keys(RedisKeys.parameters("*", "*"))
    if len(params) > 0:
        await rds.delete(*params)


async def publish_controller_update(
    rds: Redis, parameters: dict[ParameterName, WorkParameter] = {}
) -> None:
    m = MappingSequence(parts={}, sequence=[])
    cupd = ControllerUpdate(
        mapping_uuid=m.uuid,
        parameters_version={n: p.uuid for n, p in parameters.items()},
        active_streams=list(m.all_streams),
    )
    await rds.xadd(
        RedisKeys.updates(),
        {"data": cupd.model_dump_json()},
    )


async def publish_parameters(
    rds: Redis, parameters: dict[ParameterName, WorkParameter]
) -> None:
    key_names = []
    for name, param in parameters.items():
        key_names.append(RedisKeys.parameters(name, param.uuid))
    if len(key_names) > 0:
        ex_key_no = await rds.exists(*key_names)
        logging.info(
            "check for param values in redis, %d exist of %d: %s",
            ex_key_no,
            len(key_names),
            key_names,
        )
        if ex_key_no < len(key_names):
            logging.warning(
                "the redis parameters don't match the controller parameters, rewriting"
            )
            async with rds.pipeline() as pipe:
                for name, param in parameters.items():
                    await pipe.set(RedisKeys.parameters(name, param.uuid), param.data)
                await pipe.execute()


async def get_published_worker_congig(rds: Redis) -> None:
    async with rds.pipeline() as pipe:
        await pipe.keys(RedisKeys.config("worker"))
        worker_keys = await pipe.execute()
    async with rds.pipeline() as pipe:
        await pipe.mget(worker_keys[0])
        worker_json = await pipe.execute()
    return worker_json[0]


@pytest.mark.asyncio
async def test_distributed() -> None:
    dummy = DummyWorker(WorkerName("dummy"))

    rds: Redis = redis.from_url(
        f"{dummy._distributed_settings.redis_dsn}?decode_responses=True&protocol=3"
    )
    # let's clean up the DB in case a previous test failed
    await cleanup_redis(rds)

    # start the dummy worker
    t = asyncio.create_task(dummy.run())
    t.add_done_callback(done_callback)
    await asyncio.sleep(1.5)
    # test that dummy has published its config
    worker_json = await get_published_worker_congig(rds)
    assert json.loads(worker_json[0])["name"] == "dummy"
    assert json.loads(worker_json[0])["parameters_hash"] is None

    # what if we publish a new map?
    await publish_controller_update(rds)
    await asyncio.sleep(1.5)
    # now the dummy parameters should be {} and have a corresponding hash
    worker_json = await get_published_worker_congig(rds)
    assert json.loads(worker_json[0])["parameters_hash"] == parameters_hash({})

    # lets publish a new set of params
    parameters = {
        ParameterName("dummy_par"): WorkParameter(name="dummy_par", data=b"dummy")
    }
    await publish_parameters(rds, parameters)
    await publish_controller_update(rds, parameters)
    await asyncio.sleep(1.5)
    worker_json = await get_published_worker_congig(rds)
    assert json.loads(worker_json[0])["parameters_hash"] == parameters_hash(parameters)

    await dummy.close()
    await cleanup_redis(rds)

    # leave time to the redis poll task to update
    await asyncio.sleep(0.2)
