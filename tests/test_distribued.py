import logging
import asyncio
import json
from uuid import UUID

import redis.asyncio as redis
from redis import Redis
import zmq
import zmq.asyncio

from pydantic import UUID4
import pytest

from dranspose.mapping import MappingSequence
from dranspose.helpers.utils import parameters_hash, done_callback, cancel_and_wait

from dranspose.protocol import (
    GENERIC_WORKER,
    ControllerUpdate,
    RedisKeys,
    ParameterName,
    WorkParameter,
    WorkerName,
    IngesterState,
    ConnectedWorker,
    StreamName,
)

from dranspose.distributed import DistributedService, DistributedSettings
from dranspose.worker import WorkerState, Worker, WorkerSettings

from dranspose.ingester import IngesterSettings

from dranspose.protocol import IngesterName


class DummyWorker(DistributedService):
    def __init__(self, name: WorkerName):
        self._worker_settings = DistributedSettings()  # type: ignore[call-arg]

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
            async with rds.pipeline() as pipe:  # type: ignore[attr-defined]
                for name, param in parameters.items():
                    await pipe.set(RedisKeys.parameters(name, param.uuid), param.data)
                await pipe.execute()


async def get_published_worker_congig(rds: Redis) -> list[str]:
    async with rds.pipeline() as pipe:  # type: ignore[attr-defined]
        await pipe.keys(RedisKeys.config("worker"))
        worker_keys = await pipe.execute()
    async with rds.pipeline() as pipe:  # type: ignore[attr-defined]
        await pipe.mget(worker_keys[0])
        worker_json = await pipe.execute()
    return worker_json[0]


@pytest.mark.asyncio
async def test_distributed() -> None:
    dummy = DummyWorker(WorkerName("dummy"))

    assert dummy._distributed_settings is not None
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

    # what if we publish a map?
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


class IngestOnlyWorker(Worker):
    async def run(self) -> None:
        self.manage_ingester_task = asyncio.create_task(self.manage_ingesters())
        self.manage_ingester_task.add_done_callback(done_callback)
        await self.register()

    async def restart_work(self, uuid: UUID4, active_streams: list[StreamName]) -> None:
        pass

    async def close(self) -> None:
        await cancel_and_wait(self.manage_ingester_task)
        if self.dequeue_task is not None:
            await cancel_and_wait(self.dequeue_task)
        await self.redis.delete(RedisKeys.config("worker", self.state.name))
        # await super().close() # this will try to cancel non existing coro
        # I'll try to do the cleanup by hand
        await self.redis.aclose()
        await self.raw_redis.aclose()
        self.ctx.destroy()
        self._logger.info("worker closed")


@pytest.mark.asyncio
async def test_ingest_only_worker() -> None:
    dummy = IngestOnlyWorker(WorkerSettings(worker_name=WorkerName("dummy")))  # type: ignore[call-arg]

    assert dummy._distributed_settings is not None
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
    for w in worker_json:
        assert json.loads(w)["name"] == "dummy"
        assert json.loads(w)["parameters_hash"] is None

    # what if we publish a new map?
    await publish_controller_update(rds)
    await asyncio.sleep(1.5)
    # now the dummy parameters should be {} and have a corresponding hash
    worker_json = await get_published_worker_congig(rds)
    assert json.loads(worker_json[0])["parameters_hash"] == parameters_hash({})

    # Create a state/settings for our fake ingester
    # iport = 10000
    isettings = IngesterSettings(
        ingester_name=IngesterName("fake_ingester"),
        ingester_streams=[StreamName("fake_stream")],
        # ingester_url = f"tcp://localhost:{iport}", # just use the default
    )  # type: ignore[call-arg]
    istate = IngesterState(
        name=isettings.ingester_name,
        url=isettings.ingester_url,
        streams=isettings.ingester_streams,
    )

    # create a dummy ingester with a ROUTER socket
    ctx = zmq.asyncio.Context()
    out_socket = ctx.socket(zmq.ROUTER)
    out_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
    out_socket.bind(f"tcp://*:{isettings.ingester_url.port}")

    # publish ingester key/state
    # this is what the worker looks for:
    # configs = await self.redis.keys(RedisKeys.config("ingester"))
    # who publishes it, the controller or the ingester?
    # I think it's the ingester with publish_config()
    async with rds.pipeline() as pipe:  # type: ignore[attr-defined]
        await pipe.setex(
            RedisKeys.config("ingester", istate.name),
            10,
            istate.model_dump_json(),
        )
        await pipe.execute()

    await asyncio.sleep(1.5)

    # accept the worker
    # verify that worker has sent its identity to the ingester
    poller = zmq.asyncio.Poller()
    poller.register(out_socket, zmq.POLLIN)
    socks = dict(await poller.poll(timeout=1))
    for sock in socks:
        data = await sock.recv_multipart()
        connected_worker = ConnectedWorker(
            name=data[0], service_uuid=UUID(bytes=data[1])
        )
        logging.info("worker pinnged %s", connected_worker)

    # verify that worker has picked up the ingester, how?
    logging.info(f"{dummy._ingesters=}")
    assert len(dummy._ingesters) > 0

    # one can also look at the status published on redis

    # change service_uuid and verify that worker drops ingester

    await dummy.close()
    await cleanup_redis(rds)
    ctx.destroy(linger=0)
    # leave time to the redis poll task to update
    await asyncio.sleep(0.2)
