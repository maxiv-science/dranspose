import asyncio
import json
import os
from typing import Dict, List

import uvicorn
import zmq.asyncio
import logging
import time

from pydantic import BaseModel

from dranspose import protocol
from dranspose.mapping import Mapping
import redis.asyncio as redis
import redis.exceptions as rexceptions

from contextlib import asynccontextmanager
from fastapi import FastAPI

from dranspose.protocol import IngesterState, WorkerState, RedisKeys, ControllerUpdate, EnsembleState

logger = logging.getLogger(__name__)

class Controller:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True, protocol=3
        )

        self.mapping = Mapping({"":[]})
        self.completed = {}
        self.completed_events = []
        self.assign_task = None

    async def run(self):
        logger.debug("started controller run")
        self.assign_task = asyncio.create_task(self.assign_work())

    async def get_configs(self) -> EnsembleState:
        async with self.redis.pipeline() as pipe:
            await pipe.keys(RedisKeys.config("ingester"))
            await pipe.keys(RedisKeys.config("worker"))
            ingester_keys, worker_keys = await pipe.execute()
        async with self.redis.pipeline() as pipe:
            await pipe.mget(ingester_keys)
            await pipe.mget(worker_keys)
            ingester_json, worker_json = await pipe.execute()

        ingesters = [IngesterState.model_validate_json(i) for i in ingester_json]
        workers = [WorkerState.model_validate_json(w) for w in worker_json]
        return EnsembleState(ingesters=ingesters, workers=workers)

    async def set_mapping(self, m):
        self.assign_task.cancel()
        await self.redis.delete(RedisKeys.ready(self.mapping.uuid))
        await self.redis.delete(RedisKeys.assigned(self.mapping.uuid))

        # cleaned up
        self.mapping = m

        cupd = ControllerUpdate(mapping_uuid=self.mapping.uuid)
        await self.redis.xadd(
            RedisKeys.updates(),
            json.loads(cupd.model_dump_json()),
        )

        cfgs= await self.get_configs()
        while set([u.mapping_uuid for u in cfgs.ingesters]+[u.mapping_uuid for u in cfgs.workers]) != {self.mapping.uuid}:
            await asyncio.sleep(0.1)
            cfgi, cfgw = await self.get_configs()
        logger.info("new mapping with uuid %s distributed", self.mapping.uuid)
        self.assign_task = asyncio.create_task(self.assign_work())

    async def assign_work(self):
        last = 0
        event_no = 0
        start = time.perf_counter()
        while True:
            try:
                workers = await self.redis.xread(
                    {f"{protocol.PREFIX}:ready:{self.mapping.uuid}": last},
                    block=1000,
                )
                if f"{protocol.PREFIX}:ready:{self.mapping.uuid}" in workers:
                    for ready in workers[
                        f"{protocol.PREFIX}:ready:{self.mapping.uuid}"
                    ][0]:
                        logger.debug("got a ready worker %s", ready)
                        if ready[1]["state"] == "idle":
                            virt = self.mapping.assign_next(ready[1]["worker"])
                            if "new" not in ready[1]:
                                compev = int(ready[1]["completed"])
                                if compev not in self.completed:
                                    self.completed[compev] = []
                                self.completed[compev].append(ready[1]["worker"])
                                if (set([x for stream in self.mapping.get_event_workers(compev-1).values() for x in stream]) ==
                                        set(self.completed[compev])):
                                    self.completed_events.append(compev)
                            logger.debug(
                                "assigned worker %s to %s", ready[1]["worker"], virt
                            )
                            pipe = self.redis.pipeline()
                            for evn in range(event_no, self.mapping.complete_events):
                                wrks = self.mapping.get_event_workers(evn)
                                logger.debug("send out assignment %s", wrks)
                                await pipe.xadd(
                                    f"{protocol.PREFIX}:assigned:{self.mapping.uuid}",
                                    {s: json.dumps(w) for s, w in wrks.items()},
                                    id=evn + 1,
                                )
                                if evn % 1000 == 0:
                                    logger.info(
                                        "1000 events in %lf",
                                        time.perf_counter() - start,
                                    )
                                    start = time.perf_counter()
                            await pipe.execute()
                            event_no = self.mapping.complete_events
                        last = ready[0]
            except rexceptions.ConnectionError as e:
                break


    async def close(self):
        await self.redis.delete(f"{protocol.PREFIX}:controller:updates")
        queues = await self.redis.keys(f"{protocol.PREFIX}:ready:*")
        if len(queues) > 0:
            await self.redis.delete(*queues)
        assigned = await self.redis.keys(f"{protocol.PREFIX}:assigned:*")
        if len(assigned) > 0:
            await self.redis.delete(*assigned)
        await self.redis.aclose()


ctrl: Controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global ctrl
    ctrl = Controller(redis_host=os.getenv("REDIS_HOST","localhost"), redis_port=os.getenv("REDIS_PORT",6379))
    run_task = asyncio.create_task(ctrl.run())
    yield
    run_task.cancel()
    await ctrl.close()
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)


@app.get("/api/v1/config")
async def get_configs():
    return await ctrl.get_configs()

@app.get("/api/v1/status")
async def get_status():
    return {"work_completed": ctrl.completed,
            "last_assigned": ctrl.mapping.complete_events,
            "assignment": ctrl.mapping.assignments,
            "completed_events": ctrl.completed_events,
            "finished": len(ctrl.completed_events) == ctrl.mapping.len()}

@app.post("/api/v1/mapping")
async def set_mapping(mapping: Dict[str, List[List[int]|None]]):
    config = await ctrl.get_configs()
    if set(mapping.keys()) - set(config.get_streams()) != set():
        return f"streams {set(mapping.keys()) - set(config.get_streams())} not available"
    m = Mapping(mapping)
    avail_workers = await ctrl.redis.keys(f"{protocol.PREFIX}:worker:*:config")
    if len(avail_workers) < m.min_workers():
        return f"only {len(avail_workers)} workers available, but {m.min_workers()} required"
    await ctrl.set_mapping(m)
    return m.uuid
    #except Exception as e:
    #    return e.__repr__()