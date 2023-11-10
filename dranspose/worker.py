import asyncio
import time
from typing import Any

import zmq.asyncio
import logging
from dranspose import protocol
import redis.asyncio as redis
import redis.exceptions as rexceptions


class WorkerState:
    def __init__(self, name):
        self.name = name
        self.mapping_uuid = None


class Worker:
    def __init__(self, name: str, redis_host="localhost", redis_port=6379):
        self._logger = logging.getLogger(f"{__name__}+{name}")
        self.ctx = zmq.asyncio.Context()
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True, protocol=3
        )
        if ":" in name:
            raise Exception("Worker name must not container a :")
        self.state = WorkerState(name)
        self._ingesters: dict[str, Any] = {}
        self._stream_map: dict[str, zmq.Socket] = {}
        asyncio.create_task(self.register())
        asyncio.create_task(self.manage_ingesters())
        self.work_task = asyncio.create_task(self.work())
        self.eventqueue = asyncio.Queue()

    async def manage_assignments(self):
        while True:
            try:
                update = await self.redis.xread(
                    {f"{protocol.PREFIX}:controller:updates": last}, block=6000
                )
                if f"{protocol.PREFIX}:controller:updates" in update:
                    update = update[f"{protocol.PREFIX}:controller:updates"][0][-1]
                    last = update[0]
                    newuuid = update[1]["mapping_uuid"]
                    if newuuid != self.state.mapping_uuid:
                        self._logger.info("resetting config %s", newuuid)
                        self.work_task.cancel()
                        self.state.mapping_uuid = newuuid
                        self.work_task = asyncio.create_task(self.work())
            except rexceptions.ConnectionError as e:
                break

    async def work(self):
        self._logger.info("started work task")

        self._stream_map = {s: val["socket"] for ing, val in self._ingesters.items() for s in val["config"]["streams"]}
        print("stream map", self._stream_map)

        await self.redis.xadd(f"{protocol.PREFIX}:ready:{self.state.mapping_uuid}",{"idle":1, "worker": self.state.name})

        lastev = 0
        while True:
            try:
                assignments = await self.redis.xread({f"{protocol.PREFIX}:assigned:{self.state.mapping_uuid}:{stream}":lastev for stream in self._stream_map.keys()}, block=1000)
                if len(assignments) == 0:
                    continue
                self._logger.debug("got assignments %s", assignments)
                tasks = [
                    a["socket"].recv_multipart() for a in list(self._ingesters.values())
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                # print("done", done, "pending", pending)
                for res in done:
                    print("received work", res.result())
            except Exception as e:
                print("err worker ", e.__repr__())
                await asyncio.sleep(2)

    async def register(self):
        latest = await self.redis.xrevrange(
            f"{protocol.PREFIX}:controller:updates", count=1
        )
        last = 0
        if len(latest) > 0:
            last = latest[0][0]
        while True:
            await self.redis.setex(
                f"{protocol.PREFIX}:worker:{self.state.name}:present", 10, 1
            )
            await self.redis.json().set(
                f"{protocol.PREFIX}:worker:{self.state.name}:config",
                "$",
                self.state.__dict__,
            )
            try:
                update = await self.redis.xread(
                    {f"{protocol.PREFIX}:controller:updates": last}, block=6000
                )
                if f"{protocol.PREFIX}:controller:updates" in update:
                    update = update[f"{protocol.PREFIX}:controller:updates"][0][-1]
                    last = update[0]
                    newuuid = update[1]["mapping_uuid"]
                    if newuuid != self.state.mapping_uuid:
                        self._logger.info("resetting config %s", newuuid)
                        self.work_task.cancel()
                        self.state.mapping_uuid = newuuid
                        self.work_task = asyncio.create_task(self.work())
            except rexceptions.ConnectionError as e:
                break

    async def manage_ingesters(self):
        while True:
            configs = await self.redis.keys(f"{protocol.PREFIX}:ingester:*:config")
            processed = []
            for key in configs:
                cfg = await self.redis.json().get(key)
                iname = key.split(":")[2]
                processed.append(iname)
                if (
                    iname in self._ingesters
                    and self._ingesters[iname]["config"]["url"] != cfg["url"]
                ):
                    self._logger.warning(
                        "url of ingester changed from %s to %s, disconnecting",
                        self._ingesters[iname]["config"]["url"],
                        cfg["url"],
                    )
                    self._ingesters[iname]["socket"].close()
                    del self._ingesters[iname]

                if iname not in self._ingesters:
                    self._logger.info("adding new ingester %s", iname)
                    sock = self.ctx.socket(zmq.DEALER)
                    sock.setsockopt(zmq.IDENTITY, self.state.name.encode("ascii"))
                    try:
                        sock.connect(cfg["url"])
                        await sock.send(b"")
                        self._ingesters[iname] = {"config": cfg, "socket": sock}
                    except KeyError:
                        self._logger.error("invalid ingester config %s", cfg)
                        sock.close()
            for iname in set(self._ingesters.keys()) - set(processed):
                self._logger.info("removing stale ingester %s", iname)
                self._ingesters[iname]["socket"].close()
                del self._ingesters[iname]
            await asyncio.sleep(2)

    async def close(self):
        await self.redis.delete(f"{protocol.PREFIX}:worker:{self.state.name}:config")
        await self.redis.aclose()
        self.ctx.destroy()
