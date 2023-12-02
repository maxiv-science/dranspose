import asyncio
import json
import pickle
import time
from asyncio import Future
from typing import Optional

import zmq.asyncio

from pydantic import UUID4, BaseModel, ConfigDict

from dranspose.helpers import utils
import redis.exceptions as rexceptions

from dranspose.distributed import DistributedService, DistributedSettings
from dranspose.event import InternalWorkerMessage, EventData, ResultData
from dranspose.protocol import (
    WorkerState,
    RedisKeys,
    IngesterState,
    WorkerUpdate,
    WorkerStateEnum,
    WorkAssignment,
    WorkerName,
    EventNumber,
    IngesterName,
    StreamName,
    ReducerState,
    WorkerTimes,
)


class ConnectedIngester(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    socket: zmq.asyncio.Socket
    config: IngesterState


class WorkerSettings(DistributedSettings):
    worker_class: Optional[str] = None


class Worker(DistributedService):
    def __init__(self, name: WorkerName, settings: Optional[WorkerSettings] = None):
        self._worker_settings = settings
        if self._worker_settings is None:
            self._worker_settings = WorkerSettings()

        state = WorkerState(name=name)
        super().__init__(state, self._worker_settings)
        self.state: WorkerState
        self.ctx = zmq.asyncio.Context()

        self._ingesters: dict[IngesterName, ConnectedIngester] = {}
        self._stream_map: dict[StreamName, zmq._future._AsyncSocket] = {}

        self._reducer_service_uuid: Optional[UUID4] = None
        self.out_socket: Optional[zmq._future._AsyncSocket] = None

        self.custom = None
        if self._worker_settings.worker_class:
            try:
                self.custom = utils.import_class(self._worker_settings.worker_class)
                self._logger.info("custom worker class %s", self.custom)
            except Exception as e:
                self._logger.error(
                    "no custom worker class loaded, discarding events, err %s",
                    e.__repr__(),
                )

    async def run(self) -> None:
        self.manage_ingester_task = asyncio.create_task(self.manage_ingesters())
        self.manage_receiver_task = asyncio.create_task(self.manage_receiver())
        self.work_task = asyncio.create_task(self.work())
        await self.register()

    async def work(self) -> None:
        self._logger.info("started work task")

        self.worker = None
        if self.custom:
            try:
                self.worker = self.custom(self.parameters)
            except Exception as e:
                self._logger.error(
                    "Failed to instantiate custom worker: %s", e.__repr__()
                )

        await self.redis.xadd(
            RedisKeys.ready(self.state.mapping_uuid),
            {
                "data": WorkerUpdate(
                    state=WorkerStateEnum.IDLE,
                    new=True,
                    completed=EventNumber(0),
                    worker=self.state.name,
                ).model_dump_json()
            },
        )

        self._logger.info("registered ready message")

        lastev = "0"
        proced = 0
        while True:
            sub = RedisKeys.assigned(self.state.mapping_uuid)
            perf_start = time.perf_counter()
            try:
                assignments = await self.redis.xread({sub: lastev}, block=1000, count=1)
            except rexceptions.ConnectionError:
                break
            if sub not in assignments:
                continue
            assignments = assignments[sub][0][0]
            self._logger.debug("got assignments %s", assignments)
            self._logger.debug("stream map %s", self._stream_map)
            work_assignment = WorkAssignment.model_validate_json(assignments[1]["data"])
            ingesterset = set()
            for stream, workers in work_assignment.assignments.items():
                if self.state.name in workers:
                    try:
                        ingesterset.add(self._stream_map[stream])
                    except KeyError:
                        self._logger.error(
                            "ingester for stream %s not connected, available: %s",
                            stream,
                            self._ingesters,
                        )
            self._logger.debug("receive from ingesters %s", ingesterset)
            perf_got_assignments = time.perf_counter()

            lastev = assignments[0]
            if len(ingesterset) == 0:
                continue
            tasks: list[Future[list[zmq.Frame]]] = [
                sock.recv_multipart(copy=False) for sock in ingesterset
            ]
            done_pending: tuple[
                set[Future[list[zmq.Frame]]], set[Future[list[zmq.Frame]]]
            ] = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            # print("done", done, "pending", pending)
            perf_got_work = time.perf_counter()
            done, pending = done_pending
            msgs = []
            for res in done:
                prelim = json.loads(res.result()[0].bytes)
                pos = 1
                for stream, data in prelim["streams"].items():
                    data["frames"] = res.result()[pos : pos + data["length"]]
                    pos += data["length"]
                msg = InternalWorkerMessage.model_validate(prelim)
                msgs.append(msg)

            event = EventData.from_internals(msgs)
            perf_assembled_event = time.perf_counter()
            self._logger.debug("received work %s", event)
            result = None
            if self.worker:
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self.worker.process_event, event, self.parameters
                    )
                except Exception as e:
                    self._logger.error("custom worker failed: %s", e.__repr__())
            perf_custom_code = time.perf_counter()
            self._logger.debug("got result %s", result)
            rd = ResultData(
                event_number=event.event_number,
                worker=self.state.name,
                payload=result,
                parameters_uuid=self.state.parameters_uuid,
            )
            if self.out_socket:
                try:
                    header = rd.model_dump_json(exclude={"payload"}).encode("utf8")
                    body = pickle.dumps(rd.payload)
                    self._logger.debug(
                        "send result to reducer with header %s, len-payload %d",
                        header,
                        len(body),
                    )
                    await self.out_socket.send_multipart([header, body])
                except Exception as e:
                    self._logger.error("could not dump result %s", e.__repr__())
            perf_sent_result = time.perf_counter()
            proced += 1
            if proced % 500 == 0:
                self._logger.info("processed %d events", proced)
            times = WorkerTimes.from_timestamps(
                perf_start,
                perf_got_assignments,
                perf_got_work,
                perf_assembled_event,
                perf_custom_code,
                perf_sent_result,
            )
            wu = WorkerUpdate(
                state=WorkerStateEnum.IDLE,
                completed=work_assignment.event_number,
                worker=self.state.name,
                processing_times=times,
            )
            self._logger.debug("all work done, notify controller with %s", wu)
            await self.redis.xadd(
                RedisKeys.ready(self.state.mapping_uuid),
                {"data": wu.model_dump_json()},
            )

    async def finish_work(self) -> None:
        self._logger.info("finishing work")
        if self.worker:
            if hasattr(self.worker, "finish"):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.worker.finish, self.parameters)

    async def restart_work(self, new_uuid: UUID4) -> None:
        self._logger.info("resetting config %s", new_uuid)
        self.work_task.cancel()
        self.state.mapping_uuid = new_uuid
        self.work_task = asyncio.create_task(self.work())

    async def manage_receiver(self) -> None:
        while True:
            config = await self.redis.get(RedisKeys.config("reducer"))
            if config is None:
                self._logger.warning("cannot get reducer configuration")
                await asyncio.sleep(1)
                continue
            cfg = ReducerState.model_validate_json(config)
            if cfg.service_uuid != self._reducer_service_uuid:
                # connect to a new reducer
                if self.out_socket is not None:
                    self.out_socket.close()
                self.out_socket = self.ctx.socket(zmq.PUSH)
                self.out_socket.connect(str(cfg.url))
                self._reducer_service_uuid = cfg.service_uuid
                self._logger.info("connected out_socket to reducer at %s", cfg.url)
            await asyncio.sleep(10)

    async def manage_ingesters(self) -> None:
        while True:
            configs = await self.redis.keys(RedisKeys.config("ingester"))
            processed = []
            for key in configs:
                cfg = IngesterState.model_validate_json(await self.redis.get(key))
                iname = cfg.name
                processed.append(iname)
                if (
                    iname in self._ingesters
                    and self._ingesters[iname].config.service_uuid != cfg.service_uuid
                ):
                    self._logger.warning(
                        "service_uuid of ingester changed from %s to %s, disconnecting",
                        self._ingesters[iname].config.service_uuid,
                        cfg.service_uuid,
                    )
                    self._ingesters[iname].socket.close()
                    del self._ingesters[iname]

                if iname not in self._ingesters:
                    self._logger.info("adding new ingester %s", iname)
                    sock = self.ctx.socket(zmq.DEALER)
                    sock.setsockopt(zmq.IDENTITY, self.state.name.encode("ascii"))
                    sock.connect(str(cfg.url))
                    await sock.send(b"")
                    self._ingesters[iname] = ConnectedIngester(config=cfg, socket=sock)

                await self._ingesters[iname].socket.send(b"")
                self._logger.debug("pinged %s", iname)
            for iname in set(self._ingesters.keys()) - set(processed):
                self._logger.info("removing stale ingester %s", iname)
                self._ingesters[iname].socket.close()
                del self._ingesters[iname]
            self._stream_map = {
                s: conn_ing.socket
                for ing, conn_ing in self._ingesters.items()
                for s in conn_ing.config.streams
            }
            self.state.ingesters = [a.config for a in self._ingesters.values()]

            await asyncio.sleep(2)

    async def close(self) -> None:
        self.manage_ingester_task.cancel()
        self.manage_receiver_task.cancel()
        await self.redis.delete(RedisKeys.config("worker", self.state.name))
        await super().close()
        self.ctx.destroy()
        self._logger.info("worker closed")
