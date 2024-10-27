"""
This is the central service to orchestrate all distributed components

"""
import asyncio
from asyncio import Task
from collections import defaultdict
from types import UnionType
from typing import Any, AsyncGenerator, Optional, Annotated, Literal

import logging
import time

from pydantic import UUID4
from starlette.requests import Request
from starlette.responses import Response, FileResponse

from dranspose.distributed import DistributedSettings
from dranspose.helpers.utils import parameters_hash, done_callback, cancel_and_wait
from dranspose.mapping import Mapping
import redis.asyncio as redis
import redis.exceptions as rexceptions
from importlib.metadata import version

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query

from dranspose.parameters import (
    Parameter,
    ParameterType,
    StrParameter,
)
from dranspose.protocol import (
    IngesterState,
    WorkerState,
    RedisKeys,
    EnsembleState,
    WorkerUpdate,
    DistributedStateEnum,
    WorkerName,
    StreamName,
    EventNumber,
    VirtualWorker,
    WorkParameter,
    ControllerUpdate,
    ReducerState,
    WorkerTimes,
    HashDigest,
    ParameterName,
    SystemLoadType,
    IntervalLoad,
    WorkerLoad,
    DistributedUpdate,
    ReducerUpdate,
    IngesterUpdate,
    WorkAssignmentList,
    WorkAssignment,
)

logger = logging.getLogger(__name__)


class ControllerSettings(DistributedSettings):
    pass


class Controller:
    def __init__(self, settings: ControllerSettings | None = None):
        self.settings = settings
        if self.settings is None:
            self.settings = ControllerSettings()

        self.redis = redis.from_url(
            f"{self.settings.redis_dsn}?decode_responses=True&protocol=3"
        )
        self.mapping = Mapping({})
        self.mapping_update_lock = asyncio.Lock()
        self.parameters: dict[ParameterName, WorkParameter] = {}
        self.parameters_hash = parameters_hash(self.parameters)
        self.completed: dict[EventNumber, list[WorkerName]] = defaultdict(list)
        self.to_reduce: set[tuple[EventNumber, WorkerName]] = set()
        self.reduced: dict[EventNumber, list[WorkerName]] = defaultdict(list)
        self.processed_event_no: int = 0
        self.completed_events: list[int] = []
        self.finished_components: list[UnionType] = []
        self.external_stop = False
        self.assign_task: Task[None]
        self.config_fetch_time: float = 0
        self.config_cache: EnsembleState
        self.default_task: Task[None]
        self.consistent_task: Task[None]
        self.worker_timing: dict[
            WorkerName, dict[EventNumber, WorkerTimes]
        ] = defaultdict(dict)
        self.start_time: float

    async def run(self) -> None:
        logger.debug("started controller run")
        dist_lock = await self.redis.set(
            RedisKeys.lock(),
            "🔒",
            ex=10,
            nx=True,
        )
        logger.debug("result of lock acquisition %s", dist_lock)
        while dist_lock is None:
            logger.warning("another controller is already running, will retry ")
            await asyncio.sleep(2)
            dist_lock = await self.redis.set(
                RedisKeys.lock(),
                "🔒",
                ex=10,
                nx=True,
            )
        logger.info("controller lock acquired")

        self.assign_task = asyncio.create_task(self.assign_work())
        self.assign_task.add_done_callback(done_callback)
        self.default_task = asyncio.create_task(self.default_parameters())
        self.default_task.add_done_callback(done_callback)
        self.consistent_task = asyncio.create_task(self.consistent_parameters())
        self.consistent_task.add_done_callback(done_callback)
        self.lock_task = asyncio.create_task(self.hold_lock())
        self.lock_task.add_done_callback(done_callback)

    async def hold_lock(self):
        while True:
            await asyncio.sleep(7)
            dist_lock = await self.redis.set(
                RedisKeys.lock(),
                "🔒",
                ex=10,
                xx=True,
            )
            if dist_lock is False:
                logger.warning("The lock was lost")

    async def get_configs(self) -> EnsembleState:
        if time.time() - self.config_fetch_time < 0.5:
            return self.config_cache
        async with self.redis.pipeline() as pipe:
            await pipe.keys(RedisKeys.config("ingester"))
            await pipe.keys(RedisKeys.config("worker"))
            ingester_keys, worker_keys = await pipe.execute()
        async with self.redis.pipeline() as pipe:
            await pipe.mget(ingester_keys)
            await pipe.mget(worker_keys)
            await pipe.get(RedisKeys.config("reducer"))
            ingester_json, worker_json, reducer_json = await pipe.execute()

        ingesters = [IngesterState.model_validate_json(i) for i in ingester_json]
        workers = [WorkerState.model_validate_json(w) for w in worker_json]
        reducer = None
        if reducer_json:
            reducer = ReducerState.model_validate_json(reducer_json)
        dranspose_version = version("dranspose")

        parameters_version: dict[ParameterName, UUID4] = {
            n: p.uuid for n, p in self.parameters.items()
        }
        redis_param_keys = await self.redis.keys(RedisKeys.parameters("*", "*"))
        logger.debug("redis param keys are %s", redis_param_keys)
        self.config_cache = EnsembleState(
            ingesters=ingesters,
            workers=workers,
            reducer=reducer,
            controller_version=dranspose_version,
            parameters_version=parameters_version,
            parameters_hash=self.parameters_hash,
        )
        self.config_fetch_time = time.time()
        return self.config_cache

    async def get_load(self, intervals: list[int], scan: bool = True) -> SystemLoadType:
        ret = {}
        for wn, wt in self.worker_timing.items():
            last_event = max(wt.keys())
            itval: dict[int | Literal["scan"], IntervalLoad] = {}
            for interval in intervals:
                evs = list(
                    filter(
                        lambda x: x >= len(self.completed_events) - interval,
                        wt.keys(),
                    )
                )
                itval[interval] = IntervalLoad(
                    total=sum([wt[e].total for e in evs]),
                    active=sum([wt[e].active for e in evs]),
                    events=len(evs),
                )
            if scan:
                evs = list(wt.keys())
                itval["scan"] = IntervalLoad(
                    total=sum([wt[e].total for e in evs]),
                    active=sum([wt[e].active for e in evs]),
                    events=len(evs),
                )
            ret[wn] = WorkerLoad(last_event=last_event, intervals=itval)
        return ret

    async def set_mapping(self, m: Mapping) -> None:
        async with self.mapping_update_lock:
            logger.debug("cancelling assign task")
            await cancel_and_wait(self.assign_task)
            logger.debug(
                "deleting keys %s and %s",
                RedisKeys.ready(self.mapping.uuid),
                RedisKeys.assigned(self.mapping.uuid),
            )
            await self.redis.delete(RedisKeys.ready(self.mapping.uuid))
            await self.redis.delete(RedisKeys.assigned(self.mapping.uuid))

            # cleaned up
            self.mapping = m
            cupd = ControllerUpdate(
                mapping_uuid=self.mapping.uuid,
                parameters_version={n: p.uuid for n, p in self.parameters.items()},
                active_streams=list(m.mapping.keys()),
            )
            logger.debug("send controller update %s", cupd)
            await self.redis.xadd(
                RedisKeys.updates(),
                {"data": cupd.model_dump_json()},
            )

            cfgs = await self.get_configs()
            while cfgs.reducer is None or set(
                [u.mapping_uuid for u in cfgs.ingesters]
                + [u.mapping_uuid for u in cfgs.workers]
                + [cfgs.reducer.mapping_uuid]
            ) != {self.mapping.uuid}:
                await asyncio.sleep(0.1)
                cfgs = await self.get_configs()
                logger.debug("updated configs %s", cfgs)
            logger.info("new mapping with uuid %s distributed", self.mapping.uuid)
            self.assign_task = asyncio.create_task(self.assign_work())
            self.assign_task.add_done_callback(done_callback)

    async def set_param(self, name: ParameterName, data: bytes) -> HashDigest:
        param = WorkParameter(name=name, data=data)
        logger.debug("distributing parameter %s with uuid %s", param.name, param.uuid)
        await self.redis.set(RedisKeys.parameters(name, param.uuid), param.data)
        self.parameters[name] = param
        logger.debug("stored parameter %s locally", name)
        self.parameters_hash = parameters_hash(self.parameters)
        logger.debug("parameter hash is now %s", self.parameters_hash)
        cupd = ControllerUpdate(
            mapping_uuid=self.mapping.uuid,
            parameters_version={n: p.uuid for n, p in self.parameters.items()},
        )
        logger.debug("send update %s", cupd)
        await self.redis.xadd(
            RedisKeys.updates(),
            {"data": cupd.model_dump_json()},
        )
        return self.parameters_hash

    async def describe_parameters(self) -> list[ParameterType]:
        desc_keys = await self.redis.keys(RedisKeys.parameter_description())
        param_json = await self.redis.mget(desc_keys)

        params: list[ParameterType] = []
        for i in param_json:
            val: ParameterType = Parameter.validate_json(i)  # type: ignore
            params.append(val)

        params.append(
            StrParameter(
                name=ParameterName("dump_prefix"),
                description="Prefix to dump ingester values",
            )
        )
        return sorted(params, key=lambda x: x.name)

    async def default_parameters(self) -> None:
        while True:
            try:
                desc_keys = await self.redis.keys(RedisKeys.parameter_description())
                param_json = await self.redis.mget(desc_keys)
                for i in param_json:
                    val: ParameterType = Parameter.validate_json(i)  # type: ignore
                    if val.name not in self.parameters:
                        logger.info(
                            "set parameter %s to default %s, (type %s)",
                            val.name,
                            val.default,
                            val.__class__,
                        )
                        await self.set_param(
                            val.name, val.__class__.to_bytes(val.default)
                        )
                await asyncio.sleep(2)
            except asyncio.exceptions.CancelledError:
                break

    async def consistent_parameters(self) -> None:
        while True:
            try:
                # make sure self.parameters is present in redis

                key_names = []
                for name, param in self.parameters.items():
                    key_names.append(RedisKeys.parameters(name, param.uuid))
                if len(key_names) > 0:
                    ex_key_no = await self.redis.exists(*key_names)
                    logger.debug(
                        "check for param values in redis, %d exist of %d: %s",
                        ex_key_no,
                        len(key_names),
                        key_names,
                    )
                    if ex_key_no < len(key_names):
                        logger.warning(
                            "the redis parameters don't match the controller parameters, rewriting"
                        )
                        async with self.redis.pipeline() as pipe:
                            for name, param in self.parameters.items():
                                await pipe.set(
                                    RedisKeys.parameters(name, param.uuid), param.data
                                )
                            await pipe.execute()

                consistent = []
                cfg = await self.get_configs()
                if cfg.reducer and cfg.reducer.parameters_hash != self.parameters_hash:
                    consistent.append(("reducer", cfg.reducer.parameters_hash))
                for wo in cfg.workers:
                    if wo.parameters_hash != self.parameters_hash:
                        consistent.append((wo.name, wo.parameters_hash))
                for ing in cfg.ingesters:
                    if ing.parameters_hash != self.parameters_hash:
                        consistent.append((ing.name, ing.parameters_hash))

                if len(consistent) > 0:
                    logger.info(
                        "inconsistent parameters %s, redistribute hash %s",
                        consistent,
                        self.parameters_hash,
                    )
                    cupd = ControllerUpdate(
                        mapping_uuid=self.mapping.uuid,
                        parameters_version={
                            n: p.uuid for n, p in self.parameters.items()
                        },
                        target_parameters_hash=self.parameters_hash,
                    )
                    logger.debug("send consistency update %s", cupd)
                    await self.redis.xadd(
                        RedisKeys.updates(),
                        {"data": cupd.model_dump_json()},
                    )
                await asyncio.sleep(2)
            except asyncio.exceptions.CancelledError:
                break

    async def _update_processing_times(self, update: WorkerUpdate) -> None:
        if update.completed is None:
            return
        if update.processing_times:
            self.worker_timing[update.worker][
                update.completed[-1]
            ] = update.processing_times
        for compev, has_result in zip(update.completed, update.has_result):
            self.completed[compev].append(update.worker)
            logger.debug("added completed to set %s", self.completed)
            wa = self.mapping.get_event_workers(compev)
            if wa.get_all_workers() == set(self.completed[compev]):
                self.completed_events.append(compev)
            if has_result:
                toadd = True
                if compev in self.reduced:
                    logger.debug(
                        "events %s already received from reducer", self.reduced
                    )
                    if update.worker in self.reduced[compev]:
                        # process worker update very late and already got reduced
                        logger.debug(
                            "event %s from worker %s was already reduced",
                            compev,
                            update.worker,
                        )
                        toadd = False
                if toadd:
                    logger.debug(
                        "waiting for reduction for event %s from worker %s",
                        compev,
                        update.worker,
                    )
                    self.to_reduce.add((compev, update.worker))
        # logger.error("time processtime %s", time.perf_counter() - start)

    async def assign_worker_in_mapping(
        self, worker: WorkerName, completed: EventNumber
    ) -> None:
        cfg = await self.get_configs()
        # logger.error("time cfg %s", time.perf_counter() - start)
        logger.debug(
            "assigning worker %s with all %s",
            worker,
            [ws.name for ws in cfg.workers],
        )
        virt = self.mapping.assign_next(
            next(w for w in cfg.workers if w.name == worker),
            cfg.workers,
            completed=completed,
            horizon=5,
        )
        # logger.error("time assign %s", time.perf_counter() - start)
        logger.debug("assigned worker %s to %s", worker, virt)
        logger.debug(
            "send out complete events in range(%d, %d)",
            self.processed_event_no,
            self.mapping.complete_events,
        )
        assignments: list[WorkAssignment] = []
        for evn in range(self.processed_event_no, self.mapping.complete_events):
            wrks = self.mapping.get_event_workers(EventNumber(evn))
            assignments.append(wrks)
            if wrks.get_all_workers() == set():
                self.completed[wrks.event_number] = []
                self.completed_events.append(wrks.event_number)
            if evn % 1000 == 0:
                logger.info(
                    "1000 events in %lf",
                    time.perf_counter() - self.start_time,
                )
                self.start_time = time.perf_counter()
        if len(assignments) > 0:
            logger.debug("send out assignment %s", assignments)
            await self.redis.xadd(
                RedisKeys.assigned(self.mapping.uuid),
                {"data": WorkAssignmentList.dump_json(assignments)},
            )

        # logger.error("time sent out %s", time.perf_counter() - start)
        self.processed_event_no = self.mapping.complete_events

    async def _process_worker_update(self, update: WorkerUpdate) -> None:
        logger.debug("got a ready worker %s", update)
        if update.state == DistributedStateEnum.READY:
            await self.assign_worker_in_mapping(update.worker, EventNumber(0))
        if update.state == DistributedStateEnum.IDLE:
            # start = time.perf_counter()
            await self.assign_worker_in_mapping(update.worker, update.completed[-1])

            asyncio.create_task(self._update_processing_times(update))

    async def completed_finish(self) -> bool:
        fin_workers = set()
        reducer = False
        fin_ingesters = set()
        for upd in self.finished_components:
            if isinstance(upd, WorkerUpdate):
                fin_workers.add(upd.worker)
            elif isinstance(upd, ReducerUpdate):
                reducer = True
            elif isinstance(upd, IngesterUpdate):
                fin_ingesters.add(upd.ingester)

        cfgs = await self.get_configs()

        return (
            set([w.name for w in cfgs.workers]) == fin_workers
            and set([i.name for i in cfgs.ingesters]) == fin_ingesters
            and reducer
        )

    async def assign_work(self) -> None:
        last = 0
        self.external_stop = False
        self.processed_event_no = 0
        self.completed = defaultdict(list)
        self.reduced = defaultdict(list)
        self.completed_events = []
        self.to_reduce = set()
        self.finished_components = []
        self.worker_timing = defaultdict(dict)
        notify_finish = True
        self.start_time = time.perf_counter()
        while True:
            try:
                workers = await self.redis.xread(
                    {RedisKeys.ready(self.mapping.uuid): last},
                    block=1000,
                )
                if RedisKeys.ready(self.mapping.uuid) in workers:
                    for ready in workers[RedisKeys.ready(self.mapping.uuid)][0]:
                        logger.debug("got ready raw data: %s", ready)
                        update = DistributedUpdate.validate_json(ready[1]["data"])
                        if isinstance(update, WorkerUpdate):
                            await self._process_worker_update(update)
                        elif isinstance(update, ReducerUpdate):
                            if (
                                update.completed is not None
                                and update.worker is not None
                            ):
                                compev = update.completed
                                self.reduced[compev].append(update.worker)
                                logger.debug("added reduced to set %s", self.reduced)
                                if (compev, update.worker) in self.to_reduce:
                                    self.to_reduce.remove((compev, update.worker))
                        elif isinstance(update, IngesterUpdate):
                            pass

                        if hasattr(update, "state"):
                            if update.state == DistributedStateEnum.FINISHED:
                                logger.info("distributed %s has finished", update)
                                self.finished_components.append(update)

                        last = ready[0]
                logger.debug(
                    "checking if finished, completed %d, len %d, to_reduce %s",
                    len(self.completed_events),
                    self.mapping.len(),
                    self.to_reduce,
                )
                if (
                    (
                        len(self.completed_events) > 0
                        and len(self.completed_events) == self.mapping.len()
                        and len(self.to_reduce) == 0
                    )
                    or self.external_stop
                ) and notify_finish:
                    # all events done, send close
                    cupd = ControllerUpdate(
                        mapping_uuid=self.mapping.uuid,
                        parameters_version={
                            n: p.uuid for n, p in self.parameters.items()
                        },
                        finished=True,
                    )
                    logger.debug("send finished update %s", cupd)
                    await self.redis.xadd(
                        RedisKeys.updates(),
                        {"data": cupd.model_dump_json()},
                    )
                    notify_finish = False
            except rexceptions.ConnectionError:
                break

    async def close(self) -> None:
        await cancel_and_wait(self.default_task)
        await cancel_and_wait(self.consistent_task)
        await self.redis.delete(RedisKeys.updates())
        queues = await self.redis.keys(RedisKeys.ready("*"))
        if len(queues) > 0:
            await self.redis.delete(*queues)
        assigned = await self.redis.keys(RedisKeys.assigned("*"))
        if len(assigned) > 0:
            await self.redis.delete(*assigned)
        params = await self.redis.keys(RedisKeys.parameters("*", "*"))
        if len(params) > 0:
            await self.redis.delete(*params)
        await cancel_and_wait(self.lock_task)
        await self.redis.delete(RedisKeys.lock())
        await self.redis.aclose()


ctrl: Controller


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load the ML model
    global ctrl
    ctrl = Controller()
    # run_task = asyncio.create_task(ctrl.run())
    await ctrl.run()
    # run_task.add_done_callback(done_callback)
    yield
    # await cancel_and_wait(run_task)
    await ctrl.close()
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.get("/api/v1/config")
async def get_configs() -> EnsembleState:
    global ctrl
    return await ctrl.get_configs()


@app.get("/api/v1/status")
async def get_status() -> dict[str, Any]:
    global ctrl
    return {
        "work_completed": ctrl.completed,
        "last_assigned": ctrl.mapping.complete_events,
        "assignment": ctrl.mapping.assignments,
        "completed_events": ctrl.completed_events,
        "finished": await ctrl.completed_finish(),
        "processing_times": ctrl.worker_timing,
    }


@app.get("/api/v1/load")
async def get_load(
    intervals: Annotated[list[int] | None, Query()] = None, scan: bool = True
) -> SystemLoadType:
    global ctrl
    if intervals is None:
        intervals = [1, 10]
    return await ctrl.get_load(intervals, scan)


@app.get("/api/v1/progress")
async def get_progress() -> dict[str, Any]:
    global ctrl
    return {
        "last_assigned": ctrl.mapping.complete_events,
        "completed_events": len(ctrl.completed_events),
        "total_events": ctrl.mapping.len(),
        "finished": await ctrl.completed_finish(),
    }


@app.post("/api/v1/mapping")
async def set_mapping(
    mapping: dict[StreamName, list[Optional[list[VirtualWorker]]]],
    all_wrap: bool = True,
) -> UUID4 | str:
    global ctrl
    config = await ctrl.get_configs()
    if set(mapping.keys()) - set(config.get_streams()) != set():
        logger.warning(
            "bad request streams %s not available",
            set(mapping.keys()) - set(config.get_streams()),
        )
        raise HTTPException(
            status_code=400,
            detail=f"streams {set(mapping.keys()) - set(config.get_streams())} not available",
        )
    m = Mapping(mapping, add_start_end=all_wrap)
    if len(config.workers) < m.min_workers():
        logger.warning(
            "only %d workers available, but %d required",
            len(config.workers),
            m.min_workers(),
        )
        raise HTTPException(
            status_code=400,
            detail=f"only {len(config.workers)} workers available, but {m.min_workers()} required",
        )
    await ctrl.set_mapping(m)
    return m.uuid


@app.get("/api/v1/mapping")
async def get_mapping() -> dict[StreamName, list[Optional[list[VirtualWorker]]]]:
    global ctrl
    return ctrl.mapping.mapping


@app.post("/api/v1/stop")
async def stop() -> None:
    global ctrl
    logger.info("externally stopped scan")
    ctrl.external_stop = True


@app.get("/api/v1/logs")
async def get_logs(level: str = "INFO") -> Any:
    global ctrl
    data = await ctrl.redis.xrange("dranspose_logs", "-", "+")
    logs = []
    levels = logging.getLevelNamesMapping()
    minlevel = levels.get(level.upper(), "INFO")
    for entry in data:
        msglevel = levels.get(entry[1].get("levelname", "DEBUG"))
        if msglevel >= minlevel:
            logs.append(entry[1])
    return logs


@app.post("/api/v1/sardana_hook")
async def set_sardana_hook(
    info: dict[Literal["streams"] | Literal["scan"], Any]
) -> UUID4 | str:
    global ctrl
    config = await ctrl.get_configs()
    print(info)
    if "scan" not in info:
        return "no scan info"
    if "nb_points" not in info["scan"]:
        return "no nb_points in scan"
    if "streams" not in info:
        return "streams required"
    for st in set(config.get_streams()).intersection(set(info["streams"])):
        print("use stream", st)
    logger.debug("create new mapping")
    m = Mapping.from_uniform(
        set(config.get_streams()).intersection(set(info["streams"])),
        info["scan"]["nb_points"],
    )
    logger.debug("set mapping")
    await ctrl.set_mapping(m)
    return m.uuid


@app.post("/api/v1/parameter/{name}")
async def post_param(request: Request, name: ParameterName) -> HashDigest:
    data = await request.body()
    logger.info("got %s: %s", name, data)
    u = await ctrl.set_param(name, data)
    return u


@app.get("/api/v1/parameter/{name}")
async def get_param(name: ParameterName) -> Response:
    if name not in ctrl.parameters:
        raise HTTPException(status_code=404, detail="Parameter not found")

    data = ctrl.parameters[name].data
    return Response(data, media_type="application/x.bytes")


@app.get("/api/v1/parameters")
async def param_descr() -> list[ParameterType]:
    global ctrl
    return await ctrl.describe_parameters()
