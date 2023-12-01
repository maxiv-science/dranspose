import pickle
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, Literal, Annotated, Optional

from pydantic import (
    AnyUrl,
    UUID4,
    BaseModel,
    validate_call,
    UrlConstraints,
    Field,
    TypeAdapter,
)

from uuid import uuid4
import zmq
from functools import cache

from pydantic_core import Url

ZmqUrl = Annotated[Url, UrlConstraints(allowed_schemes=["tcp"])]

StreamName = NewType("StreamName", str)
"""
strongly typed stream name (str)
"""
WorkerName = NewType("WorkerName", str)
IngesterName = NewType("IngesterName", str)
VirtualWorker = NewType("VirtualWorker", int)
"""
strongly typed virtual worker (int)
"""
EventNumber = NewType("EventNumber", int)
"""
strongly typed event number (int)
"""

class RedisKeys:
    PREFIX = "dranspose"

    @staticmethod
    @cache
    @validate_call
    def config(
        typ: Literal["ingester", "worker", "reducer", "*"] = "*",
        instance: IngesterName | WorkerName | Literal["reducer", "*"] = "*",
    ) -> str:
        if typ == "reducer":
            instance = "reducer"
        return f"{RedisKeys.PREFIX}:{typ}:{instance}:config"

    @staticmethod
    @cache
    @validate_call
    def ready(uuid: Optional[UUID4 | Literal["*"]] = None) -> str:
        return f"{RedisKeys.PREFIX}:ready:{uuid}"

    @staticmethod
    @cache
    @validate_call
    def assigned(uuid: Optional[UUID4 | Literal["*"]] = None) -> str:
        return f"{RedisKeys.PREFIX}:assigned:{uuid}"

    @staticmethod
    @cache
    def updates() -> str:
        return f"{RedisKeys.PREFIX}:controller:updates"

    @staticmethod
    @cache
    @validate_call
    def parameters(
        uuid: UUID4,
    ) -> str:
        return f"{RedisKeys.PREFIX}:controller:parameters:{uuid}"


class ProtocolException(Exception):
    pass


class ControllerUpdate(BaseModel):
    mapping_uuid: UUID4
    parameters_uuid: UUID4
    finished: bool = False


class WorkParameters(BaseModel):
    pickle: bytes
    uuid: UUID4 = Field(default_factory=uuid4)


class WorkAssignment(BaseModel):
    event_number: EventNumber
    assignments: dict[StreamName, list[WorkerName]]

    def get_workers_for_streams(self, streams: list[StreamName]) -> "WorkAssignment":
        ret = WorkAssignment(event_number=self.event_number, assignments={})
        for stream in streams:
            if stream in self.assignments:
                ret.assignments[stream] = self.assignments[stream]
        return ret

    def get_all_workers(self) -> set[WorkerName]:
        return set([x for stream in self.assignments.values() for x in stream])


class WorkerStateEnum(Enum):
    IDLE = "idle"


class WorkerUpdate(BaseModel):
    state: WorkerStateEnum
    completed: EventNumber
    worker: WorkerName
    new: bool = False


class DistributedState(BaseModel):
    service_uuid: UUID4 = Field(default_factory=uuid.uuid4)
    mapping_uuid: Optional[UUID4] = None
    parameters_uuid: Optional[UUID4] = None


class IngesterState(DistributedState):
    name: IngesterName
    url: ZmqUrl
    streams: list[StreamName] = []


class WorkerState(DistributedState):
    name: WorkerName
    ingesters: list[IngesterState] = []


class ReducerState(DistributedState):
    name: str = "reducer"
    url: ZmqUrl


class EnsembleState(BaseModel):
    ingesters: list[IngesterState]
    workers: list[WorkerState]
    reducer: Optional[ReducerState]

    def get_streams(self) -> list[StreamName]:
        ingester_streams = set([s for i in self.ingesters for s in i.streams])
        worker_streams = [
            set([s for i in w.ingesters for s in i.streams]) for w in self.workers
        ]

        return list(ingester_streams.intersection(*worker_streams))
