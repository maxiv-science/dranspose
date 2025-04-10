import asyncio
import logging
from typing import Awaitable, Callable, Any, Coroutine, Optional

from dranspose.protocol import (
    StreamName,
    WorkerName,
    VirtualWorker,
    VirtualConstraint,
)
from dranspose.ingester import Ingester
from dranspose.ingesters.zmqpull_single import (
    ZmqPullSingleIngester,
    ZmqPullSingleSettings,
)
import aiohttp

import pytest
import zmq.asyncio
import zmq
from pydantic_core import Url

from dranspose.worker import Worker, WorkerSettings
from tests.utils import wait_for_controller, wait_for_finish


async def consume_repub(ctx: zmq.Context[Any], num: int) -> int:
    s = ctx.socket(zmq.PULL)
    s.connect("tcp://127.0.0.1:5555")

    for _ in range(num):
        data = await s.recv_multipart(copy=False)
        logging.info("received data %s", data)

    return num


@pytest.mark.asyncio
async def test_repub(
    controller: None,
    reducer: Callable[[Optional[str]], Awaitable[None]],
    create_worker: Callable[[WorkerName | Worker], Awaitable[Worker]],
    create_ingester: Callable[[Ingester], Awaitable[Ingester]],
    stream_eiger: Callable[
        [zmq.Context[Any], int, int, float], Coroutine[Any, Any, None]
    ],
) -> None:
    await reducer(None)
    await create_worker(
        Worker(
            settings=WorkerSettings(
                worker_name=WorkerName("w5555"),
                worker_class="examples.repub.worker:RepubWorker",
            ),
        )
    )
    await create_ingester(
        ZmqPullSingleIngester(
            settings=ZmqPullSingleSettings(
                ingester_streams=[StreamName("eiger")],
                upstream_url=Url("tcp://localhost:9999"),
            ),
        )
    )

    await wait_for_controller(streams={StreamName("eiger")})
    async with aiohttp.ClientSession() as session:
        ntrig = 10
        resp = await session.post(
            "http://localhost:5000/api/v1/mapping",
            json={
                "eiger": [
                    [
                        VirtualWorker(constraint=VirtualConstraint(2 * i)).model_dump(
                            mode="json"
                        )
                    ]
                    for i in range(1, ntrig)
                ],
            },
        )
        assert resp.status == 200
        await resp.json()

    context = zmq.asyncio.Context()

    collector = asyncio.create_task(consume_repub(context, 2 * ntrig + 2))

    asyncio.create_task(stream_eiger(context, 9999, ntrig - 1, 0.1))

    await wait_for_finish()

    async with aiohttp.ClientSession() as session:
        ntrig = 10
        resp = await session.post(
            "http://localhost:5000/api/v1/mapping",
            json={
                "eiger": [
                    [
                        VirtualWorker(constraint=VirtualConstraint(2 * i)).model_dump(
                            mode="json"
                        )
                    ]
                    for i in range(1, ntrig)
                ],
            },
        )
        assert resp.status == 200
        await resp.json()

    asyncio.create_task(stream_eiger(context, 9999, ntrig - 1, 0.1))

    await wait_for_finish()

    await collector

    context.destroy()
