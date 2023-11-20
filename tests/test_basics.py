import asyncio
import logging
import random
import time
import aiohttp
import numpy

import pytest
import pytest_asyncio
import numpy as np
import uvicorn
import zmq.asyncio
import zmq

from tests.fixtures import controller
from tests.stream1 import AcquisitionSocket
from dranspose.ingesters.streaming_single import (
    StreamingSingleIngester,
    StreamingSingleSettings,
)
from dranspose.protocol import EnsembleState, RedisKeys, StreamName
from dranspose.worker import Worker

import redis.asyncio as redis


@pytest_asyncio.fixture
async def create_worker():
    workers = []

    async def _make_worker(name):
        worker = Worker(name)
        worker_task = asyncio.create_task(worker.run())
        workers.append((worker, worker_task))
        return worker

    yield _make_worker

    for worker, task in workers:
        await worker.close()
        task.cancel()


@pytest_asyncio.fixture
async def create_ingester():
    ingesters = []

    async def _make_ingester(inst):
        ingester_task = asyncio.create_task(inst.run())
        ingesters.append((inst, ingester_task))
        return inst

    yield _make_ingester

    for inst, task in ingesters:
        await inst.close()
        task.cancel()


@pytest_asyncio.fixture
async def stream_eiger():
    async def _make_eiger(ctx, port, nframes):
        socket = AcquisitionSocket(ctx, f"tcp://*:{port}")
        acq = await socket.start(filename="")
        width = 1475
        height = 831
        for frameno in range(nframes):
            img = np.zeros((width, height), dtype=np.uint16)
            for _ in range(20):
                img[random.randint(0, width - 1)][
                    random.randint(0, height - 1)
                ] = random.randint(0, 10)
            await acq.image(img, img.shape, frameno)
            time.sleep(0.1)
        await acq.close()
        await socket.close()

    yield _make_eiger


@pytest_asyncio.fixture
async def stream_orca():
    async def _make_orca(ctx, port, nframes):
        socket = AcquisitionSocket(ctx, f"tcp://*:{port}")
        acq = await socket.start(filename="")
        width = 2000
        height = 4000
        for frameno in range(nframes):
            img = np.zeros((width, height), dtype=np.uint16)
            for _ in range(20):
                img[random.randint(0, width - 1)][
                    random.randint(0, height - 1)
                ] = random.randint(0, 10)
            await acq.image(img, img.shape, frameno)
            time.sleep(0.1)
        await acq.close()
        await socket.close()

    yield _make_orca


@pytest_asyncio.fixture
async def stream_alba():
    async def _make_alba(ctx, port, nframes):
        socket = AcquisitionSocket(ctx, f"tcp://*:{port}")
        acq = await socket.start(filename="")
        val = np.zeros((0,), dtype=numpy.float64)
        for frameno in range(nframes):
            await acq.image(val, val.shape, frameno)
            time.sleep(0.1)
        await acq.close()
        await socket.close()

    yield _make_alba


@pytest.mark.asyncio
async def test_simple(controller, create_worker, create_ingester, stream_eiger):
    await create_worker("w1")
    await create_ingester(
        StreamingSingleIngester(
            name=StreamName("eiger"),
            settings=StreamingSingleSettings(upstream_url="tcp://localhost:9999"),
        )
    )

    r = redis.Redis(host="localhost", port=6379, decode_responses=True, protocol=3)

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/api/v1/config")
        state = EnsembleState.model_validate(await st.json())
        while {"eiger"} - set(state.get_streams()) != set():
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/api/v1/config")
            state = EnsembleState.model_validate(await st.json())

        ntrig = 10
        resp = await session.post(
            "http://localhost:5000/api/v1/mapping",
            json={
                "eiger": [[2 * i] for i in range(1, ntrig)],
            },
        )
        assert resp.status == 200
        uuid = await resp.json()

    updates = await r.xread({RedisKeys.updates(): 0})
    print("updates", updates)
    keys = await r.keys("dranspose:*")
    print("keys", keys)
    present_keys = {f"dranspose:assigned:{uuid}"}
    print("presentkeys", present_keys)
    assert present_keys - set(keys) == set()

    context = zmq.asyncio.Context()

    asyncio.create_task(stream_eiger(context, 9999, ntrig - 1))

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/api/v1/progress")
        content = await st.json()
        while not content["finished"]:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/api/v1/progress")
            content = await st.json()

    context.destroy()

    await r.aclose()

    print(content)


@pytest.mark.asyncio
async def est_map(
    controller, create_worker, create_ingester, stream_eiger, stream_orca, stream_alba
):
    await create_worker("w1")
    await create_worker("w2")
    await create_worker("w3")
    await create_ingester(
        StreamingSingleIngester(
            name=StreamName("eiger"),
            settings=StreamingSingleSettings(upstream_url="tcp://localhost:9999"),
        )
    )
    await create_ingester(
        StreamingSingleIngester(
            name=StreamName("orca"),
            settings=StreamingSingleSettings(
                upstream_url="tcp://localhost:9998", worker_url="tcp://localhost:10011"
            ),
        )
    )
    await create_ingester(
        StreamingSingleIngester(
            name=StreamName("alba"),
            settings=StreamingSingleSettings(
                upstream_url="tcp://localhost:9997", worker_url="tcp://localhost:10012"
            ),
        )
    )
    await create_ingester(
        StreamingSingleIngester(
            name=StreamName("slow"),
            settings=StreamingSingleSettings(
                upstream_url="tcp://localhost:9996", worker_url="tcp://localhost:10013"
            ),
        )
    )

    r = redis.Redis(host="localhost", port=6379, decode_responses=True, protocol=3)

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/api/v1/config")
        state = EnsembleState.model_validate(await st.json())
        print("content", state.ingesters)
        while {"eiger", "orca", "alba", "slow"} - set(state.get_streams()) != set():
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/api/v1/config")
            state = EnsembleState.model_validate(await st.json())

        print("startup done")
        ntrig = 10
        resp = await session.post(
            "http://localhost:5000/api/v1/mapping",
            json={
                "eiger": [[2 * i] for i in range(1, ntrig)],
                "orca": [[2 * i + 1] for i in range(1, ntrig)],
                "alba": [[2 * i, 2 * i + 1] for i in range(1, ntrig)],
                "slow": [
                    [2 * i, 2 * i + 1] if i % 4 == 0 else None for i in range(1, ntrig)
                ],
            },
        )
        assert resp.status == 200
        uuid = await resp.json()

    print("uuid", uuid, type(uuid))
    updates = await r.xread({RedisKeys.updates(): 0})
    print("updates", updates)
    keys = await r.keys("dranspose:*")
    print("keys", keys)
    present_keys = {f"dranspose:assigned:{uuid}"}
    print("presentkeys", present_keys)
    assert present_keys - set(keys) == set()

    context = zmq.asyncio.Context()

    asyncio.create_task(stream_eiger(context, 9999, ntrig - 1))
    asyncio.create_task(stream_orca(context, 9998, ntrig - 1))
    asyncio.create_task(stream_alba(context, 9997, ntrig - 1))
    asyncio.create_task(stream_alba(context, 9996, ntrig // 4))

    async with aiohttp.ClientSession() as session:
        st = await session.get("http://localhost:5000/api/v1/progress")
        content = await st.json()
        while not content["finished"]:
            await asyncio.sleep(0.3)
            st = await session.get("http://localhost:5000/api/v1/progress")
            content = await st.json()

    context.destroy()

    await r.aclose()

    print(content)
