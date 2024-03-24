import asyncio
import logging
import struct
import time
from typing import Callable, Any, Coroutine, Optional

import pytest
import zmq.asyncio


async def sink(port, n) -> None:
    c = zmq.asyncio.Context()
    s = c.socket(zmq.PULL)
    s.connect(f"tcp://localhost:{port}")
    start = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        # logging.info("recv %d", i)
        data = await s.recv_multipart(copy=True)
        total_bytes += sum(map(len, data))
        if i % 100 == 0:
            end = time.perf_counter()
            logging.info(
                "sink 100 packets took %s: %lf p/s", end - start, 100 / (end - start)
            )
            start = end
        # logging.info("i %d, data %s", i, data)
    logging.info("total bytes %d", total_bytes)
    c.destroy()


async def time_sink(port, n) -> None:
    c = zmq.asyncio.Context()
    s = c.socket(zmq.PULL)
    s.connect(f"tcp://localhost:{port}")
    latencies = []
    for i in range(n):
        # logging.info("recv %d", i)
        data = await s.recv_multipart(copy=True)
        now = time.perf_counter()
        start = struct.unpack(">d", data[0])[0]
        # logging.info("latency was %lf", now-start)
        latencies.append(now - start)
        if i % 1000 == 0 and i != 0:
            logging.info(
                "last 1000 latency: avg %lf, max: %lf min: %lf",
                sum(latencies) / len(latencies),
                max(latencies),
                min(latencies),
            )
            latencies = []
            # end = time.perf_counter()
            # logging.info("sink 100 packets took %s: %lf p/s", end - start, 100 / (end - start))
            # start = end
        # logging.info("i %d, data %s", i, data)
    c.destroy()


async def forward(inport, outport, n) -> None:
    c = zmq.asyncio.Context()
    sin = c.socket(zmq.PULL)
    sout = c.socket(zmq.PUSH)
    sin.connect(f"tcp://localhost:{inport}")
    sout.bind(f"tcp://*:{outport}")
    start = time.perf_counter()
    for i in range(n):
        data = await sin.recv_multipart(copy=False)
        await sout.send_multipart(data)
        if i % 1000 == 0:
            end = time.perf_counter()
            logging.info(
                "forward 1000 packets took %s: %lf p/s",
                end - start,
                1000 / (end - start),
            )
            start = end
    c.destroy()


@pytest.mark.asyncio
async def test_zmq_rate_tcp(
    stream_small: Callable[
        [zmq.Context[Any], int, int, Optional[float]], Coroutine[Any, Any, None]
    ]
) -> None:
    ctask = asyncio.create_task(sink(9999, 1002))

    context = zmq.asyncio.Context()
    await stream_small(context, 9999, 1000, 0.000001)
    await ctask

    context.destroy()


@pytest.mark.asyncio
async def test_zmq_rate_forward(
    stream_small: Callable[
        [zmq.Context[Any], int, int, Optional[float]], Coroutine[Any, Any, None]
    ]
) -> None:
    asyncio.create_task(forward(9999, 9998, 1002))
    ctask = asyncio.create_task(sink(9998, 1002))

    context = zmq.asyncio.Context()
    await stream_small(context, 9999, 1000, 0.00001)
    await ctask

    context.destroy()


@pytest.mark.asyncio
async def test_zmq_latency(time_beacon) -> None:
    ctask = asyncio.create_task(time_sink(9999, 10000))

    context = zmq.asyncio.Context()
    await time_beacon(context, 9999, 10000, 0.00001)
    await ctask
    context.destroy()


@pytest.mark.asyncio
async def test_zmq_forward_latency(time_beacon) -> None:
    asyncio.create_task(forward(9999, 9998, 10000))
    ctask = asyncio.create_task(time_sink(9998, 10000))

    context = zmq.asyncio.Context()
    await time_beacon(context, 9999, 10000, 0.00001)
    await ctask
    context.destroy()


async def route(inport, outport, n, workers) -> None:
    c = zmq.asyncio.Context()
    sin = c.socket(zmq.PULL)
    sout = c.socket(zmq.ROUTER)
    sin.connect(f"tcp://localhost:{inport}")
    sout.bind(f"tcp://*:{outport}")
    for _ in workers:
        data = await sout.recv_multipart()
        logging.info("registered worker %s", data)
    start = time.perf_counter()
    for i in range(n):
        data = await sin.recv_multipart(copy=False)
        await sout.send_multipart([workers[i % len(workers)]] + data)
        if i % 1000 == 0:
            end = time.perf_counter()
            logging.info(
                "forward 1000 packets took %s: %lf p/s",
                end - start,
                1000 / (end - start),
            )
            start = end
    c.destroy()


async def dealer_sink(port, name, n) -> None:
    c = zmq.asyncio.Context()
    s = c.socket(zmq.DEALER)
    s.setsockopt(zmq.IDENTITY, name)
    s.connect(f"tcp://localhost:{port}")
    await s.send(b"ping")
    start = time.perf_counter()
    total_bytes = 0
    for i in range(n):
        # logging.info("recv %d", i)
        data = await s.recv_multipart(copy=True)
        total_bytes += sum(map(len, data))
        if i % 1000 == 0:
            end = time.perf_counter()
            logging.info(
                "%s: sink 100 packets took %s: %lf p/s",
                name,
                end - start,
                1000 / (end - start),
            )
            start = end
        # logging.info("i %d, data %s", i, data)
    logging.info("total bytes %d", total_bytes)
    c.destroy()


@pytest.mark.asyncio
async def test_zmq_full_latency(time_beacon) -> None:
    asyncio.create_task(route(9999, 9998, 10000, [b"w1", b"w2"]))
    c1task = asyncio.create_task(dealer_sink(9998, b"w1", 5000))
    c2task = asyncio.create_task(dealer_sink(9998, b"w2", 5000))

    context = zmq.asyncio.Context()
    await time_beacon(context, 9999, 10000, 0.000001)
    await c1task
    await c2task
    context.destroy()
