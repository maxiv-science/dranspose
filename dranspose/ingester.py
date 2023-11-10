import asyncio
import json
import time
from typing import Dict

import redis.asyncio as redis
import zmq.asyncio
import logging
from dranspose import protocol

logger = logging.getLogger(__name__)

class IngesterState:
    def __init__(self, name, url, streams):
        self.name = name
        self.url = url
        self.streams = streams


class Ingester:
    def __init__(self, name: str, redis_host="localhost", redis_port=6379, config=None):
        if config is None:
            config = {}
        self.ctx = zmq.asyncio.Context()
        self.out_socket = self.ctx.socket(zmq.ROUTER)
        self.out_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.out_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.out_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 300)
        self.out_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 300)
        self.out_socket.bind(f"tcp://*:{config.get('worker_port', 10000)}")
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, protocol=3)
        streams = config.get("streams",["orca","eiger"])
        self.state = IngesterState(name, config.get("worker_url", f"tcp://localhost:{config.get('worker_port', 10000)}"), streams)

    async def run(self):
        asyncio.create_task(self.register())
        asyncio.create_task(self.accept_workers())
        asyncio.create_task(self.work())

    async def work(self):
        while True:
            print("poke worker 1")
            try:
                await self.out_socket.send_multipart([b"worker1", b"hello worker 1"])
            except zmq.error.ZMQError:
                print("not reachable, try again")
            await asyncio.sleep(3)

    async def accept_workers(self):
        poller = zmq.asyncio.Poller()
        poller.register(self.out_socket, zmq.POLLIN)
        while True:
            socks = dict(await poller.poll())
            for sock in socks:
                data = await sock.recv_multipart()
                logger.info("new worker connected %s", data[0])

    async def register(self):
        while True:
            await self.redis.setex(f"{protocol.PREFIX}:ingester:{self.state.name}:present", 10, 1)
            await self.redis.json().set(f"{protocol.PREFIX}:ingester:{self.state.name}:config","$", self.state.__dict__)
            await asyncio.sleep(6)

    async def close(self):
        await self.redis.delete(f"{protocol.PREFIX}:ingester:{self.state.name}:config")
        await self.redis.aclose()

    def __del__(self):
        self.ctx.destroy()
        logger.info("stopped worker")