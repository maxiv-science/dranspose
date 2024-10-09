from typing import AsyncGenerator, Optional

import zmq

from dranspose.data.pcap import PCAPPacket, PCAPStart, PCAPData, PCAPEnd
from dranspose.event import StreamData
from dranspose.ingester import Ingester, IngesterSettings
from dranspose.protocol import StreamName, ZmqUrl


class ZmqSubPCAPSettings(IngesterSettings):
    upstream_url: ZmqUrl


class ZmqSubPCAPIngester(Ingester):
    def __init__(self, settings: Optional[ZmqSubPCAPSettings] = None) -> None:
        if settings is not None:
            self._streaming_PCAP_settings = settings
        else:
            self._streaming_PCAP_settings = ZmqSubPCAPSettings()

        super().__init__(settings=self._streaming_PCAP_settings)
        self.in_socket: Optional[zmq._future._AsyncSocket] = None

    async def run_source(self, stream: StreamName) -> AsyncGenerator[StreamData, None]:
        self.in_socket = self.ctx.socket(zmq.SUB)
        self.in_socket.connect(str(self._streaming_PCAP_settings.upstream_url))
        self.in_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._logger.info(
            "subscribed to %s", self._streaming_PCAP_settings.upstream_url
        )

        while True:
            self._logger.debug("clear up insocket")
            parts = await self.in_socket.recv_multipart(copy=False)
            try:
                packet = PCAPPacket.validate_json(parts[0].bytes)
            except Exception as e:
                self._logger.warning("packet not valid %s", e.__repr__())
                continue
            self._logger.debug("received frame with header %s", packet)
            if type(packet) is PCAPStart:
                self._logger.info("start of new sequence %s", packet)
                yield StreamData(typ="PCAP", frames=parts)
                break
        while True:
            parts = await self.in_socket.recv_multipart(copy=False)
            try:
                packet = PCAPPacket.validate_json(parts[0].bytes)
            except Exception as e:
                self._logger.error("packet not valid %s", e.__repr__())
                continue
            if type(packet) is PCAPData:
                # vect_keys = [k for k, v in parts.items() if isinstance(v, list)]
                # lenghts = {len(parts[k]) for k, v in parts.items() if isinstance(v, list)}
                vect_keys = {
                    k: len(parts[k]) for k, v in parts.items() if isinstance(v, list)
                }
                lenghts = set(vect_keys.values())
                if len(lenghts) != 1:
                    raise ValueError(
                        "All lists in a PCAP frame must have the same length"
                    )
                # iterator comprehension from hell
                # frames_it = ({k: (v[i] if k in vect_keys.keys() else v + i if k == "frame_number" else v)
                #            for k, v in parts.items()}
                #            for i in range(lenghts.pop()))
                # for frame in frames_it:
                #     yield StreamData(typ="PCAP", frames=frame)
                # equivalent for loop
                for i in range(lenghts.pop()):
                    frame = {}
                    for k, v in parts.items():
                        if k in vect_keys.keys():
                            frame[k] = v[i]
                        elif k == "frame_number":
                            frame[k] = v + i  # Increment frame_number
                        else:
                            frame[k] = v  # Keep the original scalar value
                    yield StreamData(typ="PCAP", frames=frame)
            elif type(packet) is PCAPEnd:
                yield StreamData(typ="PCAP", frames=parts)
                break
        while True:
            self._logger.debug("discarding messages until next run")
            await self.in_socket.recv_multipart(copy=False)

    async def stop_source(self, stream: StreamName) -> None:
        if self.in_socket:
            self._logger.info("closing socket without linger")
            self.in_socket.close(linger=0)
            self.in_socket = None
