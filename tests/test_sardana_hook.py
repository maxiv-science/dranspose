import json
import logging
from typing import Callable, Awaitable, Optional

import aiohttp
import pytest
from pydantic_core import Url

from dranspose.ingester import Ingester
from dranspose.ingesters.zmqpull_single import (
    ZmqPullSingleIngester,
    ZmqPullSingleSettings,
)
from dranspose.protocol import (
    StreamName,
)
from tests.utils import wait_for_controller

payload = """{"streams": ["albaem-01_ch1", "albaem-01_ch3", "orca", "pcap_rot", "pcap_trigts", "albaem-xrd_ch1"], "scan": {"_name": "timescan", "_in_pars": [10, 0.03, 0.03], "_out_pars": null, "_aborted": false, "_stopped": false, "_processingStop": false, "_parent_macro": null, "_executor": "<<non-serializable: MacroExecutor>>", "_macro_line": "timescan(10, 0.03, 0.03) -> 39d8ecd6-a0e9-11ee-96fa-0050569cfe6d", "_interactive_mode": true, "_macro_thread": "<<non-serializable: OmniWorker>>", "_id": "39d8ecd6-a0e9-11ee-96fa-0050569cfe6d", "_desc": "Macro 'timescan(10, 0.03, 0.03) -> 39d8ecd6-a0e9-11ee-96fa-0050569cfe6d'", "_macro_status": {"id": "39d8ecd6-a0e9-11ee-96fa-0050569cfe6d", "range": [0.0, 100.0], "state": "start", "step": 0.0}, "_pause_event": "<<non-serializable: PauseEvent>>", "inited_class_list": ["<<non-serializable: type>>"], "log_name": "Macro[timescan]", "log_full_name": "B304A/DOOR/01.Macro[timescan]", "log_obj": "<<non-serializable: Logger>>", "log_handlers": [], "log_parent": "<<non-serializable: weakref>>", "log_children": {"140150443868416": "<<non-serializable: weakref>>", "140150443868848": "<<non-serializable: weakref>>", "140150443868464": "<<non-serializable: weakref>>", "140150443868896": "<<non-serializable: weakref>>", "140150443868608": "<<non-serializable: weakref>>"}, "_hooks": [["<<non-serializable: ExecMacroHook>>", ["pre-scan"]], ["<<non-serializable: ExecMacroHook>>", ["pre-scan"]], ["<<non-serializable: ExecMacroHook>>", ["pre-scan"]], ["<<non-serializable: ExecMacroHook>>", ["pre-scan"]], ["<<non-serializable: ExecMacroHook>>", ["post-scan"]], ["<<non-serializable: ExecMacroHook>>", ["post-scan"]], ["<<non-serializable: ExecMacroHook>>", ["post-scan"]], ["<<non-serializable: ExecMacroHook>>", ["post-scan"]], ["<<non-serializable: ExecMacroHook>>", ["pre-scan"]]], "_hookHintsDict": {"_ALL_": ["<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>"], "_NOHINTS_": [], "pre-scan": ["<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>"], "post-scan": ["<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>", "<<non-serializable: ExecMacroHook>>"]}, "nb_points": 10, "integ_time": 0.03, "latency_time": 0.03, "_gScan": "<<non-serializable: TScan>>", "_data": {}}}"""


@pytest.mark.asyncio
async def test_not_enough_workers(
    controller: None,
    create_ingester: Callable[[Ingester], Awaitable[Ingester]],
    reducer: Callable[[Optional[str]], Awaitable[None]],
) -> None:
    await reducer(None)
    await create_ingester(
        ZmqPullSingleIngester(
            settings=ZmqPullSingleSettings(
                ingester_streams=[StreamName("orca")],
                upstream_url=Url("tcp://localhost:9999"),
            ),
        )
    )
    await wait_for_controller(streams={StreamName("orca")})
    async with aiohttp.ClientSession() as session:
        data = json.loads(payload)
        resp = await session.post(
            "http://localhost:5000/api/v1/sardana_hook",
            json=data,
        )
        assert resp.status == 200
        await resp.json()

        ma = await session.get("http://localhost:5000/api/v1/mapping")
        mapping = await ma.json()
        logging.info("map in %s", mapping)

        assert mapping == {
            "parts": {
                "sardana": {
                    "mapping": {
                        "orca": [
                            [{"tags": ["generic"], "constraint": 0}],
                            [{"tags": ["generic"], "constraint": 1}],
                            [{"tags": ["generic"], "constraint": 2}],
                            [{"tags": ["generic"], "constraint": 3}],
                            [{"tags": ["generic"], "constraint": 4}],
                            [{"tags": ["generic"], "constraint": 5}],
                            [{"tags": ["generic"], "constraint": 6}],
                            [{"tags": ["generic"], "constraint": 7}],
                            [{"tags": ["generic"], "constraint": 8}],
                            [{"tags": ["generic"], "constraint": 9}],
                        ]
                    }
                },
                "reserved_start_end": {
                    "mapping": {"orca": [[{"tags": ["generic"], "constraint": None}]]}
                },
            },
            "sequence": ["reserved_start_end", "sardana", "reserved_start_end"],
        }
