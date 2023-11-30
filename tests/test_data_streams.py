import json
import pickle

from dranspose.data.contrast import ContrastPacket
from dranspose.data.xspress3_stream import XspressPacket


def test_contrast_stream() -> None:
    with open("tests/data/contrast-dump.pkls", "rb") as f:
        while True:
            try:
                frames = pickle.load(f)
                assert len(frames) == 1
                data = pickle.loads(frames[0])
                pkg = ContrastPacket.validate_python(data)
                print(pkg)
            except EOFError:
                break


def test_xspress3_stream() -> None:
    with open("tests/data/xspress3-dump.pkls", "rb") as f:
        skip = 0
        while True:
            try:
                frames = pickle.load(f)
                assert len(frames) == 1
                if skip > 0:
                    skip += 1
                    continue
                data = json.loads(frames[0])
                pkg = XspressPacket.validate_python(data)
                print(pkg)
                if pkg.htype == "image":
                    skip = 2
            except EOFError:
                break