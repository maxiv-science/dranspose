from typing import Literal, Optional

from pydantic import BaseModel, TypeAdapter


class XspressStart(BaseModel):
    htype: Literal["header"]
    filename: str


class XspressImage(BaseModel):
    htype: Literal["image"]
    frame: int
    shape: list[int]
    exptime: Optional[float] = 1
    type: str
    compression: Optional[str] = "none"


class XspressEnd(BaseModel):
    htype: Literal["series_end"]


XspressPacket = TypeAdapter(XspressStart | XspressImage | XspressEnd)
