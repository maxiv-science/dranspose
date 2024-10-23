import base64
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI
from starlette.requests import Request
from starlette.responses import Response

router = APIRouter()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


def _get_now():
    return time.time()


def _path_to_uuid(path: list[str]) -> bytes:
    abspath = "/" + "/".join(path)
    print("abspath", abspath)
    uuid = base64.b16encode(abspath.encode())
    # print("uuid from function is", str(id(obj)).encode() + b"-" + uuid)
    return b"h5dict-" + uuid


def _uuid_to_obj(data: dict[str, Any], uuid: str):
    logger.debug("parse %s", uuid)
    idstr, path = uuid.split("-")
    path = base64.b16decode(path).decode()
    logger.debug("raw path %s", path)
    # assert id(data) == int(idstr)
    obj = data
    clean_path = []
    for p in path.split("/"):
        print("path part is:", p)
        if p == "":
            continue
        print("traverse", obj)
        obj = obj[p]
        clean_path.append(p)
    logger.debug("parser yields obj %s at path %s", obj, clean_path)
    return obj, clean_path


@router.get("/datasets/{uuid}/value")
def values(req: Request, uuid, select: str | None = None):
    data = req.app.state.get_data()
    obj, _ = _uuid_to_obj(data, uuid)
    logger.debug("return value for obj %s", obj)
    logger.debug("selection %s", select)

    slices = []
    if select is not None:
        dims = select[1:-1].split(",")
        slices = [slice(*map(int, dim.split(":"))) for dim in dims]

    logger.debug("slices %s", slices)
    ret = obj
    if len(slices) == 1:
        ret = ret[slices[0]]
    elif len(slices) > 1:
        ret = ret[*slices]

    if type(ret) is np.ndarray:
        by = ret.tobytes()
        return Response(content=by, media_type="application/octet-stream")

    logger.debug("return value %s of type %s", ret, type(ret))
    return {"value": ret}


@router.get("/datasets/{uuid}")
def datasets(req: Request, uuid):
    data = req.app.state.get_data()
    obj, _ = _uuid_to_obj(data, uuid)
    logger.debug("return dataset for obj %s", obj)
    ret = {
        "id": uuid,
        "attributeCount": 0,
        "lastModified": _get_now(),
        "created": _get_now(),
        "creationProperties": {
            "allocTime": "H5D_ALLOC_TIME_LATE",
            "fillTime": "H5D_FILL_TIME_IFSET",
            "layout": {"class": "H5D_CONTIGUOUS"},
        },
    }
    extra = None
    if type(obj) is int:
        extra = {
            "shape": {"class": "H5S_SCALAR"},
            "type": {"base": "H5T_STD_I64LE", "class": "H5T_INTEGER"},
        }
    elif type(obj) is float:
        extra = {
            "shape": {"class": "H5S_SCALAR"},
            "type": {"base": "H5T_IEEE_F64LE", "class": "H5T_FLOAT"},
        }

    elif type(obj) is str:
        extra = {
            "type": {
                "class": "H5T_STRING",
                "length": "H5T_VARIABLE",
                "charSet": "H5T_CSET_UTF8",
                "strPad": "H5T_STR_NULLTERM",
            },
            "shape": {"class": "H5S_SCALAR"},
        }

    else:
        try:
            arr = np.array(obj)
            extra = {
                "shape": {"class": "H5S_SIMPLE", "dims": arr.shape},
            }
            if arr.dtype == np.float64:
                extra["type"] = {"base": "H5T_IEEE_F64LE", "class": "H5T_FLOAT"}
            elif arr.dtype == np.int64:
                extra["type"] = {"base": "H5T_STD_I64LE", "class": "H5T_INTEGER"}
            else:
                logger.error("numpy type %s not available", arr.dtype)
        except Exception:
            pass

    ret.update(extra)
    print("return dset", ret)
    return ret


def _get_group_link(obj, path):
    if isinstance(obj, dict):
        print("path to sub is: ", path)
        link = {
            "class": "H5L_TYPE_HARD",
            "collection": "groups",
            "id": _path_to_uuid(path),
            "title": path[-1],
            "created": _get_now(),
        }
    else:
        link = {
            "class": "H5L_TYPE_HARD",
            "collection": "datasets",
            "id": _path_to_uuid(path),
            "title": path[-1],
            "created": _get_now(),
        }
    return link


def _get_group_links(obj, path):
    if isinstance(obj, dict):
        links = []
        for key, val in obj.items():
            if not isinstance(key, str):
                logger.warning("unable to use non-string key: %s as group name", key)
                continue
            link = _get_group_link(val, path + [key])
            links.append(link)
        return links


@router.get("/groups/{uuid}/links/{name}")
def link(req: Request, uuid, name):
    data = req.app.state.get_data()
    obj, path = _uuid_to_obj(data, uuid)
    path.append(name)
    obj = obj[name]
    logger.debug("generate link for %s at %s", obj, path)
    ret = {"link": _get_group_link(obj, path)}
    logger.debug("link name is %s", ret)
    return ret


@router.get("/groups/{uuid}/links")
def links(req: Request, uuid: str):
    data = req.app.state.get_data()
    obj, path = _uuid_to_obj(data, uuid)
    ret = {"links": _get_group_links(obj, path)}

    logger.debug("group links %s", ret)
    return ret


@router.get("/groups/{uuid}")
def group(req: Request, uuid: str):
    data = req.app.state.get_data()
    obj, path = _uuid_to_obj(data, uuid)
    logger.debug("start listing group id %s, obj %s, path: %s", uuid, obj, path)

    group = {
        "id": uuid,
        "root": _path_to_uuid([]),
        "linkCount": len(obj),
        "attributeCount": 0,
        "lastModified": _get_now(),
        "created": _get_now(),
        "domain": "/",
    }
    logger.debug("group is %s", group)
    return group


@router.get("/")
def read_root(request: Request):
    logging.info("data %s", request.app.state.get_data())
    data = request.app.state.get_data()
    if isinstance(data, dict):
        uuid = _path_to_uuid([])
        ret = {
            "root": uuid,
            "created": time.time(),
            "owner": "admin",
            "class": "domain",
            "lastModified": time.time(),
        }
        return ret


app = FastAPI()

app.include_router(router, prefix="/results")


def get_data():
    return {"image": {}}


app.state.get_data = get_data
