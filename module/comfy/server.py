import asyncio
import dataclasses
import uuid
from io import BytesIO
from typing import Optional

import server
from aiohttp import web

from ..core.resource_management import GLOBAL_RESOURCE_RESOLVER, ParseResult, Resource, ResourceResolver


def dict_to_simple_dataclass(cls, d):
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    keys = set(fieldtypes.keys()) & set(d.keys())
    return cls(**{f: d[f] for f in keys})


class DataObject:
    def to_dict(self):
        return dataclasses.asdict(self)


class UUIDResourceResource(Resource):
    def __init__(self, bytes_buff) -> None:
        self.bytes_buff = bytes_buff

    def get_bytes_io(self) -> BytesIO:
        return BytesIO(self.bytes_buff)


class UUIDResourceResourceResolver(ResourceResolver):
    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        return GLOBAL_UUID_RESOURCE_POOL.get(url.hostname, None)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("playgraounduuidres", UUIDResourceResourceResolver())

GLOBAL_UUID_RESOURCE_POOL = {}
GLOBAL_UUID_RESOURCE_POOL_LOCK = asyncio.Lock()


def gen_res_uuid():
    while res_uuid := uuid.uuid4().hex:
        if res_uuid not in GLOBAL_UUID_RESOURCE_POOL:
            return res_uuid


@dataclasses.dataclass
class UUIDResQueryParameters(DataObject):
    res_uuid: str = ""


@server.PromptServer.instance.routes.get("/api/paper_playground/uuid_res")
async def get_uuid_res(request):
    query_parameters = dict_to_simple_dataclass(UUIDResQueryParameters, request.rel_url.query)

    try:
        res_uuid = query_parameters.res_uuid.strip()
        if len(res_uuid) != 32:
            return web.Response(status=400)

        assert uuid.UUID(res_uuid)

        with GLOBAL_RESOURCE_RESOLVER.get_resource("playgraounduuidres://" + res_uuid).get_bytes_io() as b:
            return web.Response(body=b.read())
    except FileNotFoundError:
        return web.Response(status=404)
    except Exception:
        pass

    return web.Response(status=400)


@server.PromptServer.instance.routes.post("/api/paper_playground/uuid_res")
async def post_uuid_res(request):
    post = await request.post()
    res = post.get("res")
    file_bytes = res.file.read()
    async with GLOBAL_UUID_RESOURCE_POOL_LOCK:
        res_uuid = gen_res_uuid()
        GLOBAL_UUID_RESOURCE_POOL[res_uuid] = UUIDResourceResource(file_bytes)

    return web.json_response(UUIDResQueryParameters(res_uuid).to_dict())
