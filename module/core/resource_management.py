import logging
import os
from io import BytesIO
from typing import Optional, Union
from urllib.parse import ParseResult, urlparse

_logger = logging.getLogger(__name__)


class Resource:
    def get_bytes_io(self) -> BytesIO:
        raise NotImplementedError()

    def get_filesystem_path(self) -> Optional[str]:
        raise NotImplementedError()


class ResourceResolver:
    def get_resource(self, url: Union[str, ParseResult]) -> Optional[Resource]:
        _logger.log(logging.INFO, f"try get resource {url}")
        if isinstance(url, str):
            url = urlparse(url)

        if (resource := self._get_resource(url)) is not None:
            return resource
        else:
            raise FileNotFoundError()

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        raise NotImplementedError()


class MutiSchemeResourceResolver(ResourceResolver):
    def __init__(self) -> None:
        self.scheme_resolver_map: dict[str, ResourceResolver] = {}

    def registe_scheme_resolver(self, scheme, scheme_resolver):
        self.scheme_resolver_map[scheme] = scheme_resolver

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        scheme_resolver = self.scheme_resolver_map[url.scheme]
        return scheme_resolver._get_resource(url)


class ProxyMutiSchemeResourceResolver(MutiSchemeResourceResolver):
    def __init__(self) -> None:
        super().__init__()
        self.identical_resource_url_map: dict[str, set[str]] = {}

    def registe_identical_resource_urls(self, identical_resource_urls: set[str]):
        new_identical_resource_urls = set()
        for url in identical_resource_urls:
            new_identical_resource_urls.add(url)
            if old_identical_resource_urls := self.identical_resource_url_map.get(url):
                new_identical_resource_urls.update(old_identical_resource_urls)
        for url in new_identical_resource_urls:
            self.identical_resource_url_map[url] = new_identical_resource_urls

    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        if url.geturl() not in self.identical_resource_url_map:
            return super()._get_resource(url)

        identical_resource_urls = self.identical_resource_url_map[url.geturl()]

        for identical_url in identical_resource_urls:
            if resource := super().get_resource(identical_url):
                return resource


GLOBAL_RESOURCE_RESOLVER = ProxyMutiSchemeResourceResolver()


class FileSystemResource(Resource):
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def get_bytes_io(self) -> BytesIO:
        return open(self.file_path, "rb")

    def get_filesystem_path(self) -> Optional[str]:
        return self.file_path


class FileSchemeResourceResolver(ResourceResolver):
    def _get_resource(self, url: ParseResult) -> Optional[Resource]:
        file_path = url.path
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileSystemResource(file_path)


GLOBAL_RESOURCE_RESOLVER.registe_scheme_resolver("file", FileSchemeResourceResolver())

if __name__ == "__main__":
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls(["sha256://jdie", "http://1"])
    GLOBAL_RESOURCE_RESOLVER.registe_identical_resource_urls(["http://2", "http://1"])
    print(GLOBAL_RESOURCE_RESOLVER.identical_resource_url_map)
    with GLOBAL_RESOURCE_RESOLVER.get_resource(
        "file:///ComfyUI_paper_playground/module/core/precision.py"
    ).get_bytes_io() as f:
        print(f.read(15))
