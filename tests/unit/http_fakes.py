"""In-memory HTTP transport for ingest tests. Never opens a real socket."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from omics_agent.data_sources.http import HttpResponse


@dataclass
class FakeRoute:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    support_range: bool = True


class FakeTransport:
    def __init__(self, routes: dict[str, FakeRoute]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str]] = []

    def _lookup(self, url: str) -> FakeRoute:
        return self.routes.get(url, FakeRoute(status=404, body=b"missing"))

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        del headers, timeout
        self.calls.append((method, url))
        route = self._lookup(url)
        return HttpResponse(status=route.status, headers=dict(route.headers), body=route.body, url=url)

    def iter_body(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> tuple[HttpResponse, Iterator[bytes]]:
        del timeout
        self.calls.append((method, url))
        route = self._lookup(url)
        body = route.body
        status = route.status
        out_headers = dict(route.headers)
        range_header = (headers or {}).get("Range")
        if range_header and route.support_range and body:
            start = int(range_header.split("=")[1].split("-")[0])
            body = body[start:]
            status = 206
        out_headers["content-length"] = str(len(body))
        meta = HttpResponse(status=status, headers=out_headers, body=b"", url=url)
        return meta, iter((body,))
