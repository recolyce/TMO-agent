"""HTTP fake that matches URL prefixes. Never opens a socket."""

from __future__ import annotations

from collections.abc import Iterator

from omics_agent.data_sources.http import HttpResponse
from tests.unit.http_fakes import FakeRoute


class PrefixFakeTransport:
    """First prefix match wins; default empty-200 so literature tests stay offline."""

    def __init__(
        self, routes: dict[str, FakeRoute] | None = None, default: FakeRoute | None = None
    ) -> None:
        self.routes = routes or {}
        self.default = default or FakeRoute(body=b"{}")
        self.calls: list[tuple[str, str]] = []

    def _lookup(self, url: str) -> FakeRoute:
        if url in self.routes:
            return self.routes[url]
        for prefix, route in self.routes.items():
            if prefix in url:
                return route
        return self.default

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
        return HttpResponse(
            status=route.status, headers=dict(route.headers), body=route.body, url=url
        )

    def iter_body(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> tuple[HttpResponse, Iterator[bytes]]:
        meta = self.request(method, url, headers=headers, timeout=timeout)
        return meta, iter((meta.body,))
