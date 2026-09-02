"""HTTP transport, rate limiter, and resumable downloader.

Network calls go through :class:`HttpTransport` so tests never hit the
real internet. Downloaded bytes are stored as files; they are never
executed.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from omics_agent.errors import ChecksumMismatchError, DownloadError
from omics_agent.hashing import md5_file, sha256_file
from omics_agent.schemas.enums import ChecksumAlg
from omics_agent.schemas.ingest import (
    DownloadPolicy,
    DownloadReceipt,
    RemoteFile,
    VerificationResult,
)

_TRANSIENT = {408, 429, 500, 502, 503, 504}


@dataclass
class HttpResponse:
    """One buffered HTTP response."""

    status: int
    headers: dict[str, str]
    body: bytes
    url: str


class HttpTransport(Protocol):
    """Minimal HTTP surface used by adapters and the downloader."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        """Return the full body. Used for JSON metadata."""

    def iter_body(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> tuple[HttpResponse, Iterator[bytes]]:
        """Return headers plus a byte iterator for large files."""


class UrllibTransport:
    """Stdlib transport. Does not follow to ``file:`` URLs."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        _assert_remote_url(url)
        req = urllib.request.Request(url, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return HttpResponse(
                    status=int(resp.status),
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=resp.read(),
                    url=str(resp.geturl()),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=int(exc.code),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read() if exc.fp is not None else b"",
                url=url,
            )

    def iter_body(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> tuple[HttpResponse, Iterator[bytes]]:
        _assert_remote_url(url)
        req = urllib.request.Request(url, method=method, headers=headers or {})
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
        except urllib.error.HTTPError as exc:
            header = HttpResponse(
                status=int(exc.code),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read() if exc.fp is not None else b"",
                url=url,
            )
            return header, iter(())
        header = HttpResponse(
            status=int(resp.status),
            headers={k.lower(): v for k, v in resp.headers.items()},
            body=b"",
            url=str(resp.geturl()),
        )

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    yield block
            finally:
                resp.close()

        return header, chunks()


class HostRateLimiter:
    """Sleep so each host stays under ``requests_per_host_per_sec``."""

    def __init__(
        self, requests_per_host_per_sec: float, *, sleeper: Callable[[float], None] = time.sleep
    ) -> None:
        self.min_interval = 1.0 / requests_per_host_per_sec
        self._sleeper = sleeper
        self._last: dict[str, float] = {}

    def wait(self, url: str, *, now: float | None = None) -> float:
        host = urlparse(url).netloc.lower()
        current = time.monotonic() if now is None else now
        last = self._last.get(host)
        delayed = 0.0
        if last is not None:
            due = last + self.min_interval
            if current < due:
                delayed = due - current
                self._sleeper(delayed)
                current = due
        self._last[host] = current
        return delayed


def _assert_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError(
            f"Refusing non-HTTP URL scheme '{parsed.scheme or 'missing'}'.",
            how_to_fix="Use https:// (or http://) URLs. file:// and javascript: are not allowed.",
        )


@dataclass
class Downloader:
    """Resumable, size-limited, retried GET that writes next to ``dest``."""

    transport: HttpTransport
    policy: DownloadPolicy
    limiter: HostRateLimiter = field(init=False)

    def __post_init__(self) -> None:
        self.limiter = HostRateLimiter(self.policy.requests_per_host_per_sec)

    def fetch_json(self, url: str) -> HttpResponse:
        """GET a metadata document. Body is not executed."""

        return self._request_with_retry("GET", url)

    def download(self, remote: RemoteFile, dest_dir: Path, *, dry_run: bool) -> DownloadReceipt:
        """Download ``remote`` into ``dest_dir / filename``.

        Partial files use a ``.partial`` suffix. A completed dest is never
        overwritten; a checksum mismatch raises and leaves ``.rejected``.
        """

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / remote.filename
        if dry_run:
            return DownloadReceipt(
                filename=remote.filename,
                dest=dest,
                bytes_written=0,
                dry_run=True,
            )
        if remote.url is None:
            if remote.local_path is None:
                raise DownloadError(
                    f"No URL or local path for {remote.filename}.",
                    how_to_fix="The adapter must set url or local_path.",
                )
            return self._snapshot_local(remote, dest)
        if dest.is_file():
            # Already present: verify only, do not overwrite raw.
            receipt = DownloadReceipt(
                filename=remote.filename,
                dest=dest,
                bytes_written=dest.stat().st_size,
                resumed=False,
                sha256=sha256_file(dest),
                official_checksum=remote.official_checksum,
                official_checksum_alg=remote.official_checksum_alg,
            )
            self.verify(receipt)
            return receipt
        _assert_disk_space(dest_dir, remote.size_bytes or 0, self.policy.max_bytes)
        return self._download_http(remote, dest)

    def verify(self, receipt: DownloadReceipt) -> VerificationResult:
        """Check SHA-256 and, when present, the official publisher digest."""

        if receipt.dry_run:
            return VerificationResult(
                filename=receipt.filename, ok=True, sha256=None, official_ok=None, detail="dry-run"
            )
        if receipt.skipped_reason:
            return VerificationResult(
                filename=receipt.filename,
                ok=True,
                sha256=None,
                official_ok=None,
                detail=receipt.skipped_reason,
            )
        if not receipt.dest.is_file():
            raise DownloadError(
                f"Downloaded file missing: {receipt.dest}",
                how_to_fix="Re-run ingest. A previous attempt may have failed mid-write.",
            )
        size = receipt.dest.stat().st_size
        if size > self.policy.max_bytes:
            raise DownloadError(
                f"{receipt.filename} is {size} bytes, above max_bytes={self.policy.max_bytes}.",
                how_to_fix="Increase download policy max_bytes only after you confirm this is a processed matrix.",
            )
        digest = sha256_file(receipt.dest)
        official_ok: bool | None = None
        if receipt.official_checksum and receipt.official_checksum_alg:
            official_ok = _matches_official(
                receipt.dest, receipt.official_checksum, receipt.official_checksum_alg
            )
            if not official_ok:
                rejected = receipt.dest.with_suffix(receipt.dest.suffix + ".rejected")
                receipt.dest.replace(rejected)
                raise ChecksumMismatchError(
                    f"Official {receipt.official_checksum_alg} mismatch for {receipt.filename}. "
                    f"File moved to {rejected.name}.",
                    how_to_fix=(
                        "Do not use this file. Re-download, or confirm you have the publisher's "
                        "current checksum. The pipeline will not ignore a failed official check."
                    ),
                )
        receipt.sha256 = digest
        return VerificationResult(
            filename=receipt.filename,
            ok=True,
            sha256=digest,
            official_ok=official_ok,
            detail="sha256 ok" + ("; official checksum ok" if official_ok else ""),
        )

    def _request_with_retry(self, method: str, url: str, headers: dict[str, str] | None = None) -> HttpResponse:
        last: HttpResponse | None = None
        attempts = self.policy.retries + 1
        for attempt in range(attempts):
            self.limiter.wait(url)
            last = self.transport.request(
                method,
                url,
                headers=self._headers(headers),
                timeout=self.policy.timeout_s,
            )
            if last.status < 400 or last.status not in _TRANSIENT:
                return last
            if attempt < attempts - 1:
                time.sleep(self.policy.retry_backoff_s * (2**attempt))
        assert last is not None
        return last

    def _download_http(self, remote: RemoteFile, dest: Path) -> DownloadReceipt:
        assert remote.url is not None
        partial = dest.with_suffix(dest.suffix + ".partial")
        existing = partial.stat().st_size if partial.is_file() else 0
        headers: dict[str, str] = {}
        resumed = False
        if self.policy.resume and existing > 0:
            headers["Range"] = f"bytes={existing}-"
            resumed = True
        last_exc: Exception | None = None
        attempts = self.policy.retries + 1
        for attempt in range(attempts):
            try:
                return self._stream_to_partial(remote, dest, partial, headers, existing, resumed)
            except DownloadError as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise
                time.sleep(self.policy.retry_backoff_s * (2**attempt))
        raise last_exc if last_exc else DownloadError("Download failed.", how_to_fix="Retry ingest.")

    def _stream_to_partial(
        self,
        remote: RemoteFile,
        dest: Path,
        partial: Path,
        headers: dict[str, str],
        existing: int,
        resumed: bool,
    ) -> DownloadReceipt:
        assert remote.url is not None
        self.limiter.wait(remote.url)
        meta, chunks = self.transport.iter_body(
            "GET",
            remote.url,
            headers=self._headers(headers),
            timeout=self.policy.timeout_s,
        )
        if meta.status in _TRANSIENT or meta.status >= 400:
            if meta.status == 416 and existing:
                # Server rejected the range; restart.
                partial.unlink(missing_ok=True)
                raise DownloadError(
                    f"Range not satisfiable for {remote.filename}; will retry from scratch.",
                    how_to_fix="Re-run ingest. The incomplete .partial file was removed.",
                )
            raise DownloadError(
                f"GET {remote.url} returned HTTP {meta.status}.",
                how_to_fix="Check the accession and whether the file is still public.",
            )
        content_length = _content_length(meta.headers)
        if meta.status == 200:
            existing = 0
            resumed = False
            mode = "wb"
        elif meta.status == 206:
            mode = "ab"
        else:
            raise DownloadError(
                f"Unexpected HTTP {meta.status} while downloading {remote.filename}.",
                how_to_fix="The server must return 200 or 206 for a file GET.",
            )
        planned = existing + (content_length or 0)
        if content_length is not None and planned > self.policy.max_bytes:
            raise DownloadError(
                f"{remote.filename} would be {planned} bytes, above max_bytes={self.policy.max_bytes}.",
                how_to_fix="This is probably not a processed matrix. Do not raise the limit to pull FASTQ.",
            )
        written = existing
        with partial.open(mode) as handle:
            for chunk in chunks:
                written += len(chunk)
                if written > self.policy.max_bytes:
                    handle.close()
                    partial.unlink(missing_ok=True)
                    raise DownloadError(
                        f"{remote.filename} exceeded max_bytes while streaming.",
                        how_to_fix="Aborting to avoid filling the disk with an unexpected large file.",
                    )
                handle.write(chunk)
        partial.replace(dest)
        receipt = DownloadReceipt(
            filename=remote.filename,
            dest=dest,
            bytes_written=written,
            resumed=resumed,
            official_checksum=remote.official_checksum,
            official_checksum_alg=remote.official_checksum_alg,
        )
        self.verify(receipt)
        return receipt

    def _snapshot_local(self, remote: RemoteFile, dest: Path) -> DownloadReceipt:
        assert remote.local_path is not None
        src = remote.local_path
        if not src.is_file():
            raise DownloadError(
                f"Local file not found: {src}",
                how_to_fix="Pass an existing processed matrix, not a directory of FASTQ.",
            )
        size = src.stat().st_size
        if size > self.policy.max_bytes:
            raise DownloadError(
                f"{src.name} is {size} bytes, above max_bytes={self.policy.max_bytes}.",
                how_to_fix="Confirm this is a processed table before raising the limit.",
            )
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        receipt = DownloadReceipt(
            filename=remote.filename,
            dest=dest,
            bytes_written=size,
            sha256=sha256_file(dest),
            official_checksum=remote.official_checksum,
            official_checksum_alg=remote.official_checksum_alg,
        )
        self.verify(receipt)
        return receipt

    def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {"User-Agent": self.policy.user_agent}
        if self.policy.contact_email:
            headers["From"] = self.policy.contact_email
        if extra:
            headers.update(extra)
        return headers


def _content_length(headers: dict[str, str]) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _matches_official(path: Path, expected: str, alg: ChecksumAlg) -> bool:
    want = expected.strip().lower().replace("md5:", "").replace("sha256:", "")
    if alg is ChecksumAlg.SHA256:
        return sha256_file(path).lower() == want
    if alg is ChecksumAlg.MD5:
        return md5_file(path).lower() == want
    return False


def _assert_disk_space(dest_dir: Path, needed: int, max_bytes: int) -> None:
    usage = shutil.disk_usage(dest_dir)
    reserve = max(needed, 0) + 8 * 1024 * 1024
    if usage.free < reserve:
        raise DownloadError(
            f"Not enough free disk in {dest_dir} (free={usage.free}, need≈{reserve}).",
            how_to_fix="Free disk or choose another --dest. The pipeline will not start a partial huge write.",
        )
    del max_bytes
