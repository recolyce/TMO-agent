from __future__ import annotations

from pathlib import Path

import pytest

from omics_agent.data_sources.http import Downloader, HostRateLimiter, HttpResponse
from omics_agent.errors import ChecksumMismatchError, DownloadError
from omics_agent.hashing import sha256_bytes
from omics_agent.schemas.enums import ChecksumAlg, FileRole
from omics_agent.schemas.ingest import DownloadPolicy, RemoteFile
from tests.unit.http_fakes import FakeRoute, FakeTransport


def test_download_records_sha256_and_official_md5(tmp_path: Path) -> None:
    payload = b"gene\tS1\nG1\t1.0\n"
    url = "https://example.test/counts.tsv"
    transport = FakeTransport(
        {url: FakeRoute(body=payload, headers={"content-type": "text/tab-separated-values"})}
    )
    downloader = Downloader(transport, DownloadPolicy(max_bytes=1024, retries=0))
    remote = RemoteFile(
        url=url,
        filename="counts.tsv",
        role=FileRole.MATRIX,
        official_checksum=__import__("hashlib").md5(payload).hexdigest(),
        official_checksum_alg=ChecksumAlg.MD5,
        source_api="test",
    )
    receipt = downloader.download(remote, tmp_path, dry_run=False)
    assert receipt.sha256 == sha256_bytes(payload)
    assert (tmp_path / "counts.tsv").read_bytes() == payload


def test_official_checksum_mismatch_rejects_file(tmp_path: Path) -> None:
    url = "https://example.test/bad.tsv"
    transport = FakeTransport({url: FakeRoute(body=b"not-the-expected-bytes")})
    downloader = Downloader(transport, DownloadPolicy(max_bytes=1024, retries=0))
    remote = RemoteFile(
        url=url,
        filename="bad.tsv",
        role=FileRole.MATRIX,
        official_checksum="deadbeef" * 4,
        official_checksum_alg=ChecksumAlg.SHA256,
        source_api="test",
    )
    with pytest.raises(ChecksumMismatchError, match="Official"):
        downloader.download(remote, tmp_path, dry_run=False)
    assert not (tmp_path / "bad.tsv").exists()
    assert (tmp_path / "bad.tsv.rejected").is_file()


def test_size_limit_aborts(tmp_path: Path) -> None:
    url = "https://example.test/huge.tsv"
    transport = FakeTransport({url: FakeRoute(body=b"0123456789")})
    downloader = Downloader(transport, DownloadPolicy(max_bytes=4, retries=0))
    remote = RemoteFile(url=url, filename="huge.tsv", role=FileRole.MATRIX, source_api="test")
    with pytest.raises(DownloadError, match="max_bytes"):
        downloader.download(remote, tmp_path, dry_run=False)


def test_resume_uses_range(tmp_path: Path) -> None:
    payload = b"ABCDEFGHIJ"
    url = "https://example.test/resume.tsv"
    transport = FakeTransport({url: FakeRoute(body=payload)})
    downloader = Downloader(transport, DownloadPolicy(max_bytes=100, retries=0, resume=True))
    partial = tmp_path / "resume.tsv.partial"
    partial.write_bytes(payload[:4])
    remote = RemoteFile(url=url, filename="resume.tsv", role=FileRole.MATRIX, source_api="test")
    receipt = downloader.download(remote, tmp_path, dry_run=False)
    assert receipt.resumed is True
    assert (tmp_path / "resume.tsv").read_bytes() == payload


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    url = "https://example.test/x.tsv"
    transport = FakeTransport({url: FakeRoute(body=b"abc")})
    downloader = Downloader(transport, DownloadPolicy(retries=0))
    remote = RemoteFile(url=url, filename="x.tsv", role=FileRole.MATRIX, source_api="test")
    receipt = downloader.download(remote, tmp_path, dry_run=True)
    assert receipt.dry_run is True
    assert not (tmp_path / "x.tsv").exists()


def test_retry_then_success() -> None:
    url = "https://example.test/eutils"

    class Flaky(FakeTransport):
        def __init__(self) -> None:
            super().__init__({})
            self.n = 0

        def request(self, method: str, url: str, *, headers=None, timeout=60.0) -> HttpResponse:  # type: ignore[no-untyped-def]
            self.n += 1
            self.calls.append((method, url))
            if self.n < 3:
                return HttpResponse(status=503, headers={}, body=b"busy", url=url)
            return HttpResponse(status=200, headers={}, body=b'{"ok":true}', url=url)

    transport = Flaky()
    downloader = Downloader(
        transport, DownloadPolicy(retries=3, retry_backoff_s=0.001, requests_per_host_per_sec=10)
    )
    response = downloader.fetch_json(url)
    assert response.status == 200
    assert transport.n == 3


def test_rate_limiter_sleeps() -> None:
    slept: list[float] = []
    limiter = HostRateLimiter(2.0, sleeper=slept.append)
    limiter.wait("https://a.test/x", now=10.0)
    limiter.wait("https://a.test/y", now=10.1)
    assert slept and slept[0] == pytest.approx(0.4)


def test_refuses_file_scheme() -> None:
    from omics_agent.data_sources.http import _assert_remote_url

    with pytest.raises(DownloadError, match="non-HTTP"):
        _assert_remote_url("file:///etc/passwd")
