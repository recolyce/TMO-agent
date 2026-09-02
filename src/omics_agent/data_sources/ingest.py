"""Ingest orchestrator: resolve → optional download → readiness report.

Adapters never auto-approve a manifest. Raw files are skipped, not fetched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from omics_agent.data_sources.base import get_adapter, infer_source
from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
from omics_agent.errors import DownloadError, UnsupportedRawDataError
from omics_agent.reporting.readiness import write_readiness_report
from omics_agent.schemas.enums import FileRole
from omics_agent.schemas.ingest import (
    DownloadReceipt,
    IngestManifest,
    IngestRequest,
    RemoteFile,
)


def run_ingest(
    request: IngestRequest,
    *,
    transport: HttpTransport | None = None,
) -> dict[str, Any]:
    """Resolve a typed ingest manifest and optionally download processed files.

    Parameters
    ----------
    request:
        User locator (accession / URL / local path) plus dest and policy.
    transport:
        Injected HTTP transport. Tests pass a fake; production uses urllib.
    """

    source = infer_source(request)
    dest = request.dest_dir
    if not request.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    policy = request.policy
    http = transport or UrllibTransport()
    adapter = get_adapter(source, transport=http, policy=policy)
    manifest = adapter.resolve(request)

    plan: dict[str, Any] = {
        "dataset_id": manifest.dataset_id,
        "source": source.value,
        "review_status": manifest.review_status.value,
        "n_files": len(manifest.files),
        "n_raw_skipped": sum(1 for item in manifest.files if item.role is FileRole.REJECTED_RAW),
        "unresolved": list(manifest.unresolved),
        "dry_run": request.dry_run,
        "dest_dir": str(dest),
    }
    ingest_path = dest / "ingest_manifest.yaml"
    if request.dry_run:
        plan["would_write"] = [str(ingest_path), str(dest / "data_readiness_report.html")]
        return plan

    _write_yaml(ingest_path, manifest.model_dump(mode="json"))
    receipts: list[DownloadReceipt] = []
    if not request.resolve_only:
        downloader = Downloader(http, policy)
        raw_dir = dest / "raw"
        for remote in _downloadable(adapter.list_files(manifest)):
            try:
                receipt = downloader.download(remote, raw_dir, dry_run=False)
            except UnsupportedRawDataError:
                continue
            receipts.append(receipt)
            _apply_receipt(manifest, receipt)
        _write_yaml(ingest_path, manifest.model_dump(mode="json"))

    report = write_readiness_report(manifest, dest / "data_readiness_report.html")
    plan.update(
        {
            "ingest_manifest": str(ingest_path),
            "readiness_html": str(dest / "data_readiness_report.html"),
            "readiness_json": str(dest / "data_readiness_report.json"),
            "n_downloaded": len(receipts),
            "blocking": report.blocking,
        }
    )
    return plan


def _downloadable(files: list[RemoteFile]) -> list[RemoteFile]:
    keep: list[RemoteFile] = []
    for item in files:
        if item.role is FileRole.REJECTED_RAW:
            continue
        if item.role is FileRole.ARCHIVE:
            continue
        if item.url is None and item.local_path is None:
            continue
        keep.append(item)
    return keep


def _apply_receipt(manifest: IngestManifest, receipt: DownloadReceipt) -> None:
    for item in manifest.files:
        if item.filename == receipt.filename:
            item.path = receipt.dest
            item.sha256 = receipt.sha256
            item.size_bytes = receipt.bytes_written


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_ingest_manifest(path: Path) -> IngestManifest:
    if not path.is_file():
        raise DownloadError(
            f"Ingest manifest not found: {path}",
            how_to_fix="Run omics-agent ingest first.",
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise DownloadError(
            f"{path} is not a YAML mapping.",
            how_to_fix="Use the ingest_manifest.yaml written by omics-agent ingest.",
        )
    return IngestManifest.model_validate(payload)
