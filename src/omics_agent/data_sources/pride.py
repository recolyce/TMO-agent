"""PRIDE Archive processed-file adapter.

RAW / PEAK / vendor mass-spec files are listed as rejected_raw and are
not downloaded. SEARCH / RESULT tables may still need human review.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from omics_agent.data_sources.base import register_adapter
from omics_agent.data_sources.classify import classify_filename
from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
from omics_agent.data_sources.taxonomy import taxon_id_or_none
from omics_agent.data_sources.untrusted import capture_untrusted
from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec
from omics_agent.schemas.enums import ChecksumAlg, FileRole, ReviewStatus, SourceType
from omics_agent.schemas.ingest import (
    DownloadPolicy,
    IngestFile,
    IngestManifest,
    IngestRequest,
    ProvenanceRecord,
    RemoteFile,
)

_PROJECT = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
_FILES = "https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject"
_RAW_CATEGORIES = {"RAW", "PEAK", "SPECTRUM", "MS_RAW"}


@register_adapter
class PrideAdapter:
    """PRIDE Archive REST v2."""

    source_type = SourceType.PRIDE

    def __init__(
        self,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def landing_page(self, accession: str) -> str:
        return f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"

    def resolve(self, request: IngestRequest) -> IngestManifest:
        acc = (request.accession or "").strip().upper()
        if not acc.startswith("PXD"):
            raise SchemaError(
                f"'{request.accession}' is not a PRIDE project accession.",
                how_to_fix="Use a PXD accession such as PXD000001.",
            )
        project_url = f"{_PROJECT}/{quote(acc)}"
        files_url = f"{_FILES}?accession={quote(acc)}"
        project_resp = self.downloader.fetch_json(project_url)
        if project_resp.status >= 400:
            raise SchemaError(
                f"PRIDE API returned HTTP {project_resp.status} for {acc}.",
                how_to_fix="Check the PXD accession on https://www.ebi.ac.uk/pride/archive .",
            )
        project = _as_dict(project_resp.body)
        title = str(project.get("title") or f"PRIDE {acc}")
        organisms = project.get("organisms") or project.get("species") or []
        organism_name = "undeclared"
        if isinstance(organisms, list) and organisms:
            first = organisms[0]
            if isinstance(first, dict):
                organism_name = str(first.get("name") or first.get("value") or "undeclared")
            else:
                organism_name = str(first)
        license_name = str(project.get("license") or project.get("dataProcessingProtocol") or "unknown")
        if len(license_name) > 80:
            license_name = "unknown"
        abstract = str(project.get("projectDescription") or project.get("description") or "")
        untrusted = []
        if abstract:
            untrusted.append(capture_untrusted(project_url, abstract, content_kind="metadata"))
        files_resp = self.downloader.fetch_json(files_url)
        remotes = _files_from_pride(_as_list_or_embedded(files_resp.body))
        unresolved = [
            "Confirm sampling_design and experimental units from the PRIDE sample metadata.",
            "Confirm pairing_level; a protein table is not automatically paired to RNA.",
            "Confirm which SEARCH/RESULT files are quantified protein matrices.",
            "Confirm the project license before redistribution.",
        ]
        return IngestManifest(
            dataset_id=f"pride_{acc.lower()}",
            title=title,
            source_type=SourceType.PRIDE,
            accession=acc,
            paper_doi=request.paper_doi,
            landing_page=self.landing_page(acc),
            license=LicenseSpec(name=license_name, redistributable=False),
            organism=OrganismSpec(name=organism_name, taxon_id=taxon_id_or_none(organism_name)),
            files=[
                IngestFile(
                    filename=item.filename,
                    url=item.url,
                    role=item.role,
                    modality=item.modality or ("protein" if item.role is FileRole.MATRIX else None),
                    size_bytes=item.size_bytes,
                    official_checksum=item.official_checksum,
                    official_checksum_alg=item.official_checksum_alg,
                    skipped_reason="raw mass-spec / peak list; not downloaded"
                    if item.role is FileRole.REJECTED_RAW
                    else None,
                )
                for item in remotes
            ],
            review_status=ReviewStatus.REQUIRED,
            unresolved=unresolved,
            untrusted_text=untrusted,
            provenance=ProvenanceRecord(
                source=SourceType.PRIDE,
                retrieved_at=_now_iso(),
                api_urls=[project_url, files_url],
                adapter="pride",
            ),
        )

    def list_files(self, manifest: IngestManifest) -> list[RemoteFile]:
        return [
            RemoteFile(
                url=item.url,
                filename=item.filename,
                role=item.role,
                modality=item.modality,
                size_bytes=item.size_bytes,
                official_checksum=item.official_checksum,
                official_checksum_alg=item.official_checksum_alg,
                source_api="pride",
            )
            for item in manifest.files
            if item.url
        ]


def _files_from_pride(rows: list[Any]) -> list[RemoteFile]:
    found: list[RemoteFile] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("fileName") or row.get("filename") or "")
        if not name:
            continue
        category = str(row.get("fileCategory") or row.get("category") or "").upper()
        role = classify_filename(name)
        if category in _RAW_CATEGORIES:
            role = FileRole.REJECTED_RAW
        checksum = row.get("checksum") or row.get("md5") or row.get("sha256")
        alg = None
        if row.get("sha256"):
            alg = ChecksumAlg.SHA256
            checksum = row.get("sha256")
        elif checksum:
            alg = ChecksumAlg.MD5
        size = row.get("fileSizeBytes") or row.get("fileSize")
        url = row.get("downloadLink") or row.get("ftpLocation") or row.get("publicFileLocations")
        if isinstance(url, list) and url:
            first = url[0]
            url = first.get("value") if isinstance(first, dict) else first
        if not isinstance(url, str) or not url.startswith("http"):
            url = None
        found.append(
            RemoteFile(
                url=url,
                filename=name,
                role=role,
                size_bytes=int(size) if isinstance(size, int) else None,
                official_checksum=str(checksum) if isinstance(checksum, str) else None,
                official_checksum_alg=alg,
                source_api="pride",
                extra={"fileCategory": category},
            )
        )
    return found


def _as_dict(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list_or_embedded(body: bytes) -> list[Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        embedded = payload.get("_embedded") or payload.get("files")
        if isinstance(embedded, dict):
            files = embedded.get("files")
            if isinstance(files, list):
                return files
        if isinstance(embedded, list):
            return embedded
    return []


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
