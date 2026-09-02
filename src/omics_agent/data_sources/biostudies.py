"""BioStudies / ArrayExpress processed-file adapter."""

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

_API = "https://www.ebi.ac.uk/biostudies/api/v1/studies"
_FILES = "https://www.ebi.ac.uk/biostudies/files"


@register_adapter
class BioStudiesAdapter:
    """BioStudies REST. ArrayExpress E-MTAB / E-GEOD accessions use this API."""

    source_type = SourceType.BIOSTUDIES

    def __init__(
        self,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def landing_page(self, accession: str) -> str:
        return f"https://www.ebi.ac.uk/biostudies/studies/{accession}"

    def resolve(self, request: IngestRequest) -> IngestManifest:
        acc = (request.accession or "").strip()
        if not acc:
            raise SchemaError(
                "BioStudies ingest requires an accession.",
                how_to_fix="Pass --accession E-MTAB-xxxx or S-BSSTxxxx.",
            )
        url = f"{_API}/{quote(acc, safe='')}"
        response = self.downloader.fetch_json(url)
        if response.status >= 400:
            raise SchemaError(
                f"BioStudies API returned HTTP {response.status} for {acc}.",
                how_to_fix="Check the accession. ArrayExpress studies use E-MTAB / E-GEOD / E-MEXP ids.",
            )
        payload = _as_dict(response.body)
        attrs = _collect_attributes(payload)
        title = attrs.get("title") or attrs.get("study title") or f"BioStudies {acc}"
        organism_name = attrs.get("organism") or attrs.get("species") or "undeclared"
        license_name = attrs.get("license") or attrs.get("licence") or "unknown"
        description = attrs.get("description") or attrs.get("abstract") or ""
        untrusted = []
        if description:
            untrusted.append(capture_untrusted(url, description, content_kind="metadata"))
        remotes = _files_from_payload(payload, acc)
        unresolved = [
            "Confirm sampling_design; BioStudies attributes do not define longitudinal vs RCS.",
            "Confirm experimental_unit_id, time, and biospecimen pairing on a sample sheet.",
            "Confirm each listed file is a processed matrix.",
        ]
        if license_name == "unknown":
            unresolved.append("Confirm the BioStudies/ArrayExpress license before redistribution.")
        return IngestManifest(
            dataset_id=f"biostudies_{acc.lower().replace('-', '_')}",
            title=str(title),
            source_type=SourceType.BIOSTUDIES,
            accession=acc,
            paper_doi=request.paper_doi,
            landing_page=self.landing_page(acc),
            license=LicenseSpec(
                name=str(license_name),
                redistributable=False,
                notes="License taken from study attributes when present; still needs human confirmation.",
            ),
            organism=OrganismSpec(
                name=str(organism_name), taxon_id=taxon_id_or_none(str(organism_name))
            ),
            files=[
                IngestFile(
                    filename=item.filename,
                    url=item.url,
                    role=item.role,
                    modality=item.modality,
                    size_bytes=item.size_bytes,
                    official_checksum=item.official_checksum,
                    official_checksum_alg=item.official_checksum_alg,
                    skipped_reason="raw sequencing / raw mass-spec; not downloaded"
                    if item.role is FileRole.REJECTED_RAW
                    else None,
                )
                for item in remotes
            ],
            review_status=ReviewStatus.REQUIRED,
            unresolved=unresolved,
            untrusted_text=untrusted,
            provenance=ProvenanceRecord(
                source=SourceType.BIOSTUDIES,
                retrieved_at=_now_iso(),
                api_urls=[url],
                adapter="biostudies",
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
                source_api="biostudies",
            )
            for item in manifest.files
            if item.url
        ]


def _files_from_payload(payload: dict[str, Any], accession: str) -> list[RemoteFile]:
    found: list[RemoteFile] = []
    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        path = _file_path(node)
        if path is None:
            continue
        filename = path.split("/")[-1]
        if not filename or filename.endswith("/"):
            continue
        size = node.get("size") or node.get("sizeInBytes")
        checksum, alg = _checksum_from_node(node)
        role = classify_filename(filename)
        rel = str(node.get("path") or filename)
        found.append(
            RemoteFile(
                url=f"{_FILES}/{accession}/{rel.lstrip('/')}",
                filename=filename,
                role=role,
                size_bytes=int(size) if isinstance(size, int) else None,
                official_checksum=checksum,
                official_checksum_alg=alg,
                source_api="biostudies",
            )
        )
    return found


def _file_path(node: dict[str, Any]) -> str | None:
    """Accept BioStudies file objects that have a path, or a name plus size/type."""

    path = node.get("path") or node.get("name")
    if not isinstance(path, str) or not path:
        return None
    if "path" in node:
        return path
    if "name" in node and ("size" in node or "type" in node):
        return path
    return None


def _checksum_from_node(node: dict[str, Any]) -> tuple[str | None, ChecksumAlg | None]:
    for key, alg in (("sha256", ChecksumAlg.SHA256), ("md5", ChecksumAlg.MD5)):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), alg
    attributes = node.get("attributes")
    if isinstance(attributes, list):
        mapped = {str(item.get("name", "")).lower(): str(item.get("value", "")) for item in attributes if isinstance(item, dict)}
        if mapped.get("sha256"):
            return mapped["sha256"], ChecksumAlg.SHA256
        if mapped.get("md5"):
            return mapped["md5"], ChecksumAlg.MD5
    return None, None


def _collect_attributes(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in _walk(payload):
        if not isinstance(node, dict):
            continue
        attrs = node.get("attributes")
        if not isinstance(attrs, list):
            continue
        for item in attrs:
            if isinstance(item, dict) and "name" in item and "value" in item:
                out[str(item["name"]).strip().lower()] = str(item["value"])
    return out


def _walk(node: Any) -> list[Any]:
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk(item))
    return found


def _as_dict(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
