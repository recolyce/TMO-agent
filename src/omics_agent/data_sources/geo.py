"""GEO processed-matrix adapter (E-utilities + GEO HTTPS).

Sample/time/pairing are never inferred from the SOFT summary. Those fields
stay undeclared until a person fills the sample sheet.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

from omics_agent.data_sources.base import register_adapter
from omics_agent.data_sources.classify import classify_filename
from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
from omics_agent.data_sources.taxonomy import taxon_id_or_none
from omics_agent.data_sources.untrusted import capture_untrusted
from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec
from omics_agent.schemas.enums import FileRole, ReviewStatus, SourceType
from omics_agent.schemas.ingest import (
    DownloadPolicy,
    IngestFile,
    IngestManifest,
    IngestRequest,
    ProvenanceRecord,
    RemoteFile,
)

_GSE = re.compile(r"^GSE\d+$", re.IGNORECASE)
_GDS = re.compile(r"^GDS\d+$", re.IGNORECASE)
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


@register_adapter
class GeoAdapter:
    """GEO series (GSE) processed supplements and series matrices."""

    source_type = SourceType.GEO

    def __init__(
        self,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def landing_page(self, accession: str) -> str:
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"

    def resolve(self, request: IngestRequest) -> IngestManifest:
        acc = (request.accession or "").strip()
        if _GDS.match(acc):
            raise SchemaError(
                f"{acc} is a GEO DataSet (GDS), not a Series.",
                how_to_fix="Open the GDS page, find the GSE accession, and pass that.",
            )
        if not _GSE.match(acc):
            raise SchemaError(
                f"'{acc}' is not a GSE accession.",
                how_to_fix="Use a Series id such as GSE12345. GSM sample ids are not enough.",
            )
        acc = acc.upper()
        api_urls: list[str] = []
        unresolved: list[str] = [
            "Confirm sampling_design (longitudinal vs repeated cross-sectional).",
            "Confirm experimental_unit_id / subject_id / biospecimen_id on a sample sheet.",
            "Confirm pairing_level; GEO metadata does not prove RNA–protein pairing.",
            "Confirm the license allows the intended use and redistribution.",
            "Confirm each supplementary file is a processed matrix, not raw reads.",
        ]
        search_url = (
            f"{_EUTILS}/esearch.fcgi?db=gds&term={acc}[ACCN]&retmode=json&retmax=1&tool=omics_agent"
        )
        if self.downloader.policy.contact_email:
            search_url += f"&email={self.downloader.policy.contact_email}"
        search = self.downloader.fetch_json(search_url)
        api_urls.append(search_url)
        uids = _json_uids(search.body)
        summary_payload: dict[str, Any] = {}
        untrusted = []
        if uids:
            summary_url = (
                f"{_EUTILS}/esummary.fcgi?db=gds&id={uids[0]}&retmode=json&tool=omics_agent"
            )
            if self.downloader.policy.contact_email:
                summary_url += f"&email={self.downloader.policy.contact_email}"
            summary = self.downloader.fetch_json(summary_url)
            api_urls.append(summary_url)
            summary_payload = _json_load(summary.body)
            record = (summary_payload.get("result") or {}).get(uids[0], {})
            title = str(record.get("title") or f"GEO {acc}")
            organism_name = str(record.get("taxon") or "undeclared")
            abstract = str(record.get("summary") or "")
            if abstract:
                untrusted.append(capture_untrusted(summary_url, abstract, content_kind="metadata"))
        else:
            title = f"GEO {acc}"
            organism_name = "undeclared"
            unresolved.append(
                f"E-utilities did not return a UID for {acc}. Confirm the accession."
            )
        files = self._list_geo_files(acc)
        if not files:
            unresolved.append(
                f"No processed files listed under the GEO HTTPS directories for {acc}."
            )
        ingest_files = [
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
            for item in files
        ]
        return IngestManifest(
            dataset_id=f"geo_{acc.lower()}",
            title=title,
            source_type=SourceType.GEO,
            accession=acc,
            paper_doi=request.paper_doi,
            landing_page=self.landing_page(acc),
            license=LicenseSpec(
                name="unknown",
                redistributable=False,
                notes="GEO public accessions are not automatically redistributable. A person must confirm.",
            ),
            organism=OrganismSpec(name=organism_name, taxon_id=taxon_id_or_none(organism_name)),
            files=ingest_files,
            review_status=ReviewStatus.REQUIRED,
            unresolved=unresolved,
            untrusted_text=untrusted,
            provenance=ProvenanceRecord(
                source=SourceType.GEO,
                retrieved_at=untrusted[0].retrieved_at if untrusted else _now_iso(),
                api_urls=api_urls,
                adapter="geo",
                notes="SOFT/eutils text is stored as untrusted metadata and is not executed.",
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
                source_api="geo",
            )
            for item in manifest.files
            if item.url
        ]

    def _list_geo_files(self, acc: str) -> list[RemoteFile]:
        prefix = _geo_series_prefix(acc)
        bases = [
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/matrix/",
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/suppl/",
        ]
        found: list[RemoteFile] = []
        for base in bases:
            listing = self.downloader.fetch_json(base)
            if listing.status >= 400:
                continue
            parser = _HrefParser()
            parser.feed(listing.body.decode("utf-8", errors="replace"))
            for href in parser.hrefs:
                name = href.split("/")[-1]
                if not name or name.endswith("/") or name in {"index.html", ".."}:
                    continue
                role = classify_filename(name)
                found.append(
                    RemoteFile(
                        url=urljoin(base, name),
                        filename=name,
                        role=role,
                        source_api="geo-https",
                    )
                )
        return found


def _geo_series_prefix(acc: str) -> str:
    digits = acc[3:]
    if len(digits) <= 3:
        return "GSEnnn"
    return f"GSE{digits[:-3]}nnn"


def _json_load(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_uids(body: bytes) -> list[str]:
    payload = _json_load(body)
    idlist = ((payload.get("esearchresult") or {}).get("idlist")) or []
    return [str(item) for item in idlist]


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
