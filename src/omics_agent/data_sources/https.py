"""Generic HTTPS / user-supplied URL adapter for one processed file."""

from __future__ import annotations

from urllib.parse import unquote, urlparse

from omics_agent.data_sources.base import dataset_id_for, register_adapter
from omics_agent.data_sources.classify import classify_filename, reject_raw
from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
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


@register_adapter
class HttpsAdapter:
    """One user-provided HTTPS URL. Paper HTML is not fetched as a locator."""

    source_type = SourceType.URL

    def __init__(
        self,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def landing_page(self, accession: str) -> str:
        return accession

    def resolve(self, request: IngestRequest) -> IngestManifest:
        if not request.url:
            raise SchemaError(
                "URL ingest requires --url.",
                how_to_fix="Pass a direct https:// link to a processed matrix, not a paper landing page.",
            )
        parsed = urlparse(request.url)
        if parsed.scheme not in {"http", "https"}:
            raise SchemaError(
                f"Refusing scheme '{parsed.scheme}'.",
                how_to_fix="Use https:// or http://. file:// paths belong to --local-path.",
            )
        filename = unquote(parsed.path.rsplit("/", 1)[-1]) or "downloaded_file"
        reject_raw(filename)
        role = request.role or classify_filename(filename)
        if role is FileRole.UNKNOWN and request.role is None:
            role = FileRole.UNKNOWN
        unresolved = [
            "Confirm this URL is a processed matrix and not an HTML paper or a wrapper page.",
            "Confirm organism, sampling_design, experimental units, time, and pairing.",
            "Confirm the license of the remote file.",
        ]
        if request.modality is None:
            unresolved.append("Assign a modality (rna, protein, ...) to this file.")
        if role is FileRole.UNKNOWN:
            unresolved.append(f"Confirm the role of '{filename}' (matrix vs sample sheet vs other).")
        remote = RemoteFile(
            url=request.url,
            filename=filename,
            role=role,
            modality=request.modality,
            source_api="user-url",
        )
        return IngestManifest(
            dataset_id=dataset_id_for(SourceType.URL, request.accession, filename),
            title=f"URL ingest {filename}",
            source_type=SourceType.URL,
            accession=request.accession,
            paper_doi=request.paper_doi,
            landing_page=request.url,
            license=LicenseSpec(name="unknown", redistributable=False),
            organism=OrganismSpec(name="undeclared", taxon_id=None),
            files=[
                IngestFile(
                    filename=remote.filename,
                    url=remote.url,
                    role=remote.role,
                    modality=remote.modality,
                )
            ],
            review_status=ReviewStatus.REQUIRED,
            unresolved=unresolved,
            provenance=ProvenanceRecord(
                source=SourceType.URL,
                retrieved_at=_now_iso(),
                api_urls=[request.url],
                adapter="https",
                notes="The URL body is downloaded as a file. HTML is not parsed for further links.",
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
                source_api="user-url",
            )
            for item in manifest.files
            if item.url
        ]


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
