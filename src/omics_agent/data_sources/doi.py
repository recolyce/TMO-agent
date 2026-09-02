"""DOI-only ingest. A paper identifier is not a file locator.

The adapter never fetches the landing page or PDF. Paper HTML/PDF can
contain instructions; those must not become download URLs or shell commands.
"""

from __future__ import annotations

from omics_agent.data_sources.base import register_adapter
from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec
from omics_agent.schemas.enums import ReviewStatus, SourceType
from omics_agent.schemas.ingest import IngestManifest, IngestRequest, ProvenanceRecord, RemoteFile


@register_adapter
class DoiAdapter:
    """Record a DOI as provenance and stop for review."""

    source_type = SourceType.DOI

    def __init__(self, transport: object | None = None, policy: object | None = None) -> None:
        del transport, policy

    def landing_page(self, accession: str) -> str:
        return f"https://doi.org/{accession}"

    def resolve(self, request: IngestRequest) -> IngestManifest:
        doi = (request.paper_doi or "").strip()
        if not doi:
            raise SchemaError(
                "DOI ingest requires --paper-doi.",
                how_to_fix="Pass a DOI together with a GEO/PRIDE/BioStudies accession or a matrix URL.",
            )
        return IngestManifest(
            dataset_id="doi_" + "".join(ch if ch.isalnum() else "_" for ch in doi)[:40],
            title=f"DOI {doi} (no files resolved)",
            source_type=SourceType.DOI,
            paper_doi=doi,
            landing_page=self.landing_page(doi),
            license=LicenseSpec(name="unknown", redistributable=False),
            organism=OrganismSpec(name="undeclared", taxon_id=None),
            files=[],
            review_status=ReviewStatus.REQUIRED,
            unresolved=[
                "A paper DOI is not a file locator. Provide a GEO/PRIDE/BioStudies accession "
                "or a direct processed-matrix URL.",
                "The pipeline will not fetch or parse the paper as executable instructions.",
                "Confirm sample/time/modality/biospecimen mapping after files are identified.",
            ],
            provenance=ProvenanceRecord(
                source=SourceType.DOI,
                retrieved_at=_now_iso(),
                api_urls=[],
                adapter="doi",
                notes="No HTTP request was made. Paper text is untrusted and was not retrieved.",
            ),
        )

    def list_files(self, manifest: IngestManifest) -> list[RemoteFile]:
        del manifest
        return []


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
