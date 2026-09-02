"""Local processed-file adapter. Does not walk arbitrary trees for FASTQ."""

from __future__ import annotations

from omics_agent.data_sources.base import dataset_id_for, register_adapter
from omics_agent.data_sources.classify import classify_filename, reject_raw
from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec
from omics_agent.schemas.enums import FileRole, ReviewStatus, SourceType
from omics_agent.schemas.ingest import (
    IngestFile,
    IngestManifest,
    IngestRequest,
    ProvenanceRecord,
    RemoteFile,
)

_ALLOWED_LOCAL = {".tsv", ".csv", ".txt", ".tab", ".parquet"}


@register_adapter
class LocalFileAdapter:
    """Snapshot one local processed table into the ingest dest."""

    source_type = SourceType.LOCAL

    def __init__(self, transport: object | None = None, policy: object | None = None) -> None:
        del transport, policy

    def landing_page(self, accession: str) -> str:
        return accession

    def resolve(self, request: IngestRequest) -> IngestManifest:
        if request.local_path is None:
            raise SchemaError(
                "Local ingest requires --local-path.",
                how_to_fix="Point at a processed matrix file, not a FASTQ directory.",
            )
        path = request.local_path.expanduser().resolve()
        if path.is_dir():
            raise SchemaError(
                f"{path} is a directory. Milestone 2 will not crawl it for files.",
                how_to_fix=(
                    "Pass one processed matrix file. Crawling a folder would risk picking up "
                    "FASTQ or install scripts."
                ),
            )
        if not path.is_file():
            raise SchemaError(
                f"Local file not found: {path}",
                how_to_fix="Check the path. Use a .tsv/.csv/.parquet processed matrix.",
            )
        reject_raw(path.name)
        if path.suffix.lower() not in _ALLOWED_LOCAL:
            raise SchemaError(
                f"Local file suffix '{path.suffix}' is not a processed table.",
                how_to_fix="Use .tsv, .csv, .txt, .tab, or .parquet. Archives are not auto-extracted.",
            )
        role = request.role or classify_filename(path.name)
        unresolved = [
            "Confirm sampling_design, experimental units, time, and pairing on a sample sheet.",
            "Confirm the license of this local file.",
        ]
        if request.modality is None:
            unresolved.append("Assign a modality to the local file.")
        return IngestManifest(
            dataset_id=dataset_id_for(SourceType.LOCAL, request.accession, path.name),
            title=f"Local ingest {path.name}",
            source_type=SourceType.LOCAL,
            accession=request.accession,
            paper_doi=request.paper_doi,
            landing_page=str(path),
            license=LicenseSpec(name="unknown", redistributable=False),
            organism=OrganismSpec(name="undeclared", taxon_id=None),
            files=[
                IngestFile(
                    filename=path.name,
                    path=path,
                    role=role if role is not FileRole.UNKNOWN else FileRole.MATRIX,
                    modality=request.modality,
                )
            ],
            review_status=ReviewStatus.REQUIRED,
            unresolved=unresolved,
            provenance=ProvenanceRecord(
                source=SourceType.LOCAL,
                retrieved_at=_now_iso(),
                api_urls=[],
                adapter="local",
            ),
        )

    def list_files(self, manifest: IngestManifest) -> list[RemoteFile]:
        return [
            RemoteFile(
                local_path=item.path,
                filename=item.filename,
                role=item.role,
                modality=item.modality,
                official_checksum=item.official_checksum,
                official_checksum_alg=item.official_checksum_alg,
                source_api="local",
            )
            for item in manifest.files
            if item.path is not None
        ]


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
