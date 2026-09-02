"""Typed contracts for processed-data ingest.

Adapters produce and consume these objects only. They never invent
sample/time/biospecimen maps, and they never treat paper text as code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec, StrictModel
from omics_agent.schemas.enums import (
    ChecksumAlg,
    FileRole,
    ReviewStatus,
    SourceType,
)


class DownloadPolicy(StrictModel):
    """Hard limits for every network fetch.

    Attributes
    ----------
    max_bytes:
        Refuse a file whose Content-Length or on-disk size exceeds this.
    requests_per_host_per_sec:
        Client-side rate limit. NCBI E-utilities without an API key need ≤3/s;
        this default is 1/s so beginners do not get blocked.
    """

    max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)
    retries: int = Field(default=3, ge=0, le=8)
    retry_backoff_s: float = Field(default=1.5, gt=0)
    requests_per_host_per_sec: float = Field(default=1.0, gt=0, le=10)
    resume: bool = True
    user_agent: str = "omics-agent/0.1.0 (deterministic bulk multi-omics pipeline)"
    contact_email: str | None = None


class IngestRequest(StrictModel):
    """User entry for ``omics-agent ingest``.

    A paper DOI is provenance only. It is not a download locator and is
    never fetched as an instruction stream.
    """

    source: SourceType | None = None
    accession: str | None = None
    paper_doi: str | None = None
    url: str | None = None
    local_path: Path | None = None
    dest_dir: Path
    modality: str | None = None
    role: FileRole | None = None
    dry_run: bool = False
    resolve_only: bool = False
    policy: DownloadPolicy = Field(default_factory=DownloadPolicy)

    @model_validator(mode="after")
    def has_a_locator(self) -> Self:
        if self.source is SourceType.DOI or (
            self.paper_doi and not self.accession and not self.url and not self.local_path
        ):
            return self
        if not any([self.accession, self.url, self.local_path]):
            raise SchemaError(
                "Ingest needs an accession, a processed-matrix URL, or a local path.",
                how_to_fix=(
                    "Examples:\n"
                    "  omics-agent ingest --source geo --accession GSE12345 --dest outputs/ingest\n"
                    "  omics-agent ingest --source url --url https://.../counts.tsv --dest outputs/ingest\n"
                    "A paper DOI alone cannot locate files."
                ),
            )
        return self


class RemoteFile(StrictModel):
    """One file advertised by a repository API or a user URL."""

    url: str | None = None
    local_path: Path | None = None
    filename: str
    role: FileRole
    modality: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    official_checksum: str | None = None
    official_checksum_alg: ChecksumAlg | None = None
    source_api: str
    extra: dict[str, Any] = Field(default_factory=dict)


class DownloadReceipt(StrictModel):
    """On-disk result of one download or local snapshot. Raw dest is never overwritten in place."""

    filename: str
    dest: Path
    bytes_written: int = Field(ge=0)
    resumed: bool = False
    sha256: str | None = None
    official_checksum: str | None = None
    official_checksum_alg: ChecksumAlg | None = None
    dry_run: bool = False
    skipped_reason: str | None = None


class VerificationResult(StrictModel):
    """Checksum and policy check for one receipt."""

    filename: str
    ok: bool
    sha256: str | None = None
    official_ok: bool | None = None
    detail: str


class UntrustedText(StrictModel):
    """Repository or paper prose. Stored, never executed.

    The pipeline may show this to a reviewer. It must not be passed to a
    shell, ``eval``, or used as a file path.
    """

    source: str
    retrieved_at: str
    sha256: str
    text: str
    content_kind: str = Field(
        default="metadata",
        description="metadata|abstract|html|unknown. Never 'script'.",
    )


class ProvenanceRecord(StrictModel):
    """How and when repository metadata was obtained."""

    source: SourceType
    retrieved_at: str
    api_urls: list[str] = Field(default_factory=list)
    adapter: str
    notes: str | None = None


class IngestFile(StrictModel):
    """One file in an ingest manifest. ``modality`` may be unknown."""

    filename: str
    url: str | None = None
    path: Path | None = None
    sha256: str | None = None
    official_checksum: str | None = None
    official_checksum_alg: ChecksumAlg | None = None
    role: FileRole
    modality: str | None = None
    size_bytes: int | None = None
    skipped_reason: str | None = None


class IngestManifest(StrictModel):
    """Typed adapter output. May be incomplete; never silently filled.

    Promote to :class:`DatasetManifest` only after a person assigns design,
    pairing, modalities, and a sample sheet.
    """

    schema_version: str = "1.0"
    dataset_id: str
    title: str
    source_type: SourceType
    accession: str | None = None
    paper_doi: str | None = None
    landing_page: str | None = None
    license: LicenseSpec
    organism: OrganismSpec
    files: list[IngestFile] = Field(default_factory=list)
    sample_sheet: Path | None = None
    review_status: ReviewStatus = ReviewStatus.REQUIRED
    unresolved: list[str] = Field(default_factory=list)
    untrusted_text: list[UntrustedText] = Field(default_factory=list)
    provenance: ProvenanceRecord
    notes: str | None = None


class ReadinessLevel(StrictModel):
    """One gate in the data-readiness report."""

    name: str
    level: str = Field(description="green|yellow|red")
    detail: str
    how_to_fix: str


class DataReadinessReport(StrictModel):
    """QC gates. Red gates block training; they do not auto-repair."""

    dataset_id: str
    blocking: bool
    gates: list[ReadinessLevel]
    unresolved: list[str] = Field(default_factory=list)
