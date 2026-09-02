"""Adapter protocol and registry. Unknown sources fail; they do not no-op."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from omics_agent.errors import SchemaError
from omics_agent.schemas.enums import SourceType
from omics_agent.schemas.ingest import IngestManifest, IngestRequest, RemoteFile

_REGISTRY: dict[SourceType, type[DataSourceAdapter]] = {}


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Resolve repository metadata into a typed ingest manifest and file list."""

    source_type: SourceType

    def resolve(self, request: IngestRequest) -> IngestManifest:
        """Build a typed ingest manifest. Incomplete biology → needs_review."""

    def list_files(self, manifest: IngestManifest) -> list[RemoteFile]:
        """Return files referenced by the manifest. Does not download."""

    def landing_page(self, accession: str) -> str:
        """Human-facing URL for the accession."""


def register_adapter(cls: type[DataSourceAdapter]) -> type[DataSourceAdapter]:
    _REGISTRY[cls.source_type] = cls
    return cls


def _ensure_registered() -> None:
    from omics_agent.data_sources import biostudies as _biostudies
    from omics_agent.data_sources import doi as _doi
    from omics_agent.data_sources import geo as _geo
    from omics_agent.data_sources import https as _https
    from omics_agent.data_sources import local_files as _local
    from omics_agent.data_sources import pride as _pride

    del _biostudies, _doi, _geo, _https, _local, _pride


def get_adapter(source: SourceType, **kwargs: object) -> DataSourceAdapter:
    _ensure_registered()
    if source is SourceType.ARRAYEXPRESS:
        source = SourceType.BIOSTUDIES
    if source is SourceType.SRA or source is SourceType.MW:
        raise SchemaError(
            f"Source '{source.value}' is not implemented in milestone 2.",
            how_to_fix=(
                "SRA/raw reads and Metabolomics Workbench are out of scope. "
                "Provide a processed matrix via GEO, BioStudies, PRIDE, HTTPS, or --local-path."
            ),
        )
    if source not in _REGISTRY:
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise SchemaError(
            f"No ingest adapter for source '{source.value}'.",
            how_to_fix=f"Use one of: {known}.",
        )
    factory: Any = _REGISTRY[source]
    return factory(**kwargs)


def infer_source(request: IngestRequest) -> SourceType:
    """Infer source from an explicit flag or an accession prefix. Never from a DOI."""

    if request.source is not None and request.source is not SourceType.DOI:
        return request.source
    if request.local_path is not None and request.url is None and request.accession is None:
        return SourceType.LOCAL
    if request.url and not request.accession:
        return SourceType.URL
    acc = (request.accession or "").strip().upper()
    if acc.startswith("GSE") or acc.startswith("GDS"):
        return SourceType.GEO
    if acc.startswith("PXD"):
        return SourceType.PRIDE
    if acc.startswith("E-") or acc.startswith("S-"):
        return SourceType.BIOSTUDIES
    if request.paper_doi and not acc and not request.url:
        return SourceType.DOI
    raise SchemaError(
        "Cannot infer the repository from this request.",
        how_to_fix="Pass --source geo|biostudies|pride|url|local and an accession or URL.",
    )


def dataset_id_for(source: SourceType, accession: str | None, filename: str | None) -> str:
    raw = accession or (Path(filename).stem if filename else "ingest")
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return f"{source.value}_{cleaned}".lower()
