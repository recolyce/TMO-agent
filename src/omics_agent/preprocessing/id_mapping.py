"""ID-mapping adapters: identity, static table, and mygene.info.

All external calls go through the injectable :class:`HttpTransport`, so
tests mock the API and CI never touches the network. An API failure is an
error, not an empty map — silence would look like "no ortholog exists".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import quote

import pandas as pd

from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
from omics_agent.errors import DownloadError, SchemaError
from omics_agent.schemas.features import FeatureMap, FeatureMapping, FeatureTarget
from omics_agent.schemas.ingest import DownloadPolicy

_MYGENE = "https://mygene.info/v3/query"


class IdMappingAdapter(Protocol):
    """Maps source feature IDs to zero-or-more target IDs."""

    name: str
    target_id_type: str

    def map_ids(self, ids: Sequence[str]) -> dict[str, list[str]]:
        """Return every target for every input ID. Unmapped IDs map to []."""


class IdentityMapper:
    """No external mapping: each feature keeps its own ID."""

    name = "identity"

    def __init__(self, target_id_type: str) -> None:
        self.target_id_type = target_id_type

    def map_ids(self, ids: Sequence[str]) -> dict[str, list[str]]:
        return {str(item): [str(item)] for item in ids}


class StaticTableIdMapper:
    """Offline mapping from a curated table (one row per source→target pair)."""

    name = "static_table"

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        source_column: str = "source_id",
        target_column: str = "target_id",
        target_id_type: str = "undeclared",
    ) -> None:
        missing = [col for col in (source_column, target_column) if col not in table.columns]
        if missing:
            raise SchemaError(
                f"ID-map table is missing columns {missing}.",
                how_to_fix=(
                    "Provide a TSV with source_id and target_id columns. "
                    "A source with several targets uses several rows."
                ),
            )
        self.target_id_type = target_id_type
        self._map: dict[str, list[str]] = {}
        for source, target in zip(
            table[source_column].astype(str), table[target_column].astype(str), strict=True
        ):
            bucket = self._map.setdefault(source, [])
            if target not in bucket:
                bucket.append(target)

    def map_ids(self, ids: Sequence[str]) -> dict[str, list[str]]:
        return {str(item): list(self._map.get(str(item), [])) for item in ids}


class MyGeneInfoAdapter:
    """mygene.info query adapter. Rate-limited, retried, and fully mockable.

    One GET per identifier. The species taxon ID is required so the adapter
    never guesses the organism.
    """

    name = "mygene.info"

    def __init__(
        self,
        *,
        species_taxon_id: int,
        scopes: str = "symbol",
        fields: str = "ensembl.gene",
        target_id_type: str = "ensembl_gene_id",
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.species_taxon_id = species_taxon_id
        self.scopes = scopes
        self.fields = fields
        self.target_id_type = target_id_type
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def query_url(self, identifier: str) -> str:
        return (
            f"{_MYGENE}?q={quote(identifier, safe='')}&scopes={quote(self.scopes, safe='')}"
            f"&fields={quote(self.fields, safe='')}&species={self.species_taxon_id}&size=10"
        )

    def map_ids(self, ids: Sequence[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        path = self.fields.split(".")
        for identifier in ids:
            url = self.query_url(str(identifier))
            response = self.downloader.fetch_json(url)
            if response.status >= 400:
                raise DownloadError(
                    f"mygene.info returned HTTP {response.status} for '{identifier}'.",
                    how_to_fix=(
                        "Retry later or provide a static --id-map table. An API failure "
                        "must not be recorded as 'feature has no mapping'."
                    ),
                )
            payload = _as_dict(response.body)
            targets: list[str] = []
            for hit in payload.get("hits") or []:
                for value in _extract_values(hit, path):
                    if value not in targets:
                        targets.append(value)
            out[str(identifier)] = targets
        return out


def build_feature_map(
    *,
    modality: str,
    source_ids: Sequence[str],
    source_id_type: str,
    adapter: IdMappingAdapter,
) -> FeatureMap:
    """Run the adapter over all feature IDs and keep one-to-many explicit."""

    mapped = adapter.map_ids(source_ids)
    mappings = [
        FeatureMapping(
            source_id=str(source),
            targets=[
                FeatureTarget(target_id=target, target_id_type=adapter.target_id_type)
                for target in mapped.get(str(source), [])
            ],
        )
        for source in source_ids
    ]
    return FeatureMap(
        modality=modality,
        source_id_type=source_id_type,
        target_id_type=adapter.target_id_type,
        mapping_source=adapter.name,
        retrieved_at=datetime.now(UTC).isoformat(),
        mappings=mappings,
    )


def _extract_values(node: Any, path: list[str]) -> list[str]:
    if isinstance(node, list):
        out: list[str] = []
        for item in node:
            out.extend(_extract_values(item, path))
        return out
    if not path:
        if isinstance(node, str | int):
            return [str(node)]
        return []
    if isinstance(node, dict):
        return _extract_values(node.get(path[0]), path[1:])
    return []


def _as_dict(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
