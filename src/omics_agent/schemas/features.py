"""Feature identifier maps.

A source ID may map to zero, one, or many target IDs. One-to-many
mappings are kept explicit — they are never collapsed by picking the
first hit — and unmapped features are recorded, not dropped silently.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import Field

from omics_agent.schemas.dataset import StrictModel


class FeatureTarget(StrictModel):
    """One mapped target identifier."""

    target_id: str
    target_id_type: str


class FeatureMapping(StrictModel):
    """All targets for one source feature ID."""

    source_id: str
    targets: list[FeatureTarget] = Field(default_factory=list)

    @property
    def is_unmapped(self) -> bool:
        return not self.targets

    @property
    def is_ambiguous(self) -> bool:
        return len(self.targets) > 1


class FeatureMap(StrictModel):
    """Typed map for one modality's feature IDs."""

    modality: str
    source_id_type: str
    target_id_type: str
    mapping_source: str
    retrieved_at: str
    mappings: list[FeatureMapping] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n_unmapped = sum(1 for item in self.mappings if item.is_unmapped)
        n_ambiguous = sum(1 for item in self.mappings if item.is_ambiguous)
        return {
            "n_features": len(self.mappings),
            "n_mapped": len(self.mappings) - n_unmapped,
            "n_unmapped": n_unmapped,
            "n_ambiguous": n_ambiguous,
            "n_pairs": sum(len(item.targets) for item in self.mappings),
        }

    def to_frame(self) -> pd.DataFrame:
        """Long form: one row per (source, target) pair; unmapped rows keep NA."""

        rows: list[dict[str, Any]] = []
        for item in self.mappings:
            if item.is_unmapped:
                rows.append(
                    {
                        "source_id": item.source_id,
                        "target_id": pd.NA,
                        "target_id_type": self.target_id_type,
                        "ambiguous": False,
                        "unmapped": True,
                    }
                )
                continue
            for target in item.targets:
                rows.append(
                    {
                        "source_id": item.source_id,
                        "target_id": target.target_id,
                        "target_id_type": target.target_id_type,
                        "ambiguous": item.is_ambiguous,
                        "unmapped": False,
                    }
                )
        return pd.DataFrame(rows)
