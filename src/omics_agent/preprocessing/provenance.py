"""Provenance records that make a full-data fit impossible to hide."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from omics_agent.schemas.dataset import StrictModel


class TransformerProvenance(StrictModel):
    """Metadata attached to every fitted preprocessor.

    ``fit_split`` is literally the string ``train``. Any other value is
    rejected by the schema so a leaked fit cannot be serialized.
    """

    transformer_name: str
    fit_split: Literal["train"]
    n_fit_samples: int
    n_features: int
    fitted_at: datetime
    parameters_hash: str
    extras: dict[str, Any] = {}

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(UTC)


class StatelessTransformRecord(StrictModel):
    """Provenance for per-sample math that learns no cross-sample statistics.

    CPM uses only that sample's own library size; log transforms are
    element-wise. ``learns_statistics`` is literally ``False`` so the audit
    can distinguish these records from fitted transformers, which must
    carry ``fit_split='train'``.
    """

    transformer_name: str
    kind: Literal["stateless_per_sample"]
    learns_statistics: Literal[False] = False
    parameters: dict[str, Any] = {}
    applied_at: datetime
    extras: dict[str, Any] = {}
