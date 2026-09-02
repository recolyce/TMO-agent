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
