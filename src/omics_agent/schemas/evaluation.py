"""Typed evaluation outputs. Metrics are computed only by the evaluator."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel


class ScalarMetric(StrictModel):
    """One named metric, possibly undefined (NA) for constant vectors."""

    name: str
    value: float | None
    n_valid: int
    n_total: int
    note: str | None = None


class BootstrapCI(StrictModel):
    """Percentile interval obtained by resampling experimental units."""

    metric: str
    low: float | None
    high: float | None
    n_replicates: int
    n_units: int


class EvaluationReport(StrictModel):
    """Everything the evaluator returns for one model and one split."""

    model_name: str
    split: str
    target_modality: str
    n_instances: int
    n_features: int
    coverage: float
    n_observed_targets: int
    n_possible_targets: int
    scalars: list[ScalarMetric]
    per_feature: list[dict[str, Any]] = Field(default_factory=list)
    per_sample: list[dict[str, Any]] = Field(default_factory=list)
    bootstrap: list[BootstrapCI] = Field(default_factory=list)
    primary_metric: str
    primary_value: float | None
    warnings: list[str] = Field(default_factory=list)
