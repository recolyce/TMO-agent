"""Model I/O schemas shared by every ModelPlugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel


class FitResult(StrictModel):
    """What ``fit`` must return. No hidden side-channel metrics."""

    model_name: str
    n_train_instances: int
    n_parameters: int
    extras: dict[str, Any] = Field(default_factory=dict)


class AttributionTable(StrictModel):
    """Placeholder for later interpretation milestones.

    Milestone 1 models may return coefficients here. Attribution is not
    causation; reports must not call these values regulatory effects.
    """

    model_name: str
    method: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    caveat: str = (
        "These values are prediction contributions or coefficients, not causal effects."
    )


class ModelCard(StrictModel):
    """On-disk identity of a fitted model."""

    name: str
    params: dict[str, Any]
    feature_names: list[str]
    target_names: list[str]
    fit_split: str = "train"
    path: Path | None = None
