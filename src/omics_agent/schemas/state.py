"""Pipeline research state. Stage transitions are explicit."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import Stage


class ResearchState(StrictModel):
    """Auditable pointer to the current dataset, hashes, and open questions."""

    dataset_id: str
    experiment_id: str | None = None
    stage: Stage
    manifest_path: Path
    data_hash: str | None = None
    split_hash: str | None = None
    frozen_model_hash: str | None = None
    unresolved: list[str] = Field(default_factory=list)
