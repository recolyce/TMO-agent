"""Contracts for frozen-model attribution and the candidate table.

Attribution values are prediction contributions. They are not causal
effects. Reports must label every row as a hypothesis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import IgBaselineName
from omics_agent.schemas.literature import LiteratureSearchConfig

ABSENCE_OF_EVIDENCE = "在本次检索范围内未找到直接证据"
CLAIM_KIND: Literal["hypothesis"] = "hypothesis"
HYPOTHESIS_CAVEAT = (
    "These rows are hypotheses about model prediction contributions. "
    "Attribution is not causation. Absence of a paper is not novelty."
)


class InterpretationConfig(StrictModel):
    """Pre-registered attribution settings. Thresholds are not Agent-writable."""

    n_ig_steps: int = Field(default=16, ge=4, le=128)
    baselines: list[IgBaselineName] = Field(
        default_factory=lambda: [
            IgBaselineName.ZEROS,
            IgBaselineName.TRAIN_MEAN,
            IgBaselineName.LAST_OBSERVATION,
        ]
    )
    n_bootstrap: int = Field(default=5, ge=2, le=200)
    n_seeds: int = Field(default=5, ge=2, le=50)
    n_folds: int = Field(default=2, ge=2, le=10)
    top_k_per_target: int = Field(default=5, ge=1, le=50)
    top_n: int = Field(default=20, ge=1, le=50)
    min_sign_consistency: float = Field(default=0.6, ge=0.0, le=1.0)
    min_selection_frequency: float = Field(default=0.4, ge=0.0, le=1.0)
    min_stability: float = Field(default=0.4, ge=0.0, le=1.0)
    literature: LiteratureSearchConfig = Field(default_factory=lambda: LiteratureSearchConfig())


class CandidateRow(StrictModel):
    """One source→target hypothesis after stability screening."""

    candidate_id: str
    source_modality: str
    source_id: str
    target_modality: str
    target_id: str
    mean_attribution: float
    sign_consistency: float
    rank_median: float
    selection_frequency: float
    bootstrap_low: float
    bootstrap_high: float
    ablation_delta: float
    permutation_delta: float
    stability: float
    prior_edge_used: bool
    embedding_supported: bool
    de_novo_model_edge: bool
    passed_stability: bool
    predicted_direction: str
    claim_kind: Literal["hypothesis"] = CLAIM_KIND
    caveat: str = HYPOTHESIS_CAVEAT


class StabilityTable(StrictModel):
    """Full candidate table written by ``explain``."""

    experiment_id: str
    model_name: str
    objective_split: Literal["val"]
    test_labels_visible: Literal[False]
    claim_kind: Literal["hypothesis"] = CLAIM_KIND
    n_baselines: int
    n_bootstrap: int
    n_seeds: int
    n_folds: int
    rows: list[CandidateRow]
    notes: list[str] = Field(default_factory=list)
