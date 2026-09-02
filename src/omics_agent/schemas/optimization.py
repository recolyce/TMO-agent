"""Milestone-5 HPO schemas: budget, decision, freeze manifest, test lock.

The optimizer's objective is computed on the validation split only. Test
labels are never passed to the optimizer's API; ``OptimizationDecision``
records that invariant explicitly (``objective_split`` is literally
``"val"`` and ``test_labels_visible`` is literally ``False``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel


class PrunerConfig(StrictModel):
    """Fixed pruner settings recorded with every study."""

    kind: Literal["median"] = "median"
    n_startup_trials: int = Field(default=5, ge=0)
    n_warmup_reports: int = Field(default=2, ge=0)


class OptimizationConfig(StrictModel):
    """Validation-only HPO with a fixed budget, seed, and pruner.

    The objective metric is computed on val predictions with the standard
    masked evaluator math. The frozen ``task.primary_metric`` is reported
    alongside but is not writable here (rule 3).
    """

    n_trials: int = Field(default=20, ge=1, le=500)
    sampler: Literal["tpe"] = "tpe"
    sampler_seed: int | None = Field(
        default=None, description="Defaults to the experiment seed when unset."
    )
    objective_metric: Literal["mse", "mae"] = "mse"
    direction: Literal["minimize"] = "minimize"
    pruner: PrunerConfig = Field(default_factory=PrunerConfig)


class TrialRecord(StrictModel):
    """One Optuna trial as recorded in the decision document."""

    number: int
    state: str
    value: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class OptimizationDecision(StrictModel):
    """Structured, auditable outcome of one tuning study."""

    schema_version: str = "1.0"
    experiment_id: str
    model_name: str
    study_name: str
    sampler: str
    sampler_seed: int
    pruner: PrunerConfig
    n_trials_budget: int
    n_trials_completed: int
    n_trials_pruned: int
    n_trials_failed: int
    objective_metric: str
    objective_split: Literal["val"] = "val"
    direction: str
    best_trial_number: int
    best_params: dict[str, Any]
    best_value: float
    val_primary_metric: str
    val_primary_value: float | None = None
    test_labels_visible: Literal[False] = False
    decided_at: str
    hashes: dict[str, str] = Field(default_factory=dict)
    trials: list[TrialRecord] = Field(default_factory=list)


class FreezeManifest(StrictModel):
    """Hashes of everything the one-shot final test depends on.

    ``unlock-test`` recomputes each hash; any mismatch is a hard refusal.
    """

    schema_version: str = "1.0"
    experiment_id: str
    model_name: str
    frozen_at: str
    best_params: dict[str, Any]
    frozen_config_hash: str
    checkpoint_hashes: dict[str, str]
    decision_hash: str
    split_file_hash: str
    data_hash: str
    evaluator_code_hash: str
    splitting_code_hash: str
    status: Literal["frozen"] = "frozen"


class TestLockState(StrictModel):
    """One-shot test lock. Once consumed, tuning and re-testing are refused."""

    experiment_id: str
    model_name: str
    consumed: bool
    consumed_at: str | None = None
    final_report_hash: str | None = None
