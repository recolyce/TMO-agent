"""Search-space and pruning-hook units for the milestone-5 tuner."""

from __future__ import annotations

import optuna
import pytest

from omics_agent.errors import SchemaError
from omics_agent.optimization.search_space import suggest_params
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.optimization import OptimizationConfig, OptimizationDecision


def test_search_space_ridge_and_mlp() -> None:
    ridge = suggest_params("ridge", optuna.trial.FixedTrial({"alpha": 0.5}))
    assert ridge == {"alpha": 0.5}
    mlp = suggest_params("mlp", optuna.trial.FixedTrial({"hidden": "64x64", "alpha": 1e-4}))
    assert mlp["hidden_layer_sizes"] == [64, 64]


def test_search_space_dynamics_keys() -> None:
    params = suggest_params(
        "ode_rnn",
        optuna.trial.FixedTrial(
            {
                "lr": 1e-3,
                "hidden_dim": 48,
                "emb_dim": 32,
                "recon_weight": 0.2,
                "weight_decay": 1e-5,
            }
        ),
    )
    assert set(params) == {"lr", "hidden_dim", "emb_dim", "recon_weight", "weight_decay"}


def test_last_value_has_no_search_space() -> None:
    with pytest.raises(SchemaError, match="no tunable search space"):
        suggest_params("last_value", optuna.trial.FixedTrial({}))


def test_optimization_config_is_val_only_by_construction() -> None:
    config = OptimizationConfig()
    assert config.direction == "minimize"
    assert config.objective_metric in {"mse", "mae"}
    # The decision schema cannot even express a test objective.
    with pytest.raises(Exception, match="objective_split"):
        OptimizationDecision.model_validate(
            {
                "experiment_id": "x",
                "model_name": "ridge",
                "study_name": "x::ridge",
                "sampler": "tpe",
                "sampler_seed": 1,
                "pruner": {},
                "n_trials_budget": 1,
                "n_trials_completed": 1,
                "n_trials_pruned": 0,
                "n_trials_failed": 0,
                "objective_metric": "mse",
                "objective_split": "test",
                "direction": "minimize",
                "best_trial_number": 0,
                "best_params": {},
                "best_value": 0.0,
                "val_primary_metric": "protein_macro_pcc",
                "decided_at": "now",
            }
        )


def test_dynamics_epoch_callback_reports_and_pruning_propagates() -> None:
    pytest.importorskip("torch")
    from omics_agent.models import get_model
    from tests.unit.dyn_helpers import make_bundle, make_split_data, make_task

    train, val = make_split_data(make_bundle(), make_task())
    small = {
        "epochs": 3,
        "val_every": 1,
        "batch_size": 16,
        "hidden_dim": 16,
        "emb_dim": 12,
        "device": "cpu",
        "seed": 5,
    }
    model = get_model("gru")
    calls: list[tuple[int, float]] = []
    model.set_epoch_callback(lambda epoch, mse: calls.append((epoch, mse)))
    model.fit(train, val, ModelParams(name="gru", params=small))
    assert [epoch for epoch, _ in calls] == [0, 1, 2]
    assert all(mse >= 0 for _, mse in calls)

    pruned = get_model("gru")

    def prune(epoch: int, mse: float) -> None:
        raise optuna.TrialPruned()

    pruned.set_epoch_callback(prune)
    with pytest.raises(optuna.TrialPruned):
        pruned.fit(train, val, ModelParams(name="gru", params=small))
