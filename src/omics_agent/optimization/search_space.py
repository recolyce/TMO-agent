"""Fixed per-model search spaces.

The spaces are code, not config: an optimizer cannot widen its own budget
or search over forbidden fields (split, evaluator, primary metric).
"""

from __future__ import annotations

from typing import Any

import optuna

from omics_agent.errors import SchemaError

TUNABLE_MODELS = ("ridge", "time_spline", "mlp", "gru", "ode_rnn", "latent_ode")

_DYNAMICS = {"gru", "ode_rnn", "latent_ode"}


def suggest_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter set for ``model_name``."""

    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if model_name == "time_spline":
        return {
            "spline_df": trial.suggest_int("spline_df", 2, 4),
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
        }
    if model_name == "mlp":
        width = trial.suggest_categorical("hidden", ["32", "64x64", "128x64"])
        sizes = {"32": [32], "64x64": [64, 64], "128x64": [128, 64]}[width]
        return {
            "hidden_layer_sizes": sizes,
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        }
    if model_name in _DYNAMICS:
        return {
            "lr": trial.suggest_float("lr", 3e-4, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64]),
            "emb_dim": trial.suggest_categorical("emb_dim", [16, 32, 48]),
            "recon_weight": trial.suggest_float("recon_weight", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
    raise SchemaError(
        f"Model '{model_name}' has no tunable search space.",
        how_to_fix=f"Tunable models: {', '.join(TUNABLE_MODELS)}. last_value has no parameters.",
    )
