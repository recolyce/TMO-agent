"""MLP baseline on the same tabular design matrix as Ridge.

Nonlinear reference point for the dynamics models. Uses sklearn (CPU);
same split, same features, same evaluator.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder

from omics_agent.errors import SchemaError
from omics_agent.models.base import register_model
from omics_agent.models.ridge import _design_matrix
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.model import AttributionTable, FitResult


@register_model
class MlpModel:
    """Two-hidden-layer MLP from last inputs + time + Δt + condition."""

    name = "mlp"

    def __init__(self) -> None:
        self._model: MLPRegressor | None = None
        self._encoder: OneHotEncoder | None = None
        self._target_names: list[str] = []

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        del val
        params = cfg.params
        hidden = params.get("hidden_layer_sizes", [64, 64])
        if not isinstance(hidden, list | tuple) or not all(
            isinstance(item, int) and item > 0 for item in hidden
        ):
            raise SchemaError(
                f"hidden_layer_sizes must be a list of positive ints, got {hidden!r}.",
                how_to_fix="Example: params.hidden_layer_sizes: [64, 64]",
            )
        seed = int(params.get("seed", 20260901))
        y = train.forecast.y_true
        mask = train.forecast.y_mask
        keep = mask.any(axis=1)
        if not keep.any():
            raise SchemaError(
                "MLP has no train instances with observed targets.",
                how_to_fix="Check missingness rates and the train split.",
            )
        encoder = OneHotEncoder(handle_unknown="error", sparse_output=False)
        x = _design_matrix(train, encoder, fit_encoder=True)[keep]
        y_fit = y[keep].copy()
        col_mean = np.nanmean(y_fit, axis=0)
        inds = np.isnan(y_fit)
        y_fit[inds] = np.take(col_mean, np.where(inds)[1])
        model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden),
            alpha=float(params.get("alpha", 1e-4)),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=seed,
            solver="adam",
        )
        model.fit(x, y_fit)
        self._model = model
        self._encoder = encoder
        self._target_names = list(train.forecast.feature_names)
        n_params = int(
            sum(int(np.prod(w.shape)) for w in model.coefs_)
            + sum(int(b.shape[0]) for b in model.intercepts_)
        )
        return FitResult(
            model_name=self.name,
            n_train_instances=int(keep.sum()),
            n_parameters=n_params,
            extras={"hidden_layer_sizes": list(hidden), "seed": seed},
        )

    def predict(self, data: DataForModel) -> PredictionArrays:
        if self._model is None or self._encoder is None:
            raise SchemaError(
                "MLP was used before fit().",
                how_to_fix="Call fit() on the train split first.",
            )
        x = _design_matrix(data, self._encoder, fit_encoder=False)
        pred = np.asarray(self._model.predict(x), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return PredictionArrays(y_pred=pred, extras={})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "encoder": self._encoder, "targets": self._target_names},
            path / "model.joblib",
        )

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        del data, targets
        return AttributionTable(
            model_name=self.name,
            method="none",
            rows=[],
            caveat=(
                "MLP weights are not per-feature effects. Attribution arrives with "
                "the interpretation milestone; attribution is not causation."
            ),
        )
