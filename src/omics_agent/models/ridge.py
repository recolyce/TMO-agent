"""Multi-output Ridge on a last snapshot plus time/condition covariates."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from omics_agent.errors import SchemaError
from omics_agent.models.base import register_model
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.model import AttributionTable, FitResult


def _as_float(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SchemaError(
            f"Parameter '{key}' must be a number, got {value!r}.",
            how_to_fix=f"Set params.{key} to a number such as {default}.",
        )
    return float(value)


class _RidgeState:
    def __init__(self) -> None:
        self.model: Ridge | None = None
        self.encoder: OneHotEncoder | None = None
        self.input_names: list[str] = []
        self.target_names: list[str] = []
        self.design_names: list[str] = []


def _design_matrix(
    data: DataForModel,
    encoder: OneHotEncoder,
    *,
    fit_encoder: bool,
) -> np.ndarray:
    forecast = data.forecast
    cond = np.asarray(forecast.conditions).reshape(-1, 1)
    dummy = encoder.fit_transform(cond) if fit_encoder else encoder.transform(cond)
    dummy_arr = np.asarray(dummy.todense() if hasattr(dummy, "todense") else dummy)
    time = forecast.target_times.reshape(-1, 1)
    delta = np.nan_to_num(forecast.delta_t, nan=0.0).reshape(-1, 1)
    return np.hstack([forecast.last_inputs, time, delta, dummy_arr])


@register_model
class RidgeModel:
    """Ridge regression from last inputs + time + Δt + condition."""

    name = "ridge"

    def __init__(self) -> None:
        self._state = _RidgeState()

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        del val
        alpha = _as_float(cfg.params, "alpha", 1.0)
        y = train.forecast.y_true
        mask = train.forecast.y_mask
        # Fit only instances that have at least one observed target.
        keep = mask.any(axis=1)
        if not keep.any():
            raise SchemaError(
                "Ridge has no train instances with observed targets.",
                how_to_fix="Check missingness rates and the train split.",
            )
        encoder = OneHotEncoder(handle_unknown="error", sparse_output=False)
        x = _design_matrix(train, encoder, fit_encoder=True)[keep]
        y_fit = y[keep].copy()
        col_mean = np.nanmean(y_fit, axis=0)
        inds = np.isnan(y_fit)
        y_fit[inds] = np.take(col_mean, np.where(inds)[1])
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x, y_fit)
        self._state.model = model
        self._state.encoder = encoder
        self._state.input_names = list(train.forecast.input_feature_names)
        self._state.target_names = list(train.forecast.feature_names)
        dummy_names = list(encoder.get_feature_names_out(["condition"]))
        self._state.design_names = [
            *self._state.input_names,
            "target_time",
            "delta_t",
            *dummy_names,
        ]
        n_params = int(np.prod(model.coef_.shape) + model.intercept_.shape[0])
        return FitResult(
            model_name=self.name,
            n_train_instances=int(keep.sum()),
            n_parameters=n_params,
            extras={"alpha": alpha},
        )

    def predict(self, data: DataForModel) -> PredictionArrays:
        if self._state.model is None or self._state.encoder is None:
            raise SchemaError(
                "Ridge was used before fit().",
                how_to_fix="Call fit() on the train split first.",
            )
        x = _design_matrix(data, self._state.encoder, fit_encoder=False)
        pred = self._state.model.predict(x)
        return PredictionArrays(y_pred=np.asarray(pred, dtype=float), extras={})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._state, path / "model.joblib")

    def load(self, path: Path) -> None:
        self._state = joblib.load(path / "model.joblib")

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        del data
        if self._state.model is None:
            raise SchemaError(
                "Ridge.explain requires a fitted model.",
                how_to_fix="Fit the model before requesting coefficients.",
            )
        coef = np.asarray(self._state.model.coef_, dtype=float)
        rows = []
        wanted = set(targets) if targets else set(self._state.target_names)
        for t_i, t_name in enumerate(self._state.target_names):
            if t_name not in wanted:
                continue
            for f_i, f_name in enumerate(self._state.design_names):
                rows.append(
                    {
                        "target": t_name,
                        "feature": f_name,
                        "coefficient": float(coef[t_i, f_i]),
                    }
                )
        return AttributionTable(model_name=self.name, method="ridge_coefficients", rows=rows)
