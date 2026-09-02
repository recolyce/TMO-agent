"""Condition-specific B-spline of time, fitted on train observations only.

This is the scientifically appropriate simple baseline for repeated
cross-section: time is a continuous covariate, animals are not chained.
It is also a valid population-trajectory baseline for longitudinal data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import patsy
from sklearn.linear_model import Ridge

from omics_agent.errors import SchemaError
from omics_agent.models.base import register_model
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.model import AttributionTable, FitResult


def _as_int(params: dict[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SchemaError(
            f"Parameter '{key}' must be an integer, got {value!r}.",
            how_to_fix=f"Set params.{key} to a number such as {default}.",
        )
    return int(value)


def _as_float(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SchemaError(
            f"Parameter '{key}' must be a number, got {value!r}.",
            how_to_fix=f"Set params.{key} to a number such as {default}.",
        )
    return float(value)


@register_model
class TimeSplineModel:
    """Y(time, condition) ≈ B-spline(time) * C(condition)."""

    name = "time_spline"

    def __init__(self) -> None:
        self._model: Ridge | None = None
        self._formula: str = ""
        self._column_names: list[str] = []
        self._conditions: list[str] = []
        self._target_names: list[str] = []
        self._spline_df: int = 4

    def _formula_for(self, spline_df: int) -> str:
        return f"bs(time, df={spline_df}, include_intercept=False) * C(condition)"

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        del val
        spline_df = _as_int(cfg.params, "spline_df", 4)
        alpha = _as_float(cfg.params, "alpha", 1.0)
        obs = train.bundle.observations
        n_times = int(obs["time"].nunique())
        if spline_df >= n_times:
            raise SchemaError(
                f"time_spline spline_df={spline_df} is too large for {n_times} unique train times.",
                how_to_fix=(
                    f"Set models.time_spline.params.spline_df to {max(1, n_times - 1)} or lower. "
                    "The pipeline will not silently reduce the degrees of freedom."
                ),
            )
        target = train.forecast.feature_names
        # Fit on all train observations (not only forecast instances) so early
        # RCS times inform the trajectory.
        y = train.bundle.matrices[ _target_modality_from_names(train) ]
        keep = ~np.isnan(y).all(axis=1)
        if int(keep.sum()) < spline_df + 1:
            raise SchemaError(
                "Not enough train observations to fit the time spline.",
                how_to_fix="Increase the train split or lower spline_df.",
            )
        conditions = sorted({str(item) for item in obs["condition"].to_numpy()[keep]})
        frame = pd.DataFrame(
            {
                "time": obs["time"].to_numpy()[keep],
                "condition": pd.Categorical(
                    [str(item) for item in obs["condition"].to_numpy()[keep]],
                    categories=conditions,
                ),
            }
        )
        formula = self._formula_for(spline_df)
        x = patsy.dmatrix(formula, data=frame, return_type="dataframe")
        y_fit = y[keep].copy()
        col_mean = np.nanmean(y_fit, axis=0)
        inds = np.isnan(y_fit)
        y_fit[inds] = np.take(col_mean, np.where(inds)[1])
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x.to_numpy(), y_fit)
        self._model = model
        self._formula = formula
        self._column_names = list(x.columns)
        self._conditions = conditions
        self._target_names = list(target)
        self._spline_df = spline_df
        n_params = int(np.prod(model.coef_.shape))
        return FitResult(
            model_name=self.name,
            n_train_instances=int(keep.sum()),
            n_parameters=n_params,
            extras={"spline_df": spline_df, "alpha": alpha, "formula": formula},
        )

    def predict(self, data: DataForModel) -> PredictionArrays:
        if self._model is None or not self._formula:
            raise SchemaError(
                "TimeSpline was used before fit().",
                how_to_fix="Call fit() on the train split first.",
            )
        unknown = sorted(set(data.forecast.conditions) - set(self._conditions))
        if unknown:
            raise SchemaError(
                f"TimeSpline saw conditions {unknown} that were not in train {self._conditions}.",
                how_to_fix="Keep condition levels in every split, or treat a new condition as a separate experiment.",
            )
        frame = pd.DataFrame(
            {
                "time": data.forecast.target_times,
                "condition": pd.Categorical(data.forecast.conditions, categories=self._conditions),
            }
        )
        x = patsy.dmatrix(self._formula, data=frame, return_type="dataframe")
        x = x.reindex(columns=self._column_names, fill_value=0.0)
        pred = self._model.predict(x.to_numpy())
        return PredictionArrays(y_pred=np.asarray(pred, dtype=float), extras={})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "formula": self._formula,
                "column_names": self._column_names,
                "conditions": self._conditions,
                "target_names": self._target_names,
                "spline_df": self._spline_df,
            },
            path / "model.joblib",
        )

    def load(self, path: Path) -> None:
        payload = joblib.load(path / "model.joblib")
        self._model = payload["model"]
        self._formula = str(payload["formula"])
        self._column_names = list(payload["column_names"])
        self._conditions = list(payload["conditions"])
        self._target_names = list(payload["target_names"])
        self._spline_df = int(payload["spline_df"])

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        del data
        if self._model is None:
            raise SchemaError(
                "TimeSpline.explain requires a fitted model.",
                how_to_fix="Fit the model before requesting coefficients.",
            )
        names = list(self._column_names)
        coef = np.asarray(self._model.coef_, dtype=float)
        wanted = set(targets) if targets else set(self._target_names)
        rows = []
        for t_i, t_name in enumerate(self._target_names):
            if t_name not in wanted:
                continue
            for f_i, f_name in enumerate(names):
                rows.append(
                    {
                        "target": t_name,
                        "feature": f_name,
                        "coefficient": float(coef[t_i, f_i]),
                    }
                )
        return AttributionTable(model_name=self.name, method="spline_coefficients", rows=rows)


def _target_modality_from_names(train: DataForModel) -> str:
    """Recover the target modality key from the bundle feature-name map."""

    names = list(train.forecast.feature_names)
    for modality, features in train.bundle.feature_names.items():
        if list(features) == names:
            return modality
    raise SchemaError(
        "Could not match forecast feature names to a bundle modality.",
        how_to_fix="Keep task.target_modality consistent between instance construction and the bundle.",
    )
