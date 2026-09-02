"""LastValue baseline.

Longitudinal: the same experimental unit's last observed target vector.
Repeated cross-section: the train condition×time mean at the latest earlier
time. Never copies another animal's series into a fake trajectory.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from omics_agent.errors import SchemaError
from omics_agent.models.base import register_model
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.model import AttributionTable, FitResult


@register_model
class LastValueModel:
    """Copy the last available target. No learned parameters."""

    name = "last_value"

    def __init__(self) -> None:
        self._fitted = False
        self._n_features = 0
        self._feature_names: list[str] = []

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        del val, cfg
        self._n_features = train.forecast.y_true.shape[1]
        self._feature_names = list(train.forecast.feature_names)
        self._fitted = True
        return FitResult(
            model_name=self.name,
            n_train_instances=len(train.forecast.instance_ids),
            n_parameters=0,
            extras={"strategy": "last_observed_or_train_condition_time_mean"},
        )

    def predict(self, data: DataForModel) -> PredictionArrays:
        if not self._fitted:
            raise SchemaError(
                "LastValue was used before fit().",
                how_to_fix="Call fit() on the train split first.",
            )
        pred = data.forecast.last_target.copy()
        if pred.shape[1] != self._n_features:
            raise SchemaError(
                f"LastValue expected {self._n_features} target features, got {pred.shape[1]}.",
                how_to_fix="Use the same target modality and feature panel as training.",
            )
        return PredictionArrays(y_pred=pred, extras={})

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "n_features": self._n_features,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            },
            path / "model.joblib",
        )

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        del data, targets
        return AttributionTable(
            model_name=self.name,
            method="none",
            rows=[],
            caveat=(
                "LastValue has no learned parameters. A large last-value score "
                "means the series is slowly changing, not that a gene regulates a protein."
            ),
        )
