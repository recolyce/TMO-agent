"""Standard scaler and mean imputer that can only be fitted on train rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omics_agent.errors import PreprocessingLeakageError
from omics_agent.hashing import hash_mapping
from omics_agent.preprocessing.provenance import TransformerProvenance
from omics_agent.schemas.enums import SplitName


def _require_train_only(split_labels: np.ndarray, transformer: str) -> None:
    unique = {str(item) for item in split_labels}
    if unique != {SplitName.TRAIN.value}:
        raise PreprocessingLeakageError(
            f"{transformer} received rows from splits {sorted(unique)} during fit. "
            "Scaler, imputer, HVG, PCA, and batch-correction parameters may only "
            "be fitted on train.",
            how_to_fix=(
                "Pass only train rows to fit(). Transform val/test with the "
                "already-fitted object. Never call fit on the concatenated matrix."
            ),
        )


class TrainOnlyStandardScaler:
    """Column-wise z-score fitted exclusively on train observations.

    Zero-variance train columns keep ``scale_ = 1`` so transform stays finite.
    NaNs are ignored when estimating mean/std (NaN-aware) and preserved in
    transform; use :class:`TrainOnlyImputer` if you need filled values.
    """

    def __init__(self, name: str = "standard_scaler") -> None:
        self.name = name
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.provenance: TransformerProvenance | None = None

    def fit(self, values: np.ndarray, split_labels: np.ndarray) -> TrainOnlyStandardScaler:
        """Fit mean/std on ``values`` after checking every row is train."""

        _require_train_only(split_labels, self.name)
        if values.ndim != 2:
            raise PreprocessingLeakageError(
                f"{self.name} expected a 2-D array, got shape {values.shape}.",
                how_to_fix="Pass an observations × features numeric matrix.",
            )
        self.mean_ = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=0)
        zero = std < 1e-12
        std = np.where(zero, 1.0, std)
        self.scale_ = std
        self.provenance = TransformerProvenance(
            transformer_name=self.name,
            fit_split="train",
            n_fit_samples=int(values.shape[0]),
            n_features=int(values.shape[1]),
            fitted_at=TransformerProvenance.utc_now(),
            parameters_hash=hash_mapping(
                {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}
            ),
            extras={"n_zero_variance": int(zero.sum())},
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise PreprocessingLeakageError(
                f"{self.name} was used before fit().",
                how_to_fix="Call fit(train_matrix, train_split_labels) first.",
            )
        return (values - self.mean_) / self.scale_

    def save(self, path: Path) -> None:
        if self.provenance is None or self.mean_ is None or self.scale_ is None:
            raise PreprocessingLeakageError(
                f"{self.name} cannot be saved before fit().",
                how_to_fix="Fit on train, then save the artifact.",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean_,
            scale=self.scale_,
            provenance=np.asarray(self.provenance.model_dump_json()),
        )


class TrainOnlyImputer:
    """Column mean imputer fitted only on train. Does not fill with 0."""

    def __init__(self, name: str = "mean_imputer") -> None:
        self.name = name
        self.statistics_: np.ndarray | None = None
        self.provenance: TransformerProvenance | None = None

    def fit(self, values: np.ndarray, split_labels: np.ndarray) -> TrainOnlyImputer:
        _require_train_only(split_labels, self.name)
        stats = np.nanmean(values, axis=0)
        if np.isnan(stats).any():
            raise PreprocessingLeakageError(
                f"{self.name} found a feature that is entirely missing on train.",
                how_to_fix=(
                    "Drop that feature, or supply a scientifically justified "
                    "imputation. Filling with 0 is not allowed by default."
                ),
            )
        self.statistics_ = stats
        self.provenance = TransformerProvenance(
            transformer_name=self.name,
            fit_split="train",
            n_fit_samples=int(values.shape[0]),
            n_features=int(values.shape[1]),
            fitted_at=TransformerProvenance.utc_now(),
            parameters_hash=hash_mapping({"statistics": stats.tolist()}),
        )
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.statistics_ is None:
            raise PreprocessingLeakageError(
                f"{self.name} was used before fit().",
                how_to_fix="Call fit(train_matrix, train_split_labels) first.",
            )
        out = values.copy()
        inds = np.where(np.isnan(out))
        out[inds] = np.take(self.statistics_, inds[1])
        return out
