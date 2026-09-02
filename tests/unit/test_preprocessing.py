from __future__ import annotations

import numpy as np
import pytest

from omics_agent.errors import PreprocessingLeakageError
from omics_agent.preprocessing.scalers import TrainOnlyImputer, TrainOnlyStandardScaler


def test_scaler_rejects_fit_on_all_data() -> None:
    values = np.arange(12, dtype=float).reshape(4, 3)
    labels = np.array(["train", "train", "val", "test"])
    with pytest.raises(PreprocessingLeakageError, match="only be fitted on train"):
        TrainOnlyStandardScaler().fit(values, labels)


def test_scaler_provenance_is_train() -> None:
    values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    labels = np.array(["train", "train", "train"])
    scaler = TrainOnlyStandardScaler(name="rna_standard_scaler").fit(values, labels)
    assert scaler.provenance is not None
    assert scaler.provenance.fit_split == "train"
    transformed = scaler.transform(values)
    assert transformed.shape == values.shape
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-12)


def test_imputer_does_not_fill_with_zero_for_all_missing_column() -> None:
    values = np.array([[1.0, np.nan], [2.0, np.nan]])
    labels = np.array(["train", "train"])
    with pytest.raises(PreprocessingLeakageError, match="entirely missing"):
        TrainOnlyImputer().fit(values, labels)
