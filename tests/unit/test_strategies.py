from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from omics_agent.errors import SchemaError
from omics_agent.preprocessing.provenance import TransformerProvenance
from omics_agent.preprocessing.strategies import default_config_for, normalize
from omics_agent.schemas.dataset import ModalitySpec
from omics_agent.schemas.enums import AssayStrategy, AssayType, FeatureIdType, ValueType
from omics_agent.schemas.preprocess import ModalityPreprocessConfig


def _spec(value_type: ValueType) -> ModalitySpec:
    return ModalitySpec(
        assay=AssayType.BULK_RNASEQ,
        value_type=value_type,
        feature_id_type=FeatureIdType.UNDECLARED,
    )


def test_default_strategy_follows_value_type() -> None:
    assert default_config_for("rna", _spec(ValueType.RAW_COUNTS)).strategy is AssayStrategy.BULK_RNA_COUNTS
    assert default_config_for("protein", _spec(ValueType.INTENSITY)).strategy is AssayStrategy.PROTEIN_INTENSITY
    assert default_config_for("rna", _spec(ValueType.LOG1P)).strategy is AssayStrategy.LOG_EXPRESSION
    assert default_config_for("protein", _spec(ValueType.LOG2_INTENSITY)).strategy is AssayStrategy.LOG_EXPRESSION


def test_undeclared_value_type_refuses_to_choose() -> None:
    with pytest.raises(SchemaError, match="no preprocessing strategy"):
        default_config_for("rna", _spec(ValueType.UNDECLARED))


def test_counts_reject_negative_values() -> None:
    config = ModalityPreprocessConfig(strategy=AssayStrategy.BULK_RNA_COUNTS)
    with pytest.raises(SchemaError, match="negative"):
        normalize("rna", np.array([[1.0, -2.0]]), config)


def test_counts_cpm_log1p_math() -> None:
    raw = np.array([[90.0, 10.0], [30.0, 10.0]])
    config = ModalityPreprocessConfig(strategy=AssayStrategy.BULK_RNA_COUNTS)
    normalized, record = normalize("rna", raw, config)
    library = raw.sum(axis=1, keepdims=True)
    expected = np.log1p(raw / library * 1_000_000.0)
    assert np.allclose(normalized, expected)
    assert record.learns_statistics is False
    assert record.kind == "stateless_per_sample"


def test_counts_zero_library_sample_becomes_missing() -> None:
    raw = np.array([[0.0, 0.0], [5.0, 5.0]])
    config = ModalityPreprocessConfig(strategy=AssayStrategy.BULK_RNA_COUNTS)
    normalized, record = normalize("rna", raw, config)
    assert np.isnan(normalized[0]).all()
    assert not np.isnan(normalized[1]).any()
    assert record.extras["n_zero_library_samples"] == 1


def test_protein_zero_becomes_missing_not_zero() -> None:
    raw = np.array([[0.0, 4.0], [2.0, np.nan]])
    config = ModalityPreprocessConfig(strategy=AssayStrategy.PROTEIN_INTENSITY)
    normalized, record = normalize("protein", raw, config)
    assert np.isnan(normalized[0, 0]), "zero intensity must become missing, never a measured 0"
    assert normalized[0, 1] == pytest.approx(2.0)  # log2(4)
    assert np.isnan(normalized[1, 1])
    assert record.extras["n_zeros_as_missing"] == 1
    # Nothing anywhere was filled with 0.
    assert not np.any(normalized[np.isnan(raw) | (raw == 0)] == 0)


def test_protein_rejects_negative_intensity() -> None:
    config = ModalityPreprocessConfig(strategy=AssayStrategy.PROTEIN_INTENSITY)
    with pytest.raises(SchemaError, match="negative"):
        normalize("protein", np.array([[-1.0]]), config)


def test_protein_zeros_with_log2_but_not_missing_is_an_error() -> None:
    config = ModalityPreprocessConfig(
        strategy=AssayStrategy.PROTEIN_INTENSITY, zeros_are_missing=False, log2_transform=True
    )
    with pytest.raises(SchemaError, match="log2\\(0\\)"):
        normalize("protein", np.array([[0.0, 3.0]]), config)


def test_log_expression_is_passthrough_and_rejects_inf() -> None:
    raw = np.array([[1.5, np.nan], [0.0, -2.0]])
    config = ModalityPreprocessConfig(strategy=AssayStrategy.LOG_EXPRESSION)
    normalized, _ = normalize("rna", raw, config)
    assert np.array_equal(normalized, raw, equal_nan=True)
    with pytest.raises(SchemaError, match="infinite"):
        normalize("rna", np.array([[np.inf]]), config)


def test_fitted_provenance_cannot_claim_non_train_split() -> None:
    with pytest.raises(ValidationError):
        TransformerProvenance(
            transformer_name="rna_standard_scaler",
            fit_split="all",  # type: ignore[arg-type]
            n_fit_samples=10,
            n_features=3,
            fitted_at=TransformerProvenance.utc_now(),
            parameters_hash="x",
        )
