"""Per-assay normalization: bulk RNA counts, log expression, protein intensity.

Everything in this module is stateless per-sample math (library-size CPM,
element-wise logs). Anything that learns cross-sample statistics lives in
:mod:`omics_agent.preprocessing.scalers` and can only be fitted on train.
Protein missing values are never filled with 0.
"""

from __future__ import annotations

import numpy as np

from omics_agent.errors import SchemaError
from omics_agent.preprocessing.provenance import StatelessTransformRecord, TransformerProvenance
from omics_agent.schemas.dataset import ModalitySpec
from omics_agent.schemas.enums import AssayStrategy, ValueType
from omics_agent.schemas.preprocess import ModalityPreprocessConfig

_LOG_LIKE_VALUE_TYPES = {
    ValueType.LOG1P,
    ValueType.LOG2_INTENSITY,
    ValueType.ZSCORE,
    ValueType.SYNTHETIC_ABUNDANCE,
}


def default_config_for(modality: str, spec: ModalitySpec) -> ModalityPreprocessConfig:
    """Choose a strategy from the declared ``value_type``. Undeclared fails."""

    if spec.value_type is ValueType.RAW_COUNTS:
        return ModalityPreprocessConfig(strategy=AssayStrategy.BULK_RNA_COUNTS)
    if spec.value_type is ValueType.INTENSITY:
        return ModalityPreprocessConfig(strategy=AssayStrategy.PROTEIN_INTENSITY)
    if spec.value_type in _LOG_LIKE_VALUE_TYPES:
        return ModalityPreprocessConfig(strategy=AssayStrategy.LOG_EXPRESSION)
    raise SchemaError(
        f"Modality '{modality}' has value_type '{spec.value_type.value}'; "
        "no preprocessing strategy can be chosen.",
        how_to_fix=(
            "Set modalities.<name>.value_type to raw_counts, intensity, log1p, or "
            "log2_intensity after human review. The pipeline will not guess whether "
            "the matrix holds counts or intensities."
        ),
    )


def normalize(
    modality: str, raw: np.ndarray, config: ModalityPreprocessConfig
) -> tuple[np.ndarray, StatelessTransformRecord]:
    """Return the ``normalized`` layer and its stateless provenance record.

    NaN entries stay NaN. The caller derives the missing mask from the
    returned array (protein zeros may have become missing here).
    """

    if config.strategy is AssayStrategy.BULK_RNA_COUNTS:
        return _normalize_bulk_rna_counts(modality, raw, config)
    if config.strategy is AssayStrategy.PROTEIN_INTENSITY:
        return _normalize_protein_intensity(modality, raw, config)
    return _normalize_log_expression(modality, raw, config)


def _normalize_bulk_rna_counts(
    modality: str, raw: np.ndarray, config: ModalityPreprocessConfig
) -> tuple[np.ndarray, StatelessTransformRecord]:
    _reject_negative(modality, raw, "raw counts")
    library = np.nansum(raw, axis=1)
    zero_library = library <= 0
    safe_library = np.where(zero_library, np.nan, library)
    normalized = raw / safe_library[:, None] * config.cpm_target
    if config.log1p_after_cpm:
        normalized = np.log1p(normalized)
    return normalized, _record(
        f"{modality}_bulk_rna_counts",
        parameters={
            "cpm_target": config.cpm_target,
            "log1p_after_cpm": config.log1p_after_cpm,
        },
        extras={"n_zero_library_samples": int(zero_library.sum())},
    )


def _normalize_protein_intensity(
    modality: str, raw: np.ndarray, config: ModalityPreprocessConfig
) -> tuple[np.ndarray, StatelessTransformRecord]:
    _reject_negative(modality, raw, "linear intensities")
    work = raw.astype(float, copy=True)
    n_zeros_as_missing = 0
    if config.zeros_are_missing:
        zero_mask = work == 0
        n_zeros_as_missing = int(zero_mask.sum())
        work[zero_mask] = np.nan
    elif config.log2_transform and bool((work == 0).any()):
        raise SchemaError(
            f"Modality '{modality}' has zero intensities but zeros_are_missing=false "
            "and log2_transform=true. log2(0) is undefined.",
            how_to_fix=(
                "Either keep zeros_are_missing: true (a zero intensity means 'not "
                "quantified'), or disable log2_transform. Filling with 0 is not an option."
            ),
        )
    if config.log2_transform:
        work = np.log2(work)
    return work, _record(
        f"{modality}_protein_intensity",
        parameters={
            "zeros_are_missing": config.zeros_are_missing,
            "log2_transform": config.log2_transform,
        },
        extras={"n_zeros_as_missing": n_zeros_as_missing},
    )


def _normalize_log_expression(
    modality: str, raw: np.ndarray, config: ModalityPreprocessConfig
) -> tuple[np.ndarray, StatelessTransformRecord]:
    del config
    if bool(np.isinf(raw).any()):
        raise SchemaError(
            f"Modality '{modality}' contains infinite values.",
            how_to_fix=(
                "Infinities usually mean log2 of 0 upstream. Fix the source table; "
                "the pipeline will not silently clip them."
            ),
        )
    return raw.astype(float, copy=True), _record(
        f"{modality}_log_expression", parameters={"passthrough": True}, extras={}
    )


def _reject_negative(modality: str, raw: np.ndarray, expected: str) -> None:
    observed = raw[~np.isnan(raw)]
    if observed.size and bool((observed < 0).any()):
        raise SchemaError(
            f"Modality '{modality}' declared {expected} but contains negative values.",
            how_to_fix=(
                "Check the declared value_type. Negative values suggest the matrix is "
                "already log-scaled or centered; use the log_expression strategy instead."
            ),
        )


def _record(
    name: str, *, parameters: dict[str, object], extras: dict[str, object]
) -> StatelessTransformRecord:
    return StatelessTransformRecord(
        transformer_name=name,
        kind="stateless_per_sample",
        parameters=parameters,
        applied_at=TransformerProvenance.utc_now(),
        extras=extras,
    )
