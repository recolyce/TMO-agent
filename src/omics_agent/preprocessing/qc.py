"""Per-sample and per-feature QC metrics.

Metrics are computed on the raw layer and the missing mask. They describe
the data; they never repair it. A bad sample stays visible, not imputed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omics_agent.preprocessing.bundle import MultiOmicsBundle


def per_sample_qc(values: np.ndarray, missing: np.ndarray) -> pd.DataFrame:
    """One row per observation: coverage and raw signal totals."""

    n_observed = (~missing).sum(axis=1)
    totals = np.nansum(np.where(missing, np.nan, values), axis=1)
    medians = np.full(values.shape[0], np.nan)
    has_data = n_observed > 0
    if has_data.any():
        masked = np.where(missing, np.nan, values)
        medians[has_data] = np.nanmedian(masked[has_data], axis=1)
    return pd.DataFrame(
        {
            "qc_n_observed": n_observed.astype(int),
            "qc_pct_missing": missing.mean(axis=1),
            "qc_total_signal": np.where(has_data, totals, np.nan),
            "qc_median_signal": medians,
        }
    )


def per_feature_qc(values: np.ndarray, missing: np.ndarray) -> pd.DataFrame:
    """One row per feature: coverage, moments, and zero-variance flag."""

    masked = np.where(missing, np.nan, values)
    n_observed = (~missing).sum(axis=0)
    has_data = n_observed > 0
    means = np.full(values.shape[1], np.nan)
    stds = np.full(values.shape[1], np.nan)
    if has_data.any():
        means[has_data] = np.nanmean(masked[:, has_data], axis=0)
        stds[has_data] = np.nanstd(masked[:, has_data], axis=0)
    return pd.DataFrame(
        {
            "qc_n_observed": n_observed.astype(int),
            "qc_pct_missing": missing.mean(axis=0),
            "qc_mean": means,
            "qc_std": stds,
            "qc_zero_variance": has_data & (stds < 1e-12),
        }
    )


def compute_qc(bundle: MultiOmicsBundle) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-modality QC frames, indexed like observations / feature names."""

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for modality in bundle.matrices:
        raw = bundle.layers.get(modality, {}).get("raw", bundle.matrices[modality])
        missing = bundle.missing[modality]
        sample = per_sample_qc(raw, missing)
        sample.insert(0, "observation_id", bundle.observations["observation_id"].to_numpy())
        feature = per_feature_qc(raw, missing)
        feature.insert(0, "feature_id", bundle.feature_names[modality])
        out[modality] = {"per_sample": sample, "per_feature": feature}
    return out


def write_qc_json(bundle: MultiOmicsBundle, path: Path) -> dict[str, Any]:
    """Write ``qc_metrics.json`` and return the payload."""

    payload: dict[str, Any] = {"dataset_id": bundle.manifest.dataset_id, "modalities": {}}
    for modality, frames in compute_qc(bundle).items():
        sample = frames["per_sample"]
        feature = frames["per_feature"]
        payload["modalities"][modality] = {
            "summary": {
                "n_observations": int(len(sample)),
                "n_features": int(len(feature)),
                "pct_missing_overall": float(bundle.missing[modality].mean()),
                "n_samples_fully_missing": int((sample["qc_n_observed"] == 0).sum()),
                "n_features_fully_missing": int((feature["qc_n_observed"] == 0).sum()),
                "n_zero_variance_features": int(feature["qc_zero_variance"].sum()),
            },
            "per_sample": json.loads(sample.to_json(orient="records")),
            "per_feature": json.loads(feature.to_json(orient="records")),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
