"""Resample experimental units, never individual feature cells."""

from __future__ import annotations

import numpy as np

from omics_agent.evaluation.metrics import correlation, mae, mse, r2_score
from omics_agent.schemas.evaluation import BootstrapCI


def bootstrap_unit_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    group_ids: list[str],
    n_replicates: int,
    seed: int,
) -> list[BootstrapCI]:
    """Percentile CIs for MSE/MAE and macro PCC/Spearman/R2.

    Each replicate draws units with replacement and concatenates all of
    that unit's instances. Empty or undefined metrics become None.
    """

    units = np.array(sorted(set(group_ids)))
    index = {unit: np.where(np.array(group_ids) == unit)[0] for unit in units}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "mse": [],
        "mae": [],
        "pcc": [],
        "spearman": [],
        "r2": [],
    }
    if units.size == 0:
        return [
            BootstrapCI(metric=name, low=None, high=None, n_replicates=0, n_units=0)
            for name in collected
        ]

    for _ in range(n_replicates):
        drawn = rng.choice(units, size=units.size, replace=True)
        rows = np.concatenate([index[unit] for unit in drawn])
        yt = y_true[rows]
        yp = y_pred[rows]
        m = mask[rows]
        yt_m = np.where(m, yt, np.nan)
        yp_m = np.where(m, yp, np.nan)
        _append(collected, "mse", mse(yt_m, yp_m))
        _append(collected, "mae", mae(yt_m, yp_m))
        pcc, _ = _macro_corr(yt_m, yp_m, "pearson")
        sp, _ = _macro_corr(yt_m, yp_m, "spearman")
        _append(collected, "pcc", pcc)
        _append(collected, "spearman", sp)
        r2_macro, _ = _macro_r2(yt_m, yp_m)
        _append(collected, "r2", r2_macro)

    out: list[BootstrapCI] = []
    for name, values in collected.items():
        if len(values) < 2:
            out.append(
                BootstrapCI(
                    metric=name,
                    low=None,
                    high=None,
                    n_replicates=n_replicates,
                    n_units=int(units.size),
                )
            )
            continue
        arr = np.asarray(values, dtype=float)
        out.append(
            BootstrapCI(
                metric=name,
                low=float(np.quantile(arr, 0.025)),
                high=float(np.quantile(arr, 0.975)),
                n_replicates=n_replicates,
                n_units=int(units.size),
            )
        )
    return out


def _append(store: dict[str, list[float]], key: str, value: float | None) -> None:
    if value is not None and np.isfinite(value):
        store[key].append(float(value))


def _macro_corr(
    y_true: np.ndarray, y_pred: np.ndarray, method: str
) -> tuple[float | None, int]:
    values: list[float] = []
    n_valid_features = 0
    for j in range(y_true.shape[1]):
        value, n = correlation(y_true[:, j], y_pred[:, j], method=method)
        if value is None:
            continue
        n_valid_features += 1
        values.append(value)
    if not values:
        return None, n_valid_features
    return float(np.mean(values)), n_valid_features


def _macro_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, int]:
    values: list[float] = []
    n_valid_features = 0
    for j in range(y_true.shape[1]):
        value = r2_score(y_true[:, j], y_pred[:, j])
        if value is None:
            continue
        n_valid_features += 1
        values.append(value)
    if not values:
        return None, n_valid_features
    return float(np.mean(values)), n_valid_features
