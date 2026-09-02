"""Primitive metrics. Constant vectors yield NA, never a silent 0 correlation."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    t, p = _finite_pair(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if t.size == 0:
        return None
    return float(np.mean((t - p) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    t, p = _finite_pair(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if t.size == 0:
        return None
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    value = mse(y_true, y_pred)
    return None if value is None else float(np.sqrt(value))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    t, p = _finite_pair(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if t.size < 2:
        return None
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot <= 1e-15:
        return None
    return 1.0 - ss_res / ss_tot


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str,
) -> tuple[float | None, int]:
    """Pearson or Spearman correlation.

    Returns
    -------
    value, n_valid
        ``value`` is None (NA) when either vector is constant, has fewer
        than 2 finite pairs, or the method is undefined. ``n_valid`` is the
        number of finite pairs actually used.
    """

    t, p = _finite_pair(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    n_valid = int(t.size)
    if n_valid < 2:
        return None, n_valid
    if np.std(t) <= 1e-15 or np.std(p) <= 1e-15:
        return None, n_valid
    if method == "pearson":
        value = float(np.corrcoef(t, p)[0, 1])
        return (None if not np.isfinite(value) else value), n_valid
    if method == "spearman":
        result = spearmanr(t, p)
        value = float(result.statistic)
        return (None if not np.isfinite(value) else value), n_valid
    raise ValueError(f"Unknown correlation method {method!r}")
