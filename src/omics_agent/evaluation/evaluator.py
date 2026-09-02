"""Unified evaluator. All models are scored with the same function."""

from __future__ import annotations

import numpy as np

from omics_agent.errors import MetricError
from omics_agent.evaluation.bootstrap import bootstrap_unit_metrics
from omics_agent.evaluation.metrics import correlation, mae, mse, r2_score, rmse
from omics_agent.schemas.evaluation import EvaluationReport, ScalarMetric


def evaluate_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
    instance_ids: list[str],
    group_ids: list[str],
    model_name: str,
    split: str,
    target_modality: str,
    primary_metric: str,
    bootstrap_replicates: int,
    seed: int,
) -> EvaluationReport:
    """Score predictions at observed target positions only.

    PCC / Spearman / R2 are NA for constant features or samples. Macro
    averages skip those NA values and report how many were defined.
    """

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if yt.shape != yp.shape or yt.shape != m.shape:
        raise MetricError(
            f"y_true {yt.shape}, y_pred {yp.shape}, and mask {m.shape} differ.",
            how_to_fix="The model must return one prediction per instance and target feature.",
        )
    if yt.shape[0] != len(instance_ids) or yt.shape[1] != len(feature_names):
        raise MetricError(
            "Prediction table is not aligned with instance_ids / feature_names.",
            how_to_fix="Do not subset or transpose model output before calling the evaluator.",
        )

    yt_m = np.where(m, yt, np.nan)
    yp_m = np.where(m, yp, np.nan)
    n_obs = int(m.sum())
    n_possible = int(m.size)
    coverage = float(n_obs / n_possible) if n_possible else 0.0

    warnings: list[str] = []
    scalars: list[ScalarMetric] = []
    scalars.extend(
        [
            _scalar("mse", mse(yt_m, yp_m), n_obs, n_possible),
            _scalar("mae", mae(yt_m, yp_m), n_obs, n_possible),
            _scalar("rmse", rmse(yt_m, yp_m), n_obs, n_possible),
            _scalar("r2_pooled", r2_score(yt_m, yp_m), n_obs, n_possible),
        ]
    )
    pcc_macro, pcc_n = _macro_corr(yt_m, yp_m, "pearson")
    sp_macro, sp_n = _macro_corr(yt_m, yp_m, "spearman")
    r2_macro, r2_n = _macro_r2(yt_m, yp_m)
    scalars.append(
        ScalarMetric(
            name="pcc_macro",
            value=pcc_macro,
            n_valid=pcc_n,
            n_total=yt.shape[1],
            note="Mean of per-feature Pearson correlations; constant features are NA and excluded.",
        )
    )
    scalars.append(
        ScalarMetric(
            name="spearman_macro",
            value=sp_macro,
            n_valid=sp_n,
            n_total=yt.shape[1],
            note="Mean of per-feature Spearman correlations; constant features are NA and excluded.",
        )
    )
    scalars.append(
        ScalarMetric(
            name="r2_macro",
            value=r2_macro,
            n_valid=r2_n,
            n_total=yt.shape[1],
            note="Mean of per-feature R2; constant true vectors are NA and excluded.",
        )
    )
    pcc_pooled, n_pcc_pairs = correlation(yt_m.ravel(), yp_m.ravel(), method="pearson")
    sp_pooled, n_sp_pairs = correlation(yt_m.ravel(), yp_m.ravel(), method="spearman")
    scalars.append(
        ScalarMetric(
            name="pcc_pooled",
            value=pcc_pooled,
            n_valid=n_pcc_pairs,
            n_total=n_obs,
            note="Pooled over all observed cells. High-variance features can dominate.",
        )
    )
    scalars.append(
        ScalarMetric(
            name="spearman_pooled",
            value=sp_pooled,
            n_valid=n_sp_pairs,
            n_total=n_obs,
        )
    )

    per_feature = []
    n_const = 0
    for j, name in enumerate(feature_names):
        pcc, n_p = correlation(yt_m[:, j], yp_m[:, j], method="pearson")
        sp, n_s = correlation(yt_m[:, j], yp_m[:, j], method="spearman")
        if pcc is None:
            n_const += 1
        per_feature.append(
            {
                "feature": name,
                "mse": mse(yt_m[:, j], yp_m[:, j]),
                "mae": mae(yt_m[:, j], yp_m[:, j]),
                "pcc": pcc,
                "spearman": sp,
                "r2": r2_score(yt_m[:, j], yp_m[:, j]),
                "n_valid_pcc": n_p,
                "n_valid_spearman": n_s,
                "n_observed": int(m[:, j].sum()),
            }
        )
    if n_const:
        warnings.append(
            f"{n_const} feature(s) had undefined PCC (constant or <2 finite pairs) and are NA."
        )

    per_sample = []
    for i, instance_id in enumerate(instance_ids):
        pcc, n_p = correlation(yt_m[i], yp_m[i], method="pearson")
        sp, n_s = correlation(yt_m[i], yp_m[i], method="spearman")
        per_sample.append(
            {
                "instance_id": instance_id,
                "group_id": group_ids[i],
                "mse": mse(yt_m[i], yp_m[i]),
                "mae": mae(yt_m[i], yp_m[i]),
                "pcc": pcc,
                "spearman": sp,
                "r2": r2_score(yt_m[i], yp_m[i]),
                "n_valid_pcc": n_p,
                "n_valid_spearman": n_s,
                "n_observed": int(m[i].sum()),
            }
        )

    boot = bootstrap_unit_metrics(
        y_true=yt,
        y_pred=yp,
        mask=m,
        group_ids=group_ids,
        n_replicates=bootstrap_replicates,
        seed=seed,
    )

    named = {item.name: item.value for item in scalars}
    # Convenience aliases used in experiment YAML.
    named["protein_macro_pcc"] = named.get("pcc_macro")
    named["macro_pcc"] = named.get("pcc_macro")
    if primary_metric not in named:
        warnings.append(
            f"primary_metric '{primary_metric}' is not a computed name; "
            "available: mse, mae, rmse, pcc_macro, spearman_macro, r2_macro, "
            "pcc_pooled, protein_macro_pcc."
        )
        primary_value = named.get("pcc_macro")
    else:
        primary_value = named[primary_metric]

    return EvaluationReport(
        model_name=model_name,
        split=split,
        target_modality=target_modality,
        n_instances=int(yt.shape[0]),
        n_features=int(yt.shape[1]),
        coverage=coverage,
        n_observed_targets=n_obs,
        n_possible_targets=n_possible,
        scalars=scalars,
        per_feature=per_feature,
        per_sample=per_sample,
        bootstrap=boot,
        primary_metric=primary_metric,
        primary_value=primary_value,
        warnings=warnings,
    )


def _scalar(name: str, value: float | None, n_valid: int, n_total: int) -> ScalarMetric:
    return ScalarMetric(name=name, value=value, n_valid=n_valid, n_total=n_total)


def _macro_corr(
    y_true: np.ndarray, y_pred: np.ndarray, method: str
) -> tuple[float | None, int]:
    values: list[float] = []
    for j in range(y_true.shape[1]):
        value, _n = correlation(y_true[:, j], y_pred[:, j], method=method)
        if value is not None:
            values.append(value)
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def _macro_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, int]:
    values: list[float] = []
    for j in range(y_true.shape[1]):
        value = r2_score(y_true[:, j], y_pred[:, j])
        if value is not None:
            values.append(value)
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)
