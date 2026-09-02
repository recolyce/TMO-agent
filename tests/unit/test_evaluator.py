from __future__ import annotations

import numpy as np

from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.evaluation.metrics import correlation


def test_constant_feature_pcc_is_na() -> None:
    y_true = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    y_pred = np.array([[0.5, 2.1], [0.4, 2.8], [0.6, 4.2]])
    mask = np.ones_like(y_true, dtype=bool)
    report = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        feature_names=["const", "var"],
        instance_ids=["a", "b", "c"],
        group_ids=["u1", "u2", "u3"],
        model_name="toy",
        split="val",
        target_modality="protein",
        primary_metric="pcc_macro",
        bootstrap_replicates=20,
        seed=1,
    )
    by_name = {row["feature"]: row for row in report.per_feature}
    assert by_name["const"]["pcc"] is None
    assert by_name["const"]["n_valid_pcc"] == 3
    assert by_name["var"]["pcc"] is not None
    pcc_macro = next(item for item in report.scalars if item.name == "pcc_macro")
    assert pcc_macro.n_valid == 1
    assert pcc_macro.n_total == 2
    assert any("undefined PCC" in warning for warning in report.warnings)


def test_correlation_helper_matches_evaluator() -> None:
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([2.0, 3.0, 4.0])
    value, n = correlation(x, y, method="pearson")
    assert value is None
    assert n == 3


def test_coverage_counts_only_observed_targets() -> None:
    y_true = np.array([[1.0, np.nan], [2.0, 3.0]])
    y_pred = np.array([[1.1, 0.0], [1.8, 3.2]])
    mask = np.isfinite(y_true)
    report = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        feature_names=["a", "b"],
        instance_ids=["i1", "i2"],
        group_ids=["g1", "g2"],
        model_name="toy",
        split="val",
        target_modality="protein",
        primary_metric="mse",
        bootstrap_replicates=20,
        seed=2,
    )
    assert report.n_observed_targets == 3
    assert report.n_possible_targets == 4
    assert report.coverage == 0.75
