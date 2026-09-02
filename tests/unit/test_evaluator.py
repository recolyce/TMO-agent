from __future__ import annotations

import numpy as np
import pytest

from omics_agent.evaluation.bootstrap import bootstrap_unit_metrics
from omics_agent.evaluation.evaluator import _macro_r2, evaluate_predictions
from omics_agent.evaluation.metrics import correlation, r2_score


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


def test_bootstrap_r2_is_macro_not_pooled() -> None:
    """A high-variance feature must not dominate the bootstrap R2 (rule 5)."""

    import inspect

    from omics_agent.evaluation import bootstrap as boot_mod

    y_true = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y_pred = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    pooled = r2_score(y_true, y_pred)
    macro, _ = _macro_r2(y_true, y_pred)
    boot_macro, _ = boot_mod._macro_r2(y_true, y_pred)
    assert pooled is not None and macro is not None
    assert pooled != pytest.approx(macro, abs=0.05)
    assert boot_macro == pytest.approx(macro)
    source = inspect.getsource(boot_mod.bootstrap_unit_metrics)
    assert "_macro_r2" in source
    assert "r2_score(yt_m, yp_m)" not in source
    cis = bootstrap_unit_metrics(
        y_true=y_true,
        y_pred=y_pred,
        mask=np.ones_like(y_true, dtype=bool),
        group_ids=["a", "b", "c", "d"],
        n_replicates=20,
        seed=3,
    )
    assert any(item.metric == "r2" for item in cis)
