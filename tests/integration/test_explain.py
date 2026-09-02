"""Frozen-model explanation: val only, multiple IG baselines, no test leak."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.dyn_helpers import make_bundle, make_split_data, make_task
from tests.unit.http_prefix import PrefixFakeTransport
from typer.testing import CliRunner

from omics_agent.cli import app
from omics_agent.interpretation.runner import run_explanation
from omics_agent.models import get_model
from omics_agent.pipeline import write_experiment_yaml
from omics_agent.schemas.enums import IgBaselineName, SplitName, TaskKind
from omics_agent.schemas.experiment import (
    EvaluationConfig,
    ExperimentConfig,
    ModelParams,
    SplitConfig,
    SplitFractions,
    TaskConfig,
)
from omics_agent.schemas.interpretation import InterpretationConfig
from omics_agent.schemas.optimization import OptimizationConfig

torch = pytest.importorskip("torch")

runner = CliRunner()
FAST = {
    "epochs": 3,
    "batch_size": 16,
    "hidden_dim": 12,
    "emb_dim": 10,
    "device": "cpu",
    "patience": 2,
    "seed": 11,
}


def test_explain_ig_ablation_permutation_val_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omics_agent.preprocessing.bundle import MultiOmicsBundle

    bundle = make_bundle(n_units=8)
    task = make_task()
    train, val = make_split_data(bundle, task)
    plugin = get_model("gru")
    plugin.fit(train, val, ModelParams(name="gru", params=FAST))
    plugin.save(tmp_path / "ckpt")

    exp = ExperimentConfig(
        schema_version="1.0",
        experiment_id="m7_explain",
        dataset=Path("config/dataset.example.yaml"),
        seed=11,
        task=TaskConfig(
            kind=TaskKind.SUBJECT_FORECAST,
            target_modality="protein",
            input_modalities=["rna", "protein"],
        ),
        split=SplitConfig(fractions=SplitFractions(train=0.6, val=0.2, test=0.2)),
        models=[ModelParams(name="gru", params=FAST)],
        evaluation=EvaluationConfig(bootstrap_replicates=20),
        optimization=OptimizationConfig(n_trials=1),
        interpretation=InterpretationConfig(
            n_ig_steps=4,
            baselines=[
                IgBaselineName.ZEROS,
                IgBaselineName.TRAIN_MEAN,
                IgBaselineName.LAST_OBSERVATION,
            ],
            n_bootstrap=3,
            n_seeds=2,
            n_folds=2,
            top_n=10,
            min_sign_consistency=0.0,
            min_selection_frequency=0.0,
            min_stability=0.0,
        ),
        output_dir=tmp_path / "run",
    )
    exp_path = tmp_path / "experiment.yaml"
    write_experiment_yaml(exp_path, exp)

    seen: list[SplitName] = []
    original = MultiOmicsBundle.subset

    def spy(self: MultiOmicsBundle, split: SplitName) -> MultiOmicsBundle:
        seen.append(split)
        return original(self, split)

    monkeypatch.setattr(MultiOmicsBundle, "subset", spy)
    result = run_explanation(
        exp_path,
        model_name="gru",
        output_dir=tmp_path / "run",
        checkpoint=tmp_path / "ckpt",
        plugin=plugin,
        train_data=train,
        val_data=val,
        with_literature=True,
        transport=PrefixFakeTransport(),
    )
    assert SplitName.TEST not in seen
    assert result["objective_split"] == "val"
    assert result["test_labels_visible"] is False
    assert result["claim_kind"] == "hypothesis"
    table = result["table"]
    assert table.n_baselines == 3
    row = table.rows[0]
    assert hasattr(row, "prior_edge_used")
    assert hasattr(row, "embedding_supported")
    assert hasattr(row, "ablation_delta")
    text = Path(result["candidates_md"]).read_text(encoding="utf-8")
    assert "hypothesis" in text
    assert "是首次发现" not in text
    assert "prior_edge_used" in text
    # Fake transport has no routes: literature rows are N / absence phrase.
    lit = Path(result["literature_md"]).read_text(encoding="utf-8")
    assert "在本次检索范围内未找到直接证据" in lit or "hypothesis" in lit


def test_plugin_explain_returns_ig_rows() -> None:
    train, val = make_split_data(make_bundle(n_units=6), make_task())
    plugin = get_model("gru")
    plugin.fit(train, val, ModelParams(name="gru", params=FAST))
    table = plugin.explain(val, ["P1"])
    assert table.method == "captum_integrated_gradients"
    assert table.rows
    assert "not causation" in table.caveat.lower()


def test_cli_lists_explain_and_literature() -> None:
    explained = runner.invoke(app, ["explain", "--help"])
    assert explained.exit_code == 0
    assert "Integrated Gradients" in explained.output
    lit = runner.invoke(app, ["literature-check", "--help"])
    assert lit.exit_code == 0
    assert "needs_review" in lit.output
