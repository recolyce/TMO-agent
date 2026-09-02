"""Five-arm prior ablation: same split, same evaluator, shared HPO budget."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omics_agent.cli import app
from omics_agent.data_sources.synthetic import generate_synthetic_dataset
from omics_agent.pipeline import write_experiment_yaml
from omics_agent.priors import run_prior_ablation
from omics_agent.schemas.enums import EmbeddingModelName, PriorAblation, SamplingDesign, TaskKind
from omics_agent.schemas.experiment import (
    EvaluationConfig,
    ExperimentConfig,
    ModelParams,
    SplitConfig,
    SplitFractions,
    TaskConfig,
)
from omics_agent.schemas.optimization import OptimizationConfig
from omics_agent.schemas.priors import EmbeddingModelConfig, PriorAblationConfig

pytest.importorskip("torch")

runner = CliRunner()
SEED = 20260901
FAST = {
    "epochs": 3,
    "batch_size": 16,
    "hidden_dim": 16,
    "emb_dim": 12,
    "device": "cpu",
    "patience": 2,
    "val_every": 1,
    "seed": SEED,
}


def _workspace(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    generate_synthetic_dataset(data_dir, design=SamplingDesign.LONGITUDINAL, seed=SEED)
    run_dir = root / "run"
    config = ExperimentConfig(
        schema_version="1.0",
        experiment_id="m6_prior_ablation",
        dataset=data_dir / "dataset.yaml",
        seed=SEED,
        task=TaskConfig(
            kind=TaskKind.SUBJECT_FORECAST,
            target_modality="protein",
            input_modalities=["rna", "protein"],
            primary_metric="protein_macro_pcc",
        ),
        split=SplitConfig(
            group_columns=["experimental_unit_id", "subject_id"],
            fractions=SplitFractions(train=0.6, val=0.2, test=0.2),
            block_experiment_batch=False,
        ),
        models=[ModelParams(name="gru", params=FAST)],
        evaluation=EvaluationConfig(bootstrap_replicates=20),
        optimization=OptimizationConfig(n_trials=1),
        priors=PriorAblationConfig(
            seeds=[SEED, SEED + 1],
            share_hpo=True,
            graph_weight=0.05,
            embedding=EmbeddingModelConfig(name=EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT),
        ),
        output_dir=run_dir,
    )
    exp_path = root / "experiment.yaml"
    write_experiment_yaml(exp_path, config)
    return exp_path, run_dir


def test_five_arms_same_split_val_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omics_agent.preprocessing.bundle import MultiOmicsBundle
    from omics_agent.schemas.enums import SplitName

    exp_path, run_dir = _workspace(tmp_path)
    seen: list[SplitName] = []
    original = MultiOmicsBundle.subset

    def spy(self: MultiOmicsBundle, split: SplitName) -> MultiOmicsBundle:
        seen.append(split)
        return original(self, split)

    monkeypatch.setattr(MultiOmicsBundle, "subset", spy)
    result = run_prior_ablation(exp_path, model_name="gru", n_trials=0)
    assert SplitName.TEST not in seen
    payload = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    assert payload["objective_split"] == "val"
    assert payload["test_labels_visible"] is False
    arms = {row["ablation"] for row in payload["table"]}
    assert arms == {item.value for item in PriorAblation}
    assert payload["hpo_budget"] == 0
    # Combined (three priors) has more trainable parameters than no_prior.
    by_arm = {row["ablation"]: row for row in payload["table"]}
    assert by_arm["combined"]["n_parameters"]["mean"] > by_arm["no_prior"]["n_parameters"]["mean"]
    assert (run_dir / "splits.parquet").is_file()
    assert (run_dir / "priors" / "bundle.yaml").is_file()
    assert "STRING" in Path(result["report_md"]).read_text(encoding="utf-8")


def test_cli_lists_the_command() -> None:
    result = runner.invoke(app, ["ablate-priors", "--help"])
    assert result.exit_code == 0
    assert "random_graph" in result.output
    assert "physical" in result.output.lower()
