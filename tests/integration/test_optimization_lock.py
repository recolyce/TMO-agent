"""Security regressions for milestone 5.

Proves: the tuner never touches the test split, the final test runs exactly
once, tuning after the test is refused, and any modification of the frozen
split / checkpoint / config / decision / evaluator code is rejected.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from omics_agent.cli import app
from omics_agent.data_sources.synthetic import generate_synthetic_dataset
from omics_agent.errors import ArtifactIntegrityError
from omics_agent.errors import TestLockError as LockError
from omics_agent.optimization import run_final_test, run_tuning
from omics_agent.optimization.lock import read_lock
from omics_agent.pipeline import write_experiment_yaml
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.enums import SamplingDesign, SplitName
from omics_agent.schemas.experiment import (
    EvaluationConfig,
    ExperimentConfig,
    ModelParams,
    SplitConfig,
    SplitFractions,
    TaskConfig,
    TaskKind,
)
from omics_agent.schemas.optimization import (
    FreezeManifest,
    OptimizationConfig,
    OptimizationDecision,
)

runner = CliRunner()
SEED = 20260901


def _make_workspace(root: Path) -> tuple[Path, Path]:
    """Synthetic longitudinal dataset + experiment yaml. Returns (yaml, run_dir)."""

    data_dir = root / "data"
    generate_synthetic_dataset(data_dir, design=SamplingDesign.LONGITUDINAL, seed=SEED)
    run_dir = root / "run"
    config = ExperimentConfig(
        schema_version="1.0",
        experiment_id="m5_lock_test",
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
        models=[ModelParams(name="ridge"), ModelParams(name="last_value")],
        evaluation=EvaluationConfig(bootstrap_replicates=20),
        optimization=OptimizationConfig(n_trials=3),
        output_dir=run_dir,
    )
    exp_path = root / "experiment.yaml"
    write_experiment_yaml(exp_path, config)
    return exp_path, run_dir


@pytest.fixture(scope="module")
def tuned(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One tuned + frozen ridge workspace shared by the tamper tests."""

    root = tmp_path_factory.mktemp("m5")
    exp_path, run_dir = _make_workspace(root)
    run_tuning(exp_path, model_name="ridge")
    return exp_path, run_dir


def _clone(run_dir: Path, tmp_path: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(run_dir, dest)
    return dest


def test_tuner_is_test_blind_and_decision_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp_path, run_dir = _make_workspace(tmp_path)
    seen: list[SplitName] = []
    original = MultiOmicsBundle.subset

    def spy(self: MultiOmicsBundle, split: SplitName) -> MultiOmicsBundle:
        seen.append(split)
        return original(self, split)

    monkeypatch.setattr(MultiOmicsBundle, "subset", spy)
    result = run_tuning(exp_path, model_name="ridge")
    # The optimizer process never materializes test rows, let alone labels.
    assert SplitName.TEST not in seen
    assert {SplitName.TRAIN, SplitName.VAL} <= set(seen)

    decision = OptimizationDecision.model_validate_json(
        Path(result["decision_path"]).read_text(encoding="utf-8")
    )
    assert decision.study_name == "m5_lock_test::ridge"
    assert decision.n_trials_budget == 3
    assert len(decision.trials) == 3
    assert decision.sampler == "tpe"
    assert decision.sampler_seed == SEED
    assert decision.pruner.kind == "median"
    assert decision.objective_split == "val"
    assert decision.test_labels_visible is False
    assert "alpha" in decision.best_params
    assert result["mlflow_run_id"]

    frozen = FreezeManifest.model_validate_json(
        Path(result["freeze_manifest"]).read_text(encoding="utf-8")
    )
    assert frozen.status == "frozen"
    assert (Path(result["checkpoint_dir"]) / "model.joblib").is_file()
    assert (run_dir / "frozen" / "ridge" / "frozen_experiment.yaml").is_file()


def test_final_test_runs_once_then_everything_is_locked(
    tuned: tuple[Path, Path], tmp_path: Path
) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    result = run_final_test(exp_path, model_name="ridge", output_dir=clone)
    assert Path(result["report_json"]).is_file()
    assert result["primary_value"] is not None
    lock = read_lock(clone)
    assert lock is not None and lock.consumed and lock.final_report_hash

    with pytest.raises(LockError, match="already"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)
    with pytest.raises(LockError, match="one-shot final test"):
        run_tuning(exp_path, model_name="ridge", output_dir=clone)


def test_modified_split_is_rejected(tuned: tuple[Path, Path], tmp_path: Path) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    split_path = clone / "splits.parquet"
    frame = pd.read_parquet(split_path)
    moved = frame.copy()
    moved.loc[moved.index[0], "split"] = "val" if moved.iloc[0]["split"] != "val" else "train"
    moved.to_parquet(split_path)
    with pytest.raises(ArtifactIntegrityError, match="split assignment"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)
    # The refusal must not burn the one-shot lock.
    assert read_lock(clone) is None


def test_modified_checkpoint_is_rejected(tuned: tuple[Path, Path], tmp_path: Path) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    checkpoint = clone / "frozen" / "ridge" / "checkpoint" / "model.joblib"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="checkpoint"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)


def test_modified_frozen_config_is_rejected(tuned: tuple[Path, Path], tmp_path: Path) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    config_path = clone / "frozen" / "ridge" / "frozen_experiment.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"][0]["params"]["alpha"] = 1e-9
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="frozen experiment config"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)


def test_modified_decision_is_rejected(tuned: tuple[Path, Path], tmp_path: Path) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    decision_path = clone / "reports" / "optimization_decision_ridge.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    payload["best_value"] = 0.0
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="decision document"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)


def test_modified_evaluator_code_is_rejected(
    tuned: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp_path, run_dir = tuned
    clone = _clone(run_dir, tmp_path)
    import omics_agent.optimization.lock as lock_module

    real = lock_module.hash_source_tree

    def fake(root: Path) -> str:
        if root.name == "evaluation":
            return "deadbeef" * 8  # what a modified evaluator tree would produce
        return real(root)

    monkeypatch.setattr(lock_module, "hash_source_tree", fake)
    with pytest.raises(ArtifactIntegrityError, match="evaluator source code"):
        run_final_test(exp_path, model_name="ridge", output_dir=clone)


def test_unlock_test_cli_requires_confirm() -> None:
    result = runner.invoke(
        app, ["unlock-test", "--experiment", "does-not-matter.yaml", "--model", "ridge"]
    )
    assert result.exit_code == 1
    assert "one-shot" in result.output
