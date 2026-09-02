from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from omics_agent.cli import app
from omics_agent.data_sources.local import load_local_bundle
from omics_agent.pipeline import run_benchmark, write_experiment_yaml
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

runner = CliRunner()


def _longitudinal_experiment(dataset: Path, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        schema_version="1.0",
        experiment_id="test_longitudinal",
        dataset=dataset,
        seed=20260901,
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
        models=[
            ModelParams(name="last_value"),
            ModelParams(name="ridge", params={"alpha": 1.0}),
            ModelParams(name="time_spline", params={"spline_df": 3, "alpha": 1.0}),
        ],
        evaluation=EvaluationConfig(unlock_test=True, bootstrap_replicates=20),
        output_dir=output_dir,
    )


def test_benchmark_writes_report_and_train_provenance(
    longitudinal_dir: Path, tmp_path: Path
) -> None:
    exp_path = tmp_path / "experiment.yaml"
    run_dir = tmp_path / "run"
    write_experiment_yaml(exp_path, _longitudinal_experiment(longitudinal_dir / "dataset.yaml", run_dir))
    result = run_benchmark(exp_path, output_dir=run_dir, unlock_test=True)
    assert Path(result["report_json"]).is_file()
    assert Path(result["report_md"]).is_file()
    assert "code_hash" in result["hashes"]
    assert "data_hash" in result["hashes"]
    assert "split_hash" in result["hashes"]
    assert "config_hash" in result["hashes"]
    assert result["hashes"]["seed"] == "20260901"
    provenance = Path(result["provenance_path"]).read_text(encoding="utf-8")
    assert "fit_split: train" in provenance
    assert "fit_split: val" not in provenance
    bundle = load_local_bundle(longitudinal_dir / "dataset.yaml")
    assert bundle.sampling_design is SamplingDesign.LONGITUDINAL


def test_cli_validate_and_doctor() -> None:
    doctor = runner.invoke(app, ["doctor"])
    assert doctor.exit_code == 0, doctor.output
    valid = runner.invoke(app, ["validate-manifest", "config/dataset.example.yaml"])
    assert valid.exit_code == 0, valid.output


def test_cli_run_toy(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run-toy", "--output-dir", str(tmp_path / "toy")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "toy" / "longitudinal" / "run" / "reports" / "benchmark.md").is_file()
    assert (tmp_path / "toy" / "rcs" / "run" / "reports" / "benchmark.md").is_file()


def test_rcs_assignment_uses_group_time_forecast(rcs_dir: Path, tmp_path: Path) -> None:
    exp = ExperimentConfig(
        schema_version="1.0",
        experiment_id="test_rcs",
        dataset=rcs_dir / "dataset.yaml",
        seed=20260901,
        task=TaskConfig(
            kind=TaskKind.GROUP_TIME_FORECAST,
            target_modality="protein",
            input_modalities=["rna", "protein"],
            target_time_min=4.0,
            primary_metric="protein_macro_pcc",
        ),
        split=SplitConfig(
            group_columns=["batch"],
            block_experiment_batch=True,
            also_block=["experimental_unit_id"],
            assignment={
                "expA": SplitName.TRAIN,
                "expB": SplitName.VAL,
                "expC": SplitName.TEST,
            },
        ),
        models=[ModelParams(name="last_value"), ModelParams(name="time_spline", params={"spline_df": 3})],
        evaluation=EvaluationConfig(unlock_test=True, bootstrap_replicates=20),
    )
    exp_path = tmp_path / "rcs.yaml"
    write_experiment_yaml(exp_path, exp)
    result = run_benchmark(exp_path, output_dir=tmp_path / "rcs-run", unlock_test=True)
    assert result["n_reports"] >= 2
