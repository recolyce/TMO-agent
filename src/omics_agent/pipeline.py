"""Deterministic benchmark orchestration used by the CLI.

This is not an Agent loop. It loads a reviewed manifest, locks a split,
fits train-only preprocessors, fits registered baselines, and scores them
with the independent evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.hashing import collect_run_hashes, hash_dataframe
from omics_agent.models import get_model
from omics_agent.models.tasks import DataForModel, prepare_split_data
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.reporting.benchmark import write_benchmark_report
from omics_agent.reporting.tracking import log_benchmark_run
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import SplitName
from omics_agent.schemas.evaluation import EvaluationReport
from omics_agent.schemas.experiment import (
    ExperimentConfig,
    assert_task_matches_design,
    load_experiment,
)
from omics_agent.splitting.split import assign_splits, write_splits

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _resolve(experiment_path: Path, raw: Path) -> Path:
    return raw if raw.is_absolute() else (experiment_path.parent / raw).resolve()


def run_benchmark(
    experiment_path: Path,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
    unlock_test: bool | None = None,
) -> dict[str, Any]:
    """Run the milestone-1 baseline benchmark.

    Parameters
    ----------
    experiment_path:
        Path to ``experiment.yaml``.
    output_dir:
        Where splits, models, reports, and MLflow files are written.
        Defaults to ``experiment.output_dir`` or ``outputs/<experiment_id>``.
    dry_run:
        Validate configs and print the plan; do not fit or write models.
    unlock_test:
        Override ``evaluation.unlock_test``. Production runs should leave
        this false. The toy command sets it true with an explicit disclaimer.
    """

    experiment_path = experiment_path.resolve()
    experiment = load_experiment(experiment_path)
    manifest_path = _resolve(experiment_path, experiment.dataset)
    manifest = load_manifest(manifest_path)
    manifest.require_approved_for_training()
    assert_task_matches_design(
        task=experiment.task,
        sampling_design=manifest.design.sampling_design,
        pairing_level=manifest.design.pairing_level,
        modalities=list(manifest.modalities),
    )
    dest = output_dir or experiment.output_dir or Path("outputs") / experiment.experiment_id
    dest = dest.resolve()
    plan = {
        "experiment_id": experiment.experiment_id,
        "dataset_id": manifest.dataset_id,
        "manifest": str(manifest_path),
        "output_dir": str(dest),
        "models": [item.name for item in experiment.models],
        "task": experiment.task.kind.value,
        "seed": experiment.seed,
        "unlock_test": experiment.evaluation.unlock_test if unlock_test is None else unlock_test,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    dest.mkdir(parents=True, exist_ok=True)
    bundle = load_local_bundle(manifest_path)
    splits = assign_splits(bundle, experiment.split, seed=experiment.seed)
    split_path = dest / "splits.parquet"
    write_splits(splits, split_path)
    split_units = splits[["experimental_unit_id", "split"]].drop_duplicates()
    labeled = bundle.with_split(split_units)
    processed = labeled.apply_train_only_preprocessing()
    _assert_fit_split_train(processed)
    mudata_path = dest / "dataset.h5mu"
    processed.write_h5mu(mudata_path)

    train = processed.subset(SplitName.TRAIN)
    val = processed.subset(SplitName.VAL)
    test = processed.subset(SplitName.TEST)
    train_data = prepare_split_data(
        full=processed, train=train, split_bundle=train, split=SplitName.TRAIN, task=experiment.task
    )
    val_data = prepare_split_data(
        full=processed, train=train, split_bundle=val, split=SplitName.VAL, task=experiment.task
    )
    test_data = prepare_split_data(
        full=processed, train=train, split_bundle=test, split=SplitName.TEST, task=experiment.task
    )

    score_test = plan["unlock_test"] is True
    reports: list[EvaluationReport] = []
    for spec in experiment.models:
        plugin = get_model(spec.name)
        fit_result = plugin.fit(train_data, val_data, spec)
        model_dir = dest / "models" / spec.name
        plugin.save(model_dir)
        for split_name, data in ((SplitName.VAL, val_data), (SplitName.TEST, test_data)):
            if split_name is SplitName.TEST and not score_test:
                continue
            pred = plugin.predict(data)
            reports.append(
                evaluate_predictions(
                    y_true=data.forecast.y_true,
                    y_pred=pred.y_pred,
                    mask=data.forecast.y_mask,
                    feature_names=data.forecast.feature_names,
                    instance_ids=data.forecast.instance_ids,
                    group_ids=data.forecast.group_ids,
                    model_name=spec.name,
                    split=split_name.value,
                    target_modality=experiment.task.target_modality,
                    primary_metric=experiment.task.primary_metric,
                    bootstrap_replicates=experiment.evaluation.bootstrap_replicates,
                    seed=experiment.seed,
                )
            )
        del fit_result

    hashes = collect_run_hashes(
        package_root=PACKAGE_ROOT,
        repo_root=REPO_ROOT,
        data_frames={
            "observations": processed.observations,
            "samples": processed.sample_sheet,
        },
        split_frame=splits,
        config_payload=experiment.model_dump(mode="json"),
        seed=experiment.seed,
    )
    hashes["split_file_hash"] = hash_dataframe(pd.read_parquet(split_path))
    notes = [
        "Preprocessing transformers record fit_split=train.",
        "Evaluator, split membership, and primary_metric are not writable by an optimizer.",
    ]
    if score_test:
        notes.append(experiment.evaluation.toy_unlock_disclaimer)

    report_path = dest / "reports" / "benchmark.json"
    write_benchmark_report(
        report_path, experiment_id=experiment.experiment_id, hashes=hashes, reports=reports, notes=notes
    )
    run_id = log_benchmark_run(
        tracking_uri=dest / "mlruns",
        experiment_name=experiment.experiment_id,
        run_name=f"{experiment.experiment_id}-baselines",
        hashes=hashes,
        seed=experiment.seed,
        params={
            "task": experiment.task.kind.value,
            "target_modality": experiment.task.target_modality,
            "models": ",".join(item.name for item in experiment.models),
        },
        reports=reports,
        artifacts=[report_path, report_path.with_suffix(".md"), split_path],
    )
    provenance_path = dest / "preprocessing_provenance.json"
    provenance_path.write_text(
        yaml.safe_dump(processed.provenance, sort_keys=False), encoding="utf-8"
    )
    return {
        **plan,
        "split_path": str(split_path),
        "report_json": str(report_path),
        "report_md": str(report_path.with_suffix(".md")),
        "mlflow_run_id": run_id,
        "hashes": hashes,
        "n_reports": len(reports),
        "provenance_path": str(provenance_path),
        "mudata_path": str(mudata_path),
    }


def write_experiment_yaml(path: Path, config: ExperimentConfig) -> None:
    """Serialize an experiment config for the toy runner."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["dataset"] = str(config.dataset)
    if config.output_dir is not None:
        payload["output_dir"] = str(config.output_dir)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _assert_fit_split_train(bundle: MultiOmicsBundle) -> None:
    for record in bundle.provenance:
        if record.get("fit_split") != "train":
            raise RuntimeError(
                f"Preprocessor {record.get('transformer_name')} has fit_split="
                f"{record.get('fit_split')!r}, expected 'train'."
            )


def prepare_model_data(bundle: MultiOmicsBundle, experiment: ExperimentConfig) -> dict[str, DataForModel]:
    """Helper used by tests to build train/val/test model data."""

    train = bundle.subset(SplitName.TRAIN)
    return {
        name.value: prepare_split_data(
            full=bundle,
            train=train,
            split_bundle=bundle.subset(name),
            split=name,
            task=experiment.task,
        )
        for name in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST)
    }
