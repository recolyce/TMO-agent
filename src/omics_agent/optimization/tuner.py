"""Validation-only Optuna tuning with a fixed budget, seed, study name, pruner.

The objective closure receives train and val data only. Test rows are never
subset, never scored, and never enter the optimizer's API. After the study,
the best configuration is refit on train, saved as a frozen checkpoint, and
every hash the one-shot final test depends on is recorded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.errors import (
    OdeSolverError,
    SchemaError,
    TrainingDivergedError,
)
from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.hashing import (
    collect_run_hashes,
    hash_dataframe,
    hash_mapping,
    hash_source_tree,
    sha256_file,
)
from omics_agent.models import get_model
from omics_agent.models.tasks import DataForModel, prepare_split_data
from omics_agent.optimization.lock import assert_tuning_allowed
from omics_agent.optimization.search_space import suggest_params
from omics_agent.pipeline import PACKAGE_ROOT, REPO_ROOT, _assert_fit_split_train, _resolve
from omics_agent.reporting.tracking import log_benchmark_run
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import SplitName
from omics_agent.schemas.experiment import (
    ExperimentConfig,
    ModelParams,
    assert_task_matches_design,
    load_experiment,
)
from omics_agent.schemas.optimization import (
    FreezeManifest,
    OptimizationDecision,
    TrialRecord,
)
from omics_agent.splitting.split import assign_splits, write_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ALLOWED_OBJECTIVE_SPLITS = {SplitName.TRAIN, SplitName.VAL}


def _masked_objective(data: DataForModel, y_pred: np.ndarray, metric: str) -> float:
    """Masked val error. NaN targets are excluded, never imputed for scoring."""

    mask = data.forecast.y_mask
    if not mask.any():
        raise SchemaError(
            "Validation split has no observed targets to optimize on.",
            how_to_fix="Check missingness rates and the val split.",
        )
    diff = y_pred[mask] - data.forecast.y_true[mask]
    if metric == "mae":
        return float(np.mean(np.abs(diff)))
    return float(np.mean(diff**2))


def _assert_objective_data_is_test_free(*datasets: DataForModel) -> None:
    """Hard guard: the optimizer objective may only ever see train/val rows."""

    for data in datasets:
        if data.split not in _ALLOWED_OBJECTIVE_SPLITS:
            raise SchemaError(
                f"Tuning objective received split '{data.split.value}'.",
                how_to_fix="This is a bug guard: the optimizer must never see test rows.",
            )
        splits = set(data.bundle.observations.get("split", pd.Series(dtype=str)).astype(str))
        if "test" in splits:
            raise SchemaError(
                "Tuning objective received a bundle containing test rows.",
                how_to_fix="This is a bug guard: subset train/val before building objective data.",
            )


def run_tuning(
    experiment_path: Path,
    *,
    model_name: str,
    output_dir: Path | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Tune one model on val, then freeze the best config + checkpoint.

    Returns a summary dict with the decision path, freeze manifest path,
    and the MLflow run id.
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
    spec = next((item for item in experiment.models if item.name == model_name), None)
    if spec is None:
        raise SchemaError(
            f"Model '{model_name}' is not listed in experiment.models.",
            how_to_fix=f"Available: {[item.name for item in experiment.models]}.",
        )
    dest = output_dir or experiment.output_dir or Path("outputs") / experiment.experiment_id
    dest = dest.resolve()
    assert_tuning_allowed(dest, experiment.experiment_id)

    opt = experiment.optimization
    budget = int(n_trials) if n_trials is not None else opt.n_trials
    sampler_seed = opt.sampler_seed if opt.sampler_seed is not None else experiment.seed
    study_name = f"{experiment.experiment_id}::{model_name}"

    dest.mkdir(parents=True, exist_ok=True)
    bundle = load_local_bundle(manifest_path)
    splits = assign_splits(bundle, experiment.split, seed=experiment.seed)
    split_path = dest / "splits.parquet"
    write_splits(splits, split_path)
    labeled = bundle.with_split(splits[["experimental_unit_id", "split"]].drop_duplicates())
    processed = labeled.apply_assay_preprocessing(experiment.preprocessing.per_modality)
    _assert_fit_split_train(processed)

    # Test rows are never subset here. The objective closes over train/val only.
    train = processed.subset(SplitName.TRAIN)
    val = processed.subset(SplitName.VAL)
    train_data = prepare_split_data(
        full=processed, train=train, split_bundle=train, split=SplitName.TRAIN, task=experiment.task
    )
    val_data = prepare_split_data(
        full=processed, train=train, split_bundle=val, split=SplitName.VAL, task=experiment.task
    )
    _assert_objective_data_is_test_free(train_data, val_data)

    def objective(trial: optuna.Trial) -> float:
        params = {**spec.params, **suggest_params(model_name, trial)}
        trial.set_user_attr("model_params", _jsonable(params))
        plugin = get_model(model_name)
        if hasattr(plugin, "set_epoch_callback"):

            def report(epoch: int, val_mse: float) -> None:
                trial.report(val_mse, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            plugin.set_epoch_callback(report)
        plugin.fit(train_data, val_data, ModelParams(name=model_name, params=params))
        prediction = plugin.predict(val_data)
        return _masked_objective(val_data, prediction.y_pred, opt.objective_metric)

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=opt.pruner.n_startup_trials,
        n_warmup_steps=opt.pruner.n_warmup_reports,
    )
    study = optuna.create_study(
        study_name=study_name, direction=opt.direction, sampler=sampler, pruner=pruner
    )
    study.optimize(
        objective,
        n_trials=budget,
        catch=(TrainingDivergedError, OdeSolverError),
    )

    states = [trial.state for trial in study.trials]
    n_pruned = sum(1 for s in states if s == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for s in states if s == optuna.trial.TrialState.FAIL)
    n_complete = sum(1 for s in states if s == optuna.trial.TrialState.COMPLETE)
    if n_complete == 0:
        raise SchemaError(
            f"All {budget} trials pruned or failed; there is no best configuration.",
            how_to_fix="Raise the budget, or check the model's error messages.",
        )
    best = study.best_trial
    best_params: dict[str, Any] = dict(best.user_attrs["model_params"])

    # Refit the winner on train (val only steers early stopping) and freeze it.
    winner = get_model(model_name)
    winner.fit(train_data, val_data, ModelParams(name=model_name, params=best_params))
    frozen_dir = dest / "frozen" / model_name
    checkpoint_dir = frozen_dir / "checkpoint"
    winner.save(checkpoint_dir)
    prediction = winner.predict(val_data)
    val_report = evaluate_predictions(
        y_true=val_data.forecast.y_true,
        y_pred=prediction.y_pred,
        mask=val_data.forecast.y_mask,
        feature_names=val_data.forecast.feature_names,
        instance_ids=val_data.forecast.instance_ids,
        group_ids=val_data.forecast.group_ids,
        model_name=model_name,
        split=SplitName.VAL.value,
        target_modality=experiment.task.target_modality,
        primary_metric=experiment.task.primary_metric,
        bootstrap_replicates=experiment.evaluation.bootstrap_replicates,
        seed=experiment.seed,
    )

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
    decision = OptimizationDecision(
        experiment_id=experiment.experiment_id,
        model_name=model_name,
        study_name=study_name,
        sampler=opt.sampler,
        sampler_seed=sampler_seed,
        pruner=opt.pruner,
        n_trials_budget=budget,
        n_trials_completed=n_complete,
        n_trials_pruned=n_pruned,
        n_trials_failed=n_failed,
        objective_metric=opt.objective_metric,
        direction=opt.direction,
        best_trial_number=best.number,
        best_params=best_params,
        best_value=float(best.value if best.value is not None else float("nan")),
        val_primary_metric=experiment.task.primary_metric,
        val_primary_value=val_report.primary_value,
        decided_at=datetime.now(UTC).isoformat(),
        hashes=hashes,
        trials=[
            TrialRecord(
                number=trial.number,
                state=trial.state.name,
                value=None if trial.value is None else float(trial.value),
                params=dict(trial.user_attrs.get("model_params", trial.params)),
            )
            for trial in study.trials
        ],
    )
    decision_path = dest / "reports" / f"optimization_decision_{model_name}.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")

    frozen_config = _frozen_experiment_payload(experiment, model_name, best_params)
    frozen_config_path = frozen_dir / "frozen_experiment.yaml"
    frozen_config_path.write_text(yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8")
    manifest_doc = FreezeManifest(
        experiment_id=experiment.experiment_id,
        model_name=model_name,
        frozen_at=datetime.now(UTC).isoformat(),
        best_params=best_params,
        frozen_config_hash=hash_mapping(frozen_config),
        checkpoint_hashes={
            item.name: sha256_file(item) for item in sorted(checkpoint_dir.iterdir())
        },
        decision_hash=sha256_file(decision_path),
        split_file_hash=hash_dataframe(pd.read_parquet(split_path)),
        data_hash=hashes["data_hash"],
        evaluator_code_hash=hash_source_tree(PACKAGE_ROOT / "evaluation"),
        splitting_code_hash=hash_source_tree(PACKAGE_ROOT / "splitting"),
    )
    freeze_path = frozen_dir / "freeze_manifest.json"
    freeze_path.write_text(manifest_doc.model_dump_json(indent=2), encoding="utf-8")

    run_id = log_benchmark_run(
        tracking_uri=dest / "mlruns",
        experiment_name=experiment.experiment_id,
        run_name=f"{experiment.experiment_id}-tune-{model_name}",
        hashes=hashes,
        seed=experiment.seed,
        params={
            "study_name": study_name,
            "model": model_name,
            "n_trials_budget": budget,
            "sampler": opt.sampler,
            "sampler_seed": sampler_seed,
            "pruner": f"median(startup={opt.pruner.n_startup_trials},warmup={opt.pruner.n_warmup_reports})",
            "objective_metric": opt.objective_metric,
            "objective_split": "val",
            "test_labels_visible": False,
            "best_params": json.dumps(best_params, default=str),
        },
        reports=[val_report],
        artifacts=[decision_path, freeze_path, frozen_config_path],
        extra_metrics={
            "best_value": float(best.value if best.value is not None else float("nan")),
            "n_trials_completed": n_complete,
            "n_trials_pruned": n_pruned,
            "n_trials_failed": n_failed,
        },
    )
    return {
        "experiment_id": experiment.experiment_id,
        "model": model_name,
        "study_name": study_name,
        "n_trials": budget,
        "n_completed": n_complete,
        "n_pruned": n_pruned,
        "n_failed": n_failed,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best_params,
        "val_primary_value": val_report.primary_value,
        "decision_path": str(decision_path),
        "freeze_manifest": str(freeze_path),
        "checkpoint_dir": str(checkpoint_dir),
        "mlflow_run_id": run_id,
    }


def _frozen_experiment_payload(
    experiment: ExperimentConfig, model_name: str, best_params: dict[str, Any]
) -> dict[str, Any]:
    payload = experiment.model_dump(mode="json")
    payload["models"] = [{"name": model_name, "params": _jsonable(best_params)}]
    payload["dataset"] = str(experiment.dataset)
    if experiment.output_dir is not None:
        payload["output_dir"] = str(experiment.output_dir)
    return payload


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, default=str))
