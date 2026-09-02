"""Five-arm prior ablation on one locked split, one evaluator, one HPO budget.

Arms: no_prior, graph_only, embedding_only, combined, random_graph.
The objective never sees test rows. Combined turns on all three priors
(Reactome pathway features, graph Laplacian, frozen embedding gate).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.errors import SchemaError
from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.hashing import collect_run_hashes, hash_dataframe
from omics_agent.models import get_model
from omics_agent.models.tasks import DataForModel, prepare_split_data
from omics_agent.optimization.lock import assert_tuning_allowed
from omics_agent.pipeline import PACKAGE_ROOT, REPO_ROOT, _assert_fit_split_train, _resolve
from omics_agent.priors.embeddings import apply_embedding_model
from omics_agent.priors.runtime import align_prior
from omics_agent.priors.synthetic import build_synthetic_prior_bundle, write_synthetic_prior_bundle
from omics_agent.priors.unimol import UniMolReprFn
from omics_agent.reporting.tracking import log_benchmark_run
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import EmbeddingModelName, PriorAblation, SplitName
from omics_agent.schemas.experiment import (
    ModelParams,
    assert_task_matches_design,
    load_experiment,
)
from omics_agent.schemas.priors import (
    PriorAblationConfig,
    PriorBundle,
    flags_for,
    load_prior_bundle,
)
from omics_agent.splitting.split import assign_splits, write_splits

_DYNAMICS = {"gru", "ode_rnn", "latent_ode"}
_REQUIRED_ARMS = (
    PriorAblation.NO_PRIOR,
    PriorAblation.GRAPH_ONLY,
    PriorAblation.EMBEDDING_ONLY,
    PriorAblation.COMBINED,
    PriorAblation.RANDOM_GRAPH,
)


def run_prior_ablation(
    experiment_path: Path,
    *,
    model_name: str,
    output_dir: Path | None = None,
    n_trials: int | None = None,
    embedding_model: str | None = None,
    smiles_map: Path | None = None,
    unimol_repr_fn: UniMolReprFn | None = None,
) -> dict[str, Any]:
    """Fit every configured arm on the same split and write the comparison table."""

    experiment_path = experiment_path.resolve()
    experiment = load_experiment(experiment_path)
    if model_name not in _DYNAMICS:
        raise SchemaError(
            f"Prior ablations run on dynamics models; '{model_name}' has no embedding gate.",
            how_to_fix="Pass --model gru (or ode_rnn / latent_ode).",
        )
    spec = next((item for item in experiment.models if item.name == model_name), None)
    if spec is None:
        raise SchemaError(
            f"Model '{model_name}' is not listed in experiment.models.",
            how_to_fix=f"Available: {[item.name for item in experiment.models]}.",
        )
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
    assert_tuning_allowed(dest, experiment.experiment_id)

    dest.mkdir(parents=True, exist_ok=True)
    bundle = load_local_bundle(manifest_path)
    splits = assign_splits(bundle, experiment.split, seed=experiment.seed)
    split_path = dest / "splits.parquet"
    write_splits(splits, split_path)
    labeled = bundle.with_split(splits[["experimental_unit_id", "split"]].drop_duplicates())
    processed = labeled.apply_assay_preprocessing(experiment.preprocessing.per_modality)
    _assert_fit_split_train(processed)
    train = processed.subset(SplitName.TRAIN)
    val = processed.subset(SplitName.VAL)
    train_data = prepare_split_data(
        full=processed, train=train, split_bundle=train, split=SplitName.TRAIN, task=experiment.task
    )
    val_data = prepare_split_data(
        full=processed, train=train, split_bundle=val, split=SplitName.VAL, task=experiment.task
    )

    prior_cfg = _with_embedding_overrides(
        experiment.priors, embedding_model=embedding_model, smiles_map=smiles_map
    )
    bundle_path = None
    if prior_cfg.bundle is not None:
        bundle_path = (
            prior_cfg.bundle
            if prior_cfg.bundle.is_absolute()
            else (experiment_path.parent / prior_cfg.bundle).resolve()
        )
    prior_bundle = _load_or_build_bundle(
        bundle_path,
        dest,
        train_data,
        prior_cfg,
        experiment_dir=experiment_path.parent,
        unimol_repr_fn=unimol_repr_fn,
    )
    arms = list(prior_cfg.configs) or list(_REQUIRED_ARMS)
    seeds = list(prior_cfg.seeds)
    budget = int(n_trials) if n_trials is not None else experiment.optimization.n_trials
    shared_params = dict(spec.params)
    if prior_cfg.share_hpo and budget > 0:
        shared_params = _tune_no_prior(
            model_name,
            train_data,
            val_data,
            spec.params,
            budget,
            experiment.optimization,
            experiment.seed,
        )
        shared_params = {**spec.params, **shared_params}

    hashes = collect_run_hashes(
        package_root=PACKAGE_ROOT,
        repo_root=REPO_ROOT,
        data_frames={"observations": processed.observations, "samples": processed.sample_sheet},
        split_frame=splits,
        config_payload=experiment.model_dump(mode="json"),
        seed=experiment.seed,
    )
    hashes["split_file_hash"] = hash_dataframe(splits)
    hashes["prior_bundle_hash"] = prior_bundle.content_hash()
    hashes["prior_bundle_version"] = prior_bundle.bundle_version

    runs: list[dict[str, Any]] = []
    reports = []
    for seed in seeds:
        for arm in arms:
            runtime = align_prior(
                prior_bundle,
                feature_names=train.feature_names,
                modalities=_modalities(train_data),
                ablation=arm,
                random_graph_seed=seed + 17,
            )
            params = {
                **shared_params,
                "seed": seed,
                "graph_weight": prior_cfg.graph_weight,
                "embedding_proj_dim": prior_cfg.embedding_proj_dim,
            }
            plugin = get_model(model_name)
            setter = getattr(plugin, "set_prior", None)
            if setter is None:
                raise SchemaError(
                    f"Model '{model_name}' cannot attach a PriorRuntime.",
                    how_to_fix="Use gru / ode_rnn / latent_ode.",
                )
            setter(runtime)
            started = time.perf_counter()
            fit = plugin.fit(train_data, val_data, ModelParams(name=model_name, params=params))
            elapsed = time.perf_counter() - started
            pred = plugin.predict(val_data)
            report = evaluate_predictions(
                y_true=val_data.forecast.y_true,
                y_pred=pred.y_pred,
                mask=val_data.forecast.y_mask,
                feature_names=val_data.forecast.feature_names,
                instance_ids=val_data.forecast.instance_ids,
                group_ids=val_data.forecast.group_ids,
                model_name=f"{model_name}:{arm.value}",
                split=SplitName.VAL.value,
                target_modality=experiment.task.target_modality,
                primary_metric=experiment.task.primary_metric,
                bootstrap_replicates=experiment.evaluation.bootstrap_replicates,
                seed=seed,
            )
            reports.append(report)
            runs.append(
                {
                    "ablation": arm.value,
                    "seed": seed,
                    "n_parameters": fit.n_parameters,
                    "seconds": elapsed,
                    "mse": _scalar(report, "mse"),
                    "mae": _scalar(report, "mae"),
                    "pcc_macro": _scalar(report, "pcc_macro"),
                    "primary_metric": experiment.task.primary_metric,
                    "primary_value": report.primary_value,
                    "flags": flags_for(arm).model_dump(mode="json"),
                    "prior_bundle_hash": runtime.bundle_hash,
                    "n_unmapped_edges": runtime.n_unmapped_edges,
                }
            )
            arm_dir = dest / "priors" / arm.value / f"seed_{seed}"
            plugin.save(arm_dir)

    table = _summarize(runs, experiment.task.primary_metric)
    table_path = dest / "reports" / "prior_ablation.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment.experiment_id,
        "model": model_name,
        "objective_split": "val",
        "test_labels_visible": False,
        "hpo_budget": budget,
        "share_hpo": prior_cfg.share_hpo,
        "shared_params": shared_params,
        "prior_bundle_id": prior_bundle.bundle_id,
        "prior_bundle_version": prior_bundle.bundle_version,
        "prior_bundle_hash": prior_bundle.content_hash(),
        "embedding_model": prior_cfg.embedding.name.value,
        "embedding_spec": (
            None
            if prior_bundle.embedding_spec is None
            else prior_bundle.embedding_spec.model_dump(mode="json")
        ),
        "hashes": hashes,
        "runs": runs,
        "table": table,
        "notes": _notes(table, prior_bundle),
    }
    table_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path = table_path.with_suffix(".md")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    run_id = log_benchmark_run(
        tracking_uri=dest / "mlruns",
        experiment_name=experiment.experiment_id,
        run_name=f"{experiment.experiment_id}-prior-ablation-{model_name}",
        hashes=hashes,
        seed=experiment.seed,
        params={
            "model": model_name,
            "hpo_budget": budget,
            "share_hpo": prior_cfg.share_hpo,
            "n_seeds": len(seeds),
            "objective_split": "val",
            "test_labels_visible": False,
            "prior_bundle_version": prior_bundle.bundle_version,
        },
        reports=reports,
        artifacts=[table_path, md_path],
    )
    return {
        "experiment_id": experiment.experiment_id,
        "model": model_name,
        "report_json": str(table_path),
        "report_md": str(md_path),
        "mlflow_run_id": run_id,
        "table": table,
        "n_runs": len(runs),
    }


def _modalities(data: DataForModel) -> list[str]:
    seen: list[str] = []
    for name in data.forecast.input_feature_names:
        modality = name.split(":", 1)[0]
        if modality not in seen:
            seen.append(modality)
    return seen


def _with_embedding_overrides(
    prior_cfg: PriorAblationConfig,
    *,
    embedding_model: str | None,
    smiles_map: Path | None,
) -> PriorAblationConfig:
    embedding = prior_cfg.embedding
    if embedding_model is not None:
        try:
            name = EmbeddingModelName(embedding_model)
        except ValueError as exc:
            raise SchemaError(
                f"Unknown embedding model '{embedding_model}'.",
                how_to_fix=f"Choose one of: {[item.value for item in EmbeddingModelName]}.",
            ) from exc
        embedding = embedding.model_copy(update={"name": name})
    if smiles_map is not None:
        embedding = embedding.model_copy(update={"smiles_map": smiles_map})
    if embedding is prior_cfg.embedding:
        return prior_cfg
    return prior_cfg.model_copy(update={"embedding": embedding})


def _load_or_build_bundle(
    path: Path | None,
    dest: Path,
    train: DataForModel,
    prior_cfg: PriorAblationConfig,
    *,
    experiment_dir: Path,
    unimol_repr_fn: UniMolReprFn | None,
) -> PriorBundle:
    names = dict(train.bundle.feature_names)
    if path is not None:
        built = load_prior_bundle(path if path.is_absolute() else dest / path)
        overlay = (
            prior_cfg.embedding.smiles_map is not None
            or prior_cfg.embedding.name is EmbeddingModelName.ESM
        )
    else:
        built = build_synthetic_prior_bundle(
            rna_features=names.get("rna"),
            protein_features=names.get("protein"),
        )
        overlay = prior_cfg.embedding.name is not EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT
    if overlay:
        built = apply_embedding_model(
            built,
            prior_cfg.embedding,
            experiment_dir=experiment_dir,
            repr_fn=unimol_repr_fn,
        )
    write_synthetic_prior_bundle(dest / "priors" / "bundle.yaml", built)
    return built


def _tune_no_prior(
    model_name: str,
    train: DataForModel,
    val: DataForModel,
    base_params: dict[str, Any],
    n_trials: int,
    opt: Any,
    seed: int,
) -> dict[str, Any]:
    import optuna

    from omics_agent.errors import OdeSolverError, TrainingDivergedError
    from omics_agent.optimization.search_space import suggest_params
    from omics_agent.optimization.tuner import (
        _assert_objective_data_is_test_free,
        _masked_objective,
    )

    _assert_objective_data_is_test_free(train, val)
    sampler_seed = opt.sampler_seed if opt.sampler_seed is not None else seed

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_params(model_name, trial)}
        trial.set_user_attr("model_params", params)
        plugin = get_model(model_name)
        setter = getattr(plugin, "set_prior", None)
        if setter is not None:
            setter(None)
        if hasattr(plugin, "set_epoch_callback"):

            def report(epoch: int, val_mse: float) -> None:
                trial.report(val_mse, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            plugin.set_epoch_callback(report)
        plugin.fit(train, val, ModelParams(name=model_name, params=params))
        pred = plugin.predict(val)
        return _masked_objective(val, pred.y_pred, opt.objective_metric)

    study = optuna.create_study(
        study_name=f"prior_ablation::{model_name}::no_prior",
        direction=opt.direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=opt.pruner.n_startup_trials,
            n_warmup_steps=opt.pruner.n_warmup_reports,
        ),
    )
    study.optimize(objective, n_trials=n_trials, catch=(TrainingDivergedError, OdeSolverError))
    if study.best_trial is None or "model_params" not in study.best_trial.user_attrs:
        return dict(base_params)
    return dict(study.best_trial.user_attrs["model_params"])


def _scalar(report: Any, name: str) -> float | None:
    for item in report.scalars:
        if item.name == name:
            return item.value
    return None


def _mean_ci(values: Sequence[float | None]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "low": mean, "high": mean, "n": 1}
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return {"mean": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se, "n": int(arr.size)}


def _summarize(runs: list[dict[str, Any]], primary: str) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_arm.setdefault(row["ablation"], []).append(row)
    no_prior = {int(r["seed"]): r for r in by_arm.get(PriorAblation.NO_PRIOR.value, [])}
    rows = []
    for arm, items in by_arm.items():
        deltas_mse = []
        deltas_pcc = []
        for item in items:
            base = no_prior.get(int(item["seed"]))
            if base is None:
                continue
            if item["mse"] is not None and base["mse"] is not None:
                deltas_mse.append(float(item["mse"]) - float(base["mse"]))
            if item["pcc_macro"] is not None and base["pcc_macro"] is not None:
                deltas_pcc.append(float(item["pcc_macro"]) - float(base["pcc_macro"]))
        rows.append(
            {
                "ablation": arm,
                "n_seeds": len(items),
                "mse": _mean_ci([r["mse"] for r in items]),
                "mae": _mean_ci([r["mae"] for r in items]),
                "pcc_macro": _mean_ci([r["pcc_macro"] for r in items]),
                "primary": {"name": primary, **_mean_ci([r["primary_value"] for r in items])},
                "n_parameters": _mean_ci([r["n_parameters"] for r in items]),
                "seconds": _mean_ci([r["seconds"] for r in items]),
                "delta_mse_vs_no_prior": _mean_ci(deltas_mse),
                "delta_pcc_vs_no_prior": _mean_ci(deltas_pcc),
            }
        )
    order = {arm.value: i for i, arm in enumerate(_REQUIRED_ARMS)}
    rows.sort(key=lambda r: order.get(str(r["ablation"]), 99))
    return rows


def _notes(table: list[dict[str, Any]], bundle: PriorBundle) -> list[str]:
    notes = [
        "Attribution / prior edges are not causal effects.",
        "STRING-style edges in this bundle are functional_association, not physical PPI.",
        "Scores are validation-only. Test labels were not used.",
        f"PriorBundle {bundle.bundle_id}@{bundle.bundle_version}.",
    ]
    spec = bundle.embedding_spec
    if spec is not None and spec.model_name == EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT.value:
        notes.append("Embeddings are the synthetic_pathway_onehot fixture, not Uni-Mol/ESM.")
    elif spec is not None:
        notes.append(
            f"Embeddings: {spec.model_name} {spec.model_version} (layer={spec.extraction_layer})."
        )
    by = {row["ablation"]: row for row in table}
    graph = by.get(PriorAblation.GRAPH_ONLY.value)
    rand = by.get(PriorAblation.RANDOM_GRAPH.value)
    if graph and rand:
        g = graph["mse"]["mean"]
        r = rand["mse"]["mean"]
        if g is not None and r is not None and abs(g - r) < 0.02:
            notes.append(
                "graph_only and random_graph val MSE are within 0.02; do not claim "
                "the prior graph conferred a biological gain."
            )
    return notes


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Prior ablation: {payload['experiment_id']} / {payload['model']}",
        "",
        "Priors are not causes. STRING functional association is not physical PPI.",
        "",
        f"objective_split=`val`; test_labels_visible=`false`; "
        f"HPO budget=`{payload['hpo_budget']}` (share_hpo={payload['share_hpo']}).",
        "",
        "| arm | MSE mean [95% CI] | ΔMSE vs no_prior | PCC mean [95% CI] | "
        "ΔPCC vs no_prior | params | seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["table"]:
        lines.append(
            "| {ablation} | {mse} | {dmse} | {pcc} | {dpcc} | {nparam} | {sec} |".format(
                ablation=row["ablation"],
                mse=_fmt_ci(row["mse"]),
                dmse=_fmt_ci(row["delta_mse_vs_no_prior"]),
                pcc=_fmt_ci(row["pcc_macro"]),
                dpcc=_fmt_ci(row["delta_pcc_vs_no_prior"]),
                nparam=_fmt_ci(row["n_parameters"], digits=0),
                sec=_fmt_ci(row["seconds"], digits=2),
            )
        )
    lines.extend(["", "## Notes", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _fmt_ci(cell: dict[str, Any], digits: int = 3) -> str:
    if cell.get("mean") is None:
        return "NA"
    mean = cell["mean"]
    low, high = cell.get("low"), cell.get("high")
    if digits == 0:
        return f"{mean:.0f}"
    if low is None or high is None or cell.get("n", 0) <= 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
