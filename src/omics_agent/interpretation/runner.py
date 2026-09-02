"""Run attribution on a frozen dynamics checkpoint. Validation only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.errors import InterpretationError, SchemaError
from omics_agent.hashing import collect_run_hashes, hash_dataframe
from omics_agent.interpretation.ig import integrated_gradients
from omics_agent.interpretation.perturb import group_feature_ablation, stratified_permutation
from omics_agent.interpretation.stability import assemble_candidates, select_stable
from omics_agent.models import get_model
from omics_agent.models.tasks import DataForModel, prepare_split_data
from omics_agent.pipeline import PACKAGE_ROOT, REPO_ROOT, _assert_fit_split_train, _resolve
from omics_agent.reporting.tracking import log_benchmark_run
from omics_agent.schemas.enums import SplitName
from omics_agent.schemas.experiment import assert_task_matches_design, load_experiment
from omics_agent.schemas.interpretation import (
    ABSENCE_OF_EVIDENCE,
    CLAIM_KIND,
    HYPOTHESIS_CAVEAT,
    StabilityTable,
)
from omics_agent.schemas.priors import load_prior_bundle
from omics_agent.splitting.split import assign_splits, write_splits


def run_explanation(
    experiment_path: Path,
    *,
    model_name: str,
    output_dir: Path | None = None,
    checkpoint: Path | None = None,
    with_literature: bool = False,
    transport: Any = None,
    plugin: Any = None,
    train_data: DataForModel | None = None,
    val_data: DataForModel | None = None,
) -> dict[str, Any]:
    """IG + group ablation + stratified permutation on the frozen model.

    The explain split is literally val. Test rows are never subset here.
    """

    experiment_path = experiment_path.resolve()
    experiment = load_experiment(experiment_path)
    dest = output_dir or experiment.output_dir or Path("outputs") / experiment.experiment_id
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if train_data is None or val_data is None:
        manifest_path = _resolve(experiment_path, experiment.dataset)
        from omics_agent.schemas.dataset import load_manifest

        manifest = load_manifest(manifest_path)
        manifest.require_approved_for_training()
        assert_task_matches_design(
            task=experiment.task,
            sampling_design=manifest.design.sampling_design,
            pairing_level=manifest.design.pairing_level,
            modalities=list(manifest.modalities),
        )
        bundle = load_local_bundle(manifest_path)
        split_path = dest / "splits.parquet"
        if split_path.is_file():
            import pandas as pd

            splits = pd.read_parquet(split_path)
        else:
            splits = assign_splits(bundle, experiment.split, seed=experiment.seed)
            write_splits(splits, split_path)
        labeled = bundle.with_split(splits[["experimental_unit_id", "split"]].drop_duplicates())
        processed = labeled.apply_assay_preprocessing(experiment.preprocessing.per_modality)
        _assert_fit_split_train(processed)
        train = processed.subset(SplitName.TRAIN)
        val = processed.subset(SplitName.VAL)
        train_data = prepare_split_data(
            full=processed,
            train=train,
            split_bundle=train,
            split=SplitName.TRAIN,
            task=experiment.task,
        )
        val_data = prepare_split_data(
            full=processed,
            train=train,
            split_bundle=val,
            split=SplitName.VAL,
            task=experiment.task,
        )
        hashes = collect_run_hashes(
            package_root=PACKAGE_ROOT,
            repo_root=REPO_ROOT,
            data_frames={"observations": processed.observations, "samples": processed.sample_sheet},
            split_frame=splits,
            config_payload=experiment.model_dump(mode="json"),
            seed=experiment.seed,
        )
        hashes["split_file_hash"] = hash_dataframe(splits)
    else:
        hashes = {"note": "in-memory explanation; caller supplied train/val"}

    if plugin is None:
        plugin = get_model(model_name)
        ckpt = checkpoint or dest / "frozen" / model_name / "checkpoint"
        if not (ckpt / "card.json").is_file():
            raise InterpretationError(
                f"No frozen checkpoint at {ckpt}.",
                how_to_fix=(
                    "Run omics-agent tune ... or pass --checkpoint to a directory "
                    "written by plugin.save() (card.json + model.pt)."
                ),
            )
        plugin.load(ckpt)

    cfg = experiment.interpretation
    ig = integrated_gradients(
        plugin,
        val_data,
        train=train_data,
        n_steps=cfg.n_ig_steps,
        baselines=list(cfg.baselines),
        target_modality=experiment.task.target_modality,
    )
    ablation = group_feature_ablation(
        plugin, val_data, sources=ig["sources"], n_targets=len(ig["targets"])
    )
    permutation = stratified_permutation(
        plugin,
        val_data,
        sources=ig["sources"],
        n_targets=len(ig["targets"]),
        n_seeds=cfg.n_seeds,
        seed=experiment.seed,
    )
    prior_bundle = _load_prior(dest, experiment_path, experiment.priors.bundle)
    embedding_used = bool(getattr(plugin, "_prior_card", {}).get("use_embedding_gate"))
    table = assemble_candidates(
        experiment_id=experiment.experiment_id,
        model_name=model_name,
        attr=ig["attr"],
        sources=ig["sources"],
        targets=ig["targets"],
        group_ids=ig["group_ids"],
        ablation=ablation,
        permutation=permutation,
        config=cfg,
        seed=experiment.seed,
        bundle=prior_bundle,
        embedding_used=embedding_used,
    )
    reports_dir = dest / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "candidates.json"
    json_path.write_text(table.model_dump_json(indent=2), encoding="utf-8")
    md_path = reports_dir / "candidates.md"
    md_path.write_text(_markdown(table), encoding="utf-8")

    literature_paths: dict[str, str] = {}
    if with_literature:
        from omics_agent.literature.check import run_literature_check

        stable = select_stable(table, cfg.literature.top_n)
        lit = run_literature_check(
            stable,
            experiment_id=experiment.experiment_id,
            config=cfg.literature,
            transport=transport,
        )
        lit_json = reports_dir / "literature.json"
        lit_md = reports_dir / "literature.md"
        lit_json.write_text(lit.model_dump_json(indent=2), encoding="utf-8")
        lit_md.write_text(_literature_markdown(lit), encoding="utf-8")
        literature_paths = {"literature_json": str(lit_json), "literature_md": str(lit_md)}

    try:
        run_id = log_benchmark_run(
            tracking_uri=dest / "mlruns",
            experiment_name=experiment.experiment_id,
            run_name=f"{experiment.experiment_id}-explain-{model_name}",
            hashes=hashes if "split_file_hash" in hashes else {},
            seed=experiment.seed,
            params={
                "model": model_name,
                "objective_split": "val",
                "test_labels_visible": False,
                "claim_kind": CLAIM_KIND,
                "ig_engine": ig["engine"],
            },
            reports=[],
            artifacts=[json_path, md_path],
        )
    except Exception:  # noqa: BLE001
        run_id = None
    return {
        "experiment_id": experiment.experiment_id,
        "model": model_name,
        "candidates_json": str(json_path),
        "candidates_md": str(md_path),
        "n_candidates": len(table.rows),
        "n_stable": sum(1 for row in table.rows if row.passed_stability),
        "objective_split": "val",
        "test_labels_visible": False,
        "claim_kind": CLAIM_KIND,
        "ig_engine": ig["engine"],
        "mlflow_run_id": run_id,
        "table": table,
        **literature_paths,
    }


def _load_prior(dest: Path, experiment_path: Path, configured: Path | None) -> Any:
    candidates = []
    if configured is not None:
        candidates.append(
            configured if configured.is_absolute() else (experiment_path.parent / configured)
        )
    candidates.append(dest / "priors" / "bundle.yaml")
    for path in candidates:
        if path.is_file():
            return load_prior_bundle(path)
    return None


def _markdown(table: StabilityTable) -> str:
    lines = [
        f"# Attribution hypotheses: {table.experiment_id} / {table.model_name}",
        "",
        f"claim_kind=`{CLAIM_KIND}`. objective_split=`val`. test_labels_visible=`false`.",
        "",
        HYPOTHESIS_CAVEAT,
        "",
        f"Baselines={table.n_baselines}; bootstrap={table.n_bootstrap}; "
        f"seeds={table.n_seeds}; folds={table.n_folds}.",
        "",
        "| candidate | attr | stability | sign | sel.freq | ablation_delta | "
        "perm_delta | prior_edge_used | embedding_supported | de_novo | passed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.rows:
        lines.append(
            f"| {row.candidate_id} | {row.mean_attribution:.3f} | {row.stability:.2f} | {row.sign_consistency:.2f} | {row.selection_frequency:.2f} | {row.ablation_delta:.3f} | "
            f"{row.permutation_delta:.3f} | {str(row.prior_edge_used).lower()} | {str(row.embedding_supported).lower()} | {str(row.de_novo_model_edge).lower()} | {str(row.passed_stability).lower()} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in table.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _literature_markdown(table: Any) -> str:
    from omics_agent.schemas.literature import LiteratureTable

    assert isinstance(table, LiteratureTable)
    lines = [
        f"# Literature hypotheses: {table.experiment_id}",
        "",
        f"claim_kind=`{table.claim_kind}`.",
        f"Level N is written as: {ABSENCE_OF_EVIDENCE}.",
        "This table does not claim novelty or causality.",
        "",
        "| candidate | source | pmid | doi | stance | level | authentic | reviewer_status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in table.records:
        lines.append(
            f"| {rec.candidate_id} | {rec.source_name} | {rec.pmid or ''} | {rec.doi or ''} | "
            f"{rec.stance.value} | {rec.evidence_level.value} | "
            f"pmid={rec.pmid_authentic}/doi={rec.doi_authentic} | {rec.reviewer_status.value} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in table.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def load_candidates(path: Path) -> StabilityTable:
    if not path.is_file():
        raise SchemaError(
            f"Candidate table not found: {path}",
            how_to_fix="Run omics-agent explain first.",
        )
    return StabilityTable.model_validate_json(path.read_text(encoding="utf-8"))
