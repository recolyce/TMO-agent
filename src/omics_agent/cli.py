"""Typer CLI for the deterministic milestone-1 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from omics_agent import __version__
from omics_agent.data_sources.ingest import load_ingest_manifest, run_ingest
from omics_agent.data_sources.synthetic import generate_synthetic_dataset
from omics_agent.errors import OmicsAgentError
from omics_agent.optimization import run_final_test, run_tuning
from omics_agent.pipeline import run_benchmark, run_preprocess, write_experiment_yaml
from omics_agent.priors.ablation import run_prior_ablation
from omics_agent.reporting.readiness import write_readiness_report
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import FileRole, ReviewStatus, SamplingDesign, SourceType, SplitName
from omics_agent.schemas.experiment import (
    EvaluationConfig,
    ExperimentConfig,
    ModelParams,
    SplitFractions,
    TaskConfig,
    TaskKind,
)
from omics_agent.schemas.experiment import SplitConfig as SplitCfg
from omics_agent.schemas.ingest import DownloadPolicy, IngestRequest

app = typer.Typer(
    name="omics-agent",
    help=(
        "Deterministic bulk temporal multi-omics pipeline. "
        "This is not an autonomous agent: it will not download untrusted code "
        "or guess sample mappings."
    ),
    no_args_is_help=True,
)
console = Console()


def _fail(exc: OmicsAgentError) -> None:
    console.print(f"[red]{exc.message}[/red]")
    console.print()
    console.print("[bold]How to fix[/bold]")
    console.print(exc.how_to_fix)
    raise typer.Exit(code=1)


@app.callback()
def _root() -> None:
    """omics-agent CLI."""


@app.command()
def doctor() -> None:
    """Check Python version and required scientific dependencies."""

    table = Table(title=f"omics-agent {__version__} doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    py_ok = sys.version_info >= (3, 11)
    table.add_row(
        "python>=3.11",
        "ok" if py_ok else "fail",
        sys.version.split()[0],
    )
    modules = [
        "pydantic",
        "typer",
        "yaml",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "anndata",
        "mudata",
        "mlflow",
        "statsmodels",
        "patsy",
        "pyarrow",
    ]
    all_ok = py_ok
    for name in modules:
        try:
            __import__(name)
            table.add_row(name, "ok", "importable")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            table.add_row(name, "fail", str(exc))
    try:
        import torch

        cuda = f"cuda={torch.cuda.is_available()}"
        table.add_row(
            "torch", "ok", f"{torch.__version__} ({cuda}); enables gru/ode_rnn/latent_ode"
        )
    except ImportError:
        table.add_row(
            "torch",
            "optional",
            "not installed; gru/ode_rnn/latent_ode need: uv sync --extra dev --extra torch",
        )
    from omics_agent.hashing import git_commit
    from omics_agent.priors.embeddings import list_embedding_models
    from omics_agent.priors.unimol import UNIMOL_DEFAULT_ROOT, UNIMOL_PINNED_COMMIT

    preferred = ", ".join(
        f"{item.name.value}{'*' if item.preferred else ''}" for item in list_embedding_models()
    )
    table.add_row("embedding models", "ok", f"{preferred} (*preferred)")
    tools = UNIMOL_DEFAULT_ROOT / "unimol_tools" / "unimol_tools"
    sha = git_commit(UNIMOL_DEFAULT_ROOT)
    if tools.is_dir():
        table.add_row(
            "unimol checkout",
            "ok",
            f"{UNIMOL_DEFAULT_ROOT} HEAD={sha or 'unknown'} pin={UNIMOL_PINNED_COMMIT[:12]}",
        )
    else:
        table.add_row(
            "unimol checkout",
            "optional",
            f"not found at {UNIMOL_DEFAULT_ROOT}; set priors.embedding.unimol_root",
        )
    try:
        import rdkit  # noqa: F401

        table.add_row("rdkit", "ok", "importable (needed for live Uni-Mol)")
    except ImportError:
        table.add_row(
            "rdkit",
            "optional",
            "not installed; live Uni-Mol needs rdkit. CI uses a mock extractor.",
        )
    console.print(table)
    if not all_ok:
        console.print(
            "\nHow to fix: run [bold]uv sync --extra dev[/bold] from the repository root."
        )
        raise typer.Exit(code=1)
    console.print("Environment is ready for the CPU pipeline (milestones 1–2).")


@app.command("validate-manifest")
def validate_manifest(
    path: Annotated[Path, typer.Argument(help="Path to dataset.yaml")],
) -> None:
    """Validate a dataset manifest. Does not guess missing mappings."""

    try:
        manifest = load_manifest(path)
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(f"dataset_id: {manifest.dataset_id}")
    console.print(f"design: {manifest.design.sampling_design.value}")
    console.print(f"pairing: {manifest.design.pairing_level.value}")
    console.print(f"review: {manifest.human_review.status.value}")
    if manifest.human_review.status is ReviewStatus.APPROVED:
        console.print("Manifest is schema-valid and approved for training.")
        return
    console.print("Manifest is schema-valid but [yellow]needs_review[/yellow].")
    for item in manifest.human_review.unresolved:
        console.print(f"- {item}")
    console.print(
        "\nHow to fix: a person must resolve those items, then set "
        "human_review.status: approved. The pipeline will not infer donor/time/biospecimen links."
    )
    raise typer.Exit(code=2)


@app.command()
def ingest(
    dest: Annotated[Path, typer.Option(help="Directory for ingest_manifest.yaml and raw/")],
    source: Annotated[
        str | None,
        typer.Option(help="geo|biostudies|pride|url|local. Inferred from accession prefix if omitted."),
    ] = None,
    accession: Annotated[str | None, typer.Option(help="GSE / PXD / E-MTAB accession")] = None,
    url: Annotated[str | None, typer.Option(help="Direct HTTPS URL to a processed matrix")] = None,
    local_path: Annotated[Path | None, typer.Option(help="Local processed matrix file")] = None,
    paper_doi: Annotated[str | None, typer.Option(help="Provenance only. Never fetched as instructions.")] = None,
    modality: Annotated[str | None, typer.Option(help="Optional modality label for a single-file ingest")] = None,
    role: Annotated[str | None, typer.Option(help="Optional file role: matrix|sample_sheet")] = None,
    dry_run: Annotated[bool, typer.Option(help="Resolve and print the plan; write nothing")] = False,
    resolve_only: Annotated[bool, typer.Option(help="Write the ingest manifest without downloading")] = False,
    max_bytes: Annotated[int, typer.Option(help="Refuse files larger than this many bytes")] = 2 * 1024 * 1024 * 1024,
) -> None:
    """Resolve a typed ingest manifest from GEO, BioStudies, PRIDE, HTTPS, or a local file.

    Paper text is not executed. Uncertain sample/time/pairing fields stay needs_review.
    FASTQ and raw mass-spec are refused.
    """

    try:
        request = IngestRequest(
            source=SourceType(source) if source else None,
            accession=accession,
            paper_doi=paper_doi,
            url=url,
            local_path=local_path,
            dest_dir=dest,
            modality=modality,
            role=FileRole(role) if role else None,
            dry_run=dry_run,
            resolve_only=resolve_only,
            policy=DownloadPolicy(max_bytes=max_bytes),
        )
        result = run_ingest(request)
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("data-readiness")
def data_readiness(
    manifest: Annotated[Path, typer.Argument(help="ingest_manifest.yaml or dataset.yaml")],
    output: Annotated[Path | None, typer.Option(help="HTML report path")] = None,
) -> None:
    """Write a red/yellow/green data-readiness report. Does not guess missing mappings."""

    try:
        if manifest.name == "dataset.yaml":
            raise OmicsAgentError(
                "data-readiness for a training dataset.yaml is not the ingest report.",
                how_to_fix="Pass the ingest_manifest.yaml produced by omics-agent ingest.",
            )
        ingest_manifest = load_ingest_manifest(manifest)
        dest = output or manifest.parent / "data_readiness_report.html"
        report = write_readiness_report(ingest_manifest, dest)
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(
        {
            "dataset_id": report.dataset_id,
            "blocking": report.blocking,
            "html": str(dest),
            "n_red": sum(1 for gate in report.gates if gate.level == "red"),
        }
    )


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Annotated[Path, typer.Option(help="Directory to write the dataset into")],
    design: Annotated[
        str,
        typer.Option(help="longitudinal | repeated_cross_sectional | both"),
    ] = "both",
    seed: Annotated[int, typer.Option(help="RNG seed")] = 20260901,
    dry_run: Annotated[bool, typer.Option(help="Print the plan without writing files")] = False,
) -> None:
    """Write the small bulk dual-omics ODE fixture (CPU, no network)."""

    designs = _parse_designs(design)
    for item in designs:
        dest = output_dir if design != "both" else output_dir / item.value
        try:
            plan = generate_synthetic_dataset(dest, design=item, seed=seed, dry_run=dry_run)
        except OmicsAgentError as exc:
            _fail(exc)
        console.print(plan)


@app.command("preprocess")
def preprocess(
    experiment: Annotated[Path, typer.Option(help="Path to experiment.yaml")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory")] = None,
    id_map: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Curated TSV/CSV: modality, source_id, target_id[, target_id_type]. "
                "One row per pair; one-to-many mappings are explicit rows."
            )
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Validate and print the plan only")] = False,
) -> None:
    """Approved matrices → MuData with raw/normalized/scaled layers, QC, feature map.

    The split is locked first; scalers fit on train rows only and record
    fit_split=train. Missing protein intensities are never filled with 0.
    Without --id-map, features keep their own IDs (nothing is guessed).
    """

    try:
        result = run_preprocess(experiment, output_dir=output_dir, id_map=id_map, dry_run=dry_run)
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("benchmark")
def benchmark(
    experiment: Annotated[Path, typer.Option(help="Path to experiment.yaml")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory")] = None,
    dry_run: Annotated[bool, typer.Option(help="Validate and print the plan only")] = False,
    unlock_test: Annotated[
        bool,
        typer.Option(help="Score the test split. Toy / synthetic only unless you mean it."),
    ] = False,
) -> None:
    """Fit LastValue, Ridge, and time spline; write a benchmark report."""

    try:
        result = run_benchmark(
            experiment,
            output_dir=output_dir,
            dry_run=dry_run,
            unlock_test=unlock_test if unlock_test else None,
        )
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("tune")
def tune(
    experiment: Annotated[Path, typer.Option(help="Path to experiment.yaml")],
    model: Annotated[str, typer.Option(help="One registered model name to tune")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory")] = None,
    n_trials: Annotated[
        int | None, typer.Option(help="Override optimization.n_trials (fixed budget)")
    ] = None,
) -> None:
    """Validation-only Optuna HPO, then freeze the best config + checkpoint.

    The objective never sees test labels. Fixed budget, sampler seed, study
    name, and median pruner are recorded in the OptimizationDecision. After
    a consumed test lock, tuning the same experiment_id is refused.
    """

    try:
        result = run_tuning(
            experiment, model_name=model, output_dir=output_dir, n_trials=n_trials
        )
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("unlock-test")
def unlock_test(
    experiment: Annotated[Path, typer.Option(help="Path to experiment.yaml")],
    model: Annotated[str, typer.Option(help="Frozen model to evaluate")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory")] = None,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help="Required. The final test runs exactly once per experiment_id.",
        ),
    ] = False,
) -> None:
    """Run the one-shot final test on a frozen checkpoint.

    Refuses if any frozen hash (config, checkpoint, decision, split,
    evaluator/splitting code) changed, and consumes the test lock before
    scoring. Afterwards, tuning this experiment_id is forbidden.
    """

    if not confirm:
        console.print(
            "[red]unlock-test is one-shot per experiment_id and consumes the test "
            "lock before scoring.[/red]"
        )
        console.print()
        console.print("[bold]How to fix[/bold]")
        console.print("Re-run with --confirm if you really want to spend the final test now.")
        raise typer.Exit(code=1)
    try:
        result = run_final_test(experiment, model_name=model, output_dir=output_dir)
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("ablate-priors")
def ablate_priors(
    experiment: Annotated[Path, typer.Option(help="Path to experiment.yaml")],
    model: Annotated[str, typer.Option(help="Dynamics model: gru | ode_rnn | latent_ode")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory")] = None,
    n_trials: Annotated[
        int | None,
        typer.Option(help="Shared HPO budget. 0 skips tuning and uses experiment.models params."),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            help="Override priors.embedding.name: unimol | synthetic_pathway_onehot | esm"
        ),
    ] = None,
    smiles_map: Annotated[
        Path | None,
        typer.Option(help="Feature→SMILES TSV for Uni-Mol (modality, feature_id, smiles)"),
    ] = None,
) -> None:
    """Five-arm prior ablation on one locked split and one evaluator.

    Arms: no_prior, graph_only, embedding_only, combined, random_graph.
    Combined uses Reactome pathway features + graph Laplacian + frozen
    embedding gate. STRING functional associations are never labelled
    physical or causal. Validation only; test labels stay hidden.
    Preferred embedding model is Uni-Mol (needs a SMILES map).
    """

    try:
        result = run_prior_ablation(
            experiment,
            model_name=model,
            output_dir=output_dir,
            n_trials=n_trials,
            embedding_model=embedding_model,
            smiles_map=smiles_map,
        )
    except OmicsAgentError as exc:
        _fail(exc)
    console.print(result)


@app.command("run-toy")
def run_toy(
    output_dir: Annotated[Path, typer.Option(help="Root output directory")] = Path("outputs/toy"),
    seed: Annotated[int, typer.Option()] = 20260901,
    dry_run: Annotated[bool, typer.Option()] = False,
) -> None:
    """One-command CPU path: synthetic data → split → baselines → report.

    Scores the synthetic holdout with an explicit disclaimer. This is not a
    production test unlock.
    """

    output_dir = output_dir.resolve()
    summaries = []
    for design, folder, experiment in (
        (SamplingDesign.LONGITUDINAL, "longitudinal", _longitudinal_experiment),
        (SamplingDesign.REPEATED_CROSS_SECTIONAL, "rcs", _rcs_experiment),
    ):
        data_dir = output_dir / folder / "data"
        run_dir = output_dir / folder / "run"
        if not dry_run:
            generate_synthetic_dataset(data_dir, design=design, seed=seed, dry_run=False)
            exp_path = output_dir / folder / "experiment.yaml"
            write_experiment_yaml(exp_path, experiment(data_dir / "dataset.yaml", run_dir, seed))
        else:
            exp_path = output_dir / folder / "experiment.yaml"
            generate_synthetic_dataset(data_dir, design=design, seed=seed, dry_run=True)
            summaries.append({"design": design.value, "dry_run": True, "data_dir": str(data_dir)})
            continue
        try:
            result = run_benchmark(exp_path, output_dir=run_dir, dry_run=False, unlock_test=True)
        except OmicsAgentError as exc:
            _fail(exc)
        summaries.append(result)
    console.print(summaries)


def _parse_designs(value: str) -> list[SamplingDesign]:
    if value == "both":
        return [SamplingDesign.LONGITUDINAL, SamplingDesign.REPEATED_CROSS_SECTIONAL]
    try:
        return [SamplingDesign(value)]
    except ValueError:
        raise typer.BadParameter(
            "design must be longitudinal, repeated_cross_sectional, or both"
        ) from None


def _longitudinal_experiment(dataset: Path, output_dir: Path, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        schema_version="1.0",
        experiment_id="toy_longitudinal_protein_forecast",
        dataset=dataset,
        seed=seed,
        task=TaskConfig(
            kind=TaskKind.SUBJECT_FORECAST,
            target_modality="protein",
            input_modalities=["rna", "protein"],
            primary_metric="protein_macro_pcc",
        ),
        split=SplitCfg(
            group_columns=["experimental_unit_id", "subject_id"],
            fractions=SplitFractions(train=0.6, val=0.2, test=0.2),
            block_experiment_batch=False,
            also_block=["biospecimen_id"],
        ),
        models=_baseline_models(),
        output_dir=output_dir,
        evaluation=EvaluationConfig(unlock_test=True, bootstrap_replicates=50),
    )


def _rcs_experiment(dataset: Path, output_dir: Path, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        schema_version="1.0",
        experiment_id="toy_rcs_protein_forecast",
        dataset=dataset,
        seed=seed,
        task=TaskConfig(
            kind=TaskKind.GROUP_TIME_FORECAST,
            target_modality="protein",
            input_modalities=["rna", "protein"],
            target_time_min=4.0,
            primary_metric="protein_macro_pcc",
        ),
        split=SplitCfg(
            group_columns=["batch"],
            fractions=SplitFractions(train=0.34, val=0.33, test=0.33),
            block_experiment_batch=True,
            also_block=["experimental_unit_id", "subject_id", "biospecimen_id"],
            assignment={
                "expA": SplitName.TRAIN,
                "expB": SplitName.VAL,
                "expC": SplitName.TEST,
            },
        ),
        models=_baseline_models(),
        output_dir=output_dir,
        evaluation=EvaluationConfig(unlock_test=True, bootstrap_replicates=50),
    )


def _baseline_models() -> list[ModelParams]:
    return [
        ModelParams(name="last_value", params={}),
        ModelParams(name="ridge", params={"alpha": 1.0}),
        ModelParams(name="time_spline", params={"spline_df": 3, "alpha": 1.0}),
    ]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
