"""Local MLflow recording of hashes, seed, and evaluator metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from omics_agent.errors import TrackingError
from omics_agent.schemas.evaluation import EvaluationReport


def log_benchmark_run(
    *,
    tracking_uri: Path,
    experiment_name: str,
    run_name: str,
    hashes: dict[str, str],
    seed: int,
    params: dict[str, Any],
    reports: list[EvaluationReport],
    artifacts: list[Path],
) -> str:
    """Create one MLflow run and return the run id.

    NaN metrics are skipped (MLflow cannot store them) and the skip is
    recorded as a tag so absence is visible.
    """

    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    skipped: list[str] = []
    try:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("seed", seed)
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in hashes.items():
                mlflow.set_tag(key, value)
            mlflow.set_tag("fit_split", "train")
            for report in reports:
                prefix = f"{report.model_name}.{report.split}"
                mlflow.log_metric(f"{prefix}.coverage", report.coverage)
                mlflow.log_metric(f"{prefix}.n_instances", report.n_instances)
                if report.primary_value is not None:
                    mlflow.log_metric(f"{prefix}.primary", report.primary_value)
                for scalar in report.scalars:
                    if scalar.value is None or not _finite(scalar.value):
                        skipped.append(f"{prefix}.{scalar.name}")
                        continue
                    mlflow.log_metric(f"{prefix}.{scalar.name}", scalar.value)
                    mlflow.log_metric(f"{prefix}.{scalar.name}.n_valid", scalar.n_valid)
            if skipped:
                mlflow.set_tag("metrics_skipped_na", ",".join(skipped[:40]))
            for artifact in artifacts:
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact))
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        raise TrackingError(
            f"MLflow logging failed: {exc}",
            how_to_fix="Check that the output directory is writable and tracking_uri is a local path.",
        ) from exc


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
