"""One-shot test lock and frozen-artifact integrity checks.

Only the explicit ``unlock-test`` command runs the final test, exactly once
per experiment_id. Before scoring, every hash in the freeze manifest is
recomputed: a modified split file, checkpoint, frozen config, decision
document, or evaluator/splitting source tree is a hard refusal. The lock is
written *before* evaluation (fail-closed), and once consumed it also blocks
further tuning under the same experiment_id.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.errors import ArtifactIntegrityError, SchemaError, TestLockError
from omics_agent.evaluation.evaluator import evaluate_predictions
from omics_agent.hashing import (
    hash_dataframe,
    hash_mapping,
    hash_source_tree,
    hash_yaml_file,
    sha256_file,
)
from omics_agent.models import get_model
from omics_agent.models.tasks import prepare_split_data
from omics_agent.pipeline import (
    PACKAGE_ROOT,
    _assert_feature_maps_trainable,
    _assert_fit_split_train,
    _resolve,
)
from omics_agent.reporting.benchmark import write_benchmark_report
from omics_agent.reporting.tracking import log_benchmark_run
from omics_agent.schemas.enums import SplitName
from omics_agent.schemas.experiment import load_experiment
from omics_agent.schemas.optimization import FreezeManifest, TestLockState
from omics_agent.splitting.guard import assert_no_group_leakage

LOCK_FILENAME = "test_lock.json"


def _lock_path(dest: Path) -> Path:
    return dest / LOCK_FILENAME


def read_lock(dest: Path) -> TestLockState | None:
    path = _lock_path(dest)
    if not path.is_file():
        return None
    return TestLockState.model_validate_json(path.read_text(encoding="utf-8"))


def _write_lock(dest: Path, state: TestLockState) -> None:
    _lock_path(dest).write_text(state.model_dump_json(indent=2), encoding="utf-8")


def assert_tuning_allowed(dest: Path, experiment_id: str) -> None:
    """After the one-shot test, further tuning of this experiment_id is refused."""

    lock = read_lock(dest)
    if lock is not None and lock.consumed and lock.experiment_id == experiment_id:
        raise TestLockError(
            f"experiment_id '{experiment_id}' already ran its one-shot final test "
            f"({lock.consumed_at}). Continuing to tune it would turn the test split "
            "into a validation split.",
            how_to_fix=(
                "Start a new experiment_id (new hypothesis, new budget). The frozen "
                "result of this one stands as reported."
            ),
        )


def _integrity(condition: bool, what: str) -> None:
    if not condition:
        raise ArtifactIntegrityError(
            f"Frozen-artifact check failed: {what} does not match the freeze manifest.",
            how_to_fix=(
                "Something changed after freezing (split, checkpoint, config, decision, "
                "or evaluator/splitting code). Re-run tuning to freeze a new artifact, "
                "or restore the original files. The final test will not run on "
                "modified inputs."
            ),
        )


def run_final_test(
    experiment_path: Path,
    *,
    model_name: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen artifact, then score the test split exactly once."""

    experiment_path = experiment_path.resolve()
    live_experiment = load_experiment(experiment_path)
    experiment = live_experiment
    manifest_path = _resolve(experiment_path, experiment.dataset)
    dest = output_dir or experiment.output_dir or Path("outputs") / experiment.experiment_id
    dest = dest.resolve()
    _assert_feature_maps_trainable(dest)

    lock = read_lock(dest)
    if lock is not None and lock.consumed:
        raise TestLockError(
            f"The final test for experiment_id '{lock.experiment_id}' was already "
            f"consumed at {lock.consumed_at}.",
            how_to_fix="One test evaluation per experiment_id. Start a new experiment_id.",
        )

    frozen_dir = dest / "frozen" / model_name
    freeze_path = frozen_dir / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise SchemaError(
            f"No freeze manifest under {frozen_dir}.",
            how_to_fix="Run: omics-agent tune --experiment ... --model "
            f"{model_name} first. Only frozen models can be tested.",
        )
    manifest_doc = FreezeManifest.model_validate_json(freeze_path.read_text(encoding="utf-8"))
    if manifest_doc.experiment_id != experiment.experiment_id:
        raise ArtifactIntegrityError(
            f"Freeze manifest belongs to '{manifest_doc.experiment_id}', not "
            f"'{experiment.experiment_id}'.",
            how_to_fix="Point unlock-test at the experiment that produced the frozen model.",
        )

    # --- Integrity: recompute every recorded hash before touching test rows.
    checkpoint_dir = frozen_dir / "checkpoint"
    current_files = {item.name: sha256_file(item) for item in sorted(checkpoint_dir.iterdir())}
    _integrity(current_files == manifest_doc.checkpoint_hashes, "model checkpoint")
    _integrity(
        hash_yaml_file(frozen_dir / "frozen_experiment.yaml") == manifest_doc.frozen_config_hash,
        "frozen experiment config",
    )
    frozen_exp = load_experiment(frozen_dir / "frozen_experiment.yaml")
    if frozen_exp.task.model_dump() != live_experiment.task.model_dump():
        raise ArtifactIntegrityError(
            "live experiment task does not match the frozen experiment.",
            how_to_fix=(
                "unlock-test scores the frozen task, including primary_metric. "
                "Restore the experiment.yaml used at freeze, or start a new experiment_id."
            ),
        )
    if frozen_exp.preprocessing.model_dump() != live_experiment.preprocessing.model_dump():
        raise ArtifactIntegrityError(
            "live experiment preprocessing does not match the frozen experiment.",
            how_to_fix="Restore the preprocessing block used at freeze. Do not refit on test.",
        )
    experiment = frozen_exp
    decision_path = dest / "reports" / f"optimization_decision_{model_name}.json"
    _integrity(
        decision_path.is_file() and sha256_file(decision_path) == manifest_doc.decision_hash,
        "optimization decision document",
    )
    split_path = dest / "splits.parquet"
    _integrity(split_path.is_file(), "split file (missing)")
    splits = pd.read_parquet(split_path)
    _integrity(hash_dataframe(splits) == manifest_doc.split_file_hash, "split assignment")
    assert_no_group_leakage(
        splits, ["experimental_unit_id", "subject_id", "biospecimen_id"]
    )
    _integrity(
        hash_source_tree(PACKAGE_ROOT / "evaluation") == manifest_doc.evaluator_code_hash,
        "evaluator source code",
    )
    _integrity(
        hash_source_tree(PACKAGE_ROOT / "splitting") == manifest_doc.splitting_code_hash,
        "splitting source code",
    )

    # --- Rebuild data with the frozen split (never reassigned).
    bundle = load_local_bundle(manifest_path)
    labeled = bundle.with_split(splits[["experimental_unit_id", "split"]].drop_duplicates())
    processed = labeled.apply_assay_preprocessing(experiment.preprocessing.per_modality)
    _assert_fit_split_train(processed)
    _integrity(
        hash_mapping(
            {
                "observations": hash_dataframe(processed.observations),
                "samples": hash_dataframe(processed.sample_sheet),
            }
        )
        == manifest_doc.data_hash,
        "dataset content",
    )
    train = processed.subset(SplitName.TRAIN)
    test = processed.subset(SplitName.TEST)
    test_data = prepare_split_data(
        full=processed, train=train, split_bundle=test, split=SplitName.TEST, task=experiment.task
    )

    # --- Consume the lock BEFORE scoring (fail-closed: a crash cannot grant
    # a second attempt).
    consumed_at = datetime.now(UTC).isoformat()
    _write_lock(
        dest,
        TestLockState(
            experiment_id=experiment.experiment_id,
            model_name=model_name,
            consumed=True,
            consumed_at=consumed_at,
        ),
    )

    plugin = get_model(model_name)
    plugin.load(checkpoint_dir)
    prediction = plugin.predict(test_data)
    report = evaluate_predictions(
        y_true=test_data.forecast.y_true,
        y_pred=prediction.y_pred,
        mask=test_data.forecast.y_mask,
        feature_names=test_data.forecast.feature_names,
        instance_ids=test_data.forecast.instance_ids,
        group_ids=test_data.forecast.group_ids,
        model_name=model_name,
        split=SplitName.TEST.value,
        target_modality=experiment.task.target_modality,
        primary_metric=experiment.task.primary_metric,
        bootstrap_replicates=experiment.evaluation.bootstrap_replicates,
        seed=experiment.seed,
    )
    report_path = dest / "reports" / f"final_test_{model_name}.json"
    write_benchmark_report(
        report_path,
        experiment_id=experiment.experiment_id,
        hashes={
            "frozen_config_hash": manifest_doc.frozen_config_hash,
            "split_file_hash": manifest_doc.split_file_hash,
            "data_hash": manifest_doc.data_hash,
            "evaluator_code_hash": manifest_doc.evaluator_code_hash,
        },
        reports=[report],
        notes=[
            "One-shot final test on the frozen checkpoint. The test lock is now "
            "consumed; further tuning under this experiment_id is refused.",
        ],
    )
    _write_lock(
        dest,
        TestLockState(
            experiment_id=experiment.experiment_id,
            model_name=model_name,
            consumed=True,
            consumed_at=consumed_at,
            final_report_hash=sha256_file(report_path),
        ),
    )
    run_id = log_benchmark_run(
        tracking_uri=dest / "mlruns",
        experiment_name=experiment.experiment_id,
        run_name=f"{experiment.experiment_id}-final-test-{model_name}",
        hashes={
            "frozen_config_hash": manifest_doc.frozen_config_hash,
            "split_file_hash": manifest_doc.split_file_hash,
            "data_hash": manifest_doc.data_hash,
            "evaluator_code_hash": manifest_doc.evaluator_code_hash,
            "seed": str(experiment.seed),
        },
        seed=experiment.seed,
        params={
            "model": model_name,
            "test_lock_consumed_at": consumed_at,
            "checkpoint": str(checkpoint_dir),
        },
        reports=[report],
        artifacts=[report_path, freeze_path],
    )
    return {
        "experiment_id": experiment.experiment_id,
        "model": model_name,
        "consumed_at": consumed_at,
        "report_json": str(report_path),
        "report_md": str(report_path.with_suffix(".md")),
        "primary_metric": experiment.task.primary_metric,
        "primary_value": report.primary_value,
        "mlflow_run_id": run_id,
        "lock_path": str(_lock_path(dest)),
    }
