"""Experiment config: task, split, models, and evaluation policy.

Split membership, the evaluator, and the primary metric are not writable by
an optimizer or Agent. Changing them requires a new ``experiment_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from omics_agent.errors import SchemaError
from omics_agent.schemas.dataset import StrictModel, load_manifest
from omics_agent.schemas.enums import (
    HistoryPolicy,
    PairingLevel,
    SamplingDesign,
    SplitName,
    TaskKind,
)


class TaskConfig(StrictModel):
    """One primary endpoint. Do not mix forecast and contemporaneous tasks."""

    kind: TaskKind
    target_modality: str
    input_modalities: list[str] = Field(min_length=1)
    history: HistoryPolicy = HistoryPolicy.LAST_OBSERVATION
    horizon_steps: int = Field(default=1, ge=1)
    target_time_min: float | None = Field(
        default=None,
        description="For group_time_forecast, only times >= this value are scored.",
    )
    primary_metric: str = Field(
        default="protein_macro_pcc",
        description="Frozen primary metric name. Optimizer cannot change this field.",
    )

    @model_validator(mode="after")
    def target_is_an_input_or_explicit(self) -> Self:
        if not self.input_modalities:
            raise SchemaError(
                "task.input_modalities is empty.",
                how_to_fix="List the modalities the model may see, e.g. [rna, protein].",
            )
        return self


class SplitFractions(StrictModel):
    """Train/val/test fractions. They must sum to 1."""

    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def sums_to_one(self) -> Self:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-8:
            raise SchemaError(
                f"Split fractions sum to {total}, not 1.",
                how_to_fix="Use values such as train: 0.6, val: 0.2, test: 0.2.",
            )
        return self


class SplitConfig(StrictModel):
    """How independent units are assigned to splits.

    ``group_columns`` are checked for leakage. For repeated cross-section,
    ``block_experiment_batch`` defaults to true so a technical batch cannot
    sit in both train and test.
    """

    group_columns: list[str] = Field(default_factory=lambda: ["experimental_unit_id", "subject_id"])
    fractions: SplitFractions = Field(
        default_factory=lambda: SplitFractions(train=0.6, val=0.2, test=0.2)
    )
    block_experiment_batch: bool | None = None
    also_block: list[str] = Field(default_factory=lambda: ["biospecimen_id"])
    assignment: dict[str, SplitName] | None = Field(
        default=None,
        description="Optional explicit map of group key -> split. Used for RCS batches.",
    )


class ModelParams(StrictModel):
    """Hyperparameters for one registered baseline."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(StrictModel):
    """Train-only transformers. Fitting on all rows is a hard error."""

    scale: bool = True
    impute: bool = True


class EvaluationConfig(StrictModel):
    """Evaluator settings. Bootstrap resamples experimental units."""

    bootstrap_replicates: int = Field(default=200, ge=20)
    unlock_test: bool = Field(
        default=False,
        description="If false, only validation is scored. Toy runs may set true.",
    )
    toy_unlock_disclaimer: str = (
        "Synthetic holdout only. This is not a production test unlock."
    )


class ExperimentConfig(StrictModel):
    """Validated ``experiment.yaml``."""

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    experiment_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    dataset: Path
    seed: int = Field(ge=0)
    task: TaskConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: list[ModelParams] = Field(min_length=1)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: Path | None = None

    def load_dataset_design(self, experiment_path: Path) -> tuple[SamplingDesign, PairingLevel, list[str]]:
        """Load the linked manifest and return design fields used for task checks."""

        manifest_path = self.dataset if self.dataset.is_absolute() else (experiment_path.parent / self.dataset)
        manifest = load_manifest(manifest_path.resolve())
        return (
            manifest.design.sampling_design,
            manifest.design.pairing_level,
            list(manifest.modalities),
        )


def assert_task_matches_design(
    *,
    task: TaskConfig,
    sampling_design: SamplingDesign,
    pairing_level: PairingLevel,
    modalities: list[str],
) -> None:
    """Refuse task/design combinations that would invent biology."""

    unknown = [name for name in [task.target_modality, *task.input_modalities] if name not in modalities]
    if unknown:
        raise SchemaError(
            f"Task refers to unknown modalities {unknown}.",
            how_to_fix=f"Declared modalities are {modalities}.",
        )
    if task.kind is TaskKind.SUBJECT_FORECAST:
        if sampling_design is not SamplingDesign.LONGITUDINAL:
            raise SchemaError(
                "task.kind='subject_forecast' is illegal for repeated cross-sectional data. "
                "It would concatenate different animals into one fake trajectory.",
                how_to_fix="Use task.kind: group_time_forecast and keep experimental_unit_id unique per animal.",
            )
        if pairing_level is PairingLevel.GROUP_LEVEL_ONLY and len(task.input_modalities) > 1:
            raise SchemaError(
                "group_level_only data cannot feed multiple modalities into a sample-level forecast.",
                how_to_fix="Use a group-level task, or only after pairing_level is proven.",
            )
    if task.kind is TaskKind.GROUP_TIME_FORECAST and task.target_time_min is None:
        raise SchemaError(
            "group_time_forecast requires task.target_time_min so early times are not scored as if they were forecasts.",
            how_to_fix="Set target_time_min to the first held-out time, e.g. 4.0.",
        )


def load_experiment(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML."""

    import yaml

    from omics_agent.errors import SchemaError as _SchemaError

    if not path.is_file():
        raise _SchemaError(
            f"Experiment config not found: {path}",
            how_to_fix="Pass config/experiment.example.yaml or a generated experiment.yaml.",
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise _SchemaError(
            f"{path} is not a YAML mapping.",
            how_to_fix="Start from config/experiment.example.yaml.",
        )
    return ExperimentConfig.model_validate(payload)
