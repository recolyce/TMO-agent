from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omics_agent.errors import NeedsReviewError, SchemaError
from omics_agent.schemas.dataset import DatasetManifest, load_manifest
from omics_agent.schemas.enums import PairingLevel, SamplingDesign, TaskKind
from omics_agent.schemas.experiment import (
    ExperimentConfig,
    TaskConfig,
    assert_task_matches_design,
    load_experiment,
)


def test_example_dataset_manifest_is_valid() -> None:
    manifest = load_manifest(Path("config/dataset.example.yaml"))
    assert manifest.dataset_id == "synthetic_longitudinal_rna_protein"
    assert manifest.design.sampling_design is SamplingDesign.LONGITUDINAL


def test_example_experiment_is_valid() -> None:
    cfg = load_experiment(Path("config/experiment.example.yaml"))
    assert cfg.task.primary_metric == "protein_macro_pcc"
    assert cfg.task.kind is TaskKind.SUBJECT_FORECAST


def test_example_priors_experiment_uses_fixture_embeddings() -> None:
    cfg = load_experiment(Path("config/experiment.priors.example.yaml"))
    assert cfg.priors.embedding.name.value == "synthetic_pathway_onehot"


def test_manifest_rejects_missing_time_unit() -> None:
    payload = load_manifest(Path("config/dataset.example.yaml")).model_dump(mode="json")
    payload["design"].pop("time_unit")
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(payload)


def test_group_level_only_cannot_claim_pairing() -> None:
    payload = load_manifest(Path("config/dataset.example.yaml")).model_dump(mode="json")
    payload["design"]["pairing_level"] = PairingLevel.GROUP_LEVEL_ONLY.value
    payload["design"]["paired_modalities"] = True
    with pytest.raises(SchemaError, match="group_level_only"):
        DatasetManifest.model_validate(payload)


def test_approved_manifest_cannot_keep_unresolved_items() -> None:
    payload = load_manifest(Path("config/dataset.example.yaml")).model_dump(mode="json")
    payload["human_review"]["unresolved"] = ["Is R1 the same biospecimen as P1?"]
    with pytest.raises(SchemaError, match="unresolved"):
        DatasetManifest.model_validate(payload)


def test_needs_review_blocks_training(tmp_path: Path) -> None:
    payload = load_manifest(Path("config/dataset.example.yaml")).model_dump(mode="json")
    payload["human_review"] = {
        "status": "required",
        "unresolved": ["Confirm whether time is hours or days."],
    }
    manifest = DatasetManifest.model_validate(payload)
    with pytest.raises(NeedsReviewError, match="needs human review"):
        manifest.require_approved_for_training()


def test_subject_forecast_rejected_on_rcs() -> None:
    with pytest.raises(SchemaError, match="illegal for repeated cross-sectional"):
        assert_task_matches_design(
            task=TaskConfig(
                kind=TaskKind.SUBJECT_FORECAST,
                target_modality="protein",
                input_modalities=["rna", "protein"],
            ),
            sampling_design=SamplingDesign.REPEATED_CROSS_SECTIONAL,
            pairing_level=PairingLevel.SAME_BIOSPECIMEN,
            modalities=["rna", "protein"],
        )


def test_group_level_only_rejects_sample_level_forecast() -> None:
    with pytest.raises(SchemaError, match="group_level_only"):
        assert_task_matches_design(
            task=TaskConfig(
                kind=TaskKind.SUBJECT_FORECAST,
                target_modality="protein",
                input_modalities=["rna", "protein"],
            ),
            sampling_design=SamplingDesign.LONGITUDINAL,
            pairing_level=PairingLevel.GROUP_LEVEL_ONLY,
            modalities=["rna", "protein"],
        )


def test_approved_manifest_rejects_undeclared_design() -> None:
    payload = load_manifest(Path("config/dataset.example.yaml")).model_dump(mode="json")
    payload["design"]["sampling_design"] = "undeclared"
    payload["design"]["longitudinal"] = None
    with pytest.raises(SchemaError, match="undeclared"):
        DatasetManifest.model_validate(payload)


def test_experiment_rejects_unknown_keys() -> None:
    payload = load_experiment(Path("config/experiment.example.yaml")).model_dump(mode="json")
    payload["secret_override_test"] = True
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)
