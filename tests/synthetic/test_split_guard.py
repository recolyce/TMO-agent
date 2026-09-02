from __future__ import annotations

import pandas as pd
import pytest

from omics_agent.errors import SchemaError, SplitLeakageError
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.enums import SplitName
from omics_agent.schemas.experiment import SplitConfig, SplitFractions
from omics_agent.splitting.guard import assert_no_group_leakage
from omics_agent.splitting.split import assign_splits


def test_subject_leakage_fails() -> None:
    frame = pd.DataFrame(
        {
            "experimental_unit_id": ["S01", "S01", "S02"],
            "subject_id": ["S01", "S01", "S02"],
            "biospecimen_id": ["b1", "b2", "b3"],
            "split": ["train", "test", "val"],
        }
    )
    with pytest.raises(SplitLeakageError, match="leaks"):
        assert_no_group_leakage(frame, ["experimental_unit_id", "subject_id"])


def test_biospecimen_leakage_fails() -> None:
    frame = pd.DataFrame(
        {
            "experimental_unit_id": ["S01", "S02", "S03"],
            "subject_id": ["S01", "S02", "S03"],
            "biospecimen_id": ["shared", "shared", "other"],
            "split": ["train", "val", "test"],
        }
    )
    with pytest.raises(SplitLeakageError, match="biospecimen_id"):
        assert_no_group_leakage(frame, ["biospecimen_id"])


def test_rcs_batch_leakage_fails() -> None:
    frame = pd.DataFrame(
        {
            "experimental_unit_id": ["A1", "A2", "A3"],
            "subject_id": ["A1", "A2", "A3"],
            "batch": ["expA", "expA", "expB"],
            "split": ["train", "test", "val"],
        }
    )
    with pytest.raises(SplitLeakageError, match="batch"):
        assert_no_group_leakage(frame, ["batch"])


def test_longitudinal_split_has_no_subject_overlap(longitudinal_bundle: MultiOmicsBundle) -> None:
    splits = assign_splits(
        longitudinal_bundle,
        SplitConfig(
            group_columns=["experimental_unit_id", "subject_id"],
            fractions=SplitFractions(train=0.6, val=0.2, test=0.2),
            block_experiment_batch=False,
        ),
        seed=20260901,
    )
    assert_no_group_leakage(splits, ["experimental_unit_id", "subject_id", "biospecimen_id"])
    assert set(splits["split"]) == {"train", "val", "test"}


def test_split_config_cannot_drop_unit_and_subject_guards() -> None:
    """Rule 1: a YAML that only groups by batch must still guard subjects."""

    with pytest.raises(SchemaError, match="experimental_unit_id"):
        SplitConfig(group_columns=["batch"], also_block=[])


def test_with_split_rejects_conflicting_unit_labels(longitudinal_bundle: MultiOmicsBundle) -> None:
    """The same experimental_unit_id must not map to two splits (last-wins is leakage)."""

    unit = str(longitudinal_bundle.observations["experimental_unit_id"].iloc[0])
    frame = pd.DataFrame(
        {
            "experimental_unit_id": [unit, unit],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(SplitLeakageError, match="experimental_unit_id"):
        longitudinal_bundle.with_split(frame)


def test_rcs_split_blocks_experiment_batch(rcs_bundle: MultiOmicsBundle) -> None:
    splits = assign_splits(
        rcs_bundle,
        SplitConfig(
            group_columns=["batch"],
            block_experiment_batch=True,
            also_block=["experimental_unit_id", "subject_id", "biospecimen_id"],
            assignment={
                "expA": SplitName.TRAIN,
                "expB": SplitName.VAL,
                "expC": SplitName.TEST,
            },
        ),
        seed=20260901,
    )
    assert_no_group_leakage(splits, ["batch", "experimental_unit_id"])
    by_batch = splits.groupby("batch")["split"].nunique()
    assert (by_batch == 1).all()
