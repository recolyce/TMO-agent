"""Assign train/val/test by independent biological units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omics_agent.errors import SplitLeakageError
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.enums import SamplingDesign, SplitName
from omics_agent.schemas.experiment import SplitConfig
from omics_agent.splitting.guard import (
    assert_no_group_leakage,
    assert_time_not_confounded_with_batch,
)


def _composite_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].astype(str).agg("::".join, axis=1)


def assign_splits(
    bundle: MultiOmicsBundle,
    config: SplitConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    """Return one row per observation with a locked split label.

    Group membership is defined by ``config.group_columns``. For repeated
    cross-section, ``batch`` is also blocked when ``block_experiment_batch``
    is true (the default for that design).
    """

    observations = bundle.observations.copy()
    assert_time_not_confounded_with_batch(observations)

    block_batch = config.block_experiment_batch
    if block_batch is None:
        block_batch = bundle.sampling_design is SamplingDesign.REPEATED_CROSS_SECTIONAL

    group_columns = list(config.group_columns)
    if block_batch and "batch" not in group_columns:
        group_columns.append("batch")

    units = observations[group_columns].drop_duplicates().reset_index(drop=True)
    units["group_key"] = _composite_key(units, group_columns)

    if config.assignment:
        assigned = []
        for _, row in units.iterrows():
            # Explicit map is by the first group column (usually batch or unit).
            key = str(row[group_columns[0]])
            if key not in config.assignment:
                raise SplitLeakageError(
                    f"split.assignment is missing key '{key}'.",
                    how_to_fix="List every group value in split.assignment or drop assignment and use fractions.",
                )
            assigned.append(config.assignment[key].value)
        units["split"] = assigned
    else:
        keys = units["group_key"].to_numpy()
        unique_keys = np.unique(keys)
        rng = np.random.default_rng(seed)
        shuffled = unique_keys.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(config.fractions.train * n))
        n_val = int(round(config.fractions.val * n))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise SplitLeakageError(
                f"Only {n} independent groups; cannot form train/val/test.",
                how_to_fix=(
                    "Add more experimental units or experiment batches. "
                    "With 3-way splitting you need at least 3 independent groups."
                ),
            )
        mapping: dict[str, str] = {}
        mapping.update({k: SplitName.TRAIN.value for k in shuffled[:n_train]})
        mapping.update({k: SplitName.VAL.value for k in shuffled[n_train : n_train + n_val]})
        mapping.update({k: SplitName.TEST.value for k in shuffled[n_train + n_val :]})
        units["split"] = units["group_key"].map(mapping)

    merged = observations.merge(units[group_columns + ["split"]], on=group_columns, how="left")
    if merged["split"].isna().any():
        raise SplitLeakageError(
            "Some observations did not receive a split label.",
            how_to_fix="Check that group_columns exist on every observation row.",
        )

    guard_columns = list(dict.fromkeys([*group_columns, *config.also_block]))
    assert_no_group_leakage(merged, guard_columns)
    return merged[
        [
            "observation_id",
            "experimental_unit_id",
            "subject_id",
            "biospecimen_id",
            "batch",
            "time",
            "condition",
            "split",
        ]
    ]


def write_splits(frame: pd.DataFrame, path: Path) -> None:
    """Write the locked split table. Outputs never overwrite raw inputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
