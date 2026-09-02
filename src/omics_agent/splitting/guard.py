"""Hard checks that independent biological units do not cross splits."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from omics_agent.errors import SplitLeakageError
from omics_agent.schemas.enums import SplitName

REQUIRED_SPLITS = (SplitName.TRAIN.value, SplitName.VAL.value, SplitName.TEST.value)


def _sets_by_split(frame: pd.DataFrame, column: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for split_name, part in frame.groupby("split"):
        groups[str(split_name)] = set(part[column].astype(str))
    return groups


def assert_no_group_leakage(
    split_df: pd.DataFrame,
    group_columns: Iterable[str] = ("experimental_unit_id",),
) -> None:
    """Fail if any identity in ``group_columns`` appears in two splits.

    Parameters
    ----------
    split_df:
        Table with a ``split`` column and each requested identity column.
        One row per sample or per experimental unit is accepted.
    group_columns:
        Columns that must be disjoint across train/val/test. Typical values
        are experimental_unit_id, subject_id, biospecimen_id, and for RCS
        experiments ``batch``.
    """

    if "split" not in split_df.columns:
        raise SplitLeakageError(
            "Split table has no 'split' column.",
            how_to_fix="Write splits.parquet with columns experimental_unit_id and split.",
        )
    unknown = sorted(set(split_df["split"].astype(str)) - set(REQUIRED_SPLITS))
    if unknown:
        raise SplitLeakageError(
            f"Unknown split labels {unknown}.",
            how_to_fix="Allowed labels are exactly train, val, test.",
        )
    present = set(split_df["split"].astype(str))
    missing = [name for name in REQUIRED_SPLITS if name not in present]
    if missing:
        raise SplitLeakageError(
            f"Split table is missing partitions {missing}.",
            how_to_fix="Every experiment must isolate train, val, and test. Re-run the splitter.",
        )
    for column in group_columns:
        if column not in split_df.columns:
            raise SplitLeakageError(
                f"Split table is missing identity column '{column}'.",
                how_to_fix=f"Add {column} to the sample sheet and to the split table.",
            )
        groups = _sets_by_split(split_df, column)
        pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        for left, right in pairs:
            leaked = sorted(groups[left] & groups[right])
            if leaked:
                raise SplitLeakageError(
                    f"{column} leaks across {left} and {right}: {leaked[:12]}. "
                    "The same experimental unit, subject, or biospecimen must not "
                    "appear in more than one split.",
                    how_to_fix=(
                        "Split by the independent biological unit, not by sample rows. "
                        "For longitudinal data use subject_id / experimental_unit_id. "
                        "For repeated cross-section also block experiment batch. "
                        "Re-run: omics-agent split --experiment <experiment.yaml>"
                    ),
                )


def assert_time_not_confounded_with_batch(
    observations: pd.DataFrame,
    *,
    time_col: str = "time",
    batch_col: str = "batch",
) -> None:
    """Stop if every time point lives in a unique batch (time ≡ batch).

    In that case a time effect cannot be separated from a batch effect and
    the study is not modelable without a reviewer decision.
    """

    if batch_col not in observations.columns:
        raise SplitLeakageError(
            f"Observations have no '{batch_col}' column.",
            how_to_fix="Record the experiment batch. If unknown, add it to needs_review.",
        )
    pairs = observations[[time_col, batch_col]].drop_duplicates()
    times = set(pairs[time_col])
    batches = set(pairs[batch_col])
    if len(times) > 1 and len(batches) == len(times) and len(pairs) == len(times):
        raise SplitLeakageError(
            "Time and experiment batch are completely confounded: each time point "
            "is a unique batch. A time-course model would fit batch.",
            how_to_fix=(
                "Do not train. Add this to human_review.unresolved and ask the "
                "experimentalist whether independent batches cover multiple times. "
                "The pipeline will not pretend the confound is a time effect."
            ),
        )
