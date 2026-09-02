"""Handcrafted longitudinal data for dynamics/MLP model tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omics_agent.models.tasks import DataForModel, prepare_split_data
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import SplitName, TaskKind
from omics_agent.schemas.experiment import TaskConfig

MANIFEST_PATH = Path("config/dataset.example.yaml")


def make_task(**overrides: object) -> TaskConfig:
    base: dict[str, object] = {
        "kind": TaskKind.SUBJECT_FORECAST,
        "target_modality": "protein",
        "input_modalities": ["rna", "protein"],
    }
    base.update(overrides)
    return TaskConfig(**base)  # type: ignore[arg-type]


def make_bundle(
    *,
    n_units: int = 8,
    times: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
    constant: bool = False,
    seed: int = 7,
    with_missing: bool = True,
) -> MultiOmicsBundle:
    """Units follow smooth trajectories on an irregular time grid (gap 1,1,2)."""

    manifest = load_manifest(MANIFEST_PATH)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    rna_rows: list[np.ndarray] = []
    prot_rows: list[np.ndarray] = []
    n_val = max(1, n_units // 4)
    splits = ["train"] * (n_units - n_val) + ["val"] * n_val
    for unit in range(n_units):
        for t in times:
            rows.append(
                {
                    "observation_id": f"U{unit}_T{t}",
                    "experimental_unit_id": f"U{unit}",
                    "subject_id": f"S{unit}",
                    "time": float(t),
                    "condition": "disease" if unit % 2 else "control",
                    "batch": "B1",
                    "split": splits[unit],
                }
            )
            if constant:
                rna_rows.append(np.array([1.0, 1.0, 1.0]))
                prot_rows.append(np.array([2.0, 2.0]))
            else:
                base = 0.4 * unit
                rna_rows.append(
                    np.array([np.sin(t) + base, np.cos(t), 0.1 * t]) + rng.normal(0.0, 0.05, 3)
                )
                prot_rows.append(
                    np.array([0.5 * np.sin(t + 0.3) + base, 0.2 * t + 0.1 * base])
                    + rng.normal(0.0, 0.05, 2)
                )
    rna = np.vstack(rna_rows)
    protein = np.vstack(prot_rows)
    if with_missing and not constant:
        protein[1, 1] = np.nan  # one missing history/target entry
    observations = pd.DataFrame(rows)
    return MultiOmicsBundle(
        manifest=manifest,
        manifest_path=MANIFEST_PATH,
        observations=observations,
        matrices={"rna": rna, "protein": protein},
        feature_names={"rna": ["G1", "G2", "G3"], "protein": ["P1", "P2"]},
        missing={"rna": np.isnan(rna), "protein": np.isnan(protein)},
        sample_sheet=pd.DataFrame(),
    )


def make_split_data(
    bundle: MultiOmicsBundle, task: TaskConfig
) -> tuple[DataForModel, DataForModel]:
    train_b = bundle.subset(SplitName.TRAIN)
    val_b = bundle.subset(SplitName.VAL)
    train = prepare_split_data(
        full=bundle, train=train_b, split_bundle=train_b, split=SplitName.TRAIN, task=task
    )
    val = prepare_split_data(
        full=bundle, train=train_b, split_bundle=val_b, split=SplitName.VAL, task=task
    )
    return train, val
