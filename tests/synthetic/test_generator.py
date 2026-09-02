from __future__ import annotations

from pathlib import Path

import pandas as pd

from omics_agent.data_sources.synthetic import TRUE_EDGES, generate_synthetic_dataset
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import SamplingDesign
from omics_agent.schemas.samples import load_sample_sheet


def test_longitudinal_synthetic_has_required_fields(longitudinal_dir: Path) -> None:
    manifest = load_manifest(longitudinal_dir / "dataset.yaml")
    samples = pd.read_csv(longitudinal_dir / "samples.tsv", sep="\t")
    sheet = load_sample_sheet(
        samples,
        sampling_design=manifest.design.sampling_design,
        declared_modalities=list(manifest.modalities),
    )
    frame = sheet.to_frame()
    assert {"subject_id", "experimental_unit_id", "time", "batch"} <= set(frame.columns)
    assert set(frame["modality"]) == {"rna", "protein"}
    assert frame["time"].nunique() >= 3
    rna = pd.read_parquet(longitudinal_dir / "rna.parquet")
    protein = pd.read_parquet(longitudinal_dir / "protein.parquet")
    assert rna.isna().any().any()
    assert protein.isna().any().any()
    edges = pd.read_parquet(longitudinal_dir / "true_edges.parquet")
    assert len(edges) == len(TRUE_EDGES)
    times_per_unit = frame.groupby("experimental_unit_id")["time"].nunique()
    assert (times_per_unit > 1).all()


def test_rcs_units_are_single_time(rcs_dir: Path) -> None:
    samples = pd.read_csv(rcs_dir / "samples.tsv", sep="\t")
    times = samples.groupby("experimental_unit_id")["time"].nunique()
    assert (times == 1).all()
    assert samples["batch"].nunique() == 3
    assert set(samples["modality"]) == {"rna", "protein"}


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "empty"
    plan = generate_synthetic_dataset(
        dest, design=SamplingDesign.LONGITUDINAL, dry_run=True
    )
    assert plan["dry_run"] is True
    assert not dest.exists()
