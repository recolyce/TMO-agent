"""Shared fixtures for synthetic bulk datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.data_sources.synthetic import generate_synthetic_dataset
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.enums import SamplingDesign


@pytest.fixture
def longitudinal_dir(tmp_path: Path) -> Path:
    dest = tmp_path / "longitudinal"
    generate_synthetic_dataset(dest, design=SamplingDesign.LONGITUDINAL, seed=20260901)
    return dest


@pytest.fixture
def rcs_dir(tmp_path: Path) -> Path:
    dest = tmp_path / "rcs"
    generate_synthetic_dataset(
        dest, design=SamplingDesign.REPEATED_CROSS_SECTIONAL, seed=20260901
    )
    return dest


@pytest.fixture
def longitudinal_bundle(longitudinal_dir: Path) -> MultiOmicsBundle:
    return load_local_bundle(longitudinal_dir / "dataset.yaml")


@pytest.fixture
def rcs_bundle(rcs_dir: Path) -> MultiOmicsBundle:
    return load_local_bundle(rcs_dir / "dataset.yaml")
