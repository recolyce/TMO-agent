from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omics_agent.errors import SchemaError
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.dataset import load_manifest
from omics_agent.schemas.enums import AssayStrategy
from omics_agent.schemas.preprocess import ModalityPreprocessConfig

RNA = np.array(
    [
        [90.0, 10.0, 0.0],
        [30.0, 10.0, 10.0],
        [20.0, 20.0, 10.0],
        [50.0, 25.0, 25.0],
        [40.0, 40.0, 20.0],
        [10.0, 60.0, 30.0],
    ]
)
PROTEIN = np.array(
    [
        [0.0, 4.0],
        [2.0, np.nan],
        [8.0, 16.0],
        [4.0, 8.0],
        [16.0, 2.0],
        [32.0, 64.0],
    ]
)
CONFIGS = {
    "rna": ModalityPreprocessConfig(strategy=AssayStrategy.BULK_RNA_COUNTS),
    "protein": ModalityPreprocessConfig(strategy=AssayStrategy.PROTEIN_INTENSITY),
}


def _bundle(*, with_split: bool = True) -> MultiOmicsBundle:
    manifest = load_manifest(Path("config/dataset.example.yaml"))
    observations = pd.DataFrame(
        {
            "observation_id": [f"O{i}" for i in range(6)],
            "experimental_unit_id": [f"U{i}" for i in range(6)],
            "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
    )
    if with_split:
        observations["split"] = ["train", "train", "train", "train", "val", "test"]
    return MultiOmicsBundle(
        manifest=manifest,
        manifest_path=Path("config/dataset.example.yaml"),
        observations=observations,
        matrices={"rna": RNA.copy(), "protein": PROTEIN.copy()},
        feature_names={"rna": ["G1", "G2", "G3"], "protein": ["P1", "P2"]},
        missing={"rna": np.isnan(RNA), "protein": np.isnan(PROTEIN)},
        sample_sheet=pd.DataFrame(),
    )


def test_layers_raw_normalized_scaled_are_kept() -> None:
    processed = _bundle().apply_assay_preprocessing(CONFIGS)
    for modality in ("rna", "protein"):
        assert set(processed.layers[modality]) == {"raw", "normalized", "scaled"}
    # Raw layer is untouched: the protein zero is still 0 there.
    assert processed.layers["protein"]["raw"][0, 0] == 0.0
    library = RNA.sum(axis=1, keepdims=True)
    expected = np.log1p(RNA / library * 1_000_000.0)
    assert np.allclose(processed.layers["rna"]["normalized"], expected)
    assert np.array_equal(processed.matrices["rna"], processed.layers["rna"]["scaled"])


def test_protein_missing_stays_nan_in_every_derived_layer() -> None:
    processed = _bundle().apply_assay_preprocessing(CONFIGS)
    normalized = processed.layers["protein"]["normalized"]
    scaled = processed.layers["protein"]["scaled"]
    # The zero intensity and the original NaN are both missing, never 0.
    for row, col in ((0, 0), (1, 1)):
        assert np.isnan(normalized[row, col])
        assert np.isnan(scaled[row, col])
        assert processed.missing["protein"][row, col]
    assert not np.any(scaled[processed.missing["protein"]] == 0)


def test_preprocessing_before_split_is_refused() -> None:
    with pytest.raises(SchemaError, match="before the split"):
        _bundle(with_split=False).apply_assay_preprocessing(CONFIGS)


def test_scaler_provenance_proves_train_only_fit() -> None:
    processed = _bundle().apply_assay_preprocessing(CONFIGS)
    fitted = [rec for rec in processed.provenance if "fit_split" in rec]
    stateless = [rec for rec in processed.provenance if rec.get("learns_statistics") is False]
    assert len(fitted) == 2 and len(stateless) == 2
    for record in fitted:
        assert record["fit_split"] == "train"
        assert record["n_fit_samples"] == 4  # train rows only, not all 6
    names = {rec["transformer_name"] for rec in stateless}
    assert names == {"rna_bulk_rna_counts", "protein_protein_intensity"}


def test_mudata_has_layers_qc_and_provenance() -> None:
    processed = _bundle().apply_assay_preprocessing(CONFIGS)
    mdata = processed.to_mudata()
    protein = mdata["protein"]
    assert {"raw", "normalized", "scaled"} == set(protein.layers.keys())
    assert np.array_equal(
        np.asarray(protein.X), processed.layers["protein"]["scaled"], equal_nan=True
    )
    assert "qc_n_observed" in protein.obs.columns
    assert "qc_pct_missing" in protein.var.columns
    # Observation O0 has the zero-intensity protein: only 1 of 2 observed.
    assert int(protein.obs.loc["O0", "qc_n_observed"]) == 1
    provenance = json.loads(protein.uns["fit_split_provenance_json"])
    kinds = {rec.get("kind") for rec in provenance}
    assert "stateless_per_sample" in kinds
    assert all(rec["fit_split"] == "train" for rec in provenance if "fit_split" in rec)
