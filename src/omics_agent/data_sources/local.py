"""Load a local processed-matrix dataset described by a manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from omics_agent.errors import ManifestError
from omics_agent.preprocessing.bundle import MultiOmicsBundle
from omics_agent.schemas.dataset import DatasetManifest, load_manifest
from omics_agent.schemas.samples import load_sample_sheet


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ManifestError(
            f"Data file not found: {path}",
            how_to_fix=(
                "Run omics-agent generate-synthetic --output-dir <dir> first, "
                "or point files[].path at an existing parquet/tsv."
            ),
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ManifestError(
        f"Unsupported table format: {path.suffix}",
        how_to_fix="Use .parquet, .tsv, or .csv for milestone 1 local matrices.",
    )


def load_local_bundle(manifest_path: Path) -> MultiOmicsBundle:
    """Load matrices and the sample sheet referenced by ``manifest_path``.

    Raw files are opened read-only. The returned bundle never writes back
    into the input directory.
    """

    manifest_path = manifest_path.resolve()
    manifest: DatasetManifest = load_manifest(manifest_path)
    sample_path = manifest.resolve_path(manifest_path, manifest.sample_sheet)
    samples = _read_table(sample_path)
    sheet = load_sample_sheet(
        samples,
        sampling_design=manifest.design.sampling_design,
        declared_modalities=list(manifest.modalities),
    )
    matrices: dict[str, pd.DataFrame] = {}
    for spec in manifest.files:
        if spec.role.value != "matrix" or spec.path is None:
            continue
        matrix_path = manifest.resolve_path(manifest_path, spec.path)
        frame = _read_table(matrix_path)
        if "sample_id" not in frame.columns:
            raise ManifestError(
                f"Matrix {matrix_path} has no sample_id column.",
                how_to_fix="The first column must be sample_id matching the sample sheet.",
            )
        matrices[spec.modality] = frame.set_index("sample_id")
    return MultiOmicsBundle.from_long_tables(
        manifest=manifest,
        manifest_path=manifest_path,
        sample_sheet=sheet.to_frame(),
        matrices=matrices,
    )
