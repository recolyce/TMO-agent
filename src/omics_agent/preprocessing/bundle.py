"""Aligned multi-omics bundle: one observation row, several modality matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from omics_agent.errors import ManifestError, SchemaError
from omics_agent.preprocessing.scalers import TrainOnlyImputer, TrainOnlyStandardScaler
from omics_agent.schemas.dataset import DatasetManifest
from omics_agent.schemas.enums import PairingLevel, SamplingDesign, SplitName


@dataclass
class MultiOmicsBundle:
    """In-memory bulk multi-omics object used by split, models, and evaluator.

    ``observations`` has one row per ``observation_id`` (biospecimen/time).
    ``matrices[modality]`` is observations × features, aligned to that table.
    ``missing[modality]`` is True where the original value was NaN.
    """

    manifest: DatasetManifest
    manifest_path: Path
    observations: pd.DataFrame
    matrices: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    missing: dict[str, np.ndarray]
    sample_sheet: pd.DataFrame
    provenance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sampling_design(self) -> SamplingDesign:
        return self.manifest.design.sampling_design

    @property
    def pairing_level(self) -> PairingLevel:
        return self.manifest.design.pairing_level

    @classmethod
    def from_long_tables(
        cls,
        *,
        manifest: DatasetManifest,
        manifest_path: Path,
        sample_sheet: pd.DataFrame,
        matrices: dict[str, pd.DataFrame],
    ) -> MultiOmicsBundle:
        """Pivot a long sample sheet and per-modality sample×feature tables."""

        if manifest.design.pairing_level is PairingLevel.GROUP_LEVEL_ONLY:
            raise SchemaError(
                "group_level_only datasets cannot be assembled into a sample-aligned bundle.",
                how_to_fix=(
                    "Keep modalities at condition×time aggregates. Do not join them "
                    "on row index as if they were paired samples."
                ),
            )
        obs_rows: list[dict[str, Any]] = []
        for observation_id, grp in sample_sheet.groupby("observation_id", sort=True):
            first = grp.iloc[0]
            units = set(grp["experimental_unit_id"])
            subjects = set(grp["subject_id"])
            bios = set(grp["biospecimen_id"])
            times = set(grp["time"])
            if len(units) != 1 or len(subjects) != 1 or len(bios) != 1 or len(times) != 1:
                raise ManifestError(
                    f"observation_id '{observation_id}' mixes experimental units, subjects, "
                    "biospecimens, or times.",
                    how_to_fix="Each observation_id must be one biospecimen at one time.",
                )
            present = {str(m): True for m in grp["modality"]}
            obs_rows.append(
                {
                    "observation_id": str(observation_id),
                    "experimental_unit_id": str(first["experimental_unit_id"]),
                    "subject_id": str(first["subject_id"]),
                    "biospecimen_id": str(first["biospecimen_id"]),
                    "time": float(first["time"]),
                    "time_unit": str(first["time_unit"]),
                    "condition": str(first["condition"]),
                    "batch": str(first["batch"]),
                    "replicate_type": str(first["replicate_type"]),
                    **{f"has_{mod}": bool(present.get(mod, False)) for mod in manifest.modalities},
                }
            )
        observations = pd.DataFrame(obs_rows).sort_values("observation_id").reset_index(drop=True)
        aligned_x: dict[str, np.ndarray] = {}
        aligned_miss: dict[str, np.ndarray] = {}
        names: dict[str, list[str]] = {}
        for modality, spec_matrix in matrices.items():
            columns = [str(c) for c in spec_matrix.columns]
            names[modality] = columns
            stacked: list[np.ndarray] = []
            miss: list[np.ndarray] = []
            sheet_mod = sample_sheet[sample_sheet["modality"] == modality]
            sid_by_obs = dict(
                zip(
                    sheet_mod["observation_id"].astype(str),
                    sheet_mod["sample_id"].astype(str),
                    strict=True,
                )
            )
            for observation_id in observations["observation_id"]:
                sample_id = sid_by_obs.get(str(observation_id))
                if sample_id is None or sample_id not in spec_matrix.index:
                    row = np.full(len(columns), np.nan, dtype=float)
                else:
                    row = np.asarray(spec_matrix.loc[sample_id].reindex(columns), dtype=float)
                stacked.append(row)
                miss.append(np.isnan(row))
            aligned_x[modality] = np.vstack(stacked)
            aligned_miss[modality] = np.vstack(miss)
        return cls(
            manifest=manifest,
            manifest_path=manifest_path,
            observations=observations,
            matrices=aligned_x,
            feature_names=names,
            missing=aligned_miss,
            sample_sheet=sample_sheet,
            provenance=[],
        )

    def with_split(self, split_frame: pd.DataFrame) -> MultiOmicsBundle:
        """Attach a ``split`` column by experimental_unit_id."""

        if "experimental_unit_id" not in split_frame.columns or "split" not in split_frame.columns:
            raise SchemaError(
                "Split table must contain experimental_unit_id and split.",
                how_to_fix="Produce splits with omics-agent split.",
            )
        mapping = dict(
            zip(
                split_frame["experimental_unit_id"].astype(str),
                split_frame["split"].astype(str),
                strict=True,
            )
        )
        out = self.observations.copy()
        out["split"] = out["experimental_unit_id"].map(mapping)
        if out["split"].isna().any():
            missing_units = sorted(out.loc[out["split"].isna(), "experimental_unit_id"].unique())
            raise SchemaError(
                f"No split assignment for experimental units {missing_units[:8]}.",
                how_to_fix="Re-run the splitter so every experimental_unit_id is assigned.",
            )
        clone = MultiOmicsBundle(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            observations=out,
            matrices={k: v.copy() for k, v in self.matrices.items()},
            feature_names=self.feature_names,
            missing={k: v.copy() for k, v in self.missing.items()},
            sample_sheet=self.sample_sheet,
            provenance=list(self.provenance),
        )
        return clone

    def subset(self, split: SplitName) -> MultiOmicsBundle:
        """Return the rows belonging to one split. Does not copy the manifest."""

        if "split" not in self.observations.columns:
            raise SchemaError(
                "Bundle has no split column.",
                how_to_fix="Call bundle.with_split(splits) before subsetting.",
            )
        mask = self.observations["split"].to_numpy() == split.value
        idx = np.where(mask)[0]
        return MultiOmicsBundle(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            observations=self.observations.iloc[idx].reset_index(drop=True),
            matrices={name: matrix[idx] for name, matrix in self.matrices.items()},
            feature_names=self.feature_names,
            missing={name: matrix[idx] for name, matrix in self.missing.items()},
            sample_sheet=self.sample_sheet,
            provenance=list(self.provenance),
        )

    def apply_train_only_preprocessing(self) -> MultiOmicsBundle:
        """Fit scalers/imputers on train rows and transform every split.

        Provenance records always contain ``fit_split='train'``.
        """

        if "split" not in self.observations.columns:
            raise SchemaError(
                "Cannot preprocess before the split is attached.",
                how_to_fix="Lock the split first so transformers see only train rows.",
            )
        train_mask = self.observations["split"].to_numpy() == SplitName.TRAIN.value
        if not train_mask.any():
            raise SchemaError(
                "Split has no train rows.",
                how_to_fix="Check split fractions and group assignments.",
            )
        train_labels = np.array([SplitName.TRAIN.value] * int(train_mask.sum()))
        new_matrices = {}
        records: list[dict[str, Any]] = []
        for modality, matrix in self.matrices.items():
            # NaN-aware train-only z-score. Missing entries stay NaN so the
            # evaluator never scores imputed targets as if they were observed.
            scaler = TrainOnlyStandardScaler(name=f"{modality}_standard_scaler")
            scaler.fit(matrix[train_mask], train_labels)
            scaled = scaler.transform(matrix)
            scaled[self.missing[modality]] = np.nan
            new_matrices[modality] = scaled
            if scaler.provenance is None:
                raise RuntimeError("Preprocessor provenance missing after fit.")
            records.append(scaler.provenance.model_dump(mode="json"))
        return MultiOmicsBundle(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            observations=self.observations.copy(),
            matrices=new_matrices,
            feature_names=self.feature_names,
            missing=self.missing,
            sample_sheet=self.sample_sheet,
            provenance=self.provenance + records,
        )

    def imputed_matrix(self, modality: str) -> np.ndarray:
        """Train-mean fill of one modality for model features (not for scoring)."""

        values = self.matrices[modality]
        train_mask = self.observations["split"].to_numpy() == SplitName.TRAIN.value
        imputer = TrainOnlyImputer(name=f"{modality}_predict_imputer")
        imputer.fit(values[train_mask], np.array([SplitName.TRAIN.value] * int(train_mask.sum())))
        return imputer.transform(values)

    def to_mudata(self) -> md.MuData:
        """Convert to MuData. Each modality is an AnnData with aligned obs."""

        adatas: dict[str, ad.AnnData] = {}
        obs = self.observations.copy().set_index("observation_id")
        for modality, matrix in self.matrices.items():
            var = pd.DataFrame(index=pd.Index(self.feature_names[modality], name="feature_id"))
            adata = ad.AnnData(X=matrix.copy(), obs=obs.copy(), var=var)
            adata.layers["raw_aligned"] = matrix.copy()
            # Store provenance as JSON text. A list of dicts is not a valid
            # HDF5/AnnData uns payload and would fail on write.
            adata.uns["fit_split_provenance_json"] = json.dumps(self.provenance, default=str)
            adatas[modality] = adata
        mdata = md.MuData(adatas)
        mdata.uns["dataset_id"] = self.manifest.dataset_id
        mdata.uns["pairing_level"] = self.pairing_level.value
        mdata.uns["sampling_design"] = self.sampling_design.value
        mdata.uns["fit_split_provenance_json"] = json.dumps(self.provenance, default=str)
        return mdata

    def write_h5mu(self, path: Path) -> None:
        """Write MuData to ``path``. Never writes into the input directory."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_mudata().write_h5mu(path)
