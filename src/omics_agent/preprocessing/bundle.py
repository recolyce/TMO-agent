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

from omics_agent.errors import ManifestError, SchemaError, SplitLeakageError
from omics_agent.preprocessing.qc import per_feature_qc, per_sample_qc
from omics_agent.preprocessing.scalers import TrainOnlyImputer, TrainOnlyStandardScaler
from omics_agent.preprocessing.strategies import default_config_for, normalize
from omics_agent.schemas.dataset import DatasetManifest
from omics_agent.schemas.enums import PairingLevel, SamplingDesign, SplitName
from omics_agent.schemas.preprocess import ModalityPreprocessConfig


@dataclass
class MultiOmicsBundle:
    """In-memory bulk multi-omics object used by split, models, and evaluator.

    ``observations`` has one row per ``observation_id`` (biospecimen/time).
    ``matrices[modality]`` is observations × features, aligned to that table.
    ``missing[modality]`` is True where the value is missing (original NaN,
    or a protein zero intensity reclassified as not-quantified).
    ``layers[modality]`` holds ``raw`` / ``normalized`` / ``scaled`` after
    :meth:`apply_assay_preprocessing`.
    """

    manifest: DatasetManifest
    manifest_path: Path
    observations: pd.DataFrame
    matrices: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    missing: dict[str, np.ndarray]
    sample_sheet: pd.DataFrame
    provenance: list[dict[str, Any]] = field(default_factory=list)
    layers: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

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
        pairs = split_frame[["experimental_unit_id", "split"]].astype(str).drop_duplicates()
        conflict = pairs["experimental_unit_id"][pairs["experimental_unit_id"].duplicated()]
        if not conflict.empty:
            leaked = sorted(set(conflict.tolist()))
            raise SplitLeakageError(
                f"experimental_unit_id has conflicting split labels: {leaked[:12]}. "
                "Last-wins assignment would hide leakage.",
                how_to_fix="One split per experimental_unit_id. Re-run omics-agent split.",
            )
        mapping = dict(
            zip(
                pairs["experimental_unit_id"],
                pairs["split"],
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
            layers={
                mod: {name: values.copy() for name, values in mod_layers.items()}
                for mod, mod_layers in self.layers.items()
            },
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
            layers={
                mod: {name: values[idx] for name, values in mod_layers.items()}
                for mod, mod_layers in self.layers.items()
            },
        )

    def apply_assay_preprocessing(
        self, configs: dict[str, ModalityPreprocessConfig] | None = None
    ) -> MultiOmicsBundle:
        """Normalize per assay, then fit the scaler on train rows only.

        For each modality this produces three layers:

        - ``raw``: the aligned input matrix, untouched.
        - ``normalized``: stateless per-sample math (CPM/log1p for counts,
          zeros-to-missing plus log2 for protein intensity, pass-through for
          log-like values). Learns nothing across samples.
        - ``scaled``: train-only z-score of ``normalized``. Missing entries
          stay NaN; protein missingness is never filled with 0.

        The default strategy comes from the manifest ``value_type``;
        ``configs`` overrides it per modality.
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
        new_matrices: dict[str, np.ndarray] = {}
        new_missing: dict[str, np.ndarray] = {}
        new_layers: dict[str, dict[str, np.ndarray]] = {}
        records: list[dict[str, Any]] = []
        for modality, matrix in self.matrices.items():
            config = (configs or {}).get(modality) or default_config_for(
                modality, self.manifest.modalities[modality]
            )
            normalized, stateless = normalize(modality, matrix, config)
            missing = np.isnan(normalized)
            records.append(stateless.model_dump(mode="json"))
            if config.scale:
                # NaN-aware train-only z-score. Missing entries stay NaN so the
                # evaluator never scores imputed targets as if they were observed.
                scaler = TrainOnlyStandardScaler(name=f"{modality}_standard_scaler")
                scaler.fit(normalized[train_mask], train_labels)
                scaled = scaler.transform(normalized)
                scaled[missing] = np.nan
                if scaler.provenance is None:
                    raise RuntimeError("Preprocessor provenance missing after fit.")
                records.append(scaler.provenance.model_dump(mode="json"))
            else:
                scaled = normalized.copy()
            new_matrices[modality] = scaled
            new_missing[modality] = missing
            new_layers[modality] = {
                "raw": matrix.copy(),
                "normalized": normalized.copy(),
                "scaled": scaled.copy(),
            }
        return MultiOmicsBundle(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            observations=self.observations.copy(),
            matrices=new_matrices,
            feature_names=self.feature_names,
            missing=new_missing,
            sample_sheet=self.sample_sheet,
            provenance=self.provenance + records,
            layers=new_layers,
        )

    def apply_train_only_preprocessing(self) -> MultiOmicsBundle:
        """Manifest-default assay preprocessing. Kept for milestone-1 callers."""

        return self.apply_assay_preprocessing(None)

    def imputed_matrix(self, modality: str) -> np.ndarray:
        """Train-mean fill of one modality for model features (not for scoring)."""

        values = self.matrices[modality]
        train_mask = self.observations["split"].to_numpy() == SplitName.TRAIN.value
        imputer = TrainOnlyImputer(name=f"{modality}_predict_imputer")
        imputer.fit(values[train_mask], np.array([SplitName.TRAIN.value] * int(train_mask.sum())))
        return imputer.transform(values)

    def to_mudata(self) -> md.MuData:
        """Convert to MuData with raw/normalized/scaled layers and QC columns.

        ``X`` is the scaled matrix used by models. QC metrics live in
        ``obs`` / ``var`` with a ``qc_`` prefix and describe the raw layer.
        """

        adatas: dict[str, ad.AnnData] = {}
        obs = self.observations.copy().set_index("observation_id")
        for modality, matrix in self.matrices.items():
            var = pd.DataFrame(index=pd.Index(self.feature_names[modality], name="feature_id"))
            adata = ad.AnnData(X=matrix.copy(), obs=obs.copy(), var=var)
            mod_layers = self.layers.get(modality) or {"raw": matrix.copy()}
            for layer_name, values in mod_layers.items():
                adata.layers[layer_name] = values.copy()
            raw = mod_layers.get("raw", matrix)
            missing = self.missing[modality]
            sample_qc = per_sample_qc(raw, missing)
            for column in sample_qc.columns:
                adata.obs[column] = sample_qc[column].to_numpy()
            feature_qc = per_feature_qc(raw, missing)
            for column in feature_qc.columns:
                adata.var[column] = feature_qc[column].to_numpy()
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
