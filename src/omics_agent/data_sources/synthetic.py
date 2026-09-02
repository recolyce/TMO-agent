"""Small bulk dual-omics ODE generator with a known lagged gene→protein graph.

The simulator is a linear delayed system integrated on a fine grid, then
sampled at irregular observation times. It is an engineering fixture, not a
biological claim.

Two designs are generated from the same latent dynamics:

* longitudinal: the same subject is observed at every time.
* repeated_cross_sectional: each animal is observed at exactly one time.

Known edges are written to ``true_edges.parquet`` so later interpretation
tests can recover them. Random missingness is MCAR and recorded in the
matrices as NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

from omics_agent.errors import SchemaError
from omics_agent.schemas.enums import SamplingDesign

GENERATOR_VERSION = "synthetic_linear_delay_v1"
TIME_POINTS = (0.0, 1.0, 2.0, 4.0, 8.0)
GENE_NAMES = [f"G{i:02d}" for i in range(1, 13)]
PROTEIN_NAMES = [f"P{i:02d}" for i in range(1, 9)]
# (gene_index, protein_index, lag_time, weight)
TRUE_EDGES: tuple[tuple[int, int, float, float], ...] = (
    (0, 0, 1.0, 1.20),
    (1, 0, 1.0, 0.80),
    (2, 1, 2.0, 1.10),
    (3, 2, 1.0, 0.90),
    (4, 3, 1.0, 1.00),
    (5, 4, 2.0, 0.85),
    (6, 5, 1.0, 1.15),
    (7, 6, 1.0, 0.75),
    (8, 7, 2.0, 1.05),
    (9, 2, 1.0, 0.55),
)


def simulate_trajectory(
    rng: np.random.Generator,
    *,
    condition: str,
    n_genes: int,
    n_proteins: int,
    t_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one latent trajectory and return RNA/protein at ``t_obs``."""

    decay_r = 0.35
    decay_p = 0.25
    basal = 0.15 + 0.05 * rng.standard_normal(n_genes)
    cond_effect = np.zeros(n_genes)
    if condition == "treated":
        cond_effect[:4] = 0.8
        cond_effect[4:8] = -0.4
    rna0 = np.clip(1.0 + 0.3 * rng.standard_normal(n_genes), 0.1, None)
    prot0 = np.clip(0.6 + 0.2 * rng.standard_normal(n_proteins), 0.05, None)
    t_grid = np.linspace(0.0, float(t_obs.max()), 161)
    rna_hist = np.zeros((t_grid.size, n_genes))
    rna_hist[0] = rna0
    # Closed-form RNA (no protein feedback) so the lag is identifiable.
    for i, _t in enumerate(t_grid[1:], start=1):
        dt = t_grid[i] - t_grid[i - 1]
        drive = basal + cond_effect
        rna_hist[i] = rna_hist[i - 1] + dt * (drive - decay_r * rna_hist[i - 1])
        rna_hist[i] = np.clip(rna_hist[i], 0.0, None)

    def protein_ode(t: float, prot: np.ndarray) -> np.ndarray:
        deriv = -decay_p * prot
        for gene_i, prot_i, lag, weight in TRUE_EDGES:
            rna_lag = float(np.interp(t - lag, t_grid, rna_hist[:, gene_i]))
            deriv[prot_i] += weight * max(rna_lag, 0.0)
        return deriv

    solved = solve_ivp(
        protein_ode,
        (0.0, float(t_obs.max())),
        prot0,
        t_eval=t_obs,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    if not solved.success:
        raise RuntimeError(f"Synthetic ODE failed: {solved.message}")
    rna_obs = np.vstack(
        [np.array([np.interp(t, t_grid, rna_hist[:, g]) for g in range(n_genes)]) for t in t_obs]
    )
    prot_obs = solved.y.T
    return rna_obs, prot_obs


def _apply_missing(
    values: np.ndarray, rng: np.random.Generator, rate: float
) -> np.ndarray:
    out = values.astype(float).copy()
    mask = rng.random(out.shape) < rate
    # Keep at least one observed value per row so a sample is not empty.
    for i in range(out.shape[0]):
        row = mask[i]
        if row.all():
            row[int(rng.integers(0, out.shape[1]))] = False
        out[i, row] = np.nan
    return out


def _write_manifest(
    path: Path,
    *,
    dataset_id: str,
    title: str,
    sampling_design: SamplingDesign,
    unit: str,
    files: list[dict[str, Any]],
) -> None:
    longitudinal = sampling_design is SamplingDesign.LONGITUDINAL
    payload = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "title": title,
        "source": {
            "type": "synthetic",
            "generator": GENERATOR_VERSION,
            "local_dir": ".",
        },
        "license": {
            "name": "CC0-1.0",
            "redistributable": True,
            "notes": "Synthetic fixture; not biological data.",
        },
        "organism": {"taxon_id": 9606, "name": "Homo sapiens"},
        "design": {
            "unit_of_independence": unit,
            "sampling_design": sampling_design.value,
            "longitudinal": longitudinal,
            "pairing_level": "same_biospecimen",
            "paired_modalities": True,
            "time_unit": "hour",
        },
        "modalities": {
            "rna": {
                "assay": "synthetic",
                "value_type": "synthetic_abundance",
                "feature_id_type": "synthetic_gene",
                "n_features": len(GENE_NAMES),
            },
            "protein": {
                "assay": "synthetic",
                "value_type": "synthetic_abundance",
                "feature_id_type": "synthetic_protein",
                "n_features": len(PROTEIN_NAMES),
            },
        },
        "files": files,
        "sample_sheet": "samples.tsv",
        "human_review": {
            "status": "approved",
            "unresolved": [],
            "reviewer": "synthetic-generator",
            "notes": "Generated with known lagged edges; safe to train.",
        },
        "notes": (
            "Engineering fixture with a known sparse gene→protein lagged network. "
            "Not a biological discovery dataset."
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def generate_synthetic_dataset(
    output_dir: Path,
    *,
    design: SamplingDesign,
    seed: int = 20260901,
    rna_missing_rate: float = 0.05,
    protein_missing_rate: float = 0.10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write one synthetic dataset directory.

    Parameters
    ----------
    output_dir:
        Destination. Created if needed. Never writes into ``data/raw``.
    design:
        Longitudinal or repeated cross-sectional.
    seed:
        Full RNG seed for trajectories, missingness, and subject noise.
    dry_run:
        If true, return the plan without writing files.
    """

    output_dir = Path(output_dir)
    plan: dict[str, Any] = {
        "output_dir": str(output_dir),
        "design": design.value,
        "seed": seed,
        "n_genes": len(GENE_NAMES),
        "n_proteins": len(PROTEIN_NAMES),
        "time_points": list(TIME_POINTS),
        "n_true_edges": len(TRUE_EDGES),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_obs = np.asarray(TIME_POINTS, dtype=float)
    conditions = ("control", "treated")

    sample_rows: list[dict[str, Any]] = []
    rna_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []

    if design is SamplingDesign.LONGITUDINAL:
        n_subjects = 12
        batches = ("batchA", "batchB")
        for s_i in range(n_subjects):
            subject = f"S{s_i + 1:02d}"
            condition = conditions[s_i % 2]
            batch = batches[s_i % 2]
            rna, prot = simulate_trajectory(
                rng, condition=condition, n_genes=len(GENE_NAMES), n_proteins=len(PROTEIN_NAMES), t_obs=t_obs
            )
            rna = rna + 0.05 * rng.standard_normal(rna.shape)
            prot = prot + 0.05 * rng.standard_normal(prot.shape)
            if batch == "batchB":
                rna = rna + 0.15
                prot = prot + 0.10
            rna = _apply_missing(rna, rng, rna_missing_rate)
            prot = _apply_missing(prot, rng, protein_missing_rate)
            for t_i, t in enumerate(t_obs):
                biospecimen = f"{subject}_t{t_i}"
                observation = biospecimen
                for modality, matrix, names, file_id in (
                    ("rna", rna, GENE_NAMES, "rna.parquet"),
                    ("protein", prot, PROTEIN_NAMES, "protein.parquet"),
                ):
                    sample_id = f"{biospecimen}_{modality}"
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "observation_id": observation,
                            "experimental_unit_id": subject,
                            "subject_id": subject,
                            "biospecimen_id": biospecimen,
                            "time": float(t),
                            "time_unit": "hour",
                            "condition": condition,
                            "batch": batch,
                            "modality": modality,
                            "file_id": file_id,
                            "replicate_type": "biological",
                        }
                    )
                    values = matrix[t_i]
                    record: dict[str, Any] = {"sample_id": sample_id}
                    record.update({name: float(values[j]) for j, name in enumerate(names)})
                    if modality == "rna":
                        rna_rows.append(record)
                    else:
                        protein_rows.append(record)
        dataset_id = "synthetic_longitudinal_rna_protein"
        title = "Synthetic longitudinal bulk RNA+protein (known lagged edges)"
        unit = "subject"
    elif design is SamplingDesign.REPEATED_CROSS_SECTIONAL:
        rcs_batches = ("expA", "expB", "expC")
        reps_per_cell = 2
        animal_i = 0
        for batch in rcs_batches:
            for condition in conditions:
                for t_i, t in enumerate(t_obs):
                    for _rep in range(reps_per_cell):
                        animal_i += 1
                        unit_id = f"A{animal_i:03d}"
                        rna, prot = simulate_trajectory(
                            rng,
                            condition=condition,
                            n_genes=len(GENE_NAMES),
                            n_proteins=len(PROTEIN_NAMES),
                            t_obs=t_obs,
                        )
                        # One animal, one time: keep only that slice.
                        rna_t = rna[t_i : t_i + 1] + 0.05 * rng.standard_normal((1, len(GENE_NAMES)))
                        prot_t = prot[t_i : t_i + 1] + 0.05 * rng.standard_normal((1, len(PROTEIN_NAMES)))
                        if batch == "expB":
                            rna_t = rna_t + 0.12
                            prot_t = prot_t + 0.08
                        if batch == "expC":
                            rna_t = rna_t - 0.08
                            prot_t = prot_t - 0.05
                        rna_t = _apply_missing(rna_t, rng, rna_missing_rate)
                        prot_t = _apply_missing(prot_t, rng, protein_missing_rate)
                        biospecimen = f"{unit_id}_bs"
                        observation = biospecimen
                        for modality, matrix, names, file_id in (
                            ("rna", rna_t, GENE_NAMES, "rna.parquet"),
                            ("protein", prot_t, PROTEIN_NAMES, "protein.parquet"),
                        ):
                            sample_id = f"{biospecimen}_{modality}"
                            sample_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "observation_id": observation,
                                    "experimental_unit_id": unit_id,
                                    "subject_id": unit_id,
                                    "biospecimen_id": biospecimen,
                                    "time": float(t),
                                    "time_unit": "hour",
                                    "condition": condition,
                                    "batch": batch,
                                    "modality": modality,
                                    "file_id": file_id,
                                    "replicate_type": "biological",
                                }
                            )
                            values = matrix[0]
                            rcs_record: dict[str, Any] = {"sample_id": sample_id}
                            rcs_record.update({name: float(values[j]) for j, name in enumerate(names)})
                            record = rcs_record
                            if modality == "rna":
                                rna_rows.append(record)
                            else:
                                protein_rows.append(record)
        dataset_id = "synthetic_rcs_rna_protein"
        title = "Synthetic repeated cross-sectional bulk RNA+protein (known lagged edges)"
        unit = "experiment_batch"
    else:
        raise SchemaError(
            f"Unsupported synthetic design: {design}",
            how_to_fix="Use longitudinal or repeated_cross_sectional.",
        )

    samples = pd.DataFrame(sample_rows)
    rna_df = pd.DataFrame(rna_rows)
    protein_df = pd.DataFrame(protein_rows)
    edges = pd.DataFrame(
        [
            {
                "source_feature": GENE_NAMES[g],
                "target_feature": PROTEIN_NAMES[p],
                "lag_hours": lag,
                "weight": weight,
                "edge_type": "synthetic_gene_to_protein_lag",
            }
            for g, p, lag, weight in TRUE_EDGES
        ]
    )

    samples_path = output_dir / "samples.tsv"
    rna_path = output_dir / "rna.parquet"
    protein_path = output_dir / "protein.parquet"
    edges_path = output_dir / "true_edges.parquet"
    samples.to_csv(samples_path, sep="\t", index=False)
    rna_df.to_parquet(rna_path, index=False)
    protein_df.to_parquet(protein_path, index=False)
    edges.to_parquet(edges_path, index=False)

    files = [
        {"path": "rna.parquet", "sha256": None, "modality": "rna", "role": "matrix"},
        {"path": "protein.parquet", "sha256": None, "modality": "protein", "role": "matrix"},
        {"path": "true_edges.parquet", "sha256": None, "modality": "protein", "role": "edges"},
    ]
    _write_manifest(
        output_dir / "dataset.yaml",
        dataset_id=dataset_id,
        title=title,
        sampling_design=design,
        unit=unit,
        files=files,
    )
    # Fill checksums after write so the manifest is auditable.
    from omics_agent.hashing import sha256_file

    manifest_path = output_dir / "dataset.yaml"
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    for item in payload["files"]:
        item["sha256"] = sha256_file(output_dir / item["path"])
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan.update(
        {
            "dataset_id": dataset_id,
            "n_sample_rows": int(len(samples)),
            "n_experimental_units": int(samples["experimental_unit_id"].nunique()),
            "n_times": int(samples["time"].nunique()),
            "written": [
                str(samples_path),
                str(rna_path),
                str(protein_path),
                str(edges_path),
                str(manifest_path),
            ],
        }
    )
    return plan
