"""Embedding-model registry and PriorBundle attachment.

Preferred implemented model: Uni-Mol (3D molecular ``cls_repr``).
The synthetic pathway one-hot is a CI fixture, not a foundation model.
ESM is registered and must raise until a sequence adapter exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omics_agent.errors import PriorError
from omics_agent.priors.unimol import (
    UNIMOL_DEFAULT_ROOT,
    UNIMOL_LICENSE,
    UNIMOL_PINNED_COMMIT,
    UniMolEmbeddingAdapter,
    UniMolReprFn,
)
from omics_agent.schemas.enums import EmbeddingModelName
from omics_agent.schemas.priors import EmbeddingModelConfig, EmbeddingSpec, PriorBundle

EmbeddingStatus = Literal["implemented", "registered_not_implemented", "fixture"]


@dataclass(frozen=True)
class EmbeddingCandidate:
    """One registered extractor. Status is not a quality ranking."""

    name: EmbeddingModelName
    preferred: bool
    input_kind: str
    status: EmbeddingStatus
    description: str


EMBEDDING_CANDIDATES: tuple[EmbeddingCandidate, ...] = (
    EmbeddingCandidate(
        name=EmbeddingModelName.UNIMOL,
        preferred=True,
        input_kind="smiles",
        status="implemented",
        description=(
            "Uni-Mol 3D molecular encoder (MIT, DP Technology). "
            "Requires an explicit feature→SMILES table."
        ),
    ),
    EmbeddingCandidate(
        name=EmbeddingModelName.ESM,
        preferred=False,
        input_kind="protein_sequence",
        status="registered_not_implemented",
        description="Protein language model. Registered; adapter is not implemented.",
    ),
    EmbeddingCandidate(
        name=EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT,
        preferred=False,
        input_kind="constructed_table",
        status="fixture",
        description=(
            "Deterministic pathway one-hot + sin/cos fixture. "
            "Not ESM, Geneformer, scGPT, or Uni-Mol."
        ),
    ),
)


def list_embedding_models() -> tuple[EmbeddingCandidate, ...]:
    """Preferred model first, then the rest of the registry."""

    return EMBEDDING_CANDIDATES


def preferred_embedding_model() -> EmbeddingCandidate:
    return EMBEDDING_CANDIDATES[0]


def load_smiles_map(path: Path) -> dict[tuple[str, str], str]:
    """Load ``modality, feature_id, smiles`` TSV. Does not invent missing rows."""

    if not path.is_file():
        raise PriorError(
            f"SMILES map not found: {path}",
            how_to_fix=(
                "Write a TSV with header modality, feature_id, smiles. "
                "See config/feature_smiles.example.tsv. "
                "Gene/protein IDs are not SMILES — do not guess structures."
            ),
        )
    mapping: dict[tuple[str, str], str] = {}
    header: list[str] | None = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cells = [cell.strip() for cell in raw.split("\t")]
        if header is None:
            header = [cell.lower() for cell in cells]
            missing = {"modality", "feature_id", "smiles"} - set(header)
            if missing:
                raise PriorError(
                    f"{path} is missing columns {sorted(missing)}.",
                    how_to_fix="Use a tab-separated header: modality, feature_id, smiles.",
                )
            continue
        if len(cells) < len(header):
            raise PriorError(
                f"{path}:{line_no} has {len(cells)} columns, expected {len(header)}.",
                how_to_fix="Every data row must have modality, feature_id, and smiles.",
            )
        row = dict(zip(header, cells, strict=False))
        key = (row["modality"], row["feature_id"])
        smiles = row["smiles"]
        if not smiles:
            continue
        if key in mapping and mapping[key] != smiles:
            raise PriorError(
                f"Conflicting SMILES for {key[0]}:{key[1]} in {path}.",
                how_to_fix="Keep one SMILES per (modality, feature_id). Do not guess which is correct.",
            )
        mapping[key] = smiles
    if header is None or not mapping:
        raise PriorError(
            f"{path} has no usable SMILES rows.",
            how_to_fix="Add at least one modality/feature_id/smiles row. Empty SMILES are ignored.",
        )
    return mapping


def apply_embedding_model(
    bundle: PriorBundle,
    config: EmbeddingModelConfig,
    *,
    experiment_dir: Path,
    repr_fn: UniMolReprFn | None = None,
) -> PriorBundle:
    """Replace ``bundle.embeddings`` according to ``config.name``."""

    if config.name is EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT:
        return bundle
    if config.name is EmbeddingModelName.ESM:
        raise PriorError(
            "Embedding model 'esm' is registered but not implemented.",
            how_to_fix=(
                "Use priors.embedding.name: unimol with a SMILES map, "
                "or synthetic_pathway_onehot for the CI fixture."
            ),
        )
    if config.name is EmbeddingModelName.UNIMOL:
        tables, spec = extract_unimol_embeddings(
            bundle.features,
            config,
            experiment_dir=experiment_dir,
            species_taxon_id=bundle.species_taxon_id,
            repr_fn=repr_fn,
        )
        return with_embeddings(bundle, tables, spec)
    raise PriorError(
        f"Unknown embedding model '{config.name}'.",
        how_to_fix=f"Choose one of: {[item.value for item in EmbeddingModelName]}.",
    )


def extract_unimol_embeddings(
    features: dict[str, list[str]],
    config: EmbeddingModelConfig,
    *,
    experiment_dir: Path,
    species_taxon_id: int,
    repr_fn: UniMolReprFn | None = None,
) -> tuple[dict[str, list[list[float]]], EmbeddingSpec]:
    """One Uni-Mol ``cls_repr`` per feature. Unmapped ids become zeros."""

    if config.smiles_map is None:
        raise PriorError(
            "Uni-Mol needs an explicit feature→SMILES table. "
            "Gene and protein identifiers are not SMILES and will not be guessed.",
            how_to_fix=(
                "Set priors.embedding.smiles_map to a TSV "
                "(modality, feature_id, smiles), e.g. config/feature_smiles.example.tsv. "
                "For the synthetic RNA/protein fixture, set "
                "priors.embedding.name: synthetic_pathway_onehot."
            ),
        )
    map_path = _resolve(experiment_dir, config.smiles_map)
    smiles_of = load_smiles_map(map_path)
    ordered: list[tuple[str, str, str]] = []
    unmapped: list[str] = []
    for modality, names in features.items():
        for name in names:
            smiles = smiles_of.get((modality, name))
            if smiles is None:
                unmapped.append(f"{modality}:{name}")
            else:
                ordered.append((modality, name, smiles))
    if not ordered:
        raise PriorError(
            "No feature in the PriorBundle has a SMILES in the map. "
            "Uni-Mol will not be applied to gene/protein IDs.",
            how_to_fix=(
                f"Mapped keys look like {sorted(smiles_of)[:8]}. "
                "Add rows for the declared features, or use synthetic_pathway_onehot."
            ),
        )

    unique: list[str] = []
    index: dict[str, int] = {}
    for _, _, smiles in ordered:
        if smiles not in index:
            index[smiles] = len(unique)
            unique.append(smiles)

    root = _resolve(experiment_dir, config.unimol_root)
    adapter = UniMolEmbeddingAdapter(
        root,
        variant=config.unimol_variant,
        size=config.unimol_size,
        repr_fn=repr_fn,
    )
    if repr_fn is None:
        commit = adapter.source_commit()
        if commit != UNIMOL_PINNED_COMMIT:
            raise PriorError(
                "Uni-Mol checkout is not the pinned commit "
                f"(HEAD={commit or 'unknown'}, pin={UNIMOL_PINNED_COMMIT}).",
                how_to_fix=(
                    f"Check out {UNIMOL_PINNED_COMMIT} in priors.embedding.unimol_root. "
                    "omics-agent will not run an unpinned external repo (rule 7). "
                    "CI injects a mock repr_fn and skips this pin check."
                ),
            )
    vectors = adapter.get_cls_repr(unique)
    dim = int(vectors.shape[1])
    by_feature = {
        (modality, name): vectors[index[smiles]].astype(float).tolist()
        for modality, name, smiles in ordered
    }
    tables: dict[str, list[list[float]]] = {}
    for modality, names in features.items():
        tables[modality] = [by_feature.get((modality, name), [0.0] * dim) for name in names]
    version = (
        f"{config.unimol_variant}-{config.unimol_size}"
        if config.unimol_variant == "unimolv2"
        else config.unimol_variant
    )
    commit = adapter.source_commit()
    spec = EmbeddingSpec(
        model_name=EmbeddingModelName.UNIMOL.value,
        model_version=version,
        training_data_description=(
            "Uni-Mol 3D molecular encoder (DP Technology). "
            "Input is SMILES. This is not a gene or protein language model."
        ),
        license=UNIMOL_LICENSE,
        extraction_layer=config.extraction_layer,
        dim=dim,
        species_taxon_id=species_taxon_id,
        source_repo=str(root),
        source_commit=commit,
        input_kind="smiles",
        n_unmapped=len(unmapped),
        caveat=(
            "Frozen embeddings are an auxiliary prior, not a causal map. "
            "A foundation model may already have seen public literature, so later "
            "literature consistency is not an independent test. "
            "Uni-Mol is a small-molecule model; do not treat gene/protein IDs as SMILES."
        ),
    )
    return tables, spec


def with_embeddings(
    bundle: PriorBundle,
    tables: dict[str, list[list[float]]],
    spec: EmbeddingSpec,
) -> PriorBundle:
    """Swap the frozen table and record provenance. Graph/pathways stay put."""

    drop = (
        "Frozen embeddings are a traceable linear mix",
        "they are not ESM/Geneformer weights.",
    )
    notes = [note for note in bundle.notes if not any(part in note for part in drop)]
    notes.append(
        f"Frozen embeddings from {spec.model_name} {spec.model_version} "
        f"(layer={spec.extraction_layer}, dim={spec.dim}, "
        f"commit={spec.source_commit or 'unknown'})."
    )
    if spec.n_unmapped:
        notes.append(
            f"{spec.n_unmapped} features had no SMILES and were zero-filled; "
            "they were not guessed from gene/protein identifiers."
        )
    if spec.source_commit and spec.source_commit != UNIMOL_PINNED_COMMIT:
        notes.append(
            f"Uni-Mol checkout HEAD {spec.source_commit} differs from "
            f"the pinned adapter commit {UNIMOL_PINNED_COMMIT}."
        )
    if spec.model_name == EmbeddingModelName.UNIMOL.value:
        notes.append(f"Uni-Mol license: {UNIMOL_LICENSE}. Default root: {UNIMOL_DEFAULT_ROOT}.")
    return bundle.model_copy(update={"embeddings": tables, "embedding_spec": spec, "notes": notes})


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
