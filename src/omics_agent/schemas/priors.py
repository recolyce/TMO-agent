"""Versioned PriorBundle and the five mandatory ablation arms.

STRING functional associations keep their evidence channel, score, version,
and species. They are never labelled physical PPI or causal. Attribution
of a prior edge is not causation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from omics_agent.errors import PriorError, SchemaError
from omics_agent.hashing import hash_mapping
from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import EdgeType, EmbeddingModelName, PriorAblation

_STRING_ALIASES = ("string", "string-db", "stringdb", "string_db")
_DIRECTED_TYPES = {EdgeType.GENE_REGULATION, EdgeType.GENE_PROTEIN_CODING}
_UNDIRECTED_TYPES = {
    EdgeType.PHYSICAL_PPI,
    EdgeType.FUNCTIONAL_ASSOCIATION,
    EdgeType.PATHWAY_COMEMBERSHIP,
}


class PriorEdge(StrictModel):
    """One prior edge. ``is_causal`` is literally False — priors are not causes."""

    source_id: str
    target_id: str
    source_modality: str
    target_modality: str
    edge_type: EdgeType
    directed: bool
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    species_taxon_id: int = Field(ge=1)
    source_name: str
    source_version: str
    randomized: bool = False
    is_causal: Literal[False] = False

    @model_validator(mode="after")
    def _scientific_labels(self) -> Self:
        source = self.source_name.strip().lower().replace(" ", "")
        is_string = any(alias in source for alias in _STRING_ALIASES)
        if is_string and self.edge_type is EdgeType.PHYSICAL_PPI:
            raise SchemaError(
                "STRING functional association cannot be labelled physical_ppi. "
                "STRING scores mix evidence channels; they are not physical contacts.",
                how_to_fix=(
                    "Set edge_type: functional_association and keep evidence, score, "
                    "source_version, and species_taxon_id. Physical PPI must come from "
                    "a physical-interaction source (e.g. IntAct), never from STRING."
                ),
            )
        if is_string and self.edge_type is EdgeType.GENE_REGULATION:
            raise SchemaError(
                "STRING is not a gene-regulatory network. Do not label it gene_regulation.",
                how_to_fix="Use functional_association, or a dedicated GRN source for regulation.",
            )
        if self.edge_type in _DIRECTED_TYPES and not self.directed:
            raise SchemaError(
                f"edge_type '{self.edge_type.value}' is directed but directed=false.",
                how_to_fix="Set directed: true for gene_regulation and gene_protein_coding.",
            )
        if self.edge_type in _UNDIRECTED_TYPES and self.directed:
            raise SchemaError(
                f"edge_type '{self.edge_type.value}' is undirected but directed=true.",
                how_to_fix="Set directed: false. PPI / functional / pathway membership have no arrow.",
            )
        return self


class PathwayMembership(StrictModel):
    """One Reactome (or Reactome-like) pathway and its member features."""

    pathway_id: str
    pathway_name: str
    source_name: str = "Reactome"
    source_version: str
    species_taxon_id: int = Field(ge=1)
    member_ids: list[str] = Field(min_length=1)
    member_modality: str


class EmbeddingSpec(StrictModel):
    """Provenance of a frozen feature embedding table.

    Foundation-model embeddings may already contain public knowledge, so a
    later literature check is not an independent validation. That caveat is
    recorded here, not invented at report time.
    """

    model_name: str
    model_version: str
    training_data_description: str
    license: str
    extraction_layer: str
    dim: int = Field(ge=1)
    frozen: Literal[True] = True
    species_taxon_id: int = Field(ge=1)
    source_repo: str | None = None
    source_commit: str | None = None
    input_kind: Literal["smiles", "constructed_table", "protein_sequence"] = "constructed_table"
    n_unmapped: int = Field(default=0, ge=0)
    caveat: str = (
        "Frozen embeddings are an auxiliary prior, not a causal map. "
        "A foundation model may already have seen public literature, so later "
        "literature consistency is not an independent test."
    )


class PriorBundle(StrictModel):
    """Versioned, hashed prior artifact consumed by every ablation arm."""

    schema_version: str = "1.0"
    bundle_id: str = Field(min_length=1)
    bundle_version: str = Field(min_length=1)
    species_taxon_id: int = Field(ge=1)
    created_at: str
    features: dict[str, list[str]]
    edges: list[PriorEdge] = Field(default_factory=list)
    pathways: list[PathwayMembership] = Field(default_factory=list)
    embeddings: dict[str, list[list[float]]] = Field(default_factory=dict)
    embedding_spec: EmbeddingSpec | None = None
    license: str
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _embedding_alignment(self) -> Self:
        if self.embeddings:
            if self.embedding_spec is None:
                raise SchemaError(
                    "PriorBundle has embeddings but no embedding_spec.",
                    how_to_fix="Record model_name, model_version, license, and extraction_layer.",
                )
            for modality, rows in self.embeddings.items():
                names = self.features.get(modality)
                if names is None:
                    raise SchemaError(
                        f"Embeddings for unknown modality '{modality}'.",
                        how_to_fix=f"Declared modalities: {sorted(self.features)}.",
                    )
                if len(rows) != len(names):
                    raise SchemaError(
                        f"Embeddings for '{modality}' have {len(rows)} rows, "
                        f"but {len(names)} features are declared.",
                        how_to_fix="One embedding row per feature, same order as features[modality].",
                    )
                for i, vec in enumerate(rows):
                    if len(vec) != self.embedding_spec.dim:
                        raise SchemaError(
                            f"Embedding row {i} of '{modality}' has dim {len(vec)}, "
                            f"expected {self.embedding_spec.dim}.",
                            how_to_fix="Every vector must match embedding_spec.dim.",
                        )
        return self

    def content_hash(self) -> str:
        return hash_mapping(self.model_dump(mode="json"))


class EmbeddingModelConfig(StrictModel):
    """Which frozen embedding extractor to run. Uni-Mol is preferred.

    Uni-Mol is a 3D small-molecule encoder. It requires an explicit
    feature→SMILES table. Gene or protein identifiers are never treated
    as SMILES.
    """

    name: EmbeddingModelName = EmbeddingModelName.UNIMOL
    smiles_map: Path | None = None
    unimol_root: Path = Path("/root/workspace/Uni-Mol")
    unimol_variant: Literal["unimolv1", "unimolv2"] = "unimolv1"
    unimol_size: str = "84m"
    extraction_layer: Literal["cls_repr"] = "cls_repr"


class PriorAblationConfig(StrictModel):
    """How the five-arm prior comparison is run.

    Split, evaluator, primary metric, and HPO budget are *not* set here —
    they come from the experiment. Every arm sees the same locked split.
    """

    bundle: Path | None = None
    seeds: list[int] = Field(default_factory=lambda: [20260901, 20260902, 20260903])
    configs: list[PriorAblation] = Field(default_factory=lambda: list(PriorAblation))
    graph_weight: float = Field(default=0.1, ge=0.0)
    embedding_proj_dim: int = Field(default=16, ge=4, le=64)
    share_hpo: bool = True
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)


class AblationFlags(StrictModel):
    """Resolved switches for one named arm. Combined = all three priors."""

    ablation: PriorAblation
    use_pathway: bool
    use_graph: bool
    use_embedding: bool
    randomize_graph: bool


def flags_for(ablation: PriorAblation) -> AblationFlags:
    """Map a named arm onto the three independently ablatable priors."""

    return AblationFlags(
        ablation=ablation,
        use_pathway=ablation is PriorAblation.COMBINED,
        use_graph=ablation
        in {PriorAblation.GRAPH_ONLY, PriorAblation.COMBINED, PriorAblation.RANDOM_GRAPH},
        use_embedding=ablation in {PriorAblation.EMBEDDING_ONLY, PriorAblation.COMBINED},
        randomize_graph=ablation is PriorAblation.RANDOM_GRAPH,
    )


def load_prior_bundle(path: Path) -> PriorBundle:
    """Load and validate a PriorBundle YAML or JSON file."""

    import json

    import yaml

    if not path.is_file():
        raise PriorError(
            f"Prior bundle not found: {path}",
            how_to_fix="Pass a PriorBundle YAML written by omics-agent, or omit --priors to build the synthetic fixture.",
        )
    text = path.read_text(encoding="utf-8")
    payload: Any = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise PriorError(
            f"{path} is not a mapping.",
            how_to_fix="Start from config/priors.example.yaml.",
        )
    return PriorBundle.model_validate(payload)
