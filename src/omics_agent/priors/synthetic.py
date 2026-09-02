"""Toy PriorBundle aligned to the synthetic bulk generator.

This is an engineering fixture: Reactome-like modules, STRING-style
functional associations (never labelled physical/causal), a few IntAct
physical PPI edges, gene->protein coding edges, and a frozen traceable
embedding table. It is not a biological claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from omics_agent.data_sources.synthetic import GENE_NAMES, PROTEIN_NAMES, TRUE_EDGES
from omics_agent.schemas.enums import EdgeType
from omics_agent.schemas.priors import (
    EmbeddingSpec,
    PathwayMembership,
    PriorBundle,
    PriorEdge,
)

BUNDLE_VERSION = "synthetic_priors_v1"
TAXON = 9606  # human-shaped ids; the biology is still synthetic


def build_synthetic_prior_bundle(
    *,
    rna_features: list[str] | None = None,
    protein_features: list[str] | None = None,
    seed: int = 20260901,
    embeddings: dict[str, list[list[float]]] | None = None,
    embedding_spec: EmbeddingSpec | None = None,
) -> PriorBundle:
    """Build a versioned bundle. Feature names default to the generator panel."""

    rna = list(rna_features or GENE_NAMES)
    protein = list(protein_features or PROTEIN_NAMES)
    pathways = _reactome_like(rna, protein)
    edges = _edges(rna, protein)
    if embeddings is None:
        embeddings, spec = _frozen_embeddings(rna, protein, pathways, seed=seed)
    else:
        if embedding_spec is None:
            raise ValueError("embedding_spec is required when embeddings are supplied.")
        spec = embedding_spec
    return PriorBundle(
        bundle_id="synthetic_dual_omics",
        bundle_version=BUNDLE_VERSION,
        species_taxon_id=TAXON,
        created_at="2026-09-01T00:00:00+00:00",
        features={"rna": rna, "protein": protein},
        edges=edges,
        pathways=pathways,
        embeddings=embeddings,
        embedding_spec=spec,
        license="CC0-1.0 (synthetic fixture)",
        notes=[
            "Synthetic fixture for prior ablations. Not a Reactome or STRING extract.",
            "STRING-style edges are functional_association with evidence/score/version; "
            "they are not physical PPI and not causal.",
            "Frozen embeddings are a traceable linear mix of pathway one-hots; "
            "they are not ESM/Geneformer weights.",
        ],
    )


def write_synthetic_prior_bundle(path: Path, bundle: PriorBundle | None = None) -> PriorBundle:
    """Write YAML and return the bundle (hash is content-addressed, not the file)."""

    doc = bundle or build_synthetic_prior_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = doc.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return doc


def _reactome_like(rna: list[str], protein: list[str]) -> list[PathwayMembership]:
    groups = (
        ("R-HSA-SYN-1", "synthetic_module_a", rna[0:4], protein[0:3]),
        ("R-HSA-SYN-2", "synthetic_module_b", rna[4:8], protein[3:6]),
        ("R-HSA-SYN-3", "synthetic_module_c", rna[8:12], protein[6:8]),
    )
    out: list[PathwayMembership] = []
    for pid, name, genes, prots in groups:
        if genes:
            out.append(
                PathwayMembership(
                    pathway_id=pid,
                    pathway_name=name,
                    source_name="Reactome",
                    source_version="synthetic_v1",
                    species_taxon_id=TAXON,
                    member_ids=list(genes),
                    member_modality="rna",
                )
            )
        if prots:
            out.append(
                PathwayMembership(
                    pathway_id=f"{pid}-PROT",
                    pathway_name=f"{name}_protein",
                    source_name="Reactome",
                    source_version="synthetic_v1",
                    species_taxon_id=TAXON,
                    member_ids=list(prots),
                    member_modality="protein",
                )
            )
    return out


def _edges(rna: list[str], protein: list[str]) -> list[PriorEdge]:
    edges: list[PriorEdge] = []
    # Gene->protein coding / lag edges from the simulator, labelled as coding
    # (directed, not causal in the scientific sense — is_causal is False).
    for gene_i, prot_i, _lag, weight in TRUE_EDGES:
        if gene_i >= len(rna) or prot_i >= len(protein):
            continue
        edges.append(
            PriorEdge(
                source_id=rna[gene_i],
                target_id=protein[prot_i],
                source_modality="rna",
                target_modality="protein",
                edge_type=EdgeType.GENE_PROTEIN_CODING,
                directed=True,
                score=float(min(1.0, abs(weight) / 1.2)),
                evidence=["synthetic_lag_generator"],
                species_taxon_id=TAXON,
                source_name="synthetic_generator",
                source_version="synthetic_linear_delay_v1",
            )
        )
    # STRING-style functional associations among genes that share a module.
    # Evidence channels and score are kept; edge_type is functional_association.
    for i in range(0, min(len(rna), 12), 4):
        block = rna[i : i + 4]
        for a, b in zip(block, block[1:], strict=False):
            edges.append(
                PriorEdge(
                    source_id=a,
                    target_id=b,
                    source_modality="rna",
                    target_modality="rna",
                    edge_type=EdgeType.FUNCTIONAL_ASSOCIATION,
                    directed=False,
                    score=0.72,
                    evidence=["coexpression", "textmining"],
                    species_taxon_id=TAXON,
                    source_name="STRING",
                    source_version="12.0-synthetic",
                )
            )
    # A handful of physical PPI from a physical-interaction source — not STRING.
    for a, b in (("P01", "P02"), ("P05", "P06")):
        if a in protein and b in protein:
            edges.append(
                PriorEdge(
                    source_id=a,
                    target_id=b,
                    source_modality="protein",
                    target_modality="protein",
                    edge_type=EdgeType.PHYSICAL_PPI,
                    directed=False,
                    score=0.80,
                    evidence=["affinity_chromatography"],
                    species_taxon_id=TAXON,
                    source_name="IntAct",
                    source_version="synthetic_v1",
                )
            )
    return edges


def _frozen_embeddings(
    rna: list[str],
    protein: list[str],
    pathways: list[PathwayMembership],
    *,
    seed: int,
) -> tuple[dict[str, list[list[float]]], EmbeddingSpec]:
    """Traceable pathway-one-hot embeddings. Not a foundation-model extract."""

    dim = 8
    rng = np.random.default_rng(seed)
    pathway_ids = sorted({item.pathway_id for item in pathways})
    index = {pid: i for i, pid in enumerate(pathway_ids)}
    tables: dict[str, list[list[float]]] = {}
    for modality, names in (("rna", rna), ("protein", protein)):
        rows = []
        members = [item for item in pathways if item.member_modality == modality]
        for k, name in enumerate(names):
            vec = np.zeros(dim, dtype=np.float64)
            for item in members:
                if name in item.member_ids:
                    slot = index[item.pathway_id] % (dim - 2)
                    vec[slot] = 1.0
            vec[-2] = np.sin(0.3 * k)
            vec[-1] = np.cos(0.3 * k)
            vec = vec + 0.05 * rng.standard_normal(dim)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec.astype(float).tolist())
        tables[modality] = rows
    spec = EmbeddingSpec(
        model_name="synthetic_pathway_onehot",
        model_version=BUNDLE_VERSION,
        training_data_description=(
            "Deterministic mix of Reactome-like one-hots and a positional "
            "sin/cos. No public sequence or literature corpus."
        ),
        license="CC0-1.0 (synthetic fixture)",
        extraction_layer="constructed_table",
        dim=dim,
        species_taxon_id=TAXON,
        input_kind="constructed_table",
    )
    return tables, spec
