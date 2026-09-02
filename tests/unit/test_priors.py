"""PriorBundle, STRING labelling, Laplacian, and degree-matched random graph."""

from __future__ import annotations

import numpy as np
import pytest

from omics_agent.errors import SchemaError
from omics_agent.priors.graph import (
    degree_matched_random_graph,
    graph_laplacian,
    laplacian_quadratic,
    laplacian_weights,
)
from omics_agent.priors.pathway import membership_matrix, pathway_activity
from omics_agent.priors.runtime import align_prior
from omics_agent.priors.synthetic import build_synthetic_prior_bundle
from omics_agent.schemas.enums import EdgeType, PriorAblation
from omics_agent.schemas.priors import PriorEdge, flags_for


def _edge(**overrides: object) -> PriorEdge:
    base: dict[str, object] = {
        "source_id": "G01",
        "target_id": "G02",
        "source_modality": "rna",
        "target_modality": "rna",
        "edge_type": EdgeType.FUNCTIONAL_ASSOCIATION,
        "directed": False,
        "score": 0.7,
        "evidence": ["coexpression"],
        "species_taxon_id": 9606,
        "source_name": "STRING",
        "source_version": "12.0",
    }
    base.update(overrides)
    return PriorEdge(**base)  # type: ignore[arg-type]


def test_string_functional_cannot_be_labelled_physical() -> None:
    with pytest.raises(SchemaError, match="physical_ppi"):
        _edge(edge_type=EdgeType.PHYSICAL_PPI)


def test_string_cannot_be_labelled_gene_regulation() -> None:
    with pytest.raises(SchemaError, match="gene_regulation"):
        _edge(edge_type=EdgeType.GENE_REGULATION, directed=True)


def test_string_functional_is_not_causal() -> None:
    edge = _edge()
    assert edge.edge_type is EdgeType.FUNCTIONAL_ASSOCIATION
    assert edge.is_causal is False
    dumped = edge.model_dump()
    assert dumped["is_causal"] is False


def test_intact_physical_ppi_is_allowed() -> None:
    edge = _edge(
        source_id="P01",
        target_id="P02",
        source_modality="protein",
        target_modality="protein",
        edge_type=EdgeType.PHYSICAL_PPI,
        source_name="IntAct",
        evidence=["affinity_chromatography"],
    )
    assert edge.edge_type is EdgeType.PHYSICAL_PPI
    assert edge.is_causal is False


def test_flags_for_five_required_arms() -> None:
    assert flags_for(PriorAblation.NO_PRIOR).model_dump() == {
        "ablation": "no_prior",
        "use_pathway": False,
        "use_graph": False,
        "use_embedding": False,
        "randomize_graph": False,
    }
    assert flags_for(PriorAblation.GRAPH_ONLY).use_graph is True
    assert flags_for(PriorAblation.EMBEDDING_ONLY).use_embedding is True
    combined = flags_for(PriorAblation.COMBINED)
    assert combined.use_pathway and combined.use_graph and combined.use_embedding
    assert flags_for(PriorAblation.RANDOM_GRAPH).randomize_graph is True


def test_synthetic_bundle_is_versioned_and_hashed() -> None:
    bundle = build_synthetic_prior_bundle()
    assert bundle.bundle_version.startswith("synthetic_priors_")
    assert bundle.embedding_spec is not None and bundle.embedding_spec.frozen is True
    string_edges = [e for e in bundle.edges if "string" in e.source_name.lower()]
    assert string_edges
    assert all(e.edge_type is EdgeType.FUNCTIONAL_ASSOCIATION for e in string_edges)
    assert all(e.evidence for e in string_edges)
    assert bundle.content_hash() == build_synthetic_prior_bundle().content_hash()


def test_pathway_activity_masks_unobserved_members() -> None:
    bundle = build_synthetic_prior_bundle()
    matrix, names = membership_matrix(bundle.pathways, bundle.features["rna"], modality="rna")
    assert names
    values = np.array([[1.0, 2.0, 3.0] + [0.0] * (matrix.shape[1] - 3)])
    mask = np.zeros_like(values, dtype=bool)
    mask[0, :3] = True
    activity, observed = pathway_activity(values, mask, matrix)
    assert activity.shape == (1, matrix.shape[0])
    # Pathways whose members are all masked stay unobserved.
    fully_missing = matrix[:, 3:].sum(axis=1) == matrix.sum(axis=1)
    if fully_missing.any():
        assert not observed[0, fully_missing].any()


def test_degree_matched_random_preserves_degree_not_neighborhood() -> None:
    bundle = build_synthetic_prior_bundle()
    order = [("rna", n) for n in bundle.features["rna"]] + [
        ("protein", n) for n in bundle.features["protein"]
    ]
    original_w = laplacian_weights(bundle.edges, order)
    random_edges = degree_matched_random_graph(bundle.edges, seed=11)
    assert all(e.randomized for e in random_edges)
    assert all(e.source_name == "degree_matched_random" for e in random_edges)
    random_w = laplacian_weights(random_edges, order)

    def incidence(edges: list) -> np.ndarray:
        index = {key: i for i, key in enumerate(order)}
        deg = np.zeros(len(order))
        for edge in edges:
            i = index.get((edge.source_modality, edge.source_id))
            j = index.get((edge.target_modality, edge.target_id))
            if i is None or j is None:
                continue
            deg[i] += 1
            deg[j] += 1
        return deg

    assert np.array_equal(np.sort(incidence(bundle.edges)), np.sort(incidence(random_edges)))
    assert not np.allclose(random_w, original_w)


def test_laplacian_quadratic_matches_pairwise_sum() -> None:
    states = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pairwise = 0.0
    n = states.shape[0]
    for i in range(n):
        for j in range(n):
            pairwise += float(weights[i, j] * ((states[i] - states[j]) ** 2).sum())
    assert laplacian_quadratic(states, weights) == pytest.approx(pairwise)
    lap = graph_laplacian(weights)
    assert lap.shape == (3, 3)


def test_align_random_graph_does_not_reuse_real_weights() -> None:
    bundle = build_synthetic_prior_bundle()
    names = bundle.features
    real = align_prior(
        bundle,
        feature_names=names,
        modalities=["rna", "protein"],
        ablation=PriorAblation.GRAPH_ONLY,
        random_graph_seed=3,
    )
    rnd = align_prior(
        bundle,
        feature_names=names,
        modalities=["rna", "protein"],
        ablation=PriorAblation.RANDOM_GRAPH,
        random_graph_seed=3,
    )
    assert real.flags.use_graph and rnd.flags.use_graph
    assert not np.allclose(real.laplacian_weights, rnd.laplacian_weights)
    none = align_prior(
        bundle,
        feature_names=names,
        modalities=["rna", "protein"],
        ablation=PriorAblation.NO_PRIOR,
        random_graph_seed=3,
    )
    assert none.laplacian_weights.sum() == 0
    assert none.frozen_embeddings == {}
    assert none.pathway_membership == {}
