"""Align a PriorBundle to a model's feature order and one ablation arm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from omics_agent.priors.graph import degree_matched_random_graph, laplacian_weights
from omics_agent.priors.pathway import membership_matrix
from omics_agent.schemas.enums import PriorAblation
from omics_agent.schemas.priors import AblationFlags, PriorBundle, flags_for


@dataclass
class PriorRuntime:
    """Numpy tensors the dynamics plugin consumes. Feature order is frozen."""

    ablation: PriorAblation
    flags: AblationFlags
    order: list[tuple[str, str]]
    pathway_membership: dict[str, np.ndarray] = field(default_factory=dict)
    pathway_names: dict[str, list[str]] = field(default_factory=dict)
    frozen_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    laplacian_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bundle_hash: str = ""
    bundle_version: str = ""
    embedding_dim: int = 0
    n_unmapped_edges: int = 0
    notes: list[str] = field(default_factory=list)


def feature_order(
    feature_names: dict[str, list[str]], modalities: list[str]
) -> list[tuple[str, str]]:
    """Stable (modality, feature_id) list matching the sequence builder."""

    return [(mod, name) for mod in modalities for name in feature_names[mod]]


def align_prior(
    bundle: PriorBundle,
    *,
    feature_names: dict[str, list[str]],
    modalities: list[str],
    ablation: PriorAblation,
    random_graph_seed: int,
) -> PriorRuntime:
    """Slice the bundle onto the model's features. Unmapped ids stay unused."""

    flags = flags_for(ablation)
    order = feature_order(feature_names, modalities)
    known = set(order)
    n_unmapped = 0
    for edge in bundle.edges:
        src = (edge.source_modality, edge.source_id)
        tgt = (edge.target_modality, edge.target_id)
        if src not in known or tgt not in known:
            n_unmapped += 1

    edges = list(bundle.edges)
    notes = list(bundle.notes)
    if flags.randomize_graph:
        edges = degree_matched_random_graph(edges, seed=random_graph_seed)
        notes.append(
            "Graph is a degree-matched random rewire (negative control). "
            "Do not interpret this arm as biological structure."
        )

    weights = (
        laplacian_weights(edges, order)
        if flags.use_graph
        else np.zeros((len(order), len(order)), dtype=np.float64)
    )

    pathway: dict[str, np.ndarray] = {}
    pathway_names: dict[str, list[str]] = {}
    if flags.use_pathway:
        for modality in modalities:
            matrix, names = membership_matrix(
                bundle.pathways, feature_names[modality], modality=modality
            )
            if matrix.size and matrix.shape[0] > 0:
                pathway[modality] = matrix
                pathway_names[modality] = names

    frozen: dict[str, np.ndarray] = {}
    emb_dim = 0
    if flags.use_embedding and bundle.embedding_spec is not None:
        emb_dim = bundle.embedding_spec.dim
        for modality in modalities:
            names = feature_names[modality]
            table = np.zeros((len(names), emb_dim), dtype=np.float64)
            src_ids = bundle.features.get(modality, [])
            src_rows = bundle.embeddings.get(modality, [])
            src_index = {name: i for i, name in enumerate(src_ids)}
            for j, name in enumerate(names):
                src_i = src_index.get(name)
                if src_i is not None and src_i < len(src_rows):
                    table[j] = np.asarray(src_rows[src_i], dtype=np.float64)
            frozen[modality] = table

    return PriorRuntime(
        ablation=ablation,
        flags=flags,
        order=order,
        pathway_membership=pathway,
        pathway_names=pathway_names,
        frozen_embeddings=frozen,
        laplacian_weights=weights,
        bundle_hash=bundle.content_hash(),
        bundle_version=bundle.bundle_version,
        embedding_dim=emb_dim,
        n_unmapped_edges=n_unmapped,
        notes=notes,
    )
