"""Graph Laplacian weights and a degree-matched random-graph negative control.

The Laplacian penalty is sum_ij w_ij ||h_i - h_j||^2 with w_ij = score
(undirected edges are stored symmetrically). A degree-matched shuffle keeps
each node's degree and the edge-type / score / evidence bags; it does not
keep neighborhoods. If the real graph and this control do not differ, do
not claim a biological gain from the prior.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from omics_agent.errors import PriorError
from omics_agent.schemas.enums import EdgeType
from omics_agent.schemas.priors import PriorEdge

# Type scales are 1.0: we do not silently up-weight "physical" over STRING.
_TYPE_SCALE: dict[EdgeType, float] = {item: 1.0 for item in EdgeType}


def feature_index(order: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    return {key: i for i, key in enumerate(order)}


def laplacian_weights(
    edges: list[PriorEdge],
    order: list[tuple[str, str]],
) -> np.ndarray:
    """Dense symmetric weight matrix aligned to ``order``. Isolated nodes stay 0."""

    n = len(order)
    weights = np.zeros((n, n), dtype=np.float64)
    index = feature_index(order)
    for edge in edges:
        src = (edge.source_modality, edge.source_id)
        tgt = (edge.target_modality, edge.target_id)
        if src not in index or tgt not in index:
            continue
        i, j = index[src], index[tgt]
        if i == j:
            continue
        w = float(edge.score) * _TYPE_SCALE[edge.edge_type]
        weights[i, j] += w
        if not edge.directed:
            weights[j, i] += w
        else:
            # Directed edges still contribute an undirected smoothness term:
            # regulation A->B does not claim B causes A, but the Laplacian
            # regularizer is a similarity penalty, not a causal arrow.
            weights[j, i] += w
    return weights


def graph_laplacian(weights: np.ndarray) -> np.ndarray:
    """Unnormalized L = D - W. ``weights`` must be square and nonnegative."""

    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise PriorError(
            f"Laplacian weights must be square, got {weights.shape}.",
            how_to_fix="Build the matrix with laplacian_weights() on a single feature order.",
        )
    degree = np.diag(weights.sum(axis=1))
    return degree - weights


def laplacian_quadratic(states: np.ndarray, weights: np.ndarray) -> float:
    """2 tr(H.T @ L @ H) equals sum_ij w_ij ||h_i - h_j||^2."""

    lap = graph_laplacian(weights)
    return float(2.0 * np.einsum("id,ij,jd->", states, lap, states))


def degree_matched_random_graph(
    edges: list[PriorEdge],
    *,
    seed: int,
) -> list[PriorEdge]:
    """Configuration-model shuffle that preserves degree, type, score, evidence.

    Endpoints are rematched from stubs. Self-loops are repaired by a further
    swap. The resulting edges are marked randomized=true and
    source_name='degree_matched_random' so they cannot be mistaken for a
    biological graph.
    """

    if not edges:
        return []
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[EdgeType, bool], list[PriorEdge]] = defaultdict(list)
    for edge in edges:
        grouped[(edge.edge_type, edge.directed)].append(edge)
    out: list[PriorEdge] = []
    for (edge_type, directed), bucket in grouped.items():
        out.extend(_rewire_group(bucket, edge_type=edge_type, directed=directed, rng=rng))
    return out


def _rewire_group(
    bucket: list[PriorEdge],
    *,
    edge_type: EdgeType,
    directed: bool,
    rng: np.random.Generator,
) -> list[PriorEdge]:
    sources = [(e.source_modality, e.source_id) for e in bucket]
    targets = [(e.target_modality, e.target_id) for e in bucket]
    if directed:
        rng.shuffle(targets)
        pairs = list(zip(sources, targets, strict=True))
    else:
        stubs = sources + targets
        rng.shuffle(stubs)
        pairs = [(stubs[i], stubs[i + 1]) for i in range(0, len(stubs), 2)]
    pairs = _repair_self_loops(pairs, rng)
    rewritten: list[PriorEdge] = []
    for edge, (src, tgt) in zip(bucket, pairs, strict=True):
        rewritten.append(
            edge.model_copy(
                update={
                    "source_modality": src[0],
                    "source_id": src[1],
                    "target_modality": tgt[0],
                    "target_id": tgt[1],
                    "edge_type": edge_type,
                    "randomized": True,
                    "source_name": "degree_matched_random",
                    "is_causal": False,
                }
            )
        )
    return rewritten


def _repair_self_loops(
    pairs: list[tuple[tuple[str, str], tuple[str, str]]],
    rng: np.random.Generator,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Swap a self-loop with a later non-loop pair when possible."""

    work = list(pairs)
    n = len(work)
    for i, (src, tgt) in enumerate(work):
        if src != tgt:
            continue
        candidates = [j for j in range(n) if j != i and work[j][0] != work[j][1]]
        if not candidates:
            continue
        j = int(rng.choice(candidates))
        work[i] = (src, work[j][1])
        work[j] = (work[j][0], tgt)
    return work
