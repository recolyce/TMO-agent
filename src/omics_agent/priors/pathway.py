"""Reactome pathway activity: membership-weighted mean of observed features.

A pathway with no observed members is masked out (activity 0, mask 0).
Missing feature values never enter the mean. This is per-sample math and
learns no cross-sample statistics.
"""

from __future__ import annotations

import numpy as np

from omics_agent.schemas.priors import PathwayMembership


def membership_matrix(
    pathways: list[PathwayMembership],
    feature_ids: list[str],
    *,
    modality: str,
) -> tuple[np.ndarray, list[str]]:
    """Return (n_pathways × n_features) 0/1 matrix and pathway ids for one modality."""

    rows = [item for item in pathways if item.member_modality == modality]
    names = [item.pathway_id for item in rows]
    index = {name: i for i, name in enumerate(feature_ids)}
    matrix = np.zeros((len(rows), len(feature_ids)), dtype=np.float64)
    for r, item in enumerate(rows):
        for member in item.member_ids:
            col = index.get(member)
            if col is not None:
                matrix[r, col] = 1.0
    return matrix, names


def pathway_activity(
    values: np.ndarray,
    mask: np.ndarray,
    membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Masked mean of members. ``values``/``mask`` are (..., F); membership is (P, F)."""

    observed = np.asarray(mask, dtype=np.float64)
    filled = np.where(observed > 0, np.asarray(values, dtype=np.float64), 0.0)
    weighted = np.tensordot(filled * observed, membership.T, axes=([-1], [0]))
    support = np.tensordot(observed, membership.T, axes=([-1], [0]))
    activity_mask = support > 0
    activity = np.divide(weighted, np.where(activity_mask, support, 1.0), where=activity_mask)
    activity = np.where(activity_mask, activity, 0.0)
    return activity.astype(np.float64), activity_mask.astype(bool)
