"""Aggregate IG / ablation / permutation into a stability table.

Stability uses donor bootstrap, condition-stratified permutation seeds,
multiple IG baselines, and unit folds. Test instances are never passed in.
"""

from __future__ import annotations

import numpy as np

from omics_agent.interpretation.perturb import prior_flags
from omics_agent.schemas.enums import RelationDirection
from omics_agent.schemas.interpretation import (
    CLAIM_KIND,
    HYPOTHESIS_CAVEAT,
    CandidateRow,
    InterpretationConfig,
    StabilityTable,
)
from omics_agent.schemas.priors import PriorBundle


def _bootstrap_means(
    attr: np.ndarray,
    group_ids: list[str],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Donor-level bootstrap of mean attribution. Shape [boot, source, target]."""

    units = np.asarray(group_ids)
    unique = np.unique(units)
    rng = np.random.default_rng(seed)
    # Mean over baselines first → [B, S, T]
    per_instance = attr.mean(axis=3)
    draws = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(units == unit) for unit in chosen])
        draws.append(per_instance[rows].mean(axis=0))
    return np.stack(draws, axis=0)


def _fold_means(attr: np.ndarray, group_ids: list[str], n_folds: int) -> np.ndarray:
    units = np.unique(np.asarray(group_ids))
    n_folds = min(n_folds, max(2, len(units)))
    folds = np.array_split(units, n_folds)
    per_instance = attr.mean(axis=3)
    out = []
    for part in folds:
        if part.size == 0:
            continue
        rows = np.concatenate([np.flatnonzero(np.asarray(group_ids) == u) for u in part])
        out.append(per_instance[rows].mean(axis=0))
    return np.stack(out, axis=0) if out else per_instance.mean(axis=0, keepdims=True)


def assemble_candidates(
    *,
    experiment_id: str,
    model_name: str,
    attr: np.ndarray,
    sources: list[tuple[str, str]],
    targets: list[tuple[str, str]],
    group_ids: list[str],
    ablation: np.ndarray,
    permutation: np.ndarray,
    config: InterpretationConfig,
    seed: int,
    bundle: PriorBundle | None,
    embedding_used: bool,
) -> StabilityTable:
    """Build the candidate table. Every row is a hypothesis."""

    boot = _bootstrap_means(attr, group_ids, config.n_bootstrap, seed)
    folds = _fold_means(attr, group_ids, config.n_folds)
    mean_attr = attr.mean(axis=(0, 3))
    # Sign consistency: fraction of (bootstrap × baseline-mean) matching mean sign.
    signs = np.sign(np.where(np.abs(boot) < 1e-12, 0.0, boot))
    mean_sign = np.sign(np.where(np.abs(mean_attr) < 1e-12, 0.0, mean_attr))
    sign_ok = (signs == mean_sign) | (mean_sign == 0)
    sign_consistency = sign_ok.mean(axis=0)
    # Also require fold signs to agree when we have folds.
    fold_signs = np.sign(np.where(np.abs(folds) < 1e-12, 0.0, folds))
    fold_ok = (fold_signs == mean_sign) | (mean_sign == 0)
    sign_consistency = 0.5 * sign_consistency + 0.5 * fold_ok.mean(axis=0)

    ranks = np.empty_like(boot)
    for b in range(boot.shape[0]):
        for t in range(boot.shape[2]):
            order = np.argsort(-np.abs(boot[b, :, t]))
            rank = np.empty(boot.shape[1])
            rank[order] = np.arange(1, boot.shape[1] + 1)
            ranks[b, :, t] = rank
    rank_median = np.median(ranks, axis=0)
    selected = ranks <= config.top_k_per_target
    selection_frequency = selected.mean(axis=0)
    low = np.quantile(boot, 0.025, axis=0)
    high = np.quantile(boot, 0.975, axis=0)
    stability = 0.5 * sign_consistency + 0.5 * selection_frequency

    rows: list[CandidateRow] = []
    for s_i, source in enumerate(sources):
        for t_i, target in enumerate(targets):
            prior_edge, emb, de_novo = prior_flags(
                source=source,
                target=target,
                bundle=bundle,
                embedding_used=embedding_used,
            )
            mean = float(mean_attr[s_i, t_i])
            passed = (
                float(sign_consistency[s_i, t_i]) >= config.min_sign_consistency
                and float(selection_frequency[s_i, t_i]) >= config.min_selection_frequency
                and float(stability[s_i, t_i]) >= config.min_stability
            )
            if mean > 1e-12:
                direction = RelationDirection.UP.value
            elif mean < -1e-12:
                direction = RelationDirection.DOWN.value
            else:
                direction = RelationDirection.UNKNOWN.value
            rows.append(
                CandidateRow(
                    candidate_id=f"{source[0]}:{source[1]}->{target[0]}:{target[1]}",
                    source_modality=source[0],
                    source_id=source[1],
                    target_modality=target[0],
                    target_id=target[1],
                    mean_attribution=mean,
                    sign_consistency=float(sign_consistency[s_i, t_i]),
                    rank_median=float(rank_median[s_i, t_i]),
                    selection_frequency=float(selection_frequency[s_i, t_i]),
                    bootstrap_low=float(low[s_i, t_i]),
                    bootstrap_high=float(high[s_i, t_i]),
                    ablation_delta=float(ablation[s_i, t_i]),
                    permutation_delta=float(permutation[s_i, t_i]),
                    stability=float(stability[s_i, t_i]),
                    prior_edge_used=prior_edge,
                    embedding_supported=emb,
                    de_novo_model_edge=de_novo,
                    passed_stability=passed,
                    predicted_direction=direction,
                    claim_kind=CLAIM_KIND,
                    caveat=HYPOTHESIS_CAVEAT,
                )
            )
    rows.sort(key=lambda row: (-row.stability, -abs(row.mean_attribution)))
    notes = [
        "Every row is a hypothesis about model prediction contribution.",
        "Attribution is not causation. Do not write 首次发现 or claim a regulatory effect.",
        "Only passed_stability rows may be sent to PubMed / Europe PMC.",
        f"claim_kind={CLAIM_KIND}.",
    ]
    return StabilityTable(
        experiment_id=experiment_id,
        model_name=model_name,
        objective_split="val",
        test_labels_visible=False,
        claim_kind=CLAIM_KIND,
        n_baselines=attr.shape[3],
        n_bootstrap=config.n_bootstrap,
        n_seeds=config.n_seeds,
        n_folds=min(config.n_folds, max(2, len(set(group_ids)))),
        rows=rows,
        notes=notes,
    )


def select_stable(table: StabilityTable, top_n: int) -> list[CandidateRow]:
    """Top-N candidates that passed the pre-registered stability thresholds."""

    passed = [row for row in table.rows if row.passed_stability]
    return passed[:top_n]
