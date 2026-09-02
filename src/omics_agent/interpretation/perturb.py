"""Group feature ablation and condition-stratified permutation.

Deltas are changes in target prediction or masked MSE. They are not
causal effect sizes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from omics_agent.errors import InterpretationError
from omics_agent.models.dynamics.plugin import _Tensors
from omics_agent.models.dynamics.sequences import build_sequences
from omics_agent.models.tasks import DataForModel
from omics_agent.schemas.priors import PriorBundle


def _predict(plugin: Any, tensors: _Tensors) -> np.ndarray:
    if plugin._model is None:
        raise InterpretationError(
            "No frozen model is loaded.",
            how_to_fix="load() a checkpoint written by save().",
        )
    plugin._model.eval()
    with torch.no_grad():
        pred, _ = plugin._model(
            values=tensors.values,
            masks=tensors.masks,
            step_dt=tensors.step_dt,
            pad=tensors.pad,
            target_dt=tensors.target_dt,
            condition=tensors.condition,
        )
    return pred.detach().cpu().numpy()


def _masked_target_mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    keep = mask.astype(bool)
    err = np.where(keep, (pred - np.nan_to_num(true)) ** 2, 0.0)
    denom = keep.sum(axis=0).clip(min=1)
    return err.sum(axis=0) / denom


def _local_index(sources: list[tuple[str, str]], modality: str, source_i: int) -> int:
    return sum(1 for i, (mod, _) in enumerate(sources) if i < source_i and mod == modality)


def group_feature_ablation(
    plugin: Any,
    data: DataForModel,
    *,
    sources: list[tuple[str, str]],
    n_targets: int,
) -> np.ndarray:
    """Zero one source feature (value and mask). Return mean |Δpred| per pair."""

    seq = build_sequences(data, condition_categories=plugin._conditions, model_name=plugin.name)
    device = plugin._device
    base = _Tensors(seq, device)
    pred0 = _predict(plugin, base)
    deltas = np.zeros((len(sources), n_targets), dtype=np.float64)
    for s_i, (modality, _) in enumerate(sources):
        local_i = _local_index(sources, modality, s_i)
        values = {k: v.clone() for k, v in base.values.items()}
        masks = {k: v.clone() for k, v in base.masks.items()}
        values[modality][:, :, local_i] = 0
        masks[modality][:, :, local_i] = 0
        ablated = _Tensors(seq, device)
        ablated.values = values
        ablated.masks = masks
        pred = _predict(plugin, ablated)
        deltas[s_i] = np.abs(pred - pred0).mean(axis=0)
    return deltas


def stratified_permutation(
    plugin: Any,
    data: DataForModel,
    *,
    sources: list[tuple[str, str]],
    n_targets: int,
    n_seeds: int,
    seed: int,
) -> np.ndarray:
    """Shuffle one source feature within condition strata. Return mean ΔMSE."""

    seq = build_sequences(data, condition_categories=plugin._conditions, model_name=plugin.name)
    device = plugin._device
    base = _Tensors(seq, device)
    pred0 = _predict(plugin, base)
    mse0 = _masked_target_mse(pred0, seq.y_true, seq.y_mask)
    conditions = np.asarray(data.forecast.conditions)
    out = np.zeros((len(sources), n_targets), dtype=np.float64)
    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(0, 2**31 - 1, size=n_seeds)
    for s_i, (modality, _) in enumerate(sources):
        local_i = _local_index(sources, modality, s_i)
        drops = []
        for perm_seed in seeds:
            rng = np.random.default_rng(int(perm_seed))
            values = {k: v.clone() for k, v in base.values.items()}
            column = values[modality][:, :, local_i].detach().cpu().numpy().copy()
            for cond in np.unique(conditions):
                idx = np.flatnonzero(conditions == cond)
                if idx.size < 2:
                    continue
                shuffled = idx.copy()
                rng.shuffle(shuffled)
                column[idx] = column[shuffled]
            values[modality][:, :, local_i] = torch.as_tensor(
                column, device=device, dtype=values[modality].dtype
            )
            permuted = _Tensors(seq, device)
            permuted.values = values
            pred = _predict(plugin, permuted)
            mse = _masked_target_mse(pred, seq.y_true, seq.y_mask)
            drops.append(mse - mse0)
        out[s_i] = np.mean(np.stack(drops, axis=0), axis=0)
    return out


def prior_flags(
    *,
    source: tuple[str, str],
    target: tuple[str, str],
    bundle: PriorBundle | None,
    embedding_used: bool,
) -> tuple[bool, bool, bool]:
    """Return (prior_edge_used, embedding_supported, de_novo_model_edge)."""

    prior_edge = False
    embedding_supported = False
    if bundle is not None:
        for edge in bundle.edges:
            a = (edge.source_modality, edge.source_id)
            b = (edge.target_modality, edge.target_id)
            if {a, b} == {source, target}:
                prior_edge = True
                break
        families: dict[str, set[tuple[str, str]]] = {}
        for item in bundle.pathways:
            family = item.pathway_id.replace("-PROT", "")
            for member in item.member_ids:
                families.setdefault(family, set()).add((item.member_modality, member))
        for group in families.values():
            if source in group and target in group:
                prior_edge = True
                break
        if embedding_used and bundle.embedding_spec is not None:

            def _nonzero(mod: str, name: str) -> bool:
                ids = bundle.features.get(mod, [])
                rows = bundle.embeddings.get(mod, [])
                if name not in ids:
                    return False
                vec = rows[ids.index(name)]
                return any(abs(float(v)) > 0 for v in vec)

            embedding_supported = _nonzero(*source) and _nonzero(*target)
    return prior_edge, embedding_supported, (not prior_edge)
