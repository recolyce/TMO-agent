"""Integrated Gradients on a frozen TemporalCore.

Uses Captum when installed. Otherwise the same Riemann path integral.
Baselines: zeros, train-mean (observed train values), last observation.
Attribution is a prediction contribution, not a causal effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from omics_agent.errors import InterpretationError
from omics_agent.models.dynamics.plugin import _Tensors
from omics_agent.models.dynamics.sequences import build_sequences, modalities_from_input_names
from omics_agent.models.tasks import DataForModel
from omics_agent.schemas.enums import IgBaselineName
from omics_agent.schemas.model import AttributionTable

if TYPE_CHECKING:
    from omics_agent.models.dynamics.plugin import _DynamicsBase


def feature_keys(data: DataForModel) -> list[tuple[str, str]]:
    """(modality, feature_id) in the flattened encoder order."""

    modalities = modalities_from_input_names(data.forecast.input_feature_names)
    return [(mod, name) for mod in modalities for name in data.bundle.feature_names[mod]]


def target_keys(data: DataForModel, target_modality: str) -> list[tuple[str, str]]:
    return [(target_modality, name) for name in data.forecast.feature_names]


def _stack(values: dict[str, Tensor], modalities: list[str]) -> Tensor:
    return torch.cat([values[name] for name in modalities], dim=-1)


def _split(flat: Tensor, feature_dims: dict[str, int], modalities: list[str]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    start = 0
    for name in modalities:
        width = feature_dims[name]
        out[name] = flat[..., start : start + width]
        start += width
    return out


class _FlatForward(nn.Module):
    """Captum-facing wrapper: flat [B, T, F] values → target predictions."""

    def __init__(
        self,
        core: nn.Module,
        *,
        masks: dict[str, Tensor],
        step_dt: Tensor,
        pad: Tensor,
        target_dt: Tensor,
        condition: Tensor,
        modalities: list[str],
        feature_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.core = core
        self.masks = masks
        self.step_dt = step_dt
        self.pad = pad
        self.target_dt = target_dt
        self.condition = condition
        self.modalities = modalities
        self.feature_dims = feature_dims

    def forward(self, flat: Tensor) -> Tensor:
        values = _split(flat, self.feature_dims, self.modalities)
        pred, _ = self.core(
            values=values,
            masks=self.masks,
            step_dt=self.step_dt,
            pad=self.pad,
            target_dt=self.target_dt,
            condition=self.condition,
        )
        return pred


def _riemann_ig(
    wrapper: _FlatForward, inputs: Tensor, baseline: Tensor, target: int, n_steps: int
) -> Tensor:
    """Path-integral IG (Riemann left-rule). Matches Captum's default idea."""

    delta = inputs - baseline
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=inputs.device, dtype=inputs.dtype)
    grads = []
    for alpha in alphas:
        x = (baseline + alpha * delta).detach().requires_grad_(True)
        y = wrapper(x)[:, target].sum()
        (grad,) = torch.autograd.grad(y, x, retain_graph=False)
        grads.append(grad)
    return delta * torch.stack(grads, dim=0).mean(dim=0)


def _captum_ig(
    wrapper: _FlatForward, inputs: Tensor, baseline: Tensor, target: int, n_steps: int
) -> Tensor:
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(inputs, baselines=baseline, target=target, n_steps=n_steps)
    if not isinstance(attr, Tensor):
        raise InterpretationError(
            "Captum IntegratedGradients returned a non-tensor.",
            how_to_fix="Use captum>=0.7 with a single tensor input.",
        )
    return attr


def _attribute_one(
    wrapper: _FlatForward, inputs: Tensor, baseline: Tensor, target: int, n_steps: int
) -> Tensor:
    try:
        return _captum_ig(wrapper, inputs, baseline, target, n_steps)
    except ImportError:
        return _riemann_ig(wrapper, inputs, baseline, target, n_steps)


def _train_mean_flat(train: DataForModel, plugin: Any) -> np.ndarray:
    seq = build_sequences(train, condition_categories=plugin._conditions, model_name=plugin.name)
    parts = []
    for modality in seq.modalities:
        vals = seq.values[modality]
        mask = seq.masks[modality]
        weight = mask.sum(axis=(0, 1), keepdims=False)
        total = (vals * mask).sum(axis=(0, 1))
        mean = np.divide(total, np.maximum(weight, 1.0))
        parts.append(mean.astype(np.float64))
    return np.concatenate(parts, axis=0)


def _baseline_tensor(
    name: IgBaselineName,
    inputs: Tensor,
    pad: Tensor,
    train_mean: np.ndarray | None,
) -> Tensor:
    if name is IgBaselineName.ZEROS:
        return torch.zeros_like(inputs)
    if name is IgBaselineName.TRAIN_MEAN:
        if train_mean is None:
            raise InterpretationError(
                "train_mean IG baseline needs the train split sequences.",
                how_to_fix="Pass the train DataForModel used to fit the frozen model.",
            )
        mean = torch.as_tensor(train_mean, device=inputs.device, dtype=inputs.dtype)
        return mean.view(1, 1, -1).expand_as(inputs).contiguous()
    # last observation: broadcast the last real step to every time index
    last = inputs[:, -1:, :].clone()
    lengths = pad.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
    batch = torch.arange(inputs.shape[0], device=inputs.device)
    last = inputs[batch, lengths].unsqueeze(1)
    return last.expand_as(inputs).contiguous()


def integrated_gradients(
    plugin: _DynamicsBase,
    data: DataForModel,
    *,
    train: DataForModel | None,
    n_steps: int,
    baselines: list[IgBaselineName],
    target_modality: str,
) -> dict[str, Any]:
    """Per-instance, per-baseline attributions. Shape [B, n_sources, n_targets, n_baselines]."""

    if plugin._model is None or plugin._device is None:
        raise InterpretationError(
            f"{plugin.name} has no loaded weights.",
            how_to_fix="load() the frozen checkpoint, or fit() first in tests.",
        )
    seq = build_sequences(data, condition_categories=plugin._conditions, model_name=plugin.name)
    tensors = _Tensors(seq, plugin._device)
    feature_dims = {name: tensors.values[name].shape[-1] for name in seq.modalities}
    inputs = _stack(tensors.values, seq.modalities)
    wrapper = _FlatForward(
        plugin._model,
        masks=tensors.masks,
        step_dt=tensors.step_dt,
        pad=tensors.pad,
        target_dt=tensors.target_dt,
        condition=tensors.condition,
        modalities=seq.modalities,
        feature_dims=feature_dims,
    )
    wrapper.eval()
    train_mean = _train_mean_flat(train, plugin) if train is not None else None
    n_targets = tensors.y_true.shape[1]
    n_sources = inputs.shape[-1]
    n_base = len(baselines)
    batch = inputs.shape[0]
    table = np.zeros((batch, n_sources, n_targets, n_base), dtype=np.float64)
    pad = tensors.pad
    weight = pad.float().unsqueeze(-1)
    for b_i, baseline_name in enumerate(baselines):
        baseline = _baseline_tensor(baseline_name, inputs, pad, train_mean)
        for t_i in range(n_targets):
            attr = _attribute_one(wrapper, inputs, baseline, t_i, n_steps)
            reduced = (attr * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
            table[:, :, t_i, b_i] = reduced.detach().cpu().numpy()
    sources = feature_keys(data)
    targets = target_keys(data, target_modality)
    if len(sources) != n_sources:
        raise InterpretationError(
            f"Feature flattening mismatch: {len(sources)} keys vs {n_sources} columns.",
            how_to_fix="Input modalities and bundle.feature_names must match the checkpoint.",
        )
    return {
        "attr": table,
        "sources": sources,
        "targets": targets,
        "baselines": [item.value for item in baselines],
        "group_ids": list(data.forecast.group_ids),
        "conditions": list(data.forecast.conditions),
        "method": "captum_integrated_gradients",
        "engine": "captum" if _captum_available() else "riemann_fallback",
    }


def _captum_available() -> bool:
    try:
        import captum  # noqa: F401

        return True
    except ImportError:
        return False


def integrated_gradients_table(
    plugin: _DynamicsBase,
    data: DataForModel,
    targets: list[str],
    *,
    train: DataForModel | None = None,
    n_steps: int = 8,
) -> AttributionTable:
    """Mean IG over instances and the zeros baseline (plugin.explain surface)."""

    payload = integrated_gradients(
        plugin,
        data,
        train=train,
        n_steps=n_steps,
        baselines=[IgBaselineName.ZEROS],
        target_modality=data.forecast.feature_names[0].split(":")[0]
        if ":" in (data.forecast.feature_names[0] if data.forecast.feature_names else "")
        else "protein",
    )
    wanted = set(targets) if targets else {name for _, name in payload["targets"]}
    rows = []
    mean = payload["attr"].mean(axis=(0, 3))
    for s_i, (s_mod, s_id) in enumerate(payload["sources"]):
        for t_i, (t_mod, t_id) in enumerate(payload["targets"]):
            if t_id not in wanted:
                continue
            rows.append(
                {
                    "source": f"{s_mod}:{s_id}",
                    "target": f"{t_mod}:{t_id}",
                    "attribution": float(mean[s_i, t_i]),
                }
            )
    return AttributionTable(
        model_name=plugin.name,
        method=payload["method"],
        rows=rows,
        caveat=("Integrated Gradients values are prediction contributions, not causation."),
    )
