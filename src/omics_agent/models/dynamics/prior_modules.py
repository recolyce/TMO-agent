"""Frozen embedding projection/gate and the graph-Laplacian penalty.

h_i = gate * learned_i + (1 - gate) * MLP(LayerNorm(e_i))

The embedding table ``e_i`` is a buffer (not trained). The projection and
gate are trained. Graph smoothness is applied to the gated ``h_i`` so the
regularizer actually reaches the prediction path (via a per-feature scale).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeaturePriorModule(nn.Module):
    """Per-feature states used by the embedding gate and the Laplacian."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        *,
        proj_dim: int,
        use_embedding: bool,
        frozen: dict[str, Tensor] | None,
    ) -> None:
        super().__init__()
        self.modalities = list(feature_dims)
        self.use_embedding = use_embedding
        self.proj_dim = proj_dim
        self.learned = nn.ParameterDict(
            {
                name: nn.Parameter(0.02 * torch.randn(n_feat, proj_dim))
                for name, n_feat in feature_dims.items()
            }
        )
        self.to_scale = nn.Linear(proj_dim, 1)
        if use_embedding:
            if not frozen:
                raise ValueError("use_embedding=True requires a frozen embedding table.")
            self.frozen = nn.ParameterDict(
                {
                    name: nn.Parameter(table.clone(), requires_grad=False)
                    for name, table in frozen.items()
                }
            )
            prior_dim = next(iter(frozen.values())).shape[1]
            self.proj = nn.Sequential(
                nn.LayerNorm(prior_dim),
                nn.Linear(prior_dim, proj_dim),
                nn.SiLU(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.gate_logit = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.zeros(n_feat, 1))
                    for name, n_feat in feature_dims.items()
                }
            )
        else:
            self.frozen = None  # type: ignore[assignment]
            self.proj = None  # type: ignore[assignment]
            self.gate_logit = None  # type: ignore[assignment]

    def states(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for name in self.modalities:
            learned = self.learned[name]
            if self.use_embedding:
                assert self.proj is not None and self.gate_logit is not None
                assert self.frozen is not None
                projected = self.proj(self.frozen[name])
                gate = torch.sigmoid(self.gate_logit[name])
                out[name] = gate * learned + (1.0 - gate) * projected
            else:
                out[name] = learned
        return out

    def scales(self) -> dict[str, Tensor]:
        """Multiplicative feature scales: 1 + tanh(Linear(h_i))."""

        return {
            name: 1.0 + torch.tanh(self.to_scale(state)).squeeze(-1)
            for name, state in self.states().items()
        }

    def stacked_states(self) -> Tensor:
        return torch.cat([self.states()[name] for name in self.modalities], dim=0)


def laplacian_penalty(states: Tensor, weights: Tensor) -> Tensor:
    """sum_ij w_ij ||h_i - h_j||^2 = 2 tr(H.T @ L @ H)."""

    degree = weights.sum(dim=1)
    lap_h = degree.unsqueeze(-1) * states - weights @ states
    return 2.0 * (states * lap_h).sum()
