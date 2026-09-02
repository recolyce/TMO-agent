"""Shared network core: modality encoders → gated fusion → latent dynamics → decoders.

Three latent-dynamics variants share the encoders, fusion, and decoders:

- ``gru``: GRUCell updates at observations; delta_t enters as a cell input,
  and the jump to the target time is a zero-observation update with the
  actual gap.
- ``ode_rnn``: the hidden state evolves under an ODE field between
  observations (RK4 with per-sample dt) and is updated by a GRUCell at each
  observation.
- ``latent_ode``: a GRU encoder summarizes the history into z0; a latent
  ODE integrates through the observation times (reconstruction) and on to
  the target time (prediction). Deterministic autoencoder formulation —
  no VAE sampling, so runs are reproducible from the seed.

Only LayerNorm is used (no BatchNorm), so batch_size=1 is legal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from omics_agent.models.dynamics.odeint import odeint_rk4
from omics_agent.models.dynamics.prior_modules import FeaturePriorModule, laplacian_penalty

_MODES = ("gru", "ode_rnn", "latent_ode")


class ModalityEncoder(nn.Module):
    """[values ⊙ mask, mask] → embedding. Missing entries carry no value signal."""

    def __init__(self, n_features: int, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_features, emb_dim),
            nn.SiLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.net(torch.cat([values * mask, mask], dim=-1))


class GatedFusion(nn.Module):
    """Per-modality sigmoid gates conditioned on all embeddings + covariates."""

    def __init__(self, n_modalities: int, emb_dim: int, cond_dim: int) -> None:
        super().__init__()
        joint = n_modalities * emb_dim + cond_dim
        self.gates = nn.ModuleList(nn.Linear(joint, emb_dim) for _ in range(n_modalities))
        self.projections = nn.ModuleList(nn.Linear(emb_dim, emb_dim) for _ in range(n_modalities))

    def forward(self, embeddings: list[Tensor], condition: Tensor) -> Tensor:
        joint = torch.cat([*embeddings, condition], dim=-1)
        fused = torch.zeros_like(embeddings[0])
        for gate, projection, emb in zip(self.gates, self.projections, embeddings, strict=True):
            fused = fused + torch.sigmoid(gate(joint)) * projection(emb)
        return fused


class OdeField(nn.Module):
    """Autonomous tanh MLP vector field. Last layer starts near zero for stability."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.Tanh(), nn.Linear(2 * dim, dim))
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class ModalityDecoder(nn.Module):
    """Latent state → reconstructed modality values."""

    def __init__(self, state_dim: int, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.SiLU(), nn.Linear(state_dim, n_features)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class TemporalCore(nn.Module):
    """The full milestone-4 architecture for all three dynamics modes."""

    def __init__(
        self,
        *,
        feature_dims: dict[str, int],
        n_targets: int,
        cond_dim: int,
        emb_dim: int,
        hidden_dim: int,
        mode: str,
        rk4_substeps: int,
        pathway_dims: dict[str, int] | None = None,
        pathway_memberships: dict[str, Tensor] | None = None,
        use_feature_prior: bool = False,
        use_embedding_gate: bool = False,
        prior_proj_dim: int = 16,
        frozen_embeddings: dict[str, Tensor] | None = None,
        laplacian_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"Unknown dynamics mode '{mode}'. Use one of {_MODES}.")
        self.mode = mode
        self.modalities = list(feature_dims)
        self.rk4_substeps = rk4_substeps
        self.pathway_dims = dict(pathway_dims or {})
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(dim, emb_dim) for name, dim in feature_dims.items()}
        )
        self.pathway_encoders = nn.ModuleDict(
            {name: ModalityEncoder(dim, emb_dim) for name, dim in self.pathway_dims.items()}
        )
        n_streams = len(feature_dims) + len(self.pathway_dims)
        self.fusion = GatedFusion(n_streams, emb_dim, cond_dim)
        if pathway_memberships:
            for name, matrix in pathway_memberships.items():
                self.register_buffer(f"path_mem_{name}", matrix)
        self.feature_prior = (
            FeaturePriorModule(
                feature_dims,
                proj_dim=prior_proj_dim,
                use_embedding=use_embedding_gate,
                frozen=frozen_embeddings,
            )
            if use_feature_prior
            else None
        )
        if laplacian_weights is not None:
            self.register_buffer("laplacian_w", laplacian_weights)
        else:
            self.laplacian_w = None
        self.cell = nn.GRUCell(emb_dim + 1, hidden_dim)
        self.field = OdeField(hidden_dim)
        self.z0_net = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            if mode == "latent_ode"
            else None
        )
        self.decoders = nn.ModuleDict(
            {name: ModalityDecoder(hidden_dim, dim) for name, dim in feature_dims.items()}
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_targets),
        )
        self.hidden_dim = hidden_dim

    def _fused_steps(
        self,
        values: dict[str, Tensor],
        masks: dict[str, Tensor],
        condition: Tensor,
    ) -> Tensor:
        """Encode + fuse every time step at once → [B, T, emb]."""

        scales = self.feature_prior.scales() if self.feature_prior is not None else None
        embeddings = []
        for name in self.modalities:
            vals = values[name]
            if scales is not None:
                vals = vals * scales[name].to(device=vals.device, dtype=vals.dtype).view(1, 1, -1)
            embeddings.append(self.encoders[name](vals, masks[name]))
        if self.pathway_dims:
            for name in self.pathway_dims:
                activity, activity_mask = self._pathway_activity(name, values[name], masks[name])
                embeddings.append(self.pathway_encoders[name](activity, activity_mask))
        n_steps = embeddings[0].shape[1]
        cond_steps = condition.unsqueeze(1).expand(-1, n_steps, -1)
        joint = torch.cat([*embeddings, cond_steps], dim=-1)
        fused = torch.zeros_like(embeddings[0])
        for gate, projection, emb in zip(
            self.fusion.gates, self.fusion.projections, embeddings, strict=True
        ):
            fused = fused + torch.sigmoid(gate(joint)) * projection(emb)
        return fused

    def forward(
        self,
        *,
        values: dict[str, Tensor],
        masks: dict[str, Tensor],
        step_dt: Tensor,
        pad: Tensor,
        target_dt: Tensor,
        condition: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return (target prediction [B, n_targets], reconstructions per modality)."""

        fused = self._fused_steps(values, masks, condition)
        batch, n_steps, _ = fused.shape
        h = fused.new_zeros(batch, self.hidden_dim)
        pad_f = pad.float().unsqueeze(-1)

        if self.mode == "latent_ode":
            # Encoder pass: plain GRU over (fused, dt) → z0.
            for j in range(n_steps):
                inp = torch.cat([fused[:, j], step_dt[:, j : j + 1]], dim=-1)
                h_new = self.cell(inp, h)
                h = pad_f[:, j] * h_new + (1.0 - pad_f[:, j]) * h
            assert self.z0_net is not None
            z = self.z0_net(h)
            states = []
            for j in range(n_steps):
                if j > 0:
                    z_new = odeint_rk4(
                        self.field, z, step_dt[:, j : j + 1], substeps=self.rk4_substeps
                    )
                    z = pad_f[:, j] * z_new + (1.0 - pad_f[:, j]) * z
                states.append(z)
            final = odeint_rk4(self.field, z, target_dt.unsqueeze(-1), substeps=self.rk4_substeps)
        else:
            states = []
            for j in range(n_steps):
                if self.mode == "ode_rnn" and j > 0:
                    h_evolved = odeint_rk4(
                        self.field, h, step_dt[:, j : j + 1], substeps=self.rk4_substeps
                    )
                    h = pad_f[:, j] * h_evolved + (1.0 - pad_f[:, j]) * h
                inp = torch.cat([fused[:, j], step_dt[:, j : j + 1]], dim=-1)
                h_new = self.cell(inp, h)
                h = pad_f[:, j] * h_new + (1.0 - pad_f[:, j]) * h
                states.append(h)
            if self.mode == "ode_rnn":
                final = odeint_rk4(
                    self.field, h, target_dt.unsqueeze(-1), substeps=self.rk4_substeps
                )
            else:  # gru: zero-observation update carrying the actual gap
                zero_obs = fused.new_zeros(batch, fused.shape[-1])
                final = self.cell(torch.cat([zero_obs, target_dt.unsqueeze(-1)], dim=-1), h)

        path = torch.stack(states, dim=1)  # [B, T, hidden]
        reconstructions = {name: self.decoders[name](path) for name in self.modalities}
        head_in = torch.cat([final, target_dt.unsqueeze(-1), condition], dim=-1)
        prediction = self.target_head(head_in)
        return prediction, reconstructions

    def _pathway_activity(
        self, modality: str, values: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        membership = getattr(self, f"path_mem_{modality}")
        filled = values * mask
        weighted = torch.einsum("btf,pf->btp", filled, membership)
        support = torch.einsum("btf,pf->btp", mask, membership)
        activity = weighted / support.clamp(min=1e-6)
        activity_mask = (support > 0).to(dtype=values.dtype)
        return activity, activity_mask

    def prior_loss(self) -> Tensor:
        """Graph Laplacian on gated feature states. Zero when the graph is off."""

        if self.feature_prior is None or self.laplacian_w is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        states = self.feature_prior.stacked_states()
        return laplacian_penalty(states, self.laplacian_w)
