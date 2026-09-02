"""GRU / ODE-RNN / latent ODE ModelPlugins (pure PyTorch).

Training is deterministic from the seed, uses masked losses (missing
targets are excluded, never imputed for scoring), and stops with a typed
error if the loss or the ODE solver diverges. Only legal for longitudinal
subject_forecast.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from omics_agent.errors import SchemaError, TrainingDivergedError
from omics_agent.models.base import register_model
from omics_agent.models.dynamics.networks import TemporalCore
from omics_agent.models.dynamics.sequences import SequenceBatch, build_sequences
from omics_agent.models.dynamics.torch_env import resolve_device
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams
from omics_agent.schemas.model import AttributionTable, FitResult


def _num(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SchemaError(
            f"Parameter '{key}' must be a number, got {value!r}.",
            how_to_fix=f"Set params.{key} to a number such as {default}.",
        )
    return float(value)


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over observed entries only. NaN targets never enter."""

    keep = mask.bool()
    safe_target = torch.where(keep, torch.nan_to_num(target), torch.zeros_like(target))
    diff = (pred - safe_target) ** 2 * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


class _Tensors:
    """SequenceBatch moved onto one device."""

    def __init__(self, seq: SequenceBatch, device: torch.device) -> None:
        self.values = {k: torch.as_tensor(v, device=device) for k, v in seq.values.items()}
        self.masks = {k: torch.as_tensor(v, device=device) for k, v in seq.masks.items()}
        self.step_dt = torch.as_tensor(seq.step_dt, device=device)
        self.pad = torch.as_tensor(seq.pad, device=device)
        self.target_dt = torch.as_tensor(seq.target_dt, device=device)
        self.condition = torch.as_tensor(seq.condition, device=device)
        self.y_true = torch.as_tensor(seq.y_true, device=device)
        self.y_mask = torch.as_tensor(seq.y_mask, device=device)
        self.n = int(seq.y_true.shape[0])

    def slice(self, idx: np.ndarray) -> dict[str, Any]:
        rows = torch.as_tensor(idx, device=self.step_dt.device, dtype=torch.long)
        return {
            "values": {k: v[rows] for k, v in self.values.items()},
            "masks": {k: v[rows] for k, v in self.masks.items()},
            "step_dt": self.step_dt[rows],
            "pad": self.pad[rows],
            "target_dt": self.target_dt[rows],
            "condition": self.condition[rows],
        }


class _DynamicsBase:
    """Shared fit/predict/save/explain for the three dynamics plugins."""

    name = ""
    mode = ""

    def __init__(self) -> None:
        self._model: TemporalCore | None = None
        self._device: torch.device | None = None
        self._conditions: list[str] = []
        self._feature_dims: dict[str, int] = {}
        self._target_names: list[str] = []
        self._params: dict[str, float] = {}
        self._epoch_callback: Callable[[int, float], None] | None = None

    def set_epoch_callback(self, callback: Callable[[int, float], None] | None) -> None:
        """Called with (epoch, val_masked_mse) at every val check.

        The HPO pruner uses this hook; a raised exception (optuna.TrialPruned)
        propagates out of fit(). Only the validation loss is ever reported.
        """

        self._epoch_callback = callback

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        p = cfg.params
        seed = int(_num(p, "seed", 20260901))
        epochs = int(_num(p, "epochs", 300))
        batch_size = max(1, int(_num(p, "batch_size", 64)))
        lr = _num(p, "lr", 3e-3)
        patience = int(_num(p, "patience", 30))
        val_every = max(1, int(_num(p, "val_every", 5)))
        recon_weight = _num(p, "recon_weight", 0.3)
        grad_clip = _num(p, "grad_clip", 5.0)
        device = resolve_device(str(p.get("device", "auto")))

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)

        if not train.forecast.y_mask.any():
            raise SchemaError(
                f"{self.name} has no observed train targets.",
                how_to_fix="Check missingness rates and the train split.",
            )
        self._conditions = sorted(set(train.forecast.conditions))
        seq = build_sequences(train, condition_categories=self._conditions, model_name=self.name)
        self._feature_dims = {m: seq.values[m].shape[2] for m in seq.modalities}
        self._target_names = list(train.forecast.feature_names)
        model = TemporalCore(
            feature_dims=self._feature_dims,
            n_targets=seq.y_true.shape[1],
            cond_dim=len(self._conditions),
            emb_dim=int(_num(p, "emb_dim", 32)),
            hidden_dim=int(_num(p, "hidden_dim", 48)),
            mode=self.mode,
            rk4_substeps=int(_num(p, "rk4_substeps", 2)),
        ).to(device)
        data = _Tensors(seq, device)
        val_data = None
        if val is not None and len(val.forecast.instance_ids):
            val_data = _Tensors(
                build_sequences(val, condition_categories=self._conditions, model_name=self.name),
                device,
            )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=_num(p, "weight_decay", 1e-4)
        )
        best_val = float("inf")
        best_state: dict[str, Tensor] | None = None
        bad_rounds = 0
        epochs_run = 0
        indices = np.arange(data.n)
        for epoch in range(epochs):
            epochs_run = epoch + 1
            model.train()
            rng.shuffle(indices)
            for start in range(0, data.n, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = data.slice(batch_idx)
                rows = torch.as_tensor(batch_idx, device=device, dtype=torch.long)
                pred, recon = model(**batch)
                loss = _masked_mse(pred, data.y_true[rows], data.y_mask[rows].float())
                if recon_weight > 0:
                    for modality in seq.modalities:
                        recon_mask = batch["masks"][modality] * batch["pad"].unsqueeze(-1).float()
                        loss = loss + recon_weight * _masked_mse(
                            recon[modality], batch["values"][modality], recon_mask
                        )
                if not bool(torch.isfinite(loss)):
                    raise TrainingDivergedError(
                        f"{self.name} training loss became non-finite at epoch {epoch}.",
                        how_to_fix=(
                            "Lower params.lr, raise params.rk4_substeps, or reduce "
                            "hidden_dim. Results from a diverged run are never reported."
                        ),
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if val_data is not None and (epoch + 1) % val_every == 0:
                val_mse = self._eval_mse(model, val_data)
                if self._epoch_callback is not None:
                    self._epoch_callback(epoch, val_mse)
                if val_mse < best_val - 1e-6:
                    best_val = val_mse
                    best_state = copy.deepcopy(model.state_dict())
                    bad_rounds = 0
                else:
                    bad_rounds += 1
                    if bad_rounds >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._device = device
        self._params = {
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "recon_weight": recon_weight,
            "emb_dim": int(_num(p, "emb_dim", 32)),
            "hidden_dim": int(_num(p, "hidden_dim", 48)),
            "rk4_substeps": int(_num(p, "rk4_substeps", 2)),
        }
        n_parameters = int(sum(int(t.numel()) for t in model.parameters()))
        extras: dict[str, object] = {
            "mode": self.mode,
            "device": str(device),
            "epochs_run": epochs_run,
            "conditions": self._conditions,
        }
        if best_state is not None:
            extras["best_val_mse"] = best_val
        if self.mode == "latent_ode":
            extras["formulation"] = (
                "deterministic encoder-ODE-decoder (no VAE sampling); reproducible from the seed"
            )
        return FitResult(
            model_name=self.name,
            n_train_instances=len(train.forecast.instance_ids),
            n_parameters=n_parameters,
            extras=extras,
        )

    @staticmethod
    def _eval_mse(model: TemporalCore, data: _Tensors) -> float:
        model.eval()
        with torch.no_grad():
            pred, _ = model(
                values=data.values,
                masks=data.masks,
                step_dt=data.step_dt,
                pad=data.pad,
                target_dt=data.target_dt,
                condition=data.condition,
            )
            mse = _masked_mse(pred, data.y_true, data.y_mask.float())
        return float(mse)

    def predict(self, data: DataForModel) -> PredictionArrays:
        if self._model is None or self._device is None:
            raise SchemaError(
                f"{self.name} was used before fit().",
                how_to_fix="Call fit() on the train split first.",
            )
        seq = build_sequences(data, condition_categories=self._conditions, model_name=self.name)
        tensors = _Tensors(seq, self._device)
        self._model.eval()
        with torch.no_grad():
            pred, _ = self._model(
                values=tensors.values,
                masks=tensors.masks,
                step_dt=tensors.step_dt,
                pad=tensors.pad,
                target_dt=tensors.target_dt,
                condition=tensors.condition,
            )
        out = pred.cpu().numpy().astype(float)
        if not np.isfinite(out).all():
            raise TrainingDivergedError(
                f"{self.name} produced non-finite predictions.",
                how_to_fix="Refit with a lower lr or more rk4_substeps; do not report this run.",
            )
        return PredictionArrays(y_pred=out, extras={})

    def save(self, path: Path) -> None:
        if self._model is None:
            raise SchemaError(
                f"{self.name} cannot be saved before fit().",
                how_to_fix="Fit on train, then save the artifact.",
            )
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "model.pt")
        card = {
            "name": self.name,
            "mode": self.mode,
            "conditions": self._conditions,
            "feature_dims": self._feature_dims,
            "target_names": self._target_names,
            "params": self._params,
            "fit_split": "train",
        }
        (path / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Rebuild the network from card.json and restore the checkpoint."""

        card_path = path / "card.json"
        if not card_path.is_file():
            raise SchemaError(
                f"No card.json under {path}.",
                how_to_fix="Point at a directory written by save() (model.pt + card.json).",
            )
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("mode") != self.mode:
            raise SchemaError(
                f"Checkpoint mode '{card.get('mode')}' does not match plugin '{self.mode}'.",
                how_to_fix="Load the checkpoint with the model it was saved from.",
            )
        self._conditions = list(card["conditions"])
        self._feature_dims = {str(k): int(v) for k, v in card["feature_dims"].items()}
        self._target_names = list(card["target_names"])
        self._params = dict(card["params"])
        device = resolve_device("auto")
        model = TemporalCore(
            feature_dims=self._feature_dims,
            n_targets=len(self._target_names),
            cond_dim=len(self._conditions),
            emb_dim=int(self._params.get("emb_dim", 32)),
            hidden_dim=int(self._params.get("hidden_dim", 48)),
            mode=self.mode,
            rk4_substeps=int(self._params.get("rk4_substeps", 2)),
        ).to(device)
        state = torch.load(path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._device = device

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        del data, targets
        return AttributionTable(
            model_name=self.name,
            method="none",
            rows=[],
            caveat=(
                "Attribution for dynamics models arrives with the interpretation "
                "milestone (integrated gradients). Attribution is not causation."
            ),
        )


@register_model
class GruDynamicsModel(_DynamicsBase):
    """GRU with delta_t as a cell input; zero-observation jump to the target."""

    name = "gru"
    mode = "gru"


@register_model
class OdeRnnDynamicsModel(_DynamicsBase):
    """ODE-RNN: hidden state evolves under an ODE field between observations."""

    name = "ode_rnn"
    mode = "ode_rnn"


@register_model
class LatentOdeDynamicsModel(_DynamicsBase):
    """Latent ODE: GRU encoder → z0 → latent ODE path → decoders."""

    name = "latent_ode"
    mode = "latent_ode"
