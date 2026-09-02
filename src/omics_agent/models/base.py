"""ModelPlugin protocol and in-process registry.

Every model implements fit / predict / save / explain. The evaluator is a
separate module and is never imported by a model for scoring during fit
in a way that could mutate metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from omics_agent.errors import SchemaError
from omics_agent.models.tasks import DataForModel, PredictionArrays
from omics_agent.schemas.experiment import ModelParams, TaskConfig
from omics_agent.schemas.model import AttributionTable, FitResult

_REGISTRY: dict[str, type[ModelPlugin]] = {}


@runtime_checkable
class ModelPlugin(Protocol):
    """Required surface for a registered baseline or future dynamics model."""

    name: str

    def fit(self, train: DataForModel, val: DataForModel | None, cfg: ModelParams) -> FitResult:
        """Estimate parameters from train only. ``val`` is optional and unused by M1 baselines."""

    def predict(self, data: DataForModel) -> PredictionArrays:
        """Return predictions aligned to ``data.forecast`` instances."""

    def save(self, path: Path) -> None:
        """Serialize the fitted model into ``path`` (a directory)."""

    def explain(self, data: DataForModel, targets: list[str]) -> AttributionTable:
        """Return coefficients or an explicit 'no parameters' table. Not causation."""


def register_model(cls: type[ModelPlugin]) -> type[ModelPlugin]:
    """Class decorator that registers a plugin under ``cls.name``."""

    name = getattr(cls, "name", None)
    if not name:
        raise SchemaError(
            "Model class is missing a name attribute.",
            how_to_fix="Set name = 'last_value' (or another registered id) on the class.",
        )
    _REGISTRY[str(name)] = cls
    return cls


_TORCH_MODEL_NAMES = {"gru", "ode_rnn", "latent_ode"}


def get_model(name: str) -> ModelPlugin:
    """Instantiate a registered model. Unknown names fail with a fix hint."""

    if name not in _REGISTRY:
        if name in _TORCH_MODEL_NAMES:
            raise SchemaError(
                f"Model '{name}' needs PyTorch, which is not installed.",
                how_to_fix="Run: uv sync --extra dev --extra torch",
            )
        known = ", ".join(sorted(_REGISTRY)) or "(none loaded)"
        raise SchemaError(
            f"Unknown model '{name}'.",
            how_to_fix=f"Registered models: {known}.",
        )
    return _REGISTRY[name]()


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def assert_task_supported(plugin_name: str, task: TaskConfig) -> None:
    """Hook for later models that reject incompatible designs. M1 allows both tasks."""

    del plugin_name, task
