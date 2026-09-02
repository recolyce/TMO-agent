"""Optional-torch guard and device selection.

torch is an optional extra so milestone 1–3 environments stay light.
Importing a dynamics plugin without torch fails with the install command,
not an opaque ModuleNotFoundError.
"""

from __future__ import annotations

from typing import Any

from omics_agent.errors import SchemaError


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SchemaError(
            "PyTorch is not installed; gru / ode_rnn / latent_ode need it.",
            how_to_fix="Run: uv sync --extra dev --extra torch",
        ) from exc
    return torch


def resolve_device(requested: str) -> Any:
    """'auto' picks CUDA when available; anything else is passed through."""

    torch = require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
