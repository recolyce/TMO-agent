"""Fixed-grid RK4 integrator with explicit non-finite detection.

Pure PyTorch, no torchdiffeq. Dynamics fields here are small tanh MLPs, so
a fixed-step solver is adequate; a NaN/inf state is a hard error rather
than a silently propagated garbage prediction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omics_agent.errors import OdeSolverError


def odeint_rk4(func: Callable[[Any], Any], state: Any, dt: Any, *, substeps: int = 2) -> Any:
    """Integrate ``dstate/dt = func(state)`` over per-sample horizons ``dt``.

    Parameters
    ----------
    state:
        Tensor of shape ``[batch, dim]``.
    dt:
        Tensor of shape ``[batch, 1]``; may differ per sample (irregular time).
    substeps:
        Number of RK4 steps the horizon is divided into.
    """

    if substeps < 1:
        raise OdeSolverError(
            f"substeps must be >= 1, got {substeps}.",
            how_to_fix="Set params.rk4_substeps to a small positive integer such as 2.",
        )
    h = dt / float(substeps)
    for _ in range(substeps):
        k1 = func(state)
        k2 = func(state + 0.5 * h * k1)
        k3 = func(state + 0.5 * h * k2)
        k4 = func(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not bool(state.isfinite().all()):
            raise OdeSolverError(
                "ODE integration produced NaN/inf latent states.",
                how_to_fix=(
                    "Lower the learning rate, increase params.rk4_substeps, or reduce "
                    "hidden_dim. The pipeline will not report predictions from a "
                    "diverged solver."
                ),
            )
    return state
