"""Validation-only HPO. Not implemented in milestone 1."""

from omics_agent.errors import OmicsAgentError


def require_optimization() -> None:
    raise OmicsAgentError(
        "Optuna HPO is not part of milestone 1.",
        how_to_fix="Use the registered baselines with the YAML hyperparameters. Optimizer must never see test labels.",
    )
