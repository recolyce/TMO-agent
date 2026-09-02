"""Versioned biological priors. Not implemented in milestone 1."""

from omics_agent.errors import OmicsAgentError


def require_priors() -> None:
    raise OmicsAgentError(
        "PPI / GRN / pathway / embedding priors are not part of milestone 1.",
        how_to_fix="Train the no-prior baselines first. Prior ablations are a later milestone.",
    )
