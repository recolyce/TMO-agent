"""Integrated Gradients, ablation, permutation. Not implemented in milestone 1."""

from omics_agent.errors import OmicsAgentError


def require_interpretation() -> None:
    raise OmicsAgentError(
        "Attribution methods are not part of milestone 1.",
        how_to_fix="Ridge and time-spline expose coefficients via explain(); IG/ablation come later. Coefficients are not causal.",
    )
