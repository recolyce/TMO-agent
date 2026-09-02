"""Constrained LLM orchestration. Not implemented in milestone 1.

Milestone 1 is a deterministic pipeline. There is no LLM call here and no
silent agent fallback that would pretend curation succeeded.
"""

from omics_agent.errors import OmicsAgentError


def require_agents() -> None:
    """Refuse Agent features until a later milestone implements them."""

    raise OmicsAgentError(
        "LLM orchestration is not part of milestone 1.",
        how_to_fix=(
            "Use omics-agent run-toy or omics-agent benchmark for the deterministic "
            "pipeline. Curator / critic / literature agents arrive in later milestones."
        ),
    )
