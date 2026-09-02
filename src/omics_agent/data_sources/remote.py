"""Remote download adapters. Not implemented in milestone 1.

Real GEO / PRIDE / BioStudies clients must not be stubbed to a fake success.
"""

from omics_agent.errors import OmicsAgentError


def require_remote_download() -> None:
    raise OmicsAgentError(
        "Real download adapters are not part of milestone 1.",
        how_to_fix=(
            "Use omics-agent generate-synthetic for the CPU fixture, or wait for "
            "milestone 2. The pipeline will not fetch GEO/PRIDE silently."
        ),
    )
