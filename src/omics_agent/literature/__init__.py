"""PubMed / Europe PMC evidence tables. Not implemented in milestone 1."""

from omics_agent.errors import OmicsAgentError


def require_literature() -> None:
    raise OmicsAgentError(
        "Literature search is not part of milestone 1.",
        how_to_fix=(
            "Do not describe missing papers as a first discovery. "
            "When this module exists, absence of evidence must be written as "
            "'在本次检索范围内未找到直接证据'."
        ),
    )
