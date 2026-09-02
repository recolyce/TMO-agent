"""Backward-compatible import surface for remote ingest.

Milestone 2 implements GEO, BioStudies, PRIDE, HTTPS, and local adapters.
SRA / raw mass-spec remain unsupported and raise.
"""

from omics_agent.data_sources.ingest import run_ingest
from omics_agent.errors import UnsupportedRawDataError

__all__ = ["UnsupportedRawDataError", "run_ingest"]
