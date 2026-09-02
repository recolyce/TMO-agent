"""Data-source adapters: synthetic, local processed matrices, and ingest."""

from omics_agent.data_sources.ingest import run_ingest
from omics_agent.data_sources.local import load_local_bundle
from omics_agent.data_sources.synthetic import generate_synthetic_dataset

__all__ = ["generate_synthetic_dataset", "load_local_bundle", "run_ingest"]
