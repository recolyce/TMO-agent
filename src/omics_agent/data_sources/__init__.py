"""Data-source adapters.

Milestone 1 ships only the synthetic generator and a local processed-matrix
loader. GEO/PRIDE/BioStudies download adapters are milestone 2 and must not
be silently stubbed to "success".
"""

from omics_agent.data_sources.local import load_local_bundle
from omics_agent.data_sources.synthetic import generate_synthetic_dataset

__all__ = ["generate_synthetic_dataset", "load_local_bundle"]
