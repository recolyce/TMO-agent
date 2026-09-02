"""Per-modality preprocessing configuration.

There is deliberately no zero-fill option. Missing protein intensities
stay NaN unless a model explicitly requests train-mean imputation for its
input features (never for scored targets).
"""

from __future__ import annotations

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import AssayStrategy


class ModalityPreprocessConfig(StrictModel):
    """How one modality is normalized before the train-only scaler.

    Attributes
    ----------
    strategy:
        ``bulk_rna_counts``: library-size CPM then log1p (per-sample math).
        ``log_expression``: values are already on a log-like scale; pass through.
        ``protein_intensity``: linear intensities; zeros become missing, then log2.
    cpm_target:
        Counts-per-``cpm_target`` scaling for ``bulk_rna_counts``.
    log1p_after_cpm:
        Apply log1p after CPM (``bulk_rna_counts`` only).
    zeros_are_missing:
        ``protein_intensity`` only: a 0 intensity means "not quantified",
        not a measured zero. It becomes NaN and is never filled with 0.
    log2_transform:
        ``protein_intensity`` only: apply log2 to positive intensities.
    scale:
        Fit a train-only z-score scaler and write the ``scaled`` layer.
    """

    strategy: AssayStrategy
    cpm_target: float = Field(default=1_000_000.0, gt=0)
    log1p_after_cpm: bool = True
    zeros_are_missing: bool = True
    log2_transform: bool = True
    scale: bool = True
