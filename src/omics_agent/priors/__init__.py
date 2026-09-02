"""Versioned biological priors and the five-arm ablation (milestone 6).

``run_prior_ablation`` is imported lazily so ``omics_agent.models`` can load
the torch plugins without a circular import through this package.
"""

from pathlib import Path
from typing import Any

from omics_agent.priors.runtime import PriorRuntime, align_prior
from omics_agent.priors.synthetic import build_synthetic_prior_bundle, write_synthetic_prior_bundle
from omics_agent.schemas.priors import PriorBundle, flags_for, load_prior_bundle

__all__ = [
    "PriorBundle",
    "PriorRuntime",
    "align_prior",
    "build_synthetic_prior_bundle",
    "flags_for",
    "load_prior_bundle",
    "run_prior_ablation",
    "write_synthetic_prior_bundle",
]


def run_prior_ablation(
    experiment_path: Path,
    *,
    model_name: str,
    output_dir: Path | None = None,
    n_trials: int | None = None,
    embedding_model: str | None = None,
    smiles_map: Path | None = None,
    unimol_repr_fn: Any = None,
) -> dict[str, Any]:
    from omics_agent.priors.ablation import run_prior_ablation as _run

    return _run(
        experiment_path,
        model_name=model_name,
        output_dir=output_dir,
        n_trials=n_trials,
        embedding_model=embedding_model,
        smiles_map=smiles_map,
        unimol_repr_fn=unimol_repr_fn,
    )
