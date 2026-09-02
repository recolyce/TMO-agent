"""Uni-Mol embedding adapter.

Uses a local checkout (default ``/root/workspace/Uni-Mol``). Does not run
``setup.py`` or any install script (rule 7). CI injects ``repr_fn`` so
weights are never downloaded.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from omics_agent.errors import PriorError
from omics_agent.hashing import git_commit

UNIMOL_PINNED_COMMIT = "90f52c41299a1a582da0f9765e9f87aa21faa16a"
UNIMOL_LICENSE = "MIT"
UNIMOL_DEFAULT_ROOT = Path("/root/workspace/Uni-Mol")
UNIMOL_V1_DIM = 512

UniMolReprFn = Callable[[list[str]], np.ndarray]


class UniMolEmbeddingAdapter:
    """``cls_repr`` extractor. Live path imports ``unimol_tools`` from disk."""

    def __init__(
        self,
        root: Path = UNIMOL_DEFAULT_ROOT,
        *,
        variant: str = "unimolv1",
        size: str = "84m",
        repr_fn: UniMolReprFn | None = None,
    ) -> None:
        self.root = Path(root)
        self.variant = variant
        self.size = size
        self.repr_fn = repr_fn

    def source_commit(self) -> str | None:
        return git_commit(self.root)

    def get_cls_repr(self, smiles: list[str]) -> np.ndarray:
        if not smiles:
            return np.zeros((0, UNIMOL_V1_DIM), dtype=np.float64)
        raw = self.repr_fn(smiles) if self.repr_fn is not None else self._live_repr(smiles)
        array = _as_cls_matrix(raw)
        if array.shape[0] != len(smiles):
            raise PriorError(
                f"Uni-Mol returned {array.shape[0]} vectors for {len(smiles)} SMILES.",
                how_to_fix="The extractor must return one cls_repr row per input SMILES, same order.",
            )
        return array

    def _live_repr(self, smiles: list[str]) -> object:
        clf = self._import_repr()
        return clf.get_repr(list(smiles), return_atomic_reprs=False)

    def _import_repr(self) -> object:
        tools_root = self.root / "unimol_tools"
        package_dir = tools_root / "unimol_tools"
        if not package_dir.is_dir():
            raise PriorError(
                f"Local Uni-Mol checkout is missing unimol_tools: {self.root}",
                how_to_fix=(
                    "Point priors.embedding.unimol_root at a Uni-Mol git checkout "
                    f"(expected HEAD {UNIMOL_PINNED_COMMIT[:12]}). "
                    "omics-agent will not run Uni-Mol setup.py. "
                    "For CI, inject a mock repr_fn. "
                    "For the synthetic gene/protein fixture, set "
                    "priors.embedding.name: synthetic_pathway_onehot."
                ),
            )
        tools_path = str(tools_root)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        try:
            from unimol_tools import UniMolRepr
        except ImportError as exc:
            raise PriorError(
                "unimol_tools is not importable from the local Uni-Mol checkout.",
                how_to_fix=(
                    "Install Uni-Mol's own dependencies (rdkit, torch) in this environment. "
                    "Do not ask omics-agent to run setup.py or download scripts. "
                    "Tests inject a mock repr_fn and never load weights."
                ),
            ) from exc
        return UniMolRepr(
            data_type="molecule",
            model_name=self.variant,
            model_size=self.size,
            use_cuda=False,
        )


def _as_cls_matrix(raw: object) -> np.ndarray:
    if isinstance(raw, dict):
        if "cls_repr" not in raw:
            raise PriorError(
                "Uni-Mol output has no cls_repr key.",
                how_to_fix="Use UniMolRepr.get_repr(..., return_atomic_reprs=False).",
            )
        raw = raw["cls_repr"]
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise PriorError(
            f"Uni-Mol cls_repr must be 2-D (n, dim); got shape {array.shape}.",
            how_to_fix="Return one vector per SMILES.",
        )
    if not np.isfinite(array).all():
        raise PriorError(
            "Uni-Mol cls_repr contains NaN or inf.",
            how_to_fix="Check the SMILES and the local Uni-Mol weights. Do not proceed with a garbage table.",
        )
    return array
