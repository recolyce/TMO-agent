"""IG stability, prior flags, and the hypothesis-only candidate table."""

from __future__ import annotations

import numpy as np
import pytest

from omics_agent.interpretation.perturb import prior_flags
from omics_agent.interpretation.stability import assemble_candidates, select_stable
from omics_agent.priors.synthetic import build_synthetic_prior_bundle
from omics_agent.schemas.interpretation import InterpretationConfig
from omics_agent.schemas.literature import ABSENCE_OF_EVIDENCE


def test_candidate_table_marks_prior_embedding_and_ablation() -> None:
    bundle = build_synthetic_prior_bundle()
    sources = [("rna", bundle.features["rna"][0]), ("rna", "G99")]
    targets = [("protein", bundle.features["protein"][0])]
    rng = np.random.default_rng(0)
    attr = rng.normal(size=(12, 2, 1, 3))
    attr[:, 0, 0, :] = 1.0
    group_ids = [f"U{i % 4}" for i in range(12)]
    ablation = np.array([[0.4], [0.01]])
    permutation = np.array([[0.2], [0.0]])
    table = assemble_candidates(
        experiment_id="m7",
        model_name="gru",
        attr=attr,
        sources=sources,
        targets=targets,
        group_ids=group_ids,
        ablation=ablation,
        permutation=permutation,
        config=InterpretationConfig(
            n_bootstrap=5,
            n_seeds=5,
            n_folds=2,
            min_sign_consistency=0.5,
            min_selection_frequency=0.3,
            min_stability=0.3,
        ),
        seed=1,
        bundle=bundle,
        embedding_used=True,
    )
    assert table.objective_split == "val"
    assert table.test_labels_visible is False
    assert table.claim_kind == "hypothesis"
    by_id = {row.candidate_id: row for row in table.rows}
    prior_row = by_id[f"rna:{sources[0][1]}->protein:{targets[0][1]}"]
    novel = by_id["rna:G99->protein:" + targets[0][1]]
    assert prior_row.prior_edge_used is True or prior_row.de_novo_model_edge is True
    assert prior_row.ablation_delta == pytest.approx(0.4)
    assert "ablation_delta" in prior_row.model_dump()
    assert "embedding_supported" in prior_row.model_dump()
    assert novel.de_novo_model_edge is True
    assert all(row.claim_kind == "hypothesis" for row in table.rows)
    joined = " ".join(table.notes)
    assert "是首次发现" not in joined
    assert "causation" in joined.lower()


def test_prior_flags_mark_coding_edge_and_embedding() -> None:
    bundle = build_synthetic_prior_bundle()
    edge = next(e for e in bundle.edges if e.source_modality == "rna")
    used, emb, de_novo = prior_flags(
        source=(edge.source_modality, edge.source_id),
        target=(edge.target_modality, edge.target_id),
        bundle=bundle,
        embedding_used=True,
    )
    assert used is True
    assert de_novo is False
    assert emb is True
    used2, emb2, de_novo2 = prior_flags(
        source=("rna", "MISSING"),
        target=("protein", "MISSING"),
        bundle=bundle,
        embedding_used=True,
    )
    assert used2 is False
    assert de_novo2 is True
    assert emb2 is False


def test_runner_never_mentions_test_split() -> None:
    import inspect

    from omics_agent.interpretation import runner

    source = inspect.getsource(runner.run_explanation)
    assert "SplitName.TEST" not in source
    assert "SplitName.VAL" in source


def test_select_stable_is_top_n_passed_only() -> None:
    bundle = build_synthetic_prior_bundle()
    sources = [("rna", "G01"), ("rna", "G02")]
    targets = [("protein", "P01")]
    attr = np.ones((8, 2, 1, 3))
    attr[:, 1, 0, :] = 0.0
    table = assemble_candidates(
        experiment_id="m7",
        model_name="gru",
        attr=attr,
        sources=sources,
        targets=targets,
        group_ids=[f"U{i % 4}" for i in range(8)],
        ablation=np.ones((2, 1)),
        permutation=np.ones((2, 1)),
        config=InterpretationConfig(
            min_sign_consistency=0.9, min_selection_frequency=0.9, min_stability=0.9
        ),
        seed=0,
        bundle=bundle,
        embedding_used=False,
    )
    # High thresholds: possibly none pass; function must not invent rows.
    stable = select_stable(table, top_n=3)
    assert all(row.passed_stability for row in stable)
    assert len(stable) <= 3
    assert ABSENCE_OF_EVIDENCE.startswith("在本次检索范围内")
