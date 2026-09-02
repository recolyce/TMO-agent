"""Uni-Mol embedding adapter: mock extractor, no weights, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omics_agent.errors import PriorError
from omics_agent.priors.embeddings import (
    apply_embedding_model,
    extract_unimol_embeddings,
    list_embedding_models,
    load_smiles_map,
    preferred_embedding_model,
)
from omics_agent.priors.synthetic import build_synthetic_prior_bundle
from omics_agent.priors.unimol import UNIMOL_LICENSE, UNIMOL_PINNED_COMMIT, UniMolEmbeddingAdapter
from omics_agent.schemas.enums import EmbeddingModelName
from omics_agent.schemas.priors import EmbeddingModelConfig


def _fake_repr(smiles: list[str]) -> np.ndarray:
    table = {
        "CCO": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "c1ccccc1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        "CC(=O)Oc1ccccc1C(=O)O": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    }
    rows = []
    for item in smiles:
        if item not in table:
            raise AssertionError(f"mock Uni-Mol was asked for unknown SMILES {item!r}")
        rows.append(table[item])
    return np.stack(rows, axis=0)


def _write_map(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    lines = ["modality\tfeature_id\tsmiles"]
    lines.extend(f"{mod}\t{fid}\t{smi}" for mod, fid, smi in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_unimol_is_the_preferred_candidate() -> None:
    preferred = preferred_embedding_model()
    assert preferred.name is EmbeddingModelName.UNIMOL
    assert preferred.preferred is True
    assert preferred.status == "implemented"
    assert preferred.input_kind == "smiles"
    names = [item.name for item in list_embedding_models()]
    assert names[0] is EmbeddingModelName.UNIMOL
    assert EmbeddingModelName.ESM in names
    assert EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT in names


def test_unimol_extracts_cls_repr_from_smiles_map(tmp_path: Path) -> None:
    smiles_map = _write_map(
        tmp_path / "smiles.tsv",
        [
            ("rna", "G01", "CCO"),
            ("rna", "G02", "c1ccccc1"),
            ("protein", "P01", "CC(=O)Oc1ccccc1C(=O)O"),
        ],
    )
    features = {"rna": ["G01", "G02", "G03"], "protein": ["P01"]}
    tables, spec = extract_unimol_embeddings(
        features,
        EmbeddingModelConfig(name=EmbeddingModelName.UNIMOL, smiles_map=smiles_map),
        experiment_dir=tmp_path,
        species_taxon_id=9606,
        repr_fn=_fake_repr,
    )
    assert spec.model_name == "unimol"
    assert spec.license == UNIMOL_LICENSE
    assert spec.extraction_layer == "cls_repr"
    assert spec.input_kind == "smiles"
    assert spec.frozen is True
    assert spec.dim == 4
    assert spec.n_unmapped == 1
    assert tables["rna"][0] == [1.0, 0.0, 0.0, 0.0]
    assert tables["rna"][1] == [0.0, 1.0, 0.0, 0.0]
    assert tables["rna"][2] == [0.0, 0.0, 0.0, 0.0]
    assert tables["protein"][0] == [0.0, 0.0, 1.0, 0.0]


def test_unimol_refuses_gene_ids_without_smiles_map() -> None:
    with pytest.raises(PriorError, match="SMILES"):
        extract_unimol_embeddings(
            {"rna": ["G01", "G02"], "protein": ["P01"]},
            EmbeddingModelConfig(name=EmbeddingModelName.UNIMOL),
            experiment_dir=Path("."),
            species_taxon_id=9606,
            repr_fn=_fake_repr,
        )


def test_unimol_refuses_when_map_has_no_matching_features(tmp_path: Path) -> None:
    smiles_map = _write_map(tmp_path / "smiles.tsv", [("metabolite", "ethanol", "CCO")])
    with pytest.raises(PriorError, match="gene/protein"):
        extract_unimol_embeddings(
            {"rna": ["G01"], "protein": ["P01"]},
            EmbeddingModelConfig(name=EmbeddingModelName.UNIMOL, smiles_map=smiles_map),
            experiment_dir=tmp_path,
            species_taxon_id=9606,
            repr_fn=_fake_repr,
        )


def test_esm_is_registered_and_raises() -> None:
    bundle = build_synthetic_prior_bundle()
    with pytest.raises(PriorError, match="not implemented"):
        apply_embedding_model(
            bundle,
            EmbeddingModelConfig(name=EmbeddingModelName.ESM),
            experiment_dir=Path("."),
        )


def test_synthetic_embedding_model_leaves_fixture() -> None:
    bundle = build_synthetic_prior_bundle()
    out = apply_embedding_model(
        bundle,
        EmbeddingModelConfig(name=EmbeddingModelName.SYNTHETIC_PATHWAY_ONEHOT),
        experiment_dir=Path("."),
    )
    assert out.embedding_spec is not None
    assert out.embedding_spec.model_name == "synthetic_pathway_onehot"
    assert out.embeddings == bundle.embeddings


def test_apply_unimol_replaces_fixture_table(tmp_path: Path) -> None:
    smiles_map = _write_map(tmp_path / "smiles.tsv", [("rna", "G01", "CCO")])
    bundle = build_synthetic_prior_bundle(rna_features=["G01", "G02"], protein_features=["P01"])
    out = apply_embedding_model(
        bundle,
        EmbeddingModelConfig(name=EmbeddingModelName.UNIMOL, smiles_map=smiles_map),
        experiment_dir=tmp_path,
        repr_fn=_fake_repr,
    )
    assert out.embedding_spec is not None
    assert out.embedding_spec.model_name == "unimol"
    assert out.embeddings["rna"][0] == [1.0, 0.0, 0.0, 0.0]
    assert out.edges == bundle.edges
    assert out.pathways == bundle.pathways
    assert any("not guessed" in note for note in out.notes)


def test_load_smiles_map_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("id\tsmiles\nG01\tCCO\n", encoding="utf-8")
    with pytest.raises(PriorError, match="missing columns"):
        load_smiles_map(path)


def test_load_smiles_map_reads_example() -> None:
    mapping = load_smiles_map(Path("config/feature_smiles.example.tsv"))
    assert mapping[("metabolite", "ethanol")] == "CCO"
    assert mapping[("metabolite", "benzene")] == "c1ccccc1"


def test_unimol_live_path_refuses_unpinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omics_agent.priors import unimol as unimol_mod

    monkeypatch.setattr(unimol_mod, "git_commit", lambda _repo: "0" * 40)
    smiles_map = _write_map(tmp_path / "smiles.tsv", [("rna", "G01", "CCO")])
    with pytest.raises(PriorError, match="pinned"):
        extract_unimol_embeddings(
            {"rna": ["G01"]},
            EmbeddingModelConfig(name=EmbeddingModelName.UNIMOL, smiles_map=smiles_map),
            experiment_dir=tmp_path,
            species_taxon_id=9606,
            repr_fn=None,
        )


def test_adapter_records_local_commit_without_importing_weights() -> None:
    adapter = UniMolEmbeddingAdapter(repr_fn=_fake_repr)
    sha = adapter.source_commit()
    if sha is not None:
        assert len(sha) == 40
    assert UNIMOL_PINNED_COMMIT
    out = adapter.get_cls_repr(["CCO"])
    assert out.shape == (1, 4)
