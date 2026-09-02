from __future__ import annotations

import json

import pandas as pd
import pytest

from omics_agent.errors import DownloadError, SchemaError
from omics_agent.preprocessing.id_mapping import (
    IdentityMapper,
    MyGeneInfoAdapter,
    StaticTableIdMapper,
    build_feature_map,
)
from omics_agent.schemas.ingest import DownloadPolicy
from tests.unit.http_fakes import FakeRoute, FakeTransport


def _policy() -> DownloadPolicy:
    return DownloadPolicy(retries=0, retry_backoff_s=0.001, requests_per_host_per_sec=10)


def test_static_table_keeps_one_to_many_and_unmapped() -> None:
    table = pd.DataFrame(
        {
            "source_id": ["P1", "P1", "P2"],
            "target_id": ["G1", "G2", "G3"],
        }
    )
    mapper = StaticTableIdMapper(table, target_id_type="ensembl_gene_id")
    feature_map = build_feature_map(
        modality="protein",
        source_ids=["P1", "P2", "P3"],
        source_id_type="uniprot_accession",
        adapter=mapper,
    )
    by_source = {item.source_id: item for item in feature_map.mappings}
    assert [t.target_id for t in by_source["P1"].targets] == ["G1", "G2"]
    assert by_source["P1"].is_ambiguous
    assert not by_source["P2"].is_ambiguous
    assert by_source["P3"].is_unmapped
    summary = feature_map.summary()
    assert summary == {
        "n_features": 3,
        "n_mapped": 2,
        "n_unmapped": 1,
        "n_ambiguous": 1,
        "n_pairs": 3,
    }
    frame = feature_map.to_frame()
    assert len(frame) == 4  # 2 + 1 + 1 unmapped placeholder row
    assert frame.loc[frame["source_id"] == "P3", "unmapped"].all()
    with pytest.raises(SchemaError, match="one-to-many"):
        feature_map.assert_ready_for_training()


def test_static_table_requires_columns() -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        StaticTableIdMapper(pd.DataFrame({"a": ["x"]}))


def test_identity_mapper_maps_each_id_to_itself() -> None:
    mapper = IdentityMapper(target_id_type="synthetic_gene")
    assert mapper.map_ids(["G1", "G2"]) == {"G1": ["G1"], "G2": ["G2"]}


def test_mygene_adapter_is_mockable_and_offline() -> None:
    adapter = MyGeneInfoAdapter(
        species_taxon_id=9606, transport=FakeTransport({}), policy=_policy()
    )
    routes = {
        adapter.query_url("TP53"): FakeRoute(
            body=json.dumps(
                {"hits": [{"ensembl": [{"gene": "ENSG000A"}, {"gene": "ENSG000B"}]}]}
            ).encode()
        ),
        adapter.query_url("BRCA1"): FakeRoute(
            body=json.dumps({"hits": [{"ensembl": {"gene": "ENSG000C"}}]}).encode()
        ),
        adapter.query_url("NOPE"): FakeRoute(body=json.dumps({"hits": []}).encode()),
    }
    transport = FakeTransport(routes)
    adapter = MyGeneInfoAdapter(species_taxon_id=9606, transport=transport, policy=_policy())
    mapping = adapter.map_ids(["TP53", "BRCA1", "NOPE"])
    assert mapping == {
        "TP53": ["ENSG000A", "ENSG000B"],
        "BRCA1": ["ENSG000C"],
        "NOPE": [],
    }
    assert transport.calls, "the adapter must go through the injected transport"
    assert all(url.startswith("https://mygene.info/") for _, url in transport.calls)


def test_mygene_api_failure_is_an_error_not_an_empty_map() -> None:
    adapter = MyGeneInfoAdapter(
        species_taxon_id=10090, transport=FakeTransport({}), policy=_policy()
    )
    transport = FakeTransport({adapter.query_url("Trp53"): FakeRoute(status=500, body=b"down")})
    adapter = MyGeneInfoAdapter(species_taxon_id=10090, transport=transport, policy=_policy())
    with pytest.raises(DownloadError, match="mygene.info returned HTTP 500"):
        adapter.map_ids(["Trp53"])
