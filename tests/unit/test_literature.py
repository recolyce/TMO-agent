"""PubMed / Europe PMC adapters: mock HTTP only, hypothesis language."""

from __future__ import annotations

import json

from omics_agent.literature.check import (
    build_query,
    classify_hit,
    doi_is_well_formed,
    pmid_is_well_formed,
    run_literature_check,
    verify_hit,
)
from omics_agent.literature.sources import (
    EuropePmcAdapter,
    PaperHit,
    PubMedAdapter,
    europepmc_search_url,
    pubmed_esearch_url,
    pubmed_esummary_url,
)
from omics_agent.schemas.enums import (
    EvidenceLevel,
    LiteratureReviewerStatus,
    LiteratureStance,
    RelationDirection,
)
from omics_agent.schemas.interpretation import CandidateRow
from omics_agent.schemas.literature import ABSENCE_OF_EVIDENCE, LiteratureSearchConfig
from tests.unit.http_fakes import FakeRoute, FakeTransport


def _candidate() -> CandidateRow:
    return CandidateRow(
        candidate_id="rna:G01->protein:P01",
        source_modality="rna",
        source_id="G01",
        target_modality="protein",
        target_id="P01",
        mean_attribution=0.5,
        sign_consistency=1.0,
        rank_median=1.0,
        selection_frequency=1.0,
        bootstrap_low=0.2,
        bootstrap_high=0.8,
        ablation_delta=0.3,
        permutation_delta=0.1,
        stability=1.0,
        prior_edge_used=True,
        embedding_supported=False,
        de_novo_model_edge=False,
        passed_stability=True,
        predicted_direction=RelationDirection.UP.value,
    )


def test_query_does_not_invent_synonyms() -> None:
    query = build_query("G01", "P01", organism="human", tissue=None, condition=None)
    assert "G01" in query and "P01" in query and "human" in query
    assert "TP53" not in query


def test_no_hit_is_level_n_not_novelty() -> None:
    query = build_query("G01", "P01", organism="human", tissue=None, condition=None)
    transport = FakeTransport(
        {
            pubmed_esearch_url(query, retmax=5): FakeRoute(
                body=json.dumps({"esearchresult": {"idlist": []}}).encode()
            ),
            europepmc_search_url(query, page_size=5): FakeRoute(
                body=json.dumps({"resultList": {"result": []}}).encode()
            ),
        }
    )
    table = run_literature_check(
        [_candidate()],
        experiment_id="m7",
        config=LiteratureSearchConfig(top_n=5, max_hits_per_source=5),
        transport=transport,
        searched_at="2026-09-02T00:00:00+00:00",
    )
    assert table.claim_kind == "hypothesis"
    assert table.records[0].evidence_level is EvidenceLevel.N
    assert table.records[0].evidence_sentence == ABSENCE_OF_EVIDENCE
    assert table.records[0].reviewer_status is LiteratureReviewerStatus.NEEDS_REVIEW
    assert table.records[0].evidence_sentence == ABSENCE_OF_EVIDENCE
    assert "是首次发现" not in table.records[0].evidence_sentence
    assert all("是首次发现" not in note for note in table.notes)
    assert transport.calls  # adapters were invoked


def test_unstable_candidates_are_not_queried() -> None:
    transport = FakeTransport({})
    table = run_literature_check(
        [],
        experiment_id="m7",
        config=LiteratureSearchConfig(),
        transport=transport,
    )
    assert table.records == []
    assert transport.calls == []


def test_pmid_doi_authenticity_and_reviewer_status() -> None:
    assert pmid_is_well_formed("12345")
    assert not pmid_is_well_formed("pmid:123")
    assert doi_is_well_formed("10.1234/abc.def")
    assert not doi_is_well_formed("doi:10.1234/abc")
    query = build_query("G01", "P01", organism="human", tissue=None, condition=None)
    summary = {
        "result": {
            "uids": ["12345"],
            "12345": {
                "uid": "12345",
                "title": "G01 knockout activates P01",
                "pubdate": "2020 Jan",
                "elocationid": "doi: 10.1234/abc.def",
            },
        }
    }
    transport = FakeTransport(
        {
            pubmed_esearch_url(query, retmax=5): FakeRoute(
                body=json.dumps({"esearchresult": {"idlist": ["12345"]}}).encode()
            ),
            pubmed_esummary_url(["12345"]): FakeRoute(body=json.dumps(summary).encode()),
            europepmc_search_url(query, page_size=5): FakeRoute(
                body=json.dumps({"resultList": {"result": []}}).encode()
            ),
        }
    )
    table = run_literature_check(
        [_candidate()],
        experiment_id="m7",
        config=LiteratureSearchConfig(),
        transport=transport,
        searched_at="2026-09-02T00:00:00+00:00",
    )
    rec = table.records[0]
    assert rec.pmid == "12345"
    assert rec.doi == "10.1234/abc.def"
    assert rec.pmid_authentic is True
    assert rec.doi_authentic is True
    assert rec.reviewer_status is LiteratureReviewerStatus.NEEDS_REVIEW
    assert rec.evidence_level is EvidenceLevel.A
    assert rec.stance is LiteratureStance.SUPPORTS
    assert rec.query_string == query
    assert rec.searched_at == "2026-09-02T00:00:00+00:00"


def test_mismatched_pmid_fails_closed() -> None:
    hit = PaperHit(
        source_name="pubmed_eutils",
        pmid="999",
        doi="not-a-doi",
        title="Wrong title",
        year="2020",
        abstract=None,
        raw={},
    )
    transport = FakeTransport(
        {
            pubmed_esummary_url(["999"]): FakeRoute(
                body=json.dumps(
                    {"result": {"uids": ["999"], "999": {"uid": "123", "title": "Other"}}}
                ).encode()
            )
        }
    )
    pubmed = PubMedAdapter(transport=transport)
    pmid_ok, doi_ok = verify_hit(hit, pubmed)
    assert pmid_ok is False
    assert doi_ok is False


def test_classify_absence_phrase() -> None:
    stance, level, *_rest = classify_hit(
        PaperHit("pubmed_eutils", None, None, None, None, None, {}),
        predicted=RelationDirection.UP,
        organism="human",
        tissue=None,
        condition=None,
    )
    assert stance is LiteratureStance.UNRELATED
    assert level is EvidenceLevel.N


def test_adapters_are_constructible() -> None:
    assert PubMedAdapter().name == "pubmed_eutils"
    assert EuropePmcAdapter().name == "europepmc"
