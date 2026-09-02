"""Query construction, evidence grading, PMID/DOI checks, literature runner.

Rule-based (no LLM). Level N is 「在本次检索范围内未找到直接证据」.
The pipeline never sets reviewer_status to accepted.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from omics_agent.data_sources.http import HttpTransport
from omics_agent.literature.sources import (
    EUROPEPMC_SEARCH,
    EuropePmcAdapter,
    PaperHit,
    PubMedAdapter,
)
from omics_agent.schemas.enums import (
    EvidenceLevel,
    LiteratureReviewerStatus,
    LiteratureStance,
    RelationDirection,
)
from omics_agent.schemas.interpretation import CandidateRow
from omics_agent.schemas.literature import (
    ABSENCE_OF_EVIDENCE,
    LiteratureRecord,
    LiteratureSearchConfig,
    LiteratureTable,
)

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_PMID_RE = re.compile(r"^\d{1,12}$")

_PERTURB = ("knockout", "knock-out", "crispr", "overexpress", "perturbation", "causal")
_BINDING = ("bind", "chip-seq", "co-ip", "physical interaction", "pull-down")
_CORREL = ("correlat", "co-express", "associated with", "coexpression")
_PATHWAY = ("pathway", "reactome", "string")
_UP = ("activate", "induce", "up-regulate", "upregulate", "increase")
_DOWN = ("inhibit", "repress", "down-regulate", "downregulate", "decrease")


def build_query(
    source_id: str,
    target_id: str,
    *,
    organism: str | None,
    tissue: str | None,
    condition: str | None,
) -> str:
    """Deterministic search string. Synonyms are not invented."""

    parts = [
        f"({source_id})",
        f"({target_id})",
        "(regulat* OR bind* OR interact* OR phosphorylat*)",
    ]
    if organism:
        parts.append(f"({organism})")
    if tissue:
        parts.append(f"({tissue})")
    if condition:
        parts.append(f"({condition})")
    return " AND ".join(parts)


def doi_is_well_formed(doi: str | None) -> bool:
    if not doi:
        return False
    return bool(_DOI_RE.match(doi.strip()))


def pmid_is_well_formed(pmid: str | None) -> bool:
    if not pmid:
        return False
    return bool(_PMID_RE.match(pmid.strip()))


def verify_hit(hit: PaperHit, pubmed: PubMedAdapter | None) -> tuple[bool, bool]:
    """Format + identity checks. A fabricated PMID/DOI fails closed."""

    pmid_ok = pmid_is_well_formed(hit.pmid)
    doi_ok = doi_is_well_formed(hit.doi)
    if pmid_ok and pubmed is not None and hit.pmid:
        summaries = pubmed.summarize([hit.pmid])
        if not summaries:
            pmid_ok = False
        else:
            rec = summaries[0]
            if rec.pmid != hit.pmid:
                pmid_ok = False
            if hit.title and rec.title and _norm(hit.title) != _norm(rec.title):
                pmid_ok = False
            if hit.doi and rec.doi and _norm(hit.doi) != _norm(rec.doi):
                doi_ok = False
    return pmid_ok, doi_ok


def classify_hit(
    hit: PaperHit,
    *,
    predicted: RelationDirection,
    organism: str | None,
    tissue: str | None,
    condition: str | None,
) -> tuple[
    LiteratureStance, EvidenceLevel, RelationDirection, str, bool | None, bool | None, bool | None
]:
    text = " ".join(part for part in (hit.title, hit.abstract) if part).lower()
    if not text:
        return (
            LiteratureStance.UNRELATED,
            EvidenceLevel.N,
            RelationDirection.UNKNOWN,
            ABSENCE_OF_EVIDENCE,
            None,
            None,
            None,
        )
    reported = RelationDirection.UNKNOWN
    if any(word in text for word in _UP) and not any(word in text for word in _DOWN):
        reported = RelationDirection.UP
    elif any(word in text for word in _DOWN) and not any(word in text for word in _UP):
        reported = RelationDirection.DOWN

    stance = LiteratureStance.UNRELATED
    if predicted is not RelationDirection.UNKNOWN and reported is not RelationDirection.UNKNOWN:
        stance = (
            LiteratureStance.SUPPORTS if reported is predicted else LiteratureStance.CONTRADICTS
        )
    elif any(word in text for word in (*_PERTURB, *_BINDING, *_CORREL, *_PATHWAY)):
        stance = LiteratureStance.SUPPORTS

    if stance is LiteratureStance.CONTRADICTS:
        level = EvidenceLevel.X
    elif any(word in text for word in _PERTURB):
        level = EvidenceLevel.A
    elif any(word in text for word in _BINDING):
        level = EvidenceLevel.B
    elif any(word in text for word in _CORREL):
        level = EvidenceLevel.C
    elif any(word in text for word in _PATHWAY):
        level = EvidenceLevel.D
    else:
        level = EvidenceLevel.D
        if stance is LiteratureStance.UNRELATED:
            level = EvidenceLevel.N

    sentence = (hit.title or ABSENCE_OF_EVIDENCE).strip()
    same_species = _mentions(text, organism)
    same_tissue = _mentions(text, tissue)
    same_condition = _mentions(text, condition)
    return stance, level, reported, sentence, same_species, same_tissue, same_condition


def run_literature_check(
    candidates: list[CandidateRow],
    *,
    experiment_id: str,
    config: LiteratureSearchConfig,
    transport: HttpTransport | None = None,
    pubmed: PubMedAdapter | None = None,
    epmc: EuropePmcAdapter | None = None,
    searched_at: str | None = None,
) -> LiteratureTable:
    """Search only the supplied candidates. Unstable rows must not be passed in."""

    pubmed = pubmed or PubMedAdapter(transport=transport)
    epmc = epmc or EuropePmcAdapter(transport=transport)
    when = searched_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    records: list[LiteratureRecord] = []
    for row in candidates[: config.top_n]:
        predicted = RelationDirection(row.predicted_direction)
        query = build_query(
            row.source_id,
            row.target_id,
            organism=config.organism,
            tissue=config.tissue,
            condition=config.condition,
        )
        pubmed_hits = pubmed.search(query, retmax=config.max_hits_per_source)
        epmc_hits = epmc.search(query, page_size=config.max_hits_per_source)
        hits = _dedupe(pubmed_hits + epmc_hits)
        if not hits:
            records.append(_empty_record(row, query, when, predicted, source_name="pubmed_eutils"))
            continue
        for hit in hits:
            pmid_ok, doi_ok = verify_hit(hit, pubmed)
            stance, level, reported, sentence, same_sp, same_ti, same_co = classify_hit(
                hit,
                predicted=predicted,
                organism=config.organism,
                tissue=config.tissue,
                condition=config.condition,
            )
            if not pmid_ok and not doi_ok:
                # Fail closed: keep the row but force human review and do not
                # let an unverified identifier look like evidence.
                stance = LiteratureStance.UNRELATED
                level = EvidenceLevel.N
                sentence = ABSENCE_OF_EVIDENCE
            records.append(
                LiteratureRecord(
                    candidate_id=row.candidate_id,
                    source_id=row.source_id,
                    target_id=row.target_id,
                    predicted_direction=predicted,
                    query_string=query,
                    searched_at=when,
                    source_name=hit.source_name,
                    pmid=hit.pmid,
                    doi=hit.doi,
                    title=hit.title,
                    year=hit.year,
                    evidence_sentence=sentence,
                    relation_direction=reported,
                    same_species=same_sp,
                    same_tissue=same_ti,
                    same_condition=same_co,
                    stance=stance,
                    evidence_level=level,
                    pmid_authentic=pmid_ok,
                    doi_authentic=doi_ok,
                    reviewer_status=LiteratureReviewerStatus.NEEDS_REVIEW,
                    claim_kind="hypothesis",
                )
            )
    notes = [
        "Every row is a hypothesis. Attribution plus a paper is not causation.",
        f"Level N means: {ABSENCE_OF_EVIDENCE}.",
        "Do not write 首次发现 from an empty search.",
        "reviewer_status is needs_review until a human accepts or rejects the row.",
        f"Europe PMC search endpoint: {EUROPEPMC_SEARCH}.",
    ]
    return LiteratureTable(
        experiment_id=experiment_id,
        claim_kind="hypothesis",
        records=records,
        notes=notes,
    )


def _empty_record(
    row: CandidateRow,
    query: str,
    when: str,
    predicted: RelationDirection,
    *,
    source_name: str,
) -> LiteratureRecord:
    return LiteratureRecord(
        candidate_id=row.candidate_id,
        source_id=row.source_id,
        target_id=row.target_id,
        predicted_direction=predicted,
        query_string=query,
        searched_at=when,
        source_name=source_name,
        evidence_sentence=ABSENCE_OF_EVIDENCE,
        stance=LiteratureStance.UNRELATED,
        evidence_level=EvidenceLevel.N,
        pmid_authentic=False,
        doi_authentic=False,
        reviewer_status=LiteratureReviewerStatus.NEEDS_REVIEW,
        claim_kind="hypothesis",
    )


def _dedupe(hits: list[PaperHit]) -> list[PaperHit]:
    seen: set[str] = set()
    out: list[PaperHit] = []
    for hit in hits:
        key = hit.pmid or hit.doi or f"{hit.source_name}:{hit.title}"
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _mentions(text: str, needle: str | None) -> bool | None:
    if not needle:
        return None
    return needle.lower() in text


def run_literature_from_table(
    candidates_path: Any,
    *,
    experiment_id: str,
    config: LiteratureSearchConfig,
    dest: Any,
    transport: HttpTransport | None = None,
) -> dict[str, Any]:
    """CLI helper: load candidates.json, keep stable top-N, search, write."""

    from pathlib import Path

    from omics_agent.interpretation.stability import select_stable
    from omics_agent.schemas.interpretation import StabilityTable

    path = Path(candidates_path)
    table = StabilityTable.model_validate_json(path.read_text(encoding="utf-8"))
    stable = select_stable(table, config.top_n)
    lit = run_literature_check(
        stable, experiment_id=experiment_id, config=config, transport=transport
    )
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "literature.json"
    json_path.write_text(lit.model_dump_json(indent=2), encoding="utf-8")
    return {
        "literature_json": str(json_path),
        "n_records": len(lit.records),
        "n_queried": len(stable),
        "claim_kind": "hypothesis",
    }
