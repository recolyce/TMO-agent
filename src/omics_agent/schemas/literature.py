"""Literature-evidence rows. Absence of a hit is not a discovery."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from omics_agent.schemas.dataset import StrictModel
from omics_agent.schemas.enums import (
    EvidenceLevel,
    LiteratureReviewerStatus,
    LiteratureStance,
    RelationDirection,
)

ABSENCE_OF_EVIDENCE = "在本次检索范围内未找到直接证据"


class LiteratureSearchConfig(StrictModel):
    """How many stable candidates may be sent to PubMed / Europe PMC."""

    top_n: int = Field(default=20, ge=1, le=50)
    max_hits_per_source: int = Field(default=5, ge=1, le=20)
    species_taxon_id: int | None = Field(default=9606, ge=1)
    organism: str | None = "human"
    tissue: str | None = None
    condition: str | None = None


class LiteratureRecord(StrictModel):
    """One paper (or an explicit no-hit) for one candidate.

    ``reviewer_status`` is always needs_review when written by the pipeline.
    """

    candidate_id: str
    source_id: str
    target_id: str
    predicted_direction: RelationDirection
    query_string: str
    searched_at: str
    source_name: str
    pmid: str | None = None
    doi: str | None = None
    title: str | None = None
    year: str | None = None
    evidence_sentence: str
    relation_direction: RelationDirection = RelationDirection.UNKNOWN
    same_species: bool | None = None
    same_tissue: bool | None = None
    same_condition: bool | None = None
    stance: LiteratureStance
    evidence_level: EvidenceLevel
    pmid_authentic: bool = False
    doi_authentic: bool = False
    reviewer_status: LiteratureReviewerStatus = LiteratureReviewerStatus.NEEDS_REVIEW
    claim_kind: Literal["hypothesis"] = "hypothesis"


class LiteratureTable(StrictModel):
    """Saved evidence table. claim_kind is literally hypothesis."""

    experiment_id: str
    claim_kind: Literal["hypothesis"] = "hypothesis"
    absence_phrase: Literal["在本次检索范围内未找到直接证据"] = ABSENCE_OF_EVIDENCE
    records: list[LiteratureRecord]
    notes: list[str] = Field(default_factory=list)
