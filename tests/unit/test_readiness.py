from __future__ import annotations

from pathlib import Path

from omics_agent.reporting.readiness import evaluate_readiness, write_readiness_report
from omics_agent.schemas.dataset import LicenseSpec, OrganismSpec
from omics_agent.schemas.enums import FileRole, ReviewStatus, SourceType
from omics_agent.schemas.ingest import IngestFile, IngestManifest, ProvenanceRecord


def _manifest(**kwargs: object) -> IngestManifest:
    base = dict(
        dataset_id="toy",
        title="toy",
        source_type=SourceType.GEO,
        license=LicenseSpec(name="unknown", redistributable=False),
        organism=OrganismSpec(name="undeclared"),
        files=[],
        review_status=ReviewStatus.REQUIRED,
        unresolved=["Confirm time."],
        provenance=ProvenanceRecord(
            source=SourceType.GEO,
            retrieved_at="2026-01-01T00:00:00+00:00",
            adapter="test",
        ),
    )
    base.update(kwargs)
    return IngestManifest.model_validate(base)


def test_readiness_is_blocking_without_sample_sheet() -> None:
    report = evaluate_readiness(
        _manifest(
            files=[
                IngestFile(filename="counts.tsv", role=FileRole.MATRIX, url="https://x/counts.tsv")
            ]
        )
    )
    assert report.blocking is True
    names = {gate.name: gate.level for gate in report.gates}
    assert names["sample_sheet"] == "red"
    assert names["sampling_design"] == "red"
    assert names["human_review"] == "red"


def test_readiness_html_written(tmp_path: Path) -> None:
    html = tmp_path / "data_readiness_report.html"
    report = write_readiness_report(_manifest(), html)
    assert report.blocking is True
    text = html.read_text(encoding="utf-8")
    assert "BLOCKED" in text
    assert (tmp_path / "data_readiness_report.json").is_file()
