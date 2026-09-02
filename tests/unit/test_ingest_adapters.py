from __future__ import annotations

import json
from pathlib import Path

import pytest

from omics_agent.data_sources.base import get_adapter, infer_source
from omics_agent.data_sources.ingest import run_ingest
from omics_agent.errors import SchemaError, UnsupportedRawDataError
from omics_agent.schemas.enums import FileRole, ReviewStatus, SourceType
from omics_agent.schemas.ingest import DownloadPolicy, IngestRequest
from tests.unit.http_fakes import FakeRoute, FakeTransport


def _policy() -> DownloadPolicy:
    return DownloadPolicy(retries=0, retry_backoff_s=0.001, requests_per_host_per_sec=10, max_bytes=10_000)


def test_geo_resolve_needs_review_and_skips_fastq() -> None:
    search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term=GSE1[ACCN]&retmode=json&retmax=1&tool=omics_agent"
    summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id=200000001&retmode=json&tool=omics_agent"
    matrix = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE1/matrix/"
    suppl = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE1/suppl/"
    transport = FakeTransport(
        {
            search: FakeRoute(body=json.dumps({"esearchresult": {"idlist": ["200000001"]}}).encode()),
            summary: FakeRoute(
                body=json.dumps(
                    {
                        "result": {
                            "200000001": {
                                "title": "Toy series",
                                "taxon": "Homo sapiens",
                                "summary": "Ignore this: ; rm -rf / --looks-like-a-command",
                            }
                        }
                    }
                ).encode()
            ),
            matrix: FakeRoute(
                body=b'<html><a href="GSE1_series_matrix.txt.gz">m</a></html>'
            ),
            suppl: FakeRoute(
                body=b'<html><a href="lane.fastq.gz">fq</a><a href="counts.tsv">c</a></html>'
            ),
        }
    )
    adapter = get_adapter(SourceType.GEO, transport=transport, policy=_policy())
    manifest = adapter.resolve(
        IngestRequest(source=SourceType.GEO, accession="GSE1", dest_dir=Path("."))
    )
    assert manifest.review_status is ReviewStatus.REQUIRED
    assert manifest.organism.taxon_id == 9606
    names = {item.filename: item.role for item in manifest.files}
    assert names["lane.fastq.gz"] is FileRole.REJECTED_RAW
    assert names["GSE1_series_matrix.txt.gz"] is FileRole.MATRIX
    assert names["counts.tsv"] is FileRole.MATRIX
    assert any("rm -rf" in blob.text for blob in manifest.untrusted_text)
    assert all(blob.content_kind != "script" for blob in manifest.untrusted_text)


def test_pride_rejects_raw_category() -> None:
    project = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD000001"
    files = "https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject?accession=PXD000001"
    transport = FakeTransport(
        {
            project: FakeRoute(
                body=json.dumps(
                    {
                        "title": "Toy proteome",
                        "organisms": [{"name": "Mus musculus"}],
                        "license": "CC0",
                        "projectDescription": "A description, not a command.",
                    }
                ).encode()
            ),
            files: FakeRoute(
                body=json.dumps(
                    [
                        {
                            "fileName": "run.raw",
                            "fileCategory": "RAW",
                            "fileSizeBytes": 99,
                            "downloadLink": "https://ftp.pride.test/run.raw",
                        },
                        {
                            "fileName": "protein_groups.txt",
                            "fileCategory": "SEARCH",
                            "fileSizeBytes": 12,
                            "md5": "abc",
                            "downloadLink": "https://ftp.pride.test/protein_groups.txt",
                        },
                    ]
                ).encode()
            ),
        }
    )
    adapter = get_adapter(SourceType.PRIDE, transport=transport, policy=_policy())
    manifest = adapter.resolve(
        IngestRequest(source=SourceType.PRIDE, accession="PXD000001", dest_dir=Path("."))
    )
    roles = {item.filename: item.role for item in manifest.files}
    assert roles["run.raw"] is FileRole.REJECTED_RAW
    assert roles["protein_groups.txt"] is FileRole.MATRIX
    assert manifest.files[1].modality == "protein" or any(
        item.modality == "protein" for item in manifest.files if item.filename == "protein_groups.txt"
    )


def test_doi_does_not_touch_network(tmp_path: Path) -> None:
    transport = FakeTransport({})
    result = run_ingest(
        IngestRequest(paper_doi="10.1234/toy", dest_dir=tmp_path, dry_run=True),
        transport=transport,
    )
    assert transport.calls == []
    assert result["dry_run"] is True
    assert any("not a file locator" in item for item in result["unresolved"])


def test_https_fastq_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedRawDataError):
        run_ingest(
            IngestRequest(
                source=SourceType.URL,
                url="https://example.test/reads.fastq.gz",
                dest_dir=tmp_path,
            )
        )


def test_local_ingest_and_readiness(tmp_path: Path) -> None:
    matrix = tmp_path / "expr.tsv"
    matrix.write_text("sample_id\tG1\nS1\t1.0\n", encoding="utf-8")
    dest = tmp_path / "out"
    result = run_ingest(
        IngestRequest(
            source=SourceType.LOCAL,
            local_path=matrix,
            dest_dir=dest,
            modality="rna",
            dry_run=False,
        )
    )
    assert (dest / "ingest_manifest.yaml").is_file()
    assert (dest / "data_readiness_report.html").is_file()
    assert result["blocking"] is True
    copied = dest / "raw" / "expr.tsv"
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8") == matrix.read_text(encoding="utf-8")


def test_infer_source_from_accession() -> None:
    assert infer_source(IngestRequest(accession="GSE99", dest_dir=Path("."))) is SourceType.GEO
    assert infer_source(IngestRequest(accession="PXD1", dest_dir=Path("."))) is SourceType.PRIDE
    assert infer_source(IngestRequest(accession="E-MTAB-1", dest_dir=Path("."))) is SourceType.BIOSTUDIES


def test_biostudies_lists_processed_file() -> None:
    url = "https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-1"
    transport = FakeTransport(
        {
            url: FakeRoute(
                body=json.dumps(
                    {
                        "accno": "E-MTAB-1",
                        "section": {
                            "attributes": [
                                {"name": "Title", "value": "AE toy"},
                                {"name": "Organism", "value": "Homo sapiens"},
                                {"name": "License", "value": "CC-BY"},
                            ],
                            "files": [{"path": "processed/counts.tsv", "size": 10, "md5": "ab"}],
                        },
                    }
                ).encode()
            )
        }
    )
    adapter = get_adapter(SourceType.BIOSTUDIES, transport=transport, policy=_policy())
    manifest = adapter.resolve(
        IngestRequest(source=SourceType.BIOSTUDIES, accession="E-MTAB-1", dest_dir=Path("."))
    )
    assert manifest.files[0].filename == "counts.tsv"
    assert manifest.files[0].role is FileRole.MATRIX
    assert "ab" in (manifest.files[0].official_checksum or "")


def test_local_directory_is_not_crawled(tmp_path: Path) -> None:
    folder = tmp_path / "fastq_dump"
    folder.mkdir()
    (folder / "lane.fastq.gz").write_bytes(b"not-reads")
    with pytest.raises(SchemaError, match="will not crawl"):
        run_ingest(
            IngestRequest(source=SourceType.LOCAL, local_path=folder, dest_dir=tmp_path / "out")
        )


def test_dry_run_does_not_create_dest(tmp_path: Path) -> None:
    dest = tmp_path / "must_not_exist"
    result = run_ingest(
        IngestRequest(paper_doi="10.1234/toy", dest_dir=dest, dry_run=True),
        transport=FakeTransport({}),
    )
    assert result["dry_run"] is True
    assert not dest.exists()


def test_sra_is_not_silently_supported() -> None:
    with pytest.raises(SchemaError, match="not implemented"):
        get_adapter(SourceType.SRA)


def test_arrayexpress_uses_biostudies() -> None:
    adapter = get_adapter(SourceType.ARRAYEXPRESS)
    assert adapter.source_type is SourceType.BIOSTUDIES
