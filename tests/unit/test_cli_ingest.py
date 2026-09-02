from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from omics_agent.cli import app

runner = CliRunner()


def test_cli_ingest_doi_dry_run() -> None:
    result = runner.invoke(
        app,
        ["ingest", "--paper-doi", "10.1234/none", "--dest", "outputs/unused", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "not a file locator" in result.output


def test_cli_ingest_local(tmp_path: Path) -> None:
    matrix = tmp_path / "rna.tsv"
    matrix.write_text("sample_id\tG1\nA\t0\n", encoding="utf-8")
    dest = tmp_path / "ingest"
    result = runner.invoke(
        app,
        [
            "ingest",
            "--source",
            "local",
            "--local-path",
            str(matrix),
            "--dest",
            str(dest),
            "--modality",
            "rna",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (dest / "ingest_manifest.yaml").is_file()
    ready = runner.invoke(app, ["data-readiness", str(dest / "ingest_manifest.yaml")])
    assert ready.exit_code == 0, ready.output
    assert "blocking" in ready.output
