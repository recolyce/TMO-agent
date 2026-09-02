"""Data-readiness gates. Red means stop; the pipeline does not guess a fix."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from omics_agent.schemas.enums import FileRole, ReviewStatus, SamplingDesign
from omics_agent.schemas.ingest import DataReadinessReport, IngestManifest, ReadinessLevel
from omics_agent.schemas.samples import REQUIRED_SAMPLE_COLUMNS, load_sample_sheet


def evaluate_readiness(manifest: IngestManifest) -> DataReadinessReport:
    """Score ingest output against the milestone-2 QC gates."""

    gates: list[ReadinessLevel] = []
    processed = [item for item in manifest.files if item.role is FileRole.MATRIX]
    raw = [item for item in manifest.files if item.role is FileRole.REJECTED_RAW]
    unknown = [item for item in manifest.files if item.role is FileRole.UNKNOWN]
    archives = [item for item in manifest.files if item.role is FileRole.ARCHIVE]

    gates.append(
        _gate(
            "processed_matrix",
            "green" if processed else "red",
            f"{len(processed)} processed-looking file(s) listed."
            if processed
            else "No processed matrix was identified.",
            "Provide a series matrix, quantified protein table, or other processed abundance file.",
        )
    )
    gates.append(
        _gate(
            "raw_excluded",
            "green" if not raw else "yellow",
            "No raw FASTQ / raw mass-spec listed."
            if not raw
            else f"{len(raw)} raw file(s) listed and will not be downloaded: "
            + ", ".join(item.filename for item in raw[:8]),
            "Leave raw files out of training. Use author-processed tables only.",
        )
    )
    gates.append(
        _gate(
            "unknown_files",
            "green" if not unknown else "yellow",
            "Every file has a role."
            if not unknown
            else "Unknown files need a person to assign role/modality: "
            + ", ".join(item.filename for item in unknown[:8]),
            "Edit ingest_manifest.yaml and set role/modality, or drop the files.",
        )
    )
    gates.append(
        _gate(
            "archives",
            "green" if not archives else "yellow",
            "No archives."
            if not archives
            else "Archives were not extracted: " + ", ".join(item.filename for item in archives[:6]),
            "Do not auto-extract zip/tar. A person should unpack processed tables only.",
        )
    )
    gates.append(
        _gate(
            "checksums",
            "green"
            if processed and all(item.sha256 for item in processed if item.path)
            else ("yellow" if processed else "red"),
            "Downloaded processed files have SHA-256."
            if processed and all(item.sha256 for item in processed if item.path)
            else "SHA-256 is missing until download, or download was skipped.",
            "Run ingest without --dry-run / --resolve-only so files are fetched and hashed.",
        )
    )
    official_bad = [
        item.filename
        for item in manifest.files
        if item.official_checksum and item.path and not item.sha256
    ]
    gates.append(
        _gate(
            "official_checksum",
            "yellow" if official_bad else "green",
            "Official publisher checksums will be checked on download."
            if not official_bad
            else "Official checksum present but file not verified yet: " + ", ".join(official_bad[:6]),
            "Download the file so the official digest can be compared.",
        )
    )
    license_unknown = manifest.license.name.strip().lower() in {"unknown", "undeclared", ""}
    gates.append(
        _gate(
            "license",
            "red" if license_unknown else "yellow",
            "License is unknown." if license_unknown else f"License recorded as '{manifest.license.name}' (still confirm redistribution).",
            "A person must confirm the license. The pipeline will not assume GEO/PRIDE files are redistributable.",
        )
    )
    gates.append(
        _gate(
            "sampling_design",
            "red",
            "sampling_design is not set on the ingest manifest (adapters must not guess).",
            "After inspecting the sample sheet, set longitudinal or repeated_cross_sectional on dataset.yaml.",
        )
    )
    gates.append(
        _gate(
            "experimental_units",
            "red",
            "No reviewed sample sheet with experimental_unit_id.",
            "Create samples.tsv with the required columns. Do not invent donor/time/biospecimen IDs.",
        )
    )
    gates.append(_sample_sheet_gate(manifest))
    pairing_level = "undeclared"
    gates.append(
        _gate(
            "pairing",
            "red",
            f"pairing_level is {pairing_level}. Biospecimen identity has not been proven.",
            "Set pairing_level only when biospecimen IDs prove it. Use group_level_only otherwise.",
        )
    )
    if manifest.organism.taxon_id is None:
        gates.append(
            _gate(
                "organism",
                "yellow" if manifest.organism.name != "undeclared" else "red",
                f"organism={manifest.organism.name!r} has no NCBI taxon_id.",
                "Set organism.taxon_id from NCBI Taxonomy. Do not guess an uncommon species.",
            )
        )
    else:
        gates.append(
            _gate(
                "organism",
                "green",
                f"{manifest.organism.name} (taxon {manifest.organism.taxon_id}).",
                "",
            )
        )
    review_red = manifest.review_status is not ReviewStatus.APPROVED or bool(manifest.unresolved)
    gates.append(
        _gate(
            "human_review",
            "red" if review_red else "green",
            "Review is still required." if review_red else "Review approved.",
            "Resolve every unresolved item. The pipeline will not infer mappings from the paper.",
        )
    )
    blocking = any(gate.level == "red" for gate in gates)
    return DataReadinessReport(
        dataset_id=manifest.dataset_id,
        blocking=blocking,
        gates=gates,
        unresolved=list(manifest.unresolved),
    )


def write_readiness_report(manifest: IngestManifest, html_path: Path) -> DataReadinessReport:
    """Write HTML + JSON next to ``html_path``."""

    report = evaluate_readiness(manifest)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_to_html(report), encoding="utf-8")
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return report


def _sample_sheet_gate(manifest: IngestManifest) -> ReadinessLevel:
    if manifest.sample_sheet is None:
        return _gate(
            "sample_sheet",
            "red",
            "No sample sheet is attached.",
            "Write samples.tsv with: " + " ".join(REQUIRED_SAMPLE_COLUMNS),
        )
    path = Path(manifest.sample_sheet)
    if not path.is_file():
        return _gate(
            "sample_sheet",
            "red",
            f"Sample sheet path does not exist: {path}",
            "Point sample_sheet at an existing TSV after a person curated it.",
        )
    try:
        import pandas as pd

        frame = pd.read_csv(path, sep="\t")
        load_sample_sheet(
            frame,
            sampling_design=SamplingDesign.UNDECLARED,
            declared_modalities=sorted(
                {item.modality for item in manifest.files if item.modality}
            )
            or ["undeclared"],
        )
    except Exception as exc:  # noqa: BLE001 — surface any sheet problem as a red gate
        return _gate(
            "sample_sheet",
            "red",
            f"Sample sheet failed validation: {exc}",
            "Fix the sheet. Do not drop unknown IDs; add them to human_review.unresolved.",
        )
    return _gate(
        "sample_sheet",
        "yellow",
        "A sample sheet parsed, but design/pairing still need a person to confirm.",
        "Approve the dataset.yaml only after units, time, and pairing are explicit.",
    )


def _gate(name: str, level: str, detail: str, how_to_fix: str) -> ReadinessLevel:
    return ReadinessLevel(name=name, level=level, detail=detail, how_to_fix=how_to_fix)


def _to_html(report: DataReadinessReport) -> str:
    rows = []
    colors = {"green": "#1b7f3a", "yellow": "#a67c00", "red": "#b42318"}
    for gate in report.gates:
        color = colors.get(gate.level, "#333")
        rows.append(
            "<tr>"
            f"<td><strong style='color:{color}'>{escape(gate.level.upper())}</strong></td>"
            f"<td>{escape(gate.name)}</td>"
            f"<td>{escape(gate.detail)}</td>"
            f"<td>{escape(gate.how_to_fix)}</td>"
            "</tr>"
        )
    unresolved = "".join(f"<li>{escape(item)}</li>" for item in report.unresolved) or "<li>None listed.</li>"
    status = "BLOCKED" if report.blocking else "READY FOR REVIEW"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Data readiness — {escape(report.dataset_id)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Data readiness: {escape(report.dataset_id)}</h1>
  <p>Status: <strong>{status}</strong>. Red gates stop training. The pipeline will not guess sample mappings.</p>
  <table>
    <thead><tr><th>level</th><th>gate</th><th>detail</th><th>how to fix</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <h2>Unresolved review items</h2>
  <ul>{unresolved}</ul>
</body>
</html>
"""
