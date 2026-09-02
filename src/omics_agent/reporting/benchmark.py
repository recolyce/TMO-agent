"""Markdown + JSON benchmark report for a toy or baseline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omics_agent.schemas.evaluation import EvaluationReport


def write_benchmark_report(
    path: Path,
    *,
    experiment_id: str,
    hashes: dict[str, str],
    reports: list[EvaluationReport],
    notes: list[str],
) -> None:
    """Write ``benchmark.json`` and ``benchmark.md`` next to ``path`` stem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "hashes": hashes,
        "notes": notes,
        "reports": [item.model_dump() for item in reports],
    }
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(experiment_id, hashes, reports, notes), encoding="utf-8")


def _markdown(
    experiment_id: str,
    hashes: dict[str, str],
    reports: list[EvaluationReport],
    notes: list[str],
) -> str:
    lines = [
        f"# Benchmark report: {experiment_id}",
        "",
        "Attribution / coefficients are prediction contributions, not causal effects.",
        "",
        "## Hashes and seed",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for key, value in hashes.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Metrics", ""])
    for report in reports:
        lines.append(f"### {report.model_name} / {report.split}")
        lines.append("")
        lines.append(
            f"instances={report.n_instances}, features={report.n_features}, "
            f"coverage={report.coverage:.3f} "
            f"({report.n_observed_targets}/{report.n_possible_targets} observed targets)"
        )
        lines.append("")
        lines.append("| metric | value | n_valid | n_total |")
        lines.append("|---|---|---|---|")
        for scalar in report.scalars:
            value = "NA" if scalar.value is None else f"{scalar.value:.4f}"
            lines.append(f"| {scalar.name} | {value} | {scalar.n_valid} | {scalar.n_total} |")
        if report.bootstrap:
            lines.extend(["", "Bootstrap 95% CI (resample experimental units):", ""])
            for ci in report.bootstrap:
                low = "NA" if ci.low is None else f"{ci.low:.4f}"
                high = "NA" if ci.high is None else f"{ci.high:.4f}"
                lines.append(f"- {ci.metric}: [{low}, {high}] (n_units={ci.n_units})")
        if report.warnings:
            lines.extend(["", "Warnings:", ""])
            for warning in report.warnings:
                lines.append(f"- {warning}")
        lines.append("")
    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)
