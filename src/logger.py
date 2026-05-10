"""
Observability — DQ report writer.

Every pipeline run writes a JSON + Markdown report to `reports/`. The JSON is
for machines (alerting, dashboards). The Markdown is for humans (the team
member who just got paged).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .quality import QualityResult


def render_markdown(report: dict) -> str:
    """Render a DQ report as a human-readable Markdown document."""
    lines = []
    lines.append(f"# Data Quality Report — {report['run_at']}")
    lines.append("")
    lines.append("## Bronze")
    for tbl, n in report["bronze_counts"].items():
        lines.append(f"- {tbl}: {n:,} rows ingested")
    lines.append("")

    lines.append("## Silver — validation results")
    lines.append("")
    lines.append("| Table | Rows In | Rows Out | Dropped | Status |")
    lines.append("|---|---:|---:|---:|---|")
    for tbl, info in report["silver"].items():
        status = "✅ PASS" if info["all_passed"] else "⚠️ WARN"
        lines.append(
            f"| {tbl} | {info['rows_in']:,} | {info['rows_out']:,} | "
            f"{info['rows_in'] - info['rows_out']:,} | {status} |"
        )
    lines.append("")

    # Threshold violations
    violations = []
    for tbl, info in report["silver"].items():
        for r in info["results"]:
            if not r["passed"]:
                violations.append(f"- **{tbl}** · `{r['rule']}` — {r['detail']}")
    lines.append("## Threshold violations")
    lines.append("")
    if violations:
        lines.extend(violations)
    else:
        lines.append("None — all DQ rules passed.")
    lines.append("")

    lines.append("## Gold")
    for tbl, n in report["gold_counts"].items():
        lines.append(f"- {tbl}: {n:,} rows")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Report generated automatically by the pipeline._")
    return "\n".join(lines)


def write_report(
    *,
    bronze_counts: dict[str, int],
    silver_results: dict[str, dict],
    gold_counts: dict[str, int],
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write the per-run DQ report (JSON + Markdown). Returns the two paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    run_at = ts.strftime("%Y-%m-%d %H:%M UTC")
    slug = ts.strftime("%Y%m%dT%H%M%S")

    report = {
        "run_at": run_at,
        "bronze_counts": bronze_counts,
        "silver": silver_results,
        "gold_counts": gold_counts,
    }

    json_path = reports_dir / f"dq_report_{slug}.json"
    md_path = reports_dir / f"dq_report_{slug}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def serialize_results(results: list[QualityResult]) -> list[dict]:
    """Turn QualityResult dataclasses into JSON-friendly dicts."""
    return [asdict(r) for r in results]
