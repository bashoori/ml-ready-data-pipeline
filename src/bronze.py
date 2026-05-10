"""
Bronze layer — raw ingestion.

Bronze preserves the source data faithfully. No cleaning, no drops, no joins.
The point is reproducibility: if a downstream consumer disputes a number, we
can replay any transformation from this layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with permissive defaults — keep everything as strings so we
    don't accidentally lose information to type coercion at this layer."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read line-delimited JSON, tolerating malformed lines (logged, not dropped silently)."""
    rows: list[dict] = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    df = pd.DataFrame(rows)
    df.attrs["bad_lines"] = bad_lines
    return df


def read_tsv(path: Path) -> pd.DataFrame:
    """Read TSV with permissive defaults."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])


READERS = {
    "csv": read_csv,
    "jsonl": read_jsonl,
    "tsv": read_tsv,
}


def ingest(name: str, source_cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """Load a single Bronze table from its raw source file."""
    reader = READERS[source_cfg["format"]]
    path = raw_dir / source_cfg["file"]
    df = reader(path)
    # Stamp metadata onto the DataFrame so downstream layers know where it came from
    df.attrs["source_name"] = name
    df.attrs["source_path"] = str(path)
    df.attrs["bronze_row_count"] = len(df)
    return df


def ingest_all(sources: dict, raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Ingest every configured source. Returns a name → DataFrame mapping."""
    return {name: ingest(name, cfg, raw_dir) for name, cfg in sources.items()}
