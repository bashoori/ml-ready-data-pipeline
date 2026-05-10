"""
Data-quality rules.

Each rule is a function that takes a DataFrame and returns a QualityResult.
Rules are composable and run as part of Silver-layer validation. Results
land in the per-run DQ report so consumers can see exactly what happened.

Hard-fail rules (schema, required columns) raise. Soft rules (null rates,
duplicate rates, drop rates) return a result with `passed=False` but do not
raise — they're signals for the team to investigate, not reasons to break
the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class QualityResult:
    rule: str
    passed: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# --- Hard rules (raise) -----------------------------------------------------


def require_columns(df: pd.DataFrame, cols: list[str], table: str) -> None:
    """Hard fail if any required column is missing. Schema is a contract."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema violation in `{table}`: missing required columns {missing}. "
            f"Got: {list(df.columns)}"
        )


def require_primary_key(df: pd.DataFrame, key: str, table: str) -> None:
    """Hard fail if the primary key column doesn't exist."""
    if key not in df.columns:
        raise ValueError(f"Primary key `{key}` not found in `{table}`.")


# --- Soft rules (return QualityResult) --------------------------------------


def check_null_rate(
    df: pd.DataFrame, column: str, threshold: float
) -> QualityResult:
    """Warn if a column's null rate exceeds the threshold."""
    if column not in df.columns:
        return QualityResult(
            rule=f"null_rate({column})",
            passed=False,
            detail=f"column `{column}` missing",
        )
    null_rate = df[column].isna().mean() if len(df) else 0.0
    return QualityResult(
        rule=f"null_rate({column})",
        passed=null_rate <= threshold,
        detail=f"{null_rate:.2%} null (threshold {threshold:.0%})",
        metrics={"null_rate": float(null_rate), "threshold": threshold},
    )


def check_duplicate_rate(
    df: pd.DataFrame, key: str, threshold: float
) -> QualityResult:
    """Warn if duplicates on the primary key exceed the threshold (pre-dedup)."""
    if key not in df.columns or len(df) == 0:
        return QualityResult(
            rule=f"duplicate_rate({key})",
            passed=True,
            detail="not applicable",
        )
    dup_rate = df[key].duplicated().mean()
    return QualityResult(
        rule=f"duplicate_rate({key})",
        passed=dup_rate <= threshold,
        detail=f"{dup_rate:.2%} duplicates (threshold {threshold:.0%})",
        metrics={"duplicate_rate": float(dup_rate), "threshold": threshold},
    )


def check_drop_rate(
    rows_in: int, rows_out: int, threshold: float
) -> QualityResult:
    """Warn if more than `threshold` of input rows were dropped during cleaning."""
    if rows_in == 0:
        return QualityResult(rule="drop_rate", passed=True, detail="no input rows")
    drop_rate = (rows_in - rows_out) / rows_in
    return QualityResult(
        rule="drop_rate",
        passed=drop_rate <= threshold,
        detail=f"{drop_rate:.2%} dropped ({rows_in - rows_out}/{rows_in}, threshold {threshold:.0%})",
        metrics={
            "drop_rate": float(drop_rate),
            "rows_in": int(rows_in),
            "rows_out": int(rows_out),
            "threshold": threshold,
        },
    )


def check_value_range(
    df: pd.DataFrame, column: str, lo: float, hi: float
) -> QualityResult:
    """Warn if a numeric column has values outside [lo, hi]."""
    if column not in df.columns or len(df) == 0:
        return QualityResult(rule=f"range({column})", passed=True, detail="not applicable")
    series = pd.to_numeric(df[column], errors="coerce")
    out_of_range = ((series < lo) | (series > hi)).sum()
    return QualityResult(
        rule=f"range({column})",
        passed=out_of_range == 0,
        detail=f"{out_of_range} rows outside [{lo}, {hi}]",
        metrics={"out_of_range": int(out_of_range), "lo": lo, "hi": hi},
    )
