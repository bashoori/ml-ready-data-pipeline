"""
Silver layer — schema enforcement, type coercion, deduplication, validation.

Anything that escapes Silver is safe to use downstream. Each table has its
own cleaning function so the rules stay readable and testable. The shared
pattern: enforce schema → coerce types → drop bad rows → dedup → validate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import quality as q


# --- Per-table cleaners -----------------------------------------------------


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the users table.

    Rules applied:
    - Strip whitespace and lowercase emails (so `Alice@x` and `alice@x` collapse).
    - Coerce `signup_date` to date; drop rows with no signup date.
    - Coerce `age` to int; drop rows with age outside [13, 120].
    - Deduplicate on `user_id`, keeping the earliest signup_date.
    """
    df = df.copy()

    # Email normalization
    df["email"] = df["email"].str.strip().str.lower()

    # Signup date — required. Coerce, then drop nulls.
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce").dt.date
    df = df.dropna(subset=["signup_date"])

    # Age — coerce to nullable int, then constrain
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 13) & (df["age"] <= 120)]
    df["age"] = df["age"].astype("int16")

    # Country — strip
    df["country"] = df["country"].str.strip()

    # Dedup — keep earliest signup
    df = df.sort_values("signup_date").drop_duplicates(subset=["user_id"], keep="first")

    return df.reset_index(drop=True)


def clean_events(df: pd.DataFrame, valid_user_ids: set[str]) -> pd.DataFrame:
    """Clean the events table.

    Rules applied:
    - Normalize `event_type` to lowercase.
    - Coerce `timestamp` to datetime; drop nulls and future-dated events.
    - Drop events whose `user_id` doesn't exist in users (referential integrity).
    - Deduplicate on `event_id`.
    """
    df = df.copy()

    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # Future events are almost always source-error; drop them
    df = df[df["timestamp"] <= pd.Timestamp.utcnow().tz_localize(None)]

    # Drop blank user_ids and any user_id not in the cleaned users table
    df = df[df["user_id"].notna() & (df["user_id"] != "")]
    df = df[df["user_id"].isin(valid_user_ids)]

    # Dedup
    df = df.drop_duplicates(subset=["event_id"], keep="first")

    # Stable column ordering for downstream
    return df[["event_id", "user_id", "event_type", "timestamp"]].reset_index(drop=True)


def clean_feedback(df: pd.DataFrame, valid_user_ids: set[str]) -> pd.DataFrame:
    """Clean the feedback table.

    Rules applied:
    - Coerce rating to int; drop ratings outside [1, 5].
    - Drop empty / whitespace-only comments.
    - Drop rows whose user_id doesn't exist.
    - Deduplicate on feedback_id.
    """
    df = df.copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]
    df["rating"] = df["rating"].astype("int8")

    df["comment"] = df["comment"].fillna("").str.strip()
    df = df[df["comment"] != ""]

    df = df[df["user_id"].notna() & (df["user_id"] != "")]
    df = df[df["user_id"].isin(valid_user_ids)]

    df = df.drop_duplicates(subset=["feedback_id"], keep="first")

    return df.reset_index(drop=True)


# --- Validation orchestration ----------------------------------------------


def validate(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    *,
    table: str,
    primary_key: str,
    required_columns: list[str],
    thresholds: dict,
) -> list[q.QualityResult]:
    """Run hard + soft DQ checks on a Silver table. Returns soft-rule results."""
    # Hard rules — raise on failure
    q.require_columns(df_in, required_columns, table)
    q.require_primary_key(df_in, primary_key, table)

    # Soft rules — collect for the DQ report
    results: list[q.QualityResult] = []
    results.append(q.check_drop_rate(len(df_in), len(df_out), thresholds["max_drop_rate"]))
    results.append(
        q.check_duplicate_rate(df_out, primary_key, thresholds["max_duplicate_rate"])
    )
    for col in df_out.columns:
        results.append(q.check_null_rate(df_out, col, thresholds["max_null_rate"]))

    return results


# --- Persistence ------------------------------------------------------------


def write(df: pd.DataFrame, name: str, silver_dir: Path) -> Path:
    """Write a Silver table to Parquet. Parquet is the right choice here —
    columnar, typed, and trivially loaded by every downstream tool.

    Falls back to CSV if no Parquet engine (pyarrow / fastparquet) is
    installed, so the pipeline still runs in minimal environments.
    """
    silver_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = silver_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
    except (ImportError, ValueError):
        out = silver_dir / f"{name}.csv"
        df.to_csv(out, index=False)
    return out
