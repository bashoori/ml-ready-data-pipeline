"""
Gold layer — ML-ready feature tables.

Different consumers (BI, ML training, ad-hoc analysis) get tables shaped for
their use case, all derived from the same Silver source of truth.

Two tables produced here:
- `user_features`: one row per user, denormalized — the kind of wide,
  pre-aggregated table an ML pipeline can load directly.
- `event_facts`: long-form fact table for behavioral analysis.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_user_features(
    users: pd.DataFrame, events: pd.DataFrame, feedback: pd.DataFrame
) -> pd.DataFrame:
    """One row per user, with engineered behavioral features.

    Columns:
        user_id, email, signup_date, age, country,
        n_events, n_logins, n_purchases,
        first_event_at, last_event_at, days_active,
        n_feedback, avg_rating, has_negative_feedback
    """
    # Event-derived features
    ev_agg = (
        events.groupby("user_id")
        .agg(
            n_events=("event_id", "count"),
            first_event_at=("timestamp", "min"),
            last_event_at=("timestamp", "max"),
        )
        .reset_index()
    )
    # Per-event-type counts (pivot)
    ev_type_counts = (
        events.assign(_one=1)
        .pivot_table(
            index="user_id",
            columns="event_type",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    # Standardize the columns we care about (defaults to 0 if the user has none)
    for col in ("login", "click", "view", "purchase", "logout"):
        if col not in ev_type_counts.columns:
            ev_type_counts[col] = 0
    ev_type_counts = ev_type_counts.rename(
        columns={"login": "n_logins", "purchase": "n_purchases"}
    )[["user_id", "n_logins", "n_purchases"]]

    # Feedback-derived features
    fb_agg = (
        feedback.groupby("user_id")
        .agg(
            n_feedback=("feedback_id", "count"),
            avg_rating=("rating", "mean"),
            has_negative_feedback=("rating", lambda r: bool((r <= 2).any())),
        )
        .reset_index()
    )

    # Compose
    out = (
        users.merge(ev_agg, on="user_id", how="left")
        .merge(ev_type_counts, on="user_id", how="left")
        .merge(fb_agg, on="user_id", how="left")
    )

    # Fill NaNs that come from left joins with sensible defaults
    out["n_events"] = out["n_events"].fillna(0).astype("int32")
    out["n_logins"] = out["n_logins"].fillna(0).astype("int32")
    out["n_purchases"] = out["n_purchases"].fillna(0).astype("int32")
    out["n_feedback"] = out["n_feedback"].fillna(0).astype("int32")
    # Coerce to a clean boolean column (NaN → False) without triggering
    # the pandas object-dtype downcast warning.
    out["has_negative_feedback"] = (
        out["has_negative_feedback"].notna() & out["has_negative_feedback"].astype("boolean").fillna(False)
    ).astype(bool)

    # days_active = (last - first) in days, 0 for users with <2 events
    out["days_active"] = (
        (out["last_event_at"] - out["first_event_at"]).dt.total_seconds() / 86400
    ).fillna(0).astype("int32")

    return out


def build_event_facts(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Long-form event fact table joined to user attributes for slicing."""
    cols = ["event_id", "user_id", "event_type", "timestamp", "country", "age"]
    out = events.merge(users[["user_id", "country", "age"]], on="user_id", how="left")
    return out[cols]


def write(df: pd.DataFrame, name: str, gold_dir: Path) -> Path:
    """Write a Gold table to Parquet, falling back to CSV if no
    Parquet engine is installed."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = gold_dir / f"{name}.parquet"
        df.to_parquet(out, index=False)
    except (ImportError, ValueError):
        out = gold_dir / f"{name}.csv"
        df.to_csv(out, index=False)
    return out
