"""Unit tests for Silver-layer cleaning logic."""
from __future__ import annotations

import pandas as pd
import pytest

from src import silver
from src import quality as q


# --- users ------------------------------------------------------------------


def test_clean_users_drops_invalid_age():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "email": ["a@x.com", "b@x.com", "c@x.com"],
        "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "age": ["29", "231", "-5"],
        "country": ["CA", "US", "UK"],
    })
    out = silver.clean_users(df)
    assert list(out["user_id"]) == ["u1"]


def test_clean_users_normalizes_email_and_strips():
    df = pd.DataFrame({
        "user_id": ["u1"],
        "email": ["  Alice@Example.COM  "],
        "signup_date": ["2024-01-01"],
        "age": ["30"],
        "country": ["CA"],
    })
    out = silver.clean_users(df)
    assert out.loc[0, "email"] == "alice@example.com"


def test_clean_users_dedups_keeping_earliest_signup():
    df = pd.DataFrame({
        "user_id": ["u1", "u1"],
        "email": ["a@x.com", "a@x.com"],
        "signup_date": ["2024-03-01", "2024-01-01"],
        "age": ["30", "30"],
        "country": ["CA", "CA"],
    })
    out = silver.clean_users(df)
    assert len(out) == 1
    assert str(out.loc[0, "signup_date"]) == "2024-01-01"


def test_clean_users_drops_missing_signup_date():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "email": ["a@x.com", "b@x.com"],
        "signup_date": ["2024-01-01", None],
        "age": ["30", "30"],
        "country": ["CA", "US"],
    })
    out = silver.clean_users(df)
    assert list(out["user_id"]) == ["u1"]


# --- events -----------------------------------------------------------------


def test_clean_events_lowercases_event_type():
    df = pd.DataFrame({
        "event_id": ["e1"],
        "user_id": ["u1"],
        "event_type": ["CLICK"],
        "timestamp": ["2024-06-01T10:00:00"],
    })
    out = silver.clean_events(df, valid_user_ids={"u1"})
    assert out.loc[0, "event_type"] == "click"


def test_clean_events_drops_unknown_user():
    df = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "user_id": ["u1", "u_unknown"],
        "event_type": ["click", "click"],
        "timestamp": ["2024-06-01T10:00:00", "2024-06-01T10:00:00"],
    })
    out = silver.clean_events(df, valid_user_ids={"u1"})
    assert list(out["event_id"]) == ["e1"]


def test_clean_events_drops_future_dated():
    df = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "user_id": ["u1", "u1"],
        "event_type": ["click", "click"],
        "timestamp": ["2024-06-01T10:00:00", "2099-01-01T00:00:00"],
    })
    out = silver.clean_events(df, valid_user_ids={"u1"})
    assert list(out["event_id"]) == ["e1"]


def test_clean_events_drops_blank_user_id():
    df = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "user_id": ["", "u1"],
        "event_type": ["click", "click"],
        "timestamp": ["2024-06-01T10:00:00", "2024-06-01T10:00:00"],
    })
    out = silver.clean_events(df, valid_user_ids={"u1"})
    assert list(out["event_id"]) == ["e2"]


# --- feedback ---------------------------------------------------------------


def test_clean_feedback_drops_invalid_ratings():
    df = pd.DataFrame({
        "feedback_id": ["f1", "f2", "f3"],
        "user_id": ["u1", "u1", "u1"],
        "rating": ["5", "0", "7"],
        "comment": ["good", "bad", "wow"],
    })
    out = silver.clean_feedback(df, valid_user_ids={"u1"})
    assert list(out["feedback_id"]) == ["f1"]


def test_clean_feedback_drops_empty_comments():
    df = pd.DataFrame({
        "feedback_id": ["f1", "f2"],
        "user_id": ["u1", "u1"],
        "rating": ["5", "5"],
        "comment": ["good", "   "],
    })
    out = silver.clean_feedback(df, valid_user_ids={"u1"})
    assert list(out["feedback_id"]) == ["f1"]


# --- quality rules ----------------------------------------------------------


def test_require_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Schema violation"):
        q.require_columns(df, ["a", "b"], "test_table")


def test_check_drop_rate_flags_excessive_drops():
    res = q.check_drop_rate(rows_in=100, rows_out=70, threshold=0.10)
    assert res.passed is False
    assert "30.00%" in res.detail


def test_check_drop_rate_passes_under_threshold():
    res = q.check_drop_rate(rows_in=100, rows_out=95, threshold=0.10)
    assert res.passed is True
