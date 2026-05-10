"""Unit tests for Gold-layer feature engineering."""
from __future__ import annotations

import pandas as pd

from src import gold


def _users():
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "email": ["a@x.com", "b@x.com"],
        "signup_date": pd.to_datetime(["2024-01-01", "2024-02-01"]).date,
        "age": [30, 40],
        "country": ["CA", "US"],
    })


def _events():
    return pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "user_id": ["u1", "u1", "u1", "u2"],
        "event_type": ["login", "click", "purchase", "login"],
        "timestamp": pd.to_datetime([
            "2024-03-01T10:00", "2024-03-02T10:00",
            "2024-03-05T10:00", "2024-03-10T10:00",
        ]),
    })


def _feedback():
    return pd.DataFrame({
        "feedback_id": ["f1", "f2"],
        "user_id": ["u1", "u1"],
        "rating": [5, 1],
        "comment": ["love", "hate"],
    })


def test_user_features_one_row_per_user():
    out = gold.build_user_features(_users(), _events(), _feedback())
    assert len(out) == 2
    assert set(out["user_id"]) == {"u1", "u2"}


def test_user_features_event_counts():
    out = gold.build_user_features(_users(), _events(), _feedback()).set_index("user_id")
    assert out.loc["u1", "n_events"] == 3
    assert out.loc["u1", "n_logins"] == 1
    assert out.loc["u1", "n_purchases"] == 1
    assert out.loc["u2", "n_events"] == 1


def test_user_features_user_with_no_events_has_zero_counts():
    users = pd.DataFrame({
        "user_id": ["u_lonely"],
        "email": ["x@x.com"],
        "signup_date": pd.to_datetime(["2024-01-01"]).date,
        "age": [30],
        "country": ["CA"],
    })
    events = pd.DataFrame(columns=["event_id", "user_id", "event_type", "timestamp"])
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    feedback = pd.DataFrame(columns=["feedback_id", "user_id", "rating", "comment"])
    out = gold.build_user_features(users, events, feedback)
    assert out.loc[0, "n_events"] == 0
    assert out.loc[0, "n_logins"] == 0
    assert out.loc[0, "n_feedback"] == 0


def test_user_features_negative_feedback_flag():
    out = gold.build_user_features(_users(), _events(), _feedback()).set_index("user_id")
    # u1 has a 1-star rating → flag is True
    assert bool(out.loc["u1", "has_negative_feedback"]) is True
    # u2 has no feedback → flag defaults False
    assert bool(out.loc["u2", "has_negative_feedback"]) is False


def test_event_facts_has_user_attributes():
    out = gold.build_event_facts(_events(), _users())
    assert {"country", "age"}.issubset(out.columns)
    u1_row = out[out["user_id"] == "u1"].iloc[0]
    assert u1_row["country"] == "CA"
    assert u1_row["age"] == 30
