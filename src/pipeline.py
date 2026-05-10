"""
Pipeline orchestrator.

Wires Bronze → Silver → Gold and writes the per-run DQ report.

Run:
    python -m src.pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from . import bronze, silver, gold, logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def run() -> int:
    print("▶ ML-Ready Data Pipeline\n")

    cfg = load_config(PROJECT_ROOT / "config.yaml")
    paths = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    sources = cfg["sources"]
    thresholds = cfg["dq_thresholds"]

    # --- BRONZE ---
    print("▸ BRONZE — ingesting raw sources")
    raw = bronze.ingest_all(sources, paths["raw"])
    bronze_counts = {name: len(df) for name, df in raw.items()}
    for name, n in bronze_counts.items():
        print(f"    {name:10s}  {n:>7,} rows")

    # --- SILVER ---
    print("\n▸ SILVER — schema enforcement, cleaning, validation")
    cleaned: dict = {}
    silver_results: dict = {}

    # Users first — its keys are the referential foundation for the others
    users_in = raw["users"]
    users_clean = silver.clean_users(users_in)
    valid_user_ids = set(users_clean["user_id"])

    cleaned["users"] = users_clean
    silver_results["users"] = {
        "rows_in": len(users_in),
        "rows_out": len(users_clean),
        "results": logger.serialize_results(
            silver.validate(
                users_in, users_clean,
                table="users",
                primary_key=sources["users"]["primary_key"],
                required_columns=sources["users"]["required_columns"],
                thresholds=thresholds,
            )
        ),
    }

    events_in = raw["events"]
    events_clean = silver.clean_events(events_in, valid_user_ids)
    cleaned["events"] = events_clean
    silver_results["events"] = {
        "rows_in": len(events_in),
        "rows_out": len(events_clean),
        "results": logger.serialize_results(
            silver.validate(
                events_in, events_clean,
                table="events",
                primary_key=sources["events"]["primary_key"],
                required_columns=sources["events"]["required_columns"],
                thresholds=thresholds,
            )
        ),
    }

    feedback_in = raw["feedback"]
    feedback_clean = silver.clean_feedback(feedback_in, valid_user_ids)
    cleaned["feedback"] = feedback_clean
    silver_results["feedback"] = {
        "rows_in": len(feedback_in),
        "rows_out": len(feedback_clean),
        "results": logger.serialize_results(
            silver.validate(
                feedback_in, feedback_clean,
                table="feedback",
                primary_key=sources["feedback"]["primary_key"],
                required_columns=sources["feedback"]["required_columns"],
                thresholds=thresholds,
            )
        ),
    }

    # Add overall pass/fail flag for the summary table
    for name, info in silver_results.items():
        info["all_passed"] = all(r["passed"] for r in info["results"])

    # Persist Silver
    for name, df in cleaned.items():
        path = silver.write(df, name, paths["silver"])
        print(f"    {name:10s}  {len(df):>7,} rows  →  {path.relative_to(PROJECT_ROOT)}")

    # --- GOLD ---
    print("\n▸ GOLD — ML-ready feature tables")
    user_features = gold.build_user_features(
        cleaned["users"], cleaned["events"], cleaned["feedback"]
    )
    event_facts = gold.build_event_facts(cleaned["events"], cleaned["users"])

    gold_tables = {"user_features": user_features, "event_facts": event_facts}
    gold_counts: dict[str, int] = {}
    for name, df in gold_tables.items():
        path = gold.write(df, name, paths["gold"])
        gold_counts[name] = len(df)
        print(f"    {name:14s}  {len(df):>7,} rows  →  {path.relative_to(PROJECT_ROOT)}")

    # --- DQ REPORT ---
    print("\n▸ DQ REPORT")
    json_path, md_path = logger.write_report(
        bronze_counts=bronze_counts,
        silver_results=silver_results,
        gold_counts=gold_counts,
        reports_dir=paths["reports"],
    )
    print(f"    {json_path.relative_to(PROJECT_ROOT)}")
    print(f"    {md_path.relative_to(PROJECT_ROOT)}")

    # Surface any soft-rule failures at the end
    soft_fails = [
        f"{tbl} · {r['rule']}: {r['detail']}"
        for tbl, info in silver_results.items()
        for r in info["results"]
        if not r["passed"]
    ]
    if soft_fails:
        print("\n⚠ Soft DQ warnings (run did not fail):")
        for s in soft_fails:
            print(f"    - {s}")
    else:
        print("\n✓ All DQ checks passed.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
