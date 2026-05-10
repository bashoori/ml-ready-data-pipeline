# ML-Ready Data Pipeline

> A medallion-architecture pipeline that turns messy, multi-source raw data into validated, observable, ML-ready feature tables.

Built in Python + Pandas + SQL. Designed around the same patterns I use on Databricks / Microsoft Fabric in production, scaled down so it runs on a laptop.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SQL](https://img.shields.io/badge/SQL-PostgreSQL%2FDuckDB-336791?logo=postgresql&logoColor=white)](#)
[![Architecture](https://img.shields.io/badge/Architecture-Medallion-0e75b6)](#)

---

## Why this project exists

Real-world data that lands in front of an ML or analytics team almost never arrives clean:

- Multiple sources, multiple formats (CSV, JSONL, TSV)
- Inconsistent schemas — column names drift, types are wrong
- Quality issues hidden in plain sight — duplicates, nulls, out-of-range values
- No observability — you can't tell if today's training data is comparable to yesterday's

This pipeline shows the engineering discipline that makes that data **trustworthy**: a layered medallion architecture, embedded data-quality checks at every layer, and an observability artifact written every run so consumers (analysts, ML researchers) can see exactly what happened to their data.

The same patterns I use here scale up directly to Spark / Delta Lake / Microsoft Fabric in production — only the runtime changes.

---

## Architecture

```
              ┌──────────────┐
   raw CSV ──▶│   BRONZE     │   raw ingest, preserves fidelity
   raw JSONL ▶│  (data/raw)  │   no transformations, no drops
   raw TSV ──▶└──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   SILVER     │   schema enforcement, type coercion,
              │ (validated)  │   deduplication, embedded DQ checks
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │     GOLD     │   joined feature tables,
              │  (ML-ready)  │   ready for downstream ML / analytics
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  DQ REPORT   │   per-run observability artifact:
              │ (reports/)   │   row counts, null rates, dropped rows,
              │              │   threshold violations
              └──────────────┘
```

### Why medallion?

- **Bronze keeps raw fidelity.** If a downstream consumer disputes a number, I can replay any transformation from raw input.
- **Silver is the trust boundary.** Anything that escapes Silver has been validated and is safe to use.
- **Gold is consumer-shaped.** Different consumers (BI, ML training, ad-hoc analysis) get tables shaped for their use case, all derived from the same Silver source of truth.

---

## What's in the box

| Layer | What happens | Code |
|---|---|---|
| Bronze | Reads CSV / JSONL / TSV from `data/raw/`. Preserves every row, every column. | `src/bronze.py` |
| Silver | Schema enforcement, type coercion, deduplication, configurable DQ rules. | `src/silver.py`, `src/quality.py` |
| Gold | Joins Silver tables into ML-ready feature tables. Writes Parquet for fast downstream loads. | `src/gold.py` |
| Observability | Every run writes a JSON + Markdown DQ report to `reports/`. | `src/logger.py` |
| SQL | Feature-table schemas in DDL — drop straight into Postgres / Redshift / Fabric. | `sql/feature_tables.sql` |
| Tests | Unit tests on Silver cleaning logic and Gold joins. | `tests/` |

---

## Sample data

Three deliberately messy sources live in `data/raw/`:

- **`users.csv`** — user profiles. Includes duplicate rows, mixed-case emails, null `signup_date`s, an obviously bad age value, and trailing whitespace.
- **`events.jsonl`** — user events (line-delimited JSON). Includes events with missing `user_id`, future-dated timestamps, and inconsistent `event_type` casing.
- **`feedback.tsv`** — free-text feedback. Includes empty rows, encoding artefacts, and ratings outside the valid 1–5 range.

The pipeline's job: turn these into a single, trustworthy `user_features` Gold table you'd hand to an ML team for training, plus an event-level fact table for behavioral analysis.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/<you>/ml-ready-data-pipeline.git
cd ml-ready-data-pipeline
pip install -r requirements.txt

# 2. Run the pipeline
python -m src.pipeline

# 3. Inspect outputs
ls data/silver/   # validated parquet tables
ls data/gold/     # ML-ready feature tables
ls reports/       # DQ reports (JSON + Markdown)

# 4. Run tests
pytest -v
```

### Sample output (DQ report)

After running, `reports/dq_report_<timestamp>.md` looks like:

```
## Data Quality Report — 2026-05-10 18:42 UTC

### Bronze
- users:    1,000 rows ingested
- events:   8,742 rows ingested
- feedback:   612 rows ingested

### Silver — validation results
| Table    | In  | Out | Dropped | Null-rate (key) | Status |
|----------|-----|-----|---------|-----------------|--------|
| users    | 1000| 982 | 18      | 0.0%            | ✅ PASS |
| events   | 8742| 8501| 241     | 0.0%            | ✅ PASS |
| feedback | 612 | 598 | 14      | 0.0%            | ✅ PASS |

### Threshold violations
None — all DQ rules passed.

### Gold
- user_features:  982 rows
- event_facts:    8,501 rows
```

---

## Design choices worth calling out

**1. Quality checks are code, not comments.** Every Silver rule lives in `src/quality.py` as a function with a name, a description, and a return value — pass / fail / threshold-exceeded. They run on every load and the result lands in the DQ report. No "we'll add validation later."

**2. The pipeline fails loudly when something matters, quietly when it doesn't.** Schema mismatches and missing required columns hard-fail. Soft issues (e.g., a column whose null rate creeps up by 2%) get logged in the DQ report but don't block the run — that's a signal for the team to investigate, not a reason to break production.

**3. Bronze is append-only and reproducible.** No transformations land here. If a researcher needs to ask *"why did this user disappear from training data this week?"* — the trace starts in Bronze.

**4. Gold tables are shaped for consumers, not for storage.** `user_features` is denormalized and optimized for a single read; `event_facts` is the long-form fact table. Different consumers, different tables, both derived from the same Silver layer.

**5. Local-first, but production-shaped.** This runs on Pandas so reviewers can clone and execute it in 30 seconds. The same module structure (`bronze` / `silver` / `gold` / `quality` / `logger`) is what I'd use in PySpark on Databricks — you'd swap the I/O layer and the compute engine, but the contracts and DQ rules would stay identical.

---

## Project layout

```
ml-ready-data-pipeline/
├── README.md
├── requirements.txt
├── config.yaml                 # source paths and DQ thresholds
├── data/
│   ├── raw/                    # bronze input (committed for repro)
│   ├── silver/                 # validated parquet (gitignored)
│   └── gold/                   # ML-ready parquet (gitignored)
├── src/
│   ├── pipeline.py             # entrypoint — runs Bronze → Silver → Gold
│   ├── bronze.py
│   ├── silver.py
│   ├── gold.py
│   ├── quality.py              # DQ rules
│   └── logger.py               # DQ report writer
├── sql/
│   └── feature_tables.sql      # Gold-layer DDL
├── tests/
│   ├── test_silver.py
│   └── test_gold.py
└── reports/                    # per-run DQ reports
```

---

## Scaling to production

This project deliberately uses Pandas so it stays inspectable and runnable. Moving the same shape to a real platform is mostly a swap of the I/O and compute layer:

| Concern | This project | Production swap |
|---|---|---|
| Compute | Pandas | PySpark on Databricks / Microsoft Fabric |
| Storage | Local Parquet | Delta Lake on OneLake / S3 |
| Orchestration | Single `python -m src.pipeline` | Apache Airflow / Azure Data Factory / Fabric Pipelines |
| Quality monitoring | DQ report file | Quality dashboard, alerting on threshold breach |
| Schema governance | DDL in `sql/` | Unity Catalog / Fabric Lakehouse schemas |

Same pattern, same contracts. The discipline scales.

---

## License

MIT — feel free to reuse and adapt.
