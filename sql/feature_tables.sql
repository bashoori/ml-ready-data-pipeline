-- =====================================================================
-- Gold-layer feature tables
-- ---------------------------------------------------------------------
-- These DDL statements mirror the Parquet schemas the pipeline writes.
-- Drop into Postgres / Redshift / Microsoft Fabric Lakehouse with minor
-- type-name adjustments (e.g. `TIMESTAMP` vs `DATETIME`, `BOOLEAN` vs
-- `TINYINT`).
-- =====================================================================


-- ---------------------------------------------------------------------
-- user_features
-- One row per user, denormalized — the wide, pre-aggregated table an
-- ML training pipeline can load directly without further joins.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_features (
    user_id                 VARCHAR(64)  NOT NULL,
    email                   VARCHAR(255) NOT NULL,
    signup_date             DATE         NOT NULL,
    age                     SMALLINT     NOT NULL,
    country                 VARCHAR(64)  NOT NULL,

    -- Behavioral features
    n_events                INTEGER      NOT NULL DEFAULT 0,
    n_logins                INTEGER      NOT NULL DEFAULT 0,
    n_purchases             INTEGER      NOT NULL DEFAULT 0,
    first_event_at          TIMESTAMP    NULL,
    last_event_at           TIMESTAMP    NULL,
    days_active             INTEGER      NOT NULL DEFAULT 0,

    -- Sentiment features (from feedback)
    n_feedback              INTEGER      NOT NULL DEFAULT 0,
    avg_rating              DECIMAL(3,2) NULL,
    has_negative_feedback   BOOLEAN      NOT NULL DEFAULT FALSE,

    PRIMARY KEY (user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_features_country  ON user_features (country);
CREATE INDEX IF NOT EXISTS idx_user_features_signup   ON user_features (signup_date);


-- ---------------------------------------------------------------------
-- event_facts
-- Long-form fact table for behavioral analysis. Joined to user
-- attributes (country, age) for slicing without re-joining at query time.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_facts (
    event_id     VARCHAR(64)  NOT NULL,
    user_id      VARCHAR(64)  NOT NULL,
    event_type   VARCHAR(32)  NOT NULL,
    timestamp    TIMESTAMP    NOT NULL,
    country      VARCHAR(64)  NULL,
    age          SMALLINT     NULL,

    PRIMARY KEY (event_id)
);

CREATE INDEX IF NOT EXISTS idx_event_facts_user_ts  ON event_facts (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_event_facts_type     ON event_facts (event_type);
