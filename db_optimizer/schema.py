"""SQLite schema creation and migration."""

import sqlite3


SCHEMA_SQL = """
-- Main candidate table
CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factors_blob    BLOB NOT NULL,
    support_hash    TEXT NOT NULL,
    support_sig     TEXT NOT NULL,
    origin          TEXT NOT NULL,
    parent_ids      TEXT,
    virtual_island  INTEGER NOT NULL,
    fitness_fp32    REAL,
    fitness_fp64    REAL,
    fro_residual    REAL,
    generation      INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      REAL NOT NULL,
    evaluated_at    REAL,
    minimax_improved REAL,
    tier_reached    INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_fitness
    ON candidates(fitness_fp32);
CREATE INDEX IF NOT EXISTS idx_candidates_support_hash
    ON candidates(support_hash);
CREATE INDEX IF NOT EXISTS idx_candidates_island_fitness
    ON candidates(virtual_island, fitness_fp32);
CREATE INDEX IF NOT EXISTS idx_candidates_generation
    ON candidates(generation);

-- Shadow archive: basin-escape memory
CREATE TABLE IF NOT EXISTS shadow (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factors_blob    BLOB NOT NULL,
    support_hash    TEXT NOT NULL UNIQUE,
    fitness_fp64    REAL NOT NULL,
    origin_run      TEXT,
    archived_at     REAL NOT NULL
);

-- Blacklisted support patterns
CREATE TABLE IF NOT EXISTS blacklist (
    support_hash    TEXT PRIMARY KEY,
    reason          TEXT,
    added_at        REAL NOT NULL
);

-- Run log for audit
CREATE TABLE IF NOT EXISTS run_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event           TEXT NOT NULL,
    generation      INTEGER,
    detail_json     TEXT,
    wall_seconds    REAL,
    timestamp       REAL NOT NULL
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes (idempotent)."""
    conn.executescript(SCHEMA_SQL)
    _migrate(conn)


_MIGRATIONS = [
    # (column, table, type, default)
    ("dead_energy", "candidates", "REAL", None),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema (safe to re-run)."""
    for col, table, dtype, default in _MIGRATIONS:
        try:
            default_clause = f" DEFAULT {default}" if default is not None else ""
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}{default_clause}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return list of user table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(r[0] for r in rows)
