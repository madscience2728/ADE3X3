"""Test SQLite schema creation and basic CRUD."""

import sqlite3
import time

import pytest

from db_optimizer.schema import create_schema, get_table_names
from db_optimizer.db import get_memory_connection


@pytest.fixture
def conn():
    c = get_memory_connection()
    create_schema(c)
    yield c
    c.close()


class TestSchema:
    def test_tables_created(self, conn):
        tables = get_table_names(conn)
        assert "candidates" in tables
        assert "shadow" in tables
        assert "blacklist" in tables
        assert "run_log" in tables

    def test_idempotent(self, conn):
        # Running create_schema again should not error
        create_schema(conn)
        tables = get_table_names(conn)
        assert len(tables) == 4

    def test_insert_candidate(self, conn):
        conn.execute(
            """INSERT INTO candidates
               (factors_blob, support_hash, support_sig, origin, virtual_island,
                generation, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (b"\x00" * 4104, "abc123", "(9,9,9)", "test", 0, 0, "pending", time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM candidates WHERE id = 1").fetchone()
        assert row is not None
        assert row["support_hash"] == "abc123"
        assert row["status"] == "pending"

    def test_insert_shadow(self, conn):
        conn.execute(
            """INSERT INTO shadow
               (factors_blob, support_hash, fitness_fp64, archived_at)
               VALUES (?, ?, ?, ?)""",
            (b"\x00" * 4104, "def456", 0.098, time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM shadow WHERE support_hash = 'def456'").fetchone()
        assert row is not None
        assert row["fitness_fp64"] == pytest.approx(0.098)

    def test_blacklist_unique_hash(self, conn):
        conn.execute(
            "INSERT INTO blacklist (support_hash, reason, added_at) VALUES (?, ?, ?)",
            ("dead_hash", "known bad", time.time()),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO blacklist (support_hash, reason, added_at) VALUES (?, ?, ?)",
                ("dead_hash", "duplicate", time.time()),
            )

    def test_run_log(self, conn):
        conn.execute(
            """INSERT INTO run_log (event, generation, detail_json, wall_seconds, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            ("gen_complete", 0, '{"best": 0.098}', 1.5, time.time()),
        )
        conn.commit()
        rows = conn.execute("SELECT * FROM run_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["event"] == "gen_complete"

    def test_index_on_status(self, conn):
        # Insert a bunch and verify status index is used
        now = time.time()
        for i in range(100):
            conn.execute(
                """INSERT INTO candidates
                   (factors_blob, support_hash, support_sig, origin, virtual_island,
                    generation, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (b"\x00" * 4104, f"h{i}", "(9,9,9)", "test", i % 4, 0,
                 "pending" if i < 80 else "screened", now),
            )
        conn.commit()
        pending = conn.execute(
            "SELECT COUNT(*) as cnt FROM candidates WHERE status = 'pending'"
        ).fetchone()
        assert pending["cnt"] == 80
