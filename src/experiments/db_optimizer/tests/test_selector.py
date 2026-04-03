"""Test selector queries: tournament, elitism, pruning, archival."""

import time

import numpy as np
import pytest

from db_optimizer.config import RANK, DIM
from db_optimizer.db import get_memory_connection
from db_optimizer.schema import create_schema
from db_optimizer.blob import factors_to_blob
from db_optimizer.selector import (
    select_parents,
    select_elites,
    select_global_elites,
    fetch_pending,
    update_fitness_batch,
    prune_duplicates,
    insert_candidates,
    pending_count,
    island_stats,
    log_event,
)


@pytest.fixture
def conn():
    c = get_memory_connection()
    create_schema(c)
    yield c
    c.close()


def _insert_random_candidates(conn, n, rng, generation=0, status="pending", fitness=None):
    """Insert n random candidates for testing."""
    now = time.time()
    for i in range(n):
        a = rng.standard_normal((RANK, DIM))
        b = rng.standard_normal((RANK, DIM))
        g = rng.standard_normal((RANK, DIM))
        blob = factors_to_blob(a, b, g)
        fit = fitness if fitness is not None else rng.random() * 2
        conn.execute(
            """INSERT INTO candidates
               (factors_blob, support_hash, support_sig, origin, virtual_island,
                fitness_fp32, generation, status, created_at, evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (blob, f"hash_{i}_{rng.integers(1000000)}", "(9,9,9)", "test",
             i % 4, fit if status != "pending" else None,
             generation, status, now, now if status != "pending" else None),
        )
    conn.commit()


class TestSelectParents:
    def test_returns_k(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 50, rng, status="screened")
        parents = select_parents(conn, island=0, k=5, tournament_size=3)
        assert len(parents) == 5

    def test_empty_island(self, conn):
        parents = select_parents(conn, island=0, k=5)
        assert parents == []


class TestSelectElites:
    def test_returns_top_k(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 50, rng, status="screened")
        elites = select_elites(conn, island=0, k=3)
        assert len(elites) <= 3
        # Should be sorted by fitness
        for i in range(len(elites) - 1):
            assert elites[i]["fitness_fp32"] <= elites[i + 1]["fitness_fp32"]


class TestPending:
    def test_fetch_pending(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 20, rng, status="pending")
        rows = fetch_pending(conn, limit=10)
        assert len(rows) == 10

    def test_pending_count(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 30, rng, status="pending")
        assert pending_count(conn) == 30


class TestUpdateFitness:
    def test_batch_update(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 10, rng, status="pending")
        rows = fetch_pending(conn, limit=10)
        ids = [r["id"] for r in rows]
        fitnesses = [0.1 * i for i in range(len(ids))]
        update_fitness_batch(conn, ids, fitnesses, tier=1)

        # Verify
        for cid, fit in zip(ids, fitnesses):
            row = conn.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
            assert row["status"] == "screened"
            assert row["fitness_fp32"] == pytest.approx(fit)
            assert row["tier_reached"] == 1


class TestPruneDuplicates:
    def test_prune_keeps_best(self, conn):
        now = time.time()
        blob = factors_to_blob(
            np.zeros((RANK, DIM)),
            np.zeros((RANK, DIM)),
            np.zeros((RANK, DIM)),
        )
        # Insert 3 candidates with same support_hash, different fitness
        for fit in [0.5, 0.1, 0.3]:
            conn.execute(
                """INSERT INTO candidates
                   (factors_blob, support_hash, support_sig, origin, virtual_island,
                    fitness_fp32, generation, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (blob, "same_hash", "(0,0,0)", "test", 0, fit, 0, "screened", now),
            )
        conn.commit()

        pruned = prune_duplicates(conn)
        assert pruned == 2
        remaining = conn.execute("SELECT * FROM candidates").fetchall()
        assert len(remaining) == 1
        assert remaining[0]["fitness_fp32"] == pytest.approx(0.1)


class TestInsertCandidates:
    def test_batch_insert(self, conn):
        blob = factors_to_blob(
            np.ones((RANK, DIM)),
            np.ones((RANK, DIM)),
            np.ones((RANK, DIM)),
        )
        candidates = [
            (blob, f"hash_{i}", "(9,9,9)", "test", i % 4, 0)
            for i in range(10)
        ]
        insert_candidates(conn, candidates)
        total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
        assert total == 10


class TestIslandStats:
    def test_stats(self, conn):
        rng = np.random.default_rng(42)
        _insert_random_candidates(conn, 40, rng, status="screened")
        stats = island_stats(conn, 4)
        assert len(stats) == 4
        for s in stats:
            assert s["count"] >= 0
            if s["count"] > 0:
                assert s["best"] is not None
                assert s["best"] <= s["mean"]


class TestLogEvent:
    def test_log(self, conn):
        log_event(conn, "test_event", generation=0, detail_json='{"foo": 1}')
        rows = conn.execute("SELECT * FROM run_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["event"] == "test_event"
