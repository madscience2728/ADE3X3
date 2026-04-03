"""End-to-end test: 3-generation run on in-memory DB."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from db_optimizer.config import RANK, DIM, TARGET_TENSOR
from db_optimizer.db import get_memory_connection
from db_optimizer.schema import create_schema
from db_optimizer.blob import factors_to_blob, blob_to_factors, factors_to_json
from db_optimizer.tensor import fitness_maxabs
from db_optimizer.gpu_eval import gpu_batch_fitness
from db_optimizer.selector import (
    insert_candidates, fetch_pending, update_fitness_batch,
    select_global_elites, pending_count,
)
from db_optimizer.generator import generate_batch


class TestEndToEnd:
    def test_3_generation_loop(self):
        """Simulate 3 generations: seed → generate → screen → repeat."""
        conn = get_memory_connection()
        create_schema(conn)

        rng = np.random.default_rng(42)

        # Seed: 10 random candidates
        seed_candidates = []
        for i in range(10):
            a = rng.standard_normal((RANK, DIM)) * 0.1
            b = rng.standard_normal((RANK, DIM)) * 0.1
            g = rng.standard_normal((RANK, DIM)) * 0.1
            blob = factors_to_blob(a, b, g)
            seed_candidates.append((blob, f"seed_{i}", "(9,9,9)", "seed", i % 4, 0))
        insert_candidates(conn, seed_candidates)
        assert pending_count(conn) == 10

        for gen in range(3):
            # Evaluate pending
            pending = fetch_pending(conn, limit=1000)
            if pending:
                factors_list = [blob_to_factors(r["factors_blob"]) for r in pending]
                ids = [r["id"] for r in pending]
                fits = gpu_batch_fitness(factors_list)
                update_fitness_batch(conn, ids, fits.tolist(), tier=1)

            # Select parents
            elites = select_global_elites(conn, k=5)
            parents = [blob_to_factors(r["factors_blob"]) for r in elites]

            if parents:
                # Generate children
                children = generate_batch(parents, 20, rng)
                to_insert = []
                for a, b, g, origin in children:
                    blob = factors_to_blob(a, b, g)
                    to_insert.append((blob, f"gen{gen}_{rng.integers(1000000)}", "(9,9,9)", origin, rng.integers(4), gen + 1))
                insert_candidates(conn, to_insert)

        # Final evaluation
        pending = fetch_pending(conn, limit=1000)
        if pending:
            factors_list = [blob_to_factors(r["factors_blob"]) for r in pending]
            ids = [r["id"] for r in pending]
            fits = gpu_batch_fitness(factors_list)
            update_fitness_batch(conn, ids, fits.tolist(), tier=1)

        # Check: DB is consistent
        total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
        assert total >= 10  # at least the seeds
        assert pending_count(conn) == 0  # all evaluated

        # Best fitness should be finite
        best = conn.execute(
            "SELECT MIN(fitness_fp32) as best FROM candidates"
        ).fetchone()["best"]
        assert best is not None
        assert best > 0
        assert np.isfinite(best)

        conn.close()

    def test_seed_from_json_file(self, tmp_path):
        """Test seeding from a JSON file."""
        # Create a fake candidate JSON
        rng = np.random.default_rng(42)
        a = rng.standard_normal((RANK, DIM))
        b = rng.standard_normal((RANK, DIM))
        g = rng.standard_normal((RANK, DIM))
        data = factors_to_json(a, b, g)

        json_path = tmp_path / "optimized_test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        from db_optimizer.orchestrator import seed_from_json
        conn = get_memory_connection()
        create_schema(conn)
        count = seed_from_json(conn, [json_path])
        assert count == 1

        row = conn.execute("SELECT * FROM candidates WHERE origin = 'seed'").fetchone()
        assert row is not None
        assert row["status"] == "refined"
        assert row["fitness_fp64"] is not None

        # Shadow archive should also have it
        shadow = conn.execute("SELECT * FROM shadow").fetchone()
        assert shadow is not None

        conn.close()
