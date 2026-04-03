"""Orchestrator: main generation → evaluation → selection loop."""

import itertools
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from .config import (
    RANK, DIM, N_ISLANDS, GENERATION_BATCH_SIZE, MAX_PENDING,
    TIER1_SURVIVOR_THRESHOLD, TIER2_REFINE_THRESHOLD, HIT_THRESHOLD,
    MAX_GPU_BATCH_SIZE, ELITE_K, TOURNAMENT_SIZE,
    CPU_MINIMAX_SWEEPS, CPU_MINIMAX_FINE_RANGE,
)
from .schema import create_schema
from .db import get_connection, get_memory_connection
from .blob import (
    factors_to_blob, blob_to_factors, factors_from_json,
    compute_support_hash, compute_support_sig,
)
from .tensor import fitness_maxabs, fitness_frobenius
from .gpu_eval import gpu_batch_fitness
from .cpu_refine import cpu_minimax_refine
from .generator import generate_batch
from .selector import (
    select_parents, select_global_elites, fetch_pending, update_fitness_batch,
    fetch_survivors, fetch_refine_candidates, update_refined,
    prune_duplicates, archive_to_shadow, pending_count,
    insert_candidates, island_stats, log_event,
)


_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[orchestrator] Graceful shutdown requested (Ctrl+C). Finishing current generation...")


def seed_from_json(conn, json_paths: list[Path], generation: int = 0,
                   n_islands: int | None = None, perturbations: int = 20) -> int:
    """Import JSON candidates into the database with perturbed copies.

    For each JSON file, inserts the original plus *perturbations* mutated copies
    at geometrically increasing sigma, spread round-robin across all islands.
    """
    target_n_islands = n_islands if n_islands is not None else N_ISLANDS
    rng = np.random.default_rng()
    count = 0

    for p in json_paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        # Support both sparse ("terms") and dense ("alpha"/"beta"/"gamma") formats
        has_terms = len(data.get("terms", [])) == RANK
        has_dense = all(k in data for k in ("alpha", "beta", "gamma"))
        if not has_terms and not has_dense:
            continue
        alpha, beta, gamma = factors_from_json(data)

        # Insert original
        blob = factors_to_blob(alpha, beta, gamma)
        s_hash = compute_support_hash(alpha, beta, gamma)
        s_sig = compute_support_sig(alpha, beta, gamma)
        fit = fitness_maxabs(alpha, beta, gamma)
        fro = fitness_frobenius(alpha, beta, gamma)
        now = time.time()
        conn.execute(
            """INSERT INTO candidates
               (factors_blob, support_hash, support_sig, origin, virtual_island,
                fitness_fp32, fitness_fp64, fro_residual,
                generation, status, tier_reached, created_at, evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (blob, s_hash, s_sig, "seed", count % target_n_islands,
             float(fit), float(fit), float(fro),
             generation, "refined", 3, now, now),
        )
        conn.execute(
            """INSERT OR IGNORE INTO shadow
               (factors_blob, support_hash, fitness_fp64, origin_run, archived_at)
               VALUES (?, ?, ?, ?, ?)""",
            (blob, s_hash, float(fit), "seed", now),
        )
        count += 1

        # Insert perturbed copies at varying sigma, spread across islands
        sigmas = np.geomspace(0.001, 0.10, perturbations)
        for i, sigma in enumerate(sigmas):
            a2, b2, g2 = alpha.copy(), beta.copy(), gamma.copy()
            # Perturb a random subset of coefficients (5-30%)
            n_perturb = rng.integers(max(1, RANK), max(2, RANK * DIM // 3))
            for _ in range(n_perturb):
                fi = rng.integers(3)
                ri = rng.integers(RANK)
                di = rng.integers(DIM)
                [a2, b2, g2][fi][ri, di] += rng.normal(0, sigma)
            blob2 = factors_to_blob(a2, b2, g2)
            s_hash2 = compute_support_hash(a2, b2, g2)
            s_sig2 = compute_support_sig(a2, b2, g2)
            island = (count) % target_n_islands
            conn.execute(
                """INSERT INTO candidates
                   (factors_blob, support_hash, support_sig, origin, virtual_island,
                    fitness_fp32, fitness_fp64, fro_residual,
                    generation, status, tier_reached, created_at, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (blob2, s_hash2, s_sig2, f"seed_perturb_s{sigma:.4f}", island,
                 None, None, None,
                 generation, "pending", 0, now, None),
            )
            count += 1

    conn.commit()
    return count


def run(
    db_path: Path | str,
    max_generations: int = 10_000,
    seed_paths: list[Path] | None = None,
    verbose: bool = True,
) -> None:
    """Main orchestrator loop."""
    global _shutdown_requested
    _shutdown_requested = False

    signal.signal(signal.SIGINT, _signal_handler)

    conn = get_connection(db_path)
    create_schema(conn)

    # Seed initial candidates if DB is empty
    existing = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
    if existing == 0 and seed_paths:
        n_seeded = seed_from_json(conn, seed_paths)
        if verbose:
            print(f"[seed] Imported {n_seeded} candidates from JSON files")
        log_event(conn, "seed", generation=0, detail_json=json.dumps({"count": n_seeded}))

    rng = np.random.default_rng()
    t_start = time.time()

    for generation in range(max_generations):
        if _shutdown_requested:
            break

        t_gen = time.time()

        # ── Phase 1: Generate ──────────────────────────────────────
        # Check backpressure
        n_pending = pending_count(conn)
        if n_pending < MAX_PENDING:
            # Select parents from each island
            all_parents = []
            for island in range(N_ISLANDS):
                rows = select_parents(conn, island, k=ELITE_K, tournament_size=TOURNAMENT_SIZE)
                for row in rows:
                    a, b, g = blob_to_factors(row["factors_blob"])
                    all_parents.append((a, b, g))

            if not all_parents:
                # Fallback: global elites
                elites = select_global_elites(conn, ELITE_K * N_ISLANDS)
                for row in elites:
                    a, b, g = blob_to_factors(row["factors_blob"])
                    all_parents.append((a, b, g))

            if all_parents:
                children = generate_batch(all_parents, GENERATION_BATCH_SIZE, rng)
                # Insert children
                to_insert = []
                for alpha, beta, gamma, origin in children:
                    blob = factors_to_blob(alpha, beta, gamma)
                    s_hash = compute_support_hash(alpha, beta, gamma)
                    s_sig = compute_support_sig(alpha, beta, gamma)
                    v_island = hash(s_hash) % N_ISLANDS
                    to_insert.append((blob, s_hash, s_sig, origin, v_island, generation))
                insert_candidates(conn, to_insert)

        # ── Phase 2: Evaluate ─────────────────────────────────────

        # Tier 1: GPU bulk screen
        pending_rows = fetch_pending(conn, limit=MAX_GPU_BATCH_SIZE)
        if pending_rows:
            factors_list = [blob_to_factors(row["factors_blob"]) for row in pending_rows]
            ids = [row["id"] for row in pending_rows]
            fitness_fp32 = gpu_batch_fitness(factors_list)
            update_fitness_batch(conn, ids, fitness_fp32.tolist(), tier=1)

        # Tier 2: GPU minimax on survivors (simplified: just re-screen with tighter threshold)
        # Full GPU minimax sweep deferred to Phase 2.2

        # Tier 3: CPU FP64 refinement on elite candidates
        refine_rows = fetch_refine_candidates(conn, TIER2_REFINE_THRESHOLD)
        for row in refine_rows[:10]:  # limit to 10 per generation
            alpha, beta, gamma = blob_to_factors(row["factors_blob"])
            alpha_r, beta_r, gamma_r, fit64 = cpu_minimax_refine(
                alpha, beta, gamma,
                sweeps=CPU_MINIMAX_SWEEPS,
                fine_range=CPU_MINIMAX_FINE_RANGE,
            )
            fro = fitness_frobenius(alpha_r, beta_r, gamma_r)
            new_blob = factors_to_blob(alpha_r, beta_r, gamma_r)
            update_refined(conn, row["id"], new_blob, fit64, fro)

            # Archive to shadow if improved
            archive_to_shadow(conn, row["id"])

        # ── Selection + housekeeping ──────────────────────────────
        n_pruned = prune_duplicates(conn)

        # ── Logging ───────────────────────────────────────────────
        best_row = conn.execute(
            """SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best_fit
               FROM candidates WHERE fitness_fp32 IS NOT NULL"""
        ).fetchone()
        best_fit = best_row["best_fit"] if best_row else None

        total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
        wall = time.time() - t_gen

        if verbose:
            print(
                f"[gen {generation:4d}] best={best_fit:.10f} "
                f"total={total} pending={len(pending_rows)} "
                f"pruned={n_pruned} wall={wall:.1f}s"
            )

        log_event(conn, "gen_complete", generation=generation,
                   detail_json=json.dumps({
                       "best": best_fit,
                       "total": total,
                       "pending": len(pending_rows),
                       "pruned": n_pruned,
                   }),
                   wall_seconds=wall)

        # ── Termination check ─────────────────────────────────────
        if best_fit is not None and best_fit < HIT_THRESHOLD:
            print(f"\n*** EXACT HIT at generation {generation}! fitness={best_fit:.2e} ***")
            break

    # ── Shutdown ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n[orchestrator] Finished after {generation + 1} generations ({elapsed:.1f}s total)")

    # Export best candidate
    best = conn.execute(
        """SELECT factors_blob, COALESCE(fitness_fp64, fitness_fp32) as fit
           FROM candidates
           WHERE fitness_fp32 IS NOT NULL
           ORDER BY COALESCE(fitness_fp64, fitness_fp32) ASC
           LIMIT 1"""
    ).fetchone()
    if best:
        alpha, beta, gamma = blob_to_factors(best["factors_blob"])
        from .blob import factors_to_json
        data = factors_to_json(alpha, beta, gamma)
        out_path = Path(db_path).parent / "best_candidate.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[export] Best candidate (fit={best['fit']:.10f}) → {out_path}")

    conn.close()
