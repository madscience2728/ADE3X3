"""db_optimizer main entry point.

Runs the optimizer as a 4-stage parallel pipeline and serves a live dashboard.
  Thread A (generator)    – mutate/crossover → blob encode → insert to DB
  Thread B (GPU eval)     – fetch pending → GPU batch screen (Tier 1)
                            → GPU minimax refine (Tier 2) → push to gpu_out_queue
  Thread C (CPU refine)   – pop from gpu_out_queue → ProcessPool adaptive polish (Tier 3) → update DB
  Thread D (DB writer)    – single writer drains a queue for all DB mutations

Usage:  python main.py              (uses K:/ade3x3_optimizer/candidates.db, seeds from repo root)
        python main.py --db my.db   (custom DB path)
"""

import json
import math
import mimetypes
import os
import queue as _queue_mod
import signal
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ── paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DB_OPTIMIZER_DIR = REPO_ROOT / "db_optimizer"
STATIC_DIR = DB_OPTIMIZER_DIR / "static"
HOST = "127.0.0.1"
PORT = 8766

# ── optimizer thread state ───────────────────────────────────────────
_optimizer_thread: threading.Thread | None = None
_optimizer_lock = threading.Lock()
_optimizer_status = {
    "running": False,
    "generation": 0,
    "best_fitness": None,
    "total_candidates": 0,
    "pending": 0,
    "start_time": None,
    "error": None,
}

# ── pipeline activity tracking ───────────────────────────────────────
_activity_lock = threading.Lock()
_pipeline_activity: dict[str, dict] = {
    "generator":   {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
    "gpu_screen":  {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
    "gpu_minimax": {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
    "cpu_refine":  {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
    "db_io":       {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
    "monitor":     {"state": "idle", "detail": "", "last_active": 0.0, "cycles": 0, "last_items": 0, "last_dt": 0.0},
}
_runtime_island_layout: list[dict] = []


def _set_activity(stage: str, state: str, detail: str = "", items: int = 0, dt: float = 0.0):
    """Update the activity state for a pipeline stage."""
    with _activity_lock:
        a = _pipeline_activity[stage]
        a["state"] = state
        a["detail"] = detail
        if state != "idle":
            a["last_active"] = time.time()
        if items > 0:
            a["last_items"] = items
            a["last_dt"] = dt
            a["cycles"] += 1


def _get_activity() -> dict:
    """Return current activity snapshot for all stages."""
    now = time.time()
    with _activity_lock:
        result = {}
        for stage, a in _pipeline_activity.items():
            active = a["state"] != "idle" and (now - a["last_active"]) < 3.0
            rate = a["last_items"] / a["last_dt"] if a["last_dt"] > 0.01 else 0.0
            result[stage] = {
                "state": a["state"],
                "detail": a["detail"],
                "active": active,
                "cycles": a["cycles"],
                "rate": round(rate, 1),
            }
        return result


def _backup_db_file(db_path: Path) -> Path | None:
    """Create a timestamped SQLite backup next to the DB file."""
    if not db_path.exists():
        return None

    db_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}.backup-{timestamp}{db_path.suffix}")

    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    return backup_path


def _reset_db_contents(db_path: Path) -> None:
    """Clear all user tables in-place so a fresh run works on Windows with open handles."""
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        table_names = [row[0] for row in rows]

        for table_name in table_names:
            conn.execute(f'DELETE FROM "{table_name}"')

        has_sequence = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()
        if has_sequence:
            for table_name in table_names:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table_name,))

        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _prepare_fresh_run(db_path: Path, make_backup: bool) -> Path | None:
    """Optionally back up the current DB, then delete it for a fresh run."""
    backup_path = _backup_db_file(db_path) if make_backup else None
    _reset_db_contents(db_path)
    return backup_path


def _allocate_island_budget(islands: list[dict], total_budget: int) -> dict[int, int]:
    """Allocate offspring budget across islands using cap-weighted fair sharing."""
    allocations = {item["id"]: 0 for item in islands}
    remaining = max(0, total_budget)
    while remaining > 0:
        available = [item for item in islands if allocations[item["id"]] < item["room"]]
        if not available:
            break
        target = max(
            available,
            key=lambda item: (item["weight"] / (allocations[item["id"]] + 1), item["room"]),
        )
        allocations[target["id"]] += 1
        remaining -= 1
    return {island_id: count for island_id, count in allocations.items() if count > 0}


def _pick_evenly(values: list[int], count: int) -> list[int]:
    """Pick count values spread across an ordered list."""
    if count <= 0:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]

    chosen_idx: list[int] = []
    for i in range(count):
        idx = round(i * (len(values) - 1) / (count - 1))
        if idx not in chosen_idx:
            chosen_idx.append(idx)

    probe = 0
    while len(chosen_idx) < count:
        if probe not in chosen_idx:
            chosen_idx.append(probe)
        probe += 1

    chosen_idx.sort()
    return [values[idx] for idx in chosen_idx[:count]]


def _select_active_island_layout(cfg_module, requested_islands: int) -> list[dict]:
    """Select an active subset of the full role×size island lattice.

    The subset preserves the two-axis structure by spreading each role across
    the available size exponents rather than just taking the first N ids.
    """
    full_count = len(cfg_module.ISLAND_LAYOUT)
    target = max(1, min(int(requested_islands), full_count))
    roles = list(cfg_module.ISLAND_ROLE_ORDER)
    size_exps = list(cfg_module.ISLAND_SIZE_EXPONENTS)

    base = target // len(roles)
    remainder = target % len(roles)

    selected: list[dict] = []
    next_id = 0
    for role_idx, role in enumerate(roles):
        n_sizes = base + (1 if role_idx < remainder else 0)
        if n_sizes <= 0:
            continue
        for size_exp in _pick_evenly(size_exps, n_sizes):
            selected.append({
                "id": next_id,
                "role": role,
                "size_exp": size_exp,
                "cap": 2 ** size_exp,
            })
            next_id += 1
    return selected


def _run_optimizer(db_path: Path, seed_dir: Path, max_generations: int, config_overrides: dict | None = None):
    """Target for the optimizer background thread.

    Runs a 4-stage parallel pipeline:
      Thread A (generator)    – per-island mutation → blob encode/hash → insert to DB
      Thread B (GPU eval)     – fetch pending → Tier 1 screen → Tier 2 minimax → gpu_out_queue
      Thread C (CPU refine)   – gpu_out_queue → ProcessPool adaptive polish → update DB
    All coordinate via the DB (SQLite WAL) + in-memory queues.
    """
    global _optimizer_status

    # Late import so the web server starts fast
    from db_optimizer import config as cfg
    from db_optimizer.schema import create_schema
    from db_optimizer.db import get_connection
    from db_optimizer.blob import (
        factors_to_blob, blob_to_factors, bulk_blobs_to_stacked,
        compute_support_hash, compute_support_sig,
    )
    from db_optimizer.tensor import fitness_frobenius
    from db_optimizer.gpu_eval import gpu_batch_fitness, gpu_batch_fitness_stacked, gpu_minimax_refine, gpu_minimax_refine_stacked
    from db_optimizer.cpu_refine import cpu_minimax_refine
    from db_optimizer.generator import generate_batch
    from db_optimizer.selector import (
        select_parents, select_global_elites, fetch_pending, update_fitness_batch,
        fetch_refine_candidates, update_refined, update_minimax_batch,
        prune_duplicates, archive_to_shadow, pending_count, island_counts,
        insert_candidates, log_event,
        promote_best, demote_random, fetch_shadow_pool,
    )
    import numpy as np

    # Apply config overrides
    ov = config_overrides or {}
    requested_islands = int(ov.get("n_islands", cfg.N_ISLANDS))
    ISLAND_LAYOUT = _select_active_island_layout(cfg, requested_islands)
    N_ISLANDS = len(ISLAND_LAYOUT)
    GENERATION_BATCH_SIZE = int(ov.get("batch_size", cfg.GENERATION_BATCH_SIZE))
    MAX_PENDING = int(ov.get("max_pending", cfg.MAX_PENDING))
    ELITE_K = int(ov.get("elite_k", cfg.ELITE_K))
    TOURNAMENT_SIZE = int(ov.get("tournament_size", cfg.TOURNAMENT_SIZE))
    TIER1_SURVIVOR_THRESHOLD = float(ov.get("tier1_threshold", cfg.TIER1_SURVIVOR_THRESHOLD))
    TIER2_REFINE_THRESHOLD = float(ov.get("tier2_threshold", cfg.TIER2_REFINE_THRESHOLD))
    MAX_GPU_BATCH_SIZE = int(ov.get("max_gpu_batch", cfg.MAX_GPU_BATCH_SIZE))
    HIT_THRESHOLD = float(ov.get("hit_threshold", cfg.HIT_THRESHOLD))
    CPU_MINIMAX_SWEEPS = int(ov.get("minimax_sweeps", cfg.CPU_MINIMAX_SWEEPS))
    CPU_MINIMAX_FINE_RANGE = float(ov.get("minimax_fine_range", cfg.CPU_MINIMAX_FINE_RANGE))
    P_MUTATE_COEFF = float(ov.get("p_mutate_coeff", cfg.P_MUTATE_COEFF))
    P_MUTATE_GAUSSIAN = float(ov.get("p_mutate_gaussian", cfg.P_MUTATE_GAUSSIAN))
    P_CROSSOVER = float(ov.get("p_crossover", cfg.P_CROSSOVER))
    P_SHADOW_REINJECT = float(ov.get("p_shadow_reinject", cfg.P_SHADOW_REINJECT))
    N_CPU_WORKERS = int(ov.get("cpu_workers", max(1, (os.cpu_count() or 4) - 2)))
    N_REFINE_PER_CYCLE = int(ov.get("refine_batch", max(N_CPU_WORKERS * 2, 10)))
    GPU_MINIMAX_SWEEPS = int(ov.get("gpu_minimax_sweeps", cfg.GPU_MINIMAX_SWEEPS))
    GPU_MINIMAX_BATCH = int(ov.get("gpu_minimax_batch", cfg.GPU_MINIMAX_BATCH_SIZE))
    GPU_MINIMAX_N_TRIALS = int(ov.get("gpu_minimax_n_trials", cfg.GPU_MINIMAX_N_TRIALS))
    GPU_MINIMAX_FINE_RANGE = float(ov.get("gpu_minimax_fine_range", cfg.GPU_MINIMAX_FINE_RANGE))
    START_MODE = str(ov.get("mode", "resume")).lower()
    SEED_PATH = ov.get("seed_path")

    ISLAND_CONFIG = cfg.ISLAND_CONFIG
    ROLE_LEVELS = len({meta["role"] for meta in ISLAND_LAYOUT})

    global _runtime_island_layout
    _runtime_island_layout = ISLAND_LAYOUT

    # ── Shared state across pipeline threads ──
    _gen_counter = [0]
    _gen_lock = threading.Lock()
    _stop_event = threading.Event()
    _pipeline_errors: list[str] = []

    # ── In-memory queue: GPU → CPU (Tier 2 → Tier 3) ──
    _gpu_out_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=10_000)

    # ── DB buffer: single-writer thread drains a queue ──
    _db_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=256)
    _work_available = threading.Event()  # signal GPU/CPU when new rows exist

    class _DbResult:
        """Holder for synchronous DB query results."""
        __slots__ = ("value", "ready")
        def __init__(self):
            self.value = None
            self.ready = threading.Event()
        def get(self, timeout: float = 30.0):
            self.ready.wait(timeout)
            return self.value

    def _db_submit(op: str, args: tuple = (), *, sync: bool = False):
        """Submit a DB operation. If sync=True, blocks until result is ready."""
        if sync:
            res = _DbResult()
            _db_queue.put((op, args, res))
            return res.get()
        else:
            _db_queue.put((op, args, None))

    def _db_writer_loop(conn):
        """Single thread that owns the DB connection and drains the queue."""
        try:
            while not _stop_event.is_set() or not _db_queue.empty():
                try:
                    op, args, result = _db_queue.get(timeout=0.05)
                except _queue_mod.Empty:
                    continue

                _set_activity("db_io", "working", op)
                try:
                    val = None
                    if op == "insert_candidates":
                        insert_candidates(conn, args[0])
                        _work_available.set()
                    elif op == "fetch_pending":
                        val = fetch_pending(conn, limit=args[0])
                    elif op == "update_fitness_batch":
                        update_fitness_batch(conn, args[0], args[1], tier=args[2])
                        _work_available.set()
                    elif op == "update_minimax_batch":
                        update_minimax_batch(conn, args[0], args[1], args[2])
                        _work_available.set()
                    elif op == "fetch_refine_candidates":
                        val = fetch_refine_candidates(conn, args[0])
                    elif op == "update_refined":
                        update_refined(conn, args[0], args[1], args[2], args[3])
                    elif op == "archive_to_shadow":
                        archive_to_shadow(conn, args[0])
                    elif op == "prune_duplicates":
                        val = prune_duplicates(conn)
                    elif op == "pending_count":
                        val = pending_count(conn)
                    elif op == "island_counts":
                        val = island_counts(conn, args[0])
                    elif op == "select_parents":
                        val = select_parents(conn, args[0], k=args[1], tournament_size=args[2])
                    elif op == "select_global_elites":
                        val = select_global_elites(conn, args[0])
                    elif op == "log_event":
                        log_event(conn, args[0], generation=args[1],
                                  detail_json=args[2], wall_seconds=args[3])
                    elif op == "promote_best":
                        val = promote_best(conn, args[0], args[1], k=args[2], dst_capacity=args[3])
                    elif op == "demote_random":
                        val = demote_random(conn, args[0], args[1], k=args[2], dst_capacity=args[3])
                    elif op == "monitor_query":
                        best_row = conn.execute(
                            "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates WHERE fitness_fp32 IS NOT NULL"
                        ).fetchone()
                        total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
                        pend = pending_count(conn)
                        island_rows = conn.execute(
                            """SELECT virtual_island as isl, COUNT(*) as cnt,
                                      MIN(COALESCE(fitness_fp64, fitness_fp32)) as best
                               FROM candidates WHERE fitness_fp32 IS NOT NULL
                               GROUP BY virtual_island ORDER BY virtual_island"""
                        ).fetchall()
                        val = (best_row, total, pend, island_rows)
                    if result is not None:
                        result.value = val
                        result.ready.set()
                except Exception as exc:
                    if result is not None:
                        result.value = None
                        result.ready.set()
                    _pipeline_errors.append(f"db_writer({op}): {exc}")
                    import traceback; traceback.print_exc()
                finally:
                    _set_activity("db_io", "idle")
        except Exception as exc:
            _pipeline_errors.append(f"db_writer: {exc}")
            import traceback; traceback.print_exc()
            _stop_event.set()

    def _is_running():
        with _optimizer_lock:
            return _optimizer_status["running"] and not _stop_event.is_set()

    # ── Thread A: Generator (island-aware) ───────────────────────
    def _generator_loop(conn):
        try:
            rng = np.random.default_rng()
            cycle = 0
            last_log = 0.0
            # Own connection for reads — bypasses the DB writer queue entirely
            # (not read_only=True; Windows backslash paths break ?mode=ro URIs)
            gen_read_conn = get_connection(db_path, check_same_thread=False)

            # Shadow pool + plateau detection state
            shadow_pool = []
            shadow_refresh_cycle = 0
            plateau_gens = 0
            last_best_fitness = None

            # ── Adaptive sigma per island ──
            # Tracks per-island best and scales sigma based on stagnation
            island_best_fitness: dict[int, float] = {}
            island_stale_cycles: dict[int, int] = {}
            island_sigma_mult: dict[int, float] = {}  # multiplier on base sigma
            SIGMA_GROW = 1.3       # multiply sigma by this each stale cycle
            SIGMA_DECAY = 0.85     # decay toward 1.0 on improvement
            SIGMA_MAX_MULT = 10.0  # cap: 10x base sigma
            SIGMA_MIN_MULT = 0.5   # floor: half base sigma on rapid improvement

            # ── Operator credit assignment (per island) ──
            # EMA of success rate per operator → adaptive probabilities
            OP_NAMES = ["mutate_coeff", "mutate_gaussian", "crossover", "shadow_reinject", "mutate_support"]
            island_op_credits: dict[int, dict[str, float]] = {}
            OP_EMA_ALPHA = 0.15   # learning rate for credit updates
            OP_MIN_PROB = 0.03    # floor probability for any operator

            # ── Random immigrant injection ──
            IMMIGRANT_PLATEAU_THRESH = 30  # stale cycles before injection
            HYPERMUT_PLATEAU_THRESH = 20   # stale cycles before sigma spike

            while _is_running():
                _set_activity("generator", "checking", "backpressure")
                n_pend = pending_count(gen_read_conn)
                if n_pend >= MAX_PENDING:
                    _set_activity("generator", "idle", "backpressure")
                    now = time.time()
                    if now - last_log >= 5.0:
                        print(f"  [GEN] backpressure: pending={n_pend} >= max={MAX_PENDING}")
                        last_log = now
                    time.sleep(0.05)
                    continue

                t0 = time.time()
                counts = island_counts(gen_read_conn, N_ISLANDS)

                # Refresh shadow pool every 10 cycles
                shadow_refresh_cycle += 1
                if shadow_refresh_cycle >= 10 or not shadow_pool:
                    shadow_refresh_cycle = 0
                    shadow_rows = fetch_shadow_pool(gen_read_conn, limit=200)
                    if shadow_rows:
                        shadow_pool = [blob_to_factors(r["factors_blob"]) for r in shadow_rows]

                # Plateau detection: track if best fitness is stagnant
                cur_best_row = gen_read_conn.execute(
                    "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates WHERE fitness_fp32 IS NOT NULL"
                ).fetchone()
                cur_best = cur_best_row["best"] if cur_best_row else None
                if cur_best is not None and last_best_fitness is not None:
                    if cur_best < last_best_fitness - 1e-12:
                        plateau_gens = 0  # improving
                    else:
                        plateau_gens += 1
                if cur_best is not None:
                    last_best_fitness = cur_best
                # Shadow rate multiplier: 1x normally, ramp to 3x on plateau (after 10 stale gens)
                shadow_boost = min(3.0, 1.0 + max(0, plateau_gens - 10) * 0.2)

                # ── Update per-island adaptive state ──
                for island_meta in ISLAND_LAYOUT:
                    isl = island_meta["id"]
                    row = gen_read_conn.execute(
                        "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates "
                        "WHERE virtual_island = ? AND fitness_fp32 IS NOT NULL",
                        (isl,),
                    ).fetchone()
                    isl_best = row["best"] if row and row["best"] is not None else None
                    if isl_best is None:
                        continue
                    prev = island_best_fitness.get(isl)
                    if prev is not None and isl_best < prev - 1e-12:
                        # Improvement: decay sigma multiplier toward 1.0
                        island_stale_cycles[isl] = 0
                        island_sigma_mult[isl] = max(SIGMA_MIN_MULT,
                                                     island_sigma_mult.get(isl, 1.0) * SIGMA_DECAY)
                        # Credit the operator that produced the improvement
                        best_origin = gen_read_conn.execute(
                            "SELECT origin FROM candidates WHERE virtual_island = ? "
                            "AND COALESCE(fitness_fp64, fitness_fp32) = ? LIMIT 1",
                            (isl, isl_best),
                        ).fetchone()
                        if best_origin:
                            op = best_origin["origin"]
                            credits = island_op_credits.setdefault(isl,
                                {name: 1.0 / len(OP_NAMES) for name in OP_NAMES})
                            for name in OP_NAMES:
                                if name == op:
                                    credits[name] += OP_EMA_ALPHA * (1.0 - credits[name])
                                else:
                                    credits[name] *= (1.0 - OP_EMA_ALPHA)
                    else:
                        stale = island_stale_cycles.get(isl, 0) + 1
                        island_stale_cycles[isl] = stale
                        # Stagnation: grow sigma multiplier
                        if stale >= 3:
                            island_sigma_mult[isl] = min(SIGMA_MAX_MULT,
                                                         island_sigma_mult.get(isl, 1.0) * SIGMA_GROW)
                    island_best_fitness[isl] = isl_best

                # ── Hypermutation: spike sigma on moderate plateau ──
                hypermut_active = (plateau_gens >= HYPERMUT_PLATEAU_THRESH
                                   and plateau_gens % 5 == 0)

                # ── Random immigrant injection on deep plateau ──
                if plateau_gens >= IMMIGRANT_PLATEAU_THRESH and plateau_gens % 10 == 0:
                    n_immigrants = max(N_ISLANDS, GENERATION_BATCH_SIZE // 20)
                    imm_rows = []
                    imm_alloc = _allocate_island_budget(
                        [{"id": m["id"], "room": m["cap"], "weight": m["cap"]}
                         for m in ISLAND_LAYOUT if m["role"] in ("explore", "wide_explore", "balanced")],
                        n_immigrants,
                    )
                    gen = _gen_counter[0]
                    for isl, n_imm in imm_alloc.items():
                        for _ in range(n_imm):
                            a = rng.standard_normal((cfg.RANK, cfg.DIM))
                            b = rng.standard_normal((cfg.RANK, cfg.DIM))
                            g = rng.standard_normal((cfg.RANK, cfg.DIM))
                            blob = factors_to_blob(a, b, g)
                            s_hash = compute_support_hash(a, b, g)
                            s_sig = compute_support_sig(a, b, g)
                            imm_rows.append((blob, s_hash, s_sig, "random_immigrant", isl, gen))
                    if imm_rows:
                        _db_submit("insert_candidates", (imm_rows,))
                        print(f"  [GEN] Injected {len(imm_rows)} random immigrants (plateau={plateau_gens})")

                # Per-island generation using island-specific configs
                _set_activity("generator", "working", "selecting parents")
                parent_rows_by_island = {}
                active_islands = []

                for island_meta in ISLAND_LAYOUT:
                    island = island_meta["id"]
                    role = island_meta["role"]
                    room = max(0, island_meta["cap"] - counts[island])
                    if room <= 0:
                        continue
                    icfg = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])

                    # Select parents — direct read, no queue round-trip
                    rows = select_parents(gen_read_conn, island, ELITE_K, TOURNAMENT_SIZE)
                    if not rows:
                        continue
                    parent_rows_by_island[island] = rows
                    active_islands.append({"id": island, "room": room, "weight": island_meta["cap"]})

                allocations = _allocate_island_budget(active_islands, GENERATION_BATCH_SIZE)

                # Submit per-island batches incrementally so GPU gets work immediately
                # instead of waiting for the entire generation cycle to finish.
                total_inserted = 0
                for island, n_children in allocations.items():
                    island_meta = ISLAND_LAYOUT[island]
                    role = island_meta["role"]
                    icfg = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])
                    parents = [blob_to_factors(row["factors_blob"]) for row in parent_rows_by_island[island]]
                    effective_p_shadow = min(0.40, icfg["p_shadow"] * shadow_boost)

                    # Adaptive sigma: base × island multiplier × hypermutation spike
                    sigma_mult = island_sigma_mult.get(island, 1.0)
                    if hypermut_active:
                        sigma_mult = max(sigma_mult, 5.0)
                    effective_sigma = icfg["sigma"] * sigma_mult

                    # Adaptive operator probabilities from credit assignment
                    credits = island_op_credits.get(island)
                    if credits:
                        total_credit = sum(credits.values())
                        if total_credit > 0:
                            normed = {k: max(OP_MIN_PROB, v / total_credit) for k, v in credits.items()}
                            s = sum(normed.values())
                            eff_p_coeff = normed.get("mutate_coeff", icfg["p_coeff"]) / s
                            eff_p_gauss = normed.get("mutate_gaussian", icfg["p_gaussian"]) / s
                            eff_p_cross = normed.get("crossover", icfg["p_crossover"]) / s
                            # shadow is boosted separately; scale remaining ops
                            remaining = 1.0 - effective_p_shadow
                            op_sum = eff_p_coeff + eff_p_gauss + eff_p_cross
                            if op_sum > 0:
                                eff_p_coeff = eff_p_coeff / op_sum * remaining * 0.85
                                eff_p_gauss = eff_p_gauss / op_sum * remaining * 0.85
                                eff_p_cross = eff_p_cross / op_sum * remaining * 0.85
                            else:
                                eff_p_coeff, eff_p_gauss, eff_p_cross = icfg["p_coeff"], icfg["p_gaussian"], icfg["p_crossover"]
                        else:
                            eff_p_coeff, eff_p_gauss, eff_p_cross = icfg["p_coeff"], icfg["p_gaussian"], icfg["p_crossover"]
                    else:
                        eff_p_coeff, eff_p_gauss, eff_p_cross = icfg["p_coeff"], icfg["p_gaussian"], icfg["p_crossover"]

                    children = generate_batch(
                        parents, n_children, rng,
                        shadow_pool=shadow_pool or None,
                        p_coeff=eff_p_coeff,
                        p_gaussian=eff_p_gauss,
                        p_crossover=eff_p_cross,
                        p_shadow=effective_p_shadow,
                        sigma=effective_sigma,
                    )

                    gen = _gen_counter[0]
                    island_rows = []
                    for alpha, beta, gamma, origin in children:
                        blob = factors_to_blob(alpha, beta, gamma)
                        s_hash = compute_support_hash(alpha, beta, gamma)
                        s_sig = compute_support_sig(alpha, beta, gamma)
                        island_rows.append((blob, s_hash, s_sig, origin, island, gen))

                    if island_rows:
                        _db_submit("insert_candidates", (island_rows,))
                        total_inserted += len(island_rows)

                # Fallback: populate islands with spare capacity using global elites.
                if total_inserted < GENERATION_BATCH_SIZE:
                    remaining_budget = GENERATION_BATCH_SIZE - total_inserted
                    elites = select_global_elites(gen_read_conn, ELITE_K * N_ISLANDS)
                    if elites:
                        parents = [blob_to_factors(r["factors_blob"]) for r in elites]
                        generated_counts = {i: counts[i] for i in range(N_ISLANDS)}
                        for island, added in allocations.items():
                            generated_counts[island] += added
                        fallback_islands = []
                        for island_meta in ISLAND_LAYOUT:
                            island = island_meta["id"]
                            room = max(0, island_meta["cap"] - generated_counts[island])
                            if room > 0:
                                fallback_islands.append({"id": island, "room": room, "weight": island_meta["cap"]})
                        fallback_alloc = _allocate_island_budget(fallback_islands, remaining_budget)
                        gen = _gen_counter[0]
                        for island, n_children in fallback_alloc.items():
                            island_meta = ISLAND_LAYOUT[island]
                            role = island_meta["role"]
                            icfg = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])
                            effective_p_shadow = min(0.40, icfg["p_shadow"] * shadow_boost)
                            fb_sigma_mult = island_sigma_mult.get(island, 1.0)
                            if hypermut_active:
                                fb_sigma_mult = max(fb_sigma_mult, 5.0)
                            fb_sigma = icfg["sigma"] * fb_sigma_mult
                            children = generate_batch(
                                parents, n_children, rng,
                                shadow_pool=shadow_pool or None,
                                p_coeff=icfg["p_coeff"],
                                p_gaussian=icfg["p_gaussian"],
                                p_crossover=icfg["p_crossover"],
                                p_shadow=effective_p_shadow,
                                sigma=fb_sigma,
                            )
                            fb_rows = []
                            for alpha, beta, gamma, origin in children:
                                blob = factors_to_blob(alpha, beta, gamma)
                                s_hash = compute_support_hash(alpha, beta, gamma)
                                s_sig = compute_support_sig(alpha, beta, gamma)
                                fb_rows.append((blob, s_hash, s_sig, origin, island, gen))
                            if fb_rows:
                                _db_submit("insert_candidates", (fb_rows,))
                                total_inserted += len(fb_rows)

                if total_inserted == 0:
                    _set_activity("generator", "idle", "waiting for scored parents")
                    now = time.time()
                    if now - last_log >= 5.0:
                        print("  [GEN] waiting for scored parents")
                        last_log = now
                    time.sleep(0.10)
                    continue

                cycle += 1
                dt = time.time() - t0
                _set_activity("generator", "idle", "", items=total_inserted, dt=dt)
                now = time.time()
                if cycle % 20 == 0 or dt >= 1.0 or (now - last_log) >= 10.0:
                    print(f"  [GEN] cycle {cycle}: {total_inserted} children, pending={n_pend}, {dt:.2f}s")
                    last_log = now

        except Exception as exc:
            _set_activity("generator", "error", str(exc))
            _pipeline_errors.append(f"generator: {exc}")
            import traceback; traceback.print_exc()
            _stop_event.set()

    # ── Thread B: GPU Eval (Tier 1 Screen + Tier 2 Minimax) ─────
    def _gpu_eval_loop(conn):
        try:
            cycle = 0
            last_log = 0.0
            # Own connection for reads — GPU never waits on the DB writer
            gpu_read_conn = get_connection(db_path, check_same_thread=False)
            gpu_in_flight: set[int] = set()  # IDs submitted for write but not yet committed

            # ── Pre-decode thread: overlap CPU decode with GPU compute ──
            # GPU thread pops ready-to-go (ids, alphas, betas, gammas, islands, kind)
            # from this queue while the decoder fetches+decodes the next batch.
            _prefetch_q: _queue_mod.Queue = _queue_mod.Queue(maxsize=2)

            def _prefetch_loop():
                pf_conn = get_connection(db_path, check_same_thread=False)
                pf_in_flight: set[int] = set()
                while not _stop_event.is_set():
                    try:
                        # Try pending (screening) first
                        rows = fetch_pending(pf_conn, limit=MAX_GPU_BATCH_SIZE)
                        current_ids = {r["id"] for r in rows}
                        pf_in_flight &= current_ids
                        fresh = [r for r in rows if r["id"] not in pf_in_flight]

                        if fresh:
                            blobs = [r["factors_blob"] for r in fresh]
                            a, b, g = bulk_blobs_to_stacked(blobs)
                            ids = [r["id"] for r in fresh]
                            islands = [r["virtual_island"] for r in fresh]
                            pf_in_flight.update(ids)
                            _prefetch_q.put(("screen", ids, a, b, g, islands), timeout=1.0)
                            continue

                        # No pending — fetch best unrefined for minimax
                        mm_rows = pf_conn.execute(
                            """SELECT id, factors_blob, virtual_island FROM candidates
                               WHERE fitness_fp32 IS NOT NULL AND tier_reached <= 1
                               ORDER BY fitness_fp32 ASC LIMIT ?""",
                            (GPU_MINIMAX_BATCH,),
                        ).fetchall()

                        if not mm_rows:
                            # Nothing tier<=1 — re-minimax best tier-2 candidates
                            mm_rows = pf_conn.execute(
                                """SELECT id, factors_blob, virtual_island FROM candidates
                                   WHERE fitness_fp32 IS NOT NULL AND tier_reached = 2
                                   ORDER BY COALESCE(minimax_improved, fitness_fp32) ASC LIMIT ?""",
                                (GPU_MINIMAX_BATCH,),
                            ).fetchall()

                        if mm_rows:
                            blobs = [r["factors_blob"] for r in mm_rows]
                            a, b, g = bulk_blobs_to_stacked(blobs)
                            ids = [r["id"] for r in mm_rows]
                            islands = [r["virtual_island"] for r in mm_rows]
                            _prefetch_q.put(("minimax", ids, a, b, g, islands), timeout=1.0)
                        else:
                            time.sleep(0.02)
                    except _queue_mod.Full:
                        pass
                    except Exception:
                        if _stop_event.is_set():
                            break
                        time.sleep(0.05)

            prefetch_thread = threading.Thread(target=_prefetch_loop, name="gpu_prefetch", daemon=True)
            prefetch_thread.start()

            while _is_running():
                # Pop a pre-decoded batch (overlapped with previous GPU work)
                try:
                    kind, ids, alpha_np, beta_np, gamma_np, islands = _prefetch_q.get(timeout=0.05)
                except _queue_mod.Empty:
                    continue

                if kind == "screen":
                    # ── Tier 1: Bulk GPU screen ──
                    t0 = time.time()
                    N = alpha_np.shape[0]
                    _set_activity("gpu_screen", "working", f"evaluating {N} on GPU")

                    # Track in-flight for the main thread's view
                    gpu_in_flight.update(ids)

                    fitness_fp32 = gpu_batch_fitness_stacked(alpha_np, beta_np, gamma_np)
                    _db_submit("update_fitness_batch", (ids, fitness_fp32.tolist(), 1))

                    dt_screen = time.time() - t0
                    best_batch = float(fitness_fp32.min()) if N > 0 else 999
                    _set_activity("gpu_screen", "idle", "", items=N, dt=dt_screen)

                    # Inline tier-2: minimax survivors directly (already decoded)
                    surv_mask = fitness_fp32 < TIER1_SURVIVOR_THRESHOLD
                    n_surv = int(surv_mask.sum())
                    if n_surv > 0:
                        s_alpha = alpha_np[surv_mask]
                        s_beta = beta_np[surv_mask]
                        s_gamma = gamma_np[surv_mask]
                        s_ids = [ids[i] for i in range(N) if surv_mask[i]]
                        s_islands = [islands[i] for i in range(N) if surv_mask[i]]

                        t1 = time.time()
                        _set_activity("gpu_minimax", "working",
                                      f"refining {n_surv} survivors ({GPU_MINIMAX_SWEEPS} sweeps)")

                        minimax_results = gpu_minimax_refine_stacked(
                            s_alpha, s_beta, s_gamma,
                            sweeps=GPU_MINIMAX_SWEEPS,
                            n_trials=GPU_MINIMAX_N_TRIALS,
                            fine_range=GPU_MINIMAX_FINE_RANGE,
                        )

                        mm_ids, mm_blobs, mm_fitness = [], [], []
                        for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                            mm_ids.append(s_ids[i])
                            mm_blobs.append(factors_to_blob(a_r, b_r, g_r))
                            mm_fitness.append(fit)

                        _db_submit("update_minimax_batch", (mm_ids, mm_blobs, mm_fitness))

                        for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                            if fit < TIER2_REFINE_THRESHOLD:
                                try:
                                    _gpu_out_queue.put_nowait(
                                        (s_ids[i], a_r, b_r, g_r, fit, s_islands[i])
                                    )
                                except _queue_mod.Full:
                                    break

                        dt_mm = time.time() - t1
                        best_mm = min(r[3] for r in minimax_results) if minimax_results else 999
                        _set_activity("gpu_minimax", "idle", "", items=n_surv, dt=dt_mm)
                    else:
                        dt_mm = 0.0
                        best_mm = 999

                    now = time.time()
                    if cycle % 5 == 0 or (now - last_log) >= 10.0:
                        print(
                            f"  [GPU] cycle {cycle}: screened {N} (best={best_batch:.6f}), "
                            f"minimax {n_surv}, screen={dt_screen:.2f}s, mm={dt_mm:.2f}s"
                        )
                        last_log = now

                elif kind == "minimax":
                    # ── Fill-in: GPU minimax on best candidates ──
                    N = alpha_np.shape[0]
                    t1 = time.time()
                    _set_activity("gpu_minimax", "working",
                                  f"minimax {N} candidates ({GPU_MINIMAX_SWEEPS} sweeps)")

                    minimax_results = gpu_minimax_refine_stacked(
                        alpha_np, beta_np, gamma_np,
                        sweeps=GPU_MINIMAX_SWEEPS,
                        n_trials=GPU_MINIMAX_N_TRIALS,
                        fine_range=GPU_MINIMAX_FINE_RANGE,
                    )

                    mm_ids, mm_blobs, mm_fitness = [], [], []
                    for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                        mm_ids.append(ids[i])
                        mm_blobs.append(factors_to_blob(a_r, b_r, g_r))
                        mm_fitness.append(fit)

                    _db_submit("update_minimax_batch", (mm_ids, mm_blobs, mm_fitness))

                    for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                        if fit < TIER2_REFINE_THRESHOLD:
                            try:
                                _gpu_out_queue.put_nowait(
                                    (ids[i], a_r, b_r, g_r, fit, islands[i])
                                )
                            except _queue_mod.Full:
                                break

                    dt_mm = time.time() - t1
                    best_mm = min(r[3] for r in minimax_results) if minimax_results else 999
                    _set_activity("gpu_minimax", "idle", "", items=N, dt=dt_mm)

                    now = time.time()
                    if cycle % 5 == 0 or dt_mm >= 2.0 or (now - last_log) >= 10.0:
                        print(f"  [GPU] cycle {cycle}: minimax {N}, best={best_mm:.6f}, {dt_mm:.2f}s")
                        last_log = now

                cycle += 1

        except Exception as exc:
            _set_activity("gpu_screen", "error", str(exc))
            _set_activity("gpu_minimax", "error", str(exc))
            _pipeline_errors.append(f"gpu_eval: {exc}")
            import traceback; traceback.print_exc()
            _stop_event.set()

    # ── Thread C: CPU Refiner with ProcessPool (adaptive + queue-fed) ─
    def _cpu_refine_loop(conn):
        try:
            cycle = 0
            n_total_improved = 0
            best_fit = 999.0
            last_log = 0.0
            # Own connection for reads — CPU never waits on the DB writer
            cpu_read_conn = get_connection(db_path, check_same_thread=False)
            with ProcessPoolExecutor(max_workers=N_CPU_WORKERS) as pool:
                active_futures = {}   # fut -> (row_id, island)
                in_flight_ids = set()
                prune_counter = 0

                while _is_running():
                    # --- Feed workers from gpu_out_queue (fast path) ---
                    capacity = max(0, N_CPU_WORKERS * 2 - len(active_futures))
                    queue_fed = 0
                    while capacity > 0:
                        try:
                            row_id, alpha, beta, gamma, _, island = _gpu_out_queue.get_nowait()
                        except _queue_mod.Empty:
                            break
                        if row_id in in_flight_ids:
                            continue

                        # Island-aware sweep count
                        role = ISLAND_LAYOUT[island]["role"] if island < len(ISLAND_LAYOUT) else "balanced"
                        sweeps = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])["cpu_sweeps"]

                        fut = pool.submit(
                            cpu_minimax_refine,
                            alpha, beta, gamma,
                            sweeps=sweeps,
                            fine_range=CPU_MINIMAX_FINE_RANGE,
                        )
                        active_futures[fut] = (row_id, island)
                        in_flight_ids.add(row_id)
                        capacity -= 1
                        queue_fed += 1

                    # --- Fallback: feed from DB if queue was empty ---
                    if queue_fed == 0 and capacity > 0:
                        _set_activity("cpu_refine", "working", "fetching from DB")
                        refine_rows = fetch_refine_candidates(cpu_read_conn, TIER2_REFINE_THRESHOLD)
                        if not refine_rows:
                            # Relaxed: refine the best evaluated-but-unrefined candidates
                            # regardless of threshold — keeps CPU busy at all times
                            refine_rows = cpu_read_conn.execute(
                                """SELECT id, factors_blob, fitness_fp32, virtual_island
                                   FROM candidates
                                   WHERE fitness_fp32 IS NOT NULL AND tier_reached < 3
                                   ORDER BY fitness_fp32 ASC
                                   LIMIT 100""",
                            ).fetchall()
                        if refine_rows:
                            new_rows = [r for r in refine_rows if r["id"] not in in_flight_ids]
                            batch = new_rows[:capacity]
                            for row in batch:
                                alpha, beta, gamma = blob_to_factors(row["factors_blob"])
                                island = row["virtual_island"]
                                role = ISLAND_LAYOUT[island]["role"] if island < len(ISLAND_LAYOUT) else "balanced"
                                sweeps = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])["cpu_sweeps"]

                                fut = pool.submit(
                                    cpu_minimax_refine,
                                    alpha, beta, gamma,
                                    sweeps=sweeps,
                                    fine_range=CPU_MINIMAX_FINE_RANGE,
                                )
                                active_futures[fut] = (row["id"], island)
                                in_flight_ids.add(row["id"])

                    if active_futures:
                        _set_activity("cpu_refine", "working",
                                      f"refining {len(active_futures)} ({N_CPU_WORKERS}w)")

                    # --- Collect completed futures (non-blocking) ---
                    done = [f for f in active_futures if f.done()]
                    n_batch_improved = 0
                    for fut in done:
                        row_id, island = active_futures.pop(fut)
                        in_flight_ids.discard(row_id)
                        try:
                            a_r, b_r, g_r, fit64 = fut.result(timeout=1)
                            fro = fitness_frobenius(a_r, b_r, g_r)
                            new_blob = factors_to_blob(a_r, b_r, g_r)
                            _db_submit("update_refined", (row_id, new_blob, fit64, fro))
                            _db_submit("archive_to_shadow", (row_id,))
                            n_batch_improved += 1
                            n_total_improved += 1
                            best_fit = min(best_fit, fit64)
                        except Exception:
                            pass

                    # --- Periodic pruning ---
                    if done:
                        prune_counter += len(done)
                    if prune_counter >= N_CPU_WORKERS * 5:
                        _db_submit("prune_duplicates")
                        prune_counter = 0

                    # --- Logging ---
                    if done:
                        cycle += 1
                        _set_activity("cpu_refine", "working",
                                      f"refining {len(active_futures)} ({N_CPU_WORKERS}w)",
                                      items=len(done), dt=0.1)
                        now = time.time()
                        if cycle % 10 == 0 or (now - last_log) >= 10.0:
                            print(f"  [CPU] cycle {cycle}: +{n_batch_improved} improved, "
                                  f"active={len(active_futures)}, total={n_total_improved}, "
                                  f"best={best_fit:.10f}, {N_CPU_WORKERS}w, "
                                  f"queue={_gpu_out_queue.qsize()}")
                            last_log = now

                    # --- Yield ---
                    if not active_futures:
                        _set_activity("cpu_refine", "idle", "waiting for work")
                        time.sleep(0.02)
                    elif not done:
                        time.sleep(0.005)

        except Exception as exc:
            _set_activity("cpu_refine", "error", str(exc))
            _pipeline_errors.append(f"cpu_refine: {exc}")
            import traceback; traceback.print_exc()
            _stop_event.set()

    # ── Main: seed, launch pipeline, monitor + migration ────────
    try:
        conn = get_connection(db_path, check_same_thread=False)
        create_schema(conn)

        # Seed / bootstrap depending on mode
        existing = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]

        if START_MODE == "resume":
            # Keep existing DB as-is; only seed if DB is completely empty
            if existing == 0:
                from db_optimizer.orchestrator import seed_from_json
                seed_paths = sorted(seed_dir.glob("optimized_*.json"))
                if seed_paths:
                    n = seed_from_json(conn, seed_paths, n_islands=N_ISLANDS)
                    with _optimizer_lock:
                        _optimizer_status["total_candidates"] = n
                    log_event(conn, "seed", generation=0, detail_json=json.dumps({"count": n, "mode": "resume_fallback"}))
                    print(f"[seed] Resumed empty DB - seeded {n} from JSON")

        elif START_MODE == "seed":
            # Wipe DB, then seed from JSON
            from db_optimizer.orchestrator import seed_from_json
            if SEED_PATH:
                seed_target = Path(SEED_PATH)
                if not seed_target.is_absolute():
                    seed_target = (seed_dir / seed_target).resolve()
                if seed_target.is_dir():
                    seed_paths = sorted(seed_target.glob("optimized_*.json"))
                elif seed_target.is_file():
                    seed_paths = [seed_target]
                else:
                    raise FileNotFoundError(f"Seed path not found: {seed_target}")
            else:
                seed_paths = sorted(seed_dir.glob("optimized_*.json"))
            if seed_paths:
                n = seed_from_json(conn, seed_paths, n_islands=N_ISLANDS)
                with _optimizer_lock:
                    _optimizer_status["total_candidates"] = n
                log_event(conn, "seed", generation=0, detail_json=json.dumps({"count": n, "mode": "seed"}))
                print(f"[seed] Seeded {n} from JSON into fresh DB")
            else:
                print("[seed] WARNING: No JSON files found for seeding - falling back to random bootstrap")
                START_MODE = "fresh"

        if START_MODE == "fresh":
            existing = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
            if existing == 0:
                bootstrap_n = int(ov.get("bootstrap_size", max(256, ELITE_K * N_ISLANDS, min(GENERATION_BATCH_SIZE, 1024))))
                bootstrap_rng = np.random.default_rng()
                bootstrap_rows = []
                bootstrap_alloc = _allocate_island_budget(
                    [{"id": meta["id"], "room": meta["cap"], "weight": meta["cap"]} for meta in ISLAND_LAYOUT],
                    bootstrap_n,
                )
                for island, n_rows in bootstrap_alloc.items():
                    for _ in range(n_rows):
                        alpha = bootstrap_rng.standard_normal((cfg.RANK, cfg.DIM))
                        beta = bootstrap_rng.standard_normal((cfg.RANK, cfg.DIM))
                        gamma = bootstrap_rng.standard_normal((cfg.RANK, cfg.DIM))
                        blob = factors_to_blob(alpha, beta, gamma)
                        s_hash = compute_support_hash(alpha, beta, gamma)
                        s_sig = compute_support_sig(alpha, beta, gamma)
                        bootstrap_rows.append((blob, s_hash, s_sig, "fresh_bootstrap", island, 0))

                insert_candidates(conn, bootstrap_rows)
                with _optimizer_lock:
                    _optimizer_status["total_candidates"] = bootstrap_n
                    _optimizer_status["pending"] = bootstrap_n
                log_event(conn, "bootstrap", generation=0,
                          detail_json=json.dumps({"count": bootstrap_n, "mode": "fresh"}))
                print(f"[seed] Bootstrapped {bootstrap_n} random candidates for fresh mode")

        # Launch DB writer thread (owns the connection)
        db_writer = threading.Thread(target=_db_writer_loop, args=(conn,), name="db_writer", daemon=True)
        db_writer.start()

        # Launch pipeline threads
        threads = [
            threading.Thread(target=_generator_loop, args=(conn,), name="generator", daemon=True),
            threading.Thread(target=_gpu_eval_loop, args=(conn,), name="gpu_eval", daemon=True),
            threading.Thread(target=_cpu_refine_loop, args=(conn,), name="cpu_refine", daemon=True),
        ]
        for t in threads:
            t.start()

        print(f"[pipeline] Started: db_writer + generator + GPU eval (Tier1+Tier2) + CPU refiner ({N_CPU_WORKERS} workers)")
        print(f"[pipeline] Batch={GENERATION_BATCH_SIZE}, MaxPending={MAX_PENDING}, GPU batch={MAX_GPU_BATCH_SIZE}")
        print(f"[pipeline] GPU minimax: sweeps={GPU_MINIMAX_SWEEPS}, batch={GPU_MINIMAX_BATCH}, trials={GPU_MINIMAX_N_TRIALS}, fine={GPU_MINIMAX_FINE_RANGE}")
        print(f"[pipeline] Tier1 thresh={TIER1_SURVIVOR_THRESHOLD}, Tier2 thresh={TIER2_REFINE_THRESHOLD}")
        size_levels = sorted({meta["size_exp"] for meta in ISLAND_LAYOUT})
        print(f"[pipeline] Islands: {N_ISLANDS} active across {ROLE_LEVELS} roles and size exponents {size_levels}")

        # Monitor loop: update status, migration, check for stop
        mon_read_conn = get_connection(db_path, check_same_thread=False)
        last_gen_time = time.time()
        migration_counter = 0
        while _is_running():
            time.sleep(2.0)
            _set_activity("monitor", "working", "polling DB stats")

            # Direct reads on monitor's own connection — no DB writer queue
            try:
                best_row = mon_read_conn.execute(
                    "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates WHERE fitness_fp32 IS NOT NULL"
                ).fetchone()
                total = mon_read_conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
                pend = pending_count(mon_read_conn)
                island_rows = mon_read_conn.execute(
                    """SELECT virtual_island as isl, COUNT(*) as cnt,
                              MIN(COALESCE(fitness_fp64, fitness_fp32)) as best
                       FROM candidates WHERE fitness_fp32 IS NOT NULL
                       GROUP BY virtual_island ORDER BY virtual_island"""
                ).fetchall()
            except Exception:
                continue
            result = (best_row, total, pend, island_rows)

            has_progress = total > 0 or pend > 0 or any(r["best"] is not None for r in island_rows)
            if not has_progress:
                with _optimizer_lock:
                    _optimizer_status["best_fitness"] = None
                    _optimizer_status["total_candidates"] = 0
                    _optimizer_status["pending"] = 0
                _set_activity("monitor", "idle")
                continue

            # Advance generation counter
            with _gen_lock:
                _gen_counter[0] += 1
                gen = _gen_counter[0]

            with _optimizer_lock:
                _optimizer_status["best_fitness"] = best_row["best"] if best_row else None
                _optimizer_status["total_candidates"] = total
                _optimizer_status["pending"] = pend
                _optimizer_status["generation"] = gen

            wall = time.time() - last_gen_time
            last_gen_time = time.time()

            # Per-island bests for charting
            island_bests = {}
            for r in island_rows:
                if r["best"] is not None:
                    island_bests[str(r["isl"])] = r["best"]

            # ── Migration (every 5 generations) ──
            migration_counter += 1
            if migration_counter >= 5:
                migration_counter = 0
                import random as _rng

                # 1) Adjacent-tier exchange within each role (structured)
                islands_by_role: dict[str, list[dict]] = {}
                for meta in ISLAND_LAYOUT:
                    islands_by_role.setdefault(meta["role"], []).append(meta)
                for metas in islands_by_role.values():
                    metas.sort(key=lambda item: item["size_exp"])
                    for small_meta, large_meta in zip(metas, metas[1:]):
                        k = max(1, small_meta["cap"] // 8)
                        _db_submit("promote_best", (large_meta["id"], small_meta["id"], k, small_meta["cap"]))
                        _db_submit("promote_best", (small_meta["id"], large_meta["id"], k, large_meta["cap"]))
                        _db_submit("demote_random", (small_meta["id"], large_meta["id"], k, large_meta["cap"]))

                # 2) Cross-role random pairings (any island ↔ any island)
                n_cross = max(2, len(ISLAND_LAYOUT) // 4)
                for _ in range(n_cross):
                    a, b = _rng.sample(ISLAND_LAYOUT, 2)
                    k = max(1, min(a["cap"], b["cap"]) // 16)
                    _db_submit("promote_best", (a["id"], b["id"], k, b["cap"]))
                    _db_submit("promote_best", (b["id"], a["id"], k, a["cap"]))

            _db_submit("log_event", ("gen_complete", gen,
                                     json.dumps({
                                         "best": best_row["best"], "total": total, "pending": pend,
                                         "wall": round(wall, 2),
                                         "island_bests": island_bests,
                                     }), wall))
            _set_activity("monitor", "idle")

            best_val = best_row["best"] if best_row else None
            best_str = f"{best_val:.10f}" if best_val is not None else "N/A"
            isl_parts = [f"I{r['isl']}:{r['cnt']}({r['best']:.6f})" for r in island_rows if r["cnt"] > 0]
            isl_str = " ".join(isl_parts) if isl_parts else "no islands"
            qsize = _gpu_out_queue.qsize()
            if gen % 5 == 0:
                print(f"[MON] gen={gen} total={total} pending={pend} best={best_str} queue={qsize} | {isl_str}")

            if best_row["best"] is not None and best_row["best"] < HIT_THRESHOLD:
                print(f"[pipeline] HIT! fitness = {best_row['best']}")
                break

            if gen >= max_generations:
                break

        # Signal all threads to stop
        _stop_event.set()
        for t in threads:
            t.join(timeout=10)
        db_writer.join(timeout=5)

        if _pipeline_errors:
            err_msg = "; ".join(_pipeline_errors)
            print(f"[pipeline] Errors: {err_msg}")
            with _optimizer_lock:
                _optimizer_status["error"] = err_msg

        conn.close()

    except Exception as exc:
        with _optimizer_lock:
            _optimizer_status["error"] = str(exc)
            _optimizer_status["running"] = False
        raise
    finally:
        with _optimizer_lock:
            _optimizer_status["running"] = False


def _get_db_stats(db_path: Path) -> dict:
    """Read stats directly from the DB for the API."""
    from db_optimizer.db import get_connection
    from db_optimizer import config as cfg
    from db_optimizer.selector import island_stats
    try:
        island_layout = _runtime_island_layout or cfg.ISLAND_LAYOUT
        n_islands = len(island_layout)
        conn = get_connection(db_path, read_only=True)
        total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
        pending = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='pending'").fetchone()["cnt"]
        screened = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='screened' AND tier_reached=1").fetchone()["cnt"]
        minimax_count = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE tier_reached=2").fetchone()["cnt"]
        refined = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='refined'").fetchone()["cnt"]
        shadow_count = conn.execute("SELECT COUNT(*) as cnt FROM shadow").fetchone()["cnt"]

        best = conn.execute(
            "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates WHERE fitness_fp32 IS NOT NULL"
        ).fetchone()["best"]

        # History from run_log (includes per-island bests)
        history = []
        rows = conn.execute(
            "SELECT generation, detail_json, wall_seconds, timestamp FROM run_log WHERE event='gen_complete' ORDER BY generation ASC LIMIT 2000"
        ).fetchall()
        for row in rows:
            d = json.loads(row["detail_json"]) if row["detail_json"] else {}
            history.append({
                "generation": row["generation"],
                "best_fitness": d.get("best"),
                "total": d.get("total"),
                "wall_seconds": row["wall_seconds"],
                "island_bests": d.get("island_bests", {}),
            })

        # Top 20 candidates
        top = []
        top_rows = conn.execute(
            """SELECT id, COALESCE(fitness_fp64, fitness_fp32) as fit, fro_residual, support_sig,
                      origin, virtual_island, generation, tier_reached
               FROM candidates WHERE fitness_fp32 IS NOT NULL
               ORDER BY COALESCE(fitness_fp64, fitness_fp32) ASC LIMIT 20"""
        ).fetchall()
        for r in top_rows:
            top.append({
                "id": r["id"], "fitness": r["fit"], "fro": r["fro_residual"],
                "sig": r["support_sig"], "origin": r["origin"],
                "island": r["virtual_island"], "gen": r["generation"], "tier": r["tier_reached"],
            })

        islands = island_stats(conn, n_islands)
        for island in islands:
            meta = island_layout[island["island"]]
            island["role"] = meta["role"]
            island["size_exp"] = meta["size_exp"]
            island["cap"] = meta["cap"]
        conn.close()

        return {
            "total": total, "pending": pending, "screened": screened,
            "minimax": minimax_count, "refined": refined,
            "shadow": shadow_count, "best": best, "history": history, "top": top, "islands": islands,
        }
    except Exception as exc:
        return {"error": str(exc)}


class DashboardHandler(BaseHTTPRequestHandler):
    db_path: Path = Path("K:/ade3x3_optimizer/candidates.db")

    def log_message(self, format, *args):
        pass  # suppress noisy per-request logging

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, filename: str):
        path = STATIC_DIR / filename
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/index.html"):
            self._serve_static("dashboard.html")
        elif path == "/dashboard.css":
            self._serve_static("dashboard.css")
        elif path == "/dashboard.js":
            self._serve_static("dashboard.js")
        elif path == "/api/status":
            with _optimizer_lock:
                status = dict(_optimizer_status)
            self._json_response(status)
        elif path == "/api/stats":
            stats = _get_db_stats(self.db_path)
            self._json_response(stats)
        elif path == "/api/activity":
            self._json_response(_get_activity())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/start":
            content_len = int(self.headers.get("Content-Length", 0))
            config_overrides = {}
            if content_len > 0:
                raw = self.rfile.read(content_len)
                try:
                    config_overrides = json.loads(raw)
                except json.JSONDecodeError:
                    pass

            with _optimizer_lock:
                if _optimizer_status["running"]:
                    self._json_response({"error": "Already running"}, 409)
                    return

            mode = str(config_overrides.get("mode", "resume")).lower()
            backup_requested = bool(config_overrides.get("backup_db", False))
            backup_path = None
            try:
                if mode in ("fresh", "seed"):
                    backup_path = _prepare_fresh_run(self.db_path, backup_requested)
            except Exception as exc:
                self._json_response({"error": f"Failed to prepare fresh run: {exc}"}, 500)
                return

            with _optimizer_lock:
                _optimizer_status["running"] = True
                _optimizer_status["generation"] = 0
                _optimizer_status["best_fitness"] = None
                _optimizer_status["total_candidates"] = 0
                _optimizer_status["pending"] = 0
                _optimizer_status["error"] = None
                _optimizer_status["start_time"] = time.time()

            max_gens = int(config_overrides.get("max_generations", 10_000))

            global _optimizer_thread
            _optimizer_thread = threading.Thread(
                target=_run_optimizer,
                args=(self.db_path, REPO_ROOT, max_gens, config_overrides),
                daemon=True,
            )
            _optimizer_thread.start()
            self._json_response({
                "ok": True,
                "mode": mode,
                "backup_path": str(backup_path) if backup_path else None,
            })

        elif path == "/api/stop":
            with _optimizer_lock:
                _optimizer_status["running"] = False
            self._json_response({"ok": True})

        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="db_optimizer dashboard + runner")
    parser.add_argument("--db", type=Path, default=Path("K:/ade3x3_optimizer/candidates.db"))
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--autostart", action="store_true", help="Start optimizer immediately on server launch")
    args = parser.parse_args()

    DashboardHandler.db_path = args.db
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] http://{args.host}:{args.port}")
    print(f"[dashboard] DB: {args.db}")

    if args.autostart:
        with _optimizer_lock:
            _optimizer_status["running"] = True
            _optimizer_status["start_time"] = time.time()
        global _optimizer_thread
        _optimizer_thread = threading.Thread(
            target=_run_optimizer,
            args=(args.db, REPO_ROOT, 100_000),
            daemon=True,
        )
        _optimizer_thread.start()
        print("[dashboard] Optimizer auto-started")
    else:
        print("[dashboard] Optimizer idle; start from the UI or pass --autostart")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down...")
        with _optimizer_lock:
            _optimizer_status["running"] = False
        server.shutdown()


if __name__ == "__main__":
    main()
