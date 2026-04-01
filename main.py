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


def _reset_db_files(db_path: Path) -> None:
    """Remove the SQLite database and sidecar files for a truly fresh run."""
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()


def _prepare_fresh_run(db_path: Path, make_backup: bool) -> Path | None:
    """Optionally back up the current DB, then delete it for a fresh run."""
    backup_path = _backup_db_file(db_path) if make_backup else None
    _reset_db_files(db_path)
    return backup_path


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
        factors_to_blob, blob_to_factors,
        compute_support_hash, compute_support_sig,
    )
    from db_optimizer.tensor import fitness_frobenius
    from db_optimizer.gpu_eval import gpu_batch_fitness, gpu_minimax_refine
    from db_optimizer.cpu_refine import cpu_minimax_refine
    from db_optimizer.generator import generate_batch
    from db_optimizer.selector import (
        select_parents, select_global_elites, fetch_pending, update_fitness_batch,
        fetch_refine_candidates, update_refined, update_minimax_batch,
        prune_duplicates, archive_to_shadow, pending_count,
        insert_candidates, log_event,
        promote_best, demote_random,
    )
    import numpy as np

    # Apply config overrides
    ov = config_overrides or {}
    N_ISLANDS = int(ov.get("n_islands", cfg.N_ISLANDS))
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
    START_MODE = str(ov.get("mode", "seed")).lower()
    SEED_PATH = ov.get("seed_path")

    ISLAND_ROLES = cfg.ISLAND_ROLES
    ISLAND_CONFIG = cfg.ISLAND_CONFIG

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
                    elif op == "select_parents":
                        val = select_parents(conn, args[0], k=args[1], tournament_size=args[2])
                    elif op == "select_global_elites":
                        val = select_global_elites(conn, args[0])
                    elif op == "log_event":
                        log_event(conn, args[0], generation=args[1],
                                  detail_json=args[2], wall_seconds=args[3])
                    elif op == "promote_best":
                        val = promote_best(conn, args[0], args[1], k=args[2])
                    elif op == "demote_random":
                        val = demote_random(conn, args[0], args[1], k=args[2])
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

            while _is_running():
                _set_activity("generator", "checking", "backpressure")
                n_pend = _db_submit("pending_count", (), sync=True)
                if n_pend is None:
                    n_pend = 0
                if n_pend >= MAX_PENDING:
                    _set_activity("generator", "idle", "backpressure")
                    _work_available.wait(timeout=0.1)
                    _work_available.clear()
                    continue

                t0 = time.time()

                # Per-island generation using island-specific configs
                _set_activity("generator", "working", "selecting parents")
                all_to_insert = []

                for island in range(N_ISLANDS):
                    role = ISLAND_ROLES[island] if island < len(ISLAND_ROLES) else "balanced"
                    icfg = ISLAND_CONFIG.get(role, ISLAND_CONFIG["balanced"])

                    # Select parents for this island
                    rows = _db_submit("select_parents", (island, ELITE_K, TOURNAMENT_SIZE), sync=True)
                    if not rows:
                        continue

                    parents = []
                    for row in rows:
                        a, b, g = blob_to_factors(row["factors_blob"])
                        parents.append((a, b, g))

                    if not parents:
                        continue

                    # Batch size proportional to island size
                    n_children = max(32, GENERATION_BATCH_SIZE // N_ISLANDS)

                    children = generate_batch(
                        parents, n_children, rng,
                        p_coeff=icfg["p_coeff"],
                        p_gaussian=icfg["p_gaussian"],
                        p_crossover=icfg["p_crossover"],
                        p_shadow=icfg["p_shadow"],
                        sigma=icfg["sigma"],
                    )

                    gen = _gen_counter[0]
                    for alpha, beta, gamma, origin in children:
                        blob = factors_to_blob(alpha, beta, gamma)
                        s_hash = compute_support_hash(alpha, beta, gamma)
                        s_sig = compute_support_sig(alpha, beta, gamma)
                        all_to_insert.append((blob, s_hash, s_sig, origin, island, gen))

                # Fallback: if no island had parents, use global elites
                if not all_to_insert:
                    elites = _db_submit("select_global_elites", (ELITE_K * N_ISLANDS,), sync=True)
                    if elites:
                        parents = [blob_to_factors(r["factors_blob"]) for r in elites]
                        children = generate_batch(
                            parents, GENERATION_BATCH_SIZE, rng,
                            p_coeff=P_MUTATE_COEFF,
                            p_gaussian=P_MUTATE_GAUSSIAN,
                            p_crossover=P_CROSSOVER,
                            p_shadow=P_SHADOW_REINJECT,
                        )
                        gen = _gen_counter[0]
                        for alpha, beta, gamma, origin in children:
                            blob = factors_to_blob(alpha, beta, gamma)
                            s_hash = compute_support_hash(alpha, beta, gamma)
                            s_sig = compute_support_sig(alpha, beta, gamma)
                            v_island = hash(s_hash) % N_ISLANDS
                            all_to_insert.append((blob, s_hash, s_sig, origin, v_island, gen))

                if all_to_insert:
                    _db_submit("insert_candidates", (all_to_insert,))

                cycle += 1
                dt = time.time() - t0
                _set_activity("generator", "idle", "", items=len(all_to_insert), dt=dt)
                if cycle % 2 == 0:
                    print(f"  [GEN] cycle {cycle}: {len(all_to_insert)} children, pending={n_pend}, {dt:.2f}s")

        except Exception as exc:
            _set_activity("generator", "error", str(exc))
            _pipeline_errors.append(f"generator: {exc}")
            import traceback; traceback.print_exc()
            _stop_event.set()

    # ── Thread B: GPU Eval (Tier 1 Screen + Tier 2 Minimax) ─────
    def _gpu_eval_loop(conn):
        try:
            cycle = 0
            while _is_running():
                # ── Tier 1: Bulk GPU screen ──
                _set_activity("gpu_screen", "working", "fetching pending")
                pending_rows = _db_submit("fetch_pending", (MAX_GPU_BATCH_SIZE,), sync=True)
                if not pending_rows:
                    _set_activity("gpu_screen", "idle", "waiting for work")
                    _work_available.wait(timeout=0.1)
                    _work_available.clear()
                    continue

                t0 = time.time()
                _set_activity("gpu_screen", "working", f"decoding {len(pending_rows)}")
                factors_list = [blob_to_factors(row["factors_blob"]) for row in pending_rows]
                ids = [row["id"] for row in pending_rows]
                islands = [row["virtual_island"] for row in pending_rows]

                _set_activity("gpu_screen", "working", f"evaluating {len(ids)} on GPU")
                fitness_fp32 = gpu_batch_fitness(factors_list)

                # Update DB (screened, tier 1)
                _db_submit("update_fitness_batch", (ids, fitness_fp32.tolist(), 1))

                dt_screen = time.time() - t0
                best_batch = float(fitness_fp32.min()) if len(fitness_fp32) > 0 else 999
                _set_activity("gpu_screen", "idle", "", items=len(ids), dt=dt_screen)

                # ── Tier 2: GPU Minimax on survivors ──
                survivors = [
                    (factors_list[i], ids[i], islands[i])
                    for i in range(len(fitness_fp32))
                    if fitness_fp32[i] < TIER1_SURVIVOR_THRESHOLD
                ]

                if survivors:
                    t1 = time.time()
                    # Process in sub-batches for VRAM management
                    surv_factors = [s[0] for s in survivors]
                    surv_ids = [s[1] for s in survivors]
                    surv_islands = [s[2] for s in survivors]

                    _set_activity("gpu_minimax", "working",
                                  f"refining {len(surv_factors)} survivors ({GPU_MINIMAX_SWEEPS} sweeps)")

                    minimax_results = []
                    for batch_start in range(0, len(surv_factors), GPU_MINIMAX_BATCH):
                        batch_end = min(batch_start + GPU_MINIMAX_BATCH, len(surv_factors))
                        batch_factors = surv_factors[batch_start:batch_end]
                        batch_refined = gpu_minimax_refine(
                            batch_factors,
                            sweeps=GPU_MINIMAX_SWEEPS,
                            n_trials=cfg.GPU_MINIMAX_N_TRIALS,
                            fine_range=cfg.GPU_MINIMAX_FINE_RANGE,
                        )
                        minimax_results.extend(batch_refined)

                    # Update DB with minimax results
                    mm_ids = []
                    mm_blobs = []
                    mm_fitness = []
                    for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                        mm_ids.append(surv_ids[i])
                        mm_blobs.append(factors_to_blob(a_r, b_r, g_r))
                        mm_fitness.append(fit)

                    _db_submit("update_minimax_batch", (mm_ids, mm_blobs, mm_fitness))

                    # Push to gpu_out_queue for immediate CPU pickup
                    for i, (a_r, b_r, g_r, fit) in enumerate(minimax_results):
                        if fit < TIER2_REFINE_THRESHOLD:
                            try:
                                _gpu_out_queue.put_nowait(
                                    (surv_ids[i], a_r, b_r, g_r, fit, surv_islands[i])
                                )
                            except _queue_mod.Full:
                                break  # backpressure — queue full, skip remaining

                    dt_mm = time.time() - t1
                    best_mm = min(r[3] for r in minimax_results) if minimax_results else 999
                    _set_activity("gpu_minimax", "idle", "", items=len(minimax_results), dt=dt_mm)
                    print(f"  [GPU] cycle {cycle}: screened {len(ids)} (best={best_batch:.6f}), "
                          f"minimax {len(minimax_results)} (best={best_mm:.6f}), "
                          f"screen={dt_screen:.2f}s, minimax={dt_mm:.2f}s")
                else:
                    _set_activity("gpu_minimax", "idle")
                    print(f"  [GPU] cycle {cycle}: screened {len(ids)}, best={best_batch:.6f}, "
                          f"0 survivors, {dt_screen:.2f}s")

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
                        role = ISLAND_ROLES[island] if island < len(ISLAND_ROLES) else "balanced"
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
                        refine_rows = _db_submit("fetch_refine_candidates", (TIER2_REFINE_THRESHOLD,), sync=True)
                        if refine_rows:
                            new_rows = [r for r in refine_rows if r["id"] not in in_flight_ids]
                            batch = new_rows[:capacity]
                            for row in batch:
                                alpha, beta, gamma = blob_to_factors(row["factors_blob"])
                                island = row["virtual_island"]
                                role = ISLAND_ROLES[island] if island < len(ISLAND_ROLES) else "balanced"
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
                        if cycle % 3 == 0:
                            print(f"  [CPU] cycle {cycle}: +{n_batch_improved} improved, "
                                  f"active={len(active_futures)}, total={n_total_improved}, "
                                  f"best={best_fit:.10f}, {N_CPU_WORKERS}w, "
                                  f"queue={_gpu_out_queue.qsize()}")

                    # --- Yield ---
                    if not active_futures:
                        _set_activity("cpu_refine", "idle", "waiting for work")
                        _work_available.wait(timeout=0.1)
                        _work_available.clear()
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

        # Seed if empty and the run mode requests seeding
        existing = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
        if existing == 0 and START_MODE != "fresh":
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
                n = seed_from_json(conn, seed_paths)
                with _optimizer_lock:
                    _optimizer_status["total_candidates"] = n
                log_event(conn, "seed", generation=0, detail_json=json.dumps({"count": n}))

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
        print(f"[pipeline] GPU minimax: sweeps={GPU_MINIMAX_SWEEPS}, batch={GPU_MINIMAX_BATCH}, trials={cfg.GPU_MINIMAX_N_TRIALS}")
        print(f"[pipeline] Tier1 thresh={TIER1_SURVIVOR_THRESHOLD}, Tier2 thresh={TIER2_REFINE_THRESHOLD}")
        print(f"[pipeline] Islands: {N_ISLANDS} ({', '.join(ISLAND_ROLES[:N_ISLANDS])})")

        # Monitor loop: update status, migration, check for stop
        last_gen_time = time.time()
        migration_counter = 0
        while _is_running():
            time.sleep(2.0)
            _set_activity("monitor", "working", "polling DB stats")

            result = _db_submit("monitor_query", (), sync=True)
            if result is None:
                continue
            best_row, total, pend, island_rows = result

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

            # ── Power-law migration (every 5 generations) ──
            migration_counter += 1
            if migration_counter >= 5:
                migration_counter = 0
                for i in range(N_ISLANDS - 1):
                    # Promote: best from island i+1 → island i
                    _db_submit("promote_best", (i + 1, i, 1))
                    # Demote: random from island i → island i+1
                    _db_submit("demote_random", (i, i + 1, 1))

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
    from db_optimizer.config import N_ISLANDS
    from db_optimizer.selector import island_stats
    try:
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

        islands = island_stats(conn, N_ISLANDS)
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

            mode = str(config_overrides.get("mode", "seed")).lower()
            backup_requested = bool(config_overrides.get("backup_db", False))
            backup_path = None
            try:
                if mode == "fresh":
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
    parser.add_argument("--no-autostart", action="store_true", help="Don't auto-start optimizer")
    args = parser.parse_args()

    DashboardHandler.db_path = args.db
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] http://{args.host}:{args.port}")
    print(f"[dashboard] DB: {args.db}")

    if not args.no_autostart:
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

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down...")
        with _optimizer_lock:
            _optimizer_status["running"] = False
        server.shutdown()


if __name__ == "__main__":
    main()
