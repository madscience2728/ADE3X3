"""
Slime-mold CP-rank search for T<3,3,3>.

Seeds: three near-associative Bini tensors loaded from DB.
Pool: POOL_SIZE GREEN walker threads, always kept full.
Each walker takes random-direction steps and probes CP rank at ranks 19-22.
Starvation -> RED -> immediate pool refill from best YELLOW.
Branching at richer nodes amplifies promising regions.

Run with --dry-run for a 60-second test (POOL_SIZE=3, STARVATION_LIMIT=5).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# BLAS thread throttle — must be set BEFORE numpy/scipy are imported so the
# BLAS backend reads the caps at load time.  Without this, each of the N
# walker threads asks OpenBLAS for its default thread count (up to MAX_THREADS
# per call), producing N × MAX_THREADS BLAS threads on only cpu_count cores.
# With 82 walkers × 2 OpenBLAS threads = 164 threads on ~8–16 cores the first
# step takes 4–5 minutes instead of seconds.  Setting everything to "1" here
# lets the walker threads themselves act as the parallelism unit (1 thread ≡
# 1 core) with no internal contention.
# ---------------------------------------------------------------------------
for _blas_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Validation pipeline — imported lazily inside _validate_candidate so worker
# processes (which never call it) don't pay the import cost.
# Top-level names are resolved at function call time via local imports.

# ---------------------------------------------------------------------------
# Hyper-parameters (overridden by --dry-run)
# ---------------------------------------------------------------------------
# Default to cpu_count so we never spawn more walker threads than cores.
# Each walker = 1 thread = ~1 core (BLAS is single-threaded per the env caps
# set above), so more than cpu_count walkers gives no extra throughput and
# only increases context-switch overhead.
POOL_SIZE        = max(4, (os.cpu_count() or 8))
STARVATION_LIMIT = 20       # steps with residual > SIGNAL_THRESHOLD
SIGNAL_THRESHOLD = 0.05
BRANCHING_BASE   = 0.05
BRANCHING_RICH   = 0.30     # residual < 0.02
BRANCHING_HOT    = 0.60     # residual < 0.01
# Quick-mode override (--quick-mode flag): raise branching probability so
# walkers fork children every few steps and tree depth grows quickly.
# SIGNAL_THRESHOLD is intentionally left at its normal value so walkers still
# starve → go RED → trigger pool refill, exercising the full lifecycle.
QUICK_BRANCHING_BASE = 0.50   # 50% branch chance per step → deep tree in seconds
EPSILON_FRAC     = 0.02     # step = EPSILON_FRAC * ||C||
RANK_RANGE       = list(range(9, 20))  # [9, 10, ..., 19]
# Restarts scale with pool size: enough to cover basins at each step without
# making the tree grow too slowly.  Rule of thumb: 4 × POOL_SIZE gives ~20-30
# restarts per rank (RANK_RANGE has 11 entries), balancing coverage vs. speed.
# Override with --restarts if you want to push harder on fewer nodes.
RESTARTS_PER_STEP = max(80, 4 * POOL_SIZE)  # typically 80–160 on 8–24 core machines
RANK_TOL         = 1e-7
STATE_WRITE_INTERVAL = 5    # seconds between JSON writes
ALS_ITERS        = 80        # ALS iterations per restart (fast per-step evaluation)
LBFGS_MAXITER    = 300       # L-BFGS-B iterations per restart


def _get_k_checkpoint() -> int:
    """Return the optional ALS checkpoint step count for dataset collection."""
    try:
        from ml_seed.schema import K_CHECKPOINT
        return int(K_CHECKPOINT)
    except Exception:
        return 0

BINI_PREFIXES = ["53a3875f", "d84ea9b1", "70a95852"]
DB_PATH = ROOT / "ade" / "db" / "algebras.sqlite"
OUT_DIR = Path(__file__).parent
STATE_PATH  = OUT_DIR / "bini_state.json"
SOLVED_PATH = OUT_DIR / "SOLVED_UVW.npz"
GRAPH_DB_PATH = OUT_DIR / "slime_graph.db"


@dataclass(frozen=True)
class SeedRecord:
    name: str
    fp_hash: str
    source_name: str
    C: np.ndarray

# ---------------------------------------------------------------------------
# Build T<3,3,3>
# ---------------------------------------------------------------------------
def build_T333() -> np.ndarray:
    """Return the 9x9x9 matrix multiplication tensor for 3x3 matrices."""
    n = 3
    dim = n * n
    T = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                alpha = i * n + j   # e_{ij}
                beta  = j * n + k   # e_{jk}
                gamma = i * n + k   # e_{ik}
                T[alpha, beta, gamma] = 1.0
    return T

T333 = build_T333()

# ---------------------------------------------------------------------------
# CP rank evaluation (called within each walker thread, no sub-parallelism)
# ---------------------------------------------------------------------------
def _reconstruct(U, V, W):
    return np.einsum("ra,rb,rc->abc", U, V, W)

def _loss_and_grad(params, T, r):
    d1, d2, d3 = T.shape
    U = params[:r * d1].reshape(r, d1)
    V = params[r * d1:r * (d1 + d2)].reshape(r, d2)
    W = params[r * (d1 + d2):].reshape(r, d3)
    R = _reconstruct(U, V, W) - T
    loss = 0.5 * float(np.dot(R.ravel(), R.ravel()))
    dU = np.einsum("abc,rb,rc->ra", R, V, W)
    dV = np.einsum("abc,ra,rc->rb", R, U, W)
    dW = np.einsum("abc,ra,rb->rc", R, U, V)
    return loss, np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])

def _als_step(T, U, V, W):
    r = U.shape[0]
    d1, d2, d3 = T.shape
    reg = 1e-10 * np.eye(r)

    def _update(T_mode, F1, F2):
        KR = np.einsum("ra,rb->rab", F1, F2).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T_mode @ KR.T
        return np.linalg.lstsq(gram, rhs.T, rcond=None)[0]

    T1 = T.reshape(d1, d2 * d3)
    U = _update(T1, V, W)
    T2 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
    V = _update(T2, U, W)
    T3 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)
    W = _update(T3, U, V)
    return U, V, W

def _one_restart(
    T_init: np.ndarray,
    rank: int,
    seed: int,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    T_target: Optional[np.ndarray] = None,
    k_checkpoint: int = 0,
):
    """Single ALS + L-BFGS-B restart.

    Returns (residual, U, V, W, res_at_k).

    Two-phase:
      Phase 1 (ALS)      : warm up U,V,W against T_init (e.g. C_new).
                           Uses C's structure to find a good basin.
      Phase 2 (L-BFGS-B): refine against T_target (defaults to T_init).
                           Residual is always measured against T_target.

    When T_target=T333 the search asks: "given C as initialisation hint,
    how well can rank-R factors approximate T_ref?"
    """
    if T_target is None:
        T_target = T_init
    d1, d2, d3 = T_init.shape
    rng = np.random.default_rng(seed)
    scale = max((float(np.linalg.norm(T_init)) / rank) ** (1/3), 0.1)
    U = rng.standard_normal((rank, d1)) * scale
    V = rng.standard_normal((rank, d2)) * scale
    W = rng.standard_normal((rank, d3)) * scale

    # Phase 1: ALS on T_init — exploits C's structure as a basin navigator
    res_at_k = None
    for step in range(als_iters):
        U, V, W = _als_step(T_init, U, V, W)
        if k_checkpoint > 0 and step + 1 == k_checkpoint:
            approx = _reconstruct(U, V, W)
            res_at_k = float(np.linalg.norm(T_target - approx))

    # Phase 2: L-BFGS-B on T_target — measures quality against T_ref
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    try:
        opt = minimize(
            _loss_and_grad, x0, args=(T_target, rank),
            method="L-BFGS-B", jac=True,
            options={"maxiter": lbfgs_maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        res = float(np.sqrt(max(2.0 * opt.fun, 0.0)))
        params = opt.x
    except Exception:
        res = float(np.linalg.norm(T_target - _reconstruct(U, V, W)))
        params = x0

    d1t, d2t, d3t = T_target.shape
    U_ = params[:rank*d1t].reshape(rank, d1t)
    V_ = params[rank*d1t:rank*(d1t+d2t)].reshape(rank, d2t)
    W_ = params[rank*(d1t+d2t):].reshape(rank, d3t)
    return res, U_, V_, W_, res_at_k

def compute_algebra_rank(
    C: np.ndarray,
    ranks: list[int] = None,
    n_restarts: int = None,
    tol: float = RANK_TOL,
) -> tuple[float, int, Optional[tuple]]:
    """
    Probe C at each rank in ranks, using n_restarts total (divided across ranks).
    Returns (best_residual, best_rank, factors_or_None).
    Uses current global RANK_RANGE, RESTARTS_PER_STEP, ALS_ITERS, LBFGS_MAXITER.
    """
    if ranks is None:
        ranks = RANK_RANGE
    if n_restarts is None:
        n_restarts = RESTARTS_PER_STEP
    best_res = np.inf
    best_rank = -1
    best_factors = None
    restarts_each = max(1, n_restarts // len(ranks))

    for rank in ranks:
        rng_base = int(time.time() * 1000) % (2**31)
        for i in range(restarts_each):
            res, U, V, W, _ = _one_restart(C, rank, seed=rng_base + i,
                                           als_iters=ALS_ITERS, lbfgs_maxiter=LBFGS_MAXITER)
            if res < best_res:
                best_res = res
                best_rank = rank
                best_factors = (U, V, W)
            if best_res < tol:
                return best_res, best_rank, best_factors

    return best_res, best_rank, best_factors

# ---------------------------------------------------------------------------
# Utility: comm_defect and assoc_defect
# ---------------------------------------------------------------------------
def _comm_defect(C: np.ndarray) -> float:
    norm = np.linalg.norm(C)
    if norm < 1e-14:
        return 0.0
    return float(np.linalg.norm(C - C.transpose(0, 2, 1)) / norm)

def _assoc_defect(C: np.ndarray) -> float:
    norm = np.linalg.norm(C)
    if norm < 1e-14:
        return 0.0
    left  = np.einsum("abe,ecd->abcd", C, C)
    right = np.einsum("bce,aed->abcd", C, C)
    return float(np.linalg.norm(left - right) / (norm ** 2))

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
@dataclass
class Node:
    id: str
    label: str
    parent_id: Optional[str]
    status: str           # SEED | GREEN | YELLOW | RED
    depth: int
    direction: Optional[np.ndarray]  # unit vector shape (9,9,9)
    best_residual: float
    best_rank: int
    steps: int
    steps_since_signal: int
    cd: float
    ad: float
    children: list = field(default_factory=list)
    edge_type: str = "fork"  # edge from parent: "walk" or "fork"
    seed_name: str = ""
    fp_hash: Optional[str] = None
    source_name: Optional[str] = None
    C: Optional[np.ndarray] = field(default=None, repr=False)  # current tensor
    best_factors: Optional[tuple] = field(default=None, repr=False)
    epsilon_scale: float = 1.0          # adaptive step multiplier [0.1, 10.0]
    recent_residuals: list = field(default_factory=list)  # last 5 residuals

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "parent": self.parent_id,
            "status": self.status,
            "depth": self.depth,
            "best_residual": self.best_residual if math.isfinite(self.best_residual) else 999.0,
            "best_rank": self.best_rank,
            "steps": self.steps,
            "cd": round(self.cd, 5),
            "ad": round(self.ad, 5),
            "edge_type": self.edge_type,
            "seed_name": self.seed_name,
            "fp_hash": self.fp_hash,
            "source_name": self.source_name,
        }

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_lock       = threading.Lock()
_nodes: dict[str, Node] = {}       # id -> Node
_green_ids: list[str]   = []       # active GREEN node IDs
_stop_event = threading.Event()
_solved     = False
_solved_count = 0          # how many candidates passed RANK_TOL (kept running)
_t_start    = time.time()
_step_counter = 0
_global_best_residual = math.inf   # best residual seen across all nodes ever
_GLOBAL_BEST_FORK_THRESHOLD = 0.05 # fork burst when global best drops below this
_GLOBAL_BEST_FORK_N = 2            # number of children to spawn on a new global best
_ELITE_SIZE = 10                   # top-N nodes immune to starvation
_immune_node_ids: set[str] = set() # IDs of the current elite (guarded by _lock)
_last_status_bus_write = 0.0

def _get_stats() -> dict:
    with _lock:
        greens  = sum(1 for n in _nodes.values() if n.status == "GREEN")
        yellows = sum(1 for n in _nodes.values() if n.status == "YELLOW")
        reds    = sum(1 for n in _nodes.values() if n.status == "RED")
        all_res = [n.best_residual for n in _nodes.values() if math.isfinite(n.best_residual)]
        best_res = min(all_res) if all_res else 999.0
        best_rank_nodes = [n for n in _nodes.values() if n.best_residual == best_res]
        best_tested_rank = best_rank_nodes[0].best_rank if best_rank_nodes else -1
        return {
            "total_nodes": len(_nodes),
            "green_count": greens,
            "yellow_count": yellows,
            "red_count": reds,
            "best_residual_global": round(best_res, 8),
            "best_rank_global": best_tested_rank if _solved else -1,
            "best_tested_rank_global": best_tested_rank,
            "steps_total": _step_counter,
            "elapsed_seconds": round(time.time() - _t_start, 1),
            "solved": _solved,
            "solved_count": _solved_count,
        }


def _write_slime_status(force: bool = False) -> None:
    global _last_status_bus_write
    now = time.time()
    if not force and now - _last_status_bus_write < 5.0:
        return
    _last_status_bus_write = now
    try:
        from ml_seed.stage3.status_bus import write_section

        stats = _get_stats()
        write_section(
            "slime",
            {
                "running": not _stop_event.is_set(),
                "active_walkers": stats["green_count"],
                "total_nodes": stats["total_nodes"],
                "yellow_nodes": stats["yellow_count"],
                "red_nodes": stats["red_count"],
                "best_residual_this_session": (
                    stats["best_residual_global"]
                    if stats["best_residual_global"] < 999.0
                    else None
                ),
                "best_rank_this_session": (
                    stats["best_tested_rank_global"]
                    if stats["best_tested_rank_global"] >= 0
                    else None
                ),
                "total_steps_this_session": stats["steps_total"],
            },
        )
    except Exception:
        pass

def _write_state():
    with _lock:
        nodes_list = [n.to_dict() for n in _nodes.values()]
        edges = []
        for n in _nodes.values():
            if n.parent_id:
                edges.append({"source": n.parent_id, "target": n.id, "type": n.edge_type})
    stats = _get_stats()
    state = {"nodes": nodes_list, "edges": edges, "stats": stats}
    payload = json.dumps(state, indent=2)
    last_error: Exception | None = None
    for attempt in range(5):
        tmp = STATE_PATH.with_name(
            f"{STATE_PATH.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(STATE_PATH)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise
    if last_error is not None:
        return

# ---------------------------------------------------------------------------
# Candidate validation pipeline
# ---------------------------------------------------------------------------

_CHECKPOINTS_DIR  = OUT_DIR / "results" / "checkpoints"
_DISCOVERIES_PATH = OUT_DIR / "discoveries.jsonl"
_discoveries_lock = threading.Lock()


def _validate_candidate(
    node_id: str,
    C: np.ndarray,
    rank: int,
    res: float,
    factors: Optional[tuple],
) -> None:
    """Full validation of a low-residual candidate.  Runs in a daemon thread.

    Steps:
      1. Proximity check vs T333 (is this genuinely a matmul algorithm?)
      2. Build AlgebraState and run Tier 1 invariants.
      3. Run Tier 2 invariants + fingerprint (skip for n>8).
      4. Isomorphism check against all n=9 algebras in the main DB.
      5. Save checkpoint with CRT naming: novel_{N}_{seed_name}_d{depth}.npz
      6. If novel: insert into the main DB and append to discoveries.jsonl.
    """
    global _solved_count

    # Lazy imports — only the main process validation thread needs these.
    try:
        from ade.core.algebra_state import AlgebraState
        from ade.invariants.tier1 import tier1 as _tier1
        from ade.invariants.tier2 import update_fingerprint
        from iso_module.isomorphism import are_isomorphic
        from ade.db.algebra_db import AlgebraDB
    except ImportError as exc:
        print(f"  [VALIDATE] Import error — validation skipped: {exc}", flush=True)
        return

    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print(f"CANDIDATE FOUND  node={node_id[:8]}  rank={rank}  residual={res:.6e}", flush=True)
    print(sep, flush=True)

    # Gather node metadata for naming
    with _lock:
        node = _nodes.get(node_id)
        seed_name = (node.seed_name or "slime") if node else "slime"
        depth     = node.depth if node else 0
        parent_name = (_nodes[node.parent_id].seed_name
                       if node and node.parent_id and node.parent_id in _nodes
                       else "") or seed_name
    with _lock:
        idx = _solved_count

    novel_id = f"novel_{idx}_{seed_name}_d{depth}"

    # 1. Proximity to T333 -------------------------------------------------
    t333_dist = float(np.linalg.norm(C - T333))
    t333_dist_rel = t333_dist / float(np.linalg.norm(T333))
    print(f"  [VAL] ||C - T333|| = {t333_dist:.4e}  (relative: {t333_dist_rel:.4e})", flush=True)
    is_near_matmul = t333_dist_rel < 0.10   # within 10% of T333
    print(f"  [VAL] Near matmul tensor: {is_near_matmul}", flush=True)

    # 2. AlgebraState + Tier 1 ---------------------------------------------
    A = AlgebraState(9, "R", novel_id)
    A.set_C_dense(C)
    try:
        t1 = _tier1(A)
        print(f"  [VAL] Tier1: dim_center={t1['dim_center']}  "
              f"killing={t1['killing_signature']}  "
              f"nil={t1['nilpotency_class']}  "
              f"assoc={t1['assoc_defect_norm']:.4e}", flush=True)
    except Exception as exc:
        t1 = {"error": str(exc)}
        print(f"  [VAL] Tier 1 failed: {exc}", flush=True)

    # 3. Tier 2 ------------------------------------------------------------
    t2: dict
    try:
        t2 = update_fingerprint(A, t1=t1 if "error" not in t1 else None)
        fp = t2.get("fingerprint")
        print(f"  [VAL] Tier2: fingerprint={fp}  dim_Der={t2.get('dim_Der')}  "
              f"dim_H2={t2.get('dim_H2')}  rigid={t2.get('is_rigid')}", flush=True)
    except NotImplementedError:
        fp = None
        t2 = {"error": "n=9 > 8, H2 skipped"}
        print("  [VAL] Tier 2 H2 skipped (n=9 > limit)", flush=True)
    except Exception as exc:
        fp = None
        t2 = {"error": str(exc)}
        print(f"  [VAL] Tier 2 failed: {exc}", flush=True)

    # Build flag_set (needed for both checkpoint and DB insert)
    flag_set = {
        "is_near_matmul": is_near_matmul,
        "low_assoc_defect": (
            t1.get("assoc_defect_norm", 999.0) < 1e-6
            if "error" not in t1 else None
        ),
        "is_rigid": t2.get("is_rigid"),
    }

    # 4. Save checkpoint in CRT naming convention --------------------------
    # novel_{N}_{seed_name}_d{depth}.npz  — matches search_crt/results/checkpoints/
    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id  = novel_id.replace("/", "_").replace("\\", "_")
    ckpt_path = _CHECKPOINTS_DIR / f"{safe_id}.npz"
    np.savez_compressed(
        str(ckpt_path),
        C=C,
        flags=np.array(json.dumps(flag_set)),
        fingerprint=np.array(json.dumps(list(fp) if fp is not None else [])),
        seed_name=np.array(seed_name),
        depth=np.array(depth),
        rank=np.array(rank),
        residual=np.array(res),
    )
    print(f"  [VAL] Checkpoint saved -> {ckpt_path.name}", flush=True)

    # Also keep SOLVED_UVW naming for backward compat with external tools.
    if factors is not None:
        U, V, W = factors
        if idx <= 1:
            npz_path = SOLVED_PATH
        else:
            npz_path = SOLVED_PATH.with_stem(f"SOLVED_UVW_{idx}")
        np.savez(str(npz_path), U=U, V=V, W=W, C=C,
                 rank=np.array(rank), residual=np.array(res))
        print(f"  [VAL] Factors saved   -> {npz_path.name}", flush=True)

    # 5. Isomorphism check against known DB --------------------------------
    known_matches: list[dict] = []
    is_novel = True
    db = None
    try:
        db = AlgebraDB(str(DB_PATH))
        known_states = db.load_all_as_states()
        n9_states = {name: st for name, st in known_states.items() if st.n == 9}
        print(f"  [VAL] Checking vs {len(n9_states)} known n=9 algebras ...", flush=True)
        for name, known_A in n9_states.items():
            result = are_isomorphic(A, known_A)
            is_iso = result.get("isomorphic", False)
            if is_iso:
                is_novel = False
                known_matches.append({"name": name, "isomorphic": True,
                                       "reason": result.get("reason", "")})
                print(f"  [VAL] MATCH: isomorphic to known algebra '{name}'", flush=True)
            else:
                stage = result.get("stage", "?")
                if stage != 1:
                    print(f"  [VAL] not iso to '{name}' (stage {stage})", flush=True)
        if is_novel:
            print("  [VAL] NOVEL — no isomorphism match found!", flush=True)
    except Exception as exc:
        print(f"  [VAL] Isomorphism check error: {exc}", flush=True)
        is_novel = False   # conservative: don't insert if we couldn't check

    # 6. Insert novel finds into the main DB --------------------------------
    fp_hash_inserted: Optional[str] = None
    if is_novel and db is not None:
        try:
            fp_hash_inserted = db.insert(
                name=novel_id,
                A=A,
                fp=fp,
                flag_set=flag_set,
                generation=0,
                is_known=False,
                seed_name=parent_name,
                score=float(1.0 - res),
            )
            print(f"  [VAL] Inserted into DB: fp_hash={fp_hash_inserted}", flush=True)
        except Exception as exc:
            print(f"  [VAL] DB insert failed: {exc}", flush=True)

    # Append to JSONL log --------------------------------------------------
    factors_ser = None
    if factors is not None:
        U, V, W = factors
        factors_ser = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    record = {
        "timestamp": time.time(),
        "novel_id": novel_id,
        "node_id": node_id,
        "rank": rank,
        "residual": res,
        "t333_dist": t333_dist,
        "t333_dist_rel": t333_dist_rel,
        "is_near_matmul": is_near_matmul,
        "is_novel": is_novel,
        "fp_hash": fp_hash_inserted,
        "checkpoint": ckpt_path.name,
        "tier1": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in t1.items() if not isinstance(v, np.ndarray) or v.size < 100},
        "tier2": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in t2.items() if not isinstance(v, np.ndarray) or v.size < 100},
        "known_matches": known_matches,
        "factors": factors_ser,
    }
    with _discoveries_lock:
        with open(str(_DISCOVERIES_PATH), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    print(f"  [VAL] novel={is_novel}  near_matmul={is_near_matmul}  "
          f"known_matches={len(known_matches)}", flush=True)
    print(sep + "\n", flush=True)

# ---------------------------------------------------------------------------
# Graph DB persistence — saves the full tensor state so the graph can be
# resumed after a crash or power loss.
# ---------------------------------------------------------------------------

def _arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _blob_to_arr(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))

def _init_graph_db() -> None:
    """Create the slime_graph DB (idempotent)."""
    con = sqlite3.connect(str(GRAPH_DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            parent_id TEXT,
            status TEXT,
            depth INTEGER,
            best_residual REAL,
            best_rank INTEGER,
            steps INTEGER,
            steps_since_signal INTEGER,
            cd REAL,
            ad REAL,
            edge_type TEXT,
            seed_name TEXT,
            fp_hash TEXT,
            source_name TEXT,
            C_blob BLOB,
            direction_blob BLOB,
            best_u_blob BLOB,
            best_v_blob BLOB,
            best_w_blob BLOB,
            updated_at REAL
        )
    """)
    # Migrate existing DBs that predate the best_factors columns.
    existing = {row[1] for row in con.execute("PRAGMA table_info(nodes)").fetchall()}
    for col in ("best_u_blob", "best_v_blob", "best_w_blob"):
        if col not in existing:
            con.execute(f"ALTER TABLE nodes ADD COLUMN {col} BLOB")
    con.commit()
    con.close()

def _save_graph_to_db() -> None:
    """Upsert every node (including tensor C, direction, and best U/V/W factors) to slime_graph.db."""
    with _lock:
        snapshot = list(_nodes.values())
    now = time.time()
    con = sqlite3.connect(str(GRAPH_DB_PATH))
    for node in snapshot:
        c_blob = _arr_to_blob(node.C) if node.C is not None else None
        d_blob = _arr_to_blob(node.direction) if node.direction is not None else None
        u_blob = v_blob = w_blob = None
        if node.best_factors is not None:
            U, V, W = node.best_factors
            u_blob = _arr_to_blob(U)
            v_blob = _arr_to_blob(V)
            w_blob = _arr_to_blob(W)
        best_res = node.best_residual if math.isfinite(node.best_residual) else 999.0
        con.execute("""
            INSERT OR REPLACE INTO nodes
            (id, label, parent_id, status, depth, best_residual, best_rank,
             steps, steps_since_signal, cd, ad, edge_type, seed_name,
             fp_hash, source_name, C_blob, direction_blob,
             best_u_blob, best_v_blob, best_w_blob, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            node.id, node.label, node.parent_id, node.status, node.depth,
            best_res, node.best_rank, node.steps, node.steps_since_signal,
            node.cd, node.ad, node.edge_type, node.seed_name,
            node.fp_hash, node.source_name, c_blob, d_blob,
            u_blob, v_blob, w_blob, now,
        ))
    con.commit()
    con.close()

def _load_graph_from_db() -> bool:
    """Load the saved graph from slime_graph.db. Returns True if anything was loaded."""
    if not GRAPH_DB_PATH.exists():
        return False
    con = sqlite3.connect(str(GRAPH_DB_PATH))
    rows = con.execute("""
        SELECT id, label, parent_id, status, depth, best_residual, best_rank,
               steps, steps_since_signal, cd, ad, edge_type, seed_name,
               fp_hash, source_name, C_blob, direction_blob,
               best_u_blob, best_v_blob, best_w_blob
        FROM nodes
    """).fetchall()
    con.close()
    if not rows:
        return False

    print(f"[RESUME] Loading {len(rows)} nodes from {GRAPH_DB_PATH.name}", flush=True)
    loaded: dict[str, Node] = {}
    for row in rows:
        (nid, label, parent_id, status, depth, best_res, best_rank, steps,
         steps_since_signal, cd, ad, edge_type, seed_name, fp_hash, source_name,
         c_blob, d_blob, u_blob, v_blob, w_blob) = row

        C = _blob_to_arr(c_blob) if c_blob else None
        direction = _blob_to_arr(d_blob) if d_blob else None
        best_factors = None
        if u_blob and v_blob and w_blob:
            best_factors = (_blob_to_arr(u_blob), _blob_to_arr(v_blob), _blob_to_arr(w_blob))
        # Downgrade any in-flight GREEN to YELLOW so they're safely re-spawned
        if status == "GREEN":
            status = "YELLOW"

        node = Node(
            id=nid, label=label, parent_id=parent_id, status=status,
            depth=depth, direction=direction,
            best_residual=best_res if best_res != 999.0 else np.inf,
            best_rank=best_rank, steps=steps, steps_since_signal=steps_since_signal,
            cd=cd, ad=ad, children=[], edge_type=edge_type,
            seed_name=seed_name, fp_hash=fp_hash, source_name=source_name,
            C=C, best_factors=best_factors,
        )
        loaded[nid] = node

    # Rebuild children lists from parent_id
    for node in loaded.values():
        if node.parent_id and node.parent_id in loaded:
            loaded[node.parent_id].children.append(node.id)

    with _lock:
        _nodes.update(loaded)

    greens  = sum(1 for n in loaded.values() if n.status == "GREEN")
    yellows = sum(1 for n in loaded.values() if n.status == "YELLOW")
    seeds   = sum(1 for n in loaded.values() if n.status == "SEED")
    reds    = sum(1 for n in loaded.values() if n.status == "RED")
    print(
        f"[RESUME] Restored {len(loaded)} nodes: "
        f"SEED={seeds} GREEN={greens} YELLOW={yellows} RED={reds} "
        f"(GREEN downgraded to YELLOW)",
        flush=True,
    )
    return True

# ---------------------------------------------------------------------------
# Load seeds from DB
# ---------------------------------------------------------------------------
def load_bini_seeds() -> list[SeedRecord]:
    con = sqlite3.connect(str(DB_PATH))
    seeds = []
    for prefix in BINI_PREFIXES:
        row = con.execute(
            "SELECT fp_hash, name, C_blob FROM algebras WHERE fp_hash LIKE ?",
            (prefix + "%",)
        ).fetchone()
        if row is None or row[2] is None:
            raise RuntimeError(f"Bini seed {prefix}... not found in DB")
        fp_hash, name, c_blob = row
        source_row = con.execute(
            "SELECT discovered_from FROM provenance WHERE fp_hash=? ORDER BY rowid ASC LIMIT 1",
            (fp_hash,),
        ).fetchone()
        source_name = source_row[0] if source_row and source_row[0] else "M(3,R)+perturbation"
        C = np.load(io.BytesIO(c_blob))
        seeds.append(SeedRecord(name=name, fp_hash=fp_hash, source_name=source_name, C=C))
    con.close()
    print(f"Loaded {len(seeds)} Bini seeds from DB.")
    return seeds


def _build_m3r_seed() -> SeedRecord:
    """Pure M(3,R) structure constants as a 9x9x9 algebra tensor.
    Each basis element e_ij: e_ij * e_kl = delta_{jk} e_il.
    This is NOT T<3,3,3> — it's the associative multiplication table,
    living in a different 9-dim space and guaranteed rank ≤ 9.
    Gives the search a perfectly associative, non-Bini starting point.
    """
    n = 3
    dim = n * n
    C = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if j == k:   # e_ij * e_kl = e_il when j==k, else 0
                        alpha = i * n + j
                        beta  = k * n + l
                        gamma = i * n + l
                        C[alpha, beta, gamma] = 1.0
    return SeedRecord(
        name="M3R",
        fp_hash="m3r_seed_" + "0" * 7,
        source_name="M(3,R) structure constants",
        C=C,
    )


def _build_random_seeds(n_seeds: int = 2, seed_offset: int = 42) -> list[SeedRecord]:
    """Random 9x9x9 tensors scaled to the same Frobenius norm as T<3,3,3>.
    Gives the search uncorrelated starting basins with no algebraic bias.
    """
    target_norm = float(np.linalg.norm(T333))
    records = []
    for i in range(n_seeds):
        rng = np.random.default_rng(seed_offset + i)
        C = rng.standard_normal((9, 9, 9))
        C *= target_norm / np.linalg.norm(C)
        records.append(SeedRecord(
            name=f"rand_{seed_offset + i}",
            fp_hash=f"rand_seed_{seed_offset + i:04d}" + "0" * 3,
            source_name="random normal tensor",
            C=C,
        ))
    return records


def _build_scaffold_seed(name: str) -> SeedRecord:
    """Build a geometry-based scaffold seed tensor, normalized to T333's sphere.

    Five shapes:
      cross       — hub+spoke cross (c=0, 4 arms each with 2 elements)
      cross_diag  — cross with diagonal corners (axial + diagonal, D4 symmetry)
      hub_8cycle  — center hub with length-8 ring (cyclic locality)
      block_333   — direct sum of three Z/3Z groups (block diagonal)
      tree        — binary tree depth-2 (center → a1,a2 → b1..b6)

    All tensors are scaled to ||T333||_F = 5.1962 before returning.
    """
    n = 9
    C = np.zeros((n, n, n), dtype=np.float64)

    if name == "cross":
        # c=0, u1=1,u2=2, d1=3,d2=4, l1=5,l2=6, r1=7,r2=8
        horiz  = [5, 6, 0, 7, 8]   # l1,l2,c,r1,r2
        vert   = [1, 2, 0, 3, 4]   # u1,u2,c,d1,d2
        center = 0
        for hi, h in enumerate(horiz):
            for vi, v in enumerate(vert):
                if hi == vi:
                    C[center, h, v] = 1.0
        for h in horiz:
            C[h, h, center] = 1.0
        for v in vert:
            C[v, center, v] = 1.0
        C[center, center, center] = 1.0

    elif name == "cross_diag":
        center = 0
        axial  = [1, 2, 3, 4]
        diag   = [5, 6, 7, 8]
        for i in range(1, 9):
            C[i, center, i] = 1.0
            C[i, i, center] = 1.0
        C[center, center, center] = 1.0
        for i, a in enumerate(axial):
            C[center, a, a] = 1.0
            C[a, a, axial[(i+1) % 4]] = 0.5
            C[a, axial[(i+1) % 4], a] = 0.5
            C[diag[i], a, axial[(i+1) % 4]] = 0.5
        for i, d in enumerate(diag):
            C[center, d, d] = 1.0
            C[axial[i], d, diag[(i+1) % 4]] = 0.5
        for i in range(4):
            C[diag[i], axial[i], diag[i]] = 0.5
            C[axial[i], diag[i], axial[i]] = 0.5

    elif name == "hub_8cycle":
        center = 0
        ring   = list(range(1, 9))
        for i in range(1, 9):
            C[i, center, i] = 1.0
            C[i, i, center] = 1.0
        C[center, center, center] = 1.0
        for i, v in enumerate(ring):
            C[center, v, v] = 1.0
            C[ring[(i+4) % 8], v, v] = 0.5
        for i, v in enumerate(ring):
            vn = ring[(i+1) % 8]
            C[v, v, vn]   = 0.5
            C[vn, v, vn]  = 0.5
        for i, v in enumerate(ring):
            C[center, v, ring[(i+4) % 8]] = 1.0

    elif name == "block_333":
        U, W, Z = list(range(3)), list(range(3, 6)), list(range(6, 9))
        for i in U:
            for j in U:
                C[U[(U.index(i)+U.index(j)) % 3], i, j] = 1.0
        for i in W:
            for j in W:
                C[W[(W.index(i)+W.index(j)) % 3], i, j] = 1.0
        for i in Z:
            for j in Z:
                C[Z[(Z.index(i)+Z.index(j)) % 3], i, j] = 1.0
        for i, u in enumerate(U):
            for j, w in enumerate(W):
                C[Z[(i+j) % 3], u, w] = 1.0
        for i, w in enumerate(W):
            for j, z in enumerate(Z):
                C[U[(i+j) % 3], w, z] = 1.0
        for i, z in enumerate(Z):
            for j, u in enumerate(U):
                C[W[(i+j) % 3], z, u] = 1.0

    elif name == "tree":
        center = 0
        a = [1, 2]
        b = [[3, 4, 5], [6, 7, 8]]
        C[center, center, center] = 1.0
        for ai in a:
            C[ai, center, ai] = 1.0
            C[ai, ai, center] = 1.0
        for branch_i, branch in enumerate(b):
            ai = a[branch_i]
            for bi in branch:
                C[bi, center, bi] = 1.0
                C[bi, bi, center] = 1.0
            for i, bi in enumerate(branch):
                for j, bj in enumerate(branch):
                    C[ai, bi, bj] = 0.5
                    C[center, bi, bj] += 0.3
        for bi in b[0]:
            for bj in b[1]:
                C[center, bi, bj] = 0.3

    else:
        raise ValueError(f"Unknown scaffold: {name!r}")

    norm = float(np.linalg.norm(C))
    if norm > 1e-10:
        C *= float(np.linalg.norm(T333)) / norm

    fp_hash = f"scaffold_{name}_" + "0" * max(0, 11 - len(name))
    return SeedRecord(
        name=name,
        fp_hash=fp_hash,
        source_name=f"geometry scaffold ({name})",
        C=C,
    )


def load_all_seeds() -> list[SeedRecord]:
    """Return the full seed list.

    Loads extracted best-per-lineage C tensors from .npz files when available
    (saved by _extract_best_seeds.py), falling back to geometry scaffold seeds.
    """
    T333_NORM_val = float(np.linalg.norm(build_T333()))
    npz_specs = [
        ("cross",      "seed_cross_best.npz"),
        ("cross_diag", "seed_crossdiag_best.npz"),
        ("hub_8cycle", "seed_hub8cycle_best.npz"),
        ("block_333",  "seed_block333_best.npz"),
        ("tree",       "seed_tree_best.npz"),
    ]
    seeds: list[SeedRecord] = []
    for name, fname in npz_specs:
        path = OUT_DIR / fname
        if path.exists():
            data = np.load(str(path))
            C = data["C"].astype(np.float64)
            norm = float(np.linalg.norm(C))
            if norm > 1e-10:
                C = C * (T333_NORM_val / norm)
            fp_hash = f"extracted_{name}_" + "0" * max(0, 9 - len(name))
            seeds.append(SeedRecord(
                name=name,
                fp_hash=fp_hash,
                source_name=f"extracted best from lineage ({name})",
                C=C,
            ))
            print(f"  Loaded extracted seed: {name}  ({fname})")
        else:
            print(f"  WARNING: {fname} not found — falling back to scaffold for {name}")
            seeds.append(_build_scaffold_seed(name))

    generated_dir = OUT_DIR / "generated_seeds"
    if generated_dir.exists():
        loaded_generated = 0
        for path in sorted(generated_dir.glob("gen_*.npz")):
            try:
                with np.load(str(path)) as data:
                    C = np.asarray(data["C"], dtype=np.float64)
                if C.shape != (9, 9, 9):
                    continue
                norm = float(np.linalg.norm(C))
                if norm > 1e-10:
                    C = C * (T333_NORM_val / norm)
                seeds.append(
                    SeedRecord(
                        name=path.stem,
                        fp_hash=hashlib.md5(C.tobytes()).hexdigest(),
                        source_name="generated",
                        C=C,
                    )
                )
                loaded_generated += 1
            except Exception:
                continue
        if loaded_generated:
            print(f"  Loaded generated seeds: {loaded_generated}")
    print(f"Total seeds: {len(seeds)}")

    # Always include two fixed random-normal control seeds so there is an
    # unbiased baseline regardless of which geometry seeds were loaded.
    seeds.extend(_build_random_seeds(n_seeds=1, seed_offset=42))   # rand_42
    seeds.extend(_build_random_seeds(n_seeds=1, seed_offset=69))   # rand_69
    print(f"  Added random control seeds: rand_42, rand_69")
    print(f"Total seeds (with controls): {len(seeds)}")
    return seeds


def _filter_seeds_by_model(seeds: list[SeedRecord], top_k: int = 20) -> list[SeedRecord]:
    """Optionally rank seeds with the Stage 2 model when deployment is enabled."""
    try:
        from ml_seed.stage2.infer import is_model_deployable, score_seeds
    except Exception:
        return seeds

    if not seeds or not is_model_deployable():
        return seeds

    try:
        scores = score_seeds([seed.C for seed in seeds])
        ranked = sorted(zip(scores, seeds), key=lambda pair: pair[0]["acquisition"])
        keep = min(top_k, len(ranked))
        filtered = [seed for _, seed in ranked[:keep]]
        print(
            f"[MODEL] Enabled Stage 2 seed filter: keeping {keep}/{len(seeds)} seeds by acquisition",
            flush=True,
        )
        return filtered
    except Exception as exc:
        print(f"[MODEL] Seed filter disabled after scoring error: {exc}", flush=True)
        return seeds


def _next_seed_node_id() -> str:
    used_indices: set[int] = set()
    for node_id in _nodes:
        if not node_id.startswith("seed_"):
            continue
        try:
            used_indices.add(int(node_id.split("_", 1)[1]))
        except ValueError:
            continue

    next_index = 0
    while next_index in used_indices:
        next_index += 1
    return f"seed_{next_index}"


def _create_seed_node(seed: SeedRecord, node_id: str) -> Node:
    return Node(
        id=node_id,
        label=seed.name,
        parent_id=None,
        status="SEED",
        depth=0,
        direction=None,
        best_residual=np.inf,
        best_rank=-1,
        steps=0,
        steps_since_signal=0,
        cd=_comm_defect(seed.C),
        ad=_assoc_defect(seed.C),
        children=[],
        edge_type="fork",
        seed_name=seed.name,
        fp_hash=seed.fp_hash,
        source_name=seed.source_name,
        C=seed.C,
    )


def _probe_seed_nodes(seed_records: list[SeedRecord]) -> None:
    if not seed_records:
        return

    print("\nProbing seed residuals in parallel ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seed_records)) as executor:
        future_to_seed = {executor.submit(_probe_seed, seed): seed for seed in seed_records}
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            _, _res, rank, factors = future.result()
            with _lock:
                node = next(n for n in _nodes.values() if n.fp_hash == seed.fp_hash and n.parent_id is None)
                # NOTE: do NOT set node.best_residual here.
                # 'res' is ‖UVW−C‖ (CP rank of the algebra itself), not
                # ‖UVW−T333‖.  Storing it would corrupt the global best
                # display and make it look like T333 was nearly solved.
                node.best_rank = rank
                node.best_factors = factors


def _merge_missing_seed_nodes() -> list[SeedRecord]:
    seed_records = load_all_seeds()
    seed_records = _filter_seeds_by_model(seed_records)

    existing_root_fp_hashes = {
        node.fp_hash
        for node in _nodes.values()
        if node.parent_id is None and node.fp_hash
    }
    missing = [seed for seed in seed_records if seed.fp_hash not in existing_root_fp_hashes]
    if not missing:
        return []

    print(f"[RESUME] Adding {len(missing)} new seed roots discovered since last run.", flush=True)
    with _lock:
        for seed in missing:
            nid = _next_seed_node_id()
            _nodes[nid] = _create_seed_node(seed, nid)

    _write_state()
    _probe_seed_nodes(missing)
    _write_state()
    _save_graph_to_db()
    return missing

# ---------------------------------------------------------------------------
# New node factory
# ---------------------------------------------------------------------------
def _random_unit_direction(shape, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(shape)
    v /= np.linalg.norm(v)
    return v


GRAD_MIX = 0.25  # fraction of negative-gradient signal blended into walk direction


def _gradient_biased_direction(
    C: np.ndarray,
    factors: Optional[tuple],
    rng=None,
    grad_mix: float = GRAD_MIX,
) -> np.ndarray:
    """Return a unit direction blending a random vector with the negative CP
    gradient w.r.t. C.  By the envelope theorem, dL/dC = C - UVW, so the
    descent direction is (UVW - C), i.e. toward the current best approximation.
    Without factors falls back to a purely random direction."""
    rand_dir = _random_unit_direction(C.shape, rng)
    if factors is None:
        return rand_dir
    U, V, W = factors
    approx = _reconstruct(U, V, W)   # UVW — the current best rank-r approximation
    grad_dir = approx - C            # negative gradient: push C toward low-rank manifold
    g_norm = float(np.linalg.norm(grad_dir))
    if g_norm < 1e-14:
        return rand_dir
    grad_dir /= g_norm
    blended = grad_mix * grad_dir + (1.0 - grad_mix) * rand_dir
    b_norm = float(np.linalg.norm(blended))
    if b_norm < 1e-14:
        return rand_dir
    return blended / b_norm

def _make_green_from(parent: Node, edge_type: str = "fork") -> Node:
    rng = np.random.default_rng()
    direction = _gradient_biased_direction(parent.C, parent.best_factors, rng)
    nid = uuid.uuid4().hex[:16]
    node = Node(
        id=nid,
        label=nid[:8],
        parent_id=parent.id,
        status="GREEN",
        depth=parent.depth + 1,
        direction=direction,
        best_residual=np.inf,
        best_rank=-1,
        steps=0,
        steps_since_signal=0,
        cd=parent.cd,
        ad=parent.ad,
        children=[],
        edge_type=edge_type,
        seed_name=parent.seed_name or parent.label,
        fp_hash=parent.fp_hash,
        source_name=parent.source_name,
        C=parent.C.copy(),
        epsilon_scale=parent.epsilon_scale,
    )
    return node

# ---------------------------------------------------------------------------
# Process-pool step job — top-level so it is picklable on Windows (spawn).
# Matches the run_crt.py / parallel_core pattern: stateless, serialisable
# inputs, serialisable outputs.  All heavy numpy/scipy work runs here in an
# isolated worker process with no GIL or BLAS-thread contention.
# ---------------------------------------------------------------------------
def _step_job(
    C_flat: list,
    direction_flat: list,
    rank_range: list,
    n_restarts: int,
    als_iters: int,
    lbfgs_maxiter: int,
    rank_tol: float,
    epsilon_frac: float,
    epsilon_scale: float = 1.0,
    k_checkpoint: int = 0,
) -> tuple:
    """Worker-process function: take one slime-mold step and evaluate CP rank.
    Input/output must be plain Python types (no numpy arrays, no locks).
    Returns (C_new_flat, best_residual, best_rank, factors_ser_or_None, res_at_k_or_None).
    """
    shape = (9, 9, 9)
    C = np.array(C_flat, dtype=np.float64).reshape(shape)
    direction = np.array(direction_flat, dtype=np.float64).reshape(shape)
    epsilon = epsilon_frac * float(np.linalg.norm(C))
    C_new = C + epsilon * direction

    best_res = np.inf
    best_rank = -1
    best_factors_ser = None
    best_res_at_k = None
    # Scale restart budget with step size: big exploratory jumps get deeper
    # probing; small local steps stay cheap. Cap at 96/rank so total budget
    # (4 ranks × 96 = 384 restarts) stays within a reasonable single-step cost.
    restarts_each = min(96, max(24, int(24 * epsilon_scale)))

    for rank in rank_range:
        rng_base = int(time.time() * 1000) % (2 ** 31)
        for i in range(restarts_each):
            res, U, V, W, res_at_k = _one_restart(
                C_new, rank, seed=rng_base + i,
                als_iters=als_iters, lbfgs_maxiter=lbfgs_maxiter,
                T_target=T333,  # always measure against T<3,3,3>
                k_checkpoint=k_checkpoint,
            )
            if res < best_res:
                best_res = res
                best_rank = rank
                best_res_at_k = res_at_k
                best_factors_ser = [
                    U.ravel().tolist(),
                    V.ravel().tolist(),
                    W.ravel().tolist(),
                ]
            if best_res < rank_tol:
                return (
                    C_new.ravel().tolist(),
                    float(best_res),
                    best_rank,
                    best_factors_ser,
                    best_res_at_k,
                )
        if best_res < rank_tol:
            break

    return (
        C_new.ravel().tolist(),
        float(best_res),
        best_rank,
        best_factors_ser,
        best_res_at_k,
    )


def _submit_step(pool, node_id: str) -> "concurrent.futures.Future":
    """Submit one walker-step job to the process pool. Returns a Future."""
    node = _nodes[node_id]
    return pool.submit(
        _step_job,
        node.C.ravel().tolist(),
        node.direction.ravel().tolist(),
        list(RANK_RANGE),
        RESTARTS_PER_STEP,
        ALS_ITERS,
        LBFGS_MAXITER,
        RANK_TOL,
        EPSILON_FRAC * node.epsilon_scale,
        node.epsilon_scale,
        _get_k_checkpoint(),
    )


# Temperature for softmax-weighted refill selection.
# Low value → greedy (always pick best).  High value → uniform random.
# 2.0 gives a strong-but-not-exclusive preference for lower-residual nodes.
_REFILL_TEMPERATURE = 2.0

# How strongly to penalise large (old) lineages when selecting the next
# seed to explore.  Score = -log(residual) - TREE_SIZE_PENALTY * log(size).
# 0.15 gives a gentle nudge toward smaller trees without overriding residual.
_TREE_SIZE_PENALTY = 0.15


def _softmax_choice(candidates, subtree_sizes: dict | None = None):
    """Pick one candidate using softmax weights.

    Primary driver: -log(best_residual) — lower residual means exponentially
    higher weight.  Secondary (gentle): -TREE_SIZE_PENALTY * log(lineage_size)
    penalises old/large lineages so newly-added seeds aren't starved, without
    ever hard-excluding a lineage or overriding a clearly better basin.
    """
    sizes = subtree_sizes or {}
    scores = []
    for n in candidates:
        res_score = (
            -math.log(max(n.best_residual, 1e-12))
            if math.isfinite(n.best_residual)
            else -10.0
        )
        size = max(1, sizes.get(n.seed_name or "", 1))
        size_penalty = _TREE_SIZE_PENALTY * math.log(size)
        scores.append(res_score - size_penalty)
    max_s = max(scores)
    weights = [math.exp((s - max_s) / _REFILL_TEMPERATURE) for s in scores]
    return random.choices(candidates, weights=weights, k=1)[0]


def _active_green_counts_by_seed() -> dict[str, int]:
    active_per_seed: dict[str, int] = {}
    for nid in _green_ids:
        node = _nodes.get(nid)
        if node and node.seed_name:
            active_per_seed[node.seed_name] = active_per_seed.get(node.seed_name, 0) + 1
    return active_per_seed


def _subtree_sizes_by_seed() -> dict[str, int]:
    """Count total nodes (all statuses) per seed lineage."""
    counts: dict[str, int] = {}
    for node in _nodes.values():
        if node.seed_name:
            counts[node.seed_name] = counts.get(node.seed_name, 0) + 1
    return counts


def _do_refill(pool, node_futures: dict) -> None:
    """Fork a new GREEN node from the YELLOW/SEED pool.

    Selection is driven by residual (lower = better basin) with a mild
    log-penalty on large lineages so new/small trees aren't starved.
    Priority order:
    1. Any SEED root with zero active workers → immediate activation,
       picking among those via the penalised softmax.
    2. Under-quota lineages preferred over at-quota lineages.
       Within the eligible pool, softmax picks the node (res-primary,
       tree-size as a soft secondary).
    3. Fallback: all lineages at quota → softmax over all YELLOWs.
    Must be called while holding _lock.
    """
    yellow_nodes = [n for n in _nodes.values() if n.status in ("YELLOW", "SEED")]
    if not yellow_nodes:
        return

    active_per_seed = _active_green_counts_by_seed()
    subtree_sizes = _subtree_sizes_by_seed()
    all_seed_names = {n.seed_name for n in _nodes.values() if n.seed_name}
    num_seeds = max(1, len(all_seed_names))
    quota = max(1, POOL_SIZE // num_seeds)

    # First priority: any root seed lineage with zero active workers gets one.
    # Use penalised softmax so among inactive seeds the best basin still wins.
    zero_active_roots = [
        n for n in yellow_nodes
        if n.status == "SEED" and active_per_seed.get(n.seed_name, 0) == 0
    ]
    if zero_active_roots:
        best = _softmax_choice(zero_active_roots, subtree_sizes)
        new_node = _make_green_from(best, edge_type="fork")
        _nodes[new_node.id] = new_node
        best.children.append(new_node.id)
        _green_ids.append(new_node.id)
        f = _submit_step(pool, new_node.id)
        node_futures[f] = new_node.id
        print(
            f"  [POOL] refill: reactivated seed {best.label} "
            f"(size={subtree_sizes.get(best.seed_name or '', 0)}) with {new_node.id[:8]}..",
            flush=True,
        )
        return

    # Under-quota lineages preferred; fall back to all if everyone is at quota.
    under_quota = [n for n in yellow_nodes if active_per_seed.get(n.seed_name, 0) < quota]
    refill_pool = under_quota if under_quota else yellow_nodes
    best = _softmax_choice(refill_pool, subtree_sizes)
    new_node = _make_green_from(best, edge_type="fork")
    _nodes[new_node.id] = new_node
    best.children.append(new_node.id)
    _green_ids.append(new_node.id)
    f = _submit_step(pool, new_node.id)
    node_futures[f] = new_node.id
    print(
        f"  [POOL] refill: forked {new_node.id[:8]}.. from {best.label} "
        f"(seed={best.seed_name}, size={subtree_sizes.get(best.seed_name or '', 0)}, res={best.best_residual:.4e})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Result handler — runs in main process after each worker Future completes
# ---------------------------------------------------------------------------
def _handle_step_result(
    pool,
    node_futures: dict,
    node_id: str,
    future: "concurrent.futures.Future",
) -> None:
    global _step_counter, _solved, _solved_count, _global_best_residual, _immune_node_ids

    try:
        C_new_flat, res, rank, factors_ser, res_at_k = future.result()
    except Exception as exc:
        print(f"  [STEP ERROR {node_id[:8]}] {exc}", flush=True)
        with _lock:
            if node_id in _nodes:
                _nodes[node_id].status = "RED"
            if node_id in _green_ids:
                _green_ids.remove(node_id)
            if len(_green_ids) < POOL_SIZE and not _stop_event.is_set():
                _do_refill(pool, node_futures)
        _write_slime_status()
        return

    C_new = np.array(C_new_flat, dtype=np.float64).reshape(9, 9, 9)
    factors: Optional[tuple] = None
    if factors_ser is not None:
        U = np.array(factors_ser[0], dtype=np.float64).reshape(rank, 9)
        V = np.array(factors_ser[1], dtype=np.float64).reshape(rank, 9)
        W = np.array(factors_ser[2], dtype=np.float64).reshape(rank, 9)
        factors = (U, V, W)

    # Update node state in main process
    with _lock:
        if node_id not in _nodes:
            return
        node = _nodes[node_id]
        if node.status != "GREEN":
            return
        _step_counter += 1
        node.C = C_new
        node.steps += 1
        node.cd = _comm_defect(C_new)
        node.ad = _assoc_defect(C_new)
        if res < node.best_residual:
            node.best_residual = res
            node.best_rank = rank
            node.best_factors = factors
        try:
            from ml_seed.collector import get_collector
            get_collector().record(
                seed_C=node.C,
                res_k=res_at_k,
                res_final=res,
                rank=rank,
                node_name=(f"{node.seed_name}:{node.id}" if node.seed_name else node.id),
            )
        except Exception:
            pass
        _is_new_global_best = (
            math.isfinite(res) and res < _global_best_residual
        )
        if _is_new_global_best:
            _global_best_residual = res
        # --- Recompute top-10 elite set ---
        # Collect all nodes with a finite best_residual, sort ascending.
        ranked = sorted(
            [(n.best_residual, n.id)
             for n in _nodes.values()
             if math.isfinite(n.best_residual) and n.status not in ("RED", "SEED")],
            key=lambda t: t[0],
        )
        new_immune = {nid for _, nid in ranked[:_ELITE_SIZE]}
        _newly_immune = new_immune - _immune_node_ids   # just entered top-10
        _immune_node_ids = new_immune
        if res > SIGNAL_THRESHOLD:
            node.steps_since_signal += 1
        else:
            node.steps_since_signal = 0
        # --- Adaptive step size ---
        prev_res = node.recent_residuals[-1] if node.recent_residuals else res
        node.recent_residuals.append(res)
        if len(node.recent_residuals) > 5:
            node.recent_residuals.pop(0)
        _direction_reset = False
        if len(node.recent_residuals) == 5:
            var = sum((r - sum(node.recent_residuals)/5)**2 for r in node.recent_residuals) / 5
            if var < 1e-4:
                if node.epsilon_scale >= 10.0:
                    # Already at ceiling and still stuck — this direction is dead.
                    # Reset to a gradient-biased fresh direction, restart scale.
                    node.direction = _gradient_biased_direction(node.C, node.best_factors)
                    node.epsilon_scale = 1.0
                    node.recent_residuals.clear()
                    _direction_reset = True
                else:
                    # Stuck but not at ceiling — expand step to escape
                    node.epsilon_scale = min(node.epsilon_scale * 2.0, 10.0)
        if not _direction_reset and prev_res > 0 and (prev_res - res) / prev_res > 0.10:
            # Good improvement — tighten step to exploit
            node.epsilon_scale = max(node.epsilon_scale * 0.5, 0.1)
        _eps_scale = node.epsilon_scale
        # Snapshot for logging (outside lock)
        _depth = node.depth
        _steps = node.steps
        _best  = node.best_residual
        _since = node.steps_since_signal

    _is_newly_immune = node_id in _newly_immune

    print(
        f"  [WALKER {node_id[:8]}] step={_steps} depth={_depth} rank={rank}"
        + (" [ELITE]" if node_id in _immune_node_ids else "")
        + "\n"
        f"    res={res:.4e} best={_best:.4e} starv={_since}/{STARVATION_LIMIT} eps={_eps_scale:.2f}"
        + (" [DIR RESET]" if _direction_reset else ""),
        flush=True,
    )
    _write_slime_status()

    # --- Fork 2 fresh children when a node enters the top-10 elite ---
    if _is_newly_immune:
        with _lock:
            node = _nodes.get(node_id)
            if node and node.status == "GREEN" and not _stop_event.is_set():
                print(
                    f"  [ELITE ENTRY] {node_id[:8]} entered top-{_ELITE_SIZE} "
                    f"(res={node.best_residual:.4e}) — forking 2 fresh children",
                    flush=True,
                )
                for _ in range(2):
                    if len(_green_ids) >= POOL_SIZE * 2:
                        break
                    child = _make_green_from(node, edge_type="elite_fork")
                    child.direction = np.random.default_rng().standard_normal(
                        child.direction.shape
                    )
                    _nodes[child.id] = child
                    node.children.append(child.id)
                    _green_ids.append(child.id)
                    f = _submit_step(pool, child.id)
                    node_futures[f] = child.id

    # --- Check SOLVED (candidate below tolerance) ---
    if res < RANK_TOL:
        with _lock:
            _solved = True
            _solved_count += 1
            # Keep the node alive as YELLOW (permanent waypoint) rather than
            # killing it as RED, so walkers can keep forking from a proven
            # low-rank neighbourhood.
            if node_id in _nodes and _nodes[node_id].status == "GREEN":
                _nodes[node_id].status = "YELLOW"
                if node_id in _green_ids:
                    _green_ids.remove(node_id)
                # Immediately refill the pool so search never stalls
                if not _stop_event.is_set():
                    _do_refill(pool, node_futures)
        print(f"\n{'='*60}", flush=True)
        print(f"CANDIDATE rank={rank}  residual={res:.6e}  "
              f"(#{_solved_count} total — search CONTINUES)", flush=True)
        print(f"{'='*60}\n", flush=True)
        # Validate asynchronously so the search doesn't block
        C_snapshot = C_new.copy()
        factors_snapshot = (
            (factors[0].copy(), factors[1].copy(), factors[2].copy())
            if factors else None
        )
        t = threading.Thread(
            target=_validate_candidate,
            args=(node_id, C_snapshot, rank, res, factors_snapshot),
            daemon=True,
        )
        t.start()
        return

    # --- Global-best burst fork ---
    # When a node sets a new global best below the threshold, immediately
    # spawn _GLOBAL_BEST_FORK_N children from it with gradient-biased
    # directions so the promising neighbourhood is exploited right away,
    # without waiting for the normal branching probability to fire.
    if _is_new_global_best and res < _GLOBAL_BEST_FORK_THRESHOLD:
        with _lock:
            node = _nodes.get(node_id)
            if node and node.status == "GREEN" and not _stop_event.is_set():
                print(
                    f"  [GLOBAL BEST] {res:.4e} — burst forking "
                    f"{_GLOBAL_BEST_FORK_N} children from {node_id[:8]}",
                    flush=True,
                )
                for _ in range(_GLOBAL_BEST_FORK_N):
                    if len(_green_ids) >= POOL_SIZE * 2:
                        break  # don't over-saturate the pool
                    child = _make_green_from(node, edge_type="best_fork")
                    _nodes[child.id] = child
                    node.children.append(child.id)
                    _green_ids.append(child.id)
                    f = _submit_step(pool, child.id)
                    node_futures[f] = child.id

    # --- Check starvation ---
    with _lock:
        if node_id not in _nodes or _nodes[node_id].status != "GREEN":
            return
        node = _nodes[node_id]
        if node.steps_since_signal >= STARVATION_LIMIT:
            # --- Top-10 elite immunity ---
            # The _ELITE_SIZE nodes with the lowest best_residual are immune
            # to starvation. Immunity is re-evaluated every step: a node that
            # has been pushed out of the top-10 by a better node is no longer
            # immune and becomes eligible for normal starvation rules.
            if node_id in _immune_node_ids:
                # Grant immunity: reset starvation counter and keep running.
                node.steps_since_signal = 0
                print(
                    f"  [WALKER {node_id[:8]}] starvation PARDONED — "
                    f"in top-{_ELITE_SIZE} elite (res={node.best_residual:.4e})",
                    flush=True,
                )
                # Fall through to the branching / continue-step section below.
            else:
                # Lost elite status or never had it → normal starvation → RED.
                node.status = "RED"
                if node_id in _green_ids:
                    _green_ids.remove(node_id)
                print(
                    f"  [WALKER {node_id[:8]}] STARVED -> RED (best={node.best_residual:.4e})",
                    flush=True,
                )
                if len(_green_ids) < POOL_SIZE and not _stop_event.is_set():
                    _do_refill(pool, node_futures)
                _write_slime_status()
                return

    # --- Check branching ---
    with _lock:
        if node_id not in _nodes or _nodes[node_id].status != "GREEN":
            return
        node = _nodes[node_id]
        br = node.best_residual
        prob = (BRANCHING_HOT if br < 0.01
                else BRANCHING_RICH if br < 0.02
                else BRANCHING_BASE)
        if np.random.random() < prob:
            # Branching is a parent→YELLOW slot transfer: pool size stays constant.
            # Do NOT gate on len(_green_ids) < POOL_SIZE — that would always be
            # False when the pool is full, permanently blocking all tree growth.
            child = _make_green_from(node, edge_type="fork")
            _nodes[child.id] = child
            node.children.append(child.id)
            node.status = "YELLOW"
            if node_id in _green_ids:
                _green_ids.remove(node_id)
            _green_ids.append(child.id)
            f = _submit_step(pool, child.id)
            node_futures[f] = child.id
            print(
                f"  [BRANCH] {node_id[:8]} -> YELLOW, forked child {child.id[:8]} "
                f"(pool={len(_green_ids)})",
                flush=True,
            )
            _write_slime_status()
            return  # parent is now YELLOW — done

    # --- Continue: submit next step for this node ---
    with _lock:
        if node_id in _nodes and _nodes[node_id].status == "GREEN":
            f = _submit_step(pool, node_id)
            node_futures[f] = node_id

# ---------------------------------------------------------------------------
# State writer thread
# ---------------------------------------------------------------------------
def _state_writer_thread():
    while not _stop_event.is_set():
        try:
            _write_state()
        except Exception as e:
            print(f"  [STATE] JSON write error: {e}", flush=True)
        try:
            _save_graph_to_db()
        except Exception as e:
            print(f"  [STATE] DB write error: {e}", flush=True)
        time.sleep(STATE_WRITE_INTERVAL)
    # Final writes
    try:
        _write_state()
    except Exception:
        pass
    try:
        _save_graph_to_db()
    except Exception:
        pass


SEED_PROBE_RESTARTS = 8   # quick budget for startup probing — just enough to rank seeds


def _probe_seed(seed: SeedRecord) -> tuple[str, float, int, Optional[tuple]]:
    print(
        f"Probing seed {seed.name} ({seed.fp_hash[:8]}..., source={seed.source_name}) ...",
        flush=True,
    )
    res, rank, factors = compute_algebra_rank(seed.C, n_restarts=SEED_PROBE_RESTARTS)
    print(f"  Seed {seed.name}: best_res={res:.4e} at tested rank={rank}", flush=True)
    return seed.fp_hash, res, rank, factors

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global POOL_SIZE, STARVATION_LIMIT, RESTARTS_PER_STEP, _t_start, ALS_ITERS, LBFGS_MAXITER, BRANCHING_BASE

    parser = argparse.ArgumentParser(description="Slime mold CP-rank search for T<3,3,3>")
    parser.add_argument("--dry-run", action="store_true",
                        help="60-second dry run with POOL_SIZE=3, STARVATION_LIMIT=5")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--starvation", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None, help="Max run duration in seconds")
    parser.add_argument("--quick-mode", action="store_true",
                        help="Raise signal threshold + branching probability so depth "
                             "grows quickly — useful for visualising tree structure")
    args = parser.parse_args()

    if args.dry_run:
        POOL_SIZE = 3
        STARVATION_LIMIT = 5
        RESTARTS_PER_STEP = 4   # fast for dry run
        ALS_ITERS = 40
        LBFGS_MAXITER = 150
        duration = 60 if args.duration is None else args.duration
        print(
            "DRY RUN DEFAULTS: POOL_SIZE=3, STARVATION_LIMIT=5, "
            f"RESTARTS_PER_STEP=4, duration={duration}s after init"
        )
    else:
        if args.pool_size:
            POOL_SIZE = args.pool_size
        if args.starvation:
            STARVATION_LIMIT = args.starvation
        if args.restarts:
            RESTARTS_PER_STEP = args.restarts
        duration = args.duration

    if args.pool_size is not None:
        POOL_SIZE = args.pool_size
    if args.starvation is not None:
        STARVATION_LIMIT = args.starvation
    if args.restarts is not None:
        RESTARTS_PER_STEP = args.restarts
    if args.quick_mode:
        BRANCHING_BASE = QUICK_BRANCHING_BASE
        print("QUICK MODE: BRANCHING_BASE=0.50 (depth-growth demo — starvation/RED still active)")

    _t_start = time.time()

    print(f"\n{'='*60}")
    print(f"SLIME MOLD SEARCH  T<3,3,3>  rank target < {min(RANK_RANGE)}")
    print(f"POOL_SIZE={POOL_SIZE}  STARVATION_LIMIT={STARVATION_LIMIT}  RESTARTS_PER_STEP={RESTARTS_PER_STEP}")
    print(f"RANK_RANGE={RANK_RANGE}  EPSILON={EPSILON_FRAC}*||C||  tol={RANK_TOL:.0e}")
    print(f"{'='*60}\n")

    # 0. Init / attempt resume from persistent graph DB
    _init_graph_db()
    resumed = _load_graph_from_db()

    if not resumed:
        # 1. Load seeds (fresh start)
        seed_records = load_all_seeds()
        seed_records = _filter_seeds_by_model(seed_records)

        # 2. Create SEED nodes
        for i, seed in enumerate(seed_records):
            nid = f"seed_{i}"
            _nodes[nid] = _create_seed_node(seed, nid)

        _write_state()

        # 3. Probe seeds in parallel so startup does real work immediately.
        _probe_seed_nodes(seed_records)

        _write_state()
        _save_graph_to_db()

        # 4. Spawn initial GREEN walkers (round-robin across seeds)
        print(f"\nSpawning initial {POOL_SIZE} walkers ...", flush=True)
        with _lock:
            for i in range(POOL_SIZE):
                seed_node = _nodes[f"seed_{i % len(seed_records)}"]
                child = _make_green_from(seed_node, edge_type="fork")
                _nodes[child.id] = child
                seed_node.children.append(child.id)
                _green_ids.append(child.id)
    else:
        added_seeds = _merge_missing_seed_nodes()
        # Resumed: write JSON + save DB to confirm state is fresh
        if added_seeds:
            print(
                f"[RESUME] Graph loaded — merged {len(added_seeds)} new seeds and continuing.",
                flush=True,
            )
        else:
            print("[RESUME] Graph loaded — continuing where we left off.", flush=True)
        _write_state()
        _save_graph_to_db()

        # Spawn POOL_SIZE new walkers from the best available YELLOW/SEED nodes.
        # Priority: seed lineages with no workers (penalised softmax), then
        # YELLOW nodes sorted by penalised score (residual-primary, tree-size soft).
        print(f"[RESUME] Spawning {POOL_SIZE} walkers from best existing waypoints ...", flush=True)
        with _lock:
            subtree_sizes = _subtree_sizes_by_seed()
            active_per_seed = _active_green_counts_by_seed()
            zero_active_roots = [
                n for n in _nodes.values()
                if n.status == "SEED" and active_per_seed.get(n.seed_name, 0) == 0
            ]
            spawn_parents = []
            for root in zero_active_roots:
                if len(spawn_parents) >= POOL_SIZE:
                    break
                spawn_parents.append(root)

            # Fill remaining slots: sort YELLOW by penalised score (best first)
            if len(spawn_parents) < POOL_SIZE:
                def _penalised_key(n):
                    res = n.best_residual if math.isfinite(n.best_residual) else 999.0
                    size = max(1, subtree_sizes.get(n.seed_name or "", 1))
                    return res * (size ** _TREE_SIZE_PENALTY)
                yellow_pool = sorted(
                    [n for n in _nodes.values() if n.status == "YELLOW"],
                    key=_penalised_key,
                )
                for wp in yellow_pool:
                    if len(spawn_parents) >= POOL_SIZE:
                        break
                    spawn_parents.append(wp)

            for parent in spawn_parents:
                if parent is None:
                    break
                child = _make_green_from(parent, edge_type="fork")
                _nodes[child.id] = child
                parent.children.append(child.id)
                _green_ids.append(child.id)

    # 5. State writer runs as a lightweight daemon thread (I/O only)
    writer_thread = threading.Thread(target=_state_writer_thread, daemon=True)
    writer_thread.start()

    # Start duration countdown AFTER initialization
    _t_start = time.time()
    t_deadline = (_t_start + duration) if duration else None

    print(
        f"Search running ({POOL_SIZE} worker processes). "
        f"State -> {STATE_PATH}",
        flush=True,
    )
    print(f"Open {OUT_DIR / 'bini_render.html'} in your browser.\n", flush=True)
    _write_slime_status(force=True)

    # 6. Main parallel event loop — mirrors run_crt.py / parallel_core pattern:
    #    ProcessPoolExecutor, POOL_SIZE worker processes.
    #    Each active walker slot = one outstanding Future in the pool.
    #    On completion: update state, re-submit next step (or refill from YELLOW).
    with concurrent.futures.ProcessPoolExecutor(max_workers=POOL_SIZE) as pool:
        node_futures: dict = {}  # future -> node_id

        # Submit first step for every initial walker
        for node_id in list(_green_ids):
            f = _submit_step(pool, node_id)
            node_futures[f] = node_id

        try:
            while not _stop_event.is_set():
                if t_deadline and time.time() >= t_deadline:
                    print(f"\nDuration limit ({duration}s) reached — stopping.", flush=True)
                    _stop_event.set()
                    break

                if not node_futures:
                    time.sleep(0.1)
                    continue

                done, _ = concurrent.futures.wait(
                    list(node_futures.keys()),
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    node_id = node_futures.pop(future)
                    _handle_step_result(pool, node_futures, node_id, future)

        except KeyboardInterrupt:
            print("\nInterrupted — stopping.", flush=True)
            _stop_event.set()

    # Final state write
    _write_state()
    _save_graph_to_db()
    _write_slime_status(force=True)
    stats = _get_stats()
    print(f"\n{'='*60}")
    print(f"Final stats: {json.dumps(stats, indent=2)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
