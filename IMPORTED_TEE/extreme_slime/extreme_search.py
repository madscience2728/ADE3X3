"""
Extreme Slime — 2-D rank-boundary scanner for T<3,3,3>.

Scans a regular N×N grid of tensor positions centred on T<3,3,3>,
asking at each pixel:

    "What is the MINIMUM CP rank at which C(α,β) can be decomposed?"

The slice is parameterised as
    C(α, β) = T<3,3,3>  +  α·û  +  β·v̂

where û points toward the best available Bini seed tensor and v̂ is a
random direction orthogonal to û.

This is the CLIFF search: instead of bini_slime's basin-hunting (finding
low-residual points at fixed high rank), we map the rank boundary — the
contour line in tensor space where feasibility jumps from rank R to rank R+1.

Ranks probed: 9 (theoretical lower bound = one per input/output slot) up to
19 (best known upper bound from bini_slime lineages).

Output: extreme_state.json  (read live by extreme_render.html)
        extreme_scan.db     (SQLite backup for resume)
"""
from __future__ import annotations

import sys
# Force UTF-8 output on Windows (default console is cp1252 which can't
# encode Greek letters, arrows, etc. used in our progress prints).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import concurrent.futures
import json
import math
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Throttle BLAS threads before numpy/scipy are imported.
# Same reasoning as bini_slime: we use process-level parallelism, not
# intra-op threads, so each worker should be single-threaded internally.
# ---------------------------------------------------------------------------
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Hyper-parameters  (all overridable via CLI)
# ---------------------------------------------------------------------------
GRID_W        = 64       # cells along beta axis (columns / horizontal)
GRID_H        = 64       # cells along alpha axis (rows / vertical)
ALPHA_RANGE   = 3.0      # half-width of scan window along u
BETA_RANGE    = 3.0      # half-width of scan window along v
ALPHA_CENTER  = 0.0      # centre of scan window along u (0 = T333)
BETA_CENTER   = 0.0      # centre of scan window along v (0 = T333)
RANK_MIN      = 9        # lowest rank to probe (one factor per matrix entry col)
RANK_MAX      = 19       # highest rank to probe (bini_slime lineage best)
RESTARTS      = 24       # CP restarts per (rank, cell)
ALS_ITERS     = 60       # ALS warm-up iterations per restart
LBFGS_MAXITER = 200      # L-BFGS-B iterations per restart
RANK_TOL      = 1e-5     # residual < this → rank achieved for this cell
POOL_SIZE     = max(4, (os.cpu_count() or 8))
STATE_WRITE_INTERVAL = 3  # seconds between JSON writes

OUT_DIR      = Path(__file__).parent
STATE_PATH   = OUT_DIR / "extreme_state.json"
DB_PATH      = OUT_DIR / "extreme_scan.db"

# ---------------------------------------------------------------------------
# T<3,3,3>   (3×3 matrix multiplication tensor, 9×9×9)
# ---------------------------------------------------------------------------
def _build_T333() -> np.ndarray:
    n, dim = 3, 9
    T = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[i * n + j, j * n + k, i * n + k] = 1.0
    return T

T333 = _build_T333()
T333_NORM = float(np.linalg.norm(T333))   # sqrt(27) ≈ 5.196

# ---------------------------------------------------------------------------
# CP decomposition utilities
# All functions must be picklable top-level (no lambdas or unbound closures)
# so they can be sent to Windows spawn-based ProcessPoolExecutor workers.
# ---------------------------------------------------------------------------
def _reconstruct(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.einsum("ra,rb,rc->abc", U, V, W)


def _loss_and_grad(params: np.ndarray, T: np.ndarray, r: int):
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


def _als_step(T: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray):
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
    T: np.ndarray,
    rank: int,
    seed: int,
    als_iters: int,
    lbfgs_maxiter: int,
) -> float:
    """Single ALS + L-BFGS-B restart. Returns residual ‖UVW − T‖."""
    d1, d2, d3 = T.shape
    rng = np.random.default_rng(seed)
    scale = max((float(np.linalg.norm(T)) / max(rank, 1)) ** (1 / 3), 0.1)
    U = rng.standard_normal((rank, d1)) * scale
    V = rng.standard_normal((rank, d2)) * scale
    W = rng.standard_normal((rank, d3)) * scale
    for _ in range(als_iters):
        U, V, W = _als_step(T, U, V, W)
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    try:
        opt = minimize(
            _loss_and_grad, x0, args=(T, rank),
            method="L-BFGS-B", jac=True,
            options={"maxiter": lbfgs_maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        return float(np.sqrt(max(2.0 * opt.fun, 0.0)))
    except Exception:
        return float(np.linalg.norm(T - _reconstruct(U, V, W)))


# ---------------------------------------------------------------------------
# Direction construction
# ---------------------------------------------------------------------------
def build_directions() -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return (û, v̂, u_label, v_label).

    û: unit vector from T333 toward the best available extracted Bini seed.
       Falls back to a fixed random direction if no seed files are found.
    v̂: random unit direction orthogonal to û (Gram-Schmidt).
    """
    bini_dir = ROOT / "bini_slime"
    seed_candidates = [
        ("cross",      bini_dir / "seed_cross_best.npz"),
        ("cross_diag", bini_dir / "seed_crossdiag_best.npz"),
        ("hub_8cycle", bini_dir / "seed_hub8cycle_best.npz"),
        ("block_333",  bini_dir / "seed_block333_best.npz"),
        ("tree",       bini_dir / "seed_tree_best.npz"),
    ]

    u: np.ndarray | None = None
    u_label = "random42"

    for seed_name, path in seed_candidates:
        if not path.exists():
            continue
        try:
            C = np.load(str(path))["C"].astype(np.float64)
            norm = float(np.linalg.norm(C))
            if norm < 1e-10:
                continue
            C *= T333_NORM / norm          # normalise to same sphere as T333
            diff = C - T333
            d_norm = float(np.linalg.norm(diff))
            if d_norm < 1e-10:
                continue
            u = diff / d_norm
            u_label = f"T333->{seed_name}"
            print(f"[DIR] u: {u_label}")
            break
        except Exception as e:
            print(f"[DIR] Warning loading {path.name}: {e}")

    if u is None:
        rng = np.random.default_rng(42)
        u = rng.standard_normal(T333.shape)
        u /= np.linalg.norm(u)
        print("[DIR] u: random42 (no seed files found)")

    # v̂: Gram-Schmidt orthogonalisation
    v = np.random.default_rng(99).standard_normal(T333.shape)
    v -= np.dot(v.ravel(), u.ravel()) * u
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-10:
        v = np.random.default_rng(100).standard_normal(T333.shape)
        v -= np.dot(v.ravel(), u.ravel()) * u
        v_norm = float(np.linalg.norm(v))
    v /= v_norm
    v_label = "random_ortho99"
    print(f"[DIR] v: {v_label}")

    return u, v, u_label, v_label


# ---------------------------------------------------------------------------
# Grid-cell worker  (top-level → picklable on Windows spawn)
# ---------------------------------------------------------------------------
def _scan_cell(
    i: int,
    j: int,
    alpha: float,
    beta: float,
    T333_flat: list,
    u_flat: list,
    v_flat: list,
    rank_min: int,
    rank_max: int,
    restarts: int,
    als_iters: int,
    lbfgs_maxiter: int,
    rank_tol: float,
) -> dict:
    """Scan one grid cell C(α,β) = T333 + α·û + β·v̂.

    Probes ranks from rank_min to rank_max (bottom-up), stopping as soon
    as a rank achieves residual < rank_tol (that is the minimum feasible rank).

    Returns a JSON-serialisable dict for the state file and DB.
    """
    shape = (9, 9, 9)
    T = np.array(T333_flat, dtype=np.float64).reshape(shape)
    u = np.array(u_flat,    dtype=np.float64).reshape(shape)
    v = np.array(v_flat,    dtype=np.float64).reshape(shape)
    C = T + alpha * u + beta * v

    # Per-cell deterministic seed — avoids identical random sequences across cells
    rng_base = abs(hash((i, j, round(alpha, 6), round(beta, 6)))) % (2 ** 28)

    residuals: dict[str, float] = {}
    min_rank: int | None = None
    best_overall = math.inf

    for rank in range(rank_min, rank_max + 1):
        best = math.inf
        for k in range(restarts):
            seed = (rng_base + rank * 10000 + k) % (2 ** 31)
            res = _one_restart(C, rank, seed, als_iters, lbfgs_maxiter)
            if res < best:
                best = res
            if best < rank_tol:
                break   # converged — no need for more restarts at this rank

        residuals[str(rank)] = round(float(best), 8)
        if best < best_overall:
            best_overall = best

        if best < rank_tol and min_rank is None:
            min_rank = rank
            break   # minimum feasible rank found — stop scanning higher ranks

    return {
        "i": i,
        "j": j,
        "alpha": round(float(alpha), 6),
        "beta":  round(float(beta),  6),
        "min_rank": min_rank,
        "best_res": round(float(best_overall), 8) if not math.isinf(best_overall) else None,
        "residuals": residuals,
        "status": "done",
    }


# ---------------------------------------------------------------------------
# SQLite persistence (for resume)
# ---------------------------------------------------------------------------
def _init_db() -> None:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS cells (
            i INTEGER,
            j INTEGER,
            alpha REAL,
            beta  REAL,
            min_rank INTEGER,
            best_res REAL,
            residuals_json TEXT,
            status TEXT,
            PRIMARY KEY (i, j)
        )
    """)
    con.commit()
    con.close()


def _save_cell_to_db(cell: dict) -> None:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        INSERT OR REPLACE INTO cells
            (i, j, alpha, beta, min_rank, best_res, residuals_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cell["i"], cell["j"], cell["alpha"], cell["beta"],
        cell.get("min_rank"), cell.get("best_res"),
        json.dumps(cell.get("residuals", {})),
        cell.get("status", "done"),
    ))
    con.commit()
    con.close()


def _load_done_cells() -> dict[tuple[int, int], dict]:
    """Load previously completed cells from the DB (for resume)."""
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT i, j, alpha, beta, min_rank, best_res, residuals_json, status FROM cells"
    ).fetchall()
    con.close()
    out: dict[tuple[int, int], dict] = {}
    for row in rows:
        i, j, alpha, beta, min_rank, best_res, res_json, status = row
        out[(i, j)] = {
            "i": i, "j": j,
            "alpha": alpha, "beta": beta,
            "min_rank": min_rank,
            "best_res": best_res,
            "residuals": json.loads(res_json) if res_json else {},
            "status": status,
        }
    return out


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_lock        = threading.Lock()
_cells: dict[tuple[int, int], dict] = {}   # (i,j) → result
_grid_meta: dict = {}
_t_start     = time.time()
_stop_event  = threading.Event()
_done_count  = 0
_total       = 0


def _get_cells_list() -> list:
    nh = _grid_meta.get("grid_h", _grid_meta.get("grid_n", GRID_H))
    nw = _grid_meta.get("grid_w", _grid_meta.get("grid_n", GRID_W))
    out = []
    for i in range(nh):
        for j in range(nw):
            c = _cells.get((i, j))
            if c is not None:
                out.append(c)
    return out


def _write_state() -> None:
    with _lock:
        cells = _get_cells_list()
        meta  = dict(_grid_meta)
        done  = _done_count
        total = _total

    # Rank distribution summary
    rank_counts: dict = {}
    no_conv = 0
    for c in cells:
        r = c.get("min_rank")
        if r is None:
            no_conv += 1
        else:
            rank_counts[str(r)] = rank_counts.get(str(r), 0) + 1

    state = {
        "grid":   meta,
        "cells":  cells,
        "stats": {
            "total":       total,
            "done":        done,
            "pending":     total - done,
            "pct":         round(done * 100 / max(total, 1), 1),
            "elapsed":     round(time.time() - _t_start, 1),
            "rank_counts": rank_counts,
            "no_conv":     no_conv,
        },
    }
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _state_writer() -> None:
    while not _stop_event.is_set():
        try:
            _write_state()
        except Exception as e:
            print(f"[STATE] write error: {e}", flush=True)
        time.sleep(STATE_WRITE_INTERVAL)
    try:
        _write_state()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global GRID_W, GRID_H, ALPHA_RANGE, BETA_RANGE, ALPHA_CENTER, BETA_CENTER
    global RANK_MIN, RANK_MAX, RESTARTS, ALS_ITERS, LBFGS_MAXITER, POOL_SIZE
    global _t_start, _total, _done_count

    parser = argparse.ArgumentParser(
        description="Extreme Slime — 2-D rank-boundary scanner for T<3,3,3>"
    )
    parser.add_argument("--grid",      type=int,   default=None,
                        help="Grid size N (sets both W and H to N×N, overrides --grid-w/--grid-h)")
    parser.add_argument("--grid-w",    type=int,   default=GRID_W,
                        help=f"Grid width  (beta/columns,  default {GRID_W})")
    parser.add_argument("--grid-h",    type=int,   default=GRID_H,
                        help=f"Grid height (alpha/rows,    default {GRID_H})")
    parser.add_argument("--range",     type=float, default=ALPHA_RANGE,
                        help=f"Axis range ±R (default {ALPHA_RANGE})")
    parser.add_argument("--rank-min",  type=int,   default=RANK_MIN,
                        help=f"Minimum rank to probe (default {RANK_MIN})")
    parser.add_argument("--rank-max",  type=int,   default=RANK_MAX,
                        help=f"Maximum rank to probe (default {RANK_MAX})")
    parser.add_argument("--restarts",  type=int,   default=RESTARTS,
                        help=f"CP restarts per rank per cell (default {RESTARTS})")
    parser.add_argument("--pool",      type=int,   default=POOL_SIZE,
                        help=f"Worker process count (default {POOL_SIZE})")
    parser.add_argument("--center-alpha", type=float, default=ALPHA_CENTER,
                        help=f"Centre of scan window along u (default {ALPHA_CENTER})")
    parser.add_argument("--center-beta",  type=float, default=BETA_CENTER,
                        help=f"Centre of scan window along v (default {BETA_CENTER})")
    parser.add_argument("--fresh",     action="store_true",
                        help="Ignore existing DB and restart the scan from scratch")
    args = parser.parse_args()

    if args.grid is not None:
        GRID_W = args.grid
        GRID_H = args.grid
    else:
        GRID_W = args.grid_w
        GRID_H = args.grid_h
    ALPHA_RANGE   = args.range
    BETA_RANGE    = args.range
    ALPHA_CENTER  = args.center_alpha
    BETA_CENTER   = args.center_beta
    RANK_MIN   = args.rank_min
    RANK_MAX   = args.rank_max
    RESTARTS   = args.restarts
    POOL_SIZE  = args.pool
    _t_start   = time.time()
    _total     = GRID_W * GRID_H

    print(f"\n{'='*64}")
    print(f"EXTREME SLIME  rank-boundary scanner  T<3,3,3>")
    print(f"Grid:     {GRID_W}x{GRID_H} = {_total} cells  (W x H)")
    print(f"Center:   a={ALPHA_CENTER:+.2f}  b={BETA_CENTER:+.2f}")
    print(f"Range:    a in [{ALPHA_CENTER-ALPHA_RANGE:.2f}, {ALPHA_CENTER+ALPHA_RANGE:.2f}]  b in [{BETA_CENTER-BETA_RANGE:.2f}, {BETA_CENTER+BETA_RANGE:.2f}]")
    print(f"Ranks:    {RANK_MIN} .. {RANK_MAX}  (bottom-up, stop at first convergence)")
    print(f"Restarts: {RESTARTS}/rank/cell   ALS={ALS_ITERS}   LBFGS={LBFGS_MAXITER}")
    print(f"Workers:  {POOL_SIZE}")
    print(f"{'='*64}\n")

    _init_db()
    u, v, u_label, v_label = build_directions()

    alphas = np.linspace(ALPHA_CENTER - ALPHA_RANGE, ALPHA_CENTER + ALPHA_RANGE, GRID_H)
    betas  = np.linspace(BETA_CENTER  - BETA_RANGE,  BETA_CENTER  + BETA_RANGE,  GRID_W)

    _grid_meta.update({
        "grid_w":        GRID_W,
        "grid_h":        GRID_H,
        "grid_n":        max(GRID_W, GRID_H),  # legacy compat
        "alpha_range":   ALPHA_RANGE,
        "beta_range":    BETA_RANGE,
        "alpha_center":  ALPHA_CENTER,
        "beta_center":   BETA_CENTER,
        "rank_min":      RANK_MIN,
        "rank_max":      RANK_MAX,
        "rank_tol":      RANK_TOL,
        "restarts":      RESTARTS,
        "alphas":        alphas.tolist(),
        "betas":         betas.tolist(),
        "u_label":       u_label,
        "v_label":       v_label,
    })

    # ---- Resume from DB if available ----------------------------------------
    if not args.fresh:
        prior = _load_done_cells()
        if prior:
            with _lock:
                _cells.update(prior)
                _done_count = len(prior)
            print(f"[RESUME] Loaded {len(prior)} completed cells from {DB_PATH.name}")
        else:
            print("[RESUME] No prior scan found — starting fresh")
    else:
        print("[FRESH] Ignoring any prior scan data")

    # ---- Build pending cell list (skip already done) ------------------------
    all_coords = [(i, j) for i in range(GRID_H) for j in range(GRID_W)]

    # Dyadic (mipmap) scan order: coarsest stride first, then halve each pass.
    # Within each stride level, sort centre-out so T333 resolves early.
    ci = GRID_H // 2
    cj = GRID_W // 2

    def _tz(n: int) -> int:
        """Number of trailing zeros (dyadic level). 0 maps to a large sentinel."""
        if n == 0:
            return 99
        c = 0
        while n % 2 == 0:
            n //= 2
            c += 1
        return c

    def _dyadic_key(p: tuple[int, int]) -> tuple[int, int]:
        i, j = p
        level = min(_tz(i), _tz(j))   # higher = coarser = scanned first
        dist  = (i - ci) ** 2 + (j - cj) ** 2
        return (-level, dist)          # negate so sort puts coarsest first

    all_coords.sort(key=_dyadic_key)

    with _lock:
        pending = [(i, j) for (i, j) in all_coords if (i, j) not in _cells]

    print(f"Pending cells: {len(pending)} of {_total}\n", flush=True)

    _write_state()
    writer = threading.Thread(target=_state_writer, daemon=True)
    writer.start()

    T_flat = T333.ravel().tolist()
    u_flat = u.ravel().tolist()
    v_flat = v.ravel().tolist()

    db_lock = threading.Lock()   # serialise DB writes from result-handler thread

    print(f"Scanning {len(pending)} cells with {POOL_SIZE} workers ...\n", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=POOL_SIZE) as pool:
        future_map: dict[concurrent.futures.Future, tuple[int, int]] = {}

        for (i, j) in pending:
            alpha = float(alphas[i])
            beta  = float(betas[j])
            f = pool.submit(
                _scan_cell,
                i, j, alpha, beta,
                T_flat, u_flat, v_flat,
                RANK_MIN, RANK_MAX,
                RESTARTS, ALS_ITERS, LBFGS_MAXITER, RANK_TOL,
            )
            future_map[f] = (i, j)

        try:
            for future in concurrent.futures.as_completed(future_map):
                if _stop_event.is_set():
                    break
                i, j = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  [ERR ({i:2d},{j:2d})] {exc}", flush=True)
                    result = {
                        "i": i, "j": j,
                        "alpha": float(alphas[i]), "beta": float(betas[j]),
                        "min_rank": None, "best_res": None,
                        "residuals": {}, "status": "error",
                    }

                with _lock:
                    _cells[(i, j)] = result
                    _done_count += 1
                    done = _done_count

                with db_lock:
                    try:
                        _save_cell_to_db(result)
                    except Exception as e:
                        print(f"  [DB ERR] {e}", flush=True)

                mr  = result["min_rank"] if result["min_rank"] is not None else "-"
                br  = result["best_res"]
                br_s = f"{br:.4e}" if br is not None else "inf"
                pct = done * 100 // _total
                print(
                    f"  [{done:4d}/{_total}] ({i:2d},{j:2d})"
                    f"  a={result['alpha']:+.2f}  b={result['beta']:+.2f}"
                    f"  min_rank={mr}  best_res={br_s}"
                    f"  {pct}%",
                    flush=True,
                )

        except KeyboardInterrupt:
            print("\nInterrupted — stopping.", flush=True)
            _stop_event.set()

    _stop_event.set()
    writer.join(timeout=6)
    _write_state()

    # ---- Summary -----------------------------------------------------------
    with _lock:
        done_cells = [c for c in _cells.values() if c.get("status") == "done"]

    rank_counts: dict[int, int] = {}
    no_conv = 0
    for c in done_cells:
        r = c.get("min_rank")
        if r is None:
            no_conv += 1
        else:
            rank_counts[r] = rank_counts.get(r, 0) + 1

    print(f"\n{'='*64}")
    print(f"Scan complete.  {len(done_cells)}/{_total} cells done.")
    for r in sorted(rank_counts):
        bar = "#" * (rank_counts[r] * 40 // max(max(rank_counts.values()), 1))
        print(f"  rank {r:2d}: {rank_counts[r]:5d} cells  {bar}")
    if no_conv:
        print(f"  no-conv: {no_conv:5d} cells")
    print(f"State -> {STATE_PATH}")
    print(f"DB    -> {DB_PATH}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
