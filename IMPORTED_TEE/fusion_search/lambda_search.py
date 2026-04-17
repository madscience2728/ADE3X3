"""lambda_search.py — Line search over λ ∈ ℝ for the T₃₃₃ rank landscape.

Phase 1 — coarse grid: λ ∈ [-5, 5] step 0.1  (101 points)
Phase 2 — refinement:  ±0.5 window around the top-K minima, step 0.01

Each evaluated point is appended to results.jsonl immediately so the search
is resumable.  Already-evaluated λ values (within a small tolerance) are
skipped on restart.

Parallelism: all λ values in a phase are evaluated simultaneously via
ProcessPoolExecutor(max_workers=os.cpu_count()).  Each worker handles exactly
one λ and runs its full restart budget independently.  With 16 cores and 64
points the wall time equals roughly one serial evaluation.

Ctrl+C / Stop (from serve.py) shutdown pattern — mirrors main.py:
  1. Workers suppress SIGINT/SIGBREAK so they cannot be interrupted mid-scipy.
  2. The main process catches KeyboardInterrupt, cancels pending futures, calls
     shutdown(wait=False) on the executor, then explicitly terminates each
     worker process.  The main process exits within ~1 s regardless of how
     long the current scipy minimisation would otherwise take.

Usage:
    python -m fusion_search.lambda_search [--grid-step 0.1] [--refine-top 5]
            [--restarts 20] [--out fusion_search/results.jsonl]
            [--workers N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import signal
import time
from typing import Optional

import numpy as np

from fusion_search.rank_probe import ALS_ITERS, LBFGS_MAXITER, RANK_RANGE, probe

# ---------------------------------------------------------------------------
# BLAS thread cap (same as bini_slime) — set before numpy/scipy load
# ---------------------------------------------------------------------------
for _env in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, "1")

_HERE = pathlib.Path(__file__).parent
DEFAULT_OUT = _HERE / "results.jsonl"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _load_done(path: pathlib.Path) -> dict[float, dict]:
    """Load previously evaluated λ values → {rounded_lam: record}."""
    done: dict[float, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[round(rec["lambda"], 6)] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Parallel worker helpers  (must be top-level for pickling under spawn)
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    """Called once in each worker process at startup.

    Suppress SIGINT and SIGBREAK (Windows) so that Ctrl+C or CTRL_BREAK_EVENT
    from the parent / serve.py does NOT interrupt workers mid-scipy.  Shutdown
    is driven by the main process: it calls executor.shutdown(wait=False) and
    then explicitly terminates each worker via proc.terminate().  This matches
    the terminate-then-kill pattern used in main.py's _terminate_process().
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):   # Windows only
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


def _probe_one(args: tuple) -> tuple:
    """Worker entry point — evaluate a single λ value.

    Returns (lam, residual, rank_estimate, solved, elapsed_s, per_rank, cond_num) as
    plain Python scalars/dicts so the tuple is always picklable back to the
    main process.
    """
    lam, n_restarts, als_iters, lbfgs_maxiter, tol = args
    result = probe(
        lam,
        ranks=RANK_RANGE,
        n_restarts=n_restarts,
        als_iters=als_iters,
        lbfgs_maxiter=lbfgs_maxiter,
        tol=tol,
        verbose=False,
    )
    return (
        round(float(lam), 8),
        float(result.residual),
        int(result.rank_estimate),
        bool(result.solved),
        float(result.timestamp),
        {int(k): float(v) for k, v in (result.per_rank or {}).items()},
        float(result.cond_num),
    )


def _shutdown_executor(
    executor: concurrent.futures.ProcessPoolExecutor,
    future_map: dict,
) -> None:
    """Cancel pending futures, shut down the executor without waiting, then
    force-terminate any worker process still alive.

    Mirrors main.py's _terminate_process(): signal first, don't wait —
    so the process tree is clean within ~1 s instead of waiting up to 30 s
    for scipy L-BFGS-B to complete its current C-level call.
    """
    for f in future_map:
        f.cancel()
    executor.shutdown(cancel_futures=True, wait=False)
    # executor._processes is {pid: Process} on CPython; guard with getattr
    # so this degrades gracefully on other implementations.
    for wp in getattr(executor, "_processes", {}).values():
        try:
            if wp.is_alive():
                wp.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parallel batch evaluation
# ---------------------------------------------------------------------------

def _eval_batch(
    lambdas: list[float],
    done: dict[float, dict],
    out_path: pathlib.Path,
    n_restarts: int,
    als_iters: int,
    lbfgs_maxiter: int,
    tol: float = 1e-7,
    verbose: bool = True,
    n_workers: int | None = None,
) -> list[dict]:
    """Evaluate *lambdas* in parallel using ProcessPoolExecutor.

    Already-cached points (present in *done*) are skipped.  Results are
    written to *out_path* as they arrive so the file is always up-to-date
    for the UI.  Returns all records (cached + new), sorted by λ.

    Raises KeyboardInterrupt (after clean shutdown) if the user interrupts.
    """
    pending = [lam for lam in lambdas if round(lam, 6) not in done]
    cached  = [done[round(lam, 6)] for lam in lambdas if round(lam, 6) in done]

    if not pending:
        if verbose and cached:
            print(f"  All {len(cached)} point(s) already cached — nothing to do.", flush=True)
        return sorted(cached, key=lambda r: r["lambda"])

    n_w = n_workers if n_workers is not None else (os.cpu_count() or 1)
    if verbose:
        print(
            f"  {len(pending)} new  +  {len(cached)} cached"
            f"  │  {n_w} workers  │  {n_restarts} restarts/point",
            flush=True,
        )

    args_list = [
        (lam, n_restarts, als_iters, lbfgs_maxiter, tol)
        for lam in pending
    ]
    new_records: list[dict] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_w, initializer=_worker_init,
    ) as executor:
        future_map: dict[concurrent.futures.Future, float] = {
            executor.submit(_probe_one, a): a[0] for a in args_list
        }
        try:
            for future in concurrent.futures.as_completed(future_map):
                lam = future_map[future]
                try:
                    lam_r, residual, rank_estimate, solved, elapsed_s, per_rank, cond_num = future.result()
                except Exception as exc:
                    print(f"  λ={lam:+.4f}  ERROR: {exc}", flush=True)
                    continue
                record = {
                    "lambda":        lam_r,
                    "residual":      residual,
                    "rank_estimate": rank_estimate,
                    "solved":        solved,
                    "elapsed_s":     elapsed_s,
                    "per_rank":      per_rank,
                    "cond_num":      cond_num,
                    "timestamp":     float(time.time()),
                }
                if verbose:
                    tag = f"  ***SOLVED rank={rank_estimate}***" if solved else ""
                    print(f"  λ={lam_r:+7.4f}  res={residual:.4e}{tag}", flush=True)
                _append(out_path, record)
                done[round(lam_r, 6)] = record
                new_records.append(record)

        except KeyboardInterrupt:
            print("\nInterrupted — shutting down workers …", flush=True)
            _shutdown_executor(executor, future_map)
            raise   # propagate so run() can write a partial summary and exit

    return sorted(cached + new_records, key=lambda r: r["lambda"])


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def _print_summary(done: dict[float, dict], top_n: int = 10) -> None:
    all_eval = sorted(done.values(), key=lambda r: r["residual"])
    print("\n" + "=" * 60)
    print(f"Top-{top_n} λ values by residual:")
    for r in all_eval[:top_n]:
        rank_str = str(r["rank_estimate"]) if r.get("solved") else "—"
        print(
            f"  λ={r['lambda']:+8.4f}  res={r['residual']:.4e}"
            f"  rank={rank_str}"
        )
    print("=" * 60)


def run(
    grid_lo: float = -5.0,
    grid_hi: float = 5.0,
    grid_step: float = 0.1,
    refine_top: int = 5,
    refine_step: float = 0.01,
    refine_half_window: float = 0.5,
    n_restarts: int = 20,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    out: str = str(DEFAULT_OUT),
    verbose: bool = True,
    n_workers: int | None = None,
) -> list[dict]:
    out_path = pathlib.Path(out)
    done = _load_done(out_path)
    n_w = n_workers if n_workers is not None else (os.cpu_count() or 1)

    if verbose:
        print("=" * 60)
        print("fusion_search  λ line-search  [parallel]")
        print(f"  output   : {out_path}")
        print(f"  grid     : [{grid_lo}, {grid_hi}] step={grid_step}")
        print(f"  restarts : {n_restarts}/point")
        print(f"  workers  : {n_w}")
        print(f"  cached   : {len(done)} previously evaluated points")
        print("=" * 60)

    # -------- Phase 1: coarse grid — all points in parallel --------
    lambdas_coarse = [
        float(lam)
        for lam in np.arange(grid_lo, grid_hi + grid_step * 0.5, grid_step)
    ]

    if verbose:
        print(f"\n[Phase 1] Coarse grid  ({len(lambdas_coarse)} points, {n_w} workers)")

    try:
        records_coarse = _eval_batch(
            lambdas_coarse, done, out_path,
            n_restarts, als_iters, lbfgs_maxiter,
            verbose=verbose, n_workers=n_w,
        )
    except KeyboardInterrupt:
        if verbose:
            print("Stopped during Phase 1 — partial results saved.", flush=True)
            _print_summary(done)
        return list(done.values())

    # -------- Pick top-K by lowest residual (skip λ=0) --------
    candidates = sorted(
        [r for r in records_coarse if abs(r["lambda"]) > 1e-6],
        key=lambda r: r["residual"],
    )
    top_k = candidates[:refine_top]

    if verbose:
        print(f"\n[Phase 2] Refining top-{refine_top} minima")
        for r in top_k:
            print(
                f"  λ={r['lambda']:+.4f}  res={r['residual']:.4e}"
                f"  rank={r['rank_estimate']}"
            )

    # -------- Phase 2: fine refinement — each window in parallel --------
    try:
        for rec in top_k:
            center = rec["lambda"]
            lo = center - refine_half_window
            hi = center + refine_half_window
            lambdas_fine = [
                float(lam)
                for lam in np.arange(lo, hi + refine_step * 0.5, refine_step)
            ]
            if verbose:
                print(
                    f"\n  Refining around λ={center:+.4f}"
                    f"  ({len(lambdas_fine)} points)"
                )
            _eval_batch(
                lambdas_fine, done, out_path,
                n_restarts, als_iters, lbfgs_maxiter,
                verbose=verbose, n_workers=n_w,
            )
    except KeyboardInterrupt:
        if verbose:
            print("Stopped during Phase 2 — partial results saved.", flush=True)

    # -------- Summary --------
    if verbose:
        _print_summary(done)

    return list(done.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="λ line-search for T₃₃₃ rank  [parallel]")
    p.add_argument("--grid-lo",    type=float, default=-5.0)
    p.add_argument("--grid-hi",    type=float, default=5.0)
    p.add_argument("--grid-step",  type=float, default=0.1)
    p.add_argument("--refine-top", type=int,   default=5)
    p.add_argument("--refine-step",type=float, default=0.01)
    p.add_argument("--restarts",   type=int,   default=20,
                   help="Random restarts per λ value (default: 20; each worker "
                        "gets the full budget simultaneously)")
    p.add_argument("--workers",    type=int,   default=None,
                   help="Worker processes (default: os.cpu_count())")
    p.add_argument("--als-iters",  type=int,   default=ALS_ITERS)
    p.add_argument("--lbfgs-iter", type=int,   default=LBFGS_MAXITER)
    p.add_argument("--out",        type=str,   default=str(DEFAULT_OUT))
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        grid_lo=args.grid_lo,
        grid_hi=args.grid_hi,
        grid_step=args.grid_step,
        refine_top=args.refine_top,
        refine_step=args.refine_step,
        n_restarts=args.restarts,
        als_iters=args.als_iters,
        lbfgs_maxiter=args.lbfgs_iter,
        out=args.out,
        verbose=not args.quiet,
        n_workers=args.workers,
    )


# ---------------------------------------------------------------------------
# Multi-sector symmetry-breaking search  (additions below; nothing above
# this line is modified)
# ---------------------------------------------------------------------------

from fusion_search.fusion_state import MultiSectorFusionState  # noqa: E402
from fusion_search.rank_probe import FLOOR_RESIDUAL, probe_state  # noqa: E402

_MS_DATA_DIR = _HERE / "data"
_MS_OUT      = _MS_DATA_DIR / "multi_sector_sweep.jsonl"
_MS_STATUS   = _MS_DATA_DIR / "multi_sector_status.json"


# ---- Top-level workers (must be module-level for ProcessPoolExecutor pickling) ----

def _probe_ms_random(args: tuple) -> tuple:
    """Worker: evaluate one candidate A₁ matrix.

    Returns (A1_flat, residual, rank_estimate, solved, cond_num, elapsed_s, per_rank).
    """
    lam, A1_flat, n_restarts, als_iters, lbfgs_maxiter, tol = args
    A1    = np.array(A1_flat, dtype=np.float64).reshape(2, 2)
    state = MultiSectorFusionState(lam, A1)
    res   = probe_state(
        state,
        ranks=RANK_RANGE,
        n_restarts=n_restarts,
        als_iters=als_iters,
        lbfgs_maxiter=lbfgs_maxiter,
        tol=tol,
        verbose=False,
    )
    return (
        list(A1_flat),
        float(res.residual),
        int(res.rank_estimate),
        bool(res.solved),
        float(res.cond_num),
        float(res.timestamp),
        {int(k): float(v) for k, v in res.per_rank.items()},
    )


def _probe_ms_refine(args: tuple) -> tuple:
    """Worker: run Nelder-Mead refinement starting from A₁_flat_init.

    Runs the full optimisation loop inside the worker so the
    ProcessPoolExecutor can push one future per starting point (up to
    n_refine_starts futures in parallel, one per core).

    Returns (best_A1_flat, best_residual, n_calls).
    """
    lam, A1_flat_init, n_restarts, als_iters, lbfgs_maxiter, tol, maxfev = args
    from scipy.optimize import minimize as _minimize  # local — already a dep

    best_res = [float("inf")]
    best_A1  = [list(A1_flat_init)]
    n_calls  = [0]

    def _obj(A1_flat: np.ndarray) -> float:
        A1    = np.asarray(A1_flat, dtype=np.float64).reshape(2, 2)
        state = MultiSectorFusionState(lam, A1)
        r     = probe_state(
            state,
            ranks=RANK_RANGE,
            n_restarts=n_restarts,
            als_iters=als_iters,
            lbfgs_maxiter=lbfgs_maxiter,
            tol=tol,
            verbose=False,
        )
        n_calls[0] += 1
        if r.residual < best_res[0]:
            best_res[0] = r.residual
            best_A1[0]  = A1_flat.tolist()
        return r.residual

    _minimize(
        _obj,
        np.array(A1_flat_init, dtype=np.float64),
        method="Nelder-Mead",
        options={
            "maxfev":   maxfev,
            "xatol":    1e-4,
            "fatol":    1e-6,
            "adaptive": True,
        },
    )
    return best_A1[0], float(best_res[0]), int(n_calls[0])


# ---- Main search entry point ----

def multi_sector_search(
    lam: float = -0.46,
    n_random: int = 500,
    n_refine_starts: int = 10,
    n_restarts_random: int = 2,
    n_restarts_refine: int = 4,
    als_iters: int = ALS_ITERS,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    refine_maxfev: int = 40,
    tol: float = 1e-7,
    out: str = str(_MS_OUT),
    verbose: bool = True,
    n_workers: Optional[int] = None,
) -> list[dict]:
    """Symmetry-breaking search over the free 2×2 A₁ matrix.

    Fixes A₀ = A₂ = λI at the scalar-sweep minimum (default λ=-0.46) and
    searches the 4-dimensional space of A₁ for configurations that break the
    Z₃-symmetric structural floor at residual ≈ 1.76.

    Phase 1 — random:     sample ``n_random`` A₁ matrices from N(0,1)^4,
                          evaluate all in parallel via ProcessPoolExecutor.
    Phase 2 — refinement: take the top ``n_refine_starts`` candidates by
                          residual; run Nelder-Mead inside each worker so
                          the refinements execute in parallel (one core each).

    Results are written to ``fusion_search/data/multi_sector_sweep.jsonl``
    (one JSON record per line, same format as results.jsonl) and a summary
    is written to ``fusion_search/data/multi_sector_status.json``.

    Signal: any record with ``"broke_floor": true`` means residual < 1.76 —
    the Z₃ manifold has been escaped.

    Parameters
    ----------
    lam : float
        Fixed scalar λ; determines A₀ = A₂ = λI.  Use the scalar-sweep
        minimum (default −0.46 from the 1.76-floor landscape).
    n_random : int
        Random A₁ candidates in Phase 1.
    n_refine_starts : int
        How many top Phase-1 results to refine with Nelder-Mead.
    n_restarts_random : int
        ALS random restarts per probe in Phase 1 (keep small — speed matters).
    n_restarts_refine : int
        ALS random restarts per probe inside the Nelder-Mead objective.
    refine_maxfev : int
        Maximum Nelder-Mead function evaluations per refinement worker.
    """
    out_path    = pathlib.Path(out)
    status_path = pathlib.Path(_MS_STATUS)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_w = n_workers if n_workers is not None else (os.cpu_count() or 1)
    all_records: list[dict] = []

    if verbose:
        print("=" * 60)
        print("multi_sector_search  [A₁ symmetry-breaking]")
        print(f"  λ (fixed)     : {lam}  →  A₀ = A₂ = λI")
        print(f"  random trials : {n_random}")
        print(f"  refine starts : {n_refine_starts}")
        print(f"  workers       : {n_w}")
        print(f"  floor         : {FLOOR_RESIDUAL}  (Z₃ structural floor)")
        print(f"  out           : {out_path}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1 — parallel random search over A₁ ∈ N(0,1)^4
    # ------------------------------------------------------------------
    rng_p1   = np.random.default_rng(seed=42)
    A1_batch = rng_p1.standard_normal((n_random, 4)).tolist()

    args_list = [
        (lam, A1_flat, n_restarts_random, als_iters, lbfgs_maxiter, tol)
        for A1_flat in A1_batch
    ]

    if verbose:
        print(
            f"\n[Phase 1] Random A₁  ({n_random} candidates,"
            f" {n_restarts_random} restarts/probe, {n_w} workers)"
        )

    phase1_results: list[tuple] = []   # (A1_flat, residual, rank, solved)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_w, initializer=_worker_init,
    ) as executor:
        future_map = {executor.submit(_probe_ms_random, a): a for a in args_list}
        try:
            for future in concurrent.futures.as_completed(future_map):
                try:
                    A1_flat, residual, rank_est, solved, cond_n, elapsed_s, per_rank = (
                        future.result()
                    )
                except Exception as exc:
                    print(f"  Worker error: {exc}", flush=True)
                    continue

                broke = residual < FLOOR_RESIDUAL
                record = {
                    "phase":         "random",
                    "lambda":        float(lam),
                    "A1":            [A1_flat[:2], A1_flat[2:]],
                    "A1_flat":       A1_flat,
                    "residual":      residual,
                    "rank_estimate": rank_est,
                    "solved":        solved,
                    "cond_num":      cond_n,
                    "elapsed_s":     elapsed_s,
                    "per_rank":      per_rank,
                    "broke_floor":   broke,
                    "timestamp":     float(time.time()),
                }
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

                phase1_results.append((A1_flat, residual, rank_est, solved))
                all_records.append(record)

                if verbose and broke:
                    print(
                        f"  *** BROKE FLOOR ***  res={residual:.6f}"
                        f"  A1={[round(v,4) for v in A1_flat]}",
                        flush=True,
                    )

        except KeyboardInterrupt:
            _shutdown_executor(executor, future_map)
            if verbose:
                print("  Interrupted during Phase 1.", flush=True)

    phase1_results.sort(key=lambda x: x[1])
    top_k = phase1_results[:n_refine_starts]
    best_p1_res = phase1_results[0][1] if phase1_results else float("inf")

    if verbose:
        print(f"\n  Phase 1 best residual : {best_p1_res:.6f}")
        print(f"  Below floor ({FLOOR_RESIDUAL})     : {best_p1_res < FLOOR_RESIDUAL}")

    # ------------------------------------------------------------------
    # Phase 2 — Nelder-Mead refinement of top-K (one worker per start)
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"\n[Phase 2] Nelder-Mead refinement"
            f"  ({len(top_k)} starts, maxfev={refine_maxfev},"
            f" {n_restarts_refine} restarts/probe)"
        )

    refine_args = [
        (lam, A1_flat, n_restarts_refine, als_iters, lbfgs_maxiter, tol, refine_maxfev)
        for A1_flat, *_ in top_k
    ]

    phase2_results: list[tuple] = []   # (A1_flat, residual)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(n_w, max(len(top_k), 1)), initializer=_worker_init,
    ) as executor:
        future_map = {executor.submit(_probe_ms_refine, a): a for a in refine_args}
        try:
            for future in concurrent.futures.as_completed(future_map):
                try:
                    best_A1_flat, best_res, n_calls = future.result()
                except Exception as exc:
                    print(f"  Refine worker error: {exc}", flush=True)
                    continue

                broke = best_res < FLOOR_RESIDUAL
                record = {
                    "phase":              "refine",
                    "lambda":             float(lam),
                    "A1":                 [best_A1_flat[:2], best_A1_flat[2:]],
                    "A1_flat":            best_A1_flat,
                    "residual":           best_res,
                    "rank_estimate":      -1,
                    "solved":             False,
                    "cond_num":           None,
                    "elapsed_s":          None,
                    "per_rank":           {},
                    "broke_floor":        broke,
                    "nelder_mead_calls":  n_calls,
                    "timestamp":          float(time.time()),
                }
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

                phase2_results.append((best_A1_flat, best_res))
                all_records.append(record)

                if verbose:
                    floor_tag = "  *** BROKE FLOOR ***" if broke else ""
                    print(
                        f"  refine  res={best_res:.6f}  calls={n_calls}{floor_tag}",
                        flush=True,
                    )

        except KeyboardInterrupt:
            _shutdown_executor(executor, future_map)
            if verbose:
                print("  Interrupted during Phase 2.", flush=True)

    # ------------------------------------------------------------------
    # Summary + status file
    # ------------------------------------------------------------------
    sortable = [
        (r["A1_flat"], r["residual"])
        for r in all_records
        if "residual" in r and r["residual"] is not None
    ]
    sortable.sort(key=lambda x: x[1])

    best_A1_flat = sortable[0][0] if sortable else None
    best_residual = sortable[0][1] if sortable else float("inf")
    best_A1_2x2   = [best_A1_flat[:2], best_A1_flat[2:]] if best_A1_flat else None
    broke_floor   = best_residual < FLOOR_RESIDUAL

    status = {
        "fusion": {
            "best_lambda":      lam,
            "best_residual":    best_residual,
            "best_A1":          best_A1_2x2,
            "broke_floor":      broke_floor,
            "points_evaluated": len(all_records),
            "mode":             "multi_sector",
        }
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Overall best residual : {best_residual:.6f}")
        print(f"  Best A₁               : {best_A1_2x2}")
        print(f"  Broke Z₃ floor (1.76) : {broke_floor}")
        print(f"  Total evaluations     : {len(all_records)}")
        print(f"  Status written to     : {status_path}")
        print("=" * 60)

    return all_records
