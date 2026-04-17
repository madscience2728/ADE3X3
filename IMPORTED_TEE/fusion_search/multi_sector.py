"""fusion_search/multi_sector.py — CLI entry point for the multi-sector A₁ search.

Invoked by serve.py as a subprocess:
    python -m fusion_search.multi_sector [options]

This is intentionally a thin shim — all logic lives in lambda_search.multi_sector_search().
"""
from __future__ import annotations

import argparse
import os
import pathlib

# Cap BLAS threads so worker-spawned numpy calls don't over-subscribe the CPU.
for _env in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, "1")

from fusion_search.lambda_search import (
    ALS_ITERS,
    LBFGS_MAXITER,
    _MS_OUT,
    multi_sector_search,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-sector A₁ symmetry-breaking search for T₃₃₃"
    )
    p.add_argument(
        "--lam", type=float, default=-0.46,
        help="Fixed scalar λ for A₀=A₂=λI (default: -0.46, the scalar-sweep minimum)",
    )
    p.add_argument(
        "--random", type=int, default=500,
        dest="n_random",
        help="Number of random A₁ candidates in Phase 1 (default: 500)",
    )
    p.add_argument(
        "--refine", type=int, default=10,
        dest="n_refine_starts",
        help="Top-K Phase-1 results to refine with Nelder-Mead (default: 10)",
    )
    p.add_argument(
        "--restarts-random", type=int, default=2,
        help="ALS restarts per probe in Phase 1 (default: 2, keep small for speed)",
    )
    p.add_argument(
        "--restarts-refine", type=int, default=4,
        help="ALS restarts per Nelder-Mead objective call (default: 4)",
    )
    p.add_argument(
        "--refine-maxfev", type=int, default=40,
        help="Max Nelder-Mead function evaluations per start (default: 40)",
    )
    p.add_argument(
        "--als-iters", type=int, default=ALS_ITERS,
        help="ALS iterations per restart",
    )
    p.add_argument(
        "--lbfgs-iter", type=int, default=LBFGS_MAXITER,
        help="L-BFGS-B iterations per restart",
    )
    p.add_argument(
        "--out", type=str, default=str(_MS_OUT),
        help="Output JSONL path",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes (default: os.cpu_count())",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    multi_sector_search(
        lam=args.lam,
        n_random=args.n_random,
        n_refine_starts=args.n_refine_starts,
        n_restarts_random=args.restarts_random,
        n_restarts_refine=args.restarts_refine,
        als_iters=args.als_iters,
        lbfgs_maxiter=args.lbfgs_iter,
        refine_maxfev=args.refine_maxfev,
        out=args.out,
        verbose=not args.quiet,
        n_workers=args.workers,
    )
