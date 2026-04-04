"""
swap_main.py — Entry point for the R=19 swap optimizer.

Usage:
    python swap_main.py                          # run for R=19, 24 workers
    python swap_main.py --rank 19                # explicit rank
    python swap_main.py --workers 12             # override worker count
    python swap_main.py --db ./data              # override DB path
    python swap_main.py --time-limit 3600        # stop after 1 hour
    python swap_main.py --stall 2000             # stall threshold
    python swap_main.py --target-delta-leak 0    # stop when leak == 0

Notes:
    - Set --workers to the number of physical cores you want to use.
    - 24 workers on a 12-core/24-thread CPU is recommended.
    - TermDB is loaded read-only per worker (OS shares pages, ~12 GB total).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="ADE3x3 Phase 4 — Full-Packet Swap Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rank", type=int, default=19,
                   help="Target rank (default 19)")
    p.add_argument("--workers", type=int, default=0,
                   help="Number of worker processes (default: cpu_count)")
    p.add_argument("--db", type=str, default="",
                   help="Path to TermDB data directory")
    p.add_argument("--time-limit", type=int, default=0,
                   help="Stop after N seconds (0 = no limit)")
    p.add_argument("--stall", type=int, default=2000,
                   help="Non-improving swaps before restart")
    p.add_argument("--target-delta-leak", type=int, default=0,
                   help="Stop when global best delta_leak reaches this value")
    p.add_argument("--hit-bias", type=float, default=0.8,
                   help="Fraction of swaps sampled from hit pool (default 0.8)")
    return p.parse_args()


def main():
    # Must guard with __name__ == '__main__' for Windows multiprocessing
    args = _parse_args()

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))
    sys.path.insert(0, str(_ROOT / "CANON_DATABASE" / "scripts"))

    from swap_config    import SwapConfig
    from swap_optimizer import run

    # Figure out worker count
    n_workers = args.workers
    if n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)

    # Map rank to target_rank_H
    target_map = {13: 4, 19: 10, 20: 11}
    R = args.rank
    if R not in target_map:
        print(f"Warning: rank {R} not in known map {{13:4, 19:10, 20:11}}. "
              f"Using target_rank_H = R - 9 = {R - 9}.")
    target_H = target_map.get(R, R - 9)

    db_path = args.db or str(_ROOT / "CANON_DATABASE" / "data")

    config = SwapConfig(
        R=R,
        target_rank_H=target_H,
        n_workers=n_workers,
        stall_threshold=args.stall,
        time_limit_seconds=args.time_limit,
        target_delta_leak=args.target_delta_leak,
        hit_pool_bias=args.hit_bias,
        db_path=db_path,
    )

    run(config)


if __name__ == "__main__":
    main()
