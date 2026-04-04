"""
Gate 2 Depth Profiler
=====================

Runs N assembly trials on a single packet using the Gate 2-aware assembler
and prints the neutral-pool size at each depth (0 through 8).

This answers: "how fast does the neutral pool shrink per dependent added?"
A halving pattern → tractable. A 10x-collapse at one depth → obstruction there.

Usage:
    python CANON_DATABASE/diag_depth_profile.py               # first R=19 packet, 50 trials
    python CANON_DATABASE/diag_depth_profile.py --rank 19 --trials 200
    python CANON_DATABASE/diag_depth_profile.py --packet 5    # packet index
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from term_db import TermDB
from assembler import DependentAssembler, compute_nd_rows_batch, compute_sn_rows_batch, \
    _row_space_projector, _residual_norms, _NEUTRAL_TOL

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


class _DepthConfig:
    max_assembly_candidates = 500_000
    innovation_threshold    = 1e-8
    max_solutions_per_rank  = 999
    n_assembly_trials       = 1       # we drive the loop externally
    trial_top_k             = 64
    gate3_residual_tol      = 1e-10
    rank_tol                = 1e-10


def profile_packet(db, basis_indices, hit_indices, rank, n_trials=50):
    """
    Run n_trials Gate-2-aware greedy assemblies on one packet.
    Returns list of depth profiles (one per trial).
    """
    config = _DepthConfig()
    config.n_assembly_trials = n_trials

    assembler = DependentAssembler(db=db, config=config)
    assembler.assemble(rank=rank, basis_indices=list(basis_indices), hit_indices=hit_indices)
    return assembler.stats.depth_profiles, assembler.stats.configs_tested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=19)
    parser.add_argument("--packet", type=int, default=0, help="packet index (0-based)")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--db", default="CANON_DATABASE/data")
    parser.add_argument("--packets", default="CANON_DATABASE/packets")
    args = parser.parse_args()

    t0 = time.time()
    db = TermDB(args.db)

    files = sorted(Path(args.packets).glob(f"R{args.rank}/*.npz"))
    if not files:
        print(f"No packets found for R={args.rank} in {args.packets}")
        sys.exit(1)

    fpath = files[args.packet]
    d = np.load(str(fpath), allow_pickle=True)
    basis_indices = d["basis_record_indices"].astype(np.int64)
    hit_indices   = d["hit_record_indices"].astype(np.int64)
    rank          = int(d["rank"][0])

    n_dep = rank - len(basis_indices)
    print(f"\nPacket: {fpath.name}")
    print(f"Rank={rank}, basis={len(basis_indices)}, target dependents={n_dep}, hit pool={len(hit_indices):,}")

    # Pre-scan neutral pool size once
    ND_basis = compute_nd_rows_batch(db, basis_indices)
    Q_N, _   = _row_space_projector(ND_basis, tol=_NEUTRAL_TOL)
    QQt_N    = Q_N.T @ Q_N if Q_N.size > 0 else np.zeros((72, 72))

    # Process in chunks to handle large pools
    n_neutral = 0
    chunk = 50_000
    for start in range(0, len(hit_indices), chunk):
        end = min(start + chunk, len(hit_indices))
        ND_chunk = compute_nd_rows_batch(db, hit_indices[start:end])
        resid = _residual_norms(ND_chunk, QQt_N)
        n_neutral += int(np.sum(resid < _NEUTRAL_TOL * 100))

    basis_H   = db.H[basis_indices].astype(np.float64)
    delta_b   = db.compute_delta_batch(basis_indices).astype(np.float64)
    N_basis   = np.hstack([basis_H, delta_b])
    rank_H    = int(np.linalg.matrix_rank(basis_H))
    rank_N    = int(np.linalg.matrix_rank(N_basis))

    print(f"Basis: rank(H)={rank_H}, rank(N=[H|D])={rank_N}, delta_leak={rank_N - rank_H}")
    print(f"Neutral hits (initial): {n_neutral:,} / {len(hit_indices):,} "
          f"({100*n_neutral/len(hit_indices):.1f}%)")
    print(f"\nRunning {args.trials} trials...")

    profiles, n_complete = profile_packet(db, basis_indices, hit_indices, rank, args.trials)

    if not profiles:
        print("No trials completed (0 neutral hits in pool?)")
        return

    print(f"Trials that reached full depth: {n_complete}")
    print(f"Trials that died early: {len(profiles) - n_complete}")
    print()

    # Aggregate depth table
    max_depth = max(len(p) for p in profiles)

    if HAS_RICH and console:
        t = Table(title=f"Neutral Pool Size per Depth — R={rank}, {len(profiles)} trials",
                  box=box.SIMPLE_HEAVY)
        t.add_column("Depth", justify="right")
        t.add_column("min",    justify="right")
        t.add_column("median", justify="right")
        t.add_column("max",    justify="right")
        t.add_column("dead (=0)", justify="right")
        t.add_column("ratio vs prev", justify="right")
        prev_med = None
        for depth in range(max_depth):
            vals = [p[depth] for p in profiles if len(p) > depth]
            if not vals:
                continue
            med = float(np.median(vals))
            ratio = f"{med/prev_med:.2f}×" if prev_med and prev_med > 0 else "—"
            prev_med = med
            dead = sum(1 for v in vals if v == 0)
            t.add_row(
                str(depth),
                str(min(vals)),
                str(int(med)),
                str(max(vals)),
                f"[red]{dead}[/red]" if dead > 0 else "0",
                ratio,
            )
        console.print(t)
    else:
        print(f"{'Depth':>6} {'min':>8} {'median':>8} {'max':>8} {'dead':>6} {'ratio':>8}")
        prev_med = None
        for depth in range(max_depth):
            vals = [p[depth] for p in profiles if len(p) > depth]
            if not vals:
                continue
            med = float(np.median(vals))
            ratio = f"{med/prev_med:.2f}x" if prev_med and prev_med > 0 else "—"
            prev_med = med
            dead = sum(1 for v in vals if v == 0)
            print(f"{depth:>6} {min(vals):>8} {int(med):>8} {max(vals):>8} {dead:>6} {ratio:>8}")

    print(f"\nTime: {time.time()-t0:.1f}s")

    # Key interpretation
    if n_complete > 0:
    print(f"\n-> {n_complete}/{args.trials} trials assembled all {n_dep} dependents.")
        print(  "   Gate 3 check ran on these -- see gate_check output above.")
    else:
        # Find where most trials die
        death_depths = [len(p) - 1 for p in profiles if profiles]
        if death_depths:
            from collections import Counter
            c = Counter(death_depths)
            most_common = c.most_common(3)
            print(f"\n-> Most trials die at depth: {most_common}")
            print(  "   This depth is where the neutral/innovative pools do not overlap.")


if __name__ == "__main__":
    main()
