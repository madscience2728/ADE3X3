"""
pareto_gate23.py -- Pareto frontier of (delta_leak, sigma_innovation) via single swaps.

Starting from the Gate-2-clean packet (gate2_clean_packet.json), performs ALL
single-term swaps across a random sample of the TermDB and records the Pareto
frontier of (delta_leak, sigma_innovation).

The goal is to measure the exchange rate: how much Gate 2 violation (delta_leak)
does it cost to buy sigma_innovation > 0?

Usage:
    python pareto_gate23.py                    # use saved packet, sample 200K swaps
    python pareto_gate23.py --n-sample 500000  # larger sample
    python pareto_gate23.py --all-positions    # try all 19 positions (default: all)
    python pareto_gate23.py --seed 42          # RNG seed

Output:
    CANON_DATABASE/pareto_gate23.json         -- full Pareto frontier + statistics
    CANON_DATABASE/pareto_gate23_hist.json    -- 2D histogram (delta_leak x sigma_innov)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))
sys.path.insert(0, str(_ROOT / "CANON_DATABASE" / "scripts"))

from term_db import TermDB
from swap_scoring import _rank


def _compute_diag(H, sigma, delta, target_rank_H=10, tol=1e-9):
    H_f = H.astype(np.float64)
    delta_f = delta.astype(np.float64)
    sigma_f = sigma.astype(np.float64)
    N  = np.hstack([H_f, delta_f])
    SN = np.hstack([sigma_f, N])
    rank_H  = _rank(H_f,  tol)
    rank_N  = _rank(N,    tol)
    rank_SN = _rank(SN,   tol)
    return {
        'gate1_gap':         abs(rank_H - target_rank_H),
        'delta_leak':        rank_N - rank_H,
        'sigma_innovation':  rank_SN - rank_N,
        'augmented_gap':     H.shape[0] - rank_SN,
        'rank_H':  rank_H,
        'rank_N':  rank_N,
        'rank_SN': rank_SN,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', default=str(_ROOT / 'CANON_DATABASE' / 'gate2_clean_packet.json'),
                   help='Path to seed packet JSON (default: gate2_clean_packet.json)')
    p.add_argument('--n-sample', type=int, default=200_000,
                   help='Number of random replacement indices to sample per position')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--db', default=str(_ROOT / 'CANON_DATABASE' / 'data'))
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load packet
    with open(args.packet) as f:
        pkt = json.load(f)
    indices = np.array(pkt['indices'], dtype=np.int64)
    R = len(indices)
    print(f"Loaded packet: R={R}, indices shape={indices.shape}")
    print(f"Seed diagnostics: {pkt['diagnostics']}")

    # Load DB
    print(f"Loading TermDB from {args.db} ...")
    t0 = time.time()
    db = TermDB(load_path=args.db, generate=False)
    db.H     = np.load(Path(args.db) / 'H.npy',     mmap_mode='r')
    db.sigma = np.load(Path(args.db) / 'sigma.npy', mmap_mode='r')
    print(f"Loaded in {time.time()-t0:.1f}s")

    N_DB = db.N_PAIRS

    # Base packet arrays
    H_base     = db.H[indices].astype(np.float64)
    sigma_base = db.sigma[indices].astype(np.float64)
    delta_base = db.compute_delta_batch(indices.tolist()).astype(np.float64)

    # Pareto frontier: dict mapping (delta_leak, sigma_innovation) -> best example
    # We want all non-dominated points when minimizing delta_leak and
    # maximizing sigma_innovation.
    # 2D histogram: indexed [delta_leak][sigma_innovation]
    hist: dict[tuple[int, int], int] = {}

    # Pareto set: keep best sigma_innovation for each delta_leak level
    pareto: dict[int, dict] = {}  # delta_leak -> {sigma_innovation, position, new_idx, diag}

    n_positions = R
    n_sample = args.n_sample
    total_swaps = n_positions * n_sample
    print(f"Scanning {n_positions} positions x {n_sample:,} samples = {total_swaps:,} swaps total")

    t_start = time.time()
    evaluated = 0
    gate1_violations = 0

    for pos in range(n_positions):
        # Sample random replacement indices
        new_indices = rng.integers(0, N_DB, size=n_sample, dtype=np.int64)

        for j, nidx in enumerate(new_indices):
            evaluated += 1

            # Build trial H (in-place swap at pos)
            H_trial = H_base.copy()
            H_trial[pos] = db.H[nidx].astype(np.float64)

            # Gate 1 pre-filter
            sv = np.linalg.svd(H_trial, compute_uv=False)
            thresh = 1e-9 * (max(H_trial.shape) * sv[0]) if sv[0] > 0 else 1e-9
            rank_H = int(np.sum(sv > thresh))
            if rank_H != 10:
                gate1_violations += 1
                continue

            sigma_trial = sigma_base.copy()
            sigma_trial[pos] = db.sigma[nidx].astype(np.float64)
            delta_trial = delta_base.copy()
            delta_trial[pos] = db.compute_delta(int(nidx)).astype(np.float64)

            diag = _compute_diag(H_trial, sigma_trial, delta_trial)
            leak = diag['delta_leak']
            si   = diag['sigma_innovation']

            # Update histogram
            key = (leak, si)
            hist[key] = hist.get(key, 0) + 1

            # Update Pareto: for this leak level, keep max sigma_innovation
            if leak not in pareto or si > pareto[leak]['sigma_innovation']:
                pareto[leak] = {
                    'sigma_innovation': si,
                    'delta_leak': leak,
                    'augmented_gap': diag['augmented_gap'],
                    'rank_H': diag['rank_H'],
                    'rank_N': diag['rank_N'],
                    'rank_SN': diag['rank_SN'],
                    'position': pos,
                    'new_db_index': int(nidx),
                    'replaced_db_index': int(indices[pos]),
                }

        elapsed = time.time() - t_start
        rate = evaluated / elapsed if elapsed > 0 else 0
        g1_pass = evaluated - gate1_violations
        print(f"  pos {pos+1:2d}/{n_positions}: evaluated {evaluated:,}  "
              f"gate1_pass={g1_pass:,}  rate={rate/1000:.1f}K/s", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total evaluated: {evaluated:,}, Gate1 violations: {gate1_violations:,}")

    # Print Pareto table
    print("\n=== PARETO FRONTIER (delta_leak vs sigma_innovation) ===")
    print(f"{'delta_leak':>12}  {'sigma_innov':>12}  {'aug_gap':>8}  {'count':>8}")
    for leak in sorted(pareto.keys()):
        p = pareto[leak]
        cnt = sum(v for (lk, si), v in hist.items() if lk == leak)
        print(f"{leak:>12}  {p['sigma_innovation']:>12}  {p['augmented_gap']:>8}  {cnt:>8}")

    # Save results
    out = {
        'seed_packet_file': args.packet,
        'seed_diagnostics': pkt['diagnostics'],
        'n_sample_per_position': n_sample,
        'n_positions': n_positions,
        'total_evaluated': evaluated,
        'gate1_violations': gate1_violations,
        'gate1_pass_rate': (evaluated - gate1_violations) / max(evaluated, 1),
        'elapsed_seconds': elapsed,
        'pareto_frontier': {
            str(leak): p for leak, p in sorted(pareto.items())
        },
    }
    out_path = _ROOT / 'CANON_DATABASE' / 'pareto_gate23.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved Pareto results to {out_path}")

    # Save 2D histogram
    hist_out = {f"{k[0]},{k[1]}": v for k, v in sorted(hist.items())}
    hist_path = _ROOT / 'CANON_DATABASE' / 'pareto_gate23_hist.json'
    with open(hist_path, 'w') as f:
        json.dump(hist_out, f, indent=2)
    print(f"Saved histogram to {hist_path}")


if __name__ == '__main__':
    main()
