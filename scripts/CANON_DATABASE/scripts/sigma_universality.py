"""
sigma_universality.py -- Test whether sigma_innovation=0 is universal across ALL
Gate-1-satisfying packets, or specific to one basin.

If sigma_innovation=0 holds universally, then sigma in col(H) is an algebraic
identity for {-1,0,1} factorizations with rank(H)=target, and R is provably
impossible over {-1,0,1}.

Usage:
    python sigma_universality.py --rank 13
    python sigma_universality.py --rank 20 --n-packets 200 --workers 8
    python sigma_universality.py --rank 19 --n-packets 100 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))
sys.path.insert(0, str(_ROOT / "CANON_DATABASE" / "scripts"))

from term_db import TermDB
from swap_scoring import _rank


def _build_random_basis(H_mmap, rng, target_rank_H, max_tries=10000):
    """Pick target_rank_H linearly-independent row indices from H_mmap."""
    N_db = H_mmap.shape[0]
    basis_idx = []
    H_basis = np.empty((0, H_mmap.shape[1]), dtype=np.float64)
    attempts = 0
    while len(basis_idx) < target_rank_H and attempts < max_tries:
        attempts += 1
        i = int(rng.integers(0, N_db))
        h = H_mmap[i].astype(np.float64).reshape(1, -1)
        trial = np.vstack([H_basis, h]) if H_basis.shape[0] > 0 else h
        sv = np.linalg.svd(trial, compute_uv=False)
        thresh = 1e-9 * max(trial.shape) * sv[0] if sv[0] > 0 else 1e-9
        if np.sum(sv > thresh) == len(basis_idx) + 1:
            basis_idx.append(i)
            H_basis = trial
    return (np.array(basis_idx, dtype=np.int64), H_basis) if len(basis_idx) == target_rank_H else (None, None)


# Worker state: set by _worker_init
_H_mmap = None
_sigma_mmap = None
_db = None


def _worker_init(db_path: str):
    global _db, _H_mmap, _sigma_mmap
    _db = TermDB(load_path=db_path, generate=False)
    _H_mmap     = np.load(Path(db_path) / 'H.npy',     mmap_mode='r')
    _sigma_mmap = np.load(Path(db_path) / 'sigma.npy', mmap_mode='r')
    _db.H     = _H_mmap
    _db.sigma = _sigma_mmap


def _test_packet_from_hits(args_tuple):
    """
    Worker task: given a pre-computed hit_indices array (records in a subspace),
    build packets by sampling from it — no new DB scans.
    """
    worker_seed, R, target_rank_H, hit_indices = args_tuple
    rng = np.random.default_rng(worker_seed)
    db = _db
    H_mmap = _H_mmap

    n_dep = R - target_rank_H
    n_hits = len(hit_indices)
    if n_hits < R:
        return None

    # Build independent basis from within hit_indices
    basis_pos = []
    H_basis = np.empty((0, H_mmap.shape[1]), dtype=np.float64)
    tried = set()
    for _ in range(10000):
        if len(basis_pos) == target_rank_H:
            break
        p = int(rng.integers(0, n_hits))
        if p in tried:
            continue
        tried.add(p)
        h = H_mmap[hit_indices[p]].astype(np.float64).reshape(1, -1)
        trial = np.vstack([H_basis, h]) if H_basis.shape[0] > 0 else h
        sv = np.linalg.svd(trial, compute_uv=False)
        thresh = 1e-9 * max(trial.shape) * sv[0] if sv[0] > 0 else 1e-9
        if np.sum(sv > thresh) == len(basis_pos) + 1:
            basis_pos.append(p)
            H_basis = trial

    if len(basis_pos) < target_rank_H:
        return None

    # Fill dependents from remaining hit positions
    basis_set = set(basis_pos)
    dep_pos = []
    for _ in range(50000):
        if len(dep_pos) == n_dep:
            break
        p = int(rng.integers(0, n_hits))
        if p not in basis_set and p not in dep_pos:
            dep_pos.append(p)

    if len(dep_pos) < n_dep:
        return None

    all_pos = basis_pos + dep_pos
    idx = hit_indices[all_pos]

    H     = H_mmap[idx].astype(np.float64)
    sigma = _sigma_mmap[idx].astype(np.float64)
    delta = db.compute_delta_batch(idx.tolist()).astype(np.float64)

    N   = np.hstack([H, delta])
    SN  = np.hstack([sigma, N])
    rH  = _rank(H,  1e-9)
    rN  = _rank(N,  1e-9)
    rSN = _rank(SN, 1e-9)
    return int(rH), int(rN - rH), int(rSN - rN), idx.tolist()

_TARGET_MAP = {13: 4, 19: 10, 20: 11}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank', type=int, default=19)
    p.add_argument('--n-packets', type=int, default=100)
    p.add_argument('--workers', type=int, default=0,
                   help='Worker processes (default: cpu_count - 1)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='')
    args = p.parse_args()

    R = args.rank
    target_rank_H = _TARGET_MAP.get(R, R - 9)
    n_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 1)
    out_path = Path(args.out) if args.out else _ROOT / 'CANON_DATABASE' / f'sigma_universality_R{R}.json'
    db_path  = str(_ROOT / 'CANON_DATABASE' / 'data')

    print(f'Rank R={R}, target_rank_H={target_rank_H}, workers={n_workers}, packets={args.n_packets}')

    # Load H mmap once in main process for basis-building
    H_mmap = np.load(Path(db_path) / 'H.npy', mmap_mode='r')

    rng = np.random.default_rng(args.seed)

    N_SUBSPACES = max(1, min(10, args.n_packets))
    packets_per_subspace = max(1, args.n_packets // N_SUBSPACES)

    results = []
    nonzero_packets = []
    t0 = time.time()
    done = 0

    # Import here to avoid circular imports at module level
    db_main = TermDB(load_path=db_path, generate=False)
    db_main.H     = H_mmap
    db_main.sigma = np.load(Path(db_path) / 'sigma.npy', mmap_mode='r')

    with mp.Pool(n_workers, initializer=_worker_init, initargs=(db_path,)) as pool:
        for sub_i in range(N_SUBSPACES):
            if len(results) >= args.n_packets:
                break

            # Build random basis in the main process
            basis_idx, H_basis = _build_random_basis(H_mmap, rng, target_rank_H)
            if basis_idx is None:
                print(f'  [subspace {sub_i+1}/{N_SUBSPACES}] Could not build independent basis, skipping')
                continue

            # ONE query_subspace call per subspace — heavyweight, done only here
            hit_indices = db_main.query_subspace(H_basis.astype(np.int8))
            if len(hit_indices) < R:
                print(f'  [subspace {sub_i+1}/{N_SUBSPACES}] Only {len(hit_indices)} hits (< R={R}), skipping')
                continue

            hit_arr = np.asarray(hit_indices, dtype=np.int64)
            print(f'  [subspace {sub_i+1}/{N_SUBSPACES}] {len(hit_arr):,} hits — dispatching {packets_per_subspace} packets')

            sub_seeds = rng.integers(0, 2**31, size=packets_per_subspace).tolist()
            tasks = [(int(s), R, target_rank_H, hit_arr) for s in sub_seeds]

            for res in pool.imap_unordered(_test_packet_from_hits, tasks, chunksize=1):
                done += 1
                if res is None:
                    print(f'  [{done:4d}] FAILED (no valid packet from hits)')
                    continue
                rH, dleak, sinno, idx = res
                status = 'SIGMA_ALIVE' if sinno > 0 else 'sigma_dead'
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f'  [{done:4d}] rank_H={rH}  delta_leak={dleak}  sigma_innov={sinno}  [{status}]  {rate:.1f}pkt/s')
                results.append({'rank_H': rH, 'delta_leak': dleak, 'sigma_innovation': sinno})
                if sinno > 0:
                    nonzero_packets.append({'indices': idx, 'rank_H': rH,
                                            'delta_leak': dleak, 'sigma_innovation': sinno})

    elapsed = time.time() - t0
    total = len(results)
    n_dead  = sum(1 for r in results if r['sigma_innovation'] == 0)
    n_alive = sum(1 for r in results if r['sigma_innovation'] > 0)
    max_si  = max((r['sigma_innovation'] for r in results), default=0)
    leak_hist = {}
    for r in results:
        k = r['delta_leak']
        leak_hist[k] = leak_hist.get(k, 0) + 1

    print()
    print('=== SIGMA UNIVERSALITY RESULT ===')
    print(f'  Packets tested:      {total}')
    print(f'  sigma_innovation=0:  {n_dead}  ({100*n_dead/max(total,1):.1f}%)')
    print(f'  sigma_innovation>0:  {n_alive}  ({100*n_alive/max(total,1):.1f}%)')
    print(f'  max sigma_innov:     {max_si}')
    print(f'  delta_leak dist:     {dict(sorted(leak_hist.items()))}')
    print(f'  Elapsed: {elapsed:.1f}s')
    print()
    if n_alive == 0:
        print('VERDICT: sigma_innovation=0 appears UNIVERSAL for Gate-1 packets.')
        print('  => sigma in col(H) may be an algebraic identity over {-1,0,1}.')
        print(f'  => R={R} over {{-1,0,1}} may be provably impossible.')
    else:
        print(f'VERDICT: {n_alive} packets found with sigma_innovation>0.')
        print('  => sigma=dead is NOT universal. Different basins exist.')

    output = {
        'rank': R, 'target_rank_H': target_rank_H,
        'n_packets': total, 'n_sigma_dead': n_dead, 'n_sigma_alive': n_alive,
        'max_sigma_innovation': max_si,
        'delta_leak_distribution': {str(k): v for k, v in sorted(leak_hist.items())},
        'elapsed_seconds': elapsed,
        'per_packet': results,
        'alive_packets': nonzero_packets,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()