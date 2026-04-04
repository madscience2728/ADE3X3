"""
swap_worker.py — Single worker process for the R=19 swap optimizer.

Each worker:
  1. Loads TermDB in read-only (shared pages) mode.
  2. Generates a random Gate-1-satisfying starting packet.
  3. Runs the swap loop: single/double/triple swaps, accepting improvements.
  4. Sends score updates + best indices to the coordinator via multiprocessing.Queue.
  5. On stall: escalates move size, then restarts.
"""
from __future__ import annotations

import os
import sys
import time
import random
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np

# Add repo root so we can import CANON_DATABASE modules
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))
sys.path.insert(0, str(_ROOT / "CANON_DATABASE" / "scripts"))

from term_db import TermDB
from swap_scoring import score_packet, score_is_better, score_is_solution
from swap_config import SwapConfig


# ---------------------------------------------------------------------------
# Messages sent to coordinator queue
# ---------------------------------------------------------------------------

class _Msg:
    __slots__ = ('worker_id', 'kind', 'payload')
    def __init__(self, worker_id: int, kind: str, payload: dict):
        self.worker_id = worker_id
        self.kind      = kind   # 'update' | 'solution' | 'status'
        self.payload   = payload


# ---------------------------------------------------------------------------
# Gate-1 packet generation
# ---------------------------------------------------------------------------

def _random_gate1_packet(
    db: TermDB,
    rng: np.random.Generator,
    R: int,
    target_rank_H: int,
    rank_tol: float = 1e-9,
) -> np.ndarray:
    """
    Generate a random R-term packet with rank(H) == target_rank_H.

    Strategy:
      1. Pick target_rank_H independent H-rows as basis.
      2. Query DB for their span (hit pool).
      3. Pick (R - target_rank_H) dependents randomly from the pool.
    """
    n_dep = R - target_rank_H
    max_basis_attempts = 5000

    # Build basis: pick independent H-rows
    basis_indices: list[int] = []
    H_basis = np.empty((0, 18), dtype=np.float64)

    attempts = 0
    while len(basis_indices) < target_rank_H:
        attempts += 1
        if attempts > max_basis_attempts:
            # Give up and return whatever we have, padded with randoms
            break
        idx = int(rng.integers(0, db.N_PAIRS))
        h = db.H[idx].astype(np.float64).reshape(1, 18)
        trial = np.vstack([H_basis, h]) if H_basis.shape[0] > 0 else h
        sv = np.linalg.svd(trial, compute_uv=False)
        thresh = 1e-9 * max(trial.shape) * sv[0] if sv[0] > 0 else 1e-9
        if np.sum(sv > thresh) == len(basis_indices) + 1:
            basis_indices.append(idx)
            H_basis = trial

    basis_arr = np.array(basis_indices, dtype=np.int64)

    # Query span of basis H-rows
    try:
        hits = db.query_subspace(H_basis.astype(np.int8))
    except Exception:
        hits = np.arange(min(200000, db.N_PAIRS), dtype=np.int64)

    if len(hits) < n_dep + 1:
        hits = np.arange(db.N_PAIRS, dtype=np.int64)

    # Sample dependents (exclude basis indices)
    basis_set = set(basis_indices)
    candidate_pool = hits[~np.isin(hits, list(basis_set))]
    if len(candidate_pool) < n_dep:
        candidate_pool = np.arange(db.N_PAIRS, dtype=np.int64)

    dep_indices = rng.choice(candidate_pool, size=n_dep, replace=False)
    return np.concatenate([basis_arr, dep_indices.astype(np.int64)])


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_main(
    worker_id: int,
    config: SwapConfig,
    out_queue: mp.Queue,
    stop_event: mp.Event,
):
    """Entry point for each worker subprocess."""
    rng = np.random.default_rng(seed=worker_id * 31337 + 1)
    random.seed(worker_id * 12345 + 7)

    R             = config.R
    target_H      = config.target_rank_H
    stall_thresh  = config.stall_threshold
    rank_tol      = config.rank_tol
    hit_bias      = config.hit_pool_bias
    N_DB          = TermDB.N_PAIRS

    # Load DB. Then re-open the large arrays as read-only mmap so the OS
    # can share physical pages across all worker processes on Windows
    # (~11 GB once instead of 11 GB x N_workers).
    import numpy as _np
    from pathlib import Path as _Path
    db = TermDB(load_path=config.db_path, generate=False)
    _dp = _Path(config.db_path)
    db.H     = _np.load(_dp / "H.npy",     mmap_mode='r')
    db.sigma = _np.load(_dp / "sigma.npy", mmap_mode='r')

    # Move probabilities
    probs = np.array([
        config.single_swap_prob,
        config.double_swap_prob,
        config.triple_swap_prob,
    ], dtype=np.float64)
    probs /= probs.sum()  # normalise
    move_sizes = [1, 2, 3]

    total_swaps     = 0
    total_improv    = 0
    total_restarts  = 0
    best_score: Optional[tuple] = None
    best_indices: Optional[np.ndarray] = None

    def _send(kind: str, payload: dict):
        try:
            out_queue.put_nowait(_Msg(worker_id, kind, payload))
        except Exception:
            pass  # queue full — skip this update

    def _build_state(indices: np.ndarray):
        """Compute H, sigma, delta, hit_pool for a packet."""
        H_rows     = db.H[indices].astype(np.float64)
        sigma_rows = db.sigma[indices].astype(np.float64)
        delta_rows = db.compute_delta_batch(indices).astype(np.float64)
        # Query hit pool for biased sampling
        try:
            pool = db.query_subspace(db.H[indices].astype(np.int8))
        except Exception:
            pool = None
        return H_rows, sigma_rows, delta_rows, pool

    def _score(indices, H, sigma, delta):
        return score_packet(H, sigma, delta, target_H, rank_tol)

    def _gate1_ok(H_new: np.ndarray) -> bool:
        sv = np.linalg.svd(H_new, compute_uv=False)
        thresh = rank_tol * max(H_new.shape) * sv[0] if sv[0] > 0 else rank_tol
        return int(np.sum(sv > thresh)) == target_H

    for restart_num in range(config.max_restarts_per_worker):
        if stop_event.is_set():
            break

        # --- Generate starting packet ---
        indices = _random_gate1_packet(db, rng, R, target_H, rank_tol)
        H, sigma, delta, pool = _build_state(indices)

        # Fix rank(H) if generation failed
        rank_H_cur = _gate1_ok(H)
        if not rank_H_cur:
            # Force rebuild with tighter loop
            for _ in range(5):
                indices = _random_gate1_packet(db, rng, R, target_H, rank_tol)
                H = db.H[indices].astype(np.float64)
                if _gate1_ok(H):
                    sigma = db.sigma[indices].astype(np.float64)
                    delta = db.compute_delta_batch(indices).astype(np.float64)
                    try:
                        pool = db.query_subspace(db.H[indices].astype(np.int8))
                    except Exception:
                        pool = None
                    break

        cur_score = _score(indices, H, sigma, delta)

        if best_score is None or score_is_better(cur_score, best_score):
            best_score   = cur_score
            best_indices = indices.copy()
            _send('update', {
                'score':   list(best_score),
                'indices': best_indices.tolist(),
                'swaps':   total_swaps,
                'improv':  total_improv,
                'restart': total_restarts,
            })

        stall_count = 0
        restart_swaps = 0

        while not stop_event.is_set():
            # Choose move size
            move_k = rng.choice(move_sizes, p=probs)

            # Choose positions to swap
            swap_pos = rng.choice(R, size=move_k, replace=False)

            # Sample new record indices (biased toward hit pool)
            new_idxs = []
            for _ in range(move_k):
                if pool is not None and len(pool) > 0 and rng.random() < hit_bias:
                    new_idxs.append(int(rng.choice(pool)))
                else:
                    new_idxs.append(int(rng.integers(0, N_DB)))

            # --- Pre-filter: Gate 1 ---
            H_trial = H.copy()
            for j, pos in enumerate(swap_pos):
                H_trial[pos] = db.H[new_idxs[j]].astype(np.float64)
            if not _gate1_ok(H_trial):
                total_swaps += 1
                restart_swaps += 1
                stall_count += 1
                if stall_count >= stall_thresh:
                    break
                continue

            # --- Build full trial ---
            sigma_trial = sigma.copy()
            delta_trial = delta.copy()
            indices_trial = indices.copy()
            for j, pos in enumerate(swap_pos):
                nidx = new_idxs[j]
                indices_trial[pos] = nidx
                sigma_trial[pos]   = db.sigma[nidx].astype(np.float64)
                delta_trial[pos]   = db.compute_delta(nidx).astype(np.float64)

            trial_score = _score(indices_trial, H_trial, sigma_trial, delta_trial)
            total_swaps += 1
            restart_swaps += 1

            if score_is_better(trial_score, cur_score):
                # Accept
                indices = indices_trial
                H       = H_trial
                sigma   = sigma_trial
                delta   = delta_trial
                cur_score = trial_score
                stall_count = 0
                total_improv += 1

                # Refresh hit pool on improvement
                try:
                    pool = db.query_subspace(db.H[indices].astype(np.int8))
                except Exception:
                    pool = None

                if score_is_better(cur_score, best_score):
                    best_score   = cur_score
                    best_indices = indices.copy()
                    _send('update', {
                        'score':   list(best_score),
                        'indices': best_indices.tolist(),
                        'swaps':   total_swaps,
                        'improv':  total_improv,
                        'restart': total_restarts,
                    })

                    if score_is_solution(best_score):
                        _send('solution', {
                            'score':   list(best_score),
                            'indices': best_indices.tolist(),
                        })
                        stop_event.set()
                        return
            else:
                stall_count += 1
                if stall_count >= stall_thresh:
                    break

        total_restarts += 1
        if total_restarts % 10 == 0:
            _send('status', {
                'swaps':    total_swaps,
                'improv':   total_improv,
                'restarts': total_restarts,
                'score':    list(cur_score) if cur_score else None,
            })

    _send('status', {
        'swaps':    total_swaps,
        'improv':   total_improv,
        'restarts': total_restarts,
        'score':    list(best_score) if best_score else None,
        'done':     True,
    })
