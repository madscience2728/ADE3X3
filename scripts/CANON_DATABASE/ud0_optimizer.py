"""
UD₀-Targeted Gate-3 Optimizer
==============================
Key theorem (Phase 7):
    UΣ = 3·UD₀   where  U = ker(H^T),  D₀[k,(r,u)] = α[k,r,0]·β[k,0,u]

Gate 3 ⟺ rank(UD₀) = 9.

Gates 1+2 constrain ONLY the off-diagonal pieces δ_{st} (s≠t). They say nothing
about D₀. The Phase-4 best accidentally has UD₀=0 — this is NOT forced by Gates 1+2.

New objective (lower = better, lexicographic):
    (gate1_gap, delta_leak, -‖UD₀‖_F, recon_err)

Strategy:
  1. Start from Phase-4 best (Gates 1+2 satisfied, UD₀=0).
  2. Swap one term at a time, keeping gate1_gap=0, delta_leak=0.
  3. Maximise ‖UD₀‖_F among Gate-1+2-preserving moves.
  4. If rank(UD₀)=9 found → check full Gate 3 and recon_err → report solution.

Usage:
    python CANON_DATABASE/ud0_optimizer.py [--workers N] [--minutes M]
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))

DB_PATH = _ROOT / "CANON_DATABASE" / "data"
CHECKPOINT_IN  = _ROOT / "CANON_DATABASE" / "swap_checkpoint.json"
CHECKPOINT_OUT = _ROOT / "CANON_DATABASE" / "ud0_checkpoint.json"

TARGET_RANK_H = 10
R = 19
NULL_DIM = R - TARGET_RANK_H  # = 9

RANK_TOL = 1e-9
RANK_ABS = 1e-12


# ── helpers ──────────────────────────────────────────────────────────────────

def fast_rank(M: np.ndarray) -> int:
    if M.size == 0:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[0] == 0:
        return 0
    return int(np.sum(sv > max(RANK_ABS, RANK_TOL * sv[0])))


def col_perp_basis(M: np.ndarray):
    """Left null-space of M: returns U s.t. U @ M ≈ 0, shape (R-rank, R)."""
    _, s, Vt = np.linalg.svd(M.T, full_matrices=True)
    tol = max(RANK_ABS, RANK_TOL * (s[0] if len(s) > 0 and s[0] > 0 else 1.0))
    r = int(np.sum(s > tol))
    return Vt[r:]  # (R - rank(M), R)


def make_d0(alpha_batch: np.ndarray, beta_batch: np.ndarray) -> np.ndarray:
    """
    alpha_batch: (R, 3, 3), beta_batch: (R, 3, 3)
    Returns D₀: (R, 9) where D₀[k, r*3+u] = alpha[k,r,0] * beta[k,0,u]
    """
    # outer(alpha[:,r,0], beta[:,0,u]) for each (r,u)
    a0 = alpha_batch[:, :, 0]   # (R, 3): a0[k,r] = alpha[k,r,0]
    b0 = beta_batch[:, 0, :]    # (R, 3): b0[k,u] = beta[k,0,u]
    # D0[k, r*3+u] = a0[k,r] * b0[k,u]
    D0 = (a0[:, :, None] * b0[:, None, :]).reshape(R, 9)  # (R, 9)
    return D0.astype(np.float64)


def score_packet(H: np.ndarray, delta: np.ndarray, D0: np.ndarray,
                 sigma: np.ndarray) -> tuple:
    """
    Returns (gate1_gap, delta_leak, neg_ud0_norm_sq, recon_err).
    Lower is better (lexicographic).
    """
    H_f = H.astype(np.float64)
    rank_H = fast_rank(H_f)
    gate1_gap = abs(rank_H - TARGET_RANK_H)

    d_f = delta.astype(np.float64)
    N = np.hstack([H_f, d_f])
    rank_N = fast_rank(N)
    delta_leak = rank_N - rank_H

    # Compute ‖UD₀‖_F only when Gate 1 is satisfied (U is well-defined)
    if rank_H == TARGET_RANK_H:
        U = col_perp_basis(H_f)   # (9, 19)
        ud0 = U @ D0              # (9, 9)
        ud0_norm_sq = float(np.sum(ud0 ** 2))
        rk_ud0 = fast_rank(ud0)
    else:
        ud0_norm_sq = 0.0
        rk_ud0 = 0

    # Reconstruction error (cheap proxy: use sigma+H+delta)
    SN = np.hstack([sigma.astype(np.float64), N])
    target = np.zeros((9, 81), dtype=np.float64)
    target[:, :9] = 3.0 * np.eye(9)
    try:
        G, _, _, _ = np.linalg.lstsq(SN.T, target.T, rcond=None)
        recon_err = float(np.max(np.abs(target.T - SN.T @ G)))
    except Exception:
        recon_err = 1e6

    # Objective: gate1 first (hard), then maximize rk(UD₀) (primary),
    # then minimize delta_leak (secondary), then maximize ||UD₀||², then recon.
    # This means rk improvement always beats Gate-2 improvement.
    return (gate1_gap, -rk_ud0, delta_leak, -ud0_norm_sq, recon_err), rk_ud0


def is_solution(score: tuple, rk_ud0: int) -> bool:
    g1, neg_rk, dl, neg_ud0, recon = score
    return g1 == 0 and dl == 0 and rk_ud0 == NULL_DIM and recon < 0.1


def is_better(a: tuple, b: tuple) -> bool:
    return a < b


# ── TermDB loader ─────────────────────────────────────────────────────────────

class _DB:
    def __init__(self):
        print("Loading TermDB...", end=" ", flush=True)
        self.H        = np.load(DB_PATH / "H.npy",         mmap_mode="r")
        self.sigma    = np.load(DB_PATH / "sigma.npy",     mmap_mode="r")
        self.alpha_idx= np.load(DB_PATH / "alpha_idx.npy", mmap_mode="r")
        self.beta_idx = np.load(DB_PATH / "beta_idx.npy",  mmap_mode="r")
        self.templates= np.load(DB_PATH / "templates.npy")   # (19683,3,3) int8
        self.N        = self.H.shape[0]
        print(f"done. N={self.N:,}")

    def get_blocks(self, idx: np.ndarray):
        """Returns H(R,18), sigma(R,9), delta(R,54), D0(R,9), alpha(R,3,3), beta(R,3,3)."""
        H     = self.H[idx].astype(np.float64)
        sigma = self.sigma[idx].astype(np.float64)
        alpha = self.templates[self.alpha_idx[idx]].astype(np.float64)  # (R,3,3)
        beta  = self.templates[self.beta_idx[idx]].astype(np.float64)   # (R,3,3)
        D0    = make_d0(alpha, beta)

        pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
        delta = np.zeros((len(idx), 54), dtype=np.float64)
        for ci, (s, t) in enumerate(pairs):
            for r in range(3):
                for u in range(3):
                    col_idx = ci * 9 + r * 3 + u
                    delta[:, col_idx] = alpha[:, r, s] * beta[:, t, u]
        return H, sigma, delta, D0


# ── worker ────────────────────────────────────────────────────────────────────

def _worker(worker_id: int, seed_indices: list[int], n_minutes: float,
            out_queue: mp.Queue, stop_event: mp.Event):
    rng = np.random.default_rng(worker_id * 1000 + int(time.time()) % 1000)
    db = _DB()
    N = db.N

    idx = np.array(seed_indices, dtype=np.int64)
    H, sigma, delta, D0 = db.get_blocks(idx)
    best_score, rk_ud0 = score_packet(H, delta, D0, sigma)
    best_idx = idx.copy()

    out_queue.put({'worker': worker_id, 'event': 'start',
                   'score': best_score, 'rk_ud0': rk_ud0})

    t_end = time.time() + n_minutes * 60
    swaps = 0
    improvements = 0
    stall = 0
    stall_limit = 500

    while time.time() < t_end and not stop_event.is_set():
        # Swap one random position
        pos = int(rng.integers(R))
        new_term = int(rng.integers(N))
        if new_term in best_idx:
            continue

        trial_idx = best_idx.copy()
        trial_idx[pos] = new_term

        H2, sigma2, delta2, D02 = db.get_blocks(trial_idx)
        trial_score, trial_rk = score_packet(H2, delta2, D02, sigma2)

        swaps += 1
        if is_better(trial_score, best_score):
            best_score = trial_score
            best_idx = trial_idx
            improvements += 1
            stall = 0
            out_queue.put({
                'worker': worker_id,
                'event': 'improvement',
                'score': best_score,
                'rk_ud0': trial_rk,
                'indices': best_idx.tolist(),
                'swaps': swaps,
                'improvements': improvements,
            })
            if is_solution(best_score, trial_rk):
                out_queue.put({'worker': worker_id, 'event': 'solution',
                               'score': best_score, 'rk_ud0': trial_rk,
                               'indices': best_idx.tolist()})
                stop_event.set()
                return
        else:
            stall += 1

        # On stall: try double swap
        if stall >= stall_limit:
            stall = 0
            pos2 = int(rng.integers(R))
            new2 = int(rng.integers(N))
            if new2 not in best_idx and pos2 != pos:
                trial_idx2 = best_idx.copy()
                trial_idx2[pos] = new_term
                trial_idx2[pos2] = new2
                H3, sigma3, delta3, D03 = db.get_blocks(trial_idx2)
                s3, rk3 = score_packet(H3, delta3, D03, sigma3)
                if is_better(s3, best_score):
                    best_score = s3
                    best_idx = trial_idx2
                    improvements += 1
                    out_queue.put({
                        'worker': worker_id, 'event': 'improvement',
                        'score': best_score, 'rk_ud0': rk3,
                        'indices': best_idx.tolist(),
                        'swaps': swaps, 'improvements': improvements,
                    })

        if swaps % 5000 == 0:
            out_queue.put({'worker': worker_id, 'event': 'heartbeat',
                           'score': best_score, 'rk_ud0': rk_ud0,
                           'swaps': swaps})

    out_queue.put({'worker': worker_id, 'event': 'done',
                   'score': best_score, 'rk_ud0': rk_ud0,
                   'indices': best_idx.tolist(), 'swaps': swaps})


# ── coordinator ───────────────────────────────────────────────────────────────

def run(n_workers: int = 4, n_minutes: float = 20.0):
    # Load seed from Phase-4 best
    with open(CHECKPOINT_IN) as f:
        ckpt = json.load(f)
    seed_indices = ckpt["global_best_indices"]
    print(f"Seed: Phase-4 best ({len(seed_indices)} terms)")

    db = _DB()
    idx0 = np.array(seed_indices, dtype=np.int64)
    H0, sigma0, delta0, D00 = db.get_blocks(idx0)
    s0, rk0 = score_packet(H0, delta0, D00, sigma0)
    print(f"Seed score: gate1={s0[0]} rk(UD₀)={rk0} delta_leak={s0[2]} "
          f"‖UD₀‖²={-s0[3]:.4f} recon={s0[4]:.4f}")
    del db  # free before forking

    out_queue: mp.Queue = mp.Queue(maxsize=50000)
    stop_event: mp.Event = mp.Event()

    workers = []
    for wid in range(n_workers):
        # Stagger seeds slightly by perturbing 2 random positions
        rng = np.random.default_rng(wid + 42)
        seed_i = list(seed_indices)
        if wid > 0:
            # Random restart: perturb 2 positions with random valid Gate-1 terms
            pos = rng.choice(R, size=min(wid * 2, R), replace=False)
            N_db = 387_420_489
            for p in pos:
                seed_i[int(p)] = int(rng.integers(N_db))
        p = mp.Process(target=_worker, args=(wid, seed_i, n_minutes,
                                              out_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)

    print(f"\nRunning {n_workers} workers for {n_minutes:.0f} minutes...\n")
    rk_log_path = _ROOT / "CANON_DATABASE" / "ud0_rk_log.jsonl"
    rk_log_fh = open(rk_log_path, 'a')

    print(f"{'time':>6}  {'worker':>6}  {'gate1':>5}  {'rk_ud0':>7}  {'dleak':>6}  "
          f"{'‖UD₀‖²':>10}  {'recon':>8}  {'swaps':>8}")
    print("-" * 78)

    best_global = s0
    best_rk_global = rk0
    best_indices_global = list(seed_indices)
    t0 = time.time()
    solution_found = False
    done_count = 0

    while done_count < n_workers and not stop_event.is_set():
        try:
            msg = out_queue.get(timeout=2.0)
        except Exception:
            continue

        event = msg['event']
        w = msg['worker']
        sc = tuple(msg['score'])
        rk = msg.get('rk_ud0', 0)
        swaps = msg.get('swaps', 0)
        elapsed = time.time() - t0

        if event in ('improvement', 'solution'):
            # Always log rk > 0 packets separately (regardless of gate2)
            if rk > 0 and sc[0] == 0:
                entry = json.dumps({'t': round(elapsed,1), 'w': w, 'rk': rk,
                                    'dleak': sc[2], 'ud0sq': -sc[3],
                                    'recon': sc[4], 'swaps': swaps,
                                    'indices': msg.get('indices', [])})
                rk_log_fh.write(entry + '\n'); rk_log_fh.flush()

            if is_better(sc, best_global):
                best_global = sc
                best_rk_global = rk
                best_indices_global = msg.get('indices', best_indices_global)
                print(f"{elapsed:6.1f}  {w:6d}  {sc[0]:5d}  {rk:7d}  {sc[2]:6d}  "
                      f"{-sc[3]:10.4f}  {sc[4]:8.4f}  {swaps:8d}  ← best")
            else:
                print(f"{elapsed:6.1f}  {w:6d}  {sc[0]:5d}  {rk:7d}  {sc[2]:6d}  "
                      f"{-sc[3]:10.4f}  {sc[4]:8.4f}  {swaps:8d}")

            # Save checkpoint after each improvement
            _save_checkpoint(best_global, best_rk_global,
                             best_indices_global, elapsed)

        elif event == 'solution':
            solution_found = True
            print(f"\n{'='*60}")
            print(f"SOLUTION FOUND by worker {w}!")
            print(f"score={sc}, rk(UD₀)={rk}")
            print(f"{'='*60}\n")
            stop_event.set()

        elif event == 'done':
            done_count += 1

        elif event == 'heartbeat' and swaps % 25000 == 0:
            print(f"{elapsed:6.1f}  {w:6d}  {sc[0]:5d}  {rk:7d}  {sc[2]:6d}  "
                  f"{-sc[3]:10.4f}  {sc[4]:8.4f}  {swaps:8d}")

    for p in workers:
        p.terminate()

    print(f"\n{'='*60}")
    print(f"Run complete. Best result:")
    print(f"  gate1_gap   = {best_global[0]}")
    print(f"  rk(UD₀)     = {best_rk_global}  (need {NULL_DIM} for Gate 3)")
    print(f"  delta_leak  = {best_global[2]}")
    print(f"  ‖UD₀‖²      = {-best_global[3]:.6f}")
    print(f"  recon_err   = {best_global[4]:.6f}")
    if best_rk_global >= NULL_DIM:
        print("  *** GATE 3 SATISFIED! ***")
    elif best_rk_global > 0:
        print(f"  Progress: rk(UD₀) rose from 0 to {best_rk_global} — WALL CRACKED")
    else:
        print("  rk(UD₀) = 0 — still pinned")

    _save_checkpoint(best_global, best_rk_global, best_indices_global,
                     time.time() - t0)
    return best_global, best_rk_global, best_indices_global


def _save_checkpoint(score, rk_ud0, indices, elapsed):
    data = {
        'score': list(score),
        'rk_ud0': rk_ud0,
        'indices': indices,
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    CHECKPOINT_OUT.write_text(json.dumps(data, indent=2))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int,   default=4)
    parser.add_argument("--minutes",  type=float, default=20.0)
    args = parser.parse_args()
    run(n_workers=args.workers, n_minutes=args.minutes)
