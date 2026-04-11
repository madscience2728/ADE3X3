"""
GF(3) Tensor Rank — 18-Hour Production Run (v2)
=================================================
All-cores parallel attack on R_{GF(3)}(T_matmul(3)).

CORRECTED: Standard R=27 decomposition is rigid over GF(3).
No merging possible. Must search from scratch via ALS.

Strategy (24 cores):
  - Cores 0-11:  ALS with random restarts, target R ∈ {19..23}
  - Cores 12-17: ALS focused on R ∈ {22,23} (most likely to succeed)
  - Cores 18-23: Hybrid ALS + random perturbation, fixed R

psutil CPU/memory logging every 60s.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Value
from itertools import product
from time import time, strftime
import json
import os
import psutil
import threading
import ctypes

# ================================================================
TOTAL_TIME = 18 * 3600
LOG_INTERVAL = 60
REPORT_INTERVAL = 300
NUM_WORKERS = 24
N = 3

GF3_MUL = np.array([[0,0,0],[0,1,2],[0,2,1]], dtype=np.int8)
GF3_INV = np.array([0, 1, 2], dtype=np.int8)

def build_T():
    T = np.zeros((9, 9, 9), dtype=np.int8)
    for r, s, u in product(range(N), repeat=3):
        T[3*r+s, 3*s+u, 3*r+u] = 1
    return T

def gf3_residual_nnz(T, U, V, W, R):
    T_sum = np.zeros((9, 9, 9), dtype=np.int32)
    for r in range(R):
        T_sum += np.einsum('i,j,k->ijk', U[r].astype(np.int32),
                           V[r].astype(np.int32), W[r].astype(np.int32))
    return np.count_nonzero(np.mod(T.astype(np.int32) - T_sum, 3))

def save_solution(worker_id, strategy, R, U, V, W, start_time):
    os.makedirs("outputs", exist_ok=True)
    ts = strftime("%Y%m%d_%H%M%S")
    fname = f"outputs/gf3_R{R}_{strategy}_w{worker_id}_{ts}.json"
    data = {
        "rank": R, "strategy": strategy, "worker": worker_id,
        "elapsed_s": time() - start_time,
        "U": U.tolist(), "V": V.tolist(), "W": W.tolist()
    }
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {fname}", flush=True)

def compute_partial(T, U, V, W, R, r):
    """Compute T - sum_{s!=r} U[s] ⊗ V[s] ⊗ W[s] mod 3."""
    T_partial = T.astype(np.int32).copy()
    for s in range(R):
        if s != r:
            T_partial -= np.einsum('i,j,k->ijk', U[s].astype(np.int32),
                                   V[s].astype(np.int32), W[s].astype(np.int32))
    return np.mod(T_partial, 3).astype(np.int8)

def optimal_update_mode0(T_partial, V_r, W_r):
    """Optimally update U[r] given fixed V[r], W[r]. Each U[r][i] chosen independently."""
    # vw[j,k] = V_r[j] * W_r[k] in GF(3)
    vw = GF3_MUL[V_r, :][:, W_r]  # (9,9)
    u = np.zeros(9, dtype=np.int8)
    for i in range(9):
        best_c = 0; best_matches = 0
        for c in range(3):
            # count positions where T_partial[i,j,k] == c * vw[j,k] mod 3
            contrib = np.mod(c * vw.astype(np.int32), 3).astype(np.int8)
            matches = np.count_nonzero(T_partial[i] == contrib)
            if matches > best_matches:
                best_matches = matches; best_c = c
        u[i] = best_c
    return u

def optimal_update_mode1(T_partial, U_r, W_r):
    """Optimally update V[r] given fixed U[r], W[r]."""
    uw = GF3_MUL[U_r, :][:, W_r]  # (9,9)
    v = np.zeros(9, dtype=np.int8)
    for j in range(9):
        best_c = 0; best_matches = 0
        for c in range(3):
            contrib = np.mod(c * uw.astype(np.int32), 3).astype(np.int8)
            matches = np.count_nonzero(T_partial[:, j, :] == contrib)
            if matches > best_matches:
                best_matches = matches; best_c = c
        v[j] = best_c
    return v

def optimal_update_mode2(T_partial, U_r, V_r):
    """Optimally update W[r] given fixed U[r], V[r]."""
    uv = GF3_MUL[U_r, :][:, V_r]  # (9,9)
    w = np.zeros(9, dtype=np.int8)
    for k in range(9):
        best_c = 0; best_matches = 0
        for c in range(3):
            contrib = np.mod(c * uv.astype(np.int32), 3).astype(np.int8)
            matches = np.count_nonzero(T_partial[:, :, k] == contrib)
            if matches > best_matches:
                best_matches = matches; best_c = c
        w[k] = best_c
    return w

def als_update_factor(T, U, V, W, R, mode, rng):
    for r in range(R):
        T_partial = compute_partial(T, U, V, W, R, r)
        if mode == 0:
            U[r] = optimal_update_mode0(T_partial, V[r], W[r])
        elif mode == 1:
            V[r] = optimal_update_mode1(T_partial, U[r], W[r])
        else:
            W[r] = optimal_update_mode2(T_partial, U[r], V[r])

def local_search(T, U, V, W, R, rng, max_rounds=50):
    """After ALS converges, alternate: re-randomize worst term, re-run ALS on it."""
    nnz = gf3_residual_nnz(T, U, V, W, R)
    for _ in range(max_rounds):
        if nnz == 0: return 0
        # Find worst term: the one whose removal reduces residual most
        best_r = 0; best_drop = -1
        for r in range(R):
            # Zero out term r temporarily
            u_bak, v_bak, w_bak = U[r].copy(), V[r].copy(), W[r].copy()
            U[r][:] = 0; V[r][:] = 0; W[r][:] = 0
            nnz_without = gf3_residual_nnz(T, U, V, W, R)
            U[r][:] = u_bak; V[r][:] = v_bak; W[r][:] = w_bak
            drop = nnz - nnz_without  # positive = term is hurting
            if drop > best_drop:
                best_drop = drop; best_r = r
        # Re-randomize worst term and optimize it
        U[best_r] = rng.integers(0, 3, 9).astype(np.int8)
        V[best_r] = rng.integers(0, 3, 9).astype(np.int8)
        W[best_r] = rng.integers(0, 3, 9).astype(np.int8)
        for sweep in range(10):
            T_p = compute_partial(T, U, V, W, R, best_r)
            U[best_r] = optimal_update_mode0(T_p, V[best_r], W[best_r])
            T_p = compute_partial(T, U, V, W, R, best_r)
            V[best_r] = optimal_update_mode1(T_p, U[best_r], W[best_r])
            T_p = compute_partial(T, U, V, W, R, best_r)
            W[best_r] = optimal_update_mode2(T_p, U[best_r], V[best_r])
        new_nnz = gf3_residual_nnz(T, U, V, W, R)
        if new_nnz >= nnz:
            break  # no improvement from re-randomizing worst term
        nnz = new_nnz
    return nnz

def worker_als(worker_id, shared_best, T, rng_seed, start_time, max_time,
               target_ranks, max_als_iters=500, stale_limit=30):
    rng = np.random.default_rng(rng_seed)
    attempts = 0
    best_nnz_ever = {R: 729 for R in target_ranks}
    
    while time() - start_time < max_time:
        for target_R in target_ranks:
            if time() - start_time >= max_time:
                break
            
            U = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
            V = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
            W = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
            
            prev_nnz = 729; stale = 0
            for it in range(max_als_iters):
                if time() - start_time >= max_time: break
                als_update_factor(T, U, V, W, target_R, it % 3, rng)
                nnz = gf3_residual_nnz(T, U, V, W, target_R)
                
                if nnz == 0:
                    with shared_best.get_lock():
                        if target_R < shared_best.value:
                            shared_best.value = target_R
                    print(f"\n  [W{worker_id:02d}] *** FOUND R={target_R} *** "
                          f"(attempt {attempts}, iter {it}, {(time()-start_time)/3600:.2f}h)",
                          flush=True)
                    save_solution(worker_id, 'als', target_R, U, V, W, start_time)
                    target_ranks = [R for R in target_ranks if R < target_R]
                    if not target_ranks: return target_R
                    break
                
                if nnz < best_nnz_ever.get(target_R, 729):
                    best_nnz_ever[target_R] = nnz
                if nnz >= prev_nnz:
                    stale += 1
                    if stale > stale_limit:
                        # ALS converged — try local search
                        nnz = local_search(T, U, V, W, target_R, rng, max_rounds=30)
                        if nnz == 0:
                            with shared_best.get_lock():
                                if target_R < shared_best.value:
                                    shared_best.value = target_R
                            print(f"\n  [W{worker_id:02d}] *** FOUND R={target_R} (local search) *** "
                                  f"(attempt {attempts}, {(time()-start_time)/3600:.2f}h)",
                                  flush=True)
                            save_solution(worker_id, 'als+local', target_R, U, V, W, start_time)
                            target_ranks = [R for R in target_ranks if R < target_R]
                            if not target_ranks: return target_R
                        if nnz < best_nnz_ever.get(target_R, 729):
                            best_nnz_ever[target_R] = nnz
                        break
                else:
                    stale = 0
                prev_nnz = nnz
            
            attempts += 1
            if attempts % 500 == 0 and worker_id % 6 == 0:
                elapsed = time() - start_time
                bests = {R: best_nnz_ever.get(R, 729) for R in [19,20,21,22,23]}
                print(f"  [W{worker_id:02d}] {attempts} restarts, {elapsed/3600:.1f}h | "
                      f"best nnz: {bests}", flush=True)
    
    return min(shared_best.value, 27)

def worker_hybrid(worker_id, shared_best, T, rng_seed, start_time, max_time, target_R):
    rng = np.random.default_rng(rng_seed)
    attempts = 0; best_nnz = 729
    
    while time() - start_time < max_time:
        U = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
        V = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
        W = rng.integers(0, 3, (target_R, 9)).astype(np.int8)
        
        prev_nnz = 729
        for it in range(200):
            if time() - start_time >= max_time: break
            als_update_factor(T, U, V, W, target_R, it % 3, rng)
            nnz = gf3_residual_nnz(T, U, V, W, target_R)
            
            if nnz == 0:
                with shared_best.get_lock():
                    if target_R < shared_best.value:
                        shared_best.value = target_R
                print(f"\n  [W{worker_id:02d} HYB] *** FOUND R={target_R} *** "
                      f"({(time()-start_time)/3600:.2f}h)", flush=True)
                save_solution(worker_id, 'hybrid', target_R, U, V, W, start_time)
                return target_R
            
            if nnz < best_nnz: best_nnz = nnz
            if nnz >= prev_nnz:
                n_perturb = rng.integers(1, min(4, target_R))
                for _ in range(n_perturb):
                    r = rng.integers(target_R)
                    which = rng.integers(3)
                    if which == 0: U[r] = rng.integers(0, 3, 9).astype(np.int8)
                    elif which == 1: V[r] = rng.integers(0, 3, 9).astype(np.int8)
                    else: W[r] = rng.integers(0, 3, 9).astype(np.int8)
            prev_nnz = nnz
        
        attempts += 1
        if attempts % 200 == 0 and worker_id == 18:
            print(f"  [W{worker_id:02d} HYB R={target_R}] "
                  f"{attempts} restarts, best_nnz={best_nnz}, "
                  f"{(time()-start_time)/3600:.1f}h", flush=True)
    return 27

def cpu_monitor(start_time, max_time, logfile):
    with open(logfile, 'w') as f:
        f.write("elapsed_s,cpu_percent,mem_percent,mem_gb\n")
        while time() - start_time < max_time + 60:
            try:
                cpu = psutil.cpu_percent(interval=LOG_INTERVAL)
                mem = psutil.virtual_memory()
                elapsed = time() - start_time
                f.write(f"{elapsed:.0f},{cpu:.1f},{mem.percent:.1f},{mem.used/1e9:.2f}\n")
                f.flush()
                if int(elapsed) % REPORT_INTERVAL < LOG_INTERVAL + 5:
                    print(f"\n  [MON] {elapsed/3600:.1f}h | CPU: {cpu:.0f}% | "
                          f"RAM: {mem.used/1e9:.1f}GB ({mem.percent:.0f}%)", flush=True)
            except Exception:
                pass

def run_worker_proc(strategy, wid, shared_best, T, seed, st, mt, ranks_or_tr):
    """Top-level function for mp.Process (picklable)."""
    if strategy == 'als':
        worker_als(wid, shared_best, T, seed, st, mt, ranks_or_tr)
    else:
        worker_hybrid(wid, shared_best, T, seed, st, mt, ranks_or_tr)

def main():
    print("=" * 70)
    print("GF(3) TENSOR RANK — 18-HOUR PRODUCTION RUN v2")
    print("=" * 70)
    
    ncpu = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    print(f"Hardware: {ncpu} logical cores, {mem.total/1e9:.1f} GB RAM")
    print(f"Workers:  {NUM_WORKERS}")
    print(f"Runtime:  {TOTAL_TIME/3600:.0f} hours")
    print(f"Start:    {strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Known:    19 ≤ R_GF(3) ≤ 25")
    print()
    print("  [W00-W11] ALS random restarts, R ∈ {19..23}")
    print("  [W12-W17] ALS focused, R ∈ {22,23}")
    print("  [W18-W23] Hybrid ALS+perturb, R ∈ {19..22}")
    print()
    
    T = build_T()
    start_time = time()
    shared_best = Value(ctypes.c_int, 27)
    
    os.makedirs("outputs", exist_ok=True)
    logfile = f"outputs/gf3_cpu_log_{strftime('%Y%m%d_%H%M%S')}.csv"
    monitor = threading.Thread(target=cpu_monitor,
                               args=(start_time, TOTAL_TIME, logfile), daemon=True)
    monitor.start()
    print(f"CPU log → {logfile}")
    print("=" * 70 + "\n")
    
    procs = []
    for i in range(12):
        ranks = [23, 22, 21, 20, 19]
        start_idx = i % 5
        rotated = ranks[start_idx:] + ranks[:start_idx]
        p = mp.Process(target=run_worker_proc,
                       args=('als', i, shared_best, T, 10000+i, start_time,
                             TOTAL_TIME, rotated))
        procs.append(p)
    for i in range(6):
        p = mp.Process(target=run_worker_proc,
                       args=('als', 12+i, shared_best, T, 20000+i, start_time,
                             TOTAL_TIME, [23, 22]))
        procs.append(p)
    for i, tr in enumerate([19, 20, 21, 22, 21, 20]):
        p = mp.Process(target=run_worker_proc,
                       args=('hybrid', 18+i, shared_best, T, 30000+i, start_time,
                             TOTAL_TIME, tr))
        procs.append(p)
    
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    
    elapsed = time() - start_time
    print("\n" + "=" * 70)
    print(f"COMPLETE — {elapsed/3600:.1f}h | Best R: {shared_best.value}")
    print("=" * 70)
    
    if shared_best.value < 25:
        print(f"*** R_{{GF(3)}} ≤ {shared_best.value} — IMPROVES KNOWN BOUND ***")
    
    summary = {"best_R": shared_best.value, "hours": elapsed/3600,
               "timestamp": strftime('%Y-%m-%d %H:%M:%S')}
    sf = f"outputs/gf3_18hr_summary_{strftime('%Y%m%d_%H%M%S')}.json"
    with open(sf, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {sf}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
