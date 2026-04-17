"""
rank_compress.py — Compress AlphaTensor R=23 → R=20
====================================================
Strategy:
  1. Start from exact integer R=23 decomposition (residual = 0)
  2. For target rank R: try all C(23, 23-R) deletions of terms
     (or random subsets if combinatorially too large)
  3. For each deletion, optimize remaining R terms via Adam+LBFGS
     to minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||
  4. Best residual < threshold → FOUND

For R=22: C(23,1)=23 deletions — exhaustive
For R=21: C(23,2)=253 deletions — exhaustive  
For R=20: C(23,3)=1771 deletions — exhaustive (each fast)
"""
import itertools
import time
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── AlphaTensor R=23 exact decomposition ─────────────────────────
def alphatensor_uvw():
    u = np.array([
        [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0,-1, 0,-1,-1,-1,-1,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,-1, 1, 1, 0, 1, 0, 0,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 0, 0,-1, 0, 0],
        [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0,-1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1,-1, 0, 0, 0, 0, 0,-1, 0,-1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0,-1, 1,-1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.float64)
    v = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [-1,-1, 0, 0,-1, 0,-1,-1, 1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1,-1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0,-1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [-1,-1, 0, 0,-1, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1,-1,-1, 0, 1, 0, 1, 0,-1, 0, 0],
        [-1,-1,-1,-1,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float64)
    w = np.array([
        [0, 0, 0, 0, 0, 0,-1, 1, 1, 0, 0,-1,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1],
        [-1, 1, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0,-1, 1, 1, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
        [0, 0, 0, 1,-1, 0, 1, 0, 0, 0,-1, 0,-1, 1, 0, 0, 0,-1, 0, 0,-1, 0, 1],
        [-1, 1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 1, 0,-1, 0, 0,-1, 0, 0],
    ], dtype=np.float64)
    return u, v, w

def build_T333():
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(3):
        for s in range(3):
            for c in range(3):
                T[r*3+s, s*3+c, r*3+c] = 1.0
    return T

# ── Verify decomposition ─────────────────────────────────────────
U0, V0, W0 = alphatensor_uvw()
T333 = build_T333()
recon = np.einsum('ir,jr,kr->ijk', U0, V0, W0)
err = np.max(np.abs(recon - T333))
if err > 0.5:
    # try gamma transpose
    W0_t = np.zeros_like(W0)
    for k in range(23):
        W0_t[:, k] = W0[:, k].reshape(3,3).T.ravel()
    recon2 = np.einsum('ir,jr,kr->ijk', U0, V0, W0_t)
    err2 = np.max(np.abs(recon2 - T333))
    if err2 < 0.5:
        W0 = W0_t
        err = err2
    else:
        raise RuntimeError(f"AlphaTensor reconstruction failed: err={err}, err_T={err2}")
print(f"AlphaTensor R=23 verified: max_err={err:.1e}")

T_torch = torch.tensor(T333, dtype=torch.float64, device=device)
T_norm = torch.norm(T_torch).item()

# ── Optimization: warm-start from subset of R=23 terms ───────────
def optimize_subset(keep_indices, n_restarts=3, adam_steps=20000, lr=3e-3):
    """
    Given indices of terms to keep from R=23, optimize those terms
    to minimize reconstruction error. Warm-starts from the exact values,
    plus random perturbation restarts.
    """
    R = len(keep_indices)
    U_init = torch.tensor(U0[:, keep_indices], dtype=torch.float64, device=device)
    V_init = torch.tensor(V0[:, keep_indices], dtype=torch.float64, device=device)
    W_init = torch.tensor(W0[:, keep_indices], dtype=torch.float64, device=device)
    
    best_res = float('inf')
    best_state = None
    
    for restart in range(n_restarts):
        if restart == 0:
            # Pure warm-start from exact R=23 subset
            U = torch.nn.Parameter(U_init.clone())
            V = torch.nn.Parameter(V_init.clone())
            W = torch.nn.Parameter(W_init.clone())
        else:
            # Perturbed warm-start
            noise = 0.1 * restart
            U = torch.nn.Parameter(U_init + noise * torch.randn_like(U_init))
            V = torch.nn.Parameter(V_init + noise * torch.randn_like(V_init))
            W = torch.nn.Parameter(W_init + noise * torch.randn_like(W_init))
        
        opt = torch.optim.Adam([U, V, W], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=1e-6)
        
        local_best = float('inf')
        for step in range(adam_steps):
            opt.zero_grad()
            recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            loss = torch.sum((recon - T_torch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], 1.0)
            opt.step()
            sched.step()
            
            res = loss.item() ** 0.5
            if res < local_best:
                local_best = res
        
        # LBFGS polish
        if local_best < 0.5:
            U2 = torch.nn.Parameter(U.data.clone().contiguous())
            V2 = torch.nn.Parameter(V.data.clone().contiguous())
            W2 = torch.nn.Parameter(W.data.clone().contiguous())
            lopt = torch.optim.LBFGS([U2, V2, W2], lr=1e-2, max_iter=20, 
                                      line_search_fn='strong_wolfe')
            for _ in range(250):
                def closure():
                    lopt.zero_grad()
                    r = torch.einsum('ir,jr,kr->ijk', U2, V2, W2)
                    l = torch.sum((r - T_torch) ** 2)
                    l.backward()
                    return l
                lopt.step(closure)
            with torch.no_grad():
                r = torch.einsum('ir,jr,kr->ijk', U2, V2, W2)
                local_best = torch.sum((r - T_torch) ** 2).item() ** 0.5
        
        if local_best < best_res:
            best_res = local_best
    
    return best_res

# ── Main compression loop ────────────────────────────────────────
def compress_to_rank(target_rank):
    n_delete = 23 - target_rank
    combos = list(itertools.combinations(range(23), n_delete))
    n_combos = len(combos)
    print(f"\n{'='*60}")
    print(f"RANK {target_rank}: deleting {n_delete} terms, {n_combos} combinations")
    print(f"{'='*60}")
    
    # Phase 1: Quick screening — warm-start only, fewer steps
    print(f"\nPhase 1: Quick screen (2k Adam steps, warm-start only)...")
    quick_results = []
    t0 = time.time()
    for i, del_idx in enumerate(combos):
        keep = [j for j in range(23) if j not in del_idx]
        
        # Ultra-fast: just warm-start, 2000 steps
        R = len(keep)
        U_init = torch.tensor(U0[:, keep], dtype=torch.float64, device=device)
        V_init = torch.tensor(V0[:, keep], dtype=torch.float64, device=device)
        W_init = torch.tensor(W0[:, keep], dtype=torch.float64, device=device)
        
        U = torch.nn.Parameter(U_init.clone())
        V = torch.nn.Parameter(V_init.clone())
        W = torch.nn.Parameter(W_init.clone())
        
        opt = torch.optim.Adam([U, V, W], lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000, eta_min=1e-5)
        
        local_best = float('inf')
        for step in range(2000):
            opt.zero_grad()
            recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            loss = torch.sum((recon - T_torch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], 1.0)
            opt.step()
            sched.step()
            res = loss.item() ** 0.5
            if res < local_best:
                local_best = res
        
        quick_results.append((local_best, del_idx))
        
        if (i + 1) % 50 == 0 or i == n_combos - 1:
            elapsed = time.time() - t0
            best_so_far = min(r[0] for r in quick_results)
            print(f"  [{i+1}/{n_combos}] best_quick={best_so_far:.6e}  ({elapsed:.0f}s)")
    
    # Sort by residual
    quick_results.sort(key=lambda x: x[0])
    
    print(f"\n  Top 10 quick-screen results:")
    for rank_i, (res, del_idx) in enumerate(quick_results[:10]):
        print(f"    #{rank_i+1}: del={del_idx} res={res:.6e}")
    
    # Phase 2: Deep optimization on top candidates
    n_deep = min(20, len(quick_results))
    print(f"\nPhase 2: Deep optimize top {n_deep} (20k Adam + LBFGS, 3 restarts)...")
    
    deep_results = []
    for rank_i, (quick_res, del_idx) in enumerate(quick_results[:n_deep]):
        keep = [j for j in range(23) if j not in del_idx]
        res = optimize_subset(keep, n_restarts=3, adam_steps=20000)
        deep_results.append((res, del_idx))
        print(f"    #{rank_i+1}: del={del_idx} quick={quick_res:.6e} → deep={res:.6e}")
        if res < 1e-8:
            print(f"\n  *** EXACT DECOMPOSITION FOUND AT RANK {target_rank}! ***")
            print(f"  Delete terms: {del_idx}")
            return res, del_idx
    
    deep_results.sort(key=lambda x: x[0])
    best_res, best_del = deep_results[0]
    print(f"\n  Best at rank {target_rank}: res={best_res:.6e}, del={best_del}")
    return best_res, best_del

# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    for target in [22, 21, 20]:
        res, del_idx = compress_to_rank(target)
        if res < 1e-8:
            print(f"\nSUCCESS: Exact rank-{23 - len(del_idx)} found!")
            break
        print(f"\nRank {target}: best residual = {res:.6e} (not exact)")
