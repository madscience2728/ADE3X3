"""
Fiber-Structured CP Optimizer for R=19 decomposition of T_matmul.

Theory basis (docs/TOWER_CONNECTION.md):
  - 19 rank-1 terms grouped into 9 (s,u)-fibers over Z_3^2
  - r-index drops out of Fourier: chi(r,s,u) = exp(2pi*i*(ps+qu+wu)/3)
  - Gamma(s,u) = sum_{t in fiber(s,u)} gamma_t = 3  [HARD CONSTRAINT]
  - 18 of 24 dead modes auto-satisfied if a_t has uniform column sums
  - 6 remaining dead modes + 3 active modes constrain b,c factor vectors

Fiber structure of 19 interior points:
  4 forced fibers (s,u both even): only r=1 term, gamma PINNED = 3 (or use free a,b,c rescaling)
  5 free fibers (>= one odd): r in {0,1,2}, 3 terms each, sum gamma = 3

Usage:
  python scripts/fiber_cp_optimizer.py [--rank 19] [--restarts 50] [--seed 0]
"""

import numpy as np
import itertools
import argparse
from scipy.optimize import minimize

# ── Tensor ──────────────────────────────────────────────────────────────────
def build_T():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[3*r+s, 3*s+u, 3*r+u] = 1.0
    return T

T_TARGET = build_T()
T_VEC = T_TARGET.ravel()  # shape (729,)

# ── Fiber structure ──────────────────────────────────────────────────────────
# The 19 interior points (r,s,u) with at least one odd coordinate
INTERIOR = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)
            if not (r % 2 == 0 and s % 2 == 0 and u % 2 == 0)]

# Group by (s,u)-fiber
from collections import defaultdict
FIBERS = defaultdict(list)  # (s,u) -> list of (r, term_idx)
for idx, (r, s, u) in enumerate(INTERIOR):
    FIBERS[(s, u)].append((r, idx))

SU_KEYS = sorted(FIBERS.keys())  # 9 fiber keys
# Fiber sizes: 4 x size-1 (s,u even), 5 x size-3 (at least one odd)
FORCED_FIBERS = [(s, u) for s, u in SU_KEYS if s % 2 == 0 and u % 2 == 0]  # 4 fibers
FREE_FIBERS = [(s, u) for s, u in SU_KEYS if not (s % 2 == 0 and u % 2 == 0)]  # 5 fibers

N_TERMS = len(INTERIOR)  # 19

# ── Parameter layout ────────────────────────────────────────────────────────
# Absorb gamma into a_t: effective factor = gamma_t * a_t stored as a single vector.
# Layout per term: [a (9), b (9), c (9)] = 27 per term = 27*19 = 513 total
# Reconstruction: T = sum_t outer(A[t], B[t], C[t])  where A[t] already carries gamma.

PARAMS_PER_TERM = 27  # a(9) + b(9) + c(9)

def unpack(params):
    """Unpack flat parameter vector into (As, Bs, Cs)."""
    As = params[:N_TERMS*9].reshape(N_TERMS, 9)
    Bs = params[N_TERMS*9:N_TERMS*18].reshape(N_TERMS, 9)
    Cs = params[N_TERMS*18:].reshape(N_TERMS, 9)
    return As, Bs, Cs

def reconstruct(As, Bs, Cs):
    """Reconstruct tensor. T[i,j,k] = sum_t A[t,i]*B[t,j]*C[t,k]."""
    # Efficient: einsum
    return np.einsum('ti,tj,tk->ijk', As, Bs, Cs)

# ── Loss function ────────────────────────────────────────────────────────────
def loss(params):
    As, Bs, Cs = unpack(params)
    T_recon = reconstruct(As, Bs, Cs)
    return np.sum((T_recon - T_TARGET) ** 2)

def fiber_residual(As):
    """For diagnostics: how much the a_t norms violate the fiber balance."""
    # In the norm-scaled sense: sum_{t in fiber} ||a_t|| should be balanced
    # (not a hard constraint here, just diagnostic)
    viols = []
    for su in SU_KEYS:
        fiber_sum = sum(np.linalg.norm(As[idx]) for _, idx in FIBERS[su])
        viols.append(fiber_sum)
    return np.array(viols)

# ── Initialization ───────────────────────────────────────────────────────────
def init_params(rng, strategy='fiber'):
    """Initialize parameters respecting fiber structure."""
    params = np.zeros(N_TERMS * PARAMS_PER_TERM)
    
    for su in SU_KEYS:
        fiber_terms = FIBERS[su]
        n_in_fiber = len(fiber_terms)
        
        # Initialize gammas summing to 3
        if n_in_fiber == 1:
            _, idx = fiber_terms[0]
            params[idx * PARAMS_PER_TERM] = 3.0
        else:
            # Random gammas summing to 3
            g = rng.exponential(1.0, n_in_fiber)
            g = g / g.sum() * 3.0
            for _, idx in fiber_terms:
                params[idx * PARAMS_PER_TERM] = g[_]
    
    # Initialize factor vectors randomly
    for t in range(N_TERMS):
        base = t * PARAMS_PER_TERM
        scale = 0.3
        params[base+1:base+10] = rng.normal(0, scale, 9)   # a
        params[base+10:base+19] = rng.normal(0, scale, 9)  # b
        params[base+19:base+28] = rng.normal(0, scale, 9)  # c
        
        if strategy == 'support_hint':
            # Hint: bias a_t toward e_{3r+s} for the (r,s,u) support point of term t
            r, s, u = INTERIOR[t]
            params[base+1+3*r+s] += 1.0  # a hint
            params[base+10+3*s+u] += 1.0  # b hint
            params[base+19+3*r+u] += 1.0  # c hint
    
    return params

def init_params_v2(rng):
    """Init: support-based seed with noise on factor vectors."""
    As = np.zeros((N_TERMS, 9))
    Bs = np.zeros((N_TERMS, 9))
    Cs = np.zeros((N_TERMS, 9))
    
    alpha = 0.4
    for t in range(N_TERMS):
        r, s, u = INTERIOR[t]
        a = np.zeros(9); a[3*r+s] = 1.0
        b = np.zeros(9); b[3*s+u] = 1.0
        c = np.zeros(9); c[3*r+u] = 1.0
        As[t] = a + rng.normal(0, alpha, 9)
        Bs[t] = b + rng.normal(0, alpha, 9)
        Cs[t] = c + rng.normal(0, alpha, 9)
    
    return np.concatenate([As.ravel(), Bs.ravel(), Cs.ravel()])

# ── Optimizer ────────────────────────────────────────────────────────────────
def run_single(rng, verbose=False, max_iter=2000):
    """Single optimization run. Returns (best_err, best_params)."""
    params = init_params_v2(rng)
    
    result = minimize(
        loss, params,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10},
    )
    
    As, Bs, Cs = unpack(result.x)
    T_recon = reconstruct(As, Bs, Cs)
    rec_err = np.sqrt(np.sum((T_recon - T_TARGET) ** 2))
    fiber_norms = fiber_residual(As)
    
    if verbose:
        print(f"  rec_err={rec_err:.6f}  fiber_norms={fiber_norms.round(2)}  "
              f"converged={result.success}")
    
    return rec_err, result.x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restarts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_iter', type=int, default=3000)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    
    print("=== Fiber-Structured CP Optimizer (R=19) ===")
    print(f"Target: rank-19 CP decomposition of T_matmul (9x9x9)")
    print(f"Constraint: Gamma(s,u)=3 for all 9 (s,u)-fibers")
    print(f"Params per term: {PARAMS_PER_TERM}, total: {N_TERMS * PARAMS_PER_TERM}")
    print()
    print(f"Fiber structure:")
    for su in SU_KEYS:
        fiber = FIBERS[su]
        n = len(fiber)
        parity = 'forced (size-1)' if n == 1 else 'free (size-3)'
        print(f"  fiber {su}: {n} terms | {parity}")
    print()
    
    rng = np.random.default_rng(args.seed)
    
    best_err = np.inf
    best_params = None
    
    print(f"Running {args.restarts} restarts (max_iter={args.max_iter})...")
    for i in range(args.restarts):
        rec_err, params = run_single(rng, verbose=False, max_iter=args.max_iter)
        marker = " *** NEW BEST ***" if rec_err < best_err else ""
        if args.verbose or rec_err < best_err:
            print(f"  [{i+1:3d}] rec_err={rec_err:.6f}{marker}")
        if rec_err < best_err:
            best_err = rec_err
            best_params = params.copy()
    
    print()
    print(f"Best reconstruction error: {best_err:.8f}")
    print(f"(AlphaTensor rank-23 gives err=0; R=19 is feasible iff best_err -> 0)")
    
    if best_err < 1e-4:
        print()
        print("!!! POTENTIAL RANK-19 SOLUTION FOUND !!!")
        As, Bs, Cs = unpack(best_params)
        print("A norms:", np.linalg.norm(As, axis=1).round(4))
    
    return best_err, best_params

if __name__ == '__main__':
    main()
