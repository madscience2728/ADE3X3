"""Focused search on top kernel candidates with solvability diagnostics."""
import sys, time, numpy as np
sys.path.insert(0, 'CANON OPTIMIZER')
from kernel_sweep import *
from scipy.optimize import minimize

irreps = decompose_irreps()
candidates = enumerate_kernel_candidates(irreps, 10)
dims = [b.shape[1] for b in irreps]

print(f"{len(irreps)} irreps, dims={dims}")
print(f"{len(candidates)} kernel candidates")

# Test a batch of kernels — 20 restarts each, 1000 maxiter
# Focus on getting solvability right

def deep_probe(kidx, combo, n_restarts=30, maxiter=1500):
    K_basis = np.hstack([irreps[i] for i in combo])
    U, _, _ = np.linalg.svd(K_basis, full_matrices=True)
    Q_perp = U[:, K_basis.shape[1]:]  # (19, 9) — K^perp basis
    Q_K = U[:, :K_basis.shape[1]]     # (19, 10) — K basis
    
    loss_fn = make_kernel_loss(Q_perp)
    
    best_loss = np.inf
    best_params = None
    
    for restart in range(n_restarts):
        rng = np.random.default_rng(kidx * 10000 + restart)
        x0 = rng.standard_normal(81) * 0.5
        result = minimize(loss_fn, x0, method="L-BFGS-B",
                          options={"maxiter": maxiter, "disp": False})
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()
    
    alpha, beta, gamma_dummy = expand_seeds(best_params)
    sigma, H, delta = build_H_Delta(alpha, beta)
    nuisance = np.hstack([H, delta])
    
    # Detailed rank analysis at multiple tolerances
    sv_H = np.linalg.svd(H, compute_uv=False)
    sv_N = np.linalg.svd(nuisance, compute_uv=False)
    
    # Check solvability: can we solve Gamma?
    gamma_result, gamma_info = solve_gamma_from_kernel(alpha, beta)
    
    return {
        "kidx": kidx,
        "combo": combo,
        "dims": [dims[i] for i in combo],
        "best_loss": best_loss,
        "sv_H_tail": sv_H[8:].tolist(),  # SVs 8..17
        "sv_N_tail": sv_N[8:].tolist(),
        "rank_H_1e6": int(np.sum(sv_H > 1e-6)),
        "rank_H_1e4": int(np.sum(sv_H > 1e-4)),
        "rank_N_1e6": int(np.sum(sv_N > 1e-6)),
        "gamma_info": gamma_info,
        "solved": gamma_result is not None,
    }


# Run on first 50 candidates (should cover the most promising combos)
# Use parallel execution
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

N = min(50, len(candidates))
print(f"\nProbing {N} kernels × 30 restarts × 1500 maxiter (sequential)")
print("=" * 90)

start = time.time()
results = []

for i in range(N):
    r = deep_probe(i, candidates[i], n_restarts=10, maxiter=800)
    results.append(r)
    dims_str = "+".join(str(d) for d in r["dims"])
    star = " ***" if r["solved"] else ""
    print(f"  K_{r['kidx']:>3d} [{dims_str:>15s}] "
          f"loss={r['best_loss']:.3e} "
          f"rk(H@1e-6)={r['rank_H_1e6']:>2d} "
          f"rk(H@1e-4)={r['rank_H_1e4']:>2d} "
          f"rk(N@1e-6)={r['rank_N_1e6']:>2d} "
          f"sv_H_tail=[{','.join(f'{v:.1e}' for v in r['sv_H_tail'][:5])}]"
          f"{star}")

results.sort(key=lambda r: r["best_loss"])
elapsed = time.time() - start

print(f"\n{'=' * 90}")
print(f"Done in {elapsed:.0f}s")
print(f"\nTop 15 by loss:")
for r in results[:15]:
    dims_str = "+".join(str(d) for d in r["dims"])
    print(f"  K_{r['kidx']:>3d} [{dims_str:>15s}] "
          f"loss={r['best_loss']:.6e} "
          f"rk(H)={r['rank_H_1e6']} rk(N)={r['rank_N_1e6']} "
          f"solv={r['solved']}")

solved = [r for r in results if r["solved"]]
if solved:
    print(f"\n*** {len(solved)} EXACT SOLUTIONS ***")
else:
    print("\nNo exact solutions yet.")
    # Show kernels where rank(H) ≤ 10 at some tolerance
    low_rank = [r for r in results if r["rank_H_1e4"] <= 10]
    if low_rank:
        print(f"\n{len(low_rank)} kernels with rank(H)<=10 at tol=1e-4:")
        for r in low_rank:
            dims_str = "+".join(str(d) for d in r["dims"])
            print(f"  K_{r['kidx']:>3d} [{dims_str:>15s}] "
                  f"loss={r['best_loss']:.6e} "
                  f"sv_H_tail={[f'{v:.1e}' for v in r['sv_H_tail']]}")
