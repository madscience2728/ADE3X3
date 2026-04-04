"""Deep search on promising kernel candidates."""
import sys, time, numpy as np
sys.path.insert(0, 'CANON OPTIMIZER')
from kernel_sweep import *

irreps = decompose_irreps()
candidates = enumerate_kernel_candidates(irreps, 10)
dims = [b.shape[1] for b in irreps]

# K_32: the only kernel that reached rank(H)=10 in smoke test
combo = candidates[32]
print(f"Deep search on K_32: combo={combo}")
print(f"Irreps: " + "+".join(f"V{j}({dims[j]})" for j in combo))

r = sweep_kernel(32, combo, irreps, n_restarts=200, maxiter=3000, seed_base=0)
print(f"best_loss={r['best_loss']:.6e}  rk(H)={r['rank_H']}  rk(N)={r['rank_nuisance']}")
print(f"gamma_info={r['gamma_info']}")
print(f"solved={r['solved']}")
