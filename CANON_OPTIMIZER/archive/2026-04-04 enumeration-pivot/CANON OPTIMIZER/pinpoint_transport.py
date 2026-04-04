"""Pinpoint why seed→transport doesn't match standard algorithm."""
import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()
def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]
def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))
def _apply_factors_fixed(pi, eps, a, b, g):
    def _pm(e): return np.eye(3)[[0,2,1],:] if e else np.eye(3)
    P = [_pm(eps[i]) for i in range(3)]
    pair_to_factor = {(0,1): a, (1,2): b, (0,2): g}
    nf = {}
    for (i,j), nm in [((0,1),'a'), ((1,2),'b'), ((0,2),'g')]:
        oi, oj = pi[i], pi[j]
        key = (min(oi,oj), max(oi,oj))
        F = pair_to_factor[key]
        F_o = F.copy() if oi <= oj else F.T
        nf[nm] = P[i] @ F_o @ P[j].T
    return nf['a'], nf['b'], nf['g']
def e(i): v = np.zeros(3); v[i] = 1; return v

# The Edge orbit: seed (0,0,1)
seed = (0, 0, 1)
a_seed = np.outer(e(0), e(0))  # standard α for (0,0,1): e_r e_s^T = e0 e0^T
b_seed = np.outer(e(0), e(1))  # standard β for (0,0,1): e_s e_u^T = e0 e1^T
g_seed = np.outer(e(0), e(1))  # standard γ for (0,0,1): e_r e_u^T = e0 e1^T

# The problem was: multiple group elements map seed→same target.
# Which one do we pick? They should ALL give the same result IF stabilizer acts trivially.

target = (1, 0, 0)
flush(f"All group elements mapping {seed} → {target}:")

transporters = []
for pi, eps in _GROUP:
    if _apply_triple(pi, eps, seed) == target:
        a_t, b_t, g_t = _apply_factors_fixed(pi, eps, a_seed, b_seed, g_seed)
        transporters.append((pi, eps, a_t, b_t, g_t))
        flush(f"  π={list(pi)} ε={[int(x) for x in eps]}")
        flush(f"    α = {a_t.ravel().tolist()}")
        flush(f"    β = {b_t.ravel().tolist()}")

# Expected for (1,0,0): α=e1 e0^T, β=e0 e0^T, γ=e1 e0^T
flush(f"\nExpected for {target}: α=e1e0^T, β=e0e0^T, γ=e1e0^T")
flush(f"  α = {np.outer(e(1),e(0)).ravel().tolist()}")
flush(f"  β = {np.outer(e(0),e(0)).ravel().tolist()}")

flush(f"\nDo ALL transporters agree? {len(set(str(t[2].ravel().tolist()) for t in transporters)) == 1}")

# Check: do stabilizer elements of the seed produce identity on factors?
flush(f"\nStabilizer of {seed}:")
for pi, eps in _GROUP:
    if _apply_triple(pi, eps, seed) == seed:
        a_t, b_t, g_t = _apply_factors_fixed(pi, eps, a_seed, b_seed, g_seed)
        same_a = np.allclose(a_t, a_seed)
        same_b = np.allclose(b_t, b_seed)
        same_g = np.allclose(g_t, g_seed)
        flush(f"  π={list(pi)} ε={[int(x) for x in eps]}: "
              f"fixes α={same_a} β={same_b} γ={same_g}")
        if not (same_a and same_b and same_g):
            flush(f"    α_seed={a_seed.ravel().tolist()}")
            flush(f"    α_stab={a_t.ravel().tolist()}")
            flush(f"    β_seed={b_seed.ravel().tolist()}")
            flush(f"    β_stab={b_t.ravel().tolist()}")
