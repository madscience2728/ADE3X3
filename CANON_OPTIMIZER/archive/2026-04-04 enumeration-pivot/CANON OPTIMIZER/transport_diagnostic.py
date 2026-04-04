"""
DIAGNOSTIC: Trace exactly what _apply_factors produces for the standard
algorithm seed factors and compare to the expected standard factors.
"""
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

def _apply_factors(pi, eps, a, b, g):
    def _pm(e): return np.eye(3)[[0,2,1],:] if e else np.eye(3)
    P = [_pm(eps[i]) for i in range(3)]
    pi_inv = [0]*3
    for i in range(3): pi_inv[pi[i]] = i
    facs = {(0,1): a, (1,2): b, (0,2): g}
    res = {}
    for (x,y), nm in [((0,1),'a'), ((1,2),'b'), ((0,2),'g')]:
        oa, ob = pi_inv[x], pi_inv[y]
        key = (min(oa,ob), max(oa,ob))
        F = facs[key].T if (oa,ob) != key else facs[key].copy()
        res[nm] = P[x] @ F @ P[y].T
    return res['a'], res['b'], res['g']

# Standard algorithm: term (r,s,u) has α=e_r e_s^T, β=e_s e_u^T, γ=e_r e_u^T
def e(i): v = np.zeros(3); v[i] = 1; return v

flush("Tracing the Edge orbit from seed (0,0,1)")
flush("Standard: α=e_0 e_0^T, β=e_0 e_1^T, γ=e_0 e_1^T")
flush()

seed = (0, 0, 1)
a_seed = np.outer(e(0), e(0))  # e_r e_s^T for (r,s,u)=(0,0,1)
b_seed = np.outer(e(0), e(1))  # e_s e_u^T
g_seed = np.outer(e(0), e(1))  # e_r e_u^T

flush(f"Seed α:\n{a_seed}\nSeed β:\n{b_seed}\nSeed γ:\n{g_seed}\n")

# Find all transporters for the Edge orbit
orbit = set()
transporters = {}
for pi, eps in _GROUP:
    t = _apply_triple(pi, eps, seed)
    if t not in transporters:
        transporters[t] = (pi, eps)
        orbit.add(t)

for t in sorted(orbit):
    pi, eps = transporters[t]
    r, s, u = t
    
    # Expected standard factors
    a_exp = np.outer(e(r), e(s))
    b_exp = np.outer(e(s), e(u))
    g_exp = np.outer(e(r), e(u))
    
    # Transported factors
    a_got, b_got, g_got = _apply_factors(pi, eps, a_seed, b_seed, g_seed)
    
    err_a = np.linalg.norm(a_exp - a_got)
    err_b = np.linalg.norm(b_exp - b_got)
    err_g = np.linalg.norm(g_exp - g_got)
    
    status = "✓" if max(err_a, err_b, err_g) < 1e-10 else "✗"
    flush(f"  {t}: π={list(pi)} ε={[int(x) for x in eps]}  "
          f"err(α)={err_a:.1e} err(β)={err_b:.1e} err(γ)={err_g:.1e}  {status}")
    
    if max(err_a, err_b, err_g) > 1e-10:
        flush(f"    Expected α:\n{a_exp}")
        flush(f"    Got α:\n{a_got}")
        flush(f"    Expected β:\n{b_exp}")
        flush(f"    Got β:\n{b_got}")

flush("\n\nTracing the Face orbit from seed (0,1,1)")
seed2 = (0, 1, 1)
a_seed2 = np.outer(e(0), e(1))
b_seed2 = np.outer(e(1), e(1))
g_seed2 = np.outer(e(0), e(1))

transporters2 = {}
for pi, eps in _GROUP:
    t = _apply_triple(pi, eps, seed2)
    if t not in transporters2:
        transporters2[t] = (pi, eps)

for t in sorted(transporters2):
    pi, eps = transporters2[t]
    r, s, u = t
    a_exp = np.outer(e(r), e(s))
    b_exp = np.outer(e(s), e(u))
    g_exp = np.outer(e(r), e(u))
    a_got, b_got, g_got = _apply_factors(pi, eps, a_seed2, b_seed2, g_seed2)
    err_a = np.linalg.norm(a_exp - a_got)
    err_b = np.linalg.norm(b_exp - b_got)
    err_g = np.linalg.norm(g_exp - g_got)
    status = "✓" if max(err_a, err_b, err_g) < 1e-10 else "✗"
    flush(f"  {t}: err(α)={err_a:.1e} err(β)={err_b:.1e} err(γ)={err_g:.1e}  {status}")
    if max(err_a, err_b, err_g) > 1e-10:
        flush(f"    Expected α:\n{a_exp}")
        flush(f"    Got α:\n{a_got}")
