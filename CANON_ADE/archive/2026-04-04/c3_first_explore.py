# coding: utf-8
"""
c3_first_explore.py — C3-first exploration of the algebra space.

All losses (associativity, C5, power-assoc, Jacobi) are PRECOMPUTED as
quadratic forms in family parameters alpha, so gradient descent runs
in milliseconds per step.

Families:
  Root:    C4 (baked in) + C2 (r-blindness) → 27-dim
  HULL_ZD: Root + hull zero-divisor → 6-dim

STRUCTURED: C5, assoc, joint C5+assoc in both families
WILDCARD:   power-assoc, Jacobi/Lie, random survey, max R_rank
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
from axiom_bfs import (build_canon, enc_hullzd, N, N_ORBITS, orbit_of, OCS)
from canon_constraints import (
    INTERIOR, IDX, C3_FiberConstraint, C4_Symmetry, C5_SedenionZDGraph,
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
rng = np.random.default_rng(2026)

# ==================================================================
# Build families
# ==================================================================
print("=" * 70)
print("PHASE D: C3-FIRST EXPLORATION")
print("=" * 70)

def build_nullbasis(cs):
    A, b = cs.build()
    if A.shape[0] == 0:
        return np.eye(N_ORBITS), np.zeros(N_ORBITS)
    U, sv, Vt = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * sv[0] * 1e-10
    rank = int(np.sum(sv > tol))
    null_basis = Vt[rank:].T
    part = np.linalg.lstsq(A, b, rcond=None)[0] if np.max(np.abs(b)) > 1e-12 else np.zeros(N_ORBITS)
    return null_basis, part

def expand_orbit(v):
    f = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f[i, j, k] = v[orbit_of(i, j, k)]
    return f

cs_root = build_canon()
cs_hz = build_canon(); enc_hullzd(cs_hz)

nb_hz, _ = build_nullbasis(cs_hz)
nb_root, _ = build_nullbasis(cs_root)
DIM_HZ = nb_hz.shape[1]
DIM_ROOT = nb_root.shape[1]
print(f"  Root d.o.f.: {DIM_ROOT},  HULL_ZD d.o.f.: {DIM_HZ}")

print("  Expanding basis tensors...", flush=True)
t0 = time.time()
Fv_hz = [expand_orbit(nb_hz[:, vi]) for vi in range(DIM_HZ)]
Fv_root = [expand_orbit(nb_root[:, vi]) for vi in range(DIM_ROOT)]
print(f"  Done in {time.time()-t0:.1f}s")

# ==================================================================
# Precompute quadratic forms
# ==================================================================
print("\n  Precomputing quadratic constraint matrices...", flush=True)
t0 = time.time()

# --- C5 ---
ind_a, ind_b = C5_SedenionZDGraph._build_zd_indicators()
n_edges = ind_a.shape[0]

def build_c5_qform(Fv, DIM):
    n_c = n_edges * N
    M = np.zeros((n_c, DIM))
    for vi in range(DIM):
        C = np.einsum('ea,abz,eb->ez', ind_a, Fv[vi], ind_b)
        M[:, vi] = C.ravel()
    return M.T @ M

Q_c5_hz = build_c5_qform(Fv_hz, DIM_HZ)
Q_c5_root = build_c5_qform(Fv_root, DIM_ROOT)
print(f"    C5 forms built", flush=True)

# --- Associativity ---
def build_assoc_qforms(Fv, DIM):
    mats = []; seen = {}
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    M = np.zeros((DIM, DIM))
                    for i in range(DIM):
                        for j in range(DIM):
                            M[i, j] = (np.dot(Fv[i][a,b,:], Fv[j][:,c,d])
                                       - np.dot(Fv[i][b,c,:], Fv[j][a,:,d]))
                    if np.max(np.abs(M)) < 1e-12: continue
                    Ms = (M + M.T) / 2.0
                    sc = np.max(np.abs(Ms))
                    key = tuple(np.round(Ms.flatten()/sc, 4))
                    if key not in seen:
                        seen[key] = len(mats); mats.append(Ms/sc)
    return np.array(mats) if mats else np.zeros((0, DIM, DIM))

print("    Assoc HULL_ZD...", flush=True)
Mc_hz = build_assoc_qforms(Fv_hz, DIM_HZ)
print(f"    Assoc HULL_ZD: {Mc_hz.shape[0]} forms", flush=True)

print("    Assoc Root (27-dim, may be slow)...", flush=True)
Mc_root = build_assoc_qforms(Fv_root, DIM_ROOT)
print(f"    Assoc Root: {Mc_root.shape[0]} forms", flush=True)

# --- Power-associativity ---
def build_pa_qforms(Fv, DIM):
    mats = []; seen = {}
    for i in range(N):
        for l in range(N):
            M = np.zeros((DIM, DIM))
            for v in range(DIM):
                for w in range(DIM):
                    M[v,w] = (np.dot(Fv[v][i,i,:], Fv[w][:,i,l])
                              - np.dot(Fv[v][i,i,:], Fv[w][i,:,l]))
            if np.max(np.abs(M)) < 1e-12: continue
            Ms = (M + M.T) / 2.0
            sc = np.max(np.abs(Ms))
            key = tuple(np.round(Ms.flatten()/sc, 4))
            if key not in seen:
                seen[key] = len(mats); mats.append(Ms/sc)
    return np.array(mats) if mats else np.zeros((0, DIM, DIM))

print("    PA HULL_ZD...", flush=True)
Mc_pa_hz = build_pa_qforms(Fv_hz, DIM_HZ)
print(f"    PA HULL_ZD: {Mc_pa_hz.shape[0]} forms", flush=True)

# --- Jacobi (commutator bracket) ---
Cv_hz = [Fv_hz[v] - Fv_hz[v].transpose(1,0,2) for v in range(DIM_HZ)]

def build_jacobi_qforms(Cv, DIM):
    mats = []; seen = {}
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    M = np.zeros((DIM, DIM))
                    for v in range(DIM):
                        for w in range(DIM):
                            M[v,w] = (np.dot(Cv[v][j,k,:], Cv[w][i,:,l])
                                      + np.dot(Cv[v][k,i,:], Cv[w][j,:,l])
                                      + np.dot(Cv[v][i,j,:], Cv[w][k,:,l]))
                    if np.max(np.abs(M)) < 1e-12: continue
                    Ms = (M + M.T) / 2.0
                    sc = np.max(np.abs(Ms))
                    key = tuple(np.round(Ms.flatten()/sc, 4))
                    if key not in seen:
                        seen[key] = len(mats); mats.append(Ms/sc)
    return np.array(mats) if mats else np.zeros((0, DIM, DIM))

print("    Jacobi HULL_ZD...", flush=True)
Mc_jac_hz = build_jacobi_qforms(Cv_hz, DIM_HZ)
print(f"    Jacobi HULL_ZD: {Mc_jac_hz.shape[0]} forms", flush=True)

print(f"  Precomputation done in {time.time()-t0:.1f}s")

# ==================================================================
# Fast loss/grad
# ==================================================================
def qf_loss(Mc, a):
    vals = np.einsum('i,tij,j->t', a, Mc, a)
    return np.sum(vals**2)

def qf_grad(Mc, a):
    vals = np.einsum('i,tij,j->t', a, Mc, a)
    return 2.0 * np.einsum('t,tij,j->i', vals, Mc + Mc.transpose(0,2,1), a)

def c5_loss(Q, a): return float(a @ Q @ a)
def c5_grad(Q, a): return 2.0 * Q @ a

def gd_sphere(loss_fn, grad_fn, a0, lr=1e-2, steps=3000, tol=1e-14):
    a = a0.copy()
    for s in range(steps):
        l = loss_fn(a)
        if l < tol: return a, l, s
        g = grad_fn(a)
        g -= np.dot(g, a) * a
        a -= lr * g; a /= np.linalg.norm(a)
        if s % 500 == 499: lr *= 0.5
    return a, loss_fn(a), steps

def make_f(Fv, alpha):
    return sum(alpha[i] * Fv[i] for i in range(len(alpha)))

def algebra_props(f, label=""):
    L_ranks = [np.linalg.matrix_rank(f[i], tol=1e-8) for i in range(N)]
    R_ranks = [np.linalg.matrix_rank(f[:,i,:], tol=1e-8) for i in range(N)]
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i,j] = np.trace(f[i] @ f[j])
    return {"L_min": min(L_ranks), "L_max": max(L_ranks),
            "R_min": min(R_ranks), "R_max": max(R_ranks),
            "kill": np.linalg.matrix_rank(K, tol=1e-8),
            "comm": np.max(np.abs(f - f.transpose(1,0,2))), "label": label}

def pp(p):
    print(f"  [{p['label']}] L=[{p['L_min']},{p['L_max']}] R=[{p['R_min']},{p['R_max']}] "
          f"Kill={p['kill']} comm={p['comm']:.2e}")

def multi_restart(loss_fn, grad_fn, dim, n=50, lr=1e-2, steps=5000):
    results = []
    for _ in range(n):
        a0 = rng.standard_normal(dim); a0 /= np.linalg.norm(a0)
        sol, loss, st = gd_sphere(loss_fn, grad_fn, a0, lr=lr, steps=steps)
        results.append((loss, sol))
    results.sort(key=lambda x: x[0])
    return results

# ==================================================================
# TEST 1: C5 in HULL_ZD
# ==================================================================
print("\n" + "=" * 70)
print("TEST 1: HULL_ZD — C5 loss minimization (50 restarts)")
print("=" * 70)
res = multi_restart(lambda a: c5_loss(Q_c5_hz, a), lambda a: c5_grad(Q_c5_hz, a), DIM_HZ, 50)
print(f"  Top 5 C5 losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
f_ = make_f(Fv_hz, res[0][1]); pp(algebra_props(f_, "HZ C5-best"))
print(f"  Assoc loss: {qf_loss(Mc_hz, res[0][1]):.4e}")

# ==================================================================
# TEST 2: Assoc in HULL_ZD (verify known solution)
# ==================================================================
print("\n" + "=" * 70)
print("TEST 2: HULL_ZD — associativity (50 restarts)")
print("=" * 70)
res = multi_restart(lambda a: qf_loss(Mc_hz, a), lambda a: qf_grad(Mc_hz, a), DIM_HZ, 50)
print(f"  Top 5 assoc losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
n_ex = sum(1 for l,_ in res if l < 1e-10)
print(f"  Exact assoc solutions: {n_ex}/50")
if n_ex > 0:
    f_ = make_f(Fv_hz, res[0][1]); pp(algebra_props(f_, "HZ assoc"))
    print(f"  C5 loss: {c5_loss(Q_c5_hz, res[0][1]):.4e}")

# ==================================================================
# TEST 3: Joint C5+assoc in HULL_ZD
# ==================================================================
print("\n" + "=" * 70)
print("TEST 3: HULL_ZD — joint C5 + assoc (100 restarts)")
print("=" * 70)
def jl_hz(a): return c5_loss(Q_c5_hz, a) + qf_loss(Mc_hz, a)
def jg_hz(a): return c5_grad(Q_c5_hz, a) + qf_grad(Mc_hz, a)
res = multi_restart(jl_hz, jg_hz, DIM_HZ, 100)
print(f"  Top 5 joint losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
for i in range(min(3, len(res))):
    s = res[i][1]
    print(f"    #{i+1}: C5={c5_loss(Q_c5_hz,s):.4e}  assoc={qf_loss(Mc_hz,s):.4e}")

# ==================================================================
# TEST 4: C5 in 27-dim root
# ==================================================================
print("\n" + "=" * 70)
print("TEST 4: Root 27-dim — C5 (100 restarts)")
print("=" * 70)
res = multi_restart(lambda a: c5_loss(Q_c5_root, a), lambda a: c5_grad(Q_c5_root, a), DIM_ROOT, 100)
print(f"  Top 5 C5 losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
n_ex = sum(1 for l,_ in res if l < 1e-8)
print(f"  Near-zero C5 (< 1e-8): {n_ex}/100")
if res[0][0] < 1.0:
    f_ = make_f(Fv_root, res[0][1]); pp(algebra_props(f_, "Root C5-best"))

# ==================================================================
# TEST 5: Joint C5+assoc in 27-dim root
# ==================================================================
print("\n" + "=" * 70)
print("TEST 5: Root 27-dim — joint C5 + 0.1*assoc (100 restarts)")
print("=" * 70)
def jl_r(a): return c5_loss(Q_c5_root, a) + 0.1 * qf_loss(Mc_root, a)
def jg_r(a): return c5_grad(Q_c5_root, a) + 0.1 * qf_grad(Mc_root, a)
res = multi_restart(jl_r, jg_r, DIM_ROOT, 100)
print(f"  Top 5 joint losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
for i in range(min(3, len(res))):
    s = res[i][1]
    print(f"    #{i+1}: C5={c5_loss(Q_c5_root,s):.4e}  assoc={qf_loss(Mc_root,s):.4e}")

# ==================================================================
# W1: Power-associativity in HULL_ZD
# ==================================================================
print("\n" + "=" * 70)
print("W1: Power-associativity in HULL_ZD (50 restarts)")
print("=" * 70)
if Mc_pa_hz.shape[0] > 0:
    res = multi_restart(lambda a: qf_loss(Mc_pa_hz, a), lambda a: qf_grad(Mc_pa_hz, a), DIM_HZ, 50)
    print(f"  Top 5 PA losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
    n_ex = sum(1 for l,_ in res if l < 1e-10)
    print(f"  Exact PA solutions: {n_ex}/50")
    if n_ex > 0:
        f_ = make_f(Fv_hz, res[0][1]); pp(algebra_props(f_, "HZ PA"))
        print(f"  C5={c5_loss(Q_c5_hz, res[0][1]):.4e}  assoc={qf_loss(Mc_hz, res[0][1]):.4e}")
else:
    print("  PA trivially satisfied (0 constraint forms)")

# ==================================================================
# W2: PA + C5 in HULL_ZD
# ==================================================================
print("\n" + "=" * 70)
print("W2: PA + C5 in HULL_ZD (50 restarts)")
print("=" * 70)
if Mc_pa_hz.shape[0] > 0:
    def pac5l(a): return qf_loss(Mc_pa_hz, a) + c5_loss(Q_c5_hz, a)
    def pac5g(a): return qf_grad(Mc_pa_hz, a) + c5_grad(Q_c5_hz, a)
    res = multi_restart(pac5l, pac5g, DIM_HZ, 50)
    print(f"  Top 5 PA+C5: {[f'{l:.4e}' for l,_ in res[:5]]}")
    for i in range(min(3, len(res))):
        s = res[i][1]
        print(f"    #{i+1}: PA={qf_loss(Mc_pa_hz,s):.4e}  C5={c5_loss(Q_c5_hz,s):.4e}")
else:
    print("  PA trivial, see TEST 1 for C5")

# ==================================================================
# W3: Jacobi/Lie in HULL_ZD
# ==================================================================
print("\n" + "=" * 70)
print("W3: Lie/Jacobi in HULL_ZD (50 restarts)")
print("=" * 70)
if Mc_jac_hz.shape[0] > 0:
    res = multi_restart(lambda a: qf_loss(Mc_jac_hz, a), lambda a: qf_grad(Mc_jac_hz, a), DIM_HZ, 50)
    print(f"  Top 5 Jacobi losses: {[f'{l:.4e}' for l,_ in res[:5]]}")
    n_lie = sum(1 for l,_ in res if l < 1e-10)
    print(f"  Exact Lie algebras: {n_lie}/50")
    if n_lie > 0:
        f_ = make_f(Fv_hz, res[0][1]); pp(algebra_props(f_, "HZ Lie"))
        print(f"  C5={c5_loss(Q_c5_hz, res[0][1]):.4e}  assoc={qf_loss(Mc_hz, res[0][1]):.4e}")
else:
    print("  Jacobi trivially satisfied — commutator always Lie in this family.")

# ==================================================================
# W4: Random landscape survey (27-dim root)
# ==================================================================
print("\n" + "=" * 70)
print("W4: Random landscape survey (500 samples, 27-dim root)")
print("=" * 70)
survey = []
for t in range(500):
    a = rng.standard_normal(DIM_ROOT); a /= np.linalg.norm(a)
    f_t = make_f(Fv_root, a)
    p = algebra_props(f_t, f"s{t}")
    p["c5"] = c5_loss(Q_c5_root, a)
    p["assoc"] = qf_loss(Mc_root, a)
    survey.append(p)
L_mins = [p["L_min"] for p in survey]
R_mins = [p["R_min"] for p in survey]
kills = [p["kill"] for p in survey]
print(f"  L_rank_min: {min(L_mins)}-{max(L_mins)} mean={np.mean(L_mins):.1f}")
print(f"  R_rank_min: {min(R_mins)}-{max(R_mins)} mean={np.mean(R_mins):.1f}")
print(f"  Kill_rank:  {min(kills)}-{max(kills)} mean={np.mean(kills):.1f}")
print(f"  C5 range:   [{min(p['c5'] for p in survey):.2e}, {max(p['c5'] for p in survey):.2e}]")
best_R = max(survey, key=lambda p: p["R_min"])
print(f"  Best R_min={best_R['R_min']}:"); pp(best_R)

# ==================================================================
# W5: Max R_rank in root (numerical gradient, 200 samples + refine)
# ==================================================================
print("\n" + "=" * 70)
print("W5: Maximize min(R_rank) in 27-dim root")
print("=" * 70)
def r_loss(alpha):
    f = make_f(Fv_root, alpha)
    loss = 0.0
    for i in range(N):
        sv = np.linalg.svd(f[:,i,:], compute_uv=False)
        loss += np.sum(np.exp(-100*sv))
    return loss

best_rl, best_ar = float('inf'), None
for _ in range(200):
    a = rng.standard_normal(DIM_ROOT); a /= np.linalg.norm(a)
    l = r_loss(a)
    if l < best_rl: best_rl = l; best_ar = a.copy()
print(f"  Best seed R-loss: {best_rl:.4e}")

a = best_ar.copy(); lr = 1e-3
for step in range(300):
    l = r_loss(a)
    if step % 100 == 0:
        f_t = make_f(Fv_root, a)
        rr = [np.linalg.matrix_rank(f_t[:,i,:], tol=1e-8) for i in range(N)]
        print(f"    step {step}: loss={l:.4e} R min={min(rr)} max={max(rr)}")
    eps = 1e-6; g = np.zeros(DIM_ROOT)
    for d in range(DIM_ROOT):
        a[d] += eps; g[d] = (r_loss(a) - l)/eps; a[d] -= eps
    g -= np.dot(g,a)*a; a -= lr*g; a /= np.linalg.norm(a)
    if step % 100 == 99: lr *= 0.7

f_ = make_f(Fv_root, a); pp(algebra_props(f_, "Root max-R"))
print(f"  C5={c5_loss(Q_c5_root, a):.4e}  assoc={qf_loss(Mc_root, a):.4e}")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)
