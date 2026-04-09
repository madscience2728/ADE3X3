"""
Two jobs:
1. Run 50 restarts at R=22, save best decomposition
2. Project best residual onto standard 27-term basis to identify which O3 terms are missing
"""
import numpy as np
from scipy.optimize import minimize
import json, os

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

# Standard rank-1 basis: outer products of T's nonzero terms
# Each term e_{rsu} = e_{3r+s} ⊗ e_{3s+u} ⊗ e_{3r+u}
basis_terms = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            a = np.zeros(9); a[3*r+s] = 1.0
            b = np.zeros(9); b[3*s+u] = 1.0
            c = np.zeros(9); c[3*r+u] = 1.0
            basis_terms.append(((r,s,u), np.einsum('i,j,k->ijk', a, b, c)))

R = 22
DIM = 9

def obj(x):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
    loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC = np.einsum('ijl,ki,kj->kl', E, A, B)
    return loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])

rng = np.random.default_rng(1337)
best_frob = np.inf
best_x = None

print(f"Running 50 restarts at R={R}...")
for trial in range(50):
    scale = rng.uniform(0.3, 1.5) / R**0.5
    x0 = rng.standard_normal(R * 3 * DIM) * scale
    res = minimize(obj, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10})
    frob = np.sqrt(max(0.0, 2*res.fun))
    if frob < best_frob:
        best_frob = frob
        best_x = res.x.copy()
    if (trial+1) % 10 == 0:
        print(f"  trial {trial+1:>3}/50  best_frob={best_frob:.8f}  (ε²={best_frob**2:.6f})")

print(f"\nBest ε = {best_frob:.8f},  ε² = {best_frob**2:.8f}")
print(f"1/56 = {1/56:.8f},  diff = {abs(best_frob**2 - 1/56):.2e}")

# ── Project residual onto standard 27-term basis ────────────────────────────
A = best_x[:R*DIM].reshape(R, DIM)
B = best_x[R*DIM:2*R*DIM].reshape(R, DIM)
C = best_x[2*R*DIM:].reshape(R, DIM)
T_approx = np.einsum('ki,kj,kl->ijl', A, B, C)
resid = T - T_approx  # what the approximation is MISSING

print(f"\n── Standard-basis projection of residual ──")
print(f"(positive = under-represented, negative = over-represented)")
print(f"{'rsu':>6}  {'orbit':>6}  {'proj':>10}  {'|proj|':>8}")
print("-" * 45)

# Classify orbits
def classify(r, s, u):
    zeros = (r==0) + (s==0) + (u==0)
    return f"O{3-zeros}"

projs = []
for (r,s,u), B_term in basis_terms:
    # projection: <resid, B_term> / ||B_term||² — but B_term is unit norm (it's a standard e_i⊗e_j⊗e_k)
    p = float(np.tensordot(resid, B_term, axes=3))
    orb = classify(r, s, u)
    projs.append(((r,s,u), orb, p))

projs.sort(key=lambda x: abs(x[2]), reverse=True)
for (r,s,u), orb, p in projs:
    marker = " <<<<" if abs(p) > 0.05 else ""
    print(f"({r},{s},{u})  {orb:>6}  {p:>10.6f}  {abs(p):>8.6f}{marker}")

print(f"\nOrbit summary (sum of |proj|):")
from collections import defaultdict
by_orbit = defaultdict(list)
for (r,s,u), orb, p in projs:
    by_orbit[orb].append(p)
for orb in ['O0','O1','O2','O3']:
    ps = by_orbit[orb]
    print(f"  {orb}: n={len(ps)}  sum_abs={sum(abs(x) for x in ps):.6f}  rms={np.sqrt(np.mean(np.array(ps)**2)):.6f}")

# Save best
os.makedirs('CANON_ADE_EPSILON/results', exist_ok=True)
out = {
    'rank': R,
    'frobenius': best_frob,
    'frob_sq': best_frob**2,
    'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist()
}
with open('CANON_ADE_EPSILON/results/rank22_best.json', 'w') as f:
    json.dump(out, f)
print(f"\nSaved to CANON_ADE_EPSILON/results/rank22_best.json")
