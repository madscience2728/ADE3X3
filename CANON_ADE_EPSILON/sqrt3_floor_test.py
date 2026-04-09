"""
Test conjecture: ε*(R) = √3 for all R < 23, ε*(R) = 0 for R ≥ 23.
For each rank in {13, 19, 20, 21, 22, 23, 27}:
  - 5 independent random warm starts
  - L-BFGS-B, maxiter=2000
  - Report final best Frobenius residual
"""
import numpy as np
from scipy.optimize import minimize

SQRT3 = np.sqrt(3)

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

DIM = 9

def make_obj(R):
    def f(x):
        A = x[:R*DIM].reshape(R, DIM)
        B = x[R*DIM:2*R*DIM].reshape(R, DIM)
        C = x[2*R*DIM:].reshape(R, DIM)
        E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
        loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
        dA = np.einsum('ijl,kj,kl->ki', E, B, C)
        dB = np.einsum('ijl,ki,kl->kj', E, A, C)
        dC = np.einsum('ijl,ki,kj->kl', E, A, B)
        return loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])
    return f

RANKS = [13, 19, 20, 21, 22, 23, 27]
N_STARTS = 5
rng = np.random.default_rng(42)

print(f"√3 = {SQRT3:.6f}")
print(f"{'R':>4}  {'best_frob':>12}  {'/ √3':>8}  {'verdict'}")
print("-" * 50)

results = {}

for R in RANKS:
    obj = make_obj(R)
    best = np.inf
    for trial in range(N_STARTS):
        scale = 1.0 / R**0.5
        x0 = rng.standard_normal(R * 3 * DIM) * scale
        res = minimize(obj, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-9})
        frob = np.sqrt(2 * res.fun)
        if frob < best:
            best = frob

    ratio = best / SQRT3
    if best < 1e-4:
        verdict = "→ 0  (exact)"
    elif abs(ratio - 1.0) < 0.02:
        verdict = f"→ √3  (ratio={ratio:.4f})"
    elif ratio < 1.0:
        verdict = f"BELOW √3  (!!)  ratio={ratio:.4f}"
    else:
        verdict = f"above √3  ratio={ratio:.4f}"

    results[R] = best
    print(f"{R:>4}  {best:>12.6f}  {ratio:>8.4f}  {verdict}")

print()
print("Summary:")
print(f"  √3 = {SQRT3:.6f}")
print(f"  Ranks < 23:  {[results[r] for r in RANKS if r < 23]}")
print(f"  Ranks ≥ 23:  {[results[r] for r in RANKS if r >= 23]}")
