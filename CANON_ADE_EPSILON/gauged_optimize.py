"""
Gauge-fixed optimization: pin ||alpha_k||=1 for all k.
This eliminates 22 scaling freedoms, normalizes factor magnitudes,
and is required for the bfloat16 budget formula to apply in practice.

Parametrize: alpha_k = a_k / ||a_k||, absorb scale into beta_k.
So: T ≈ sum_k (a_k/||a_k||) ⊗ (s_k * b_k/||b_k||) ⊗ (t_k * c_k/||c_k||)
  = sum_k s_k*t_k * hat_a_k ⊗ hat_b_k ⊗ hat_c_k

Free params: a_k in R^9 (direction only, but optimize freely and normalize),
             b_k in R^9, c_k in R^9, scalar scales absorbed into b.
Simpler: just keep A,B,C but add ||A[k]||=1 penalty or use unit-sphere param.

Simplest: optimize A,B,C freely, but after each step normalize rows of A.
Use projected gradient: optimize, then project A rows to unit sphere.

Even simpler for L-BFGS-B: fix gauge by requiring A[k,0] >= 0 and ||A[k]||=1
via spherical coords, or just add L2 regularization to keep entries small.

We'll use the simplest approach: add a soft regularization that penalizes
large factor entries, driving them toward unit scale.
"""
import numpy as np, json, os
from scipy.optimize import minimize

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

R, DIM = 22, 9

def obj_gauged(x, lam=1e-3):
    """Frobenius loss + soft penalty on row norms of A away from 1."""
    A=x[:R*DIM].reshape(R,DIM); B=x[R*DIM:2*R*DIM].reshape(R,DIM); C=x[2*R*DIM:].reshape(R,DIM)
    E=np.einsum('ki,kj,kl->ijl',A,B,C)-T
    loss=0.5*float(np.dot(E.ravel(),E.ravel()))

    # Gauge penalty: sum_k (||A[k]||^2 - 1)^2
    norms_sq = np.sum(A**2, axis=1)  # (R,)
    gauge_pen = lam * float(np.sum((norms_sq - 1.0)**2))
    total = loss + gauge_pen

    dA=np.einsum('ijl,kj,kl->ki',E,B,C)
    dA += lam * 4 * ((norms_sq - 1.0)[:,None] * A)
    dB=np.einsum('ijl,ki,kl->kj',E,A,C)
    dC=np.einsum('ijl,ki,kj->kl',E,A,B)
    return total, np.concatenate([dA.ravel(),dB.ravel(),dC.ravel()])

def frob_only(x):
    A=x[:R*DIM].reshape(R,DIM); B=x[R*DIM:2*R*DIM].reshape(R,DIM); C=x[2*R*DIM:].reshape(R,DIM)
    E=np.einsum('ki,kj,kl->ijl',A,B,C)-T
    return float(np.linalg.norm(E))

# Load current best, normalize A rows as warm start
with open('CANON_ADE_EPSILON/results/rank22_best.json') as f:
    d = json.load(f)
A0=np.array(d['alpha']); B0=np.array(d['beta']); C0=np.array(d['gamma'])

# Normalize: A[k] -> A[k]/||A[k]||, B[k] -> B[k]*||A[k]||
norms = np.linalg.norm(A0, axis=1, keepdims=True)
A0n = A0 / norms
B0n = B0 * norms  # absorb scale into B
x0 = np.concatenate([A0n.ravel(), B0n.ravel(), C0.ravel()])

eps0 = frob_only(x0)
norms_check = np.linalg.norm(A0n, axis=1)
print(f'Starting ε={eps0:.6f}  A_row_norms: min={norms_check.min():.4f} max={norms_check.max():.4f}')
print(f'Factor entry magnitudes: A_max={np.max(np.abs(A0n)):.3f} B_max={np.max(np.abs(B0n)):.3f}')

best_x = x0.copy()
best_frob = eps0

rng = np.random.default_rng(42)

for phase, lam in [(50, 1e-2), (50, 1e-3), (50, 1e-4), (100, 1e-5)]:
    print(f'\nPhase λ={lam}: {phase} restarts...')
    phase_best = np.inf
    for trial in range(phase):
        if trial == 0:
            x_init = best_x.copy()
        elif trial % 5 == 0:
            # perturb best
            x_init = best_x + rng.standard_normal(len(best_x)) * 0.01
        else:
            scale = rng.uniform(0.3, 1.0) / R**0.5
            x_init = rng.standard_normal(len(best_x)) * scale
            # normalize A rows in init
            Ai = x_init[:R*DIM].reshape(R,DIM)
            ni = np.linalg.norm(Ai, axis=1, keepdims=True)
            ni = np.maximum(ni, 1e-8)
            Ai = Ai / ni
            x_init[:R*DIM] = Ai.ravel()

        res = minimize(lambda x: obj_gauged(x, lam), x_init, jac=True, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10})
        # Re-normalize A rows after optimization
        xr = res.x.copy()
        Ar = xr[:R*DIM].reshape(R,DIM)
        nr = np.linalg.norm(Ar, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-10)
        Ar = Ar / nr
        Br = xr[R*DIM:2*R*DIM].reshape(R,DIM) * nr
        xr[:R*DIM] = Ar.ravel(); xr[R*DIM:2*R*DIM] = Br.ravel()

        f = frob_only(xr)
        if f < best_frob:
            best_frob = f
            best_x = xr.copy()
        if f < phase_best:
            phase_best = f

    Af=best_x[:R*DIM].reshape(R,DIM); Bf=best_x[R*DIM:2*R*DIM].reshape(R,DIM); Cf=best_x[2*R*DIM:].reshape(R,DIM)
    nf = np.linalg.norm(Af, axis=1)
    print(f'  best_frob={best_frob:.8f}  A_norms: [{nf.min():.4f},{nf.max():.4f}]  '
          f'A_max={np.max(np.abs(Af)):.3f} B_max={np.max(np.abs(Bf)):.3f} C_max={np.max(np.abs(Cf)):.3f}')

# Final result
Af=best_x[:R*DIM].reshape(R,DIM); Bf=best_x[R*DIM:2*R*DIM].reshape(R,DIM); Cf=best_x[2*R*DIM:].reshape(R,DIM)
print(f'\nFinal ε = {best_frob:.8f}')
print(f'Factor max entries: A={np.max(np.abs(Af)):.4f} B={np.max(np.abs(Bf)):.4f} C={np.max(np.abs(Cf)):.4f}')
BF16_EPS = 2**-7
budget = 3 * R * BF16_EPS
print(f'Theoretical budget: 3×{R}×2^-7 = {budget:.4f}')
print(f'ε vs budget: {best_frob:.6f} vs {budget:.4f}  ({"WITHIN" if best_frob < budget else "EXCEEDS"})')

os.makedirs('CANON_ADE_EPSILON/results', exist_ok=True)
with open('CANON_ADE_EPSILON/results/rank22_gauged.json','w') as f:
    json.dump({'rank':R,'frobenius':best_frob,'alpha':Af.tolist(),'beta':Bf.tolist(),'gamma':Cf.tolist()},f)
print('Saved to rank22_gauged.json')
