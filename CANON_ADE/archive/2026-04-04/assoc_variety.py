# coding: utf-8
"""
assoc_variety.py  --  Phase 2: Quadratic BFS on the 6-dim HULL_ZD family.

Find all alpha in R^6 (up to scale, on RP^5) such that the algebra
  f = sum_i alpha_i * v_i
is associative:
  sum_m f[a,b,m]*f[m,c,d] = sum_m f[b,c,m]*f[a,m,d]  for all a,b,c,d

This gives 34 distinct symmetric 6x6 quadratic forms M_t.
We find the real variety  {alpha in RP^5 : alpha^T M_t alpha = 0 for all t}.

Strategy:
  1. Build all 34 quadratic forms.
  2. Check if their common null space on S^5 is empty (no solution) or has points.
  3. If nonempty: extract the solution(s) and expand to full f tensor.
  4. For each solution: check C3 fiber normalization and C5 ZD pairs.

Methods:
  - Random sampling on S^5 to find approximate solutions.
  - Gradient descent on sum_t (alpha^T M_t alpha)^2 to refine.
  - Eigenvalue analysis: find alpha that is simultaneously in null(M_t) for all t.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from axiom_bfs import (build_canon, enc_hullzd, N, N_ORBITS, orbit_of)
from canon_constraints import C3_FiberConstraint, IDX

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ------------------------------------------------------------------
# 1. Build the 6-dim HULL_ZD null basis
# ------------------------------------------------------------------
print("Building HULL_ZD null basis...", flush=True)
cs = build_canon()
enc_hullzd(cs)
A, b = cs.build()
U, sv, Vt = np.linalg.svd(A, full_matrices=True)
tol = max(A.shape) * sv[0] * 1e-10
rank = int(np.sum(sv > tol))
null_basis = Vt[rank:].T  # (439, 6)
DIM = null_basis.shape[1]
print(f"  Null basis: {DIM} vectors x {N_ORBITS} orbit coords")

def expand(v):
    f = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f[i, j, k] = v[orbit_of(i, j, k)]
    return f

Fv = [expand(null_basis[:, vi]) for vi in range(DIM)]
print(f"  Expanded {DIM} basis tensors (shape {Fv[0].shape})")

# ------------------------------------------------------------------
# 2. Build the quadratic constraint matrices
# ------------------------------------------------------------------
print("\nBuilding quadratic constraint matrices...", flush=True)

constraint_matrices = []
seen_keys = {}

for a in range(N):
    for b in range(N):
        for c in range(N):
            for d in range(N):
                M = np.zeros((DIM, DIM))
                for i in range(DIM):
                    for j in range(DIM):
                        M[i, j] = (np.dot(Fv[i][a, b, :], Fv[j][:, c, d])
                                 - np.dot(Fv[i][b, c, :], Fv[j][a, :, d]))
                if np.max(np.abs(M)) < 1e-12:
                    continue
                Ms = (M + M.T) / 2.0  # symmetrize: alpha^T M alpha = alpha^T Ms alpha
                scale = np.max(np.abs(Ms))
                key = tuple(np.round(Ms.flatten() / scale, 4))
                if key not in seen_keys:
                    seen_keys[key] = len(constraint_matrices)
                    constraint_matrices.append(Ms / scale)

print(f"  Distinct quadratic forms: {len(constraint_matrices)}")

# Stack into tensor for fast evaluation
Mc = np.array(constraint_matrices)  # shape (n_constraints, DIM, DIM)

def eval_constraints(alpha):
    """Evaluate all quadratic forms at alpha. Returns array of shape (n_constraints,)."""
    return np.einsum('i,tij,j->t', alpha, Mc, alpha)

def total_loss(alpha):
    """Sum of squared constraint values."""
    vals = eval_constraints(alpha)
    return np.sum(vals**2)

def total_loss_grad(alpha):
    """Gradient of total_loss w.r.t. alpha."""
    vals = eval_constraints(alpha)  # (T,)
    # d/d_alpha_k sum_t v_t^2 = sum_t 2*v_t * d(v_t)/d_alpha_k
    # d(v_t)/d_alpha_k = sum_j (Mc[t,k,j] + Mc[t,j,k]) * alpha_j = 2 * (Mc[t] @ alpha)[k]
    grad = 2.0 * np.einsum('t,tij,j->i', vals, Mc + Mc.transpose(0,2,1), alpha)
    return grad

# ------------------------------------------------------------------
# 3. Sample S^5 to find approximate solutions
# ------------------------------------------------------------------
print("\nSampling S^5 for approximate solutions...", flush=True)

N_SAMPLES = 100000
rng = np.random.default_rng(42)
alphas = rng.standard_normal((N_SAMPLES, DIM))
alphas /= np.linalg.norm(alphas, axis=1, keepdims=True)

losses = np.array([total_loss(a) for a in alphas])
top_k = np.argsort(losses)[:20]
print(f"  Min loss in {N_SAMPLES} samples: {losses[top_k[0]]:.6e}")
print(f"  Top 5 losses: {losses[top_k[:5]]}")

# ------------------------------------------------------------------
# 4. Gradient descent from best samples
# ------------------------------------------------------------------
print("\nRefining top candidates via gradient descent...", flush=True)

def gradient_descent(alpha0, lr=1e-3, steps=5000, tol=1e-14):
    alpha = alpha0.copy()
    for step in range(steps):
        loss = total_loss(alpha)
        if loss < tol:
            return alpha, loss, step
        grad = total_loss_grad(alpha)
        # Project gradient onto tangent space of S^5
        grad_tan = grad - np.dot(grad, alpha) * alpha
        alpha = alpha - lr * grad_tan
        alpha /= np.linalg.norm(alpha)
        # Adaptive lr
        if step % 500 == 499:
            lr *= 0.5
    return alpha, total_loss(alpha), steps

solutions = []
for k in top_k[:10]:
    sol, loss, steps = gradient_descent(alphas[k], lr=1e-2)
    if loss < 1e-10:
        solutions.append((sol, loss))
        print(f"  SOLUTION FOUND: loss={loss:.4e} in {steps} steps  alpha={sol}")
    else:
        print(f"  No convergence: final loss={loss:.4e}")

# ------------------------------------------------------------------
# 5. Also check: simultaneous null space of all Mc
# ------------------------------------------------------------------
print("\nChecking simultaneous null space of all constraint matrices...", flush=True)
# Sum all Mc (as PSD: Mc^T Mc) and find eigenvectors with eigenvalue 0
PSD_sum = np.sum(Mc @ Mc.transpose(0, 2, 1), axis=0) + np.sum(Mc.transpose(0,2,1) @ Mc, axis=0)
evals, evecs = np.linalg.eigh(PSD_sum)
print(f"  Eigenvalues of sum(Mc^T Mc + Mc Mc^T): {evals}")
for i, (ev, vec) in enumerate(zip(evals, evecs.T)):
    loss = total_loss(vec)
    print(f"    eigvec {i}: eigenvalue={ev:.4e}  assoc_loss={loss:.4e}")
    if loss < 1e-10:
        solutions.append((vec, loss))
        print(f"      -> SOLUTION: alpha={vec}")

# ------------------------------------------------------------------
# 6. Report
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Phase 2 result: {len(solutions)} associative point(s) found in HULL_ZD family")

if not solutions:
    print("\nNO associative algebra in the HULL_ZD family.")
    print("Implication: rank-19 requires relaxing HULL_ZD OR associativity.")
    print("Next: check COMM family for near-associativity, or probe partial associativity.")
else:
    for idx, (sol, loss) in enumerate(solutions):
        print(f"\nSolution {idx+1}: loss={loss:.4e}")
        f_sol = sum(sol[i] * Fv[i] for i in range(DIM))
        # Normalize
        fmax = np.max(np.abs(f_sol))
        if fmax > 1e-12:
            f_sol /= fmax

        comm_err = np.max(np.abs(f_sol - f_sol.transpose(1,0,2)))
        print(f"  Commutativity: {'YES' if comm_err < 1e-8 else 'NO'} (err={comm_err:.2e})")

        # C3 fiber sums
        print("  Fiber sums:")
        for (s, u), indices in C3_FiberConstraint.FIBERS.items():
            fsum = sum(f_sol[i, :, :] for i in indices).mean()
            print(f"    ({s},{u}): mean={fsum:.4f}")

        # Save
        out_path = os.path.join(os.path.dirname(__file__), f"assoc_solution_{idx+1}.npy")
        np.save(out_path, f_sol)
        print(f"  Saved to {out_path}")

print("\nDone.")
