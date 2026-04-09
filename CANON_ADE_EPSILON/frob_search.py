"""
Direct Frobenius minimization for R=19 CP decomposition of T_matmul.
Objective: ||T - sum_k a_k ⊗ b_k ⊗ c_k||_F
Targets:   ε ≤ 1.056  (int4 closure)
           ε ≤ 0.134  (bfloat16 closure)
"""
import numpy as np
import json
from scipy.optimize import minimize

# ── T_matmul ────────────────────────────────────────────────────────────────
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_vec = T.ravel()   # 729
T_norm = float(np.linalg.norm(T))

R = 19
DIM = 9
N_PARAMS = R * 3 * DIM   # 19 * 27 = 513

INT4_TARGET   = 1.056
BF16_TARGET   = 0.134

# ── Objective and gradient ───────────────────────────────────────────────────
def unpack(x):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    return A, B, C

def residual_and_grad(x):
    A, B, C = unpack(x)
    # T_approx[i,j,l] = sum_k A[k,i] B[k,j] C[k,l]
    T_approx = np.einsum('ki,kj,kl->ijl', A, B, C)
    E = T_approx - T     # 9x9x9
    loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    frob = float(np.linalg.norm(E))

    # Gradients via chain rule:
    # dL/dA[k,i] = sum_{j,l} E[i,j,l] * B[k,j] * C[k,l]
    dA = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC = np.einsum('ijl,ki,kj->kl', E, A, B)

    grad = np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])
    return loss, frob, grad

def obj_and_grad(x):
    loss, frob, grad = residual_and_grad(x)
    return loss, grad

# ── Load best known candidate as warm start ──────────────────────────────────
print("Loading warm-start from slp_best_at_0.074.json ...")
with open('tests/slp_best_at_0.074.json') as f:
    d = json.load(f)
A0 = np.array(d['alpha'])   # 19x9
B0 = np.array(d['beta'])
C0 = np.array(d['gamma'])
x0 = np.concatenate([A0.ravel(), B0.ravel(), C0.ravel()])
_, frob0, _ = residual_and_grad(x0)
print(f"  Warm-start Frobenius residual: {frob0:.6f}  (target int4: {INT4_TARGET}, bf16: {BF16_TARGET})")
print()

best_x = x0.copy()
best_frob = frob0

def callback(x, state=None):
    global best_x, best_frob
    _, frob, _ = residual_and_grad(x)
    if frob < best_frob:
        best_frob = frob
        best_x = x.copy()
    if frob <= BF16_TARGET:
        tag = "*** BF16 CLOSED ***"
    elif frob <= INT4_TARGET:
        tag = "*** INT4 CLOSED ***"
    else:
        tag = ""
    print(f"  frob={frob:.6f}  best={best_frob:.6f}  {tag}")

# ── Run L-BFGS-B ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 1: L-BFGS-B from warm start (maxiter=2000)")
print("=" * 60)
callback(x0)

result = minimize(
    obj_and_grad,
    x0,
    method='L-BFGS-B',
    jac=True,
    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-8},
    callback=callback
)
_, frob1, _ = residual_and_grad(result.x)
print(f"\nPhase 1 done. Final frob: {frob1:.6f}")
if frob1 < best_frob:
    best_frob = frob1
    best_x = result.x.copy()

# ── Multi-restart with random perturbations ──────────────────────────────────
print()
print("=" * 60)
print("Phase 2: Multi-restart around best found (50 restarts)")
print("=" * 60)

rng = np.random.default_rng(42)
for trial in range(50):
    # Perturb best solution
    scale = rng.choice([0.01, 0.05, 0.1, 0.3])
    x_init = best_x + rng.standard_normal(N_PARAMS) * scale

    res = minimize(
        obj_and_grad,
        x_init,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-7},
    )
    _, frob_t, _ = residual_and_grad(res.x)

    if frob_t < best_frob:
        best_frob = frob_t
        best_x = res.x.copy()
        tag = ""
        if best_frob <= BF16_TARGET: tag = " *** BF16 CLOSED ***"
        elif best_frob <= INT4_TARGET: tag = " *** INT4 CLOSED ***"
        print(f"  trial {trial:3d}: NEW BEST frob={best_frob:.6f}{tag}")
    elif trial % 10 == 0:
        print(f"  trial {trial:3d}: frob={frob_t:.6f}  best={best_frob:.6f}")

# ── Phase 3: Cold random restarts ────────────────────────────────────────────
print()
print("=" * 60)
print("Phase 3: Cold random restarts (200 restarts, short runs)")
print("=" * 60)

for trial in range(200):
    scale = rng.choice([0.3, 0.5, 1.0])
    x_init = rng.standard_normal(N_PARAMS) * scale

    # Quick short run
    res = minimize(
        obj_and_grad,
        x_init,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6},
    )
    _, frob_t, _ = residual_and_grad(res.x)

    # Refine if promising
    if frob_t < best_frob * 1.5:
        res2 = minimize(
            obj_and_grad,
            res.x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8},
        )
        _, frob_t, _ = residual_and_grad(res2.x)
        if frob_t < best_frob:
            best_frob = frob_t
            best_x = res2.x.copy()
            tag = ""
            if best_frob <= BF16_TARGET: tag = " *** BF16 CLOSED ***"
            elif best_frob <= INT4_TARGET: tag = " *** INT4 CLOSED ***"
            print(f"  cold trial {trial:3d}: NEW BEST frob={best_frob:.6f}{tag}")

    if trial % 50 == 0:
        print(f"  cold trial {trial:3d}: best so far = {best_frob:.6f}")

# ── Final report ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
A_best, B_best, C_best = unpack(best_x)
T_best = np.einsum('ki,kj,kl->ijl', A_best, B_best, C_best)
final_frob = float(np.linalg.norm(T - T_best))
final_max  = float(np.max(np.abs(T - T_best)))

print(f"  R = {R}")
print(f"  Frobenius residual: {final_frob:.8f}")
print(f"  Max-entry residual: {final_max:.8f}")
print(f"  ||T||_F = {T_norm:.4f}")
print(f"  Relative error:     {final_frob/T_norm:.6f} ({100*final_frob/T_norm:.2f}%)")
print()
print(f"  INT4    target ε ≤ {INT4_TARGET:.4f}: {'CLOSED ✓' if final_frob <= INT4_TARGET else f'OPEN  (gap {final_frob/INT4_TARGET:.2f}×)'}")
print(f"  BF16    target ε ≤ {BF16_TARGET:.4f}: {'CLOSED ✓' if final_frob <= BF16_TARGET else f'OPEN  (gap {final_frob/BF16_TARGET:.2f}×)'}")
print()
print(f"  total_error (int4,  b=4):  {final_frob + 3*R*2**-4:.6f}")
print(f"  total_error (bfloat16,b=7): {final_frob + 3*R*2**-7:.6f}")
print(f"  total_error (float16, b=10): {final_frob + 3*R*2**-10:.6f}")
print()

# Save best result
out = {
    'R': R,
    'frobenius_residual': final_frob,
    'max_entry_residual': final_max,
    'int4_closed': bool(final_frob <= INT4_TARGET),
    'bf16_closed': bool(final_frob <= BF16_TARGET),
    'alpha': A_best.tolist(),
    'beta':  B_best.tolist(),
    'gamma': C_best.tolist(),
}
with open('CANON_ADE_EPSILON/results/frob_search_r19.json', 'w') as f:
    import os; os.makedirs('CANON_ADE_EPSILON/results', exist_ok=True)
    json.dump(out, f, indent=2)
print("  Saved to CANON_ADE_EPSILON/results/frob_search_r19.json")
