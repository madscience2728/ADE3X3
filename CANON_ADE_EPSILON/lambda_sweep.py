import json, numpy as np

DEAD_PAIRS = [(s,t) for s in range(3) for t in range(3) if s != t]

def compute_Sigma_Delta(alpha_flat, beta_flat):
    R = alpha_flat.shape[0]
    A = alpha_flat.reshape(R,3,3); B = beta_flat.reshape(R,3,3)
    Sigma = np.zeros((R,9))
    Delta = np.zeros((R, 54))
    for k in range(R):
        for r in range(3):
            for u in range(3):
                Sigma[k, r*3+u] = sum(A[k,r,s]*B[k,s,u] for s in range(3))
        col = 0
        for (s,t) in DEAD_PAIRS:
            for r in range(3):
                for u in range(3):
                    Delta[k, col] = A[k,r,s]*B[k,t,u]
                    col += 1
    return Sigma, Delta

with open('outputs/ade3x3_attack/phase4_gradient_search/best_decompositions/10_border23_0000.json') as f:
    d = json.load(f)
terms = d['terms']
R = len(terms)
alpha = np.array([t['alpha'] for t in terms])
beta  = np.array([t['beta']  for t in terms])

Sigma, Delta = compute_Sigma_Delta(alpha, beta)

# SVD of Sigma: R×9 -> U(R×9), sv(9), Vt(9×9)
U, sv, Vt = np.linalg.svd(Sigma, full_matrices=False)
V = Vt.T  # 9×9

# Precompute UtDelta = U^T @ Delta  (9×54)
UtDelta = U.T @ Delta

# Budgets for R=22, bfloat16 (b=7)
b = 7
budget_frob = 3 * R * 2**(-b)
budget_fiber = budget_frob / 2   # allocate half to fiber, half to leakage

print(f"R = {R}")
print(f"bfloat16 budget_frob    = {budget_frob:.6f}")
print(f"budget_fiber (half)     = {budget_fiber:.6f}")
print(f"budget_leakage (half)   = {budget_fiber:.6f}")
print()

# Sweep lambda
lambdas = np.concatenate([
    np.linspace(0, 0.001, 20),
    np.logspace(-3, 2, 200)
])

print(f"{'lambda':>12}  {'leakage(λ)':>14}  {'fiber_err(λ)':>14}  {'leakage≤bgt':>12}  {'fiber≤bgt':>10}")
print("-"*72)

best_feasible = None
for lam in lambdas:
    # Tikhonov weights: d/(d^2 + lambda)
    weights = sv / (sv**2 + lam)   # shape (9,)

    # Leakage = 3 * ||diag(weights) @ UtDelta||_F
    leakage = 3.0 * np.linalg.norm(weights[:, None] * UtDelta)

    # Fiber error = 3*lambda * sqrt(sum 1/(d^2+lambda)^2)
    fiber_err = 3.0 * lam * np.sqrt(np.sum(1.0 / (sv**2 + lam)**2))

    ok_leak = leakage <= budget_frob
    ok_fiber = fiber_err <= budget_fiber

    if ok_leak and ok_fiber and best_feasible is None:
        best_feasible = (lam, leakage, fiber_err)
        marker = "  <<< FIRST FEASIBLE"
    else:
        marker = ""

    # Only print sparse rows to keep output readable
    if lam < 0.002 or lam > 0.5:
        continue
    print(f"{lam:12.6f}  {leakage:14.6f}  {fiber_err:14.6f}  {str(ok_leak):>12}  {str(ok_fiber):>10}{marker}")

print()
print("=== Full sweep summary (every 10th point) ===")
for i, lam in enumerate(lambdas):
    if i % 10 != 0:
        continue
    weights = sv / (sv**2 + lam)
    leakage = 3.0 * np.linalg.norm(weights[:, None] * UtDelta)
    fiber_err = 3.0 * lam * np.sqrt(np.sum(1.0 / (sv**2 + lam)**2))
    ok_leak = leakage <= budget_frob
    ok_fiber = fiber_err <= budget_fiber
    marker = " <-- FEASIBLE" if (ok_leak and ok_fiber) else ""
    print(f"  lambda={lam:10.5f}  leakage={leakage:.6f}  fiber_err={fiber_err:.6f}{marker}")

print()
if best_feasible is not None:
    lam, leakage, fiber_err = best_feasible
    print(f"FEASIBLE REGION EXISTS.")
    print(f"  First feasible lambda = {lam:.6f}")
    print(f"  Leakage at that lambda = {leakage:.6f}  (budget: {budget_frob:.6f})")
    print(f"  Fiber error at that lambda = {fiber_err:.6f}  (budget: {budget_fiber:.6f})")
else:
    # Find where leakage would cross budget and what fiber error is there
    weights0 = sv / (sv**2 + 0)
    leak0 = 3.0 * np.linalg.norm(weights0[:, None] * UtDelta)
    print(f"NO FEASIBLE LAMBDA FOUND in sweep.")
    print(f"  Leakage at lambda=0: {leak0:.6f}")
    # Find lambda where leakage = budget_frob
    from scipy.optimize import brentq
    def leak_fn(lam):
        w = sv / (sv**2 + lam)
        return 3.0 * np.linalg.norm(w[:, None] * UtDelta) - budget_frob
    # Find if leakage ever reaches budget
    leak_at_100 = leak_fn(100) + budget_frob
    print(f"  Leakage at lambda=100: {leak_at_100:.6f}")
    try:
        lam_cross = brentq(leak_fn, 0.001, 1000)
        weights_cross = sv / (sv**2 + lam_cross)
        fiber_at_cross = 3.0 * lam_cross * np.sqrt(np.sum(1.0 / (sv**2 + lam_cross)**2))
        print(f"  Lambda where leakage = budget_frob: {lam_cross:.6f}")
        print(f"  Fiber error at that lambda: {fiber_at_cross:.6f}  (budget: {budget_fiber:.6f})")
        print(f"  Fiber error / budget_frob: {fiber_at_cross / budget_frob:.4f}x")
        if fiber_at_cross <= budget_frob:
            print(f"  => FEASIBLE if we use full budget_frob for fiber too (not split)!")
    except Exception as e:
        print(f"  brentq failed: {e}")
