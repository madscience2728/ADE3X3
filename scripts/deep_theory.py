#!/usr/bin/env python
"""Deep theory analysis: null space structure, lower bounds, and optimality.

Questions answered:
  1. What are the 38 dead-null directions? (gauge symmetries vs real freedom)
  2. Tighter lower bound via semidefinite / dual arguments
  3. Is the equalized residual a KKT condition for minimax optimality?
  4. Lagrange multiplier structure at the current point
"""

import json, sys
from pathlib import Path
import numpy as np
from scipy import linalg as la
from scipy.optimize import linprog

np.set_printoptions(precision=8, linewidth=140, suppress=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T

with open(Path(__file__).resolve().parent.parent / "shotgun_best.json") as f:
    data = json.load(f)
alpha, beta, gamma = np.array(data["alpha"]), np.array(data["beta"]), np.array(data["gamma"])
fit = data["fitness"]

R, D = RANK, DIM
LIVE = (T != 0)
DEAD = (T == 0)
live_idx = np.where(LIVE.ravel())[0]
dead_idx = np.where(DEAD.ravel())[0]

T_hat = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
Res = T_hat - T
maxabs = np.max(np.abs(Res))

print("=" * 72)
print("  DEEP THEORY: NULL SPACE, KKT, AND LOWER BOUNDS")
print(f"  Current fitness: {maxabs:.10f}")
print("=" * 72)

# ══════════════════════════════════════════════════════════════
# 1. Build full Jacobian J (729 x 513)
# ══════════════════════════════════════════════════════════════
print("\n--- Building full Jacobian (729 x 513) ---")

n_entries = D**3  # 729
n_params = 3 * R * D  # 513

J = np.zeros((n_entries, n_params))
for k in range(R):
    for idx in range(n_entries):
        a, b, c = np.unravel_index(idx, (D, D, D))
        J[idx, k*D + a]       += beta[k, b] * gamma[k, c]   # d/d alpha[k,a]
        J[idx, R*D + k*D + b] += alpha[k, a] * gamma[k, c]  # d/d beta[k,b]
        J[idx, 2*R*D + k*D + c] += alpha[k, a] * beta[k, b] # d/d gamma[k,c]

J_dead = J[dead_idx, :]  # (702, 513)
J_live = J[live_idx, :]  # (27, 513)

U_d, s_d, Vh_d = la.svd(J_dead, full_matrices=False)
jdead_rank = np.sum(s_d > 1e-10)
null_dim = n_params - jdead_rank
print(f"J_dead rank: {jdead_rank},  null dim: {null_dim}")

# ══════════════════════════════════════════════════════════════
# 2. What are the 38 null-space directions?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 1: NULL SPACE DECOMPOSITION")
print(f"{'='*72}")

null_basis = Vh_d[jdead_rank:, :].T  # (513, 38)

# Do these null directions affect LIVE entries?
J_live_on_null = J_live @ null_basis  # (27, 38)
print(f"\nJ_live projected onto dead-null basis: shape {J_live_on_null.shape}")
s_live_null = la.svdvals(J_live_on_null)
live_null_rank = np.sum(s_live_null > 1e-10)
print(f"Rank of J_live restricted to dead-null space: {live_null_rank}")
print(f"Singular values: {s_live_null}")

# Interpretation: directions that change neither dead NOR live entries
pure_gauge_dim = null_dim - live_null_rank
print(f"\nDirections changing dead=0, live=0 (pure gauge): {pure_gauge_dim}")
print(f"Directions changing dead=0, live!=0 (useful freedom): {live_null_rank}")

# What does "useful freedom" look like?
if live_null_rank > 0:
    # Which live entries can these directions affect?
    # U_ln, s_ln, Vh_ln from SVD of J_live_on_null
    U_ln, s_ln, Vh_ln = la.svd(J_live_on_null, full_matrices=False)
    active_live_dirs = U_ln[:, :live_null_rank]  # (27, live_null_rank)

    print(f"\nLive entries most affected by dead-null directions:")
    for i in range(min(live_null_rank, 5)):
        top_entry = np.argmax(np.abs(active_live_dirs[:, i]))
        a, b, c = np.unravel_index(live_idx[top_entry], (D, D, D))
        r_a, s_a = divmod(a, 3)
        s_b, u_b = divmod(b, 3)
        print(f"  mode {i}: sv={s_ln[i]:.6f}, strongest at T[{a},{b},{c}] "
              f"(C[{r_a},{u_b}]), weight={active_live_dirs[top_entry, i]:.4f}")

# ══════════════════════════════════════════════════════════════
# 3. KKT analysis: is this a minimax optimum?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 2: KKT OPTIMALITY CHECK")
print(f"{'='*72}")

# At a minimax optimum, there exist non-negative multipliers mu_i >= 0
# for all entries i where |R_i| = maxabs, such that:
#   sum_i mu_i * sign(R_i) * J[i,:] = 0   (stationarity)
#   sum_i mu_i = 1                          (normalization)
#   mu_i = 0 if |R_i| < maxabs             (complementarity)

flat_res = Res.ravel()
tol_tight = 0.001  # entries within 0.1% of max-abs
tight_mask = np.abs(flat_res) > (1 - tol_tight) * maxabs
tight_idx = np.where(tight_mask)[0]
n_tight = len(tight_idx)
print(f"\nTight entries (within {tol_tight*100}% of max-abs): {n_tight}")

# Stationarity: J_tight.T @ (mu * sign) = 0, sum(mu) = 1, mu >= 0
# This is a linear feasibility problem
signs_tight = np.sign(flat_res[tight_idx])
J_tight = J[tight_idx, :]  # (n_tight, 513)

# Signed Jacobian rows
SJ = (signs_tight[:, None] * J_tight)  # (n_tight, 513)

# Check if 0 is in the convex hull of the rows of SJ
# i.e., does there exist mu >= 0, sum(mu) = 1, SJ.T @ mu = 0?
# LP: minimize 0 subject to SJ.T @ mu = 0, sum(mu) = 1, mu >= 0
# If feasible -> KKT satisfied

A_eq = np.vstack([SJ.T, np.ones((1, n_tight))])  # (514, n_tight)
b_eq = np.zeros(514)
b_eq[-1] = 1.0  # sum(mu) = 1

c_obj = np.zeros(n_tight)  # feasibility check

result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq,
                 bounds=[(0, None)] * n_tight,
                 method='highs', options={'presolve': True})

if result.success:
    mu = result.x
    n_active = np.sum(mu > 1e-8)
    print(f"\n  KKT FEASIBLE: minimax stationarity holds!")
    print(f"  Active multipliers (mu > 1e-8): {n_active} out of {n_tight}")
    print(f"  Residual of stationarity: {la.norm(SJ.T @ mu):.2e}")

    # Which entries have the largest multipliers?
    top_mu = np.argsort(mu)[::-1][:10]
    print(f"\n  Top 10 active entries:")
    for rank_i, idx_in_tight in enumerate(top_mu):
        entry_idx = tight_idx[idx_in_tight]
        a, b, c = np.unravel_index(entry_idx, (D, D, D))
        r_a, s_a = divmod(a, 3)
        s_b, u_b = divmod(b, 3)
        is_live = T[a, b, c] != 0
        print(f"    mu={mu[idx_in_tight]:.6f}  R[{a},{b},{c}]={flat_res[entry_idx]:+.8f}  "
              f"{'LIVE' if is_live else 'DEAD'}  C[{r_a},{u_b}]")
else:
    print(f"\n  KKT NOT feasible with {tol_tight*100}% tolerance")
    print(f"  Status: {result.message}")
    # Try with looser tolerance
    for loose_tol in [0.005, 0.01, 0.02, 0.05]:
        tight2 = np.abs(flat_res) > (1 - loose_tol) * maxabs
        tight2_idx = np.where(tight2)[0]
        signs2 = np.sign(flat_res[tight2_idx])
        SJ2 = signs2[:, None] * J[tight2_idx, :]
        A2 = np.vstack([SJ2.T, np.ones((1, len(tight2_idx)))])
        b2 = np.zeros(514); b2[-1] = 1.0
        res2 = linprog(np.zeros(len(tight2_idx)), A_eq=A2, b_eq=b2,
                       bounds=[(0, None)] * len(tight2_idx), method='highs')
        if res2.success:
            print(f"  KKT feasible at {loose_tol*100}% tolerance ({len(tight2_idx)} entries)")
            mu2 = res2.x
            print(f"  Stationarity residual: {la.norm(SJ2.T @ mu2):.2e}")
            break

# ══════════════════════════════════════════════════════════════
# 4. Tighter lower bound: dual of the fitting problem
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 3: LOWER BOUND ANALYSIS")
print(f"{'='*72}")

# Frobenius-based bounds
R_fro = np.sqrt(np.sum(Res**2))
print(f"\n||R||_F = {R_fro:.8f}")
print(f"||R||_F / sqrt(729) = {R_fro / np.sqrt(729):.8f}  (trivial lower bound)")
print(f"Actual max|R| = {maxabs:.8f}")

# Better: use the distribution of residuals
sorted_abs = np.sort(np.abs(flat_res))[::-1]
# If top-k entries all have value v, then ||R||^2 >= k*v^2
# So v <= ||R||/sqrt(k). But we know ||R||^2 = sum of all entries
# More useful: the top k entries contribute at least k*sorted[k-1]^2

print(f"\nResidual distribution (how equalized?):")
print(f"  Top 1:   {sorted_abs[0]:.8f}")
print(f"  Top 10:  {sorted_abs[9]:.8f}  (ratio: {sorted_abs[9]/sorted_abs[0]:.6f})")
print(f"  Top 50:  {sorted_abs[49]:.8f}  (ratio: {sorted_abs[49]/sorted_abs[0]:.6f})")
print(f"  Top 100: {sorted_abs[99]:.8f}  (ratio: {sorted_abs[99]/sorted_abs[0]:.6f})")
print(f"  Top 200: {sorted_abs[199]:.8f}  (ratio: {sorted_abs[199]/sorted_abs[0]:.6f})")
print(f"  Top 400: {sorted_abs[399]:.8f}  (ratio: {sorted_abs[399]/sorted_abs[0]:.6f})")
print(f"  Top 600: {sorted_abs[599]:.8f}  (ratio: {sorted_abs[599]/sorted_abs[0]:.6f})")
print(f"  Bottom:  {sorted_abs[-1]:.8f}")

# Equalization entropy: how close to uniform is |R|?
p = np.abs(flat_res) / np.sum(np.abs(flat_res))
entropy = -np.sum(p * np.log(p + 1e-30))
max_entropy = np.log(729)
print(f"\nResidual entropy: {entropy:.4f} / {max_entropy:.4f} = {entropy/max_entropy:.4f}")
print(f"  (1.0 = perfectly uniform, <0.9 = concentrated)")

# ══════════════════════════════════════════════════════════════
# 5. Jacobian rank at ALL entries (not just dead)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 4: FULL JACOBIAN RANK ANALYSIS")
print(f"{'='*72}")

U_full, s_full, Vh_full = la.svd(J, full_matrices=False)
jfull_rank = np.sum(s_full > 1e-10)
print(f"Full Jacobian J (729 x 513) rank: {jfull_rank}")
print(f"Singular values (top 20): {s_full[:20]}")
print(f"Singular values (bottom 20): {s_full[-20:]}")
full_null_dim = n_params - jfull_rank
print(f"Full null space dim: {full_null_dim}")
print(f"  (pure gauge symmetries of the entire tensor)")

# These gauge symmetries correspond to transformations that
# don't change T_hat at all. For CP decomposition, these are:
# - Simultaneous scaling: alpha[k] *= c, gamma[k] /= c (R dims)
# - Permutations of terms (discrete, not in Jacobian null space)
# So we expect ~R or fewer continuous gauge dims

# ══════════════════════════════════════════════════════════════
# 6. Sensitivity: how much can max-abs decrease per unit parameter change?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 5: SENSITIVITY ANALYSIS")
print(f"{'='*72}")

# The max-abs is at entry idx_max. Gradient of |R[idx_max]| w.r.t. params:
idx_max = np.argmax(np.abs(flat_res))
a_m, b_m, c_m = np.unravel_index(idx_max, (D, D, D))
sign_max = np.sign(flat_res[idx_max])
grad_max = sign_max * J[idx_max, :]  # gradient of R[idx_max] w.r.t. params

# Gradient of the smooth-max approximation (all near-max entries contribute)
beta_sm = 10000  # smooth-max parameter
absR = np.abs(flat_res)
mx_val = np.max(absR)
shifted = beta_sm * (absR - mx_val)
exp_s = np.exp(shifted)
Z = np.sum(exp_s)
weights_sm = exp_s / Z  # (729,) weights for smooth-max gradient
grad_smooth = J.T @ (weights_sm * np.sign(flat_res))  # (513,)

print(f"Max entry: R[{a_m},{b_m},{c_m}] = {flat_res[idx_max]:+.10f}")
print(f"||grad(max entry)||  = {la.norm(grad_max):.6f}")
print(f"||grad(smooth-max)|| = {la.norm(grad_smooth):.6f}")

# Project smooth-max gradient onto dead-null space
proj_null = null_basis @ (null_basis.T @ grad_smooth)
proj_range = grad_smooth - proj_null
print(f"\nSmooth-max gradient decomposition:")
print(f"  Component in dead-null space:  {la.norm(proj_null):.8f}")
print(f"  Component in dead-range space: {la.norm(proj_range):.8f}")
print(f"  Ratio (null/total):            {la.norm(proj_null)/la.norm(grad_smooth):.8f}")

# How about the full-null space?
if full_null_dim > 0:
    full_null_basis = Vh_full[jfull_rank:, :].T  # (513, full_null_dim)
    proj_full_null = full_null_basis @ (full_null_basis.T @ grad_smooth)
    print(f"  Component in FULL null space:  {la.norm(proj_full_null):.8f}")

# ══════════════════════════════════════════════════════════════
# 7. Can we move along dead-null to improve live residual?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 6: LIVE IMPROVEMENT VIA DEAD-NULL")
print(f"{'='*72}")

live_res = Res[LIVE]  # (27,)
print(f"Live residual: max|R_live| = {np.max(np.abs(live_res)):.8f}")
print(f"  All close to {np.mean(live_res):.8f} +/- {np.std(live_res):.8f}")

# Gradient of live max-abs w.r.t. params
live_flat_res = flat_res[live_idx]
idx_live_max = np.argmax(np.abs(live_flat_res))
sign_live_max = np.sign(live_flat_res[idx_live_max])
grad_live_max = sign_live_max * J[live_idx[idx_live_max], :]

# Project onto dead-null
proj_live_null = null_basis @ (null_basis.T @ grad_live_max)
print(f"\nGradient of live max-abs:")
print(f"  ||grad|| = {la.norm(grad_live_max):.6f}")
print(f"  Component in dead-null: {la.norm(proj_live_null):.8f}")
print(f"  Ratio: {la.norm(proj_live_null)/la.norm(grad_live_max):.8f}")

if la.norm(proj_live_null) > 1e-10:
    # We can improve live without changing dead!
    step = -proj_live_null / la.norm(proj_live_null)
    predicted = J[live_idx[idx_live_max], :] @ step
    print(f"  -> CAN improve live max without changing dead!")
    print(f"     Predicted improvement per unit step: {predicted:.8f}")
else:
    print(f"  -> NO live improvement possible without changing dead")

# ══════════════════════════════════════════════════════════════
# 8. Optimal equalization: LP for best possible max-abs
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  SECTION 7: EQUALIZATION OPTIMALITY")
print(f"{'='*72}")

# The question: given the Jacobian at this point, what's the best
# achievable max-abs via infinitesimal parameter change?
# 
# minimize t
# subject to: |R_i + J[i,:] @ dx| <= t  for all i
# i.e.: linear program in (t, dx)
#   -t <= R_i + J[i,:] @ dx <= t
#   R_i + J[i,:] @ dx <= t   =>  J[i,:] @ dx - t <= -R_i
#   -(R_i + J[i,:] @ dx) <= t  =>  -J[i,:] @ dx - t <= R_i

print(f"Solving LP for locally optimal max-abs (linearized)...")
# Variables: [dx (513), t (1)]
n_vars = n_params + 1  # 514

# Constraints: for each entry i (729 total):
#   J[i,:] @ dx - t <= -R_i    (upper bound)
#   -J[i,:] @ dx - t <= R_i    (lower bound)
A_ub = np.zeros((2 * n_entries, n_vars))
b_ub = np.zeros(2 * n_entries)

for i in range(n_entries):
    # J[i,:] @ dx - t <= -R_i
    A_ub[2*i, :n_params] = J[i, :]
    A_ub[2*i, n_params] = -1.0
    b_ub[2*i] = -flat_res[i]

    # -J[i,:] @ dx - t <= R_i
    A_ub[2*i+1, :n_params] = -J[i, :]
    A_ub[2*i+1, n_params] = -1.0
    b_ub[2*i+1] = flat_res[i]

# Objective: minimize t (last variable)
c_lp = np.zeros(n_vars)
c_lp[n_params] = 1.0

# Bound dx to be small (trust region)
dx_bound = 0.01
bounds_lp = [(-dx_bound, dx_bound)] * n_params + [(0, None)]

lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp,
                 method='highs', options={'presolve': True, 'time_limit': 30})

if lp_res.success:
    t_opt = lp_res.x[n_params]
    dx_opt = lp_res.x[:n_params]
    dx_norm = la.norm(dx_opt)
    print(f"  LP optimal t = {t_opt:.10f}  (current = {maxabs:.10f})")
    print(f"  Improvement: {maxabs - t_opt:.2e}")
    print(f"  ||dx|| = {dx_norm:.8f}")
    print(f"  dx bound used: {dx_bound}")

    # Verify: what actually happens at this dx?
    new_res = flat_res + J @ dx_opt
    new_maxabs = np.max(np.abs(new_res))
    print(f"  Predicted new max-abs: {new_maxabs:.10f}")

    # Component of dx in dead-null space
    dx_null = null_basis @ (null_basis.T @ dx_opt)
    dx_range = dx_opt - dx_null
    print(f"  dx dead-null component: {la.norm(dx_null):.8f} ({la.norm(dx_null)/dx_norm*100:.1f}%)")
    print(f"  dx dead-range component: {la.norm(dx_range):.8f} ({la.norm(dx_range)/dx_norm*100:.1f}%)")

    # Try larger trust regions
    for tr in [0.05, 0.1, 0.5, 1.0]:
        bounds_tr = [(-tr, tr)] * n_params + [(0, None)]
        res_tr = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_tr,
                         method='highs', options={'presolve': True, 'time_limit': 10})
        if res_tr.success:
            print(f"  Trust region {tr}: t_opt = {res_tr.x[n_params]:.10f}")
else:
    print(f"  LP failed: {lp_res.message}")

# ══════════════════════════════════════════════════════════════
# 9. Summary
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  DEEP THEORY SUMMARY")
print(f"{'='*72}")
print(f"Current max-abs: {maxabs:.10f}")
print(f"Full Jacobian rank: {jfull_rank} / 513  (gauge dim: {full_null_dim})")
print(f"Dead Jacobian rank: {jdead_rank} / 513  (dead-null dim: {null_dim})")
print(f"Dead-null directions that affect live: {live_null_rank}")
print(f"Pure gauge (affect nothing): {pure_gauge_dim}")
print(f"Residual entropy: {entropy/max_entropy:.4f}")
print(f"Near-max entries (0.1% tol): {n_tight}")
