#!/usr/bin/env python
"""Pure-theory combinatorial analysis of the rank-19 CP decomposition.

No brute-force search. Uses linear algebra, eigendecomposition, null spaces,
and combinatorial structure to derive theoretical bounds and identify
the structural bottleneck preventing improvement below 0.0775.

Analyses:
  §1  Per-term outer products: Gram matrices on live/dead subspaces
  §2  Dead null space: directions that preserve cancellation
  §3  Signal matrix Σ and its rank structure
  §4  Per-fiber residual budget: which C[r,u] is the bottleneck?
  §5  Theoretical lower bound via LP relaxation
  §6  S3 symmetry orbit analysis
  §7  Eigenspectrum of dead coupling matrix
  §8  Residual direction analysis: where does the max-abs live?
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import linalg as la

np.set_printoptions(precision=6, linewidth=140, suppress=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T

# ── Load best known solution ──────────────────────────────────
with open(Path(__file__).resolve().parent.parent / "shotgun_best.json") as f:
    data = json.load(f)
alpha = np.array(data["alpha"])   # (19, 9)
beta  = np.array(data["beta"])    # (19, 9)
gamma = np.array(data["gamma"])   # (19, 9)
fit   = data["fitness"]

R = RANK  # 19
D = DIM   # 9

# Masks
LIVE = (T != 0)              # (9,9,9) 27 True
DEAD = (T == 0)              # (9,9,9) 702 True
live_idx = np.where(LIVE.ravel())[0]
dead_idx = np.where(DEAD.ravel())[0]

# ── Reconstruct and verify ────────────────────────────────────
T_hat = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
Res = T_hat - T
maxabs = np.max(np.abs(Res))
print(f"{'='*72}")
print(f"  PURE-THEORY ANALYSIS OF RANK-19 CP DECOMPOSITION")
print(f"  Best fitness (max-abs): {maxabs:.10f}  (loaded: {fit:.10f})")
print(f"  ||T||² = {np.sum(T**2):.1f}   ||T_hat||² = {np.sum(T_hat**2):.6f}")
print(f"  ||R||² = {np.sum(Res**2):.6f}")
print(f"{'='*72}\n")

# ══════════════════════════════════════════════════════════════
# §1  Per-term rank-1 tensors and Gram matrices
# ══════════════════════════════════════════════════════════════
print("§1  GRAM MATRICES ON LIVE/DEAD SUBSPACES")
print("-" * 60)

# Each rank-1 term: T_k[a,b,c] = α[k,a]·β[k,b]·γ[k,c]
terms = np.einsum('ra,rb,rc->rabc', alpha, beta, gamma)  # (R, 9, 9, 9)
terms_flat = terms.reshape(R, 729)  # (R, 729)

# Project onto live/dead
live_proj = terms_flat[:, live_idx]   # (R, 27)
dead_proj = terms_flat[:, dead_idx]   # (R, 702)

G_live = live_proj @ live_proj.T   # (R, R)
G_dead = dead_proj @ dead_proj.T   # (R, R)

# Per-term energies
live_energy = np.diag(G_live)   # (R,)
dead_energy = np.diag(G_dead)   # (R,)
total_energy = live_energy + dead_energy

print(f"Total live energy (sum diag G_live): {np.sum(live_energy):.6f}")
print(f"Total dead energy (sum diag G_dead): {np.sum(dead_energy):.6f}")
print(f"Cross-term dead energy (off-diag G_dead): {np.sum(G_dead) - np.sum(dead_energy):.6f}")
print(f"  → Cancellation fraction: {1 - (np.sum(Res[DEAD]**2)) / np.sum(dead_energy):.6f}")

# Eigenvalues of G_dead
eig_dead = la.eigvalsh(G_dead)[::-1]
print(f"\nG_dead eigenvalues (top 10): {eig_dead[:10]}")
print(f"G_dead rank (>1e-10): {np.sum(eig_dead > 1e-10)}")
print(f"G_dead condition number: {eig_dead[0]/eig_dead[np.sum(eig_dead>1e-10)-1]:.2e}")

eig_live = la.eigvalsh(G_live)[::-1]
print(f"\nG_live eigenvalues (top 10): {eig_live[:10]}")
print(f"G_live rank (>1e-10): {np.sum(eig_live > 1e-10)}")

# ══════════════════════════════════════════════════════════════
# §2  Dead null space: what perturbations preserve cancellation?
# ══════════════════════════════════════════════════════════════
print(f"\n\n§2  DEAD NULL SPACE ANALYSIS")
print("-" * 60)

# dead_proj is (R, 702). The combined dead leakage is:
#   d = dead_proj.T @ w   where w = (1,...,1) for current decomposition
# But we care about the column space of dead_proj.
# SVD of dead_proj.T = (702, R)
U_d, s_d, VhT_d = la.svd(dead_proj.T, full_matrices=False)  # U(702,R), s(R), Vh(R,R)
print(f"SVD of dead_proj.T (702×{R}):")
print(f"  Singular values: {s_d}")
dead_rank = np.sum(s_d > 1e-10)
print(f"  Rank: {dead_rank}")
print(f"  Null space dimension: {R - dead_rank}")

if R > dead_rank:
    null_dirs = VhT_d[dead_rank:]  # rows = null vectors in R-space
    print(f"  Null vectors (changes to term weights that don't affect dead leakage):")
    for i, v in enumerate(null_dirs):
        print(f"    null[{i}] = {v}")
else:
    print(f"  No null space — all term weight changes affect dead leakage.")

# How much dead energy is "irreducible"?
# Project current dead residual onto column space of dead_proj
dead_residual = Res[DEAD]  # (702,) — current dead residual
print(f"\n  ||dead residual||² = {np.sum(dead_residual**2):.8f}")
print(f"  max |dead residual| = {np.max(np.abs(dead_residual)):.8f}")

# ══════════════════════════════════════════════════════════════
# §3  Signal matrix Σ: how terms serve fibers
# ══════════════════════════════════════════════════════════════
print(f"\n\n§3  SIGNAL MATRIX Σ (term × fiber contribution)")
print("-" * 60)

# Signal matrix: Σ[k, 3r+u] = Σ_s α[k,3r+s] · β[k,3s+u]
# This is the "outer product" of α and β projected onto live fibers
Sigma = np.zeros((R, 9))  # 9 fibers C[r,u]
for r in range(3):
    for u in range(3):
        fiber = 3*r + u
        for s in range(3):
            a_idx = 3*r + s
            b_idx = 3*s + u
            Sigma[:, fiber] += alpha[:, a_idx] * beta[:, b_idx]

# The live signal condition: γ.T @ Σ = 3·I_9
# (because each C[r,u] entry should be 1.0, and there are 3 per fiber)
GS = gamma.T @ Sigma  # (9, 9)... actually γ is (R,9), so γ.T is (9,R), Σ is (R,9)
# Wait: the target is T[3r+s, 3s+u, 3r+u] = 1 for all s
# So Σ_k Σ[k, 3r+u] · γ[k, 3r+u] = 3  (sum over 3 values of s)
# That gives: for each fiber f: Σ_k Σ[k,f] · γ[k,f] = 3

signal_check = np.array([np.dot(Sigma[:, f], gamma[:, f]) for f in range(9)])
print(f"Signal check (should all be 3.0): {signal_check}")
print(f"Max deviation from 3.0: {np.max(np.abs(signal_check - 3.0)):.2e}")

# Rank of Sigma
s_sigma = la.svdvals(Sigma)
print(f"\nΣ singular values: {s_sigma}")
print(f"Σ rank (>1e-10): {np.sum(s_sigma > 1e-10)}")

# Per-term signal strength
term_signal = np.linalg.norm(Sigma, axis=1)  # (R,)
print(f"\nPer-term |Σ_k| (signal norm):")
for k in range(R):
    print(f"  term {k:2d}: |Σ|={term_signal[k]:.4f}  "
          f"live_E={live_energy[k]:.4f}  dead_E={dead_energy[k]:.4f}  "
          f"efficiency={live_energy[k]/total_energy[k]:.3f}")

# ══════════════════════════════════════════════════════════════
# §4  Per-fiber residual budget
# ══════════════════════════════════════════════════════════════
print(f"\n\n§4  PER-FIBER RESIDUAL BUDGET")
print("-" * 60)

# For each output fiber C[r,u], the residual has 3 live entries and many dead
fiber_labels = [f"C[{r},{u}]" for r in range(3) for u in range(3)]

print(f"{'Fiber':<8} {'Live max':>10} {'Live RMS':>10} {'Dead max':>10} {'Dead RMS':>10} {'Total max':>10}")
for r in range(3):
    for u in range(3):
        c_idx = 3*r + u
        # Live entries for this fiber: T[3r+s, 3s+u, 3r+u] for s=0,1,2
        live_res = []
        for s in range(3):
            live_res.append(Res[3*r+s, 3*s+u, 3*r+u])
        live_res = np.array(live_res)

        # Dead entries in this output slice: Res[:, :, 3r+u] minus the live ones
        out_slice = Res[:, :, c_idx].ravel()  # 81 entries
        # Live entries in this slice are at (3r+s, 3s+u) for s=0,1,2
        dead_mask_slice = np.ones(81, dtype=bool)
        for s in range(3):
            dead_mask_slice[9*(3*r+s) + (3*s+u)] = False
        dead_res = out_slice[dead_mask_slice]

        print(f"C[{r},{u}]   "
              f"{np.max(np.abs(live_res)):10.6f} "
              f"{np.sqrt(np.mean(live_res**2)):10.6f} "
              f"{np.max(np.abs(dead_res)):10.6f} "
              f"{np.sqrt(np.mean(dead_res**2)):10.6f} "
              f"{max(np.max(np.abs(live_res)), np.max(np.abs(dead_res))):10.6f}")

# Which entries achieve the global max-abs?
flat_res = Res.ravel()
top_idx = np.argsort(np.abs(flat_res))[::-1][:20]
print(f"\nTop 20 residual entries (global max-abs = {maxabs:.10f}):")
for rank_i, idx in enumerate(top_idx):
    a, b, c = np.unravel_index(idx, (9, 9, 9))
    r_a, s_a = divmod(a, 3)
    s_b, u_b = divmod(b, 3)
    r_c, u_c = divmod(c, 3)
    is_live = (s_a == s_b) and (r_a == r_c) and (u_b == u_c)
    print(f"  #{rank_i+1:2d}: R[{a},{b},{c}] = {flat_res[idx]:+.8f}  "
          f"(r={r_a},s={s_a}|t={s_b},u={u_b}|out={r_c},{u_c})  "
          f"{'LIVE' if is_live else 'DEAD'}")

# ══════════════════════════════════════════════════════════════
# §5  Theoretical lower bound via LP relaxation
# ══════════════════════════════════════════════════════════════
print(f"\n\n§5  THEORETICAL BOUNDS")
print("-" * 60)

# Frobenius lower bound: ||R||² ≥ ||T||² - ||T_hat||² when ||T_hat||² < ||T||²
# Pythagorean: at ALS optimum, ||R||² + ||T_hat||² = ||T||² = 27
R_fro2 = np.sum(Res**2)
T_hat_fro2 = np.sum(T_hat**2)
print(f"Pythagorean check: ||R||²={R_fro2:.6f} + ||T_hat||²={T_hat_fro2:.6f} = {R_fro2+T_hat_fro2:.6f} (should ≈ 27)")

# Maxabs vs Frobenius: max|R| ≥ ||R||/√729
# But also: max|R| ≤ ||R|| (trivially)
fro_lower = np.sqrt(R_fro2) / np.sqrt(729)
print(f"Frobenius lower bound on max|R|: {fro_lower:.8f}  (||R||/√729)")
print(f"Actual max|R|: {maxabs:.8f}")
print(f"Ratio actual/lower: {maxabs/fro_lower:.4f}")

# Better bound: max|R| ≥ ||R||/√N where N = number of non-zero residuals 
# Count entries with |R| > 1e-10
n_active = np.sum(np.abs(flat_res) > 1e-10)
better_lower = np.sqrt(R_fro2) / np.sqrt(n_active)
print(f"\nActive residual entries (|R|>1e-10): {n_active}")
print(f"Active lower bound: {better_lower:.8f}  (||R||/√{n_active})")

# Rank-related bound: For rank-R CP, live signal uses R×9 parameters
# Dead cancellation uses the same parameters
# Degrees of freedom: 3×R×D = 513 parameters
# Live constraints: 27 (one per live entry)
# Dead constraints: 702 (one per dead entry) — want all zero
# Free parameters after live: 513 - 27 = 486
# But 702 dead constraints >> 486 free → underdetermined for zero dead
print(f"\nDegree-of-freedom analysis:")
print(f"  Parameters: 3×{R}×{D} = {3*R*D}")
print(f"  Live constraints: 27")
print(f"  Dead zero-constraints: 702")
print(f"  Free after live: {3*R*D - 27}")
print(f"  Dead deficit: {702 - (3*R*D - 27)} constraints beyond free params")
print(f"  → System is {'overdetermined' if 702 > 3*R*D-27 else 'underdetermined'} for exact dead zeroing")

# ══════════════════════════════════════════════════════════════
# §6  S3 symmetry orbit analysis
# ══════════════════════════════════════════════════════════════
print(f"\n\n§6  SYMMETRY & STRUCTURE OF RANK-1 TERMS")
print("-" * 60)

# The matrix multiplication tensor has S3 symmetry:
# T[a,b,c] is invariant under cyclic permutation (r,s,u) → (s,u,r)
# which maps (a,b,c) = (3r+s, 3s+u, 3r+u) → (3s+u, 3u+r, 3s+r)
# In factor space: this permutes (α, β, γ) cyclically

# Analyze each term's "fiber fingerprint"
# For term k, which fibers does it primarily serve?
print("Per-term fiber assignment (dominant fiber by |Σ·γ| contribution):")
# Contribution of term k to fiber f: Σ[k,f] · γ[k,f]
contrib = Sigma * gamma  # (R, 9) element-wise
for k in range(R):
    top_fiber = np.argmax(np.abs(contrib[k]))
    r_top, u_top = divmod(top_fiber, 3)
    print(f"  term {k:2d}: contrib = {contrib[k]}  "
          f"dom=C[{r_top},{u_top}] ({contrib[k, top_fiber]:+.4f})")

# How many terms primarily serve each fiber?
fiber_counts = np.zeros(9, dtype=int)
fiber_load = np.zeros(9)
for f in range(9):
    serving = np.abs(contrib[:, f]) > 0.1  # non-trivial contribution
    fiber_counts[f] = np.sum(serving)
    fiber_load[f] = np.sum(np.abs(contrib[:, f]))

print(f"\nFiber load distribution:")
for f in range(9):
    r, u = divmod(f, 3)
    print(f"  C[{r},{u}]: {fiber_counts[f]} serving terms, "
          f"total |contrib|={fiber_load[f]:.4f}, "
          f"target=3.0, actual signal={signal_check[f]:.6f}")

# ══════════════════════════════════════════════════════════════
# §7  Dead coupling eigenstructure
# ══════════════════════════════════════════════════════════════
print(f"\n\n§7  DEAD COUPLING MATRIX EIGENSTRUCTURE")
print("-" * 60)

# The dead cross-term matrix: C[i,j] = <T_i, T_j>_dead
# Its eigenvalues tell us about the interference structure
evals_d, evecs_d = la.eigh(G_dead)
evals_d = evals_d[::-1]
evecs_d = evecs_d[:, ::-1]

print(f"G_dead eigenvalues:")
for i, ev in enumerate(evals_d):
    print(f"  λ_{i:2d} = {ev:12.6f}  ({ev/np.sum(evals_d)*100:6.2f}%)")

# The top eigenvectors tell us which linear combinations of terms
# have the most dead energy (hardest to cancel)
print(f"\nTop 3 eigenvectors of G_dead (term coefficients):")
for i in range(min(3, R)):
    v = evecs_d[:, i]
    top_terms = np.argsort(np.abs(v))[::-1][:5]
    print(f"  v_{i}: eigenvalue={evals_d[i]:.6f}")
    print(f"    top terms: {[(t, f'{v[t]:+.4f}') for t in top_terms]}")

# Smallest eigenvectors: directions with least dead energy (most cancelled)
print(f"\nBottom 3 eigenvectors (most cancelled directions):")
for i in range(max(0, R-3), R):
    v = evecs_d[:, i]
    print(f"  v_{i}: eigenvalue={evals_d[i]:.8f}")

# ══════════════════════════════════════════════════════════════
# §8  Residual direction analysis: the bottleneck
# ══════════════════════════════════════════════════════════════
print(f"\n\n§8  BOTTLENECK ANALYSIS")
print("-" * 60)

# The max-abs is achieved at specific entries. What constrains those?
# Find all entries within 1% of max-abs
near_max = np.abs(flat_res) > 0.99 * maxabs
near_max_idx = np.where(near_max)[0]
print(f"Entries within 1% of max-abs ({maxabs:.8f}): {len(near_max_idx)}")

for idx in near_max_idx:
    a, b, c = np.unravel_index(idx, (9, 9, 9))
    r_a, s_a = divmod(a, 3)
    s_b, u_b = divmod(b, 3)
    r_c, u_c = divmod(c, 3)
    is_live = (s_a == s_b) and (r_a == r_c) and (u_b == u_c)

    # Per-term contributions to this entry
    per_term = alpha[:, a] * beta[:, b] * gamma[:, c]
    top_contrib = np.argsort(np.abs(per_term))[::-1][:5]

    print(f"\n  R[{a},{b},{c}] = {flat_res[idx]:+.10f}  {'LIVE' if is_live else 'DEAD'}")
    print(f"    (r={r_a},s={s_a}|t={s_b},u={u_b}) → out=({r_c},{u_c})")
    print(f"    Sum of {R} terms = {np.sum(per_term):.10f}, target = {T[a,b,c]:.0f}")
    print(f"    Top contributing terms: ", end="")
    for t in top_contrib:
        print(f"  k={t}:{per_term[t]:+.6f}", end="")
    print()

# ══════════════════════════════════════════════════════════════
# §9  Null-space directions for improvement
# ══════════════════════════════════════════════════════════════
print(f"\n\n§9  IMPROVEMENT DIRECTIONS (THEORY)")
print("-" * 60)

# Q: Is there a direction in parameter space that reduces max-abs
#    while keeping dead leakage controlled?
#
# The Jacobian of the residual w.r.t. all 513 parameters:
# J[entry, param] = ∂R[a,b,c]/∂θ_p
# For α: ∂R[a,b,c]/∂α[k,a'] = δ_{a,a'} · β[k,b] · γ[k,c]
# For β: ∂R[a,b,c]/∂β[k,b'] = α[k,a] · δ_{b,b'} · γ[k,c]
# For γ: ∂R[a,b,c]/∂γ[k,c'] = α[k,a] · β[k,b] · δ_{c,c'}
#
# Build Jacobian at dead entries only
print("Building Jacobian of dead residual w.r.t. all 513 parameters...")
n_dead = len(dead_idx)
n_params = 3 * R * D  # 513

J_dead = np.zeros((n_dead, n_params))
dead_coords = np.array(np.unravel_index(dead_idx, (9, 9, 9))).T  # (702, 3)

for k in range(R):
    for di in range(n_dead):
        a, b, c = dead_coords[di]
        # ∂R_dead[di]/∂α[k,a] = β[k,b]·γ[k,c]  (only if a matches)
        J_dead[di, k*D + a] += beta[k, b] * gamma[k, c]          # α block
        J_dead[di, R*D + k*D + b] += alpha[k, a] * gamma[k, c]   # β block
        J_dead[di, 2*R*D + k*D + c] += alpha[k, a] * beta[k, b]  # γ block

# SVD of J_dead
U_j, s_j, Vh_j = la.svd(J_dead, full_matrices=False)
print(f"J_dead shape: {J_dead.shape}")
print(f"J_dead singular values (top 20): {s_j[:20]}")
print(f"J_dead rank (>1e-10): {np.sum(s_j > 1e-10)}")
jdead_rank = np.sum(s_j > 1e-10)
print(f"Null space dimension: {n_params - jdead_rank}")
print(f"  → {n_params - jdead_rank} parameter directions that don't change dead residual at all")

# Now do the same for the bottleneck entries
print(f"\nBuilding Jacobian at near-max entries...")
J_max = np.zeros((len(near_max_idx), n_params))
max_coords = np.array(np.unravel_index(near_max_idx, (9, 9, 9))).T

for k in range(R):
    for mi in range(len(near_max_idx)):
        a, b, c = max_coords[mi]
        J_max[mi, k*D + a] += beta[k, b] * gamma[k, c]
        J_max[mi, R*D + k*D + b] += alpha[k, a] * gamma[k, c]
        J_max[mi, 2*R*D + k*D + c] += alpha[k, a] * beta[k, b]

# Directions that reduce the bottleneck entries: J_max.T @ sign(R_max)
R_max_signs = np.sign(flat_res[near_max_idx])
descent_dir = -J_max.T @ R_max_signs  # steepest descent for max entries
descent_dir /= (la.norm(descent_dir) + 1e-30)

# Project onto null space of J_dead to get a direction that
# reduces bottleneck WITHOUT increasing dead leakage
if n_params > jdead_rank:
    null_basis = Vh_j[jdead_rank:].T  # (513, null_dim)
    proj_descent = null_basis @ (null_basis.T @ descent_dir)
    proj_norm = la.norm(proj_descent)
    if proj_norm > 1e-10:
        proj_descent /= proj_norm
        # Predicted improvement
        improvement = J_max @ proj_descent
        print(f"\nDead-null projected descent direction found (norm={proj_norm:.6f})")
        print(f"  Predicted change at max entries: {improvement}")
        print(f"  Max predicted reduction: {np.max(np.abs(improvement)):.8f}")
    else:
        print(f"\nDescent direction has NO component in dead null space!")
        print(f"  → Bottleneck entries are STRUCTURALLY LOCKED by dead constraints.")
else:
    print(f"\nNo dead null space exists — all directions affect dead residual.")

# ══════════════════════════════════════════════════════════════
# §10  Coupling matrix: who interferes with whom
# ══════════════════════════════════════════════════════════════
print(f"\n\n§10  PAIRWISE COUPLING ANALYSIS")
print("-" * 60)

# For each pair (i,j), G_dead[i,j] measures dead-space coupling
# Normalize by geometric mean of self-energies
coupling = np.zeros((R, R))
for i in range(R):
    for j in range(R):
        denom = np.sqrt(dead_energy[i] * dead_energy[j])
        if denom > 1e-15:
            coupling[i, j] = G_dead[i, j] / denom

# Find most destructive pairs (most negative coupling = most cancellation)
pairs = []
for i in range(R):
    for j in range(i+1, R):
        pairs.append((i, j, coupling[i, j], G_dead[i, j]))
pairs.sort(key=lambda x: x[2])  # most negative first

print(f"Top 10 DESTRUCTIVE pairs (most cancellation):")
for i, j, norm_c, raw_c in pairs[:10]:
    print(f"  ({i:2d},{j:2d}): normalized={norm_c:+.6f}  raw=<T_i,T_j>_dead={raw_c:+.6f}")

print(f"\nTop 10 CONSTRUCTIVE pairs (most reinforcement):")
for i, j, norm_c, raw_c in pairs[-10:][::-1]:
    print(f"  ({i:2d},{j:2d}): normalized={norm_c:+.6f}  raw=<T_i,T_j>_dead={raw_c:+.6f}")

# ══════════════════════════════════════════════════════════════
# §11  Fiber-mode decomposition
# ══════════════════════════════════════════════════════════════
print(f"\n\n§11  RANK-1 TERM STRUCTURE IN FIBER MODES")
print("-" * 60)

# Orthonormal fiber basis
p0 = np.array([1, 1, 1]) / np.sqrt(3)   # signal mode
p1 = np.array([1, -1, 0]) / np.sqrt(2)  # nuisance 1
p2 = np.array([1, 1, -2]) / np.sqrt(6)  # nuisance 2

# For each term k, decompose α[k,:] into 3×3 block structure
# α[k, 3r+s] → row r, column s → project column onto p0,p1,p2
print("Per-term fiber mode decomposition (α factor):")
print(f"  {'term':>4}  {'|σ_α|':>8}  {'|η1_α|':>8}  {'|η2_α|':>8}  {'ratio σ/(η1+η2)':>16}")
for k in range(R):
    # α[k] reshaped as (3, 3): row r, column s
    ak = alpha[k].reshape(3, 3)
    sigma_a = ak @ p0   # (3,) signal projection
    eta1_a  = ak @ p1   # (3,) nuisance-1
    eta2_a  = ak @ p2   # (3,) nuisance-2
    ns_a = la.norm(sigma_a)
    ne1_a = la.norm(eta1_a)
    ne2_a = la.norm(eta2_a)
    ratio = ns_a / (ne1_a + ne2_a + 1e-15)
    print(f"  {k:4d}  {ns_a:8.4f}  {ne1_a:8.4f}  {ne2_a:8.4f}  {ratio:16.4f}")

# Same for β
print(f"\nPer-term fiber mode decomposition (β factor):")
print(f"  {'term':>4}  {'|σ_β|':>8}  {'|η1_β|':>8}  {'|η2_β|':>8}  {'ratio σ/(η1+η2)':>16}")
for k in range(R):
    bk = beta[k].reshape(3, 3)
    sigma_b = bk @ p0
    eta1_b  = bk @ p1
    eta2_b  = bk @ p2
    ns_b = la.norm(sigma_b)
    ne1_b = la.norm(eta1_b)
    ne2_b = la.norm(eta2_b)
    ratio = ns_b / (ne1_b + ne2_b + 1e-15)
    print(f"  {k:4d}  {ns_b:8.4f}  {ne1_b:8.4f}  {ne2_b:8.4f}  {ratio:16.4f}")

# ══════════════════════════════════════════════════════════════
# §12  SUMMARY: The structural picture
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'='*72}")
print(f"  STRUCTURAL SUMMARY")
print(f"{'='*72}")

live_maxabs = np.max(np.abs(Res[LIVE]))
dead_maxabs = np.max(np.abs(Res[DEAD]))
print(f"Max-abs breakdown: live={live_maxabs:.8f}, dead={dead_maxabs:.8f}")
print(f"  → Bottleneck is {'LIVE' if live_maxabs >= dead_maxabs else 'DEAD'} entries")

print(f"\nDead cancellation: {1 - np.sum(Res[DEAD]**2)/np.sum(dead_energy):.4%}")
print(f"Dead null space dim: {R - dead_rank} (in term-weight space)")
print(f"Jacobian null space dim: {n_params - jdead_rank} (in full 513-param space)")
print(f"Σ rank: {np.sum(s_sigma > 1e-10)}")
print(f"DOF deficit for exact dead zeroing: {702 - (3*R*D - 27)}")

# Count how saturated the bottleneck is
n_tight = np.sum(np.abs(flat_res) > 0.95 * maxabs)
print(f"\nEntries within 5% of max-abs: {n_tight}")
print(f"  → {'Highly distributed' if n_tight > 50 else 'Localized'} bottleneck")

print(f"\nDone.")
