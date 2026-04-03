#!/usr/bin/env python3
"""
Deep structural portrait of the best rank-19 candidate.
What does this decomposition look like from the inside?
"""

import json, numpy as np
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=120)

with open(Path(__file__).parent.parent / "slp_turbo_best.json") as f:
    data = json.load(f)

alpha = np.array(data["alpha"])  # (19, 9)
beta = np.array(data["beta"])    # (19, 9)
gamma = np.array(data["gamma"])  # (19, 9)
R, n2 = alpha.shape
n = 3
fitness = data["fitness"]

print(f"Rank-19 candidate, fitness (max-abs residual) = {fitness:.6f}")
print(f"Shape: {R} terms, each with alpha(3x3), beta(3x3), gamma(3x3)")

# ── 1. RECONSTRUCT TENSOR AND EXAMINE RESIDUAL ──
T_target = np.zeros((n2, n2, n2))
for r in range(n):
    for u in range(n):
        T_target[n*r + 0, 0 + u, n*r + u] = 1  # s=t=0... 
# Actually build it properly
T_target = np.zeros((9, 9, 9))
for i in range(3):
    for j in range(3):
        for k in range(3):
            T_target[3*i+j, 3*j+k, 3*i+k] = 1  # T[a,b,c] = delta(a=ir+s, b=st+u, c=ir+u) sum_s

T_recon = np.zeros((9, 9, 9))
for k in range(R):
    T_recon += np.einsum('i,j,k', alpha[k], beta[k], gamma[k])

residual = T_recon - T_target
print(f"\n{'='*70}")
print(f"  1. RESIDUAL ANATOMY")
print(f"{'='*70}")
print(f"Max-abs residual: {np.max(np.abs(residual)):.6f}")
print(f"Frobenius norm:   {np.linalg.norm(residual):.6f}")
print(f"L1 norm:          {np.sum(np.abs(residual)):.4f}")

# Live vs dead
live_mask = T_target != 0
dead_mask = T_target == 0
n_live = np.sum(live_mask)
n_dead = np.sum(dead_mask)

live_resid = residual[live_mask]
dead_resid = residual[dead_mask]
print(f"\nLive entries ({n_live}):  max-abs={np.max(np.abs(live_resid)):.6f}, rms={np.sqrt(np.mean(live_resid**2)):.6f}")
print(f"Dead entries ({n_dead}): max-abs={np.max(np.abs(dead_resid)):.6f}, rms={np.sqrt(np.mean(dead_resid**2)):.6f}")

# Where are the worst residuals?
flat_idx = np.argsort(np.abs(residual.flatten()))[::-1]
print(f"\nTop 10 worst residual entries:")
for i in range(10):
    idx = flat_idx[i]
    a, b, c = np.unravel_index(idx, (9, 9, 9))
    target_val = T_target[a, b, c]
    recon_val = T_recon[a, b, c]
    typ = "LIVE" if target_val != 0 else "DEAD"
    print(f"  T[{a},{b},{c}] = {recon_val:+.6f} (target {target_val:.0f}) [{typ}] resid={residual[a,b,c]:+.6f}")

# ── 2. FACTOR NORMS AND SCALES ──
print(f"\n{'='*70}")
print(f"  2. FACTOR SCALES")
print(f"{'='*70}")
alpha_norms = np.linalg.norm(alpha, axis=1)
beta_norms = np.linalg.norm(beta, axis=1)
gamma_norms = np.linalg.norm(gamma, axis=1)
term_weights = alpha_norms * beta_norms * gamma_norms

print(f"Term weights (||α||·||β||·||γ||):")
order = np.argsort(term_weights)[::-1]
for i, k in enumerate(order):
    print(f"  term {k:2d}: |α|={alpha_norms[k]:.3f}  |β|={beta_norms[k]:.3f}  |γ|={gamma_norms[k]:.3f}  weight={term_weights[k]:.3f}")

print(f"\nWeight stats: mean={np.mean(term_weights):.3f}, std={np.std(term_weights):.3f}, "
      f"min={np.min(term_weights):.3f}, max={np.max(term_weights):.3f}")

# ── 3. GAMMA STRUCTURE ──
print(f"\n{'='*70}")
print(f"  3. GAMMA STRUCTURE (output weights)")
print(f"{'='*70}")
G = gamma  # (19, 9)
print(f"Gamma shape: {G.shape}")
print(f"rank(Gamma^T) = {np.linalg.matrix_rank(G.T, tol=1e-3)}")

# Check if gamma rows are approximately sparse or structured
for k in range(R):
    g = gamma[k].reshape(3, 3)
    row_norms = np.linalg.norm(g, axis=1)
    col_norms = np.linalg.norm(g, axis=0)
    # Check near-zero rows/cols
    near_zero_rows = np.sum(row_norms < 0.05)
    near_zero_cols = np.sum(col_norms < 0.05)
    diag_dom = np.sum(np.abs(np.diag(g))) / np.sum(np.abs(g))
    tags = []
    if near_zero_rows > 0: tags.append(f"{near_zero_rows} zero-rows")
    if near_zero_cols > 0: tags.append(f"{near_zero_cols} zero-cols")
    if diag_dom > 0.6: tags.append("diag-dominant")
    if np.max(np.abs(g - g.T)) < 0.05 * np.max(np.abs(g)): tags.append("~symmetric")
    if np.max(np.abs(g + g.T)) < 0.05 * np.max(np.abs(g)): tags.append("~antisym")
    tag_str = ", ".join(tags) if tags else "-"
    print(f"  γ_{k:2d}: norms=[{row_norms[0]:.2f},{row_norms[1]:.2f},{row_norms[2]:.2f}] {tag_str}")

# ── 4. FIBER PRODUCTS AND MODE DECOMPOSITION ──
print(f"\n{'='*70}")
print(f"  4. FIBER MODE DECOMPOSITION")
print(f"{'='*70}")

# Build fiber products p_k[s,r,u] = α_k[3r+s] · β_k[3s+u]
P = np.zeros((R, 3, 3, 3))  # [k, s, r, u]
for k in range(R):
    a = alpha[k].reshape(3, 3)  # a[r,s]
    b = beta[k].reshape(3, 3)   # b[s,u] (but stored as b[t,u])
    for s in range(3):
        for r in range(3):
            for u in range(3):
                P[k, s, r, u] = a[r, s] * b[s, u]

# Sigma_k[r,u] = sum_s P[k,s,r,u]
Sigma = np.sum(P, axis=1)  # (R, 3, 3) = (R, r, u)
Sigma_flat = Sigma.reshape(R, 9)

# Eta1 = P[:,0,:,:] - P[:,1,:,:]
# Eta2 = P[:,1,:,:] - P[:,2,:,:]
Eta1 = (P[:, 0] - P[:, 1]).reshape(R, 9)
Eta2 = (P[:, 1] - P[:, 2]).reshape(R, 9)
H = np.hstack([Eta1, Eta2])  # (R, 18)

# Delta: cross-fiber products
Delta_cols = []
for r in range(3):
    for s in range(3):
        for t in range(3):
            if s == t:
                continue
            for u in range(3):
                col = np.array([alpha[k, 3*r+s] * beta[k, 3*t+u] for k in range(R)])
                Delta_cols.append(col)
Delta = np.column_stack(Delta_cols)  # (R, 54)

Gamma_T = gamma  # (R, 9) — but Gamma as used is (9, R)
G_mat = gamma.T  # (9, R)

print(f"Sigma: {Sigma_flat.shape}, rank = {np.linalg.matrix_rank(Sigma_flat, tol=1e-3)}")
print(f"H = [Eta1|Eta2]: {H.shape}, rank = {np.linalg.matrix_rank(H, tol=1e-3)}")
print(f"Delta: {Delta.shape}, rank = {np.linalg.matrix_rank(Delta, tol=1e-3)}")

# Check Gamma @ Sigma ≈ 3I?
GS = G_mat @ Sigma_flat  # (9, 9)
print(f"\nGamma @ Sigma (should be 3·I₉):")
print(f"  max |GΣ - 3I| = {np.max(np.abs(GS - 3*np.eye(9))):.6f}")

# Check Gamma @ H ≈ 0?
GH = G_mat @ H  # (9, 18)
print(f"  max |GH| = {np.max(np.abs(GH)):.6f}  (should be 0)")

# Check Gamma @ Delta ≈ 0?
GD = G_mat @ Delta  # (9, 54)
print(f"  max |GΔ| = {np.max(np.abs(GD)):.6f}  (should be 0)")

# ker(Gamma) dimension
ker_dim = R - np.linalg.matrix_rank(G_mat, tol=1e-3)
print(f"\ndim(ker(Γ)) = {ker_dim}  (R - rank(Γ) = {R} - {np.linalg.matrix_rank(G_mat, tol=1e-3)})")

# H rank in ker(Gamma)
U_ker = np.linalg.svd(G_mat, full_matrices=True)[2][np.linalg.matrix_rank(G_mat, tol=1e-3):]  # (ker_dim, R)
H_proj = U_ker @ H
Delta_proj = U_ker @ Delta
print(f"rank(H_proj) = {np.linalg.matrix_rank(H_proj, tol=1e-3)}  (need {ker_dim} to span ker(Γ))")
print(f"rank(Δ_proj) = {np.linalg.matrix_rank(Delta_proj, tol=1e-3)}")
print(f"rank([H|Δ]_proj) = {np.linalg.matrix_rank(np.hstack([H_proj, Delta_proj]), tol=1e-3)}")

# Sigma_min of H_proj
sv = np.linalg.svd(H_proj, compute_uv=False)
print(f"σ_min(H_proj) = {sv[-1]:.6f}  (0 = rank deficient)")

# ── 5. NEAR-SYMMETRIES ──
print(f"\n{'='*70}")
print(f"  5. TERM PAIR RELATIONSHIPS")
print(f"{'='*70}")

# Check for approximately parallel/antiparallel term pairs
def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)

print("Near-parallel/antiparallel alpha pairs (|cos| > 0.95):")
for i in range(R):
    for j in range(i+1, R):
        c = cosine_sim(alpha[i], alpha[j])
        if abs(c) > 0.95:
            print(f"  α_{i} vs α_{j}: cos = {c:+.4f}")

print("Near-parallel/antiparallel gamma pairs (|cos| > 0.95):")
for i in range(R):
    for j in range(i+1, R):
        c = cosine_sim(gamma[i], gamma[j])
        if abs(c) > 0.95:
            print(f"  γ_{i} vs γ_{j}: cos = {c:+.4f}")

# Check for α_i ≈ -α_j patterns
print("\nNear-negation pairs (α_i ≈ -α_j AND similar beta):")
for i in range(R):
    for j in range(i+1, R):
        ca = cosine_sim(alpha[i], alpha[j])
        if ca < -0.95:
            cb = cosine_sim(beta[i], beta[j])
            cg = cosine_sim(gamma[i], gamma[j])
            print(f"  terms ({i},{j}): cos_α={ca:+.4f}, cos_β={cb:+.4f}, cos_γ={cg:+.4f}")

# ── 6. SPECTRAL STRUCTURE ──
print(f"\n{'='*70}")
print(f"  6. SPECTRAL STRUCTURE")
print(f"{'='*70}")

# SVD of the factor matrices
for name, mat in [("Alpha", alpha), ("Beta", beta), ("Gamma", gamma)]:
    sv = np.linalg.svd(mat, compute_uv=False)
    print(f"  {name} singular values: {sv[:5]}{'...' if len(sv)>5 else ''}")
    print(f"    rank={np.sum(sv > 1e-3)}, cond={sv[0]/sv[-1]:.1f}, eff_rank={np.sum(sv)**2/np.sum(sv**2):.2f}")

# ── 7. PYTHAGOREAN CHECK ──
print(f"\n{'='*70}")
print(f"  7. PYTHAGOREAN / ENERGY BUDGET")
print(f"{'='*70}")
live_energy = np.sum(T_recon[live_mask]**2)
dead_energy = np.sum(T_recon[dead_mask]**2)
target_energy = np.sum(T_target**2)  # = 27
recon_energy = np.sum(T_recon**2)
print(f"||T_target||² = {target_energy:.1f}")
print(f"||T_recon||²  = {recon_energy:.4f}")
print(f"  live component: {live_energy:.4f}")
print(f"  dead component: {dead_energy:.4f}")
print(f"||residual||²  = {np.sum(residual**2):.6f}")
print(f"Pythagorean: ||T_recon||² = ||T_target||² + ||residual||² - 2<T_target,residual>")
cross = np.sum(T_target * residual)
print(f"  <T_target, residual> = {cross:.6f}")
print(f"  Check: {target_energy:.1f} + {np.sum(residual**2):.4f} - 2*{cross:.4f} = {target_energy + np.sum(residual**2) - 2*cross:.4f} vs {recon_energy:.4f}")

# ── 8. TERM-BY-TERM CONTRIBUTION ──
print(f"\n{'='*70}")
print(f"  8. TERM CONTRIBUTIONS TO LIVE/DEAD")
print(f"{'='*70}")
for k in range(R):
    Tk = np.einsum('i,j,k', alpha[k], beta[k], gamma[k])
    live_contrib = np.sum(np.abs(Tk[live_mask]))
    dead_contrib = np.sum(np.abs(Tk[dead_mask]))
    ratio = live_contrib / (dead_contrib + 1e-15)
    print(f"  term {k:2d}: live_L1={live_contrib:.3f}  dead_L1={dead_contrib:.3f}  ratio={ratio:.3f}")

# ── 9. THE GHOST: WHAT'S MISSING? ──
print(f"\n{'='*70}")
print(f"  9. THE GHOST: WHAT RANK-19 CANNOT REACH")
print(f"{'='*70}")
print(f"Residual has rank = {np.linalg.matrix_rank(residual.reshape(9,81), tol=0.01)}")
print(f"If we had a 20th term that exactly cancelled the residual, it would need:")
# The residual as a (9,9,9) tensor — what's its CP structure?
R_flat = residual.reshape(9, 81)
U, S, Vt = np.linalg.svd(R_flat, full_matrices=False)
print(f"  SVD of residual (9×81): {S[:5]}")
print(f"  The residual is approximately rank-{np.sum(S > 0.01)} as a matrix")

# Perfect 20th term would be: residual = α₂₀ ⊗ β₂₀ ⊗ γ₂₀
# Check if residual is close to rank-1
best_rank1_approx = S[0]
total = np.linalg.norm(residual)
print(f"  Best rank-1 approx captures {best_rank1_approx/total*100:.1f}% of ||residual||")
print(f"  The residual is {'nearly' if best_rank1_approx/total > 0.9 else 'NOT'} rank-1")

# Residual distribution
print(f"\nResidual histogram (absolute values):")
abs_resid = np.abs(residual.flatten())
abs_resid_nonzero = abs_resid[abs_resid > 1e-6]
for threshold in [0.01, 0.02, 0.03, 0.05, 0.07, 0.073]:
    count = np.sum(abs_resid > threshold)
    print(f"  |r| > {threshold:.3f}: {count:4d} entries")

# ── 10. WHAT DO I LOOK LIKE? ──
print(f"\n{'='*70}")
print(f"  10. SELF-PORTRAIT")
print(f"{'='*70}")

# Term 10 is interesting — tiny norm
print(f"\nTerm 10 (the ghost term):")
print(f"  α = {alpha[10]}")
print(f"  β = {beta[10]}")
print(f"  γ = {gamma[10]}")
print(f"  |α|={np.linalg.norm(alpha[10]):.4f}, |β|={np.linalg.norm(beta[10]):.4f}, |γ|={np.linalg.norm(gamma[10]):.4f}")
print(f"  weight = {np.prod([np.linalg.norm(alpha[10]), np.linalg.norm(beta[10]), np.linalg.norm(gamma[10])]):.6f}")

# Overall structure summary
print(f"\n--- SUMMARY: WHAT I LOOK LIKE ---")
print(f"I am 19 rank-1 terms trying to multiply 3×3 matrices.")
print(f"My best residual is {fitness:.6f} — tantalizingly close to 0.")
print(f"")

# Count near-integer gammas
gamma_rounded = np.round(gamma)
gamma_int_frac = np.mean(np.abs(gamma - gamma_rounded) < 0.05)
alpha_rounded = np.round(alpha)
alpha_int_frac = np.mean(np.abs(alpha - alpha_rounded) < 0.05)
print(f"Fraction of near-integer entries: α={alpha_int_frac:.1%}, γ={gamma_int_frac:.1%}")

# Effective number of "big" terms vs "small" terms
big_terms = np.sum(term_weights > 5)
medium_terms = np.sum((term_weights > 1) & (term_weights <= 5))
small_terms = np.sum(term_weights <= 1)
print(f"Term size distribution: {big_terms} large (>5), {medium_terms} medium (1-5), {small_terms} small (<1)")

# Check balancedness: how well does each output entry get served?
output_coverage = np.zeros(9)
for c in range(9):
    output_coverage[c] = np.sum(np.abs(gamma[:, c]))
print(f"Output coverage (Σ|γ[:,c]|): min={np.min(output_coverage):.2f}, max={np.max(output_coverage):.2f}, "
      f"cv={np.std(output_coverage)/np.mean(output_coverage):.3f}")
