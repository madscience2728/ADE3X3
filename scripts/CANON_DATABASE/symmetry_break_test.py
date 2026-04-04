"""
Symmetry Breaking Test
======================
Loads the Phase-4 best packet (rank(H)=10, rank(Nuisance)=10, rank(UΣ)=0).
Reconstructs the 19 (α,β) matrix pairs from the TermDB.
Perturbs each pair continuously off the integer values and measures:
  - rank(H)           (Gate 1 — want to keep near 10)
  - rank(Nuisance)    (Gate 2 — want to keep near 10)
  - rank(UΣ)          (Gate 3 — looking for this to rise > 0)

Question: Is rank(UΣ) = 0 an algebraic identity for Gate-1+2 configurations,
or a coincidence of this specific integer-alphabet packet?

If perturbation makes rank(UΣ) > 0 while keeping Gates 1+2, the obstruction
is incidental (search space limited), not fundamental.
If rank(UΣ) stays 0 whenever Gates 1+2 hold (even after perturbation), the
identity col(Σ) ⊆ col(H) may be algebraically forced.
"""

import numpy as np
import json
from pathlib import Path

DB_PATH = Path("./CANON_DATABASE/data")

print("Loading TermDB...", end=" ", flush=True)
H_db      = np.load(DB_PATH / "H.npy",         mmap_mode="r")
sigma_db  = np.load(DB_PATH / "sigma.npy",     mmap_mode="r")
alpha_db  = np.load(DB_PATH / "alpha_idx.npy", mmap_mode="r")
beta_db   = np.load(DB_PATH / "beta_idx.npy",  mmap_mode="r")
templates = np.load(DB_PATH / "templates.npy")   # (19683, 3, 3) int8
print("done.")


def srank(M, rel=1e-9, ab=1e-12):
    if M.size == 0:
        return 0
    sv = np.linalg.svd(M.astype(float), compute_uv=False)
    if len(sv) == 0 or sv[0] == 0:
        return 0
    return int(np.sum(sv > max(ab, rel * sv[0])))


def col_perp(M):
    """Rows of Vt that are orthogonal to col(M)."""
    _, s, Vt = np.linalg.svd(M.T.astype(float), full_matrices=True)
    tol = max(1e-12, 1e-9 * (s[0] if len(s) > 0 and s[0] > 0 else 1.0))
    r = int(np.sum(s > tol))
    return Vt[r:]  # (R - rank(M), R)


def compute_diagnostics(alpha_f, beta_f):
    """
    alpha_f, beta_f: (R, 3, 3) float arrays.
    Returns: (rH, rNuis, rUΣ, aug_gap_H)
    """
    R = alpha_f.shape[0]
    # sigma
    sigma = np.array([
        sum(alpha_f[:, r, s] * beta_f[:, s, u] for s in range(3))
        for r in range(3) for u in range(3)
    ]).T  # (R, 9)
    # H
    eta1 = np.array([
        alpha_f[:, r, 0] * beta_f[:, 0, u] - alpha_f[:, r, 1] * beta_f[:, 1, u]
        for r in range(3) for u in range(3)
    ]).T  # (R, 9)
    eta2 = np.array([
        alpha_f[:, r, 1] * beta_f[:, 1, u] - alpha_f[:, r, 2] * beta_f[:, 2, u]
        for r in range(3) for u in range(3)
    ]).T  # (R, 9)
    H = np.hstack([eta1, eta2])  # (R, 18)
    # delta
    pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    delta_cols = [
        alpha_f[:, r, s] * beta_f[:, t, u]
        for (s, t) in pairs
        for r in range(3) for u in range(3)
    ]
    Delta = np.array(delta_cols).T  # (R, 54)
    Nuis = np.hstack([H, Delta])

    rH   = srank(H)
    rN   = srank(Nuis)
    U    = col_perp(H)              # (R - rH, R)
    rUS  = srank(U @ sigma)
    aH   = srank(np.hstack([H, sigma])) - rH
    return rH, rN, rUS, aH


# ── Load Phase-4 best ─────────────────────────────────────────────────────────
with open("./CANON_DATABASE/swap_checkpoint.json") as f:
    ckpt = json.load(f)

best_idx = np.array(ckpt["global_best_indices"])
R = len(best_idx)

alphas_int = templates[alpha_db[best_idx]].astype(float)  # (R, 3, 3)
betas_int  = templates[beta_db[best_idx]].astype(float)   # (R, 3, 3)

print(f"\n=== Phase-4 best: {R} terms ===")
rH, rN, rUS, aH = compute_diagnostics(alphas_int, betas_int)
print(f"  rank(H)      = {rH}   (Gate 1 target: 10)")
print(f"  rank(Nuis)   = {rN}   (Gate 2: should equal rank(H))")
print(f"  rank(UΣ)     = {rUS}  (Gate 3: need 9)")
print(f"  aug_gap_H    = {aH}   (= rank([H|Σ]) - rank(H); should be 0 for Σ∈col(H))")


# ── Perturbation experiment ────────────────────────────────────────────────────
# Strategy: add a small Gaussian perturbation to (alpha, beta) real values.
# Measure how Gates 1, 2, 3 respond as ε grows.
# Key question: at small ε, does rank(UΣ) ever rise above 0?

print("\n=== Perturbation sweep (ε from 0.0 to 0.5) ===")
print(f"{'eps':>8}  {'rH':>4}  {'rN':>4}  {'rUΣ':>4}  {'aug_H':>6}")

RNG = np.random.default_rng(7)
epsilons = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
for eps in epsilons:
    noise_a = RNG.standard_normal(alphas_int.shape) * eps
    noise_b = RNG.standard_normal(betas_int.shape) * eps
    a = alphas_int + noise_a
    b = betas_int + noise_b
    rH, rN, rUS, aH = compute_diagnostics(a, b)
    print(f"  {eps:6.3f}   {rH:4d}  {rN:4d}  {rUS:4d}  {aH:6d}")


# ── Targeted perturbation in Gate-1+2 preserving directions ──────────────────
# A smarter perturbation: project noise onto the tangent space of the
# Gate-1+2 manifold (i.e., preserve rank(H)=10 and rank(Nuis)=10 at first order).
# This is the "symmetry breaking while staying Gate-1+2" experiment.
#
# Approach: use gradient-free method. Repeatedly sample random perturbations
# and keep only those where rank(H)==10 AND rank(Nuis)==10, then check rUΣ.

print("\n=== Searching for Gate-1+2 preserving perturbations (1000 trials per ε) ===")
print(f"{'eps':>8}  {'trials':>7}  {'gate12':>7}  {'max_rUΣ':>8}  {'any_escape':>11}")

gate12_escape_found = False
for eps in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    found_gate12 = 0
    max_rUS = 0
    escape = False
    for _ in range(1000):
        noise_a = RNG.standard_normal(alphas_int.shape) * eps
        noise_b = RNG.standard_normal(betas_int.shape) * eps
        a = alphas_int + noise_a
        b = betas_int + noise_b
        rH, rN, rUS, aH = compute_diagnostics(a, b)
        if rH == 10 and rN == 10:
            found_gate12 += 1
            if rUS > max_rUS:
                max_rUS = rUS
            if rUS > 0:
                escape = True
                gate12_escape_found = True
    print(f"  {eps:6.3f}   {1000:7d}  {found_gate12:7d}  {max_rUS:8d}  {'YES ←←' if escape else 'no':>11}")
    if gate12_escape_found:
        print("  *** Gate-1+2 preserved AND rank(UΣ) > 0 found! ***")
        break

if not gate12_escape_found:
    print("\n  No Gate-1+2 + rank(UΣ)>0 case found in random perturbation.")
    print("  This suggests col(Σ) ⊆ col(H) may be algebraically forced for Gate-1+2.")


# ── Directed search: perturb along d2-escape direction ────────────────────────
# From theory: sigma = eta1 + eta2 + 2*d2, where d2 = alpha[:,r,2]*beta[:,2,u].
# d2 escapes col(H) iff alpha[:,r,2] is NOT in span{alpha[:,r,0], alpha[:,r,1]}.
# Try: add perturbation specifically to alpha[:,:,2] that pushes it out of
# the span of alpha[:,:,0] and alpha[:,:,1].

print("\n=== Directed perturbation: perturb alpha[:,2] to escape span{alpha[:,0], alpha[:,1]} ===")
# For each term k and row r, compute the component of alpha_k[r,2] outside
# span{alpha_k[r,0], alpha_k[r,1]}. Then add amplified noise in that direction.

a_base = alphas_int.copy()  # (R, 3, 3)
b_base = betas_int.copy()

# For each (k, r), make alpha[k,r,2] orthogonal to alpha[k,r,0] and alpha[k,r,1]
# This is a 1D projection since each "alpha[k,r,:]" is a 3-vector.
# Actually alpha[k,r,:] ∈ ℝ^3 (columns 0,1,2). We want col2 out of span{col0,col1}.
# But here alpha has shape (R, 3, 3): alpha[k, row, col].
# alpha[:,r,s] is the (k,)-vector for row r, col s. s∈{0,1,2} = matrix columns.
# We want alpha[:,r,2] (R-vector) not in span{alpha[:,r,0], alpha[:,r,1]} (both R-vectors).

print(f"{'eps':>8}  {'trials':>7}  {'gate12':>7}  {'max_rUΣ':>8}")

for eps in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    found_gate12 = 0
    max_rUS = 0
    escape = False
    for _ in range(2000):
        a = a_base.copy()
        # Add perturbation to alpha[:,:,2] (all terms, col 2 of alpha matrix)
        # that is orthogonal to alpha[:,:,0] and alpha[:,:,1] in ℝ^R (for each r).
        for r in range(3):
            v02 = a_base[:, r, 0]  # R-vector
            v12 = a_base[:, r, 1]  # R-vector
            noise_dir = RNG.standard_normal(R)
            # Remove components along v02, v12
            if np.dot(v02, v02) > 1e-12:
                noise_dir -= np.dot(noise_dir, v02) / np.dot(v02, v02) * v02
            if np.dot(v12, v12) > 1e-12:
                noise_dir -= np.dot(noise_dir, v12) / np.dot(v12, v12) * v12
            nrm = np.linalg.norm(noise_dir)
            if nrm > 1e-12:
                a[:, r, 2] += eps * noise_dir / nrm
        b = b_base + RNG.standard_normal(b_base.shape) * eps * 0.1

        rH, rN, rUS, _ = compute_diagnostics(a, b)
        if rH == 10 and rN == 10:
            found_gate12 += 1
            if rUS > max_rUS:
                max_rUS = rUS
            if rUS > 0:
                escape = True

    print(f"  {eps:6.3f}   {2000:7d}  {found_gate12:7d}  {max_rUS:8d}  {'ESCAPE!' if escape else ''}")
    if escape:
        print("  *** Directed perturbation found Gate-1+2 + rank(UΣ)>0! ***")
        break

print("\nDone.")
