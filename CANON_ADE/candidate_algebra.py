# coding: utf-8
"""
candidate_algebra.py
Extract and verify the unique ADE candidate algebra:
  C4 + C2(1st-index) + COMM + R_BLIND(3rd-index)  ->  d.o.f. = 1

Checks:
  1. Null vector extraction and normalization
  2. C3 fiber constraint: sum over r-fiber in 1st index is constant per fiber
  3. C5 ZD pairs: which of the 84 pairs have f[xi,eta,:]=0 ?
  4. Associativity pre-check: how far from associative?
  5. Save full f[i,j,k] tensor to numpy file for downstream use
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from axiom_bfs import (OCS, N_ORBITS, N, _triple_orbits, orbit_of,
                       build_canon, enc_comm, enc_rblind, INTERIOR,
                       C3_FiberConstraint)
from canon_constraints import IDX, C4_Symmetry, C5_SedenionZDGraph

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ------------------------------------------------------------------
# 1. Build system and extract unique null vector
# ------------------------------------------------------------------
print("Building C4 + C2(1st) + COMM + R_BLIND(3rd)...", flush=True)

def enc_rblind_third(cs):
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for i in range(N):
                for j in range(N):
                    cs.force_equal(i, j, rep, i, j, other)

cs = build_canon()
enc_comm(cs)
enc_rblind_third(cs)

A, b = cs.build()
print(f"  Constraint matrix: {A.shape}")

U, sv, Vt = np.linalg.svd(A, full_matrices=True)
tol = max(A.shape) * sv[0] * 1e-10
rank = int(np.sum(sv > tol))
null_dim = N_ORBITS - rank
print(f"  Rank={rank}, null dim={null_dim}  (expected 1)")

v = Vt[rank:].T[:, 0]   # unique null vector

# Normalize: set sum of all orbit-rep values weighted by orbit size = 1
weights = np.array([len(_triple_orbits[oid]) for oid in range(N_ORBITS)], dtype=float)
total = np.dot(weights, v)
if abs(total) < 1e-10:
    # fallback: normalize by max abs
    v = v / np.max(np.abs(v))
else:
    v = v / total

print(f"  Null vector: {np.sum(np.abs(v) > 1e-8)} nonzero / {N_ORBITS} orbits")
print(f"  Value range: [{v.min():.6f}, {v.max():.6f}]")

# ------------------------------------------------------------------
# 2. Expand to full f[i,j,k] tensor
# ------------------------------------------------------------------
print("\nExpanding to full 19x19x19 tensor...", flush=True)
f = np.zeros((N, N, N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            f[i, j, k] = v[orbit_of(i, j, k)]

nnz = np.sum(np.abs(f) > 1e-8)
print(f"  Nonzero entries: {nnz} / {N**3}")
print(f"  f range: [{f.min():.6f}, {f.max():.6f}]")

# Symmetry checks
comm_err = np.max(np.abs(f - f.transpose(1, 0, 2)))
print(f"  Commutativity f[i,j,k]=f[j,i,k]: max_err={comm_err:.2e}  {'OK' if comm_err < 1e-8 else 'FAIL'}")

# ------------------------------------------------------------------
# 3. C3 fiber constraint check
# ------------------------------------------------------------------
print("\n=== C3: Fiber sum constraint ===")
print("  For each (s,u) fiber, sum_{r in fiber} f[(r,s,u), j, k] should be constant over j,k")
c3_ok = True
for (s, u), indices in C3_FiberConstraint.FIBERS.items():
    # Sum over r-variants in first index
    fiber_sum = sum(f[idx, :, :] for idx in indices)  # shape (N, N)
    fmin, fmax = fiber_sum.min(), fiber_sum.max()
    spread = fmax - fmin
    # Also check: is the sum constant (same for all j,k)?
    constant = np.allclose(fiber_sum, fiber_sum[0, 0], atol=1e-8)
    status = "CONST" if constant else f"SPREAD={spread:.4f}"
    print(f"  Fiber (s={s},u={u}) size={len(indices)}: sum range=[{fmin:.4f},{fmax:.4f}]  {status}")
    if not constant:
        c3_ok = False
print(f"  C3 overall: {'SATISFIED' if c3_ok else 'NOT satisfied'}")

# C3 stronger: gamma(s,u) = 3, i.e. sum_r f[(r,s,u), j, k] = 3 for all j,k?
print("\n  C3 gamma=3 check (fiber sums should equal 3 after unnormalized scaling):")
for (s, u), indices in C3_FiberConstraint.FIBERS.items():
    fiber_sum = sum(f[idx, :, :] for idx in indices)
    mean_val = fiber_sum.mean()
    print(f"  Fiber (s={s},u={u}): mean fiber_sum = {mean_val:.6f}")

# ------------------------------------------------------------------
# 4. C5 ZD pair check
# ------------------------------------------------------------------
print("\n=== C5: Sedenion ZD pairs ===")
print("  Building sedenion ZD pairs (may take a moment)...", flush=True)
raw_pairs = C5_SedenionZDGraph.build_zd_pairs()
print(f"  Total sedenion ZD pairs found: {len(raw_pairs)}")

# These are pairs of sedenion weight-2 elements, indexed by sedenion indices 0..15.
# We need to map them to INTERIOR indices. The bridge: sedenion e_i for i in 1..7
# maps to interior point via bit pattern: i -> (bit2, bit1, bit0) of i.
# e.g. e_1=001->(0,0,1), e_7=111->(1,1,1). Sedenion indices 9..15 (upper sector)
# map to (2,x,x) range. We test which pairs land within INTERIOR.

def sed_idx_to_interior(ij_pair):
    """Try to map a sedenion weight-2 pair index tuple to INTERIOR coords."""
    results = []
    for idx in ij_pair:
        # Sedenion basis e_idx: map via bit pattern
        # INTERIOR points with at least one odd coordinate
        r = (idx >> 2) & 1
        s = (idx >> 1) & 1
        u = idx & 1
        pt = (r, s, u)
        if pt in IDX:
            results.append(IDX[pt])
        else:
            return None
    return tuple(results) if len(results) == 2 else None

satisfied = 0
violated = 0
unmapped = 0
violations = []

for (ij, kl) in raw_pairs:
    m1 = sed_idx_to_interior(ij)
    m2 = sed_idx_to_interior(kl)
    if m1 is None or m2 is None:
        unmapped += 1
        continue
    i, j_idx = m1
    k, l_idx = m2
    # Check f[i, k, :] = 0 (product of first elements of each pair)
    prod = f[i, k, :]
    norm = np.max(np.abs(prod))
    if norm < 1e-8:
        satisfied += 1
    else:
        violated += 1
        violations.append((INTERIOR[i], INTERIOR[k], norm))

print(f"  ZD pairs checked: {satisfied + violated}")
print(f"  Satisfied (product=0): {satisfied}")
print(f"  Violated (product≠0): {violated}")
if violations:
    print(f"  Top violations:")
    for xi, eta, norm in sorted(violations, key=lambda x: -x[2])[:10]:
        print(f"    {xi} * {eta}: max|f|={norm:.6f}")

# ------------------------------------------------------------------
# 5. Associativity pre-check: compute (e_i * e_j) * e_k - e_i * (e_j * e_k)
# ------------------------------------------------------------------
print("\n=== Associativity check ===")
# (f*f)[i,j,k,l] = sum_m f[i,j,m]*f[m,k,l]  vs  sum_m f[j,k,m]*f[i,m,l]
assoc_errors = []
# Sample over all i,j,k,l (expensive: N^4 = 130321, manageable)
max_assoc_err = 0.0
worst = None
for i in range(N):
    for j in range(N):
        lhs = np.einsum('m,mkl->kl', f[i, j, :], f)   # (e_i*e_j)*e_k = sum_m f[i,j,m]*f[m,k,l]
        # Actually: ((e_i * e_j) * e_k)_l = sum_m f[i,j,m] * f[m,k,l]
        # e_i * (e_j * e_k))_l = sum_m f[j,k,m] * f[i,m,l]  -- NO
        # e_i * (e_j * e_k) = e_i * (sum_m f[j,k,m] e_m) = sum_m f[j,k,m] * (e_i * e_m)
        # (e_i * e_m)_l = f[i,m,l]
        # So: (e_i * (e_j * e_k))_l = sum_m f[j,k,m] * f[i,m,l]
        rhs = np.einsum('jkm,iml->kl', f[np.newaxis, j:j+1, :].reshape(1,N,N), f)
        # Simpler:
        for k in range(N):
            lhs_k = f[i, j, :] @ f[:, k, :]   # shape (N,), component l of (e_i*e_j)*e_k
            rhs_k = f[j, k, :] @ f[i, :, :]   # shape (N,), component l of e_i*(e_j*e_k)
            err = np.max(np.abs(lhs_k - rhs_k))
            if err > max_assoc_err:
                max_assoc_err = err
                worst = (i, j, k)

print(f"  Max associativity error: {max_assoc_err:.6e}")
if worst:
    i, j, k = worst
    print(f"  Worst triple: e_{INTERIOR[i]} * e_{INTERIOR[j]} * e_{INTERIOR[k]}")
    lhs_k = f[i, j, :] @ f[:, k, :]
    rhs_k = f[j, k, :] @ f[i, :, :]
    print(f"  (e_i*e_j)*e_k = {lhs_k[:6]}...")
    print(f"  e_i*(e_j*e_k) = {rhs_k[:6]}...")

# ------------------------------------------------------------------
# 6. Save tensor
# ------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(__file__), "candidate_f_tensor.npy")
np.save(out_path, f)
print(f"\nSaved f[i,j,k] to {out_path}")

# Also save the INTERIOR index map
int_path = os.path.join(os.path.dirname(__file__), "interior_points.npy")
np.save(int_path, np.array(INTERIOR))
print(f"Saved INTERIOR points to {int_path}")

print("\nDone.")
