# coding: utf-8
"""
comm_basis.py
Extract the 3-dimensional basis of the {COMM} solution space (C4+C2+commutativity).
Then test which additional constraints (C3 fiber, C5 ZD, AX_R_BLIND) pin a unique point.

The 3 basis vectors should correspond structurally to the 3 G-orbits on INTERIOR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from axiom_bfs import (OCS, N_ORBITS, N, _triple_orbits, orbit_of,
                       build_canon, enc_comm, enc_rblind, enc_hullzd,
                       INTERIOR, C3_FiberConstraint)
from canon_constraints import IDX, C4_Symmetry

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ------------------------------------------------------------------
# 1. Build C4+C2+COMM system and extract null basis
# ------------------------------------------------------------------
print("Building C4 + C2(first-index) + COMM system...", flush=True)
cs = build_canon()
enc_comm(cs)
A, b = cs.build()
print(f"  Constraint matrix: {A.shape}")

U, sv, Vt = np.linalg.svd(A, full_matrices=True)
tol = max(A.shape) * sv[0] * 1e-10
rank = int(np.sum(sv > tol))
null_dim = N_ORBITS - rank
print(f"  Rank={rank}, null dim={null_dim}  (expected 3)")

null_basis = Vt[rank:].T  # shape (439, null_dim)
print(f"  Null basis shape: {null_basis.shape}")

# ------------------------------------------------------------------
# 2. Characterize each basis vector
# ------------------------------------------------------------------
print("\n=== Null basis vectors ===")

# G-orbit sizes on INTERIOR (3 orbits, sizes 3, 7, 9)
from canon_constraints import C4_Symmetry
pair_orbits = []
seen = set()
for i in range(N):
    for j in range(N):
        oid = orbit_of(i, j, 0)  # use k=0 as probe — not right
        pass

# Better: characterize by which INTERIOR points appear as reps
# The 3 INTERIOR orbits: let's get them from C4_Symmetry
int_orbit_sets = C4_Symmetry.interior_orbits()
int_orbits = [[IDX[pt] for pt in orb] for orb in int_orbit_sets]

print(f"\n  INTERIOR G-orbits ({len(int_orbits)} orbits):")
for oi, orb in enumerate(int_orbits):
    pts = [INTERIOR[i] for i in orb]
    print(f"    Orbit {oi} (size {len(orb)}): {pts[:4]}{'...' if len(pts)>4 else ''}")

print()
for vi in range(null_dim):
    v = null_basis[:, vi]
    # Nonzero pattern
    nz = np.sum(np.abs(v) > 1e-8)
    vmax = np.max(np.abs(v))
    # Which orbit indices are nonzero
    nz_orbits = [oid for oid in range(N_ORBITS) if abs(v[oid]) > 1e-8]
    # Check: is this vector supported only on triples (i,j,k) where i,j,k are in specific INTERIOR orbits?
    print(f"  v{vi}: {nz}/{N_ORBITS} nonzero orbits, max={vmax:.4f}")
    # Sample a few rep triples
    for oid in nz_orbits[:5]:
        rep = _triple_orbits[oid][0]
        print(f"    orbit {oid:3d}: val={v[oid]:+.4f}  rep={rep}  orb_size={len(_triple_orbits[oid])}")

# ------------------------------------------------------------------
# 3. Expand each basis vector to full f tensor and test properties
# ------------------------------------------------------------------
print("\n=== Properties of each basis algebra ===")

def expand(v):
    f = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f[i, j, k] = v[orbit_of(i, j, k)]
    return f

for vi in range(null_dim):
    v = null_basis[:, vi]
    f = expand(v)
    comm_err = np.max(np.abs(f - f.transpose(1, 0, 2)))
    # Fiber constraint: sum over r-variants in first index should be proportional
    fiber_sums = []
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        for j in range(N):
            for k in range(N):
                s_val = sum(f[idx, j, k] for idx in indices)
                fiber_sums.append(s_val)
    fs = np.array(fiber_sums)
    # r-blindness of OUTPUT (k-index): f[i,j,k1] vs f[i,j,k2] same fiber
    rblind_k_err = 0.0
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for i in range(N):
                for j in range(N):
                    rblind_k_err = max(rblind_k_err, abs(f[i, j, rep] - f[i, j, other]))
    print(f"  v{vi}: comm_err={comm_err:.2e}  fiber_sum range=[{fs.min():.4f},{fs.max():.4f}]  rblind_k_err={rblind_k_err:.2e}")

# ------------------------------------------------------------------
# 4. What pins a unique point? Test C3, C5, R_BLIND(3rd idx) each alone
# ------------------------------------------------------------------
print("\n=== Constraints that select within the 3-dim family ===")

def dof_with(extra_enc):
    cs2 = build_canon()
    enc_comm(cs2)
    extra_enc(cs2)
    return cs2.dof()

# C3 fiber constraint: sum over r-fiber in first index = constant
def enc_fiber_sum(cs):
    """C3: for each (s,u) fiber, sum of f[(r,s,u), j, k] over r is the same for all r-equivs."""
    # More precisely: each fiber contributes equally, so total gamma = n_fiber * (each term)
    # Encode: for each forced fiber (size 1), the term equals 1 (normalized)
    # For free fibers (size 3): all r-variants equal, which enc_rblind_first already handles  
    # The remaining C3 constraint: gamma(s,u)=3 for all fibers
    # Encode as: sum_{r in fiber} f[(r,s,u), j, k] = 3 * f[rep, j, k] (automatically from r_blind_first)
    # What's NOT yet encoded: the *value* of gamma. Add: gamma = 3 normalization
    # This pins the scale within each fiber.
    for (s, u), indices in C3_FiberConstraint.FORCED_FIBERS.items():
        idx = indices[0]
        # Force sum = 1 (normalized): f[idx, j, k] = 1/3 for all j,k? No — that's too strong.
        # Actually C3 says sum_r gamma_r = 3, which with r-blindness means each gamma_r = 1.
        # gamma here is not f directly — let's just test the r-blind 3rd index
        pass

def enc_rblind_third(cs):
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for i in range(N):
                for j in range(N):
                    cs.force_equal(i, j, rep, i, j, other)

def enc_rblind_second(cs):
    for (s, u), indices in C3_FiberConstraint.FIBERS.items():
        if len(indices) < 2:
            continue
        rep = indices[0]
        for other in indices[1:]:
            for i in range(N):
                for k in range(N):
                    cs.force_equal(i, rep, k, i, other, k)

tests = [
    ("HULL_ZD",        enc_hullzd),
    ("R_BLIND(2nd idx)", enc_rblind_second),
    ("R_BLIND(3rd idx)", enc_rblind_third),
    ("R_BLIND(all 3)",   enc_rblind),
]

for name, enc in tests:
    d = dof_with(enc)
    print(f"  COMM + {name}: d.o.f. = {d}")

# ------------------------------------------------------------------
# 5. Combination: which pairs collapse to 0 or 1?
# ------------------------------------------------------------------
print("\n=== Pairs that collapse COMM's 3-dim family ===")
from itertools import combinations as _comb

enc_map = dict(tests)
for (n1, e1), (n2, e2) in _comb([(n, e) for n, e in tests], 2):
    def enc_both(cs, _e1=e1, _e2=e2):
        _e1(cs); _e2(cs)
    d = dof_with(enc_both)
    print(f"  COMM + {n1} + {n2}: d.o.f. = {d}")

print("\nDone.")
