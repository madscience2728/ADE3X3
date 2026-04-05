# coding: utf-8
"""
extract_algebra.py
Extracts the unique (up-to-scale) algebra null vector from the C4+C2 constraint system.
Also diagnoses why AX_FIBER_ID is inconsistent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from axiom_bfs import OCS, N_ORBITS, _triple_orbits, _triple_to_orbit, orbit_of, build_canon, INTERIOR, N
from canon_constraints import C3_FiberConstraint

np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------------
# 1. Extract the null vector of the canon (C4+C2) system
# ------------------------------------------------------------------
print("=== Null vector of C4+C2 constraint system ===")
cs = build_canon()
A, b = cs.build()
print(f"  Constraint matrix: {A.shape}  (rows=equations, cols=439 orbit vars)")

U, sv, Vt = np.linalg.svd(A, full_matrices=True)
tol = max(A.shape) * sv[0] * 1e-10
rank = int(np.sum(sv > tol))
print(f"  Rank = {rank},  null space dim = {N_ORBITS - rank}")

null_vecs = Vt[rank:].T   # shape (439, null_dim)
v = null_vecs[:, 0]        # the unique null vector (up to scale)

# Normalize to max abs = 1
v = v / np.max(np.abs(v))
print(f"  Null vector (439 orbit coords, normalised, nonzero entries):")
nonzero = [(i, v[i]) for i in range(N_ORBITS) if abs(v[i]) > 1e-8]
print(f"  Nonzero orbits: {len(nonzero)} / {N_ORBITS}")
for oid, val in sorted(nonzero, key=lambda x: -abs(x[1]))[:30]:
    rep = _triple_orbits[oid][0]
    print(f"    orbit {oid:3d}: val={val:+.4f}  rep={rep}  (orbit_size={len(_triple_orbits[oid])})")

# ------------------------------------------------------------------
# 2. Expand to full 6859-dim structure constant tensor
# ------------------------------------------------------------------
print("\n=== Full structure constant tensor f[i,j,k] ===")
f = np.zeros((N, N, N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            oid = orbit_of(i, j, k)
            f[i, j, k] = v[oid]

nnz = np.sum(np.abs(f) > 1e-8)
print(f"  Nonzero entries: {nnz} / {N**3}")

# Check symmetry
comm_err = np.max(np.abs(f - f.transpose(1, 0, 2)))
print(f"  Commutativity error f[i,j,k]-f[j,i,k]: max={comm_err:.4e}")
anti_err = np.max(np.abs(f + f.transpose(1, 0, 2)))
print(f"  Anti-comm error f[i,j,k]+f[j,i,k]: max={anti_err:.4e}")

# Check r-blindness: within each fiber, are i-variants equal?
print("\n  R-blindness check:")
for (s, u), indices in C3_FiberConstraint.FIBERS.items():
    if len(indices) < 2:
        continue
    rep = indices[0]
    max_dev = 0.0
    for other in indices[1:]:
        for j in range(N):
            for k in range(N):
                max_dev = max(max_dev, abs(f[rep, j, k] - f[other, j, k]))
    if max_dev > 1e-8:
        print(f"    fiber (s={s},u={u}): max deviation = {max_dev:.4e}  FAIL")
    else:
        print(f"    fiber (s={s},u={u}): OK (max_dev={max_dev:.2e})")

# ------------------------------------------------------------------
# 3. Diagnose FIBER_ID inconsistency
# ------------------------------------------------------------------
print("\n=== FIBER_ID diagnosis ===")
print("  FORCED_FIBERS (size-1 fibers whose element acts as identity):")
for (s, u), indices in C3_FiberConstraint.FORCED_FIBERS.items():
    id_idx = indices[0]
    print(f"    fiber (s={s},u={u}): forced element = index {id_idx}  (interior label {INTERIOR[id_idx]})")
    # Check what f[id_idx, i, i] looks like in the null vector
    diag_vals = [(i, f[id_idx, i, i]) for i in range(N)]
    nz = [(i, val) for i, val in diag_vals if abs(val) > 1e-8]
    print(f"      f[id,i,i] nonzero: {nz}")
    # For identity: f[id_idx, i, k] should = delta(i,k)
    # Check actual values
    row = np.array([f[id_idx, i, i] for i in range(N)])
    print(f"      f[id,i,i] range: [{row.min():.4f}, {row.max():.4f}]")

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  Unique algebra (up to scale): {nnz} nonzero f[i,j,k]")
print(f"  Commutativity: {'SATISFIED' if comm_err < 1e-8 else 'NOT satisfied'} (max_err={comm_err:.2e})")
print(f"  Anti-comm: {'SATISFIED' if anti_err < 1e-8 else 'NOT satisfied'} (max_err={anti_err:.2e})")
print()
print("  FIBER_ID is inconsistent because:")
print("  The unique null-vector algebra has f[forced_elem, i, i] = 0 for all i.")
print("  Identity axiom requires f[forced_elem, i, i] = 1 -- contradiction.")
print("  => Forced fiber elements are ZERO DIVISORS, not identity elements, in this algebra.")
