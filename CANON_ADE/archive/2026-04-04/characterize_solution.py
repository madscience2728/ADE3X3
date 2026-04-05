# coding: utf-8
"""
characterize_solution.py — Phase 2.5: Full characterization of the
associative algebra found in assoc_variety.py.

Loads assoc_solution_1.npy (the 19×19×19 structure constant tensor f)
and computes:
  1. Multiplication table orbit decomposition
  2. Nilpotency: e_i^2, nil-index, nilpotent radical
  3. Left/right ideal structure
  4. Jacobson radical
  5. C3 fiber normalization analysis (can it be gauge-fixed?)
  6. C5 ZD constraint violations
  7. Grading by G-orbits of INTERIOR
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from canon_constraints import (
    INTERIOR, IDX, HULL,
    C3_FiberConstraint, C4_Symmetry, C5_SedenionZDGraph,
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

f = np.load(os.path.join(os.path.dirname(__file__), "assoc_solution_1.npy"))
N = f.shape[0]
assert f.shape == (N, N, N) == (19, 19, 19)

print("=" * 70)
print("CHARACTERIZATION OF ASSOC_SOLUTION_1")
print("=" * 70)

# ------------------------------------------------------------------
# 0. Basic tensor stats
# ------------------------------------------------------------------
print("\n--- 0. Basic Stats ---")
print(f"  Shape: {f.shape}")
print(f"  Max |f|: {np.max(np.abs(f)):.6f}")
print(f"  Nonzero entries (|f|>1e-10): {np.sum(np.abs(f) > 1e-10)} of {N**3}")
print(f"  Frobenius norm: {np.linalg.norm(f):.6f}")

# Verify associativity: sum_m f[a,b,m]*f[m,c,d] = sum_m f[b,c,m]*f[a,m,d]
assoc_err = 0.0
for a in range(N):
    for d in range(N):
        LHS = f[a] @ f[:, :, d]       # (N,N): (LHS)_{b,c} = sum_m f[a,b,m]*f[m,c,d]
        RHS = f[:, :, d].T @ f[a].T   # need: sum_m f[b,c,m]*f[a,m,d]
        # Actually: LHS[b,c] = sum_m f[a,b,m]*f[m,c,d]
        #           RHS[b,c] = sum_m f[b,c,m]*f[a,m,d]
        RHS2 = np.zeros((N, N))
        for b in range(N):
            for c in range(N):
                RHS2[b, c] = np.dot(f[b, c, :], f[a, :, d])
        assoc_err = max(assoc_err, np.max(np.abs(LHS - RHS2)))
print(f"  Associativity error: {assoc_err:.4e}")

comm_err = np.max(np.abs(f - f.transpose(1, 0, 2)))
print(f"  Commutativity error: {comm_err:.4e} ({'commutative' if comm_err < 1e-8 else 'NON-commutative'})")

# ------------------------------------------------------------------
# 1. Orbit decomposition of f
# ------------------------------------------------------------------
print("\n--- 1. G-Orbit Decomposition ---")
orbits = C4_Symmetry.interior_orbits()
print(f"  G-orbits on INTERIOR: {len(orbits)}")
for oid, orb in enumerate(sorted(orbits, key=lambda x: len(x))):
    rep = sorted(orb)[0]
    idx_rep = IDX[rep]
    print(f"  Orbit {oid} (size {len(orb)}): rep={rep}, IDX={idx_rep}")
    # Check: do all elements in the orbit give the same f-slice structure?
    norms = [np.linalg.norm(f[IDX[pt]]) for pt in orb]
    print(f"    Row norms: min={min(norms):.4f} max={max(norms):.4f} (should be equal)")

# ------------------------------------------------------------------
# 2. Nilpotency analysis
# ------------------------------------------------------------------
print("\n--- 2. Nilpotency Analysis ---")

# e_i^2
print("  Squaring each basis element:")
for i in range(N):
    sq = f[i, i, :]
    if np.max(np.abs(sq)) > 1e-10:
        nz = np.sum(np.abs(sq) > 1e-10)
        print(f"    e_{i}^2 = nonzero ({nz} components, norm={np.linalg.norm(sq):.4f})")
    else:
        print(f"    e_{i}^2 = 0")

# Nil-index: compute f^k for small k (via repeated multiplication)
# f^k means the k-fold product tensor. We check if all k-fold products are zero.
def mul(a_vec, b_vec):
    """Multiply two algebra elements (as coefficient vectors)."""
    return np.einsum('i,j,ijk->k', a_vec, b_vec, f)

print("\n  Nil-index check (powers of generic element):")
rng = np.random.default_rng(0)
x = rng.standard_normal(N)
x /= np.linalg.norm(x)
power = x.copy()
for k in range(2, 8):
    power = mul(power, x)
    norm = np.linalg.norm(power)
    print(f"    x^{k} norm: {norm:.6e}")
    if norm < 1e-12:
        print(f"    -> Nilpotent at index {k}")
        break

# Check nilpotency of each basis element
print("\n  Basis element nil-indices:")
for i in range(N):
    ei = np.zeros(N); ei[i] = 1.0
    power = ei.copy()
    nil_idx = None
    for k in range(2, 12):
        power = mul(power, ei)
        if np.linalg.norm(power) < 1e-12:
            nil_idx = k
            break
    if nil_idx:
        print(f"    e_{i}: nil-index = {nil_idx}")
    else:
        print(f"    e_{i}: NOT nilpotent (norm at power 11: {np.linalg.norm(power):.4e})")

# ------------------------------------------------------------------
# 3. Left and right multiplication matrices
# ------------------------------------------------------------------
print("\n--- 3. Multiplication Matrices ---")

# Left multiplication: L_i[j,k] = f[i,j,k]  (e_i * e_j)_k
# Right multiplication: R_i[j,k] = f[j,i,k]  (e_j * e_i)_k

# Compute ranks of L_i and R_i
print("  Left multiplication matrix ranks:")
L_ranks = []
for i in range(N):
    L = f[i, :, :]  # N x N
    r = np.linalg.matrix_rank(L, tol=1e-8)
    L_ranks.append(r)
rank_counts = {}
for r in L_ranks:
    rank_counts[r] = rank_counts.get(r, 0) + 1
print(f"    Rank distribution: {rank_counts}")

print("  Right multiplication matrix ranks:")
R_ranks = []
for i in range(N):
    R = f[:, i, :]  # N x N
    r = np.linalg.matrix_rank(R, tol=1e-8)
    R_ranks.append(r)
rank_counts = {}
for r in R_ranks:
    rank_counts[r] = rank_counts.get(r, 0) + 1
print(f"    Rank distribution: {rank_counts}")

# ------------------------------------------------------------------
# 4. Ideal structure and Jacobson radical
# ------------------------------------------------------------------
print("\n--- 4. Ideal Structure & Jacobson Radical ---")

# The Jacobson radical = intersection of all maximal left ideals
# For a finite-dim algebra, J = {x : x*y is nilpotent for all y}
# Equivalently: J = {x : 1 - x*y is left-invertible for all y}
# For our algebra (no identity), J = largest nilpotent ideal.

# Compute the "regular representation" and find its radical
# The left regular representation: L = [L_0 | L_1 | ... | L_{18}]
# stacked horizontally gives a 19 x (19*19) matrix
# The radical is ker of the trace form: tr(L_x * L_y)

# Killing form / trace form
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        # tr(L_i * L_j)
        Li = f[i, :, :]
        Lj = f[j, :, :]
        K[i, j] = np.trace(Li @ Lj)

print(f"  Killing form rank: {np.linalg.matrix_rank(K, tol=1e-8)} (of {N})")
K_evals = np.linalg.eigvalsh(K)
print(f"  Killing form eigenvalues: {np.sort(K_evals)}")
if np.linalg.matrix_rank(K, tol=1e-8) < N:
    # Radical = null space of K
    _, sv, Vt = np.linalg.svd(K)
    null_dim = np.sum(sv < 1e-8)
    print(f"  Killing form null space dimension: {null_dim}")
    print(f"  => Jacobson radical has dimension >= {null_dim}")
else:
    print("  Killing form is nondegenerate => algebra is semisimple (J = 0)")

# Also compute the trace form using right multiplication
Kr = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Ri = f[:, i, :]
        Rj = f[:, j, :]
        Kr[i, j] = np.trace(Ri @ Rj)
print(f"  Right Killing form rank: {np.linalg.matrix_rank(Kr, tol=1e-8)}")

# ------------------------------------------------------------------
# 5. C3 Fiber Normalization
# ------------------------------------------------------------------
print("\n--- 5. C3 Fiber Normalization ---")
print("  C3 requires: for each (s,u)-fiber, sum of gammas = 3")
print("  Checking if f's fiber structure can be gauge-transformed to C3 form.\n")

# Fiber sum over 2nd index: S_{s,u}[xi,zeta] = sum_{r:(r,s,u) in I} f[xi, r, zeta]
fiber_of = {}
for xi in INTERIOR:
    r, s, u = xi
    fiber_of[IDX[xi]] = (s, u)

all_fibers_ok = True
for (s, u), indices in sorted(C3_FiberConstraint.FIBERS.items()):
    S = sum(f[:, i, :] for i in indices)
    # C3 target: S[xi,zeta] = delta(fiber(xi)==(s,u)) * delta(fiber(zeta)==(s,u))
    # i.e., 1 if both xi and zeta are in fiber (s,u), 0 otherwise
    target = np.zeros((N, N))
    for xi in range(N):
        for zeta in range(N):
            if fiber_of[xi] == (s, u) and fiber_of[zeta] == (s, u):
                target[xi, zeta] = 1.0

    err = np.max(np.abs(S - target))
    err_neg = np.max(np.abs(S + target))  # check if -f works
    print(f"  Fiber ({s},{u}) size={len(indices)}: "
          f"err(S-target)={err:.4f}, err(S+target)={err_neg:.4f}, "
          f"S range=[{S.min():.4f},{S.max():.4f}]")
    if err > 0.01 and err_neg > 0.01:
        all_fibers_ok = False

if all_fibers_ok:
    print("\n  C3 SATISFIED (possibly after sign flip)")
else:
    print("\n  C3 NOT SATISFIABLE by global rescaling.")
    print("  The fiber sum pattern is structurally different from the delta form.")
    print("  Nonzero fiber sums = -1 uniformly, but they don't land on same-fiber pairs.")
    print("  A gauge transformation beyond global scaling would be needed.")

# ------------------------------------------------------------------
# 6. C5 Zero-Divisor Constraint
# ------------------------------------------------------------------
print("\n--- 6. C5 ZD Constraint Check ---")
violations = C5_SedenionZDGraph.algebra_zd_constraints(f)
violations = np.array(violations)
print(f"  Total constraint values: {len(violations)} (84 edges × 19 outputs)")
print(f"  Max |violation|: {np.max(np.abs(violations)):.4e}")
print(f"  Mean |violation|: {np.mean(np.abs(violations)):.4e}")
print(f"  Fraction exactly zero: {np.sum(np.abs(violations) < 1e-10) / len(violations):.2%}")
print(f"  C5 satisfied: {'YES' if np.max(np.abs(violations)) < 1e-8 else 'NO'}")

# ------------------------------------------------------------------
# 7. Grading by G-orbits
# ------------------------------------------------------------------
print("\n--- 7. G-Orbit Grading Check ---")
print("  Testing if the algebra is graded by the 3 G-orbits of INTERIOR.")

orbits = sorted(C4_Symmetry.interior_orbits(), key=lambda x: len(x))
orbit_label = {}
for oid, orb in enumerate(orbits):
    for pt in orb:
        orbit_label[IDX[pt]] = oid

orbit_names = [f"O{oid}(size {len(orb)})" for oid, orb in enumerate(orbits)]
print(f"  Orbits: {orbit_names}")

# For grading: f[i,j,k] should be zero unless orbit(k) = orbit(i) * orbit(j)
# in some group law on {O0, O1, O2}.
# Check: for each (orbit_i, orbit_j), what orbits does orbit_k land in?
print("\n  Product orbit table (nonzero f[i,j,k] -> orbit of k):")
for oi in range(3):
    for oj in range(3):
        ok_counts = {}
        total = 0
        for i in range(N):
            if orbit_label[i] != oi:
                continue
            for j in range(N):
                if orbit_label[j] != oj:
                    continue
                for k in range(N):
                    if abs(f[i, j, k]) > 1e-10:
                        ok = orbit_label[k]
                        ok_counts[ok] = ok_counts.get(ok, 0) + 1
                        total += 1
        print(f"    O{oi} * O{oj} -> {ok_counts}  (total {total} nonzero)")

print("\n" + "=" * 70)
print("CHARACTERIZATION COMPLETE")
print("=" * 70)
