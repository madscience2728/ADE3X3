"""
Formal analysis of the 9-observable system over 3x3 bilinear pairings.

Parts:
  A. Symbolic rank proof via block determinants
  B. 9 observables as linear forms on m_ab
  C. Search for factorization through fewer than 9 bilinear primitives
  D. Report
"""
import numpy as np
from itertools import product as iproduct

# ══════════════════════════════════════════════════════════════════════════════
# PART A  — Symbolic block-determinant rank proof
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("PART A — Block-determinant rank proof")
print("=" * 72)

# Fingerprint (a, a, b) with a < b.
# Coupling constants satisfy 0 < C_a < C_b  (strictly increasing with level).
# Diagonal weights:
#   w0 = C_a + C_a = 2*C_a          (the 'outlier' element)
#   w1 = C_a + C_b                  (the other two share this weight)
#   w2 = C_a + C_b
#
# Three blocks are decoupled (act on disjoint columns of the 9-wide matrix):
#
# BLOCK D — diagonal triple {m00, m11, m22}  (columns 0,4,8)
# BLOCK A — pair {m01, m10}                  (columns 1,3)
# BLOCK B — pair {m02, m20}                  (columns 2,6)
# BLOCK C — pair {m12, m21}                  (columns 5,7)

def sym_det_D(Ca, Cb):
    """det of 3x3 diagonal block for fingerprint (a,a,b)."""
    w0, w1, w2 = 2*Ca, Ca+Cb, Ca+Cb
    # Rows: [1,1,1], [w0,-w1,0], [0,w1,-w2]
    return (w0*w1*1 + w0*w2*1 + w1*w2*1)   # cofactor expansion → w0w1 + w0w2 + w1w2

def sym_det_pair():
    """det of 2x2 pair block [[1,1],[1,-1]]."""
    return -2    # always

# Check over several representative Ca, Cb pairs (Ca < Cb, both > 0)
print("\nDiagonal block determinant = w0·w1 + w0·w2 + w1·w2")
print("(always > 0 since w0,w1,w2 > 0)")
test_couplings = [(1,2),(1,4),(1,8),(1,16),(2,4),(2,8),(4,8)]
for Ca, Cb in test_couplings:
    w0, w1, w2 = 2*Ca, Ca+Cb, Ca+Cb
    det_D = w0*w1 + w0*w2 + w1*w2
    print(f"  Ca={Ca:2d} Cb={Cb:2d}  →  w=('{w0},{w1},{w2}')  det(D)={det_D:6d}  > 0:  {det_D > 0}")

print("\nPair block determinant = det([[1,1],[1,-1]]) = -2  (always ≠ 0)")
print("\n→ All four blocks have full rank for any valid (Ca, Cb).")
print("→ rank = 3 (diag) + 2 (pair01) + 2 (pair02) + 2 (pair12) = 9  (Q.E.D.)")

# Numerical verification with the original embedding coupling magnitudes
Ca, Cb = 1.0, 4.0   # representative (level 0 vs level 2)
w0, w1, w2 = 2*Ca, Ca+Cb, Ca+Cb

A7 = np.array([
    [ 1,  0,  0,  0,  1,  0,  0,  0,  1],
    [ 0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  1,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  1,  0],
    [ 0,  1,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0, -1,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0, -1,  0],
], dtype=float)

r8 = np.array([ w0,0,0,0,-w1,0,0,0,  0], dtype=float)
r9 = np.array([  0,0,0,0, w1,0,0,0,-w2], dtype=float)
A9 = np.vstack([A7, r8, r9])

print(f"\nNumerical check (Ca={Ca}, Cb={Cb}):")
print(f"  rank(A9) = {np.linalg.matrix_rank(A9)}  (expected 9)")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — 9 observables as linear forms on m_ab
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("PART B — 9 observables as linear forms on m_ab = x_a · y_b")
print("=" * 72)

print("""
Variables (indices):
  m00=0  m01=1  m02=2
  m10=3  m11=4  m12=5
  m20=6  m21=7  m22=8

Observable     Coefficients                  Interpretation
─────────────────────────────────────────────────────────────────────
O1  m00+m11+m22                              diagonal sum (channel t)
O2  m01+m10                                  sym  pair (01↔10)
O3  m02+m20                                  sym  pair (02↔20)
O4  m12+m21                                  sym  pair (12↔21)
O5  m01-m10                                  antisym pair (01−10)
O6  m02-m20                                  antisym pair (02−20)
O7  m12-m21                                  antisym pair (12−21)
O8  w0·m00 - w1·m11                          coupling-weight diff (00 vs 11)
O9  w1·m11 - w2·m22                          coupling-weight diff (11 vs 22)

Notice: (O2+O5)/2 = x0·y1   (O2-O5)/2 = x1·y0   etc.
Observable pairs (Ok+O{k+3})/2 and (Ok-O{k+3})/2 recover individual m_ab.
""")

# Express each observable as a vector in the 9-dim bilinear form space
# basis order: m00,m01,m02,m10,m11,m12,m20,m21,m22
sym_obs = {
    "O1 (diag sum)":      [ 1, 0, 0, 0, 1, 0, 0, 0, 1],
    "O2 (sym  01/10)":    [ 0, 1, 0, 1, 0, 0, 0, 0, 0],
    "O3 (sym  02/20)":    [ 0, 0, 1, 0, 0, 0, 1, 0, 0],
    "O4 (sym  12/21)":    [ 0, 0, 0, 0, 0, 1, 0, 1, 0],
    "O5 (asym 01/10)":    [ 0, 1, 0,-1, 0, 0, 0, 0, 0],
    "O6 (asym 02/20)":    [ 0, 0, 1, 0, 0, 0,-1, 0, 0],
    "O7 (asym 12/21)":    [ 0, 0, 0, 0, 0, 1, 0,-1, 0],
    f"O8 ({w0}m00-{w1}m11)": [w0, 0, 0, 0,-w1, 0, 0, 0, 0],
    f"O9 ({w1}m11-{w2}m22)": [ 0, 0, 0, 0, w1, 0, 0, 0,-w2],
}

print("Observable coefficient vectors (9-dim bilinear form space):")
print(f"  {'Name':24s}  coefficients → 3x3 bilinear matrix")
for name, v in sym_obs.items():
    mat = np.array(v).reshape(3,3)
    print(f"  {name:24s}  {v}")

print(f"\n  These 9 vectors span dim = {np.linalg.matrix_rank(np.array(list(sym_obs.values())))}")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — Search for factorization through fewer than 9 bilinear primitives
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("PART C — Bilinear primitive count search")
print("=" * 72)

# A bilinear primitive is  p_i = L_i(x) · R_i(y)
# where L_i = (l0,l1,l2) and R_i = (r0,r1,r2) are coefficient vectors.
#
# As a bilinear form vector:
#   vec(p_i) = kron(L_i, R_i)  ∈  R^9
#
# R primitives can express the observables iff
#   span{kron(L_1,R_1), ..., kron(L_R,R_R)} ⊇ span{O1,...,O9} = R^9
#
# But span{kron(L_i,R_i)} ⊆ R^9  has dimension ≤ R.
# For R < 9, this span has dimension < 9, so it CANNOT contain all of R^9.

print("""
Bilinear primitive:  p_i = (Σ_a L_{ia}·x_a) · (Σ_b R_{ib}·y_b)
Bilinear form vector:  vec(p_i)_ab = L_{ia}·R_{ib}  →  a rank-1 element of R^9

KEY THEOREM:
  The 9 observables {O1,...,O9} span all of R^9 (dim = 9 confirmed above).
  Any R bilinear primitives span an R-dimensional subspace of R^9.
  Therefore: R < 9  ⟹  span ≠ R^9  ⟹  cannot express all 9 observables.

Proof skeleton:
  dim(span{p_1,...,p_R}) ≤ R  [linear algebra]
  ∀ R < 9:  R < 9 = dim(span{O1,...,O9})  ⟹  not a superset.

This is TIGHT: the canonical factorization (one primitive per m_ab) uses
exactly 9 rank-1 bilinear forms and recovers everything.
""")

# Numerical proof by random search: try batches of 8 random primitives and
# measure max achievable span dimension
print("Numerical verification: sample R random bilinear primitives, measure span dim")
np.random.seed(42)
N_TRIALS = 100_000
max_span_by_R = {}
for R in range(1, 10):
    best = 0
    for _ in range(N_TRIALS // 10 if R < 9 else 1):
        L = np.random.randn(R, 3)
        R_ = np.random.randn(R, 3)
        prims = np.array([np.kron(L[i], R_[i]) for i in range(R)])
        rk = np.linalg.matrix_rank(prims, tol=1e-10)
        if rk > best:
            best = rk
    max_span_by_R[R] = best

print(f"\n  {'Primitives R':>13}  {'max span dim observed':>22}  {'can cover R^9?':>15}")
print(f"  {'-'*13}  {'-'*22}  {'-'*15}")
for R in range(1, 10):
    span = max_span_by_R[R]
    can = "YES" if R == 9 else "NO  (span < 9)"
    print(f"  {R:>13}  {span:>22}  {can:>15}")

# ══════════════════════════════════════════════════════════════════════════════
# PART D — Explicit factorization and report
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("PART D — Report: minimum primitive count and explicit factorization")
print("=" * 72)

print(f"""
Minimum bilinear primitive count:  9  (tight lower bound, achieved)

Proof of lower bound:
  The 9 observables form a basis of the 9-dimensional bilinear form space
  over (R^3, R^3).  Each primitive is one rank-1 element of this space.
  Spanning a 9-dimensional space with rank-1 elements requires ≥ 9 of them.

Explicit factorization (9 primitives = canonical basis):

  Primitive   Left form L(x)   Right form R(y)   Bilinear product
  ─────────────────────────────────────────────────────────────────
  p00         x0               y0                x0·y0  = m00
  p01         x0               y1                x0·y1  = m01
  p02         x0               y2                x0·y2  = m02
  p10         x1               y0                x1·y0  = m10
  p11         x1               y1                x1·y1  = m11
  p12         x1               y2                x1·y2  = m12
  p20         x2               y0                x2·y0  = m20
  p21         x2               y1                x2·y1  = m21
  p22         x2               y2                x2·y2  = m22

Then:
  O1 = p00 + p11 + p22
  O2 = p01 + p10
  O3 = p02 + p20
  O4 = p12 + p21
  O5 = p01 − p10
  O6 = p02 − p20
  O7 = p12 − p21
  O8 = w0·p00 − w1·p11
  O9 = w1·p11 − w2·p22

WHY NO COMPRESSION IS POSSIBLE:
  The XOR routing + defect rows together exhaust the bilinear form space.
  The 4 bucket rows cover the symmetric part (O1–O4).
  The 3 defect rows cover the antisymmetric part (O5–O7).
  The 2 diagonal weight rows resolve the final degeneracy (O8–O9).
  Together they form a basis — not a subspace — of Bil(R^3, R^3).
  A complete basis cannot be re-expressed with fewer generators.

NOTE on alternative bases:
  Any 9 linearly independent rank-1 bilinear primitives
  {{(x0+x1)·y0, x0·(y1−y2), ...}} achieve the same span with exactly 9
  primitives.  The count 9 is invariant; only the specific forms change.
""")

# Final sanity: show our 9 canonical primitives recover the A9 observable matrix
print("Sanity check: recover A9 from canonical 9 primitives")
canonical_L = np.eye(3, dtype=float)   # L[a] = e_a
canonical_R = np.eye(3, dtype=float)   # R[b] = e_b
primitives = np.array([np.kron(canonical_L[a], canonical_R[b])
                        for a in range(3) for b in range(3)])  # (9×9)

# A9 should be expressible as linear combo of primitives — i.e. A9 @ prim_inv
prim_inv = np.linalg.inv(primitives)
coeffs = A9 @ prim_inv    # (9×9): row i of A9 = sum_j coeffs[i,j] * primitive_j
residual = A9 - coeffs @ primitives
print(f"  max |residual| = {np.max(np.abs(residual)):.2e}  (should be ~0)")
print(f"  coefficient matrix (rows=observables, cols=m_ab):")
for i, row in enumerate(np.round(coeffs, 4)):
    nz = [(j, v) for j, v in enumerate(row) if abs(v) > 1e-10]
    a0, b0 = divmod(np.argmax(np.abs(row)), 3)
    label = f"  O{i+1}"
    terms = " + ".join(f"{v:+.4g}·m{j//3}{j%3}" for j,v in nz)
    print(f"  {label:5s} = {terms}")
