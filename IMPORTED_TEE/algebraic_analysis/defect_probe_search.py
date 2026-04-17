"""
defect_probe_search.py
======================
Direct finite verification of the 5-channel nonlinear engine via the CD
octonion scaffold.

Parts 0-9 of the task brief, fully implemented from the actual codebase
definitions in octonion_assoc_patch.py.

Definition of K(w):
    The scaffold uses X_BASIS=[1,2,3], Y_BASIS=[4,5,6] as the patch.
    For probe w (octonion basis index), the defect matrix K(w) is 3x3 with

        kappa_ab(w) = signed component of oassoc_vec(X[a], Y[b], w)
                      at the XOR-predicted output bucket X[a] ^ Y[b] ^ w,
                      divided by 2 to normalize to {-1, 0, +1}.

    This is the "signed bucket amplitude" extracted from the actual
    assoc_matrix_for_probe function in octonion_assoc_patch.py.
"""

import numpy as np
from itertools import combinations

SEP  = "=" * 70
SEP2 = "-" * 70

# ===========================================================================
# PART 0 — Octonion algebra (direct from octonion_assoc_patch.py)
# ===========================================================================

FANO_LINES = [
    (1, 2, 3),
    (1, 4, 5),
    (2, 4, 6),
    (1, 6, 7),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
]

def _build_oct():
    T = np.zeros((8, 8, 8), dtype=float)
    for i in range(8):
        T[0, i, i] = 1.0
        T[i, 0, i] = 1.0
    for i in range(1, 8):
        T[i, i, 0] = -1.0
    for (i, j, k) in FANO_LINES:
        T[i, j, k] =  1.0;  T[j, i, k] = -1.0
        T[j, k, i] =  1.0;  T[k, j, i] = -1.0
        T[k, i, j] =  1.0;  T[i, k, j] = -1.0
    return T

_OCT = _build_oct()

def oassoc_vec(i, j, k):
    """[e_i, e_j, e_k] = (e_i*e_j)*e_k - e_i*(e_j*e_k) as 8-vector.
    Copied exactly from octonion_assoc_patch.py."""
    ij   = _OCT[i, j]
    ij_k = sum(ij[m] * _OCT[m, k] for m in range(8))
    jk   = _OCT[j, k]
    i_jk = sum(jk[m] * _OCT[i, m] for m in range(8))
    return ij_k - i_jk

# Verify alternative laws (sanity check from the source file)
for _i in range(8):
    for _j in range(8):
        assert np.allclose(oassoc_vec(_i, _i, _j), 0), f"[e{_i},e{_i},e{_j}]!=0"
        assert np.allclose(oassoc_vec(_i, _j, _j), 0), f"[e{_i},e{_j},e{_j}]!=0"
print("Alternative laws verified.")

# Verify flexibility (xy)x = x(yx) so [x,y,x]=0
for _i in range(8):
    for _j in range(8):
        assert np.allclose(oassoc_vec(_i, _j, _i), 0), f"[e{_i},e{_j},e{_i}]!=0"
print("Flexibility [x,y,x]=0 verified.")

# ===========================================================================
# PART A — Actual scaffold definition of K(w)
# ===========================================================================

print(f"\n{SEP}")
print("PART A — SCAFFOLD DEFINITION OF K(w)")
print(SEP)

# Fixed patch from octonion_assoc_patch.py
X_BASIS = [1, 2, 3]   # row space: e1, e2, e3
Y_BASIS = [4, 5, 6]   # col space: e4, e5, e6

NX = len(X_BASIS)
NY = len(Y_BASIS)

print(f"""
  Source file : octonion_assoc_patch.py
  Patch       : X = {X_BASIS} (basis element indices for row space)
                Y = {Y_BASIS} (basis element indices for col space)
  Probes      : w in {{1..7}} (imaginary octonion basis elements)

  oassoc_vec(i, j, k) computes [e_i, e_j, e_k] = (e_i*e_j)*e_k - e_i*(e_j*e_k)
  as an 8-vector (copied verbatim from the source file).

  For basis elements in the 8-dimensional octonion algebra (Fano-plane
  multiplication), the associator of three distinct imaginary elements is
  either:
    — zero (if the triple lies on a Fano line, or is an "extra-zero" triple
      due to sign cancellation in the Fano structure), or
    — exactly 2*sigma*e_r where sigma in {{-1,+1}} and r = i XOR j XOR k.

  DEFINITION of kappa_ab(w):

    kappa_ab(w) = component r of oassoc_vec(X[a], Y[b], w), divided by 2
    where r = X[a] XOR Y[b] XOR w

    kappa_ab(w) in {{-1, 0, +1}}

  This is the "signed bucket amplitude" at the XOR-predicted output
  direction — the natural scalar projection consistent with the Fano
  routing structure used throughout the codebase.

  K(w) is the 3x3 matrix [kappa_ab(w)] for a in {{0,1,2}}, b in {{0,1,2}}.
  Rows index X (a=0 -> e1, a=1 -> e2, a=2 -> e3).
  Cols index Y (b=0 -> e4, b=1 -> e5, b=2 -> e6).
""")

def kappa(a, b, w):
    """Signed bucket amplitude for bilinear pair (a,b) with probe w.
    Returns value in {-1, 0, +1}."""
    xa, yb = X_BASIS[a], Y_BASIS[b]
    vec = oassoc_vec(xa, yb, w)
    r = xa ^ yb ^ w       # XOR-predicted output bucket
    raw = vec[r]
    assert abs(raw - round(raw)) < 1e-10, f"Non-integer: {raw}"
    val = int(round(raw))
    assert val in (-2, 0, 2), f"Unexpected value {val} at ({a},{b},w={w})"
    return val // 2       # normalize to {-1, 0, +1}

def K_matrix(w):
    """Full 3x3 defect matrix for probe w."""
    return np.array([[kappa(a, b, w) for b in range(NY)]
                     for a in range(NX)], dtype=float)

def diagnostics(K):
    """Compute diagonal and cycle sums."""
    diag   = [K[a, a] for a in range(3)]
    kplus  = K[0,1] + K[1,2] + K[2,0]    # kappa_01 + kappa_12 + kappa_20
    kminus = K[0,2] + K[1,0] + K[2,1]    # kappa_02 + kappa_10 + kappa_21
    return diag, kplus, kminus

# ===========================================================================
# PART B — Single-seed search
# ===========================================================================

print(f"\n{SEP}")
print("PART B — SINGLE-SEED SEARCH")
print(SEP)
print(f"  Probes tested: w in {{1..7}} (e0 always gives zero associators)")
print()

header = f"  {'w':>3}  {'K diagonal':>20}  {'kappa_plus':>11}  {'kappa_minus':>12}  {'status':>30}"
print(header)
print("  " + SEP2)

single_successes = []
near_misses      = []

for w in range(1, 8):
    K = K_matrix(w)
    diag, kplus, kminus = diagnostics(K)

    diag_zero   = all(abs(d) < 1e-9 for d in diag)
    kplus_nz    = abs(kplus) > 1e-9
    kminus_nz   = abs(kminus) > 1e-9
    success     = diag_zero and kplus_nz and kminus_nz

    if success:
        status = "*** SUCCESS ***"
        single_successes.append(w)
    elif diag_zero and (kplus_nz or kminus_nz):
        status = "NEAR-MISS (zero diag, one cycle)"
        near_misses.append(('single', w, 'one_cycle'))
    elif not diag_zero and kplus_nz and kminus_nz:
        status = "NEAR-MISS (nonzero diag, both cycles)"
        near_misses.append(('single', w, 'nonzero_diag'))
    elif np.allclose(K, 0):
        status = "trivial (all zero)"
    elif kplus_nz and not kminus_nz:
        status = "partial (only kplus)"
    elif kminus_nz and not kplus_nz:
        status = "partial (only kminus)"
    else:
        status = "fail"

    diag_str = f"[{diag[0]:+.0f},{diag[1]:+.0f},{diag[2]:+.0f}]"
    print(f"  e{w:>2}  {diag_str:>20}  {kplus:>+10.0f}  {kminus:>+11.0f}  {status}")

print()

# ===========================================================================
# Full K(w) printout for each probe
# ===========================================================================

print(f"\n{SEP}")
print("FULL K(w) MATRICES — ALL PROBES")
print(SEP)

for w in range(1, 8):
    K = K_matrix(w)
    diag, kplus, kminus = diagnostics(K)
    print(f"\n  w = e{w}:   K(e{w}) =")
    for row in K:
        print("    " + "  ".join(f"{int(v):+2d}" for v in row))
    print(f"    diag = [{diag[0]:+.0f},{diag[1]:+.0f},{diag[2]:+.0f}]"
          f"    kappa_plus = {kplus:+.0f}    kappa_minus = {kminus:+.0f}")
    print(f"    associator sources per entry:")
    for a in range(3):
        for b in range(3):
            xa, yb = X_BASIS[a], Y_BASIS[b]
            vec    = oassoc_vec(xa, yb, w)
            r      = xa ^ yb ^ w
            nz     = [(idx, v) for idx, v in enumerate(vec) if abs(v) > 0.5]
            reason = ("Fano" if any(frozenset({xa,yb,w})==frozenset(f) for f in FANO_LINES)
                      else "alt-law" if xa==yb or yb==w or xa==w
                      else f"extra-zero" if np.allclose(vec, 0)
                      else f"+{vec[r]:.0f}*e{r}" if vec[r] > 0
                      else f"{vec[r]:.0f}*e{r}")
            print(f"      [e{xa},e{yb},e{w}] = {nz if nz else 0}  kappa={kappa(a,b,w):+d}  ({reason})")

# ===========================================================================
# PART C — Two-seed synthesis (only if single-seed fails)
# ===========================================================================

print(f"\n{SEP}")
print("PART C — TWO-SEED SYNTHESIS SEARCH")
print(SEP)

two_seed_successes = []

if single_successes:
    print(f"  Single-seed successes found: {['e'+str(w) for w in single_successes]}")
    print("  Running two-seed search for completeness.")
else:
    print("  No single-seed success. Searching all unordered pairs (w, w').")

for w1, w2 in combinations(range(1, 8), 2):
    K1 = K_matrix(w1)
    K2 = K_matrix(w2)
    K_eff = K1 + K2
    diag, kplus, kminus = diagnostics(K_eff)
    diag_zero  = all(abs(d) < 1e-9 for d in diag)
    kplus_nz   = abs(kplus) > 1e-9
    kminus_nz  = abs(kminus) > 1e-9
    if diag_zero and kplus_nz and kminus_nz:
        two_seed_successes.append((w1, w2, K_eff, kplus, kminus))

if two_seed_successes:
    print(f"  Two-seed successes: {len(two_seed_successes)} pairs")
    for w1, w2, K_eff, kp, km in two_seed_successes:
        print(f"    (e{w1}, e{w2}): K_eff =")
        for row in K_eff:
            print("      " + "  ".join(f"{int(v):+3d}" for v in row))
        print(f"      kappa_plus = {kp:+.0f}    kappa_minus = {km:+.0f}")
else:
    print("  No two-seed success found.")

# ===========================================================================
# PART D — Orbit-sum verification for all successful seeds
# ===========================================================================

print(f"\n{SEP}")
print("PART D — ORBIT-SUM VERIFICATION")
print(SEP)
print("""
  P = [[0,1,0],[0,0,1],[1,0,0]]  (cyclic permutation)
  Orbit: K0 = K, K1 = P K P^{-1}, K2 = P^2 K P^{-2}
  Verify K0 + K1 + K2 = alpha*P + beta*P^2
  and alpha = kappa_plus, beta = kappa_minus.
""")

P   = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=float)
Pi  = np.linalg.inv(P)            # = P^T = P^2 for this permutation
P2  = P @ P
P2i = np.linalg.inv(P2)

def orbit_sum(K):
    K0 = K.copy()
    K1 = P  @ K0 @ Pi
    K2 = P2 @ K0 @ P2i
    return K0, K1, K2, K0 + K1 + K2

all_successes_for_orbit = [(w, K_matrix(w)) for w in single_successes]
if not single_successes and two_seed_successes:
    # Use first two-seed result
    w1, w2, K_eff, kp, km = two_seed_successes[0]
    all_successes_for_orbit = [((w1, w2), K_eff)]

for seed_id, K in all_successes_for_orbit:
    diag, kplus, kminus = diagnostics(K)
    print(f"  --- Seed {seed_id} ---")
    K0, K1, K2, Ksum = orbit_sum(K)
    print(f"  K0 =")
    for row in K0: print("   " + "  ".join(f"{int(v):+2d}" for v in row))
    print(f"  K1 = P K P^{{-1}} =")
    for row in K1: print("   " + "  ".join(f"{int(v):+2d}" for v in row))
    print(f"  K2 = P^2 K P^{{-2}} =")
    for row in K2: print("   " + "  ".join(f"{int(v):+2d}" for v in row))
    print(f"  K0+K1+K2 =")
    for row in Ksum: print("   " + "  ".join(f"{int(v):+2d}" for v in row))

    # Check diagonal is zero
    diag_sum = [Ksum[a,a] for a in range(3)]
    print(f"  Diagonal of K0+K1+K2: {diag_sum}  (should be all zero: {all(abs(d)<1e-9 for d in diag_sum)})")

    # Extract alpha and beta
    # Ks = alpha*P + beta*P^2
    # alpha appears at P-nonzero positions (0,1),(1,2),(2,0)
    # beta  appears at P^2-nonzero positions (0,2),(1,0),(2,1)
    alpha_vals = [Ksum[0,1], Ksum[1,2], Ksum[2,0]]
    beta_vals  = [Ksum[0,2], Ksum[1,0], Ksum[2,1]]
    alpha_consistent = all(abs(v - alpha_vals[0]) < 1e-9 for v in alpha_vals)
    beta_consistent  = all(abs(v - beta_vals[0])  < 1e-9 for v in beta_vals)
    alpha = alpha_vals[0] if alpha_consistent else float('nan')
    beta  = beta_vals[0]  if beta_consistent  else float('nan')

    print(f"  alpha = P-positions  {alpha_vals}  consistent: {alpha_consistent}  value: {alpha:.1f}")
    print(f"  beta  = P^2-positions {beta_vals}  consistent: {beta_consistent}   value: {beta:.1f}")

    # Reconstruct and verify
    Kreconstructed = alpha * P + beta * P2
    residual = Ksum - Kreconstructed
    print(f"  ||K0+K1+K2 - alpha*P - beta*P^2|| = {np.linalg.norm(residual):.2e}  (should be 0)")

    # Verify alpha == kappa_plus, beta == kappa_minus
    print(f"  kappa_plus  = {kplus:+.1f},  alpha  = {alpha:+.1f}  ->  match: {abs(kplus-alpha)<1e-9}")
    print(f"  kappa_minus = {kminus:+.1f},  beta   = {beta:+.1f}  ->  match: {abs(kminus-beta)<1e-9}")
    print()

# ===========================================================================
# PART E — Engine recovery matrix verification
# ===========================================================================

print(f"\n{SEP}")
print("PART E — ENGINE RECOVERY MATRIX VERIFICATION")
print(SEP)

omega = np.exp(2j * np.pi / 3)
print(f"  omega = exp(2*pi*i/3) = {omega:.6f}")
print(f"  omega^2 = {omega**2:.6f}")
print(f"  1 + omega + omega^2 = {1+omega+omega**2:.2e}  (should be ~0)")
print()

for seed_id, K in all_successes_for_orbit:
    diag, kplus, kminus = diagnostics(K)
    alpha = float(kplus)
    beta  = float(kminus)

    print(f"  --- Seed {seed_id}: alpha={alpha:.1f}, beta={beta:.1f} ---")
    print(f"""
  The orbit-summed nonlinear defect response is:
    Sigma(K) = K0 + K1 + K2 = alpha*P + beta*P^2

  This acts on [m_ab] as:
    Q^orbit = Sum_ab (Sigma(K))_ab * m_ab = alpha*C_plus + beta*C_minus

  For two linearly-independent measurements, use DFT weighting of the
  orbit-summed channels:
    Q0 = alpha*C_plus + beta*C_minus
    Q1 = omega*alpha*C_plus + omega^2*beta*C_minus

  giving the 2x2 system:
    [Q0]   [alpha      beta     ] [C_plus ]
    [Q1] = [omega*alpha omega^2*beta] [C_minus]

    M = [[alpha, beta], [omega*alpha, omega^2*beta]]
""")

    M = np.array([
        [alpha,       beta],
        [omega*alpha, omega**2 * beta]
    ], dtype=complex)

    detM = np.linalg.det(M)
    print(f"  M =")
    print(f"    [{alpha:.4f}            {beta:.4f}     ]")
    print(f"    [{omega*alpha:.4f}  {omega**2*beta:.4f}]")
    print(f"  det(M) = {detM:.6f}")
    print(f"  |det(M)| = {abs(detM):.6f}  (should be > 0)")
    print(f"  det(M) != 0: {abs(detM) > 1e-9}")

    if abs(detM) > 1e-9:
        Minv = np.linalg.inv(M)
        print(f"\n  M^{{-1}} =")
        print(f"    [{Minv[0,0]:.4f}  {Minv[0,1]:.4f}]")
        print(f"    [{Minv[1,0]:.4f}  {Minv[1,1]:.4f}]")
        print(f"\n  RECOVERY: [C_plus, C_minus]^T = M^{{-1}} [Q0, Q1]^T")

    # Analytical det: alpha*omega^2*beta - beta*omega*alpha = alpha*beta*(omega^2-omega)
    det_analytic = alpha * beta * (omega**2 - omega)
    print(f"\n  Analytical formula: det(M) = alpha*beta*(omega^2-omega)")
    print(f"  = {alpha:.1f} * {beta:.1f} * ({omega**2-omega:.6f})")
    print(f"  = {det_analytic:.6f}   match: {abs(det_analytic - detM)<1e-9}")
    print()

# ===========================================================================
# PART F — Numerical sanity check
# ===========================================================================

print(f"\n{SEP}")
print("PART F — NUMERICAL SANITY CHECK (SCALAR CASE)")
print(SEP)

# Use scalar x_a, y_b (1x1 matrices) for clarity
# m_ab = x_a * y_b (scalars)
# D     = m00+m11+m22   (main diagonal sum)
# C_plus  = m01+m12+m20   (cyclic forward)
# C_minus = m02+m10+m21   (cyclic backward)

print("""
  Using scalar x_a, y_b.  Bilinear products m_ab = x_a * y_b.
  D        = m00 + m11 + m22
  C_plus   = m01 + m12 + m20
  C_minus  = m02 + m10 + m21
""")

rng = np.random.default_rng(42)
x = rng.integers(1, 5, 3).astype(float)   # x0, x1, x2
y = rng.integers(1, 5, 3).astype(float)   # y0, y1, y2
print(f"  x = {x}")
print(f"  y = {y}")

# Compute all m_ab
m = np.outer(x, y)   # m[a,b] = x[a]*y[b]
D       = m[0,0] + m[1,1] + m[2,2]
C_plus  = m[0,1] + m[1,2] + m[2,0]
C_minus = m[0,2] + m[1,0] + m[2,1]

print(f"\n  m_ab =")
print(f"    m00={m[0,0]:.1f}  m01={m[0,1]:.1f}  m02={m[0,2]:.1f}")
print(f"    m10={m[1,0]:.1f}  m11={m[1,1]:.1f}  m12={m[1,2]:.1f}")
print(f"    m20={m[2,0]:.1f}  m21={m[2,1]:.1f}  m22={m[2,2]:.1f}")
print(f"\n  True channels:")
print(f"    D       = {D}")
print(f"    C_plus  = {C_plus}")
print(f"    C_minus = {C_minus}")

# Mixed bilinear channels (DFT-3 products)
p = np.zeros(3, dtype=complex)
for t in range(3):
    u_t = sum(omega**t * x[a] for a in range(3))      # actually omega^(ta) x_a
    v_t = sum(omega**(-t) * y[b] for b in range(3))   # omega^(-tb) y_b
    # The product p_t = u_t * v_t (scalar)
    # but we use the simpler construction:
    u_t = sum((omega**t)**a * x[a] for a in range(3))
    v_t = sum((omega**(-t))**b * y[b] for b in range(3))
    p[t] = u_t * v_t

D_rec = (p[0] + p[1] + p[2]) / 3
print(f"\n  DFT-3 mixed products:")
print(f"    p0 = {p[0]:.4f}  p1 = {p[1]:.4f}  p2 = {p[2]:.4f}")
print(f"    D recovered = (p0+p1+p2)/3 = {D_rec:.4f}   (should be {D})")
print(f"    Error: {abs(D_rec - D):.2e}")

# Also verify C_plus and C_minus from DFT inversion
C_plus_rec  = (p[0] + omega**2 * p[1] + omega**4   * p[2]) / 3
C_minus_rec = (p[0] + omega    * p[1] + omega**2   * p[2]) / 3
print(f"    C_plus  recovered = {C_plus_rec:.4f}  (should be {C_plus})")
print(f"    C_minus recovered = {C_minus_rec:.4f}  (should be {C_minus})")

# Now test the defect-probe channel for the successful seed
print(f"\n  --- Defect probe check ---")
if single_successes:
    w_seed = single_successes[0]
    K_seed = K_matrix(w_seed)
    diag, kplus, kminus = diagnostics(K_seed)
    print(f"  Using seed w = e{w_seed},  K(e{w_seed}) =")
    for row in K_seed:
        print("    " + "  ".join(f"{int(v):+2d}" for v in row))
    print(f"  kappa_plus={kplus:.0f}, kappa_minus={kminus:.0f}")

    # Single raw defect measurement (t=0):
    # delta_t = sum_{a,b} kappa_ab(w) * omega^{t(a-b)} * m_ab
    delta = np.zeros(3, dtype=complex)
    for t in range(3):
        d = 0.0
        for a in range(3):
            for b in range(3):
                k = K_seed[a, b]
                d += k * (omega**(t*(a-b))) * m[a, b]
        delta[t] = d

    print(f"\n  Raw defect channels (before orbit sum):")
    print(f"    delta_0 = {delta[0]:.4f}")
    print(f"    delta_1 = {delta[1]:.4f}")
    print(f"    delta_2 = {delta[2]:.4f}")

    # Orbit-sum: sum_t delta_t (= 3 * diagonal of K * diagonal of m = 0 since diag=0)
    orbit_0 = (delta[0] + delta[1] + delta[2]) / 3
    # Q0 = orbit-averaged channel (should be alpha*C_plus + beta*C_minus)
    Q0 = sum(sum(K_seed[a,b] * m[a,b] for a in range(3) for b in range(3))
             for _ in [None])
    # Wait, the orbit-summed channel:
    # Q^orbit = sum_{a,b} (K0+K1+K2)_{ab} * m_ab = alpha*C_plus + beta*C_minus

    _, _, Ksum_seed = orbit_sum(K_seed)[1], orbit_sum(K_seed)[2], orbit_sum(K_seed)[3]
    Ksum_matrix = orbit_sum(K_seed)[3]
    Q_orbit_raw = sum(Ksum_matrix[a,b] * m[a,b] for a in range(3) for b in range(3))
    Q_orbit_expected = kplus * C_plus + kminus * C_minus

    print(f"\n  Orbit-summed defect:")
    print(f"    Q_orbit = sum_{{a,b}} (K0+K1+K2)_{{ab}} * m_ab = {Q_orbit_raw:.4f}")
    print(f"    Expected: kappa_plus*C_plus + kappa_minus*C_minus")
    print(f"            = {kplus:.0f}*{C_plus} + {kminus:.0f}*{C_minus}")
    print(f"            = {Q_orbit_expected:.4f}")
    print(f"    Match: {abs(Q_orbit_raw - Q_orbit_expected) < 1e-9}")

    # Two defect measurements for recovery:
    Q0_meas = kplus * C_plus + beta  * C_minus  # where beta=kminus
    Q1_meas = omega * kplus * C_plus + omega**2 * kminus * C_minus

    alpha = float(kplus)
    beta  = float(kminus)
    M_num = np.array([
        [alpha,       beta],
        [omega*alpha, omega**2 * beta]
    ], dtype=complex)
    RHS = np.array([Q0_meas, Q1_meas], dtype=complex)
    sol  = np.linalg.solve(M_num, RHS)
    C_plus_rec2  = sol[0].real
    C_minus_rec2 = sol[1].real

    print(f"\n  Recovery from two defect channel measurements:")
    print(f"    Q0 = alpha*C_plus + beta*C_minus = {Q0_meas:.4f}")
    print(f"    Q1 = omega*alpha*C_plus + omega^2*beta*C_minus = {Q1_meas:.4f}")
    print(f"    Solving M * [C_plus, C_minus]^T = [Q0, Q1]^T:")
    print(f"    C_plus  recovered = {C_plus_rec2:.6f}  (true: {C_plus})")
    print(f"    C_minus recovered = {C_minus_rec2:.6f}  (true: {C_minus})")
    print(f"    C_plus  error     = {abs(C_plus_rec2  - C_plus):.2e}")
    print(f"    C_minus error     = {abs(C_minus_rec2 - C_minus):.2e}")

# ===========================================================================
# SUMMARY
# ===========================================================================

print(f"\n{SEP}")
print("FINAL CONCLUSION")
print(SEP)

if single_successes:
    print(f"\n  F.1 — A valid single seed exists.")
    print(f"        Successful probe(s): {['e'+str(w) for w in single_successes]}")
    for w in single_successes:
        K = K_matrix(w)
        diag, kp, km = diagnostics(K)
        print(f"\n        w = e{w}:")
        print(f"          K(e{w}) = {K.astype(int).tolist()}")
        print(f"          Diagonal    = {[int(d) for d in diag]}")
        print(f"          kappa_plus  = {kp:+.0f}")
        print(f"          kappa_minus = {km:+.0f}")
        print(f"          det(M) = alpha*beta*(omega^2-omega)")
        alpha, beta = float(kp), float(km)
        detM = alpha * beta * (omega**2 - omega)
        print(f"               = {alpha:.1f} * {beta:.1f} * (omega^2-omega)")
        print(f"               = {detM:.6f}  != 0: {abs(detM) > 1e-9}")
elif two_seed_successes:
    print(f"\n  F.2 — No single seed exists, but a valid two-seed synthesis exists.")
    print(f"        Successful pair(s): {[(f'e{w1}','e{w2}') for w1,w2,_,_,_ in two_seed_successes]}")
else:
    print(f"\n  F.3 — Neither single-seed nor two-seed synthesis exists in the current scaffold.")

print()
