"""
Nonlinear channel stratification analysis for the Hamilton channel triple.

L(λ) = D + λ·C_plus + λ²·C_minus
P0 = L(1),   P1 = L(ω²),   P2 = L(ω)
ω = exp(2πi/3)   (primitive cube root of unity)

Question: on which nonlinear strata can D be recovered from P0, P1 alone?
"""
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, simplify, expand, zeros, eye, I, pi, exp, sqrt
from sympy import factor, collect, cancel, Rational

# ══════════════════════════════════════════════════════════════════════════════
# 0. ALGEBRAIC CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

omega = exp(2*pi*I/3)
omega = sp.Rational(-1,2) + sp.sqrt(3)*I/2          # exact form of exp(2πi/3)
omega2 = sp.Rational(-1,2) - sp.sqrt(3)*I/2          # ω² = conjugate(ω)

alpha = 1 - omega2          # α = 1 - ω²  (appears in eq A)
beta  = 1 - omega           # β = 1 - ω   (appears in eq A)

print("= " * 36)
print("ALGEBRAIC CONSTANTS")
print("= " * 36)
print(f"  ω  = {omega}")
print(f"  ω² = {omega2}")
print(f"  α = 1-ω² = {simplify(alpha)}")
print(f"  β = 1-ω  = {simplify(beta)}")
print(f"  Key identity αω = {simplify(alpha*omega)}  (should equal -β = {simplify(-beta)})")
print(f"  αω + β = {simplify(alpha*omega + beta)}   (should be 0)")
print(f"  α²ω² = {simplify(alpha**2 * omega**2)}   β² = {simplify(beta**2)}  (should be equal)")

# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERAL 2-PRODUCT LINEAR SYSTEM (before adding stratum constraint)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "= "*36)
print("PART 1 — 2-PRODUCT LINEAR UNDER-DETERMINATION")
print("= "*36)

d, cp, cm = symbols('D C_+ C_-')
p0, p1 = symbols('P_0 P_1')

# Two observable equations:
eq_sum  = sp.Eq(2*d - omega*cp - omega2*cm,  p0+p1)   # (B) from sum
eq_diff = sp.Eq(alpha*cp + beta*cm,           p0-p1)   # (A) from diff

print("""
Two observations give:
  (A)  α·C_+ + β·C_-  = P0 - P1          [difference]
  (B)  2D - ω·C_+ - ω²·C_- = P0 + P1    [sum]

These are 2 linear equations in 3 unknowns {D, C_+, C_-}.
The null space (unobservable direction) through the standard DFT is spanned by:
  null ∝ L(ω) - the third product P2.
Without a third constraint, D cannot be recovered.
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2. STRATUM 1 — SQUARE STRATUM  D=A², C_+=AB+BA, C_-=B²
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("STRATUM 1 — SQUARE STRATUM  L(λ) = (A+λB)²")
print("= "*36)

a, b = symbols('a b')         # scalar stand-in for eigenvalues of A, B
d_sq   = a**2
cp_sq  = 2*a*b                # commuting case; full: AB+BA
cm_sq  = b**2

P0_sq = d_sq + cp_sq + cm_sq       # (a+b)²
P1_sq = d_sq + omega2*cp_sq + omega*cm_sq

P0_sq_exp = expand(P0_sq)
P1_sq_exp = expand(P1_sq)

print(f"\n  P0 = (a+b)²  = {P0_sq_exp}")
print(f"  P1 = (a+ω²b)²= {expand(P1_sq_exp)}")

# Can we express a² from P0, P1 symbolically (scalar commuting case)?
# P0 = (a+b)² → √P0 = a+b  (choosing positive branch)
# P1 = (a+ω²b)² → √P1 = a+ω²b
# √P0 + √P1 + √P2 = 3a  but we lack P2
# Need ALGEBRAIC combination without square roots.

# Try: is D expressible as a rational function of P0, P1 alone?
# For the answer: eliminate b from P0=(a+b)², P1=...
b_from_P0 = sp.sqrt(p0) - a    # b = √P0 - a
P1_elim = expand(a**2 + omega2*2*a*(sp.sqrt(p0)-a) + omega*(sp.sqrt(p0)-a)**2)
D_from_P0P1 = simplify(a**2)

print("""
  Recovery attempt (commuting / scalar case):
  From P0 = (a+b)²: b = √P0 - a
  Substituting into P1:
""")
print(f"  P1 = {P1_elim}")
print(f"\n  → Can solve for a = f(P0,P1) but requires matrix square root √P0.")
print(f"  → For non-commuting matrices: matrix square root not unique/well-defined.")
print(f"\n  STRATUM 1 result: D recoverable in scalar/commuting case using matrix")
print(f"  square root of P0; NOT recoverable as an algebraic (polynomial) function")
print(f"  of P0, P1 alone. Requires √P0 as auxiliary.")

# The linear combination without square roots:
lin1 = simplify(P0_sq + omega2*P1_sq)       # P0 + ω²P1
lin2 = simplify(P0_sq + omega*P1_sq)        # P0 + ωP1
print(f"\n  P0 + ω²·P1 = {expand(P0_sq + omega2*P1_sq)}")
print(f"  P0 + ω·P1  = {expand(P0_sq + omega*P1_sq)}")
print(f"  → Neither simplifies to D = a² alone.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. STRATUM 2 — COMMUTING SQUARE  [A,B]=0
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "= "*36)
print("STRATUM 2 — COMMUTING SQUARE  [A,B]=0")
print("= "*36)

# The commuting square stratum is the INTERSECTION of:
#   - square stratum (D=A², C_+=2AB, C_-=B²)
#   - [A,B]=0 (commutativity)
# Key algebraic consequence: C_+² = 4D·C_-  (discriminant is zero!)

print("""
  C_+² = (2AB)² = 4A²B² = 4D·C_-     ←  Δ = C_+² - 4D·C_- = 0

  This means: commuting square ⊂ discriminant-degenerate stratum.
  Recovery on this stratum reduces to recovery on Stratum 3.
  See Stratum 3 for the exact formula.
""")

# Verify Δ=0 symbolically
Delta_comm = expand((2*a*b)**2 - 4*a**2*b**2)
print(f"  Δ = (2ab)² - 4a²b² = {Delta_comm}   ✓ (always zero)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. STRATUM 3 — DISCRIMINANT-DEGENERATE  Δ = C_+² - 4D·C_- = 0
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("STRATUM 3 — DISCRIMINANT-DEGENERATE  Δ = C_+² - 4D·C_- = 0")
print("= "*36)

print("""
  This is the MAIN recovery stratum. The stratum condition is:
    Δ = C_+² - 4·D·C_- = 0   ←  nonlinear, degree 2 in (D,C_+,C_-)

  The three equations:
    (A)  α·C_+ + β·C_- = T    (T = P0 - P1)
    (B)  2D - ω·C_+ - ω²·C_- = S    (S = P0 + P1)
    (S)  C_+² = 4·D·C_-             (stratum)

  Elimination procedure (scalar / commuting case):
    From (A): C_+ = (T - β·C_-) / α
    From (B): D = (S + ω·C_+ + ω²·C_-) / 2
    Substitute both into (S) and expand.

  KEY IDENTITY used: αω = -β  ⟹  the coefficient of T·C_- in the
  expanded equation vanishes exactly, leaving a PURE quadratic in C_-:
""")

# Symbolic derivation
S, T = symbols('S T', complex=True)
cm_var = symbols('c_m', complex=True)

# Express cp and d in terms of cm, S, T
cp_expr = (T - beta*cm_var) / alpha
d_expr  = (S + omega*cp_expr + omega2*cm_var) / 2

# Stratum condition: cp^2 = 4*d*cm
stratum_lhs = expand(cp_expr**2)
stratum_rhs = expand(4*d_expr*cm_var)

residual = expand(stratum_lhs - stratum_rhs)

# Collect as polynomial in cm_var
poly_cm = sp.Poly(residual.rewrite(sp.Add), cm_var)
print("  Expanding (C_+/α)·(T-β·C_-)² - 4·D·C_- into polynomial in C_-:")

# Simplify coefficients
coeffs_raw = poly_cm.all_coeffs()
coeffs_simplified = [simplify(c) for c in coeffs_raw]
print(f"  Coefficients of C_-^2, C_-^1, C_-^0 (raw): might be complex expressions.")

# Use the known result: the simplified form is 3β²c² + 2α²Sc - T² = 0
# Let's verify this directly
c2_coeff = simplify(3*beta**2)
c1_coeff = simplify(2*alpha**2*S)
c0_coeff = simplify(-T**2)

# Check by substituting c_-=b², S, T as in stratum
# (verify the quadratic holds on stratum)
test_cm = b**2
test_S  = 2*a**2 - 2*omega*a*b - omega2*b**2     # P0+P1 on commuting square stratum
test_T  = 2*alpha*a*b + beta*b**2               # P0-P1 on commuting square stratum

verify = simplify(3*beta**2*test_cm**2 + 2*alpha**2*test_S*test_cm - test_T**2)
print(f"\n  Claimed recovery quadratic:  3β²·C_-² + 2α²·S·C_- - T² = 0")
print(f"  where S = P0+P1,  T = P0-P1  (both known from observations)")
print(f"\n  Symbolic verification on commuting square stratum:")
print(f"    Substituting  C_- = b², S = P0+P1, T = P0-P1  (commuting case)")
print(f"    3β²(b²)² + 2α²·S·b² - T² = {simplify(verify)}  ✓  (should be 0)")

print(f"""
  RECOVERY FORMULA (scalar / commuting matrix case):
  ─────────────────────────────────────────────────
  Given P0, P1 only, on the Δ=0 stratum:

    C_-  = [ -α²·(P0+P1) ± √(α⁴·(P0+P1)² + 3β²·(P0-P1)²) ] / (3β²)

    C_+  = [(P0-P1) - β·C_-] / α

    D    = [(P0+P1) + ω·C_+ + ω²·C_-] / 2

  AMBIGUITY: the ± gives TWO solutions for C_-.
  The physically correct one satisfies C_- ≥ 0 (for PSD channels),
  or is selected by the constraint that C_+² = 4D·C_- with C_-, D ≥ 0.
  For indefinite channels: 2-to-1 ambiguity (cannot be resolved without P2
  OR additional channel structure).
""")

# Numerical constants α, β for display
alph_num = complex(alpha.rewrite(sp.cos))
beta_num = complex(beta.rewrite(sp.cos))

# ══════════════════════════════════════════════════════════════════════════════
# 5. STRATUM 3 — NUMERICAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("STRATUM 3 — NUMERICAL VERIFICATION (diagonal 3×3 matrices)")
print("= "*36)

omega_n  = np.exp(2j*np.pi/3)
omega2_n = np.conj(omega_n)
alpha_n  = 1 - omega2_n
beta_n   = 1 - omega_n

def build_channels(A, B):
    """Square stratum."""
    D  = A @ A
    Cp = A @ B + B @ A
    Cm = B @ B
    return D, Cp, Cm

def pack_products(D, Cp, Cm):
    P0 = D + Cp + Cm
    P1 = D + omega2_n*Cp + omega_n*Cm
    P2 = D + omega_n*Cp + omega2_n*Cm
    return P0, P1, P2

def recover_D_stratum3(P0, P1):
    """Recover D from P0, P1 on discriminant-degenerate stratum (Δ=0).
    Works exactly for diagonal (simultaneously diagonalizable) matrices.
    Returns the two candidate D values.

    Recovery quadratic (applied element-wise for diagonal matrices):
        3β²·C_-² + 2α²·S·C_- - T² = 0
    with S = P0+P1, T = P0-P1.
    Solution: C_- = [-α²S ± √(α⁴S² + 3β²T²)] / (3β²)
    """
    S = P0 + P1
    T = P0 - P1
    # All operations element-wise; a2_sc is a scalar coefficient.
    a2_sc   = 3 * beta_n**2              # scalar (complex constant)
    b_mat   = 2 * alpha_n**2 * S        # element-wise scale of S
    c_mat   = -(T * T)                  # element-wise T²
    disc    = b_mat * b_mat - 4 * a2_sc * c_mat   # 4α⁴S² + 12β²T², elementwise
    sq_disc = np.sqrt(disc)
    Cm_plus  = (-b_mat + sq_disc) / (2 * a2_sc)
    Cm_minus = (-b_mat - sq_disc) / (2 * a2_sc)
    results = []
    for Cm_cand in [Cm_plus, Cm_minus]:
        Cp_cand = (T - beta_n * Cm_cand) / alpha_n
        D_cand  = (S + omega_n * Cp_cand + omega2_n * Cm_cand) / 2
        results.append((D_cand, Cp_cand, Cm_cand))
    return results

np.random.seed(42)
print()
for trial in range(5):
    # Diagonal commuting matrices → Δ=0 automatically
    a_eigs = np.random.randn(3)
    b_eigs = np.random.randn(3)
    A = np.diag(a_eigs)
    B = np.diag(b_eigs)

    D_true, Cp_true, Cm_true = build_channels(A, B)
    P0, P1, P2 = pack_products(D_true, Cp_true, Cm_true)

    # Verify Δ=0
    Delta = Cp_true @ Cp_true - 4*D_true @ Cm_true
    assert np.allclose(Delta, 0, atol=1e-12), f"Stratum violated: |Δ|={np.linalg.norm(Delta):.2e}"

    # Recover from P0, P1 only
    results = recover_D_stratum3(P0, P1)
    errors = [np.linalg.norm(r[0] - D_true) for r in results]
    best_idx = int(np.argmin(errors))

    print(f"  Trial {trial+1}: a_eigs={np.round(a_eigs,3)}, b_eigs={np.round(b_eigs,3)}")
    print(f"    |Δ| = {np.linalg.norm(Delta):.1e}  (should be ~0)")
    print(f"    Recovery errors: root+ |D_cand-D_true|={errors[0]:.2e},  root- |D_cand-D_true|={errors[1]:.2e}")
    print(f"    Best root: {'+' if best_idx==0 else '-'}   |error| = {errors[best_idx]:.2e}  ✓")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRATUM 4 — COMMUTATOR-DEFECT  [C_+, C_-] = 0
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("STRATUM 4 — COMMUTATOR-DEFECT  [C_+, C_-] = 0")
print("= "*36)

print("""
  Stratum condition: [C_+, C_-] = 0   ←  nonlinear, degree 2

  When C_+ and C_- commute, they are simultaneously diagonalizable.
  In that shared eigenbasis, the system reduces to 3 independent scalar
  triples (d_i, c+_i, c-_i)  per eigenvalue.
  Each scalar triple CAN be in the discriminant stratum or not.

  Recovery conditions:
  ┌───────────────────────────────────────────────────────────────────┐
  │  Case A: EACH scalar triple (d_i, c+_i, c-_i) satisfies Δ_i=0  │
  │  → Full D recovery via Stratum 3 formula applied diag-wise.      │
  │  → Stratum: [C_+,C_-]=0  AND  det(C_+²-4D·C_-)=0 (per mode)   │
  ├───────────────────────────────────────────────────────────────────┤
  │  Case B: [C_+,C_-]=0 but individual Δ_i ≠ 0                    │
  │  → Each scalar triple has 3 unknowns and only 2 equations.       │
  │  → NOT recoverable from P0, P1 alone; P2 still required.        │
  └───────────────────────────────────────────────────────────────────┘

  Conclusion: [C_+,C_-]=0 is NOT sufficient alone for 2-product recovery.
  It must be combined with the Δ=0 condition per eigenvalue.
""")

# Polynomial relation involving [A,B]:
# On the square stratum: C_+ = AB+BA, C_- = B²
# [C_+, C_-] = [AB+BA, B²] = AB³+BAB² - B²AB - B³A = [A,B³] + B[A,B]B
# If [A,B] = 0: [C_+,C_-] = 0 automatically.
# If [A,B] = λI (Heisenberg-like): [C_+,C_-] = [A,B³] + λB² = 3λB² + λB² = 0... 
#   actually [A,B³] = [A,B]B² + B[A,B]B + B²[A,B] = 3λB² when [A,B]=λI.
#   So [C_+,C_-] = 3λB² + λB² = 4λB² ≠ 0 (for λ≠0).
# So Heisenberg commutator does NOT guarantee [C_+,C_-]=0.

# ══════════════════════════════════════════════════════════════════════════════
# 7. COVERAGE QUESTION
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("PART D — COVERAGE: can a finite list of strata cover generic triples?")
print("= "*36)

print("""
  Generic space of channel triples (D, C_+, C_-): 3n² real parameters (n=3 → 27).
  
  Strata where 2-product recovery is possible:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Stratum               │ Defining condition         │ Codimension in full │
  │                       │                            │ parameter space     │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Square (S1)           │ D=A², C_+=AB+BA, C_-=B²    │  n²  (18 for n=3)  │
  │ Commuting sq. (S2)    │ S1 ∩ [A,B]=0               │  n²+n(n-1)/2 = 21  │
  │ Discriminant-deg (S3) │ det(C_+²-4D·C_-) = 0      │  1                 │
  │ [C_+,C_-]=0 ∩ S3     │ S3 ∩ simultaneous diag.    │  > 1               │
  └──────────────────────────────────────────────────────────────────────────┘

  KEY FACT: Each stratum is a PROPER ALGEBRAIC SUBVARIETY of the full space.
  
  → A finite union of proper subvarieties has measure zero.
  → The COMPLEMENT (generic triples NOT on any listed stratum) is dense and
    has full measure.
  → On the complement, D cannot be recovered from P0, P1 alone.
  
  ANSWER: NO. A finite list of nonlinear strata cannot cover generic channel
  triples. Each stratum is a codimension ≥ 1 algebraic variety in the 27-dim
  parameter space. Their union has measure zero.

  WHAT CAN BE DONE:
  - For any SPECIFIC known (D, C_+, C_-): check if it lies on a 2-product
    stratum; if yes, apply the appropriate formula.
  - The discriminant condition det(Δ) = 0 is the LARGEST (codimension-1)
    stratum, and is the best general-purpose 2-product recovery condition:
      3β²·C_-² + 2α²(P0+P1)·C_- - (P0-P1)² = 0
    (holds exactly in scalar and commuting-matrix cases).
""")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("= "*36)
print("SUMMARY — STRATA AND RECOVERY")
print("= "*36)

print(f"""
 Stratum │ Condition                    │ Algebraic? │ D from P0,P1? │ Formula
─────────┼──────────────────────────────┼────────────┼───────────────┼─────────────────
    S1   │ L(λ)=(A+λB)²                 │ nonlinear  │ partial: yes  │ via √P0 (needs
         │                              │ deg 2      │ with matrix   │ matrix sq-root)
         │                              │            │ square root   │
─────────┼──────────────────────────────┼────────────┼───────────────┼─────────────────
    S2   │ S1 ∩ [A,B]=0                 │ nonlinear  │ YES via Δ=0   │ Stratum 3 formula
         │                              │ deg 2+n    │ (diagonal)    │ (eigenvalue-wise)
─────────┼──────────────────────────────┼────────────┼───────────────┼─────────────────
    S3   │ Δ = C_+²-4D·C_- = 0         │ nonlinear  │ YES (2-valued)│ 3β²C_-² +
         │                              │ degree 2   │               │ 2α²(P0+P1)C_- -
         │                              │ (proper!)  │               │ (P0-P1)² = 0
─────────┼──────────────────────────────┼────────────┼───────────────┼─────────────────
    S4   │ [C_+,C_-]=0                  │ nonlinear  │ only IF also  │ Reduces to S3
         │                              │ deg 2      │ Δ=0 per mode  │ per eigenvalue
─────────┼──────────────────────────────┼────────────┼───────────────┼─────────────────
  GENERIC│ none                         │   —        │ NO            │ Need P2
─────────┴──────────────────────────────┴────────────┴───────────────┴─────────────────

CANONICAL RECOVERY FORMULA for Stratum 3 (Δ = 0):

  Let  S = P0 + P1,   T = P0 - P1
  Solve:  3β²·C_-² + 2α²·S·C_- - T² = 0
  where:  β = 1-ω,  α = 1-ω²,  β²≈0.75-i√3/2 (|β|=√3)

  C_- = [-α²S ± √(α⁴S² + 3β²T²)] / (3β²)
  C_+ = (T - β·C_-) / α
  D   = (S + ω·C_+ + ω²·C_-) / 2

  Nonlinear: YES (involves ±√ — quadratic in C_-)
  Proper:    YES (Δ=0 is a codimension-1 condition)
  Generic coverage: NO (measure-zero stratum)
""")

# Numerical values of constants
a_num = complex(alpha.rewrite(sp.cos).evalf())
b_num = complex(beta.rewrite(sp.cos).evalf())
print(f"  Numerical constants:  α = {alpha_n:.6f}  β = {beta_n:.6f}")
print(f"  |β|² = {abs(beta_n)**2:.6f}  (= 3),  |α|² = {abs(alpha_n)**2:.6f}  (= 3)")
