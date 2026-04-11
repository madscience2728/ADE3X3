"""
Independent verification of tower number theory claims.
Builds everything from scratch — no dependency on zn_tower_sweep.py.
"""
import numpy as np
from fractions import Fraction

def make_level1(n):
    """Level-1 Z_n algebra with λ=-1."""
    C = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            C[i, j, k] = 1.0 if (i == 0 or j == 0) else -1.0
    return C

def conj_matrix(d):
    """Hypercomplex conjugation: negate all but slot 0."""
    J = -np.eye(d)
    J[0, 0] = 1.0
    return J if d > 1 else np.eye(1)

def cd_level_up(C, n):
    """Cayley-Dickson-style level-up with conjugation."""
    d = C.shape[0]
    D = n * d
    J = conj_matrix(d)
    Cnew = np.zeros((D, D, D))

    for i1 in range(n):
        for i2 in range(n):
            m = (i1 + i2) % n
            for a in range(d):
                for b in range(d):
                    row, col = i1*d+a, i2*d+b
                    if i1 == 0 and i2 == 0:
                        for c in range(d):
                            Cnew[row, col, m*d+c] += C[a, b, c]
                    elif i1 == 0 and i2 != 0:
                        for c in range(d):
                            Cnew[row, col, m*d+c] += C[b, a, c]
                    elif i1 != 0 and i2 == 0:
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                Cnew[row, col, m*d+c] += J[bp, b] * C[a, bp, c]
                    else:
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                Cnew[row, col, m*d+c] += -J[bp, b] * C[bp, a, c]
    return Cnew

def count_violations(C):
    d = C.shape[0]
    # (AB)C via einsum
    # (e_i * e_j) = sum_k C[i,j,k] e_k
    # ((e_i * e_j) * e_l)_m = sum_k C[i,j,k] * C[k,l,m]
    # (e_i * (e_j * e_l))_m = sum_k C[j,l,k] * C[i,k,m]
    LHS = np.einsum('ijk,klm->ijlm', C, C)
    RHS = np.einsum('jlk,ikm->ijlm', C, C)
    diff = LHS - RHS
    violations = np.count_nonzero(np.abs(diff).sum(axis=-1) > 1e-10)
    total = d**3
    return violations, total

print("=" * 60)
print("INDEPENDENT VERIFICATION")
print("=" * 60)

# ── Claim A: V_1(n) = 2(n-1)(n-2) ──
print("\nCLAIM A: V_1(n) = 2(n-1)(n-2)")
for n in [2, 3, 4, 5, 7, 11, 13]:
    C = make_level1(n)
    v, t = count_violations(C)
    pred = 2*(n-1)*(n-2)
    ok = "✓" if v == pred else "✗"
    print(f"  n={n:>2}: V_1={v:>5}, 2(n-1)(n-2)={pred:>5}  {ok}")

# ── Claim B: V_2/V_1 = 2n³ for n≥3 ──
print("\nCLAIM B: V_2/V_1 = 2n³  (equivalently, violation fraction doubles)")
for n in [3, 4, 5, 7]:
    C = make_level1(n)
    v1, t1 = count_violations(C)
    C2 = cd_level_up(C, n)
    v2, t2 = count_violations(C2)
    ratio = v2/v1 if v1 > 0 else float('inf')
    pred = 2*n**3
    ok = "✓" if abs(ratio - pred) < 0.01 else "✗"
    frac1 = Fraction(v1, t1)
    frac2 = Fraction(v2, t2)
    print(f"  n={n}: V_1={v1}, V_2={v2}, ratio={ratio:.1f}, 2n³={pred}  {ok}")
    print(f"         frac1={frac1}, frac2={frac2}, frac2/frac1={float(frac2)/float(frac1):.4f}")

# ── Claim C: Z_2 octonion violations = 168 ──
print("\nCLAIM C: Z_2 level-3 (octonions) violations = 168")
C = make_level1(2)
C = cd_level_up(C, 2)  # dim 4 = quaternions
v_q, _ = count_violations(C)
print(f"  Level 2 (dim 4, quaternions): V={v_q}")
C = cd_level_up(C, 2)  # dim 8 = octonions
v_o, t_o = count_violations(C)
print(f"  Level 3 (dim 8, octonions):   V={v_o}")
print(f"  168 = |PSL(2,7)| = |GL(3,F_2)| ? {v_o == 168}  {'✓' if v_o == 168 else '✗'}")
print(f"  Fraction: {Fraction(v_o, t_o)} = {v_o}/{t_o}")

# ── Claim D: Z_2 limit → 1/2 exactly ──
print("\nCLAIM D: Z_2 tower S_assoc → 1/2")
C = make_level1(2)
for level in range(1, 8):
    C = cd_level_up(C, 2)
    v, t = count_violations(C)
    s = Fraction(t - v, t)
    print(f"  Level {level+1} (dim {C.shape[0]:>4}): S={float(s):.12f}  = {s}")
    if C.shape[0] >= 128:
        break

# ── Claim E: Is 168 really |Aut(O)|? ──
print("\nCLAIM E: Is 168-as-violations meaningful or coincidence?")
print("  |PSL(2,7)| = 7·6·4/gcd = 168  ✓")
print("  |GL(3,F_2)| = (8-1)(8-2)(8-4) = 7·6·4 = 168  ✓")
print(f"  Octonion violations: {v_o}")
print(f"  Non-associative fraction: {v_o}/512 = {Fraction(v_o, 512)} = 21/64")
# Check: is 21 = C(7,2)? 7*6/2 = 21. Yes.
print(f"  21 = C(7,2) = 7·6/2, and 64 = 2⁶ = 8³/8")
print(f"  So violation density = C(7,2)/2⁶ = (imaginary pairs)/(2^{'{'}6{'}'})")

# ── Claim F: Universal 1/2 basin? Check Z_3 deeper ──
print("\nCLAIM F: Z_3 convergence direction")
z3_s = []
C = make_level1(3)
for level in range(1, 5):
    C = cd_level_up(C, 3)
    v, t = count_violations(C)
    s = float(Fraction(t-v, t))
    z3_s.append(s)
    print(f"  Level {level+1} (dim {C.shape[0]:>4}): S={s:.10f}")
    if C.shape[0] >= 81:
        break

print(f"  Decreasing? {all(z3_s[i] > z3_s[i+1] for i in range(len(z3_s)-1))}")
print(f"  Last value {z3_s[-1]:.6f} {'> 0.5' if z3_s[-1] > 0.5 else '<= 0.5'}")
print(f"  Distance from 0.5: {z3_s[-1] - 0.5:.6f}")

# ── Bonus: n=11 level-1 and level-2 ──
print("\nBONUS: n=11 (not in original sweep)")
C = make_level1(11)
v1, t1 = count_violations(C)
pred1 = 2*10*9
print(f"  Level 1: V={v1}, pred=2·10·9={pred1}  {'✓' if v1==pred1 else '✗'}")
# level 2 would be dim 121 — doable
C2 = cd_level_up(C, 11)
v2, t2 = count_violations(C2)
pred2 = 4 * 11**3 * 10 * 9
print(f"  Level 2: V={v2}, pred=4·11³·10·9={pred2}  {'✓' if v2==pred2 else '✗'}")
print(f"  Fraction: {Fraction(v2, t2)}")
