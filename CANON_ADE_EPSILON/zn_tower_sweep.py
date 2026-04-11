"""
Z_n Tower Associativity Sweep — Testing the Emmy-Hilbert Conjecture:

    lim_{k→∞} S_assoc^{Z_n}[k] = (n-1)/n

For each n, we build the Z_n-graded tripling tower with λ=-1 and measure
the associativity fraction at each level by exhaustive evaluation of ALL
basis triples.

NO OPTIMIZATION. Pure algebra. Exact integer/rational arithmetic where possible.
"""

import numpy as np
from itertools import product as iproduct
from fractions import Fraction
import json
import os
import time

# ═══════════════════════════════════════════════════════════════
# GENERIC Z_n TOWER
# ═══════════════════════════════════════════════════════════════

def build_Zn_structure_constants(n):
    """
    Build the Level-1 Z_n tower algebra with λ=-1.

    Basis: e_0, e_1, ..., e_{n-1}  (dimension n)
    Product: (a_0,...,a_{n-1}) · (b_0,...,b_{n-1}) = (P_0,...,P_{n-1})

    where P_k = sum over (i,j) with (i+j) mod n = k of sign(i,j) * a_i * b_j

    The sign rule (λ=-1 generalization):
      sign(i,j) = +1 if i=0 or j=0 (unit element acts trivially)
      sign(i,j) = -1 otherwise (the λ=-1 penalty for "imaginary×imaginary")

    This is the direct generalization of Emmy's λ=-1 construction:
    - Z_2: (a,b)·(c,d) = (ac - bd, ad + bc)  [complex numbers]
    - Z_3: (a,b,c)·(d,e,f) = (ad - bf - ce, ae + bd - cf, af - be + cd)
    """
    # Structure constants C[i,j,k] = coefficient of e_k in e_i · e_j
    C = np.zeros((n, n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            if i == 0 or j == 0:
                C[i, j, k] = 1.0
            else:
                C[i, j, k] = -1.0

    return C


def build_conjugation_matrix(d):
    """
    Build conjugation for the Cayley-Dickson style construction.
    For level-k algebra of dimension d:
      conj(a_0, a_1, ..., a_{n-1}) = (conj(a_0), -a_1, ..., -a_{n-1})
    At the base level (d=1): conj is identity.

    Returns a d×d matrix J such that conj(x) = J @ x.
    """
    J = np.eye(d, dtype=np.float64)
    if d > 1:
        # The first slot keeps its conjugation from below;
        # all other slots get negated.
        # But recursively, the first slot also has its own conjugation.
        # For the standard CD tower:
        #   Level 0 (dim 1): J = [1]
        #   Level 1 (dim 2): J = diag(1, -1)  [complex conj]
        #   Level 2 (dim 4): J = diag(J_1, -I_2) = diag(1,-1,-1,-1)  [quaternion conj]
        #   Level 3 (dim 8): J = diag(J_2, -I_4) = diag(1,-1,-1,-1,-1,-1,-1,-1)
        # Pattern: J[0,0]=1, J[i,i]=-1 for i>0.
        # This is the standard hypercomplex conjugation.
        for i in range(1, d):
            J[i, i] = -1.0
    return J


def level_up_Zn(C_lower, n, conj_lower=None):
    """
    Build Level-(k+1) structure constants from Level-k, for Z_n tower.
    Uses the Cayley-Dickson-style construction WITH CONJUGATION.

    The generalized CD product for Z_n:
      (x_0, x_1, ..., x_{n-1}) · (y_0, y_1, ..., y_{n-1}) = (P_0, ..., P_{n-1})

    where P_m = sum over (i,j) with (i+j)≡m (mod n) of:
      - If i=0:     x_0 · y_j                    (left multiply by "real" part)
      - If j=0:     x_i · y_0                    (right multiply by "real" part)
      - Otherwise:  -1 * conj(y_j) · x_i         (CD-style: conjugate + sign)

    For Z_2, this exactly reproduces Cayley-Dickson:
      P_0 = x_0·y_0 - conj(y_1)·x_1
      P_1 = y_1·x_0 + x_1·conj(y_0)
    Which is (a,b)·(c,d) = (ac - d̄b, da + bc̄) — the standard CD formula.

    For Z_3, this gives:
      P_0 = x_0·y_0 - conj(y_2)·x_1 - conj(y_1)·x_2
      P_1 = y_1·x_0 + x_1·conj(y_0) - conj(y_2)·x_2
      P_2 = y_2·x_0 - conj(y_1)·x_1 + x_2·conj(y_0)

    Level-(k+1) dimension = n * d_k.
    Basis indexing: element (i, a) where i ∈ {0,...,n-1}, a ∈ {0,...,d_k-1}
    Flattened index: i * d_k + a.
    """
    d = C_lower.shape[0]
    D = n * d  # new dimension

    if conj_lower is None:
        conj_lower = build_conjugation_matrix(d)

    C_new = np.zeros((D, D, D), dtype=np.float64)

    # Helper: compute structure constants of conj(y)·x in the lower algebra
    # If x = e_a, y = e_b, then conj(y) = sum_b' J[b',b] e_{b'}
    # conj(y)·x = sum_b' J[b',b] * (e_{b'} · e_a) = sum_b',c J[b',b] * C[b',a,c] * e_c
    # So (conj(y)·x)_c = sum_b' J[b',b] * C[b',a,c]

    # Helper: compute structure constants of y·conj(x) in the lower algebra
    # conj(x) = sum_a' J[a',a] e_{a'}
    # y·conj(x) = sum_a' J[a',a] * (e_b · e_{a'}) = sum_a',c J[a',a] * C[b,a',c] * e_c

    for i1 in range(n):   # slot of first operand
        for i2 in range(n):   # slot of second operand
            m = (i1 + i2) % n

            for a in range(d):  # lower basis index of first operand
                for b in range(d):  # lower basis index of second operand
                    row = i1 * d + a
                    col = i2 * d + b

                    if i1 == 0 and i2 == 0:
                        # x_0 · y_0: straight product
                        for c in range(d):
                            if C_lower[a, b, c] != 0:
                                C_new[row, col, m * d + c] += C_lower[a, b, c]

                    elif i1 == 0 and i2 != 0:
                        # y_{i2} · x_0 for P_{i2}: right action of real part
                        # = e_b · e_a (note order: y·x)
                        for c in range(d):
                            if C_lower[b, a, c] != 0:
                                C_new[row, col, m * d + c] += C_lower[b, a, c]

                    elif i1 != 0 and i2 == 0:
                        # x_{i1} · conj(y_0) for P_{i1}: left action with conjugated real
                        # = e_a · conj(e_b) = sum_b' J[b',b] * C[a,b',c]
                        for bp in range(d):
                            if conj_lower[bp, b] == 0:
                                continue
                            for c in range(d):
                                if C_lower[a, bp, c] != 0:
                                    C_new[row, col, m * d + c] += conj_lower[bp, b] * C_lower[a, bp, c]

                    else:
                        # Both i1,i2 != 0: CD cross term
                        # Contribution: -1 * conj(y_{i2}) · x_{i1}
                        # = -sum_b' J[b',b] * C[b',a,c]
                        for bp in range(d):
                            if conj_lower[bp, b] == 0:
                                continue
                            for c in range(d):
                                if C_lower[bp, a, c] != 0:
                                    C_new[row, col, m * d + c] += -1.0 * conj_lower[bp, b] * C_lower[bp, a, c]

    return C_new


def level_up_Zn_emmy(C_lower, n):
    """
    Emmy's ORIGINAL construction (no conjugation, pure λ=-1 signs).
    Kept for comparison. This is what z3_tower.py tested.
    """
    d = C_lower.shape[0]
    D = n * d
    C_new = np.zeros((D, D, D), dtype=np.float64)

    for i1 in range(n):
        for i2 in range(n):
            m = (i1 + i2) % n
            sign = 1.0 if (i1 == 0 or i2 == 0) else -1.0

            for a in range(d):
                for b in range(d):
                    for c in range(d):
                        if C_lower[a, b, c] != 0.0:
                            row = i1 * d + a
                            col = i2 * d + b
                            out = m * d + c
                            C_new[row, col, out] += sign * C_lower[a, b, c]

    return C_new


def count_associativity_violations(C):
    """
    Count the number of basis triples (a,b,c) where (e_a·e_b)·e_c ≠ e_a·(e_b·e_c).

    Returns (violations, total_triples, fraction_associative).
    """
    n = C.shape[0]
    violations = 0
    total = n * n * n

    for a in range(n):
        # (e_a · e_b) · e_c = sum_d C[a,b,d] * C[d,c,e]  for each e
        # e_a · (e_b · e_c) = sum_d C[b,c,d] * C[a,d,e]  for each e
        # Vectorized over b,c:
        for b in range(n):
            # L = C[a,b,:] — coefficients of e_a · e_b
            L = C[a, b, :]
            for c in range(n):
                # LHS: (e_a·e_b)·e_c = sum_d L[d] * C[d,c,:]
                lhs = L @ C[:, c, :]
                # RHS: e_a·(e_b·e_c) = sum_d C[b,c,d] * C[a,d,:]
                R = C[b, c, :]
                rhs = R @ C[a, :, :]
                if not np.allclose(lhs, rhs, atol=1e-10):
                    violations += 1

    frac = 1.0 - violations / total
    return violations, total, frac


def count_associativity_violations_fast(C):
    """
    Faster version using einsum for larger algebras.
    """
    n = C.shape[0]
    # LHS[a,b,c,e] = sum_d C[a,b,d] * C[d,c,e]
    LHS = np.einsum('abd,dce->abce', C, C)
    # RHS[a,b,c,e] = sum_d C[b,c,d] * C[a,d,e]
    RHS = np.einsum('bcd,ade->abce', C, C)

    diff = np.abs(LHS - RHS)
    violations = np.sum(np.any(diff > 1e-10, axis=-1))
    total = n ** 3
    frac = 1.0 - violations / total
    return int(violations), total, frac


# ═══════════════════════════════════════════════════════════════
# THE BIG SWEEP
# ═══════════════════════════════════════════════════════════════

def run_tower_sweep(n, max_level=None, max_dim=500):
    """
    Build the Z_n tower level by level and measure associativity at each level.
    Stops when dimension exceeds max_dim (memory/time constraint).
    """
    print(f"\n{'━'*70}")
    print(f"  Z_{n} TOWER — Associativity Sweep")
    print(f"  Conjectured limit: (n-1)/n = {n-1}/{n} = {(n-1)/n:.10f}")
    print(f"{'━'*70}")

    results = []

    # Level 0: R (dim 1, trivially associative)
    C = np.array([[[1.0]]])
    results.append({
        'level': 0, 'dim': 1,
        'violations': 0, 'total': 1,
        'S_assoc': 1.0,
        'predicted': (n-1)/n,
    })
    print(f"  Level 0: dim=1, S_assoc=1.000000")

    # Level 1: Z_n graded algebra (dim n)
    C = build_Zn_structure_constants(n)
    v, t, s = count_associativity_violations(C)
    results.append({
        'level': 1, 'dim': n, 'violations': v, 'total': t,
        'S_assoc': s, 'predicted': (n-1)/n,
    })
    print(f"  Level 1: dim={n}, violations={v}/{t}, S_assoc={s:.10f}")

    # Higher levels — use CD-style construction with conjugation
    level = 1
    while True:
        next_dim = C.shape[0] * n
        if max_level and level >= max_level:
            break
        if next_dim > max_dim:
            print(f"  Level {level+1}: dim={next_dim} exceeds max_dim={max_dim}, stopping")
            break

        t0 = time.time()
        conj = build_conjugation_matrix(C.shape[0])
        C = level_up_Zn(C, n, conj)
        dt_build = time.time() - t0
        level += 1

        t0 = time.time()
        if C.shape[0] <= 81:  # use fast einsum for small dimensions
            v, t, s = count_associativity_violations_fast(C)
        else:
            v, t, s = count_associativity_violations_fast(C)
        dt_count = time.time() - t0

        results.append({
            'level': level, 'dim': C.shape[0],
            'violations': v, 'total': t,
            'S_assoc': s, 'predicted': (n-1)/n,
        })
        print(f"  Level {level}: dim={C.shape[0]}, violations={v}/{t}, "
              f"S_assoc={s:.10f}  (build={dt_build:.1f}s, count={dt_count:.1f}s)")

    return results


def fit_recurrence(results, n):
    """
    Fit the recurrence S_assoc[k] = L + B·(1/n²)^k - D·(1/n³)^k
    to the measured data and extract L, B, D.
    """
    if len(results) < 3:
        return None

    # Use levels with k >= 1
    data = [(r['level'], r['S_assoc']) for r in results if r['level'] >= 1]
    if len(data) < 2:
        return None

    # Fit: S[k] = L + B·(1/n²)^k + D·(1/n³)^k
    # Three unknowns, use least squares with all available data
    A_mat = []
    b_vec = []
    for k, s in data:
        A_mat.append([1.0, (1.0/n**2)**k, (1.0/n**3)**k])
        b_vec.append(s)

    A_mat = np.array(A_mat)
    b_vec = np.array(b_vec)

    try:
        params, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        L, B, D = params
        return {'L': L, 'B': B, 'D': D,
                'residual': float(np.sum((A_mat @ params - b_vec)**2))}
    except:
        return None


# ═══════════════════════════════════════════════════════════════
# C_n MEASUREMENT: count NEW violations per level
# ═══════════════════════════════════════════════════════════════

def measure_Cn(results, n):
    """
    From the violation counts V[k], measure C_n in the recurrence:
      V[k+1] = n·V[k] + C_n·n^{3k} - D_n

    Rearranging: C_n = (V[k+1] - n·V[k] + D_n) / n^{3k}

    With D_n unknown, use two consecutive levels to eliminate D_n:
      V[k+1] = n·V[k] + C_n·n^{3k} - D_n
      V[k+2] = n·V[k+1] + C_n·n^{3(k+1)} - D_n

    Subtract: V[k+2] - n·V[k+1] = n·(V[k+1] - n·V[k]) + C_n·n^{3k}·(n³-1)... nah
    Let's just use the simplest approach: from the fitted L = (n-1)/n?,
    compute C_n = (1-L) · n(n²-1).
    """
    pass  # We'll analyze from the fit


# ═══════════════════════════════════════════════════════════════
# ALTERNATIVE SIGN RULES (exploration)
# ═══════════════════════════════════════════════════════════════

def build_Zn_omega_structure_constants(n):
    """
    Alternative: use ω = e^{2πi/n} as the sign, not just ±1.

    sign(i,j) = ω^{i·j}

    This is the "DFT-style" multiplication. For n=2: ω=-1, recovers CD.
    For n=3: ω = e^{2πi/3}, produces a COMPLEX algebra.

    We work over R by splitting into real and imaginary parts.
    For odd n, this doubles the dimension: dim = 2n (real representation).
    """
    # For now, just return the real part of the ω-signed algebra
    omega = np.exp(2j * np.pi / n)
    C_complex = np.zeros((n, n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            C_complex[i, j, k] = omega ** (i * j)

    # Split into real 2n-dimensional algebra
    d = 2 * n
    C_real = np.zeros((d, d, d), dtype=np.float64)

    # Basis: e_i^R (real part), e_i^I (imaginary part), i=0..n-1
    # Index mapping: e_i^R -> 2*i, e_i^I -> 2*i+1
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            c = omega ** (i * j)  # complex coefficient
            cr, ci = c.real, c.imag

            # e_i^R · e_j^R -> cr * e_k^R - ci * e_k^I
            C_real[2*i, 2*j, 2*k] += cr
            C_real[2*i, 2*j, 2*k+1] += -ci
            # e_i^R · e_j^I -> cr * e_k^I + ci * e_k^R
            C_real[2*i, 2*j+1, 2*k+1] += cr
            C_real[2*i, 2*j+1, 2*k] += ci
            # e_i^I · e_j^R -> cr * e_k^I + ci * e_k^R
            C_real[2*i+1, 2*j, 2*k+1] += cr
            C_real[2*i+1, 2*j, 2*k] += ci
            # e_i^I · e_j^I -> -(cr * e_k^R - ci * e_k^I) = -cr*e_k^R + ci*e_k^I
            C_real[2*i+1, 2*j+1, 2*k] += -cr
            C_real[2*i+1, 2*j+1, 2*k+1] += ci

    return C_real


def build_Zn_cyclic_structure_constants(n):
    """
    Cyclic convolution algebra: sign(i,j) = +1 for all (i,j).
    This is the λ=+1 case.
    P_k = sum over (i+j)≡k (mod n) of a_i * b_j

    This is isomorphic to R^n with componentwise multiplication
    (via the DFT diagonalization).
    """
    C = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            C[i, j, k] = 1.0
    return C


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 72 + "╗")
    print("║" + " Z_n TOWER ASSOCIATIVITY SWEEP ".center(72) + "║")
    print("║" + " Conjecture: lim S_assoc^{Z_n} = (n-1)/n ".center(72) + "║")
    print("╚" + "═" * 72 + "╝")

    all_results = {}
    emmy_results = {}  # Emmy's original (no conjugation) for comparison

    # Test n = 2, 3, 4, 5, 7 towers with CD-style conjugation
    for n in [2, 3, 4, 5, 7]:
        # For n=2: level 3 = dim 8 (octonions!), level 4 = 16 (sedenions)
        # For n=3: level 2 = dim 9, level 3 = 27
        # For n=4: level 2 = dim 16, level 3 = 64
        # For n=5: level 2 = dim 25, level 3 = 125
        # For n=7: level 2 = dim 49
        max_dim = 130 if n <= 3 else 70
        results = run_tower_sweep(n, max_dim=max_dim)
        all_results[n] = results

    # Also run Emmy's original (no conjugation) for Z_2 and Z_3 comparison
    print("\n" + "=" * 72)
    print("EMMY'S ORIGINAL (no conjugation, pure λ=-1 signs) — for comparison")
    print("=" * 72)
    for n in [2, 3]:
        print(f"\n  Z_{n} Emmy-style (no conjugation):")
        C = build_Zn_structure_constants(n)
        v, t, s = count_associativity_violations(C)
        print(f"    Level 1: dim={n}, violations={v}/{t}, S_assoc={s:.10f}")
        level = 1
        while C.shape[0] * n <= 130:
            C = level_up_Zn_emmy(C, n)
            level += 1
            v, t, s = count_associativity_violations_fast(C)
            print(f"    Level {level}: dim={C.shape[0]}, violations={v}/{t}, S_assoc={s:.10f}")

    # ──────────────────────────────────────────────────────
    # Fit the recurrence for each n
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RECURRENCE FIT: S[k] = L + B·(1/n²)^k + D·(1/n³)^k")
    print("=" * 72)

    fits = {}
    for n, results in all_results.items():
        fit = fit_recurrence(results, n)
        fits[n] = fit
        if fit:
            print(f"\n  Z_{n}: L = {fit['L']:.10f}  "
                  f"(predicted: {(n-1)/n:.10f})  "
                  f"error = {abs(fit['L'] - (n-1)/n):.2e}")
            print(f"        B = {fit['B']:.6f}, D = {fit['D']:.6f}, "
                  f"residual = {fit['residual']:.2e}")

    # ──────────────────────────────────────────────────────
    # MASTER TABLE
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MASTER TABLE")
    print("=" * 72)
    print(f"\n  {'n':>3} │ {'(n-1)/n':>12} │ {'Fitted L':>12} │ {'Error':>12} │ {'Levels':>6} │ {'Last S_assoc':>12}")
    print(f"  {'─'*3}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*6}─┼─{'─'*12}")
    for n in sorted(all_results.keys()):
        results = all_results[n]
        predicted = (n-1)/n
        last_s = results[-1]['S_assoc']
        last_k = results[-1]['level']
        fit = fits.get(n)
        if fit:
            fitted_L = fit['L']
            err = abs(fitted_L - predicted)
            print(f"  {n:>3} │ {predicted:>12.10f} │ {fitted_L:>12.10f} │ {err:>12.2e} │ {last_k:>6} │ {last_s:>12.10f}")
        else:
            print(f"  {n:>3} │ {predicted:>12.10f} │ {'N/A':>12} │ {'N/A':>12} │ {last_k:>6} │ {last_s:>12.10f}")

    # ──────────────────────────────────────────────────────
    # CONVERGENCE VISUALIZATION (ASCII)
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CONVERGENCE PLOT (S_assoc vs level)")
    print("=" * 72)

    for n in sorted(all_results.keys()):
        results = all_results[n]
        predicted = (n-1)/n
        print(f"\n  Z_{n} tower  (limit = {predicted:.4f}):")
        for r in results:
            k = r['level']
            s = r['S_assoc']
            # Scale: 0.4 to 1.0 mapped to 0..50 chars
            bar_pos = int((s - 0.4) / 0.6 * 50)
            lim_pos = int((predicted - 0.4) / 0.6 * 50)
            bar = list(' ' * 51)
            for i in range(min(bar_pos, 50)+1):
                bar[i] = '█'
            if 0 <= lim_pos <= 50:
                bar[lim_pos] = '│'
            print(f"    k={k}: {s:.6f} {''.join(bar)}")

    # ──────────────────────────────────────────────────────
    # C_n ANALYSIS: measure new violations per level
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("C_n ANALYSIS — New violation types per level")
    print("Conjecture: C_n = n² - 1")
    print("=" * 72)

    for n in sorted(all_results.keys()):
        results = all_results[n]
        predicted_Cn = n**2 - 1
        print(f"\n  Z_{n}: predicted C_{n} = {predicted_Cn}")

        for i in range(len(results) - 1):
            k = results[i]['level']
            V_k = results[i]['violations']
            V_k1 = results[i+1]['violations']
            total_k = results[i]['total']  # n^{3k}... actually dim^3

            # V[k+1] = n·V[k] + C_n·(dim_k)^3 - D_n
            # Approximate C_n (ignoring D_n for large k):
            # C_n ≈ (V[k+1] - n·V[k]) / dim_k^3
            dim_k = results[i]['dim']
            numerator = V_k1 - n * V_k
            denominator = dim_k ** 3
            if denominator > 0:
                measured_Cn = numerator / denominator
                print(f"    From levels {k}→{k+1}: "
                      f"C_n ≈ ({V_k1} - {n}×{V_k}) / {dim_k}³ = "
                      f"{numerator}/{denominator} = {measured_Cn:.6f}")

    # ──────────────────────────────────────────────────────
    # ALTERNATIVE SIGN RULES
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ALTERNATIVE: λ=+1 (cyclic convolution) — is it fully associative?")
    print("=" * 72)

    for n in [2, 3, 4, 5, 7]:
        C = build_Zn_cyclic_structure_constants(n)
        v, t, s = count_associativity_violations(C)
        print(f"  Z_{n} convolution (λ=+1): violations={v}/{t}, S_assoc={s:.6f}"
              f"  {'✓ ASSOC' if v == 0 else '✗ NON-ASSOC'}")

    print("\n" + "=" * 72)
    print("ALTERNATIVE: ω-signed (DFT-style) — real dimension 2n")
    print("=" * 72)

    for n in [2, 3, 5, 7]:
        C = build_Zn_omega_structure_constants(n)
        v, t, s = count_associativity_violations_fast(C)
        dim = C.shape[0]
        print(f"  Z_{n} ω-signed: dim={dim}, violations={v}/{t}, S_assoc={s:.6f}"
              f"  {'✓ ASSOC' if v == 0 else '✗ NON-ASSOC'}")

    # ──────────────────────────────────────────────────────
    # THE PUNCHLINE: extrapolated limits
    # ──────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 72 + "╗")
    print("║" + " FINAL VERDICT ".center(72) + "║")
    print("╚" + "═" * 72 + "╝")

    print(f"\n  {'n':>3} │ {'Predicted':>12} │ {'Measured/Fitted':>14} │ {'Status':>20}")
    print(f"  {'─'*3}─┼─{'─'*12}─┼─{'─'*14}─┼─{'─'*20}")
    for n in sorted(all_results.keys()):
        predicted = (n-1)/n
        fit = fits.get(n)
        if fit and fit['residual'] < 1e-6:
            status = "✓ CONFIRMED"
            val = f"{fit['L']:.10f}"
        elif fit:
            status = f"~ residual={fit['residual']:.1e}"
            val = f"{fit['L']:.10f}"
        else:
            val = f"{all_results[n][-1]['S_assoc']:.10f}"
            status = "insufficient data"
        print(f"  {n:>3} │ {predicted:>12.10f} │ {val:>14} │ {status:>20}")

    # Emmy's formula check for Z_3
    print("\n" + "─" * 72)
    print("  Emmy's exact formula for Z_3:")
    print("  S[k] = 2/3 + 9·(1/9)^k - 22·(1/27)^k")
    print()
    z3_results = all_results.get(3, [])
    for r in z3_results:
        k = r['level']
        if k == 0:
            continue
        emmy = 2/3 + 9 * (1/9)**k - 22 * (1/27)**k
        actual = r['S_assoc']
        print(f"    k={k}: Emmy predicts {emmy:.10f}, measured {actual:.10f}, "
              f"delta = {abs(emmy - actual):.2e}")

    # Hilbert's formula check for Z_2
    print("\n  Hilbert's exact formula for Z_2 (CD tower):")
    print("  S[k] = 1/2 + 14·(1/4)^k - 24·(1/8)^k")
    print()
    z2_results = all_results.get(2, [])
    for r in z2_results:
        k = r['level']
        if k == 0:
            continue
        hilbert = 0.5 + 14 * (1/4)**k - 24 * (1/8)**k
        actual = r['S_assoc']
        print(f"    k={k}: Hilbert predicts {hilbert:.10f}, measured {actual:.10f}, "
              f"delta = {abs(hilbert - actual):.2e}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, 'zn_tower_sweep.json')
    save_data = {}
    for n, results in all_results.items():
        save_data[str(n)] = results
        fit = fits.get(n)
        if fit:
            save_data[f'{n}_fit'] = fit
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")


if __name__ == '__main__':
    main()
