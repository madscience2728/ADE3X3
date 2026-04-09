"""
Analytical rank-precision table for T_matmul (3x3 matrix multiplication).

For each R, computes:
  Lower bound on ε*(R): from mode-unfolding SVDs (Eckart-Young theorem)
    Any CP rank-R approximation T_approx satisfies
      ||T - T_approx||_F >= max_mode ||T_(mode) - best rank-R matrix||_F
    (because the mode-R matricization of a CP rank-R tensor has matrix rank <= R)

  Upper bound on ε*(R): standard basis truncation = sqrt(27 - R)
    Achieved by keeping any R of the 27 orthogonal standard terms.

  Known exact points from literature:
    R >= 23: exact decomposition exists (Laderman 1976, and others)
    R = 21:  exact over C (Smirnov 2013 / AlphaTensor 2022)
    R = 27:  trivial

  Precision required:
    b*(R) = log2(3R / ε*(R))
"""

import numpy as np

# ── Build T_matmul ──────────────────────────────────────────────────────────
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_norm = float(np.linalg.norm(T))   # sqrt(27)

# ── Mode unfoldings ──────────────────────────────────────────────────────────
# Mode-n unfolding: reshape tensor to (dim_n, prod of other dims)
def unfold(T, mode):
    dims = list(range(T.ndim))
    dims.remove(mode)
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))

modes = [unfold(T, 0), unfold(T, 1), unfold(T, 2)]  # each 9 x 81

# SVD of each mode unfolding
svds = []
for i, M in enumerate(modes):
    sv = np.linalg.svd(M, compute_uv=False)
    svds.append(sv)
    print(f"Mode-{i} unfolding shape: {M.shape}, nonzero SVs: {np.sum(sv > 1e-10)}")
    print(f"  singular values: {np.round(sv, 6)}")

print()

# Eckart-Young lower bound: for CP rank-R, matrix rank of each unfolding <= R
# => ||T - T_approx||_F >= ||T_(mode) - best_rank_R_matrix||_F
#    = sqrt(sum of squared SVs beyond index R)
def eckart_young_lb(svs, R):
    """Lower bound from one set of singular values."""
    sv = np.sort(svs)[::-1]
    tail = sv[R:] if R < len(sv) else np.array([])
    return float(np.linalg.norm(tail))

# Best lower bound across all three modes
def best_lb(R):
    return max(eckart_young_lb(sv, R) for sv in svds)

# Upper bound: standard basis truncation
def upper_std(R):
    return float(np.sqrt(max(0, 27 - R)))

# Known exact ranks (ε=0)
EXACT_REAL    = set(range(23, 28))   # R>=23 exact over R (Laderman etc.)
EXACT_COMPLEX = {21, 22}             # R=21,22 exact over C (Smirnov/AlphaTensor)
UNKNOWN       = set(range(19, 21))   # R=19,20: open question over R

def prec_label(b):
    if b is None: return '—            '
    if b <= 0:    return 'trivial      '
    if b <= 4:    return 'int4  (b≤4)  '
    if b <= 7:    return 'bfloat16(b≤7)'
    if b <= 8:    return 'int8  (b≤8)  '
    if b <= 10:   return 'float16(b≤10)'
    if b <= 23:   return 'float32(b≤23)'
    return f'need b≥{b:.1f}   '

SEP = "=" * 125
print(SEP)
print("  ANALYTICAL RANK-PRECISION TABLE  |  3×3 matmul tensor  |  ||T||_F = √27 ≈ 5.196")
print(SEP)
header = (f"{'R':>3} | {'lb (EY)':>9} | {'ub (std)':>9} | "
          f"{'b*(ub)':>8} | {'b*(lb)':>8} | {'uncertainty':>14} | "
          f"{'ub precision':>14} | {'lb precision':>14} | notes")
print(header)
print("-" * 125)

for R in range(1, 28):
    lb  = best_lb(R)
    ub  = upper_std(R)

    # Exact override
    if R in EXACT_REAL:
        lb = 0.0; ub = 0.0
    elif R in EXACT_COMPLEX:
        lb = 0.0  # exact over C, unknown over R -> lb=0 is wrong actually
        # Over reals: we don't know. Use EY lb, no override for ub.
        lb = best_lb(R)
        # But ub from standard basis still applies
        ub = upper_std(R)

    b_ub = float(np.log2(3*R / ub)) if ub > 1e-12 else 0.0
    b_lb = float(np.log2(3*R / lb)) if lb > 1e-12 else None

    if ub < 1e-12:
        ub_s = '     exact'
        b_ub_s = '    exact'
    else:
        ub_s = f'{ub:9.4f}'
        b_ub_s = f'{b_ub:8.3f}'

    if lb < 1e-12:
        lb_s = '     0.000'
        b_lb_s = '        —'
    else:
        lb_s = f'{lb:9.4f}'
        b_lb_s = f'{b_lb:8.3f}' if b_lb else '        —'

    # Uncertainty interval on ε
    if lb < 1e-12 and ub < 1e-12:
        uncert = '    exact=0    '
    elif lb < 1e-12:
        uncert = f'[0, {ub:.4f}]    '
    else:
        uncert = f'[{lb:.3f}, {ub:.3f}]'

    notes = ''
    if R in EXACT_REAL:    notes = 'exact/real (Laderman+)'
    elif R == 22:          notes = 'exact/C (AlphaTensor) | real unknown'
    elif R == 21:          notes = 'exact/C (Smirnov/AlphaTensor) | real unknown'
    elif R == 20:          notes = '*** open question over R ***'
    elif R == 19:          notes = '*** open question over R ***'
    elif R == 18:          notes = 'provably approximate'
    elif R < 18:           notes = f'provably approximate (rank lb > {R})'

    print(f"{R:3d} | {lb_s} | {ub_s} | {b_ub_s} | {b_lb_s} | {uncert:>14} | "
          f"{prec_label(b_ub):>14} | {prec_label(b_lb):>14} | {notes}")

print(SEP)
print()
print("COLUMN MEANINGS:")
print("  lb (EY)      = Eckart-Young lower bound on ε*(R) from mode-unfolding SVDs")
print("                 Any CP rank-R approx MUST have error >= this value (by E-Y theorem)")
print("  ub (std)     = Upper bound = √(27-R), achieved by standard basis truncation")
print("  b*(ub)       = Bits needed for upper bound: log2(3R / ub)")
print("  b*(lb)       = Bits needed for lower bound: log2(3R / lb)  [best-case scenario]")
print("  uncertainty  = [lb, ub], the provably correct interval for ε*(R)")
print()
print("INTERPRETATION:")
print("  If lb ≈ ub: ε*(R) is tightly known analytically, no optimizer needed")
print("  If lb << ub: there is room for optimizer to beat standard basis")
print("  At R=19: lb from EY tells us the information-theoretic floor on residual")
print()
print("MODE UNFOLDING SVD SPECTRA (used for EY bounds):")
for i, sv in enumerate(svds):
    print(f"  Mode-{i}: {np.round(sv[sv > 1e-10], 4)}")
print()

# Print the EY bounds specifically for key ranks
print("KEY R VALUES — EY LOWER BOUNDS IN DETAIL:")
print(f"{'R':>3}  {'mode0_tail':>12}  {'mode1_tail':>12}  {'mode2_tail':>12}  {'best_lb':>10}")
for R in [9, 13, 17, 18, 19, 20, 21, 22, 23]:
    lb0 = eckart_young_lb(svds[0], R)
    lb1 = eckart_young_lb(svds[1], R)
    lb2 = eckart_young_lb(svds[2], R)
    best = max(lb0, lb1, lb2)
    print(f"{R:3d}  {lb0:12.6f}  {lb1:12.6f}  {lb2:12.6f}  {best:10.6f}")
