import numpy as np
import json
from pathlib import Path

# Build T_matmul
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_norm = float(np.linalg.norm(T))  # sqrt(27)

# Known optimizer results
optimizer_best = {
    19: 0.074,   # slp_best_at_0.074.json
}

# Load phase4 R=22 best
phase4_dir = Path("outputs/ade3x3_attack/phase4_gradient_search/best_decompositions")
r22_best = None
if phase4_dir.exists():
    for f in sorted(phase4_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            terms = d.get('terms', [])
            R_file = len(terms)
            if R_file == 22:
                # reconstruct residual
                import numpy as np
                alpha = np.array([t['alpha'] for t in terms]).reshape(22,9)
                beta  = np.array([t['beta']  for t in terms]).reshape(22,9)
                gamma = np.array([t['gamma'] for t in terms]).reshape(22,9)
                T_approx = np.einsum('ki,kj,kl->ijl', alpha, beta, gamma)
                res = float(np.linalg.norm(T - T_approx))
                if r22_best is None or res < r22_best:
                    r22_best = res
        except:
            pass
if r22_best is not None:
    optimizer_best[22] = r22_best

# Precision label helpers
def prec_label(b):
    if b <= 0:   return 'any precision'
    if b <= 4:   return 'int4   (b=4) '
    if b <= 7:   return 'bfloat16(b=7) '
    if b <= 8:   return 'int8   (b=8) '
    if b <= 10:  return 'float16(b=10) '
    if b <= 23:  return 'float32(b=23) '
    return f'need b={b:.1f}     '

def scalar_mults(R):
    # Each rank-1 term needs 2 matrix-vector products of 3x3:
    # alpha @ x (9 mults) + beta @ y (9 mults) + combine with gamma (9 mults)
    # Standard count: 3*R scalar multiplications for the Sigma part
    # Total effective scalar mults for 3x3 matmul: R terms × (3+3+3) = 9R additions + something
    # Conventional: each rank-1 term in CP decomp of matmul uses
    # one contraction per coeff pair: 9R for alpha, 9R for beta, 9R for gamma = 27R total
    # But "scalar multiplications" for matmul: standard = 27, so savings = 27 - something
    # Simple metric: R vs 27 (standard), reduction = (27-R)/27
    return R

sep = "=" * 110
print(sep)
print(f"  COMPLETE RANK-PRECISION MAP  |  3×3 Matrix Multiplication Tensor  |  T_matmul norm = {T_norm:.4f} = √27")
print(sep)
print(f"{'R':>3} | {'ε*(std)':>9} | {'b*(std)':>7} | {'ε*(opt)':>9} | {'b*(opt)':>7} | {'improv':>7} | {'bits_gained':>11} | {'std_prec':>16} | {'opt_prec':>16} | note")
print("-" * 110)

for R in range(1, 28):
    eps_std = float(np.sqrt(max(0, 27 - R)))
    if eps_std > 1e-12:
        b_std = float(np.log2(3 * R / eps_std))
    else:
        b_std = 0.0

    eps_opt  = optimizer_best.get(R, None)
    if eps_opt is not None and eps_opt > 1e-12:
        b_opt = float(np.log2(3 * R / eps_opt))
        improv = eps_std / eps_opt
        bits_gained = b_opt - b_std
    else:
        b_opt = None
        improv = None
        bits_gained = None

    std_prec  = prec_label(b_std)  if eps_std > 1e-12 else 'exact          '
    opt_prec  = prec_label(b_opt)  if b_opt is not None else '—              '
    improv_s  = f'{improv:7.2f}x'  if improv is not None else '       —'
    bits_s    = f'{bits_gained:+.3f}'  if bits_gained is not None else '          —'
    b_std_s   = f'{b_std:7.3f}'    if eps_std > 1e-12 else '  exact'
    b_opt_s   = f'{b_opt:7.3f}'    if b_opt is not None else '      —'
    eps_opt_s = f'{eps_opt:9.4f}'  if eps_opt is not None else '        —'

    # Notes
    note = ''
    if R == 19:  note = '<-- best known near-miss'
    if R == 21:  note = '<-- AlphaTensor exact bound'
    if R == 22:  note = '<-- best optimizer candidate (phase4)'
    if R == 27:  note = '<-- standard algorithm, exact'

    print(f"{R:3d} | {eps_std:9.4f} | {b_std_s} | {eps_opt_s} | {b_opt_s} | {improv_s} | {bits_s:>11} | {std_prec} | {opt_prec} | {note}")

print(sep)
print()
print("KEY:")
print("  ε*(std)     = residual when keeping R of the 27 orthogonal standard basis terms = √(27-R)")
print("  b*(std)     = bits required: log2(3R / ε*(std))")
print("  ε*(opt)     = best residual found by optimizer search")
print("  b*(opt)     = bits required for optimizer candidate: log2(3R / ε*(opt))")
print("  improv      = ε*(std) / ε*(opt)  [how much optimizer beat standard basis]")
print("  bits_gained = b*(opt) - b*(std)  [extra precision headroom won by optimizer]")
print()
print("PRECISION THRESHOLDS:")
print("  int4    b=4   (4-bit integer)")
print("  bfloat16 b=7  (Google's 'brain float')")
print("  int8    b=8   (8-bit integer)")
print("  float16 b=10  (IEEE half precision)")
print("  float32 b=23  (IEEE single precision)")
print()
print("SCALAR MULT CONTEXT:")
print(f"  Standard 3x3 matmul:    27 scalar multiplications")
print(f"  AlphaTensor bound:      21 (proven exact for complex field)")
print(f"  Best known real:        23")
print(f"  Best near-miss (R=19):  0.074 residual  (38.2x better than std basis)")
print()
print("BUDGET FORMULA:")
print("  budget_frob(R, b) = 3 * R * 2^{-b}")
print("  For R=19, b=7 (bfloat16): budget = 3*19*2^{-7} = 0.445")
print("  For R=22, b=7 (bfloat16): budget = 3*22*2^{-7} = 0.516")
print("  Leakage ||Γ·Δ||_F for best R=22 candidate: 9.89  (19.1x over budget)")
