"""
The Free-Bits Worksheet for 3x3 Matrix Multiplication
======================================================

total_error  =  ε(R)  +  3·R·2^{-b}·M²

free_bits(R, ε, τ)  =  log₂(27·(τ − ε(R))  /  (R·τ))

τ_min(R)  =  27·ε(R) / (27 − R)   ← minimum τ for trick to break even
"""
import numpy as np

# Current best known ε(R): Frobenius residual
# Standard basis = sqrt(27-R); optimizer finds may be lower
epsilon_best = {
    # exact algorithms (real field)
    27: 0.0,
    26: 0.0,   # trivially drop 1 term... wait, no. sqrt(1)=1.0 for standard basis
    # Let's use standard basis for all unless we have a confirmed better value
}

# Standard basis ε(R) = sqrt(27-R)
# Optimizer best (Frobenius) from our measurements:
optimizer_frob = {
    19: 1.9269,   # slp_best_at_0.074.json — fitness was max-entry, Frob = 1.927
    22: None,     # need to check
}

def eps_std(R):
    return float(np.sqrt(max(0, 27 - R)))

def eps_best(R):
    if R >= 23: return 0.0   # exact real decompositions known
    opt = optimizer_frob.get(R)
    return min(eps_std(R), opt) if opt is not None else eps_std(R)

def b_needed(R, eps, tau):
    """Minimum bits needed: b >= log2(3R / (tau - eps))"""
    headroom = tau - eps
    if headroom <= 0:
        return float('inf')
    return float(np.log2(3 * R / headroom))

def b_exact_27(tau):
    """Bits needed for exact 27-term at tolerance tau: log2(81/tau)"""
    return float(np.log2(81 / tau))

def free_bits(R, eps, tau):
    """Bits saved vs exact 27: b_exact - b_needed"""
    bn = b_needed(R, eps, tau)
    if np.isinf(bn): return -float('inf')
    return b_exact_27(tau) - bn

def tau_min(R, eps):
    """Minimum tau for break-even: 27*eps / (27-R)"""
    if R >= 27: return 0.0
    return 27 * eps / (27 - R)

SEP = "=" * 100

# ── TABLE 1: The fundamental formula at selected tau values ─────────────────
print(SEP)
print("TABLE 1 — free_bits at various target tolerances τ")
print("  free_bits > 0 → rank-R trick SAVES precision vs exact 27-term at that τ")
print("  free_bits < 0 → trick COSTS more precision")
print(SEP)

taus = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
print(f"\n  {'R':>3}  {'ε(R)':>8}  {'τ_min':>8}", end="")
for tau in taus:
    print(f"  τ={tau:>4}", end="")
print()
print("  " + "-"*90)

for R in [7, 9, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27]:
    eps = eps_best(R)
    t_min = tau_min(R, eps)
    print(f"  {R:>3}  {eps:>8.4f}  {t_min:>8.3f}", end="")
    for tau in taus:
        fb = free_bits(R, eps, tau)
        if np.isinf(fb) and fb < 0:
            s = " —inf "
        elif abs(fb) > 99:
            s = f" {fb:>+6.0f}"
        else:
            s = f" {fb:>+6.2f}"
        print(f"  {s:>6}", end="")
    print()

# ── TABLE 2: The target for each precision tier ──────────────────────────────
print()
print(SEP)
print("TABLE 2 — At each R, what ε*(R) is needed to close at each precision tier?")
print()
print("  Condition:  ε*(R)  ≤  τ · (1 − R/27)  where τ = budget_frob = 3·R·2^{-b}")
print("  Simplified: ε*(R)  ≤  3·R·2^{-b} · (1 − R/27)  =  3·(27−R)·2^{-b} / 27 * R/(something)")
print()
print("  Correct derivation from free_bits = 0:")
print("    27·(τ − ε) = R·τ  →  ε = τ·(27−R)/27  =  3·R·2^{-b}·(27−R)/27")
print()

precisions = [(4, 'int4   '), (7, 'bfloat16'), (8, 'int8   '), (10, 'float16'), (16, 'float16 IEEE'), (23, 'float32')]

print(f"  {'R':>3}  {'ε_std':>8}  {'ε_opt':>8}", end="")
for b, name in precisions:
    print(f"  {name}(b={b})", end="")
print()
print("  " + "-"*110)

for R in [7, 9, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27]:
    e_std = eps_std(R)
    e_opt = eps_best(R)
    print(f"  {R:>3}  {e_std:>8.4f}  {e_opt:>8.4f}", end="")
    for b, name in precisions:
        tau = 3 * R * 2**(-b)       # budget at this precision
        eps_target = tau * (27 - R) / 27  # target ε for break-even
        # Mark whether current best already meets it
        meets = e_opt <= eps_target
        marker = "✓" if meets else "✗"
        print(f"  {marker} ε≤{eps_target:.5f} ", end="")
    print()

# ── TABLE 3: The specific R=19 worksheet ─────────────────────────────────────
print()
print(SEP)
print("TABLE 3 — R=19 WORKSHEET (pen-and-paper verification)")
print(SEP)

R = 19
eps_current = 1.9269
eps_std_19 = eps_std(19)

print(f"""
Given:
  R = {R} multiplications
  ‖T_matmul‖_F = √27 = {np.sqrt(27):.4f}
  ε_std(19) = √(27-19) = √8 = {eps_std_19:.4f}   ← standard basis drop (any 8 terms removed)
  ε_opt(19) = {eps_current:.4f}                    ← best known optimizer (Frobenius)

The total error formula:
  total_error(b) = ε(R) + 3·R·2^(-b)
                 = ε(R) + 57·2^(-b)

For ε = ε_std = {eps_std_19:.4f} (standard basis, drop 8 terms):
""")

for b, name in precisions:
    budget = 57 * 2**(-b)
    total  = eps_std_19 + budget
    print(f"  b={b:>2} ({name}):  total = {eps_std_19:.4f} + {budget:.5f} = {total:.4f}")

print(f"""
For ε = ε_opt = {eps_current:.4f} (best optimizer candidate):
""")
for b, name in precisions:
    budget = 57 * 2**(-b)
    total = eps_current + budget
    print(f"  b={b:>2} ({name}):  total = {eps_current:.4f} + {budget:.5f} = {total:.4f}")

print(f"""
For EXACT (ε = 0, R=23, Laderman algorithm):
""")
for b, name in precisions:
    budget = 3 * 23 * 2**(-b)
    total = budget
    print(f"  b={b:>2} ({name}):  total = 0.0000 + {budget:.5f} = {total:.5f}")

print(f"""
TARGET: ε*(19) needed for free_bits = 0 at each precision tier:
  (i.e. break-even with exact 27-term at same precision)
""")
for b, name in precisions:
    tau = 3 * R * 2**(-b)
    eps_target = tau * (27 - R) / 27
    gap = eps_current / eps_target if eps_target > 0 else float('inf')
    print(f"  b={b:>2} ({name}):  need ε ≤ {eps_target:.6f}   current={eps_current:.4f}  gap={gap:.1f}×")

print(f"""
BOTTOM LINE at R=19:
  To beat exact-27 at bfloat16 (b=7): need ε ≤ {3*19*2**-7*(27-19)/27:.4f}
  Best known Frobenius residual:             {eps_current:.4f}
  Gap to close:                              {eps_current / (3*19*2**-7*(27-19)/27):.1f}×

  The 14× gap is the real target. Not a precision question — a decomposition quality question.
""")
