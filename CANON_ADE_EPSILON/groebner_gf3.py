"""
Gröbner basis / SAT attack on R_{GF(3)}(T_matmul(3))
=====================================================
Encode: does Σ_{r=1}^R u_r⊗v_r⊗w_r = T_matmul (mod 3) have a solution?

Use Z3 solver with variables in {0,1,2}.
Field equations x³=x are automatic since variables are enumerated.

Key trick: over GF(3), each variable has only 3 values.
With R terms and 27 variables per term (9+9+9), total = 27R variables.
729 constraints (one per tensor entry).

We use symmetry to reduce: fix first term and exploit S_3 permutation.
"""

import numpy as np
from itertools import product
from time import time
import sys

try:
    from z3 import (
        Int, Solver, And, Or, sat, unsat, If, 
        IntVector, Sum, set_param
    )
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    print("Z3 not available. Install with: pip install z3-solver")

N = 3
# Build T_matmul(3)
T = np.zeros((9, 9, 9), dtype=int)
for r, s, u in product(range(N), repeat=3):
    T[3*r+s, 3*s+u, 3*r+u] = 1

print("=" * 70)
print("GRÖBNER / SAT ATTACK ON R_{GF(3)}(T_matmul(3))")
print("=" * 70)
print(f"Target tensor: 9×9×9, nnz = {T.sum()}")
print()

if not HAS_Z3:
    sys.exit(1)

# Speed settings
set_param('parallel.enable', True)

def build_and_solve(target_R, timeout_s=300, use_symmetry=True):
    """
    Build Z3 model for rank-R decomposition over GF(3).
    Returns (sat/unsat/unknown, time, solution_if_sat).
    """
    print(f"\n--- R = {target_R} ---")
    t0 = time()
    
    s = Solver()
    s.set("timeout", timeout_s * 1000)  # milliseconds
    
    # Variables: u[r][i], v[r][j], w[r][k] for r=0..R-1, i,j,k=0..8
    # Each in {0, 1, 2}
    U = [[Int(f"u_{r}_{i}") for i in range(9)] for r in range(target_R)]
    V = [[Int(f"v_{r}_{j}") for j in range(9)] for r in range(target_R)]
    W = [[Int(f"w_{r}_{k}") for k in range(9)] for r in range(target_R)]
    
    # Domain constraints: each variable in {0, 1, 2}
    for r in range(target_R):
        for i in range(9):
            s.add(Or(U[r][i] == 0, U[r][i] == 1, U[r][i] == 2))
            s.add(Or(V[r][i] == 0, V[r][i] == 1, V[r][i] == 2))
            s.add(Or(W[r][i] == 0, W[r][i] == 1, W[r][i] == 2))
    
    # Symmetry breaking: order first few terms lexicographically
    if use_symmetry and target_R >= 2:
        # Fix u[0] ≤ u[1] ≤ ... lexicographically (first nonzero component)
        # Simple version: just fix u[0][0] >= u[1][0] to break some symmetry
        for r in range(min(target_R - 1, 3)):
            # Weak ordering on first component
            s.add(U[r][0] >= U[r+1][0])
    
    # Tensor constraints: Σ_r u[r][i]*v[r][j]*w[r][k] ≡ T[i][j][k] (mod 3)
    # 
    # Over GF(3), multiplication table:
    # 0*x = 0, 1*x = x, 2*x = 2x mod 3
    # We need: (Σ_r u_r[i]*v_r[j]*w_r[k]) mod 3 = T[i,j,k]
    #
    # Z3 approach: use integer arithmetic and mod-3 constraint.
    # For efficiency, encode the triple product u*v*w mod 3 using a lookup.
    
    def mul3(a, b):
        """a*b mod 3 for Z3 Int expressions in {0,1,2}."""
        # Direct: a*b can be 0,1,2,4. We need mod 3.
        # 0→0, 1→1, 2→2, 4→1
        return If(a == 0, 0, If(b == 0, 0, If(a == b, 1, 2)))
    
    def mul3_triple(a, b, c):
        """a*b*c mod 3."""
        return mul3(mul3(a, b), c)
    
    # Only constrain nonzero entries and a sample of zero entries
    # (729 constraints total — constrain all for correctness)
    
    # For tractability: encode sum mod 3
    # Σ_r (u_r[i]*v_r[j]*w_r[k]) mod 3 = T[i,j,k]
    #
    # Use auxiliary variables for partial products to speed up
    
    constraint_count = 0
    
    # Precompute triple products as auxiliary variables
    # P[r][i][j][k] = u[r][i]*v[r][j]*w[r][k] mod 3
    # But that's 9*9*9*R = 729R auxiliaries. For R=19: 13851. Manageable.
    
    print(f"  Building model: {27*target_R} primary vars, 729 constraints...")
    
    P = {}
    for r in range(target_R):
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    P[(r,i,j,k)] = mul3_triple(U[r][i], V[r][j], W[r][k])
    
    for i in range(9):
        for j in range(9):
            for k in range(9):
                # Sum of P[r][i][j][k] for r=0..R-1, mod 3
                # Since each P is in {0,1,2}, sum is in [0, 2R].
                # We need sum mod 3 = T[i,j,k].
                total = Sum([P[(r,i,j,k)] for r in range(target_R)])
                target_val = int(T[i, j, k])
                
                # total ≡ target_val (mod 3)
                # Encode: (total - target_val) % 3 == 0
                # Z3: ∃q. total - target_val == 3*q
                # Since total ∈ [0, 2R] and target ∈ {0,1}, q ∈ [-1, 2R/3]
                # Use modular arithmetic trick:
                remainder = Int(f"rem_{i}_{j}_{k}")
                s.add(total == 3 * remainder + target_val)
                # bound remainder
                s.add(remainder >= 0)
                s.add(remainder <= (2 * target_R) // 3 + 1)
                
                constraint_count += 1
    
    t_build = time() - t0
    print(f"  Model built in {t_build:.1f}s. {constraint_count} constraints.")
    print(f"  Solving (timeout={timeout_s}s)...")
    
    result = s.check()
    t_total = time() - t0
    
    if result == sat:
        print(f"  *** SAT *** in {t_total:.1f}s")
        m = s.model()
        # Extract solution
        sol_U = np.array([[m.evaluate(U[r][i]).as_long() for i in range(9)] 
                          for r in range(target_R)], dtype=np.int8)
        sol_V = np.array([[m.evaluate(V[r][j]).as_long() for j in range(9)] 
                          for r in range(target_R)], dtype=np.int8)
        sol_W = np.array([[m.evaluate(W[r][k]).as_long() for k in range(9)] 
                          for r in range(target_R)], dtype=np.int8)
        
        # Verify
        T_check = np.zeros((9, 9, 9), dtype=np.int64)
        for r in range(target_R):
            T_check += np.einsum('i,j,k->ijk', 
                                 sol_U[r].astype(np.int64),
                                 sol_V[r].astype(np.int64),
                                 sol_W[r].astype(np.int64))
        T_check = np.mod(T_check, 3).astype(np.int8)
        
        if np.array_equal(T_check, T.astype(np.int8)):
            print(f"  *** VERIFIED: R_{{GF(3)}} ≤ {target_R} ***")
            return "SAT", t_total, (sol_U, sol_V, sol_W)
        else:
            diff = np.count_nonzero(T_check - T.astype(np.int8))
            print(f"  VERIFICATION FAILED: {diff} mismatches")
            return "ERROR", t_total, None
    elif result == unsat:
        print(f"  UNSAT in {t_total:.1f}s")
        print(f"  *** R_{{GF(3)}} > {target_R} ***")
        return "UNSAT", t_total, None
    else:
        print(f"  UNKNOWN/TIMEOUT in {t_total:.1f}s")
        return "TIMEOUT", t_total, None


# ================================================================
# Phase 1: Quick feasibility. Try small R first to calibrate timing.
# ================================================================
print("\nPHASE 1: CALIBRATION")
print("=" * 70)
print("Testing small R values to gauge solver difficulty...")

# R=9 is the multilinear rank. Should be SAT quickly if it works.
# But tensor rank ≥ 19, so R=9 should be UNSAT.
# Let's start with R=9 with short timeout to see speed.

for R_test in [9, 14, 19]:
    timeout = 60 if R_test <= 14 else 300
    status, elapsed, sol = build_and_solve(R_test, timeout_s=timeout)
    
    if status == "SAT":
        print(f"\n!!! BREAKTHROUGH: R_{{GF(3)}} ≤ {R_test}")
        print("Decomposition found! Vectors:")
        U, V, W = sol
        for r in range(R_test):
            print(f"  Term {r}: u={list(U[r])}, v={list(V[r])}, w={list(W[r])}")
        break
    elif status == "UNSAT":
        print(f"\n  Confirmed: R_{{GF(3)}} > {R_test}")
    else:
        print(f"\n  R={R_test}: solver couldn't decide in {elapsed:.0f}s")
        if R_test == 9:
            print("  UNSAT at R=9 expected but might be slow. Adjusting strategy...")

# ================================================================
# Phase 2: Targeted search near known bounds
# ================================================================
print("\n\nPHASE 2: TARGETED SEARCH NEAR BOUNDS")
print("=" * 70)
print("Searching R=22 down to R=19 with 5min timeout each...")

for R_target in [23, 22, 21, 20, 19]:
    status, elapsed, sol = build_and_solve(R_target, timeout_s=300)
    
    if status == "SAT":
        print(f"\n{'='*70}")
        print(f"RESULT: R_{{GF(3)}}(T_matmul(3)) ≤ {R_target}")
        print(f"{'='*70}")
        if R_target < 23:
            print("*** CHARACTERISTIC-3 GIVES LOWER RANK! ***")
            print("*** THE KETH-VARAI WERE RIGHT. ***")
        U, V, W = sol
        # Save to file
        np.savez(f"outputs/gf3_rank{R_target}_decomposition.npz",
                 U=U, V=V, W=W)
        print(f"Saved to outputs/gf3_rank{R_target}_decomposition.npz")
        break
    elif status == "UNSAT":
        print(f"  R_{{GF(3)}} > {R_target}: PROVEN")
        if R_target == 23:
            print("  *** R_{GF(3)} > 23: CHARACTERISTIC 3 MAKES IT HARDER! ***")
        continue
    else:
        print(f"  R={R_target}: inconclusive (timeout)")
        continue

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
Over GF(3), the tensor rank decomposition problem becomes a finite
constraint satisfaction problem: 729 equations in {0,1,2}-valued variables.

The Z3 SMT solver attacks this directly. Results above show which
ranks are achievable, unachievable, or inconclusive over GF(3).

If R_{GF(3)} < 23: characteristic-dependent tensor rank is REAL.
If R_{GF(3)} = 23: the gap 19-23 is field-independent.
If R_{GF(3)} > 23: characteristic 3 is an OBSTRUCTION.
""")
