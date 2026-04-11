"""
Gröbner / SAT attack v2: BitVector encoding over GF(3)
=======================================================
Key insight: encode each GF(3) element as 2 bits (BitVec(2)).
Multiplication table as direct lookup. No If-chains.

Uses Z3 BitVec for tight SAT encoding.
"""

import numpy as np
from itertools import product
from time import time
from z3 import (
    BitVec, BitVecVal, Solver, And, Or, sat, unsat,
    Extract, Concat, ZeroExt, If, Sum, set_param, BoolVal
)

N = 3
T = np.zeros((9, 9, 9), dtype=int)
for r, s, u in product(range(N), repeat=3):
    T[3*r+s, 3*s+u, 3*r+u] = 1

print("=" * 70)
print("SAT ATTACK v2: BITVECTOR ENCODING OVER GF(3)")
print("=" * 70)

set_param('parallel.enable', True)

def gf3_mul_bv(a, b):
    """Multiply two GF(3) elements (each BitVec(2), values 0/1/2).
    Returns BitVec(2).
    GF(3) multiplication: 0*x=0, 1*x=x, 2*x=2x mod 3.
    Table: (0,0)→0 (0,1)→0 (0,2)→0
           (1,0)→0 (1,1)→1 (1,2)→2
           (2,0)→0 (2,1)→2 (2,2)→1
    """
    zero = BitVecVal(0, 2)
    one = BitVecVal(1, 2)
    two = BitVecVal(2, 2)
    return If(a == zero, zero,
           If(b == zero, zero,
           If(a == b, one, two)))

def build_and_solve_bv(target_R, timeout_s=300):
    """Encode as pure SAT via BitVec."""
    print(f"\n--- R = {target_R} ({27*target_R} GF(3) vars) ---")
    t0 = time()
    s = Solver()
    s.set("timeout", timeout_s * 1000)
    
    # Variables: 2-bit bitvectors in {0,1,2}
    U = [[BitVec(f"u{r}_{i}", 2) for i in range(9)] for r in range(target_R)]
    V = [[BitVec(f"v{r}_{j}", 2) for j in range(9)] for r in range(target_R)]
    W = [[BitVec(f"w{r}_{k}", 2) for k in range(9)] for r in range(target_R)]
    
    bv3 = BitVecVal(3, 2)
    # Domain: exclude value 3 (bit pattern 11)
    for r in range(target_R):
        for i in range(9):
            s.add(U[r][i] != bv3)
            s.add(V[r][i] != bv3)
            s.add(W[r][i] != bv3)
    
    # Symmetry breaking: lexicographic ordering on first components
    if target_R >= 2:
        # u[0] >= u[1] on first nonzero element (weak)
        for r in range(min(target_R - 1, 5)):
            s.add(U[r][0] >= U[r+1][0])
    
    # Precompute triple products
    # For the sum constraint, we need to add R values in GF(3).
    # Addition in GF(3): a+b mod 3.
    # For BitVec(2), we need a custom adder.
    
    # Strategy: accumulate sum using wider bitvectors to avoid overflow,
    # then take mod 3 at the end.
    # Max sum of R terms each in {0,1,2} is 2R.
    # Need ceil(log2(2R+1)) bits. For R=23: 2*23=46, need 6 bits.
    
    import math
    sum_bits = max(4, math.ceil(math.log2(2 * target_R + 1)) + 1)
    
    print(f"  Sum accumulator: {sum_bits} bits (max sum = {2*target_R})")
    print(f"  Building constraints...")
    
    # For each (i,j,k), compute sum of u[r][i]*v[r][j]*w[r][k] mod 3
    constraints_added = 0
    
    # Only iterate over entries that matter
    # T has 27 nonzero entries (value 1) and 702 zero entries
    # All constraints are needed for correctness
    
    for i in range(9):
        for j in range(9):
            for k in range(9):
                target_val = int(T[i, j, k])
                
                # Build sum of products
                terms = []
                for r in range(target_R):
                    prod = gf3_mul_bv(gf3_mul_bv(U[r][i], V[r][j]), W[r][k])
                    terms.append(ZeroExt(sum_bits - 2, prod))
                
                total = terms[0]
                for t in terms[1:]:
                    total = total + t
                
                # total mod 3 == target_val
                # Use: total = 3*q + target_val
                # Encode as: total - target_val is divisible by 3
                # BitVec modular arithmetic: total % 3 in bitvector isn't great
                # Better: URem(total, 3) == target_val
                from z3 import URem
                tv = BitVecVal(target_val, sum_bits)
                three = BitVecVal(3, sum_bits)
                s.add(URem(total, three) == tv)
                
                constraints_added += 1
    
    t_build = time() - t0
    print(f"  Built in {t_build:.1f}s. {constraints_added} constraints.")
    print(f"  Solving...")
    
    result = s.check()
    t_total = time() - t0
    
    if result == sat:
        print(f"  *** SAT in {t_total:.1f}s ***")
        m = s.model()
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
            print(f"  Verification FAILED")
            return "ERROR", t_total, None
    elif result == unsat:
        print(f"  UNSAT in {t_total:.1f}s → R_{{GF(3)}} > {target_R}")
        return "UNSAT", t_total, None
    else:
        print(f"  TIMEOUT in {t_total:.1f}s")
        return "TIMEOUT", t_total, None


# ================================================================
# Strategy: start from R=23 (should be SAT since standard R=27
# works and Smirnov R=23 is likely 3-integral), then go down.
# ================================================================

print("\nPhase 1: Calibrate with R=9 (should be UNSAT, tensor rank ≥ 19)")
status9, t9, _ = build_and_solve_bv(9, timeout_s=120)

# If R=9 resolves quickly, try the interesting range
if status9 != "TIMEOUT":
    print(f"\nR=9 resolved in {t9:.1f}s — solver is working.")
    print("Proceeding to interesting range.")
else:
    print(f"\nR=9 timed out. Trying R=23 (expected SAT)...")

# Go straight to R=23, which is where we expect the known upper bound
print("\nPhase 2: R=23 (Smirnov bound — should be SAT)")
status23, t23, sol23 = build_and_solve_bv(23, timeout_s=600)

if status23 == "SAT":
    print("\n*** R=23 is achievable over GF(3). Now searching for R < 23... ***")
    for R in [22, 21, 20, 19]:
        status, t, sol = build_and_solve_bv(R, timeout_s=600)
        if status == "SAT":
            print(f"\n{'='*70}")
            print(f"*** R_{{GF(3)}} ≤ {R} ***")
            if R < 23:
                print("*** CHARACTERISTIC 3 LOWERS THE RANK ***")
            U, V, W = sol
            np.savez(f"outputs/gf3_rank{R}_decomp.npz", U=U, V=V, W=W)
            for r in range(R):
                print(f"  t{r}: u={list(U[r])} v={list(V[r])} w={list(W[r])}")
            print(f"{'='*70}")
            # Keep going to find minimum
        elif status == "UNSAT":
            print(f"\n*** R_{{GF(3)}} > {R}: PROVEN over GF(3) ***")
            break
        else:
            print(f"\n  R={R}: inconclusive")
            continue
elif status23 == "UNSAT":
    print("\n*** R_{GF(3)} > 23: CHARACTERISTIC 3 IS AN OBSTRUCTION ***")
    print("This would be a major result. Checking R=24, 25...")
    for R in [24, 25, 26, 27]:
        status, t, sol = build_and_solve_bv(R, timeout_s=300)
        if status == "SAT":
            print(f"\n  R_{{GF(3)}} ≤ {R}")
            break
else:
    print("\n  R=23 timed out. Problem may be too hard for direct SAT.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
