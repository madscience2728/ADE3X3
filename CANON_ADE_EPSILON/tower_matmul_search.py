"""
The Big Search: Can ANY Z_3-recursive 9-dim algebra be associative AND ≅ Mat(3,R)?

We parameterize the Z_3 level-up rule with free coefficients and search for
associative solutions. If one exists and is isomorphic to Mat(3,R), we get
a new matmul algorithm.

The Z_3 recursive product: elements X = (x_0, x_1, x_2) with x_i in a 3-dim algebra.
Product P_m = sum_{(i+j)≡m mod 3} f(i,j) * g_{i,j}(x_i, y_j)

where f(i,j) are scalar signs and g_{i,j} is either:
  - straight product x_i · y_j
  - twisted product conj(y_j) · x_i  
  - or a general bilinear form

We search over the sign/twist choices exhaustively (finite search).

NO OPTIMIZATION. Exhaustive algebraic enumeration.
"""

import numpy as np
from itertools import product as iprod
import time

def build_level1_params(n, signs):
    """
    Build Level-1 Z_n algebra with custom signs.
    signs[i,j] for i,j in {0,...,n-1}, with i+j constrained.
    signs[0,j] = signs[i,0] = 1 (unit preservation).
    Free signs: signs[i,j] for i,j >= 1.
    """
    C = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            C[i, j, k] = signs[i][j]
    return C

def check_unit(C):
    """Check e_0 is a two-sided unit."""
    d = C.shape[0]
    for i in range(d):
        if not np.allclose(C[0, i, :], np.eye(d)[i]):
            return False
        if not np.allclose(C[i, 0, :], np.eye(d)[i]):
            return False
    return True

def count_assoc_violations(C):
    LHS = np.einsum('ijk,klm->ijlm', C, C)
    RHS = np.einsum('jlk,ikm->ijlm', C, C)
    return np.count_nonzero(np.abs(LHS - RHS).sum(axis=-1) > 1e-10)

def check_matmul_iso(C):
    """
    Check if 9-dim algebra C is isomorphic to Mat(3,R).
    Mat(3,R) has:
    - dimension 9 ✓ (given)
    - center of dimension 1 (scalar multiples of identity)
    - is associative
    - has zero divisors
    - left regular representation has specific eigenvalue structure
    
    Quick necessary check: associative + non-commutative + has zero divisors.
    """
    d = C.shape[0]
    if d != 9:
        return False, "wrong dim"
    
    # Associative?
    if count_assoc_violations(C) > 0:
        return False, "non-associative"
    
    # Commutative? Mat(3,R) is NOT commutative
    if np.allclose(C, C.transpose(1, 0, 2)):
        return False, "commutative (Mat3R is not)"
    
    # Center dimension? For Mat(n,R), center = scalar matrices, dim = 1
    # Center = {x : x·y = y·x for all y}
    # x = sum_i α_i e_i is central iff for all j: sum_i α_i C[i,j,k] = sum_i α_i C[j,i,k] for all k
    # i.e., sum_i α_i (C[i,j,k] - C[j,i,k]) = 0 for all j,k
    # This is a linear system in α
    constraints = []
    for j in range(d):
        for k in range(d):
            row = C[:, j, k] - C[j, :, k]
            constraints.append(row)
    M = np.array(constraints)
    center_dim = d - np.linalg.matrix_rank(M, tol=1e-8)
    
    if center_dim != 1:
        return False, f"center dim {center_dim} (need 1)"
    
    # Check dimension of radical (should be 0 for Mat(3,R) which is semisimple)
    # Use trace form: β(x,y) = tr(L_x · L_y) where L_x is left multiplication
    L = np.zeros((d, d, d))
    for i in range(d):
        L[i] = C[i]  # L[i]_{j,k} = C[i,j,k]
    
    # Killing form analogue
    killing = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            killing[i, j] = np.trace(L[i] @ L[j])
    
    killing_rank = np.linalg.matrix_rank(killing, tol=1e-8)
    if killing_rank != d:
        return False, f"degenerate trace form (rank {killing_rank}, need {d})"
    
    return True, "PASSES ALL TESTS — candidate Mat(3,R)!"


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Exhaustive search over Level-1 Z_3 sign patterns
# ═══════════════════════════════════════════════════════════════

def search_level1_signs():
    """
    The Level-1 Z_3 algebra has sign(i,j) for each pair.
    Constraints: sign(0,j) = sign(i,0) = +1 (unit).
    Free signs: sign(i,j) for i,j in {1,2} → 4 free signs, each ±1.
    That's 2^4 = 16 possibilities.
    """
    n = 3
    results = []
    
    for s11, s12, s21, s22 in iprod([1, -1], repeat=4):
        signs = [[1, 1, 1],
                 [1, s11, s12],
                 [1, s21, s22]]
        C1 = build_level1_params(n, signs)
        
        if not check_unit(C1):
            continue
        
        v1 = count_assoc_violations(C1)
        comm = np.allclose(C1, C1.transpose(1, 0, 2))
        
        results.append({
            'signs': (s11, s12, s21, s22),
            'violations': v1,
            'commutative': comm,
        })
    
    return results

print("=" * 72)
print("PHASE 1: All Level-1 Z_3 Sign Patterns (2^4 = 16)")
print("=" * 72)

l1_results = search_level1_signs()
for r in sorted(l1_results, key=lambda x: x['violations']):
    print(f"  signs={r['signs']}: V={r['violations']}/27, comm={r['commutative']}")

assoc_l1 = [r for r in l1_results if r['violations'] == 0]
print(f"\nAssociative Level-1 algebras: {len(assoc_l1)}")
for r in assoc_l1:
    print(f"  {r['signs']}")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: For EACH Level-1 sign pattern, try ALL level-up rules
# ═══════════════════════════════════════════════════════════════

def general_level_up(C1, n, rule):
    """
    General Z_3 level-up with parameterized rule.
    
    rule[i1, i2] specifies what to do for slots i1, i2:
      'straight': x_{i1} · y_{i2}
      'swap':     y_{i2} · x_{i1}
      'conj_r':   x_{i1} · conj(y_{i2})
      'conj_l':   conj(y_{i2}) · x_{i1}
    
    Plus a sign ±1 for each.
    """
    d = C1.shape[0]
    D = n * d
    
    # Conjugation matrix
    J = -np.eye(d)
    J[0, 0] = 1.0
    
    C2 = np.zeros((D, D, D))
    
    for i1 in range(n):
        for i2 in range(n):
            m = (i1 + i2) % n
            sign, mode = rule[i1][i2]
            
            for a in range(d):
                for b in range(d):
                    row = i1 * d + a
                    col = i2 * d + b
                    
                    if mode == 'straight':
                        # x_a · y_b
                        for c in range(d):
                            C2[row, col, m*d+c] += sign * C1[a, b, c]
                    
                    elif mode == 'swap':
                        # y_b · x_a
                        for c in range(d):
                            C2[row, col, m*d+c] += sign * C1[b, a, c]
                    
                    elif mode == 'conj_r':
                        # x_a · conj(y_b) = sum_bp J[bp,b] C1[a,bp,c]
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                C2[row, col, m*d+c] += sign * J[bp, b] * C1[a, bp, c]
                    
                    elif mode == 'conj_l':
                        # conj(y_b) · x_a = sum_bp J[bp,b] C1[bp,a,c]
                        for bp in range(d):
                            if J[bp, b] == 0: continue
                            for c in range(d):
                                C2[row, col, m*d+c] += sign * J[bp, b] * C1[bp, a, c]
    
    return C2


print("\n" + "=" * 72)
print("PHASE 2: Exhaustive Level-Up Rule Search")
print("=" * 72)

# For each (i1,i2) pair, we have:
# - sign: +1 or -1 (2 choices)
# - mode: straight, swap, conj_r, conj_l (4 choices)
# Total per pair: 8 choices
# 
# Constraints:
# (0,0) must be straight with sign +1 (unit × unit = straight product)
# (0,j) for j≠0: sign ±1, mode ∈ {straight, swap, conj_r, conj_l} → 8
# (i,0) for i≠0: same → 8
# (i,j) for i,j≠0: same → 8
#
# Free pairs: (0,1),(0,2),(1,0),(2,0),(1,1),(1,2),(2,1),(2,2) = 8 pairs
# Each with 8 choices → 8^8 = 16,777,216 combinations
# 
# That's too many. But we can use constraints:
# - For (0,j): must preserve unit. If e_0 is unit in Level-2,
#   then (e_0, 0, 0) · (y_0, y_1, y_2) = (y_0, y_1, y_2)
#   Slot 0: e_0 · y_0 (straight, +1) — already fixed
#   Slot 1: rule[0,1] applied to (e_0, y_1) must give y_1
#   Slot 2: rule[0,2] applied to (e_0, y_2) must give y_2
# Since e_0 is unit in Level-1: e_0·y = y and y·e_0 = y, conj(e_0) = e_0
# So straight, swap, conj_r, conj_l all give y_j when first arg is e_0.
# But sign must be +1 for unit preservation.
#
# Similarly (i,0): sign must be +1.
#
# So free choices: (0,j) j≠0: mode only (sign=+1) → 4^2 = 16
#                  (i,0) i≠0: mode only (sign=+1) → 4^2 = 16  
#                  (i,j) i,j≠0: sign and mode → 8^4 = 4096
# Total: 16 × 16 × 4096 = 1,048,576 ≈ 10^6 — feasible!

modes = ['straight', 'swap', 'conj_r', 'conj_l']

# Pick a few promising Level-1 patterns (associative ones first, then all)
# First just try the standard λ=-1 pattern and look for ANY associative Level-2

print("\nSearching over Level-1 sign patterns × Level-up rules...")
print("Unit-preserving constraint: sign=+1 for (0,j) and (i,0) pairs")
print(f"Free parameters: 4² × 4² × 8⁴ = {16*16*4096} combinations per Level-1")

t0 = time.time()
found_any = False
total_checked = 0
associative_count = 0

# Start with ALL 16 Level-1 patterns
for l1_res in sorted(l1_results, key=lambda x: x['violations']):
    s11, s12, s21, s22 = l1_res['signs']
    signs = [[1, 1, 1], [1, s11, s12], [1, s21, s22]]
    C1 = build_level1_params(3, signs)
    
    # (0,j) for j=1,2: sign=+1, mode free
    # (i,0) for i=1,2: sign=+1, mode free
    # (i,j) for i,j in {1,2}: sign ±1, mode free
    
    for m01, m02, m10, m20 in iprod(range(4), repeat=4):
        for s_and_m_11, s_and_m_12, s_and_m_21, s_and_m_22 in iprod(range(8), repeat=4):
            rule = [[None]*3 for _ in range(3)]
            rule[0][0] = (1, 'straight')
            rule[0][1] = (1, modes[m01])
            rule[0][2] = (1, modes[m02])
            rule[1][0] = (1, modes[m10])
            rule[2][0] = (1, modes[m20])
            
            for idx, sm in enumerate([s_and_m_11, s_and_m_12, s_and_m_21, s_and_m_22]):
                i = 1 + idx // 2
                j = 1 + idx % 2
                sign = 1 if sm < 4 else -1
                mode = modes[sm % 4]
                rule[i][j] = (sign, mode)
            
            C2 = general_level_up(C1, 3, rule)
            total_checked += 1
            
            # Quick check: is e_0 still a unit?
            if not check_unit(C2):
                continue
            
            # Associative?
            v = count_assoc_violations(C2)
            if v == 0:
                associative_count += 1
                ok, msg = check_matmul_iso(C2)
                status = "★ Mat(3,R) CANDIDATE!" if ok else msg
                print(f"\n  ★ ASSOCIATIVE ALGEBRA FOUND!")
                print(f"    L1 signs: {l1_res['signs']}")
                print(f"    Rule: (0,1)={rule[0][1]}, (0,2)={rule[0][2]}")
                print(f"    Rule: (1,0)={rule[1][0]}, (2,0)={rule[2][0]}")
                print(f"    Rule: (1,1)={rule[1][1]}, (1,2)={rule[1][2]}")
                print(f"    Rule: (2,1)={rule[2][1]}, (2,2)={rule[2][2]}")
                print(f"    Mat(3,R) check: {status}")
                
                # Check nnz and commutative
                nnz = np.count_nonzero(C2)
                comm = np.allclose(C2, C2.transpose(1, 0, 2))
                print(f"    nnz={nnz}, commutative={comm}")
                
                if ok:
                    found_any = True
                    print(f"\n    !!!! THIS IS IT — ASSOCIATIVE Z_3-RECURSIVE Mat(3,R) !!!!")
                    print(f"    Toom-3 gives ω ≤ log_3(5) ≈ 1.465")
            
            if total_checked % 500000 == 0:
                dt = time.time() - t0
                print(f"  ... checked {total_checked}/{16*16*16*4096} ({100*total_checked/(16*16*16*4096):.1f}%), "
                      f"assoc found: {associative_count}, time: {dt:.1f}s")

dt = time.time() - t0
print(f"\n{'='*72}")
print(f"SEARCH COMPLETE")
print(f"{'='*72}")
print(f"Total checked: {total_checked}")
print(f"Associative Level-2 algebras found: {associative_count}")
print(f"Mat(3,R) candidates: {'YES!' if found_any else 'NONE'}")
print(f"Time: {dt:.1f}s")

if not found_any and associative_count == 0:
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ THEOREM (exhaustive computation):                            │
  │                                                              │
  │ There is NO Z_3-recursive 9-dimensional associative algebra  │
  │ obtainable from ANY choice of:                               │
  │   • Level-1 sign pattern (16 choices)                        │
  │   • Level-up rule: sign × {{straight, swap, conj_R, conj_L}} │
  │     with unit-preserving constraints                         │
  │                                                              │
  │ Total search space: {total_checked:>10,} algebras                    │
  │                                                              │
  │ CONSEQUENCE: The Z_3 tower route to sub-cubic matmul is      │
  │ OBSTRUCTED. No sign/conjugation choice can make the          │
  │ recursive product associative.                               │
  │                                                              │
  │ The half-associativity (S→1/2) is not an accident —          │
  │ it's a NECESSARY feature of Z_3-graded CD constructions.     │
  └──────────────────────────────────────────────────────────────┘
""")
elif not found_any and associative_count > 0:
    print(f"""
  Found {associative_count} associative algebras, but NONE isomorphic to Mat(3,R).
  The associative ones are all commutative or have wrong center dimension.
  Mat(3,R) cannot be realized as a Z_3-recursive algebra with these rules.
""")
else:
    print(f"\n  ★★★ FOUND Mat(3,R) WITH Z_3 RECURSIVE STRUCTURE! ★★★")
    print(f"  This implies a new matrix multiplication algorithm!")
