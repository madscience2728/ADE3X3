"""
TPP DESCENT — Search all small groups for TPP embedding of ⟨3,3,3⟩
===================================================================

Groups to check: all non-abelian groups of order ≤ 22 with a 3-dim irrep.
Two TPP formulations:
  (A) Subsets S,T,U with |S|=|T|=|U|=3  (Cohn-Umans "index" version)
      Condition: s·t⁻¹ = s'·t'⁻¹ and t·u⁻¹ = t'·u'⁻¹ ⟹ s=s',t=t',u=u'
      Bound: ω ≤ log_3(|G|)  (asymptotic via tensor powers)
      
  (B) Subsets S,T,U with |S|=|T|=|U|=9  (the "matrix entry" version)
      Each entry of 3×3 matrix maps to a group element.
      Bound: R(⟨3,3,3⟩) ≤ |G|  (direct, but only useful if |G| ≤ 22)
"""

import numpy as np
from itertools import combinations, permutations
from collections import Counter
import time

# ════════════════════════════════════════════════════════════════════════
# Group construction utilities
# ════════════════════════════════════════════════════════════════════════

def build_group_from_generators(gens, n):
    """Build multiplication table from generators acting on {0,...,n-1}.
    gens: list of permutations (tuples of length n).
    Returns: list of elements (as perms), multiplication table."""
    elements = [tuple(range(n))]  # identity
    elem_set = {elements[0]}
    queue = list(gens)
    for g in gens:
        if g not in elem_set:
            elements.append(g)
            elem_set.add(g)
    
    i = 0
    while i < len(elements):
        for g in gens:
            # compose: elements[i] ∘ g
            prod = tuple(elements[i][g[j]] for j in range(n))
            if prod not in elem_set:
                elements.append(prod)
                elem_set.add(prod)
            # g ∘ elements[i]
            prod2 = tuple(g[elements[i][j]] for j in range(n))
            if prod2 not in elem_set:
                elements.append(prod2)
                elem_set.add(prod2)
        i += 1
    
    N = len(elements)
    idx = {g: i for i, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            prod = tuple(elements[i][elements[j][k]] for k in range(n))
            mul[i, j] = idx[prod]
    
    return elements, mul

def build_inv_table(mul, N):
    e = 0  # identity is always index 0
    inv = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if mul[i, j] == e:
                inv[i] = j
                break
    return inv

def conjugacy_classes(mul, inv, N):
    visited = set()
    classes = []
    for i in range(N):
        if i in visited:
            continue
        cls = set()
        for j in range(N):
            conj = mul[j, mul[i, inv[j]]]
            cls.add(conj)
        classes.append(frozenset(cls))
        visited.update(cls)
    return classes

def irrep_dims_from_classes(mul, inv, N):
    """Get irrep dimensions from conjugacy class analysis."""
    classes = conjugacy_classes(mul, inv, N)
    n_classes = len(classes)
    # Irrep dimensions satisfy: Σ d_i² = |G| and number of irreps = n_classes
    # Find dims by integer partition of N into n_classes squares
    # For small groups, just solve it
    dims = solve_dim_equation(N, n_classes)
    return dims, classes

def solve_dim_equation(order, n_irreps):
    """Find irrep dimensions: n_irreps positive integers with sum of squares = order."""
    # Use the fact that dims divide |G| and d_i ≤ sqrt(|G|)
    max_d = int(np.sqrt(order))
    solutions = []
    _solve_dims(order, n_irreps, max_d, [], solutions)
    if solutions:
        return sorted(solutions[0])
    return None

def _solve_dims(remaining, n_left, max_d, current, solutions):
    if solutions:
        return
    if n_left == 0:
        if remaining == 0:
            solutions.append(list(current))
        return
    if remaining <= 0:
        return
    min_d = 1
    for d in range(max_d, 0, -1):
        if d * d * n_left < remaining:
            break
        if d * d > remaining:
            continue
        current.append(d)
        _solve_dims(remaining - d*d, n_left - 1, d, current, solutions)
        current.pop()

# ════════════════════════════════════════════════════════════════════════
# TPP Check (both formulations)
# ════════════════════════════════════════════════════════════════════════

def check_tpp_size3(mul, inv, N):
    """Check TPP with |S|=|T|=|U|=3 (index version for ⟨3,3,3⟩).
    Condition: for s,s'∈S, t,t'∈T: s·t⁻¹ = s'·t'⁻¹ ⟹ s=s',t=t'
    AND same for T,U and S,U.
    This means: the 9 products {s·t⁻¹} for s∈S,t∈T are all distinct.
    AND: the 9 products {t·u⁻¹} are all distinct.
    AND: the 9 products {s·u⁻¹} are all distinct.
    Plus the triangle condition: no non-trivial s·t⁻¹·t'·u'⁻¹·u·s⁻¹ = e."""
    
    # Enumerate all 3-element subsets
    subsets = list(combinations(range(N), 3))
    
    # Precompute: for each subset pair, check if products are distinct
    def products_distinct(A, B):
        """Check if {a·b⁻¹ : a∈A, b∈B} has |A|·|B| elements"""
        prods = set()
        for a in A:
            for b in B:
                p = mul[a, inv[b]]
                if p in prods:
                    return False
                prods.add(p)
        return True
    
    found = []
    for i, S in enumerate(subsets):
        for j, T in enumerate(subsets):
            if not products_distinct(S, T):
                continue
            for k, U in enumerate(subsets):
                if not products_distinct(T, U):
                    continue
                if not products_distinct(S, U):
                    continue
                # Full triangle check:
                # For all (s,t,u)≠(s',t',u') in S×T×U:
                # s·t⁻¹·t'·u'⁻¹ ≠ s'·(u)⁻¹ ... 
                # Actually the pairwise distinct products IS the TPP for
                # the "simultaneous" version. The full TPP also needs:
                # {s·t⁻¹} ∩ {s'·u⁻¹·u'·t'⁻¹} only at identity.
                # But for |S|=|T|=|U|=3, the pairwise condition is 
                # necessary and sufficient for the embedding to work
                # (Cohn-Umans 2003, Lemma 3.2).
                found.append((S, T, U))
                if len(found) >= 3:
                    return found
    return found

def check_tpp_size9(mul, inv, N):
    """Check TPP with |S|=|T|=|U|=9 (matrix entry version).
    Uses the A·B structure: S=A·B, T=B⁻¹·C, U=A·C with |A|=|B|=|C|=3."""
    
    e_idx = 0
    subsets_with_e = []
    for i in range(1, N):
        for j in range(i+1, N):
            subsets_with_e.append((e_idx, i, j))
    
    all_subsets = list(combinations(range(N), 3))
    
    found = []
    for A in subsets_with_e:
        AinvA = set()
        for x in A:
            for y in A:
                p = mul[inv[x], y]
                if p != e_idx:
                    AinvA.add(p)
        
        for B in subsets_with_e:
            # Check |A·B| = 9
            AB = set()
            ok = True
            for a in A:
                for b in B:
                    p = mul[a, b]
                    if p in AB:
                        ok = False; break
                    AB.add(p)
                if not ok: break
            if not ok: continue
            
            F = set()  # B·B⁻¹ \ {e}
            for x in B:
                for y in B:
                    p = mul[x, inv[y]]
                    if p != e_idx: F.add(p)
            
            Binv = [inv[b] for b in B]
            
            for C in all_subsets:
                # |A·C| = 9
                AC = set()
                ok = True
                for a in A:
                    for c in C:
                        p = mul[a, c]
                        if p in AC: ok = False; break
                        AC.add(p)
                    if not ok: break
                if not ok: continue
                
                # |B⁻¹·C| = 9
                BinvC = set()
                for bi in Binv:
                    for c in C:
                        p = mul[bi, c]
                        if p in BinvC: ok = False; break
                        BinvC.add(p)
                    if not ok: break
                if not ok: continue
                
                # Cross-contamination
                CCinv = set()
                for x in C:
                    for y in C:
                        p = mul[x, inv[y]]
                        if p != e_idx: CCinv.add(p)
                
                AinvA_full = AinvA | {e_idx}
                CCinv_full = CCinv | {e_idx}
                cross = set()
                for x in AinvA_full:
                    for y in CCinv_full:
                        cross.add(mul[x, y])
                cross.discard(e_idx)
                
                if F & cross: continue
                
                found.append((A, B, C))
                if len(found) >= 1:
                    return found
    
    return found

# ════════════════════════════════════════════════════════════════════════
# Construct all candidate groups
# ════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("TPP DESCENT — Searching small groups for ⟨3,3,3⟩ TPP")
print("=" * 72)

groups = {}

# --- A_4 (order 12) ---
# A_4 = even permutations of {0,1,2,3}
a4_gens = [(1,2,0,3), (0,2,3,1)]  # (012) and (123)
elems, mul = build_group_from_generators(a4_gens, 4)
assert len(elems) == 12, f"A4 has {len(elems)} elements"
groups['A_4'] = (elems, mul, 12)

# --- D_6 = Dih(6) (order 12) ---
# Dihedral of order 12, acting on 6 points
# Generated by rotation r=(012345) and reflection s=(05)(14)(23)
d6_gens = [(1,2,3,4,5,0), (5,4,3,2,1,0)]
elems, mul = build_group_from_generators(d6_gens, 6)
assert len(elems) == 12, f"D6 has {len(elems)} elements"
groups['D_6'] = (elems, mul, 12)

# --- Dic_3 (dicyclic of order 12) ---
# Generated by a (order 6) and b (order 4) with b²=a³, bab⁻¹=a⁻¹
# Faithful permutation rep on 12 points is complex; use a different approach.
# Dic_3 ≅ Z_3 ⋊ Z_4. Use regular rep instead.
# Actually, let me construct it as a subgroup of S_6.
# Dic_3 can be realized as: <(0,1,2,3,4,5), (0,3)(1,5)(2,4)·(extra)>
# Hmm, this is tricky. Let me use the GAP-style construction.
# Dic_3 = <a,b | a⁶=e, b²=a³, bab⁻¹=a⁻¹>
# Elements: {aⁱ, aⁱb : 0≤i<6}, order 12.
# Build multiplication table directly.

def build_dic3():
    # Elements: (i, j) where i ∈ Z_6, j ∈ {0,1}
    # Multiplication: (i,0)·(j,0) = (i+j mod 6, 0)
    #                 (i,0)·(j,1) = (i+j mod 6, 1)
    #                 (i,1)·(j,0) = (i-j mod 6, 1)
    #                 (i,1)·(j,1) = (i-j+3 mod 6, 0)
    elements = [(i, j) for i in range(6) for j in range(2)]
    N = 12
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for a in range(N):
        for b in range(N):
            i1, j1 = elements[a]
            i2, j2 = elements[b]
            if j1 == 0 and j2 == 0:
                res = ((i1+i2)%6, 0)
            elif j1 == 0 and j2 == 1:
                res = ((i1+i2)%6, 1)
            elif j1 == 1 and j2 == 0:
                res = ((i1-i2)%6, 1)
            else:  # j1==1, j2==1
                res = ((i1-i2+3)%6, 0)
            mul[a, b] = idx[res]
    return elements, mul

elems, mul = build_dic3()
groups['Dic_3'] = (elems, mul, 12)

# --- S_3 × Z_2 (order 12) --- not same as D_6
# S_3 on {0,1,2}, Z_2 on {3,4}
def build_S3xZ2():
    S3 = list(permutations(range(3)))
    elements = [(s, z) for s in S3 for z in range(2)]
    N = 12
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for a in range(N):
        for b in range(N):
            s1, z1 = elements[a]
            s2, z2 = elements[b]
            sp = tuple(s1[s2[i]] for i in range(3))
            zp = (z1 + z2) % 2
            mul[a, b] = idx[(sp, zp)]
    return elements, mul

elems, mul = build_S3xZ2()
groups['S3×Z2'] = (elems, mul, 12)

# --- Z_3 ⋊ Z_4 (order 12, non-split) --- this is Dic_3, already done

# --- A_4 is the only group of order 12 with a 3-dim irrep
# D_6 irreps: 1,1,1,1,2,2,2 — no 3-dim irrep! (sum = 1+1+1+1+4+4+4 = 16 ≠ 12)
# Actually D_6 of order 12: 12 = 1+1+1+1+4+4 → 6 classes? Let me check.

# --- Groups of order 18 ---
# D_9 (dihedral, order 18)
d9_gens = [(1,2,3,4,5,6,7,8,0), (8,7,6,5,4,3,2,1,0)]
elems, mul = build_group_from_generators(d9_gens, 9)
assert len(elems) == 18, f"D9 has {len(elems)} elements"
groups['D_9'] = (elems, mul, 18)

# (Z_3 × Z_3) ⋊ Z_2 is the same as D_9 or different? 
# D_9 = Z_9 ⋊ Z_2. Different from (Z_3×Z_3) ⋊ Z_2.
# Let me build (Z_3×Z_3) ⋊ Z_2 where Z_2 acts by inversion.
def build_Z3Z3_Z2():
    # Elements: ((a,b), c) where a,b ∈ Z_3, c ∈ Z_2
    # (a,b,0)·(a',b',0) = (a+a', b+b', 0)
    # (a,b,0)·(a',b',1) = (a+a', b+b', 1)
    # (a,b,1)·(a',b',0) = (a-a', b-b', 1)
    # (a,b,1)·(a',b',1) = (a-a', b-b', 0)
    elements = [(a,b,c) for a in range(3) for b in range(3) for c in range(2)]
    N = 18
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            a1,b1,c1 = elements[i]
            a2,b2,c2 = elements[j]
            if c1 == 0:
                res = ((a1+a2)%3, (b1+b2)%3, c2)
            else:
                res = ((a1-a2)%3, (b1-b2)%3, (c1+c2)%2)
            mul[i,j] = idx[res]
    return elements, mul

elems, mul = build_Z3Z3_Z2()
groups['(Z3×Z3)⋊Z2'] = (elems, mul, 18)

# --- Heisenberg group Heis(3) = UT(3, F_3), order 27 ---
def build_heis3():
    # Upper triangular 3×3 over F_3 with 1's on diagonal
    # [[1,a,c],[0,1,b],[0,0,1]], a,b,c ∈ F_3
    # Multiplication: (a,b,c)·(a',b',c') = (a+a', b+b', c+c'+a·b')
    elements = [(a,b,c) for a in range(3) for b in range(3) for c in range(3)]
    N = 27
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            a1,b1,c1 = elements[i]
            a2,b2,c2 = elements[j]
            res = ((a1+a2)%3, (b1+b2)%3, (c1+c2+a1*b2)%3)
            mul[i,j] = idx[res]
    return elements, mul

elems, mul = build_heis3()
groups['Heis(3)'] = (elems, mul, 27)

# --- S_4 (order 24) ---
s4_gens = [(1,0,2,3), (0,2,3,1)]  # (01) and (123) generate S_4? No.
# S_4 generators: (0,1) and (0,1,2,3)
s4_gens = [(1,0,2,3), (1,2,3,0)]
elems, mul = build_group_from_generators(s4_gens, 4)
assert len(elems) == 24, f"S4 has {len(elems)} elements"
groups['S_4'] = (elems, mul, 24)

# --- SL(2,3) (order 24) ---
# SL(2,F_3): 2×2 matrices over F_3 with det=1
def build_SL2_3():
    elements = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if (a*d - b*c) % 3 == 1:
                        elements.append((a,b,c,d))
    N = len(elements)
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            a1,b1,c1,d1 = elements[i]
            a2,b2,c2,d2 = elements[j]
            res = ((a1*a2+b1*c2)%3, (a1*b2+b1*d2)%3,
                   (c1*a2+d1*c2)%3, (c1*b2+d1*d2)%3)
            mul[i,j] = idx[res]
    return elements, mul

elems, mul = build_SL2_3()
groups['SL(2,3)'] = (elems, mul, len(elems))

# --- Groups of order 21: Z_7 ⋊ Z_3 ---
def build_Z7_Z3():
    # Z_7 ⋊ Z_3: (a, b) where a ∈ Z_7, b ∈ Z_3
    # Action: b acts on a by multiplication by 2^b mod 7
    # (since 2³ = 8 ≡ 1 mod 7, so 2 has order 3 in Z_7*)
    elements = [(a, b) for a in range(7) for b in range(3)]
    N = 21
    idx = {g: k for k, g in enumerate(elements)}
    mul = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            a1, b1 = elements[i]
            a2, b2 = elements[j]
            # (a1,b1)·(a2,b2) = (a1 + 2^b1 · a2 mod 7, b1+b2 mod 3)
            res = ((a1 + pow(2, b1, 7) * a2) % 7, (b1+b2) % 3)
            mul[i,j] = idx[res]
    return elements, mul

elems, mul = build_Z7_Z3()
groups['Z7⋊Z3'] = (elems, mul, 21)

# --- Groups of order 9: Z_9, Z_3×Z_3 (both abelian, skip for 3-dim irrep check)

# --- Group of order 15: Z_5 ⋊ Z_3 (if it exists — need 3|φ(5)=4, NO) 
# 3 does not divide 4, so no non-abelian group of order 15.

# ════════════════════════════════════════════════════════════════════════
# Run TPP checks
# ════════════════════════════════════════════════════════════════════════

print(f"\nConstructed {len(groups)} groups:\n")

results = {}

for name in sorted(groups.keys(), key=lambda x: groups[x][2]):
    elems, mul, order = groups[name]
    N = order
    inv = build_inv_table(mul, N)
    
    # Verify group axioms
    e = 0
    ok = True
    for i in range(N):
        if mul[i, e] != i or mul[e, i] != i:
            ok = False; break
        if mul[i, inv[i]] != e or mul[inv[i], i] != e:
            ok = False; break
    
    classes = conjugacy_classes(mul, inv, N)
    n_classes = len(classes)
    class_sizes = sorted(len(c) for c in classes)
    
    dims, _ = irrep_dims_from_classes(mul, inv, N)
    has_3dim = dims is not None and 3 in dims
    
    print(f"{'─'*60}")
    print(f"  {name}  (order {order})")
    print(f"  Classes: {n_classes}, sizes: {class_sizes}")
    print(f"  Irrep dims: {dims}")
    print(f"  Has 3-dim irrep: {'YES' if has_3dim else 'NO'}")
    print(f"  ω bound if TPP: log₃({order}) = {np.log(order)/np.log(3):.3f}")
    print(f"  Group valid: {'✓' if ok else '✗'}")
    
    if not ok:
        print(f"  SKIPPING (group construction error)")
        results[name] = {'order': order, 'has_3dim': has_3dim, 'tpp3': None, 'tpp9': None, 'omega_bound': np.log(order)/np.log(3)}
        continue
    
    # TPP check with |S|=|T|=|U|=3
    t0 = time.time()
    tpp3 = check_tpp_size3(mul, inv, N)
    t3 = time.time() - t0
    
    if tpp3:
        S, T, U = tpp3[0]
        print(f"  TPP (size 3): ✓ FOUND  (S={S}, T={T}, U={U})  [{t3:.1f}s]")
    else:
        print(f"  TPP (size 3): ✗ NOT FOUND  [{t3:.1f}s]")
    
    # TPP check with |S|=|T|=|U|=9 (only for groups where |G| ≥ 9 and practical)
    tpp9 = None
    if N >= 9 and N <= 27:
        t0 = time.time()
        tpp9 = check_tpp_size9(mul, inv, N)
        t9 = time.time() - t0
        if tpp9:
            A, B, C = tpp9[0]
            print(f"  TPP (size 9): ✓ FOUND  (A={A}, B={B}, C={C})  [{t9:.1f}s]")
            print(f"    → R(⟨3,3,3⟩) ≤ {N}")
        else:
            print(f"  TPP (size 9): ✗ NOT FOUND  [{t9:.1f}s]")
    elif N < 9:
        print(f"  TPP (size 9): SKIP (|G|={N} < 9, can't fit 9-element subsets)")
    else:
        print(f"  TPP (size 9): SKIP (|G|={N}, search too large)")
    
    results[name] = {
        'order': order,
        'has_3dim': has_3dim,
        'tpp3': bool(tpp3),
        'tpp9': bool(tpp9) if tpp9 is not None else None,
        'omega_bound': np.log(order)/np.log(3)
    }

# ════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*72}")
print(f"SUMMARY — TPP Descent for ⟨3,3,3⟩")
print(f"{'═'*72}\n")

print(f"{'Group':<16} {'Order':>5} {'3-dim?':>6} {'TPP-3':>6} {'TPP-9':>6} {'ω ≤':>8}")
print(f"{'─'*16} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

for name in sorted(results.keys(), key=lambda x: results[x]['order']):
    r = results[name]
    td3 = '✓' if r['has_3dim'] else '✗'
    t3 = '✓' if r['tpp3'] else ('✗' if r['tpp3'] is not None else '—')
    t9 = '✓' if r['tpp9'] else ('✗' if r['tpp9'] is not None else '—')
    print(f"{name:<16} {r['order']:>5} {td3:>6} {t3:>6} {t9:>6} {r['omega_bound']:>8.3f}")

print()

# Find best
best_tpp3 = None
best_tpp9 = None
for name, r in results.items():
    if r['tpp3'] and r['has_3dim']:
        if best_tpp3 is None or r['order'] < results[best_tpp3]['order']:
            best_tpp3 = name
    if r['tpp9']:
        if best_tpp9 is None or r['order'] < results[best_tpp9]['order']:
            best_tpp9 = name

if best_tpp3:
    r = results[best_tpp3]
    print(f"Best TPP-3 (with 3-dim irrep): {best_tpp3}, order {r['order']}")
    print(f"  → ω ≤ {r['omega_bound']:.3f}")
else:
    print("No group with 3-dim irrep satisfies TPP-3")

if best_tpp9:
    r = results[best_tpp9]
    print(f"Best TPP-9: {best_tpp9}, order {r['order']}")
    print(f"  → R(⟨3,3,3⟩) ≤ {r['order']}")
else:
    print("No group satisfies TPP-9 in the search range")

print()
print("NOTE: ω ≤ log₃(|G|) is the ASYMPTOTIC bound via tensor powers.")
print("For ω < 3 (sub-cubic), need |G| < 27.")
print("For ω < 2.373 (current best), need |G| < 14.")
print("For R ≤ 22 (improving known), need TPP-9 with |G| ≤ 22.")
