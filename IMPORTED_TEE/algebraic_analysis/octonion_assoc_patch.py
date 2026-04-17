"""
octonion_assoc_patch.py
=======================
Associator-first observable analysis for one octonion patch.

x = x0*e_{X[0]} + x1*e_{X[1]} + x2*e_{X[2]}
y = y0*e_{Y[0]} + y1*e_{Y[1]} + y2*e_{Y[2]}
w = fixed probe (one octonion basis element)

9 monomials  m_{ab} = x_a * y_b

Product observable  P_k : bucket e_k of (x*y)
Associator observable  A_{w,k} : component e_k of [x, y, w] = (x*y)*w - x*(y*w)

Goal: find minimal observable set with rank 9.
"""

import numpy as np
from numpy.linalg import matrix_rank
from itertools import combinations
from collections import defaultdict

SEP  = "=" * 70
SEP2 = "-" * 70

# ─────────────────────────────────────────────────────────────────────────────
# 0. Octonion algebra
# ─────────────────────────────────────────────────────────────────────────────
# Standard Fano-plane multiplication.
# Lines {i,j,k} with cyclic orientation i->j->k->i give
#   e_i * e_j = +e_k  (and all cyclic rotations are +)
#   e_j * e_i = -e_k  (reversed = -)
FANO_LINES = [
    (1, 2, 3),
    (1, 4, 5),
    (2, 4, 6),
    (1, 6, 7),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
]

def _build_oct():
    T = np.zeros((8, 8, 8), dtype=float)
    for i in range(8):          # e0 = identity
        T[0, i, i] = 1.0
        T[i, 0, i] = 1.0
    for i in range(1, 8):       # e_i^2 = -1
        T[i, i, 0] = -1.0
    for (i, j, k) in FANO_LINES:
        T[i, j, k] =  1.0;  T[j, i, k] = -1.0
        T[j, k, i] =  1.0;  T[k, j, i] = -1.0
        T[k, i, j] =  1.0;  T[i, k, j] = -1.0
    return T

_OCT = _build_oct()

def omul_vec(i, j):
    """e_i * e_j as 8-vector."""
    return _OCT[i, j].copy()

def oassoc_vec(i, j, k):
    """[e_i, e_j, e_k] = (e_i*e_j)*e_k - e_i*(e_j*e_k) as 8-vector."""
    ij   = _OCT[i, j]
    ij_k = np.einsum('m,mkr->r', ij, _OCT) [k]   # wrong shape -- use loop
    # redo cleanly:
    ij_k = sum(ij[m] * _OCT[m, k] for m in range(8))
    jk   = _OCT[j, k]
    i_jk = sum(jk[m] * _OCT[i, m] for m in range(8))
    return ij_k - i_jk

# Verify alternative laws: [x,x,y]=0 and [x,y,y]=0 for all basis pairs.
for _i in range(8):
    for _j in range(8):
        assert np.allclose(oassoc_vec(_i, _i, _j), 0), f"[e{_i},e{_i},e{_j}]!=0"
        assert np.allclose(oassoc_vec(_i, _j, _j), 0), f"[e{_i},e{_j},e{_j}]!=0"
print("Alternative laws verified for all basis pairs.")

# Quick spot-check: [e1,e2,e5] = 2*e6
_spot = oassoc_vec(1, 2, 5)
assert abs(_spot[6] - 2.0) < 1e-12 and np.allclose(_spot[[k for k in range(8) if k!=6]], 0), \
    f"Spot-check [e1,e2,e5]=2*e6 failed: {_spot}"
print("Spot-check [e1,e2,e5] = 2*e6 verified.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fixed patch
# ─────────────────────────────────────────────────────────────────────────────
X_BASIS = [1, 2, 3]   # x-space: e1, e2, e3
Y_BASIS = [4, 5, 6]   # y-space: e4, e5, e6

NX = len(X_BASIS)
NY = len(Y_BASIS)
N_MON = NX * NY          # 9

def midx(a, b): return a * NY + b
MON_LABELS = [f"m{a}{b}" for a in range(NX) for b in range(NY)]

print(f"\n{SEP}")
print(f"PATCH:  X={X_BASIS}, Y={Y_BASIS}")
print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Product routing table
# ─────────────────────────────────────────────────────────────────────────────
prod_vecs = {}     # (a,b) -> 8-vector
for a in range(NX):
    for b in range(NY):
        prod_vecs[(a, b)] = omul_vec(X_BASIS[a], Y_BASIS[b])

# Gather buckets: which monomials contribute to each output component?
bucket_members = defaultdict(list)    # k -> [(a,b,sign)]
for (a, b), v in prod_vecs.items():
    for k in range(8):
        if v[k] != 0:
            bucket_members[k].append((a, b, int(v[k])))

prod_buckets = sorted(bucket_members.keys())

print("\nProduct routing:")
for a in range(NX):
    for b in range(NY):
        v = prod_vecs[(a, b)]
        k = int(np.argmax(np.abs(v)))
        s = int(v[k])
        print(f"  e{X_BASIS[a]} * e{Y_BASIS[b]} -> {'+' if s>0 else '-'}e{k}")

print("\nProduct buckets:")
collided_buckets = set()
for k in prod_buckets:
    mems = bucket_members[k]
    terms = " + ".join(f"({'+' if s>0 else '-'}1)*m{a}{b}" for a,b,s in mems)
    tag = "  <-- COLLIDED" if len(mems) > 1 else ""
    print(f"  bucket e{k}:  {terms}{tag}")
    if len(mems) > 1:
        collided_buckets.add(k)

collided_monomials = {(a, b)
                      for k in collided_buckets
                      for (a, b, s) in bucket_members[k]}
coll_col_idxs = {midx(a, b) for (a, b) in collided_monomials}

print(f"\n  Total product buckets  : {len(prod_buckets)}")
print(f"  Collided buckets       : {sorted(collided_buckets)}")
print(f"  Collided monomials     : {sorted(collided_monomials)}  ({len(collided_monomials)} of 9)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Product observation matrix
# ─────────────────────────────────────────────────────────────────────────────
P_mat = np.zeros((len(prod_buckets), N_MON), dtype=float)
for ri, k in enumerate(prod_buckets):
    for (a, b, s) in bucket_members[k]:
        P_mat[ri, midx(a, b)] = float(s)

P_rank = matrix_rank(P_mat)
print(f"\n  Product matrix shape   : {P_mat.shape}")
print(f"  Product matrix rank    : {P_rank}  (need 9 for full recovery)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Associator routing table builder
# ─────────────────────────────────────────────────────────────────────────────
def assoc_matrix_for_probe(X_b, Y_b, w):
    """8 x N_MON matrix of associator coefficients for probe w."""
    A = np.zeros((8, len(X_b) * len(Y_b)), dtype=float)
    for a, xa in enumerate(X_b):
        for b, yb in enumerate(Y_b):
            av = oassoc_vec(xa, yb, w)
            A[:, a * len(Y_b) + b] = av
    return A


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build combined map with selective filters
# ─────────────────────────────────────────────────────────────────────────────
def build_combined(X_b, Y_b, P, coll_col_set, w_probes):
    """
    Incrementally build combined matrix.
    Filters:
      (i)  Only assoc rows with support overlapping collided monomials.
      (ii) Only rows NOT already in span of current matrix.

    Returns (M, kept_labels, rank, n_prod_rows, n_assoc_rows)
    """
    rows       = list(P)           # start with product rows (each as 1-D array)
    span_rows  = list(P)
    kept_labels = []               # human-readable label for each assoc row kept

    for w in w_probes:
        A = assoc_matrix_for_probe(X_b, Y_b, w)
        for j in range(8):
            row = A[j]
            if np.allclose(row, 0):
                continue
            # Filter (i): support intersects collided monomials?
            support = {i for i in range(len(row)) if abs(row[i]) > 1e-10}
            if not support & coll_col_set:
                continue
            # Filter (ii): novel?
            cand = np.vstack(span_rows + [row])
            if matrix_rank(cand) <= matrix_rank(np.vstack(span_rows)):
                continue
            span_rows.append(row)
            rows.append(row)
            kept_labels.append(f"[x,y,e{w}]_e{j}")

    M  = np.vstack(rows)
    rk = matrix_rank(M)
    return M, kept_labels, rk, len(P), len(kept_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Single-probe scan
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SINGLE-PROBE SCAN")
print(SEP)
print(f"  {'w':>4}  {'prod_rows':>9}  {'assoc_rows':>10}  {'total':>6}  {'rank':>6}")
print("  " + "-" * 45)

P_rows = [P_mat[i] for i in range(P_mat.shape[0])]

single_results = []
for w in range(8):
    M, kept, rk, np_, na = build_combined(X_BASIS, Y_BASIS, P_rows, coll_col_idxs, [w])
    total = np_ + na
    marker = "  *** RANK 9" if rk == N_MON else ""
    print(f"  e{w:>2}  {np_:>9}  {na:>10}  {total:>6}  {rk:>6}{marker}")
    single_results.append((total, rk, [w], kept, M))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pair-probe scan
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("PAIR-PROBE SCAN")
print(SEP)

pair_results = []
for w1, w2 in combinations(range(8), 2):
    M, kept, rk, np_, na = build_combined(X_BASIS, Y_BASIS, P_rows, coll_col_idxs, [w1, w2])
    pair_results.append((np_ + na, rk, [w1, w2], kept, M))

pair_results.sort(key=lambda x: (x[1] != N_MON, x[0]))
print(f"  {'probes':>12}  {'prod_rows':>9}  {'assoc_rows':>10}  {'total':>6}  {'rank':>6}")
print("  " + "-" * 50)
shown = 0
for (total, rk, ws, kept, M) in pair_results:
    if shown >= 12:
        break
    na = len(kept)
    np_ = total - na
    marker = "  ***" if rk == N_MON else ""
    print(f"  {str(['e'+str(w) for w in ws]):>14}  {np_:>9}  {na:>10}  {total:>6}  {rk:>6}{marker}")
    shown += 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Best combination detail
# ─────────────────────────────────────────────────────────────────────────────
all_results = single_results + pair_results
rank9 = [(tot, rk, ws, kept, M) for (tot, rk, ws, kept, M) in all_results if rk == N_MON]
rank9.sort(key=lambda x: x[0])

print(f"\n{SEP}")
print("BEST COMBINATION — EXPLICIT ROUTING TABLES")
print(SEP)

if rank9:
    best_tot, best_rk, best_ws, best_kept, best_M = rank9[0]
    print(f"\n  Probe(s):          {['e'+str(w) for w in best_ws]}")
    print(f"  Product buckets:   {len(P_rows)}  (components e{prod_buckets})")
    print(f"  Assoc rows kept:   {len(best_kept)}")
    print(f"  Total observables: {best_tot}")
    print(f"  Rank:              {best_rk}  (FULL)")

    # Product routing
    print(f"\n  --- Product routing ---")
    for k in prod_buckets:
        mems = bucket_members[k]
        terms = " + ".join(f"({s:+d})*{MON_LABELS[midx(a,b)]}" for a,b,s in mems)
        print(f"  P_{k}  =  {terms}")

    # Associator routing for best probes
    print(f"\n  --- Associator routing for kept rows ---")
    for w in best_ws:
        A = assoc_matrix_for_probe(X_BASIS, Y_BASIS, w)
        for j in range(8):
            row = A[j]
            label = f"[x,y,e{w}]_e{j}"
            if label not in best_kept:
                continue
            nz = [(MON_LABELS[i], row[i]) for i in range(N_MON) if abs(row[i]) > 1e-10]
            terms = " + ".join(f"({s:+.0f})*{m}" for m, s in nz)
            print(f"  {label:22}  =  {terms}")

    # Full combined matrix
    print(f"\n  --- Combined matrix ({best_M.shape[0]} x {N_MON}) ---")
    print("  " + " ".join(f"{l:>7}" for l in MON_LABELS))
    for i, row in enumerate(best_M):
        if i < len(P_rows):
            lbl = f"P_e{prod_buckets[i]}"
        else:
            lbl = best_kept[i - len(P_rows)]
        print(f"  {lbl:22s}" + " ".join(f"{v:>7.1f}" for v in row))

    # Null-space check
    _, sv, Vt = np.linalg.svd(best_M)
    print(f"\n  Singular values: {np.round(sv, 4)}")
    print(f"  Smallest singular value: {sv[-1]:.4e}  (> 0 confirms rank {best_rk})")

else:
    print("\n  Rank 9 NOT achieved with any single or pair probe on this patch!")
    best = max(all_results, key=lambda x: x[1])
    print(f"  Best: probes={['e'+str(w) for w in best[2]]}, total={best[0]}, rank={best[1]}")

    # Show null space
    _, sv, Vt = np.linalg.svd(best[4])
    print(f"\n  Null space (dimension {N_MON - best[1]}):")
    for nv in Vt[best[1]:]:
        nz = [(MON_LABELS[j], round(nv[j], 4)) for j in range(N_MON) if abs(nv[j]) > 1e-6]
        print(f"    {nz}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full associator routing detail for all probes
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("FULL ASSOCIATOR ROUTING — ALL 8 PROBES")
print(SEP)
print(f"  Rows: 9 monomials.  Columns: e0..e7 components of [x,y,w].")
print(f"  Non-zero entries only shown.\n")

for w in range(8):
    A = assoc_matrix_for_probe(X_BASIS, Y_BASIS, w)
    print(f"  w = e{w}:")
    any_nz = False
    for a in range(NX):
        for b in range(NY):
            col = midx(a, b)
            nz = {k: A[k, col] for k in range(8) if abs(A[k, col]) > 1e-10}
            if nz:
                parts = ", ".join(f"{s:+.0f}*e{k}" for k, s in sorted(nz.items()))
                print(f"    [e{X_BASIS[a]},e{Y_BASIS[b]},e{w}] = {parts}")
                any_nz = True
    if not any_nz:
        print(f"    All zero (associator vanishes for this probe).")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Patch variation: fix X=[1,2,3], vary Y over all 3-subsets of {1..7}
#     and search all w probes. Goal: minimize total rows at rank 9.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("PATCH VARIATION SEARCH  (X fixed = [1,2,3], Y vary over all C(7,3)=35 choices)")
print(SEP)

X_FIX = [1, 2, 3]
records = []

for Y_cand in combinations(range(1, 8), 3):
    Y_cand = list(Y_cand)

    # product routing for this Y
    bk = defaultdict(list)
    for a in range(3):
        for b in range(3):
            v = omul_vec(X_FIX[a], Y_cand[b])
            for k in range(8):
                if v[k] != 0:
                    bk[k].append((a, b, int(v[k])))

    bk_keys   = sorted(bk.keys())
    P_c       = np.zeros((len(bk_keys), 9), dtype=float)
    for ri, k in enumerate(bk_keys):
        for (a, b, s) in bk[k]:
            P_c[ri, a * 3 + b] = float(s)

    coll_bk   = {k for k, v in bk.items() if len(v) > 1}
    coll_cols  = {a * 3 + b for k in coll_bk for (a, b, s) in bk[k]}
    P_rows_c  = [P_c[i] for i in range(P_c.shape[0])]

    for w in range(8):
        M_, kept_, rk_, np__, na_ = build_combined(X_FIX, Y_cand, P_rows_c, coll_cols, [w])
        records.append((np__ + na_, rk_, X_FIX, Y_cand, [w], np__, na_))

records.sort(key=lambda x: (x[1] != N_MON, x[0]))

print(f"\n  {'Y_BASIS':>18}  {'w':>4}  {'prod':>5}  {'assoc':>6}  {'tot':>5}  {'rank':>5}")
print("  " + "-" * 55)
shown_keys = set()
for row in records:
    tot, rk, Xb, Yb, ws, np_, na_ = row
    key = (tuple(Yb), tuple(ws))
    if key in shown_keys:
        continue
    shown_keys.add(key)
    star = " ***" if rk == N_MON else ""
    print(f"  {str(Yb):>18}  e{ws[0]:>2}  {np_:>5}  {na_:>6}  {tot:>5}  {rk:>5}{star}")
    if len(shown_keys) >= 25:
        break

# Among rank-9 results, find the minimal total
rank9_var = [(tot, rk, Xb, Yb, ws, np_, na_)
             for (tot, rk, Xb, Yb, ws, np_, na_) in records if rk == N_MON]

if rank9_var:
    best_var = rank9_var[0]
    print(f"\n  Minimum-observable rank-9 result:")
    print(f"    Y_BASIS = {best_var[3]},  probe w = e{best_var[4][0]}")
    print(f"    {best_var[5]} product rows + {best_var[6]} assoc rows = {best_var[0]} total")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

print(f"""
  Patch:   X = {X_BASIS}    Y = {Y_BASIS}

  Product observations only:
    {len(prod_buckets)} buckets, rank = {P_rank}
    Collided buckets: {sorted(collided_buckets)}
    All 9 monomials involved in at least one collision.

  Best single-probe result:""")

sp_rank9 = [(t, rk, ws, k, M) for (t, rk, ws, k, M) in single_results if rk == N_MON]
sp_rank9.sort()
if sp_rank9:
    t, rk, ws, k, M = sp_rank9[0]
    print(f"    Probe e{ws[0]}:  {len(P_rows)} prod + {len(k)} assoc = {t} rows,  rank = {rk}")
else:
    best_single = max(single_results, key=lambda x: x[1])
    print(f"    NO single probe achieves rank 9.  Best: rank={best_single[1]}")

print(f"\n  Best pair-probe result:")
pr_rank9 = [(t, rk, ws, k, M) for (t, rk, ws, k, M) in pair_results if rk == N_MON]
pr_rank9.sort()
if pr_rank9:
    t, rk, ws, k, M = pr_rank9[0]
    print(f"    Probes e{ws[0]},e{ws[1]}:  {len(P_rows)} prod + {len(k)} assoc = {t} rows,  rank = {rk}")
else:
    print(f"    NO pair achieves rank 9.")

if rank9_var:
    bv = rank9_var[0]
    print(f"\n  Global minimum (over Y-patch variants, single probe):")
    print(f"    X={bv[2]},  Y={bv[3]},  w=e{bv[4][0]}")
    print(f"    {bv[0]} total observables,  rank = {bv[1]}")
