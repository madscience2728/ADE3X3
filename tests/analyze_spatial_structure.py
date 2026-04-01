"""Analyze spatial structure of the best candidate's residual."""
import json, numpy as np

bp = r'outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json'
with open(bp) as f:
    data = json.load(f)

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

candidate = np.zeros((9,9,9))
for term in data['terms']:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate += np.einsum('c,a,b->cab', g, a, b)
R = candidate - T

# ── LIVE entries through (r,s,u) lens ──
print("=== LIVE ENTRY RESIDUALS as 3x3x3 cube (r,s,u) ===")
print("Each 3x3 slice is a fixed r value, rows=s, cols=u\n")
for r in range(3):
    print(f"  r={r} (output row):")
    for s in range(3):
        row = []
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            row.append(f"{R[c,a,b]:+.4f}")
        print(f"    s={s}: [{', '.join(row)}]")
    print()

# ── Flat 3x9 grid: r vs (s,u) ──
print("=== LIVE RESIDUALS: rows=r, cols=(s,u) flat ===")
hdr = "        " + "  ".join(f"({s},{u})" for s in range(3) for u in range(3))
print(hdr)
for r in range(3):
    vals = []
    for s in range(3):
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            vals.append(f"{R[c,a,b]:+.4f}")
    print(f"  r={r}: {'  '.join(vals)}")

# ── Checkerboard: (r+s+u) parity ──
print("\n=== CHECKERBOARD TEST: (r+s+u) parity ===")
even_res, odd_res = [], []
for r in range(3):
    for s in range(3):
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            val = R[c,a,b]
            if (r+s+u) % 2 == 0:
                even_res.append(val)
            else:
                odd_res.append(val)
print(f"Even parity (r+s+u even): {len(even_res)} entries")
print(f"  mean residual: {np.mean(even_res):+.4f}")
print(f"  mean |residual|: {np.mean(np.abs(even_res)):.4f}")
print(f"  max |residual|: {np.max(np.abs(even_res)):.4f}")
print(f"Odd parity (r+s+u odd): {len(odd_res)} entries")
print(f"  mean residual: {np.mean(odd_res):+.4f}")
print(f"  mean |residual|: {np.mean(np.abs(odd_res)):.4f}")
print(f"  max |residual|: {np.max(np.abs(odd_res)):.4f}")

# ── Diagonal vs off-diagonal ──
print("\n=== DIAGONAL vs OFF-DIAGONAL ===")
pure_diag, partial_diag, off_diag = [], [], []
for r in range(3):
    for s in range(3):
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            val = R[c,a,b]
            if r==s==u:
                pure_diag.append(val)
            elif r==s or s==u or r==u:
                partial_diag.append(val)
            else:
                off_diag.append(val)
print(f"Pure diagonal r=s=u (3):     mean|R|={np.mean(np.abs(pure_diag)):.4f}  vals: {[f'{v:+.4f}' for v in pure_diag]}")
print(f"Partial diagonal (12):       mean|R|={np.mean(np.abs(partial_diag)):.4f}")
print(f"Full off-diagonal (12):      mean|R|={np.mean(np.abs(off_diag)):.4f}")

# ── S3 orbit analysis: which permutation class of (r,s,u)? ──
print("\n=== S3 SYMMETRY ORBIT ANALYSIS ===")
# classify each (r,s,u) by its sorted multiset type
from collections import defaultdict
orbits = defaultdict(list)
for r in range(3):
    for s in range(3):
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            val = R[c,a,b]
            key = tuple(sorted([r,s,u]))
            orbits[key].append(val)

print(f"{'Orbit (sorted)':>15} | count | mean(R) | mean|R| | max|R| | std(R)")
for key in sorted(orbits.keys()):
    vs = orbits[key]
    print(f"  {key!s:>13} | {len(vs):5d} | {np.mean(vs):+.4f} | {np.mean(np.abs(vs)):.4f} | {np.max(np.abs(vs)):.4f} | {np.std(vs):.4f}")

# ── Index-level coloring: which value of r (or s, or u) carries most error ──
print("\n=== PER-INDEX BREAKDOWN ===")
for name, idx_fn in [("r", lambda r,s,u: r), ("s", lambda r,s,u: s), ("u", lambda r,s,u: u)]:
    buckets = defaultdict(list)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                c,a,b = r*3+u, r*3+s, s*3+u
                buckets[idx_fn(r,s,u)].append(R[c,a,b])
    print(f"  By {name}:")
    for k in range(3):
        vs = buckets[k]
        print(f"    {name}={k}: mean|R|={np.mean(np.abs(vs)):.4f}  mean(R)={np.mean(vs):+.4f}  max|R|={np.max(np.abs(vs)):.4f}")

# ── DEAD entries: which (r,s,u) triples leak most? ──
print("\n=== DEAD ENTRY ANALYSIS (702 entries) ===")
dead_abs = np.abs(R.copy())
# Zero out the 27 live entries
for r in range(3):
    for s in range(3):
        for u in range(3):
            dead_abs[r*3+u, r*3+s, s*3+u] = 0.0

# Flatten the 9x9x9 dead residual and look at the pattern per axis
print(f"Dead max|R|: {dead_abs.max():.4f}")
print(f"Dead mean|R|: {dead_abs[dead_abs > 0].mean():.6f}")
print(f"Dead ||R||^2: {np.sum(dead_abs[dead_abs > 0]**2):.4f}")

# Which c-slice (output matrix element) has worst dead leakage?
print("\n  Dead leakage by c-index (output element):")
for c in range(9):
    sl = dead_abs[c,:,:]
    nz = sl[sl > 0]
    if len(nz) > 0:
        print(f"    c={c}: max={nz.max():.4f}  mean={nz.mean():.4f}  count={len(nz)}")

# Which a-slice has worst dead leakage?
print("\n  Dead leakage by a-index (left matrix element):")
for a in range(9):
    sl = dead_abs[:,a,:]
    nz = sl[sl > 0]
    if len(nz) > 0:
        print(f"    a={a}: max={nz.max():.4f}  mean={nz.mean():.4f}  count={len(nz)}")

print("\n=== SIGN PATTERN ON LIVE ENTRIES ===")
for r in range(3):
    print(f"  r={r}:")
    for s in range(3):
        signs = []
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            v = R[c,a,b]
            signs.append("+" if v >= 0 else "-")
        print(f"    s={s}: [{', '.join(signs)}]")

# ── BIMODAL CLASSIFICATION ──
print("\n=== BIMODAL CLASSIFICATION of live entries ===")
print("  [0] = |R| < 0.01   (solved)")
print("  [H] = |R| > 0.45   (half-missing, ~-0.5)")
print("  [~] = intermediate\n")
for r in range(3):
    print(f"  r={r}:     u=0  u=1  u=2")
    for s in range(3):
        tags = []
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            v = abs(R[c,a,b])
            if v < 0.01: tags.append(" 0 ")
            elif v > 0.45: tags.append(" H ")
            else: tags.append(f"{R[c,a,b]:+.1f}")
        sep = "  "
        print(f"    s={s}: [{sep.join(tags)}]")
    print()

solved = near_half = mid = 0
for r in range(3):
    for s in range(3):
        for u in range(3):
            v = abs(R[r*3+u, r*3+s, s*3+u])
            if v < 0.01: solved += 1
            elif v > 0.45: near_half += 1
            else: mid += 1
print(f"Solved (|R|<0.01): {solved}")
print(f"Near-half (|R|>0.45): {near_half}")
print(f"Intermediate: {mid}")

print("\n=== SOLVED ENTRIES as (r,s,u) triples ===")
for r in range(3):
    for s in range(3):
        for u in range(3):
            if abs(R[r*3+u, r*3+s, s*3+u]) < 0.01:
                print(f"  ({r},{s},{u})  R={R[r*3+u, r*3+s, s*3+u]:+.4f}")

print("\n=== HALF-MISSING ENTRIES as (r,s,u) triples ===")
for r in range(3):
    for s in range(3):
        for u in range(3):
            c,a,b = r*3+u, r*3+s, s*3+u
            if abs(R[c,a,b]) > 0.45:
                print(f"  ({r},{s},{u})  R={R[c,a,b]:+.4f}")
