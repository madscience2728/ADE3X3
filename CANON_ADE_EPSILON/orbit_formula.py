"""
Search for combinatorial formula: ε*(R)² = f(orbit structure of deleted terms)
"""
import numpy as np
from fractions import Fraction

sizes = [1, 6, 12, 8]   # O0, O1, O2, O3
names = ['O0', 'O1', 'O2', 'O3']
total = 27

eps_data = {13: 3.321242, 19: 1.742612, 20: 1.442099, 21: 1.004765, 22: 0.133602}

def orbit_split(R):
    kept_counts = {}
    del_counts = {}
    remaining = R
    for nm, sz in zip(names, sizes):
        if remaining >= sz:
            kept_counts[nm] = sz
            del_counts[nm] = 0
            remaining -= sz
        elif remaining > 0:
            kept_counts[nm] = remaining
            del_counts[nm] = sz - remaining
            remaining = 0
        else:
            kept_counts[nm] = 0
            del_counts[nm] = sz
    return kept_counts, del_counts

print(f"{'R':>3}  {'ε²':>10}  {'del_O0':>6} {'del_O1':>6} {'del_O2':>6} {'del_O3':>6}  {'d_total':>7}")
print('-'*70)
for R, eps in sorted(eps_data.items()):
    sq = eps**2
    kc, dc = orbit_split(R)
    d = sum(dc.values())
    print(f"{R:>3}  {sq:>10.6f}  {dc['O0']:>6} {dc['O1']:>6} {dc['O2']:>6} {dc['O3']:>6}  {d:>7}")

print()
print("--- Testing ε² = (27-R) / X ---")
print(f"{'R':>3}  {'27-R':>5}  {'ε²':>10}  {'X = (27-R)/ε²':>16}  {'nearest fraction':>20}")
for R, eps in sorted(eps_data.items()):
    sq = eps**2
    numer = 27 - R
    X = numer / sq
    frac = Fraction(X).limit_denominator(500)
    err = abs(float(frac) - X)
    print(f"{R:>3}  {numer:>5}  {sq:>10.6f}  {X:>16.6f}  {str(frac):>20}  err={err:.5f}")

print()
print("--- Testing ε² = N / (27-R) ---")
for R, eps in sorted(eps_data.items()):
    sq = eps**2
    denom = 27 - R
    N = sq * denom
    frac = Fraction(N).limit_denominator(500)
    err = abs(float(frac) - N)
    print(f"  R={R}: ε²×(27-{R}) = {N:.6f}  → {frac}  err={err:.5f}")

print()
print("--- Testing ε² × various combos ---")
for R, eps in sorted(eps_data.items()):
    sq = eps**2
    kc, dc = orbit_split(R)
    d = sum(dc.values())
    k = R
    hits = []
    for num_name, num_val in [
        ('d',d), ('d²',d*d), ('d(d-1)',d*(d-1)), ('d(d+1)',d*(d+1)),
        ('k',k), ('k²',k*k), ('k(k-1)',k*(k-1)),
        ('dk',d*k), ('27d',27*d), ('d·k/27', d*k),
        ('(27-R)',27-R), ('R(27-R)',R*(27-R)),
    ]:
        if num_val == 0: continue
        ratio = sq * num_val
        frac = Fraction(ratio).limit_denominator(500)
        if abs(float(frac) - ratio) < 0.003 * max(1, ratio):
            hits.append(f"ε²·{num_name}={num_val}≈{frac}")
    sep = "  |  "
    print(f"  R={R}: " + sep.join(hits))
