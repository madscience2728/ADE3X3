import numpy as np

triples = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]
Sigma_full = np.zeros((27, 9))
for k, (r,s,u) in enumerate(triples):
    Sigma_full[k, r*3+u] = 1.0

orbits = {
    'O0': [k for k,(r,s,u) in enumerate(triples) if r==0 and s==0 and u==0],
    'O1': [k for k,(r,s,u) in enumerate(triples) if [r,s,u].count(0)==2 and not(r==0 and s==0 and u==0)],
    'O2': [k for k,(r,s,u) in enumerate(triples) if [r,s,u].count(0)==1],
    'O3': [k for k,(r,s,u) in enumerate(triples) if 0 not in (r,s,u)],
}

def compute_eps(kept):
    kept = sorted(kept)
    deleted = [k for k in range(27) if k not in kept]
    if not deleted:
        return 0.0
    S_kept = Sigma_full[kept]
    S_del  = Sigma_full[deleted]
    coeffs, _, _, _ = np.linalg.lstsq(S_kept.T, S_del.T, rcond=None)
    return float(np.linalg.norm(S_del.T - S_kept.T @ coeffs, 'fro'))

orbit_lists = [orbits['O0'], orbits['O1'], orbits['O2'], orbits['O3']]
orbit_sizes = [len(o) for o in orbit_lists]

print(f"Orbit sizes: O0={orbit_sizes[0]}, O1={orbit_sizes[1]}, O2={orbit_sizes[2]}, O3={orbit_sizes[3]}")
print()
print(f"{'R':>3}  {'eps*':>12}  {'b*':>8}  precision")
print("-"*55)

rng = np.random.default_rng(42)

results = {}
for R in range(1, 27):
    best_eps = np.inf

    # G-stable subsets (unions of complete orbits of size R)
    for mask in range(1, 16):
        kept = []
        for i in range(4):
            if mask & (1 << i):
                kept.extend(orbit_lists[i])
        if len(kept) == R:
            eps = compute_eps(kept)
            if eps < best_eps:
                best_eps = eps

    # Random non-stable subsets (5000 samples)
    for _ in range(5000):
        kept = rng.choice(27, R, replace=False).tolist()
        eps = compute_eps(kept)
        if eps < best_eps:
            best_eps = eps

    M = 1.0
    if best_eps < 1e-12:
        b_req = 0.0
        prec = 'exact'
    else:
        b_req = float(np.log2(3 * R * M**2 / best_eps))
        if b_req <= 7:
            prec = 'bfloat16 (b<=7)'
        elif b_req <= 10:
            prec = 'float16  (b<=10)'
        elif b_req <= 23:
            prec = 'float32  (b<=23)'
        else:
            prec = f'NEED b={b_req:.1f}'

    results[R] = (best_eps, b_req, prec)
    print(f"{R:3d}  {best_eps:12.8f}  {b_req:8.3f}  {prec}")

print(f" 27  {'0.00000000':>12}  {'0.000':>8}  exact")
print()
print("=== Summary by precision tier ===")
exact = [R for R,(e,b,p) in results.items() if e < 1e-12]
bf16  = [R for R,(e,b,p) in results.items() if 1e-12 <= e and b <= 7]
fp16  = [R for R,(e,b,p) in results.items() if b > 7 and b <= 10]
fp32  = [R for R,(e,b,p) in results.items() if b > 10 and b <= 23]
need_more = [R for R,(e,b,p) in results.items() if b > 23]
print(f"  Exact (eps=0):      R = {exact}")
print(f"  bfloat16 (b<=7):    R = {bf16}")
print(f"  float16  (b<=10):   R = {fp16}")
print(f"  float32  (b<=23):   R = {fp32}")
print(f"  Need b>23:          R = {need_more}")
