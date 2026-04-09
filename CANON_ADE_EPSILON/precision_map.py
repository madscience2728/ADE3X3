import numpy as np, json

with open('CANON_ADE_EPSILON/results/rank22_best.json') as f:
    d = json.load(f)
A = np.array(d['alpha']); B = np.array(d['beta']); C = np.array(d['gamma'])
R = d['rank']; eps_F = d['frobenius']

rng = np.random.default_rng(42)
N = 1000
e_approx = []
for _ in range(N):
    M1 = rng.uniform(-1,1,(3,3)); M2 = rng.uniform(-1,1,(3,3))
    exact = M1 @ M2
    c = (C.T @ ((A @ M1.ravel()) * (B @ M2.ravel()))).reshape(3,3)
    e_approx.append(np.linalg.norm(exact - c, 'fro'))
e_approx = np.array(e_approx)

print(f'R={R} approximation (float64, N={N} pairs):')
print(f'  eps_F          = {eps_F:.8f}')
print(f'  mean Frob err  = {e_approx.mean():.6e}')
print(f'  max  Frob err  = {e_approx.max():.6e}')
print()

min_b = np.log2(1.0 / (eps_F / (3 * R)))
print(f'Min mantissa bits for closure: b <= {min_b:.2f}')
print(f'  -> closes at 13-bit arithmetic and coarser')
print()

formats = [
    ('float32',  23),
    ('float16',  10),
    ('bfloat16',  7),
    ('int8',      8),
    ('int4',      4),
    ('int2',      2),
]
header = '{:<12} {:>4}  {:>17}  {:>13}  {}'.format('Precision', 'b', 'Budget 3*R*2^-b', 'Budget/eps_F', 'Status')
print(header)
print('-'*65)
for name, b in formats:
    budget = 3 * R * 2**(-b)
    ratio = budget / eps_F
    status = 'CLOSED' if ratio >= 1.0 else 'OPEN'
    marker = '  <- boundary' if 0.5 < ratio < 2.0 else ''
    row = '{:<12} {:>4}  {:>17.6f}  {:>13.1f}x  {}{}'.format(name, b, budget, ratio, status, marker)
    print(row)
