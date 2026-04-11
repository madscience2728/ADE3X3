"""Parity theorem test v2: doubled budget, ranks 13/19/20/21/22."""
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import time

T = np.zeros((9,9,9))
ORBITS = {0: [], 1: [], 2: [], 3: []}
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
            zeros = (r == 0) + (s == 0) + (u == 0)
            if zeros == 3: ORBITS[0].append((r,s,u))
            elif zeros == 2: ORBITS[1].append((r,s,u))
            elif zeros == 1: ORBITS[2].append((r,s,u))
            else: ORBITS[3].append((r,s,u))

DIM = 9
BASE = ORBITS[0] + ORBITS[1] + ORBITS[2]  # 19 terms
assert len(BASE) == 19

O3_EVEN = [(r,s,u) for r in range(1,3) for s in range(1,3) for u in range(1,3) if (r+s+u) % 2 == 0]
O3_ODD  = [(r,s,u) for r in range(1,3) for s in range(1,3) for u in range(1,3) if (r+s+u) % 2 == 1]

def obj(x, R):
    A = x[:R*DIM].reshape(R,DIM)
    B = x[R*DIM:2*R*DIM].reshape(R,DIM)
    C = x[2*R*DIM:].reshape(R,DIM)
    E = np.einsum('ki,kj,kl->ijl',A,B,C) - T
    loss = 0.5*float(np.dot(E.ravel(),E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki',E,B,C)
    dB = np.einsum('ijl,ki,kl->kj',E,A,C)
    dC = np.einsum('ijl,ki,kj->kl',E,A,B)
    return loss, np.concatenate([dA.ravel(),dB.ravel(),dC.ravel()])

def run(args):
    label, R, terms, seed = args
    rng = np.random.default_rng(seed)
    A = np.zeros((R,DIM)); B = np.zeros((R,DIM)); C = np.zeros((R,DIM))
    sc = 1.0/R**0.5
    for k,(r,s,u) in enumerate(terms):
        A[k,3*r+s] = rng.choice([-1.,1.])*(1+0.2*rng.standard_normal())
        B[k,3*s+u] = rng.choice([-1.,1.])*(1+0.2*rng.standard_normal())
        C[k,3*r+u] = rng.choice([-1.,1.])*(1+0.2*rng.standard_normal())
        A[k] += rng.standard_normal(DIM)*0.05*sc
        B[k] += rng.standard_normal(DIM)*0.05*sc
        C[k] += rng.standard_normal(DIM)*0.05*sc
    x0 = np.concatenate([A.ravel(),B.ravel(),C.ravel()])
    res = minimize(lambda x: obj(x,R), x0, jac=True, method='L-BFGS-B',
                   options={'maxiter':8000,'ftol':1e-15,'gtol':1e-12})
    Ar = res.x[:R*DIM].reshape(R,DIM)
    Br = res.x[R*DIM:2*R*DIM].reshape(R,DIM)
    Cr = res.x[2*R*DIM:].reshape(R,DIM)
    ef = float(np.linalg.norm(np.einsum('ki,kj,kl->ijl',Ar,Br,Cr)-T))
    return label, ef

def main():
    RESTARTS = 400  # doubled budget

    configs = {
        # No O3 headroom — baselines
        'R13_O0O2':           (13, ORBITS[0] + ORBITS[2]),
        'R19_BASE':           (19, BASE),
        # 1 O3 term
        'R20_1xO3_even':      (20, BASE + [O3_EVEN[0]]),           # (1,1,1)
        'R20_1xO3_odd':       (20, BASE + [O3_ODD[0]]),            # (2,2,2)
        # 2 O3 terms — parity comparison
        'R21_same_even':      (21, BASE + O3_EVEN[:2]),             # (1,1,1)+(1,2,2)
        'R21_same_odd':       (21, BASE + O3_ODD[:2]),              # (2,2,2)+(2,1,1)
        'R21_mixed':          (21, BASE + [O3_EVEN[0], O3_ODD[0]]),# (1,1,1)+(2,2,2)
        # 3 O3 terms — parity comparison
        'R22_same_even':      (22, BASE + O3_EVEN[:3]),             # all even
        'R22_same_odd':       (22, BASE + O3_ODD[:3]),              # all odd
        'R22_mixed_2e1o':     (22, BASE + O3_EVEN[:2] + O3_ODD[:1]),
        'R22_mixed_1e2o':     (22, BASE + O3_EVEN[:1] + O3_ODD[:2]),
    }

    tasks = []
    for label, (R, terms) in configs.items():
        assert len(terms) == R, f"{label}: expected {R}, got {len(terms)}"
        for i in range(RESTARTS):
            tasks.append((label, R, terms, hash((label, i)) & 0xFFFFFFFF))

    total = len(tasks)
    print(f'Running {total} tasks ({RESTARTS} restarts x {len(configs)} configs), 24 workers...')
    for label, (R, terms) in configs.items():
        o3_terms = [t for t in terms if all(x > 0 for x in t)]
        parities = [(r+s+u) % 2 for r,s,u in o3_terms]
        ptag = 'none' if not parities else ('SAME' if len(set(parities)) == 1 else 'MIXED')
        print(f'  {label:25s}  R={R:2d}  O3={len(o3_terms)}  parity={ptag}  terms={o3_terms}')

    t0 = time.time()
    results = {}
    done = 0
    with Pool(24) as pool:
        for label, ef in pool.imap_unordered(run, tasks):
            results.setdefault(label, []).append(ef)
            done += 1
            if done % 200 == 0:
                print(f'  [{done}/{total}]', flush=True)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s\n')
    print('=' * 72)
    print(f'{"config":25s} {"R":>3s} {"O3":>3s} {"parity":>6s} {"best_eF":>10s} {"best_eF2":>10s} {"mean_eF":>10s}')
    print('-' * 72)
    for label in configs:
        vals = results[label]
        R, terms = configs[label]
        o3_terms = [t for t in terms if all(x > 0 for x in t)]
        parities = [(r+s+u) % 2 for r,s,u in o3_terms]
        ptag = 'none' if not parities else ('SAME' if len(set(parities)) == 1 else 'MIXED')
        print(f'{label:25s} {R:3d} {len(o3_terms):3d} {ptag:>6s} {min(vals):10.6f} {min(vals)**2:10.6f} {np.mean(vals):10.6f}')

if __name__ == '__main__':
    main()
