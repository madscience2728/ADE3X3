"""Parity theorem confirmation: two predictions."""
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import time

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
DIM = 9
BASE = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            if (r==0)+(s==0)+(u==0) >= 1:
                BASE.append((r,s,u))

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
    # Test 1: R=21 antipodal pair (d=3, parity-mixed)
    antipodal = [(1,1,1),(2,2,2)]  # even + odd
    # Test 2: R=22 equilateral triple (same parity = even)
    equilateral = [(1,1,1),(1,2,2),(2,1,2)]  # all even

    RESTARTS = 200
    tasks = []
    for i in range(RESTARTS):
        tasks.append(('R21_antipodal', 21, BASE+antipodal, i))
        tasks.append(('R22_equilateral', 22, BASE+equilateral, i+10000))

    print(f'Running {len(tasks)} tasks (200 each x 2 configs), 24 workers...')
    print(f'  R=21 antipodal pair: (1,1,1)+(2,2,2) — parity MIXED')
    print(f'  R=22 equilateral triple: (1,1,1)+(1,2,2)+(2,1,2) — parity SAME (even)')
    print(f'  Emmy predicts: R21_antipodal ~ 0.06, R22_equilateral ~ 1.0')
    t0 = time.time()
    results = {}
    done = 0
    with Pool(24) as pool:
        for label, ef in pool.imap_unordered(run, tasks):
            results.setdefault(label, []).append(ef)
            done += 1
            if done % 100 == 0:
                print(f'  [{done}/{len(tasks)}]', flush=True)

    print(f'\nDone in {time.time()-t0:.1f}s\n')
    print('=' * 60)
    for label in ['R21_antipodal', 'R22_equilateral']:
        vals = results[label]
        parity = 'MIXED' if 'antipodal' in label else 'SAME (even)'
        print(f'{label}:')
        print(f'  parity   = {parity}')
        print(f'  best eF  = {min(vals):.8f}')
        print(f'  best eF2 = {min(vals)**2:.8f}')
        print(f'  mean eF  = {np.mean(vals):.8f}')
        print()

if __name__ == '__main__':
    main()
