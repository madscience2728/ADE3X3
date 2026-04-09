"""
Multi-rank deep search: runs R=21, 20, 19, 13 simultaneously.
Allocates 6 workers per rank, runs until 200 consecutive non-improving
restarts or 2 hours, checkpoints best per rank.
"""
import numpy as np
import json
import os
import time
from datetime import datetime
from multiprocessing import Pool, Manager
from scipy.optimize import minimize

# ── Target tensor ─────────────────────────────────────────────────────────────
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_FLAT = T.ravel()
DIM = 9

RANKS = [21, 20, 19, 13]
WORKERS_PER_RANK = 6
MAX_CONSEC_PLATEAU = 200
WALL_SECONDS = 7200  # 2 hours

CKPT_DIR = 'CANON_ADE_EPSILON/results/cliff_search'
os.makedirs(CKPT_DIR, exist_ok=True)

BF16_EPS = 2**-7

# ── Objective ─────────────────────────────────────────────────────────────────
def obj(x, R):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
    loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC = np.einsum('ijl,ki,kj->kl', E, A, B)
    return loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])

def frob(x, R):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    return float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', A, B, C) - T))

def one_restart(args):
    R, seed, x_warm = args
    rng = np.random.default_rng(seed)
    n = 3 * R * DIM

    if x_warm is not None and rng.random() < 0.4:
        # Perturb incumbent
        sigma = rng.uniform(0.001, 0.1)
        x0 = x_warm + rng.standard_normal(n) * sigma
    else:
        scale = rng.uniform(0.3, 1.2) / R**0.5
        x0 = rng.standard_normal(n) * scale

    res = minimize(lambda x: obj(x, R), x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-11})
    f = frob(res.x, R)
    return f, res.x


def search_rank(args):
    R, shared_state, start_time = args
    rng = np.random.default_rng(R * 1000 + os.getpid())
    best_f = shared_state.get(f'best_{R}', np.inf)
    best_x = shared_state.get(f'bestx_{R}', None)
    consec = 0
    trial = 0

    ckpt_path = os.path.join(CKPT_DIR, f'rank{R}_best.json')

    while True:
        elapsed = time.time() - start_time
        if elapsed > WALL_SECONDS:
            break
        if consec >= MAX_CONSEC_PLATEAU:
            break

        seed = int(rng.integers(0, 2**31))
        f, x = one_restart((R, seed, best_x))
        trial += 1

        if f < best_f:
            best_f = f
            best_x = x.copy()
            consec = 0
            # Update shared state
            shared_state[f'best_{R}'] = best_f
            shared_state[f'bestx_{R}'] = best_x
            # Checkpoint
            A = x[:R*DIM].reshape(R, DIM)
            B = x[R*DIM:2*R*DIM].reshape(R, DIM)
            C = x[2*R*DIM:].reshape(R, DIM)
            with open(ckpt_path, 'w') as fp:
                json.dump({'rank': R, 'frobenius': best_f,
                           'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist()}, fp)
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] R={R:2d}  trial={trial:5d}  NEW BEST ε={best_f:.8f}  '
                  f'min_b={np.log2(3*R/max(best_f,1e-12)):.2f}  elapsed={elapsed:.0f}s', flush=True)
        else:
            consec += 1

    return R, best_f, best_x, trial


def worker_entry(args):
    """Each worker is one of 6 for a given rank."""
    R, worker_id, shared_state, start_time = args
    rng = np.random.default_rng(R * 10000 + worker_id + int(time.time()) % 1000)
    consec_key = f'consec_{R}'
    best_key = f'best_{R}'
    bestx_key = f'bestx_{R}'
    trial = 0

    ckpt_path = os.path.join(CKPT_DIR, f'rank{R}_best.json')

    while True:
        elapsed = time.time() - start_time
        if elapsed > WALL_SECONDS:
            break
        consec = shared_state.get(consec_key, 0)
        if consec >= MAX_CONSEC_PLATEAU:
            break

        best_x = shared_state.get(bestx_key, None)
        seed = int(rng.integers(0, 2**31))
        f, x = one_restart((R, seed, best_x))
        trial += 1

        global_best = shared_state.get(best_key, np.inf)
        if f < global_best:
            shared_state[best_key] = f
            shared_state[bestx_key] = x.copy()
            shared_state[consec_key] = 0

            A = x[:R*DIM].reshape(R, DIM)
            B_m = x[R*DIM:2*R*DIM].reshape(R, DIM)
            C_m = x[2*R*DIM:].reshape(R, DIM)
            with open(ckpt_path, 'w') as fp:
                json.dump({'rank': R, 'frobenius': float(f),
                           'alpha': A.tolist(), 'beta': B_m.tolist(), 'gamma': C_m.tolist()}, fp)
            ts = datetime.now().strftime('%H:%M:%S')
            min_b = np.log2(3*R/max(f, 1e-12))
            print(f'[{ts}] R={R:2d} w{worker_id}  NEW BEST ε={f:.8f}  '
                  f'min_b={min_b:.2f}  elapsed={elapsed:.0f}s', flush=True)
        else:
            old = shared_state.get(consec_key, 0)
            shared_state[consec_key] = old + 1

    return R, worker_id, trial


def print_summary(shared_state, start_time):
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'{"R":>4}  {"ε":>12}  {"min_b":>8}  {"Status at each precision"}')
    print('-'*70)
    for R in RANKS:
        f = shared_state.get(f'best_{R}', np.inf)
        if f == np.inf or f <= 0:
            print(f'{R:>4}  {"no result":>12}')
            continue
        min_b = np.log2(3*R / f)
        # Which formats close?
        formats = [('int4',4), ('int8',8), ('bf16',7), ('fp16',10), ('fp32',23)]
        closed = [name for name,b in formats if 3*R*2**(-b) >= f]
        open_ = [name for name,b in formats if 3*R*2**(-b) < f]
        consec = shared_state.get(f'consec_{R}', 0)
        status = 'PLATEAUED' if consec >= MAX_CONSEC_PLATEAU else 'STILL RUNNING'
        print(f'{R:>4}  {f:>12.8f}  {min_b:>8.2f}  closed={closed}  open={open_}')
        print(f'       elapsed={time.time()-start_time:.0f}s  consecutive_no_improve={consec}  [{status}]')
    print('='*70)


if __name__ == '__main__':
    print(f'Cliff search: R={RANKS}  {WORKERS_PER_RANK} workers/rank  '
          f'plateau={MAX_CONSEC_PLATEAU}  wall={WALL_SECONDS//3600}h', flush=True)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    start_time = time.time()

    with Manager() as manager:
        shared = manager.dict()
        for R in RANKS:
            shared[f'best_{R}'] = np.inf
            shared[f'bestx_{R}'] = None
            shared[f'consec_{R}'] = 0

        # Build worker args: 6 workers per rank
        worker_args = []
        for R in RANKS:
            for w in range(WORKERS_PER_RANK):
                worker_args.append((R, w, shared, start_time))

        total_workers = len(RANKS) * WORKERS_PER_RANK  # 24
        print(f'Launching {total_workers} workers...', flush=True)

        with Pool(processes=total_workers) as pool:
            results = pool.map(worker_entry, worker_args)

        print_summary(shared, start_time)
        elapsed = time.time() - start_time
        print(f'\nTotal elapsed: {elapsed:.1f}s = {elapsed/3600:.2f}h')
