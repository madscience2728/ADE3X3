"""
Parallel Frobenius minimization at R=22, testing ε*(22) = 0.
24 workers, continuous restarts until ε < 1e-4 or timeout.
"""
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import time, os, json

SQRT3 = np.sqrt(3)
R = 22
DIM = 9
N_PARAMS = R * 3 * DIM

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

TARGET_EXACT = 1e-4
N_WORKERS = 24
TIMEOUT = 600  # seconds


def obj(x):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
    loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC = np.einsum('ijl,ki,kj->kl', E, A, B)
    return loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])


def worker(args):
    worker_id, seed, shared_best, result_queue = args
    rng = np.random.default_rng(seed)
    trial = 0
    local_best = np.inf
    local_best_x = None

    while True:
        scale = rng.uniform(0.3, 1.5) / R**0.5
        x0 = rng.standard_normal(N_PARAMS) * scale

        # occasionally perturb around local best
        if local_best_x is not None and trial % 5 == 0:
            noise = rng.standard_normal(N_PARAMS) * 0.1
            x0 = local_best_x + noise

        res = minimize(obj, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10})
        frob = np.sqrt(max(0.0, 2 * res.fun))

        if frob < local_best:
            local_best = frob
            local_best_x = res.x.copy()

        result_queue.put((worker_id, trial, frob, local_best))

        if frob <= TARGET_EXACT:
            result_queue.put(('EXACT', worker_id, frob, res.x.tolist()))
            return

        # check if global best is exact — stop
        try:
            gb = shared_best.value
            if gb <= TARGET_EXACT:
                return
        except Exception:
            pass

        trial += 1


def listener(result_queue, shared_best, start_time):
    global_best = np.inf
    global_best_x = None
    trial_count = 0
    last_print = 0.0
    buckets = {}  # frob rounded to 2dp -> count

    while True:
        item = result_queue.get()
        if item[0] == 'DONE':
            break

        if item[0] == 'EXACT':
            _, wid, frob, x = item
            print(f"\n{'='*60}")
            print(f"  EXACT SOLUTION FOUND by worker {wid}: ε = {frob:.8f}")
            print(f"{'='*60}")
            os.makedirs('CANON_ADE_EPSILON/results', exist_ok=True)
            A = np.array(x[:R*DIM]).reshape(R, DIM)
            B = np.array(x[R*DIM:2*R*DIM]).reshape(R, DIM)
            C = np.array(x[2*R*DIM:]).reshape(R, DIM)
            with open('CANON_ADE_EPSILON/results/rank22_exact.json', 'w') as f:
                json.dump({'rank': R, 'frobenius': frob,
                           'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist()}, f)
            shared_best.value = frob
            continue

        wid, trial, frob, local_best = item
        trial_count += 1

        if frob < global_best:
            global_best = frob
            try:
                shared_best.value = frob
            except Exception:
                pass

        bucket = round(frob, 2)
        buckets[bucket] = buckets.get(bucket, 0) + 1

        now = time.time() - start_time
        if now - last_print >= 5.0:
            last_print = now
            ratio = global_best / SQRT3
            top5 = sorted(buckets.items())[:8]
            basin_str = '  '.join(f'{b:.2f}×{c}' for b,c in top5)
            print(f"  t={now:5.0f}s  trials={trial_count:5d}  "
                  f"global_best={global_best:.6f}  /√3={ratio:.4f}  "
                  f"basins: {basin_str}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()
    shared_best = manager.Value('d', np.inf)

    print(f"R=22 parallel search: {N_WORKERS} workers, target ε < {TARGET_EXACT}")
    print(f"√3 = {SQRT3:.6f},  √(22-22) = 0  (exact rank conjecture)")
    print(f"Starting {N_WORKERS} workers...\n")

    start = time.time()

    seeds = [int(np.random.default_rng(i).integers(0, 2**31)) for i in range(N_WORKERS)]
    args = [(i, seeds[i], shared_best, result_queue) for i in range(N_WORKERS)]

    pool = mp.Pool(N_WORKERS)
    pool.map_async(worker, args)

    listen_proc = mp.Process(target=listener, args=(result_queue, shared_best, start))
    listen_proc.start()

    deadline = start + TIMEOUT
    while time.time() < deadline:
        try:
            if shared_best.value <= TARGET_EXACT:
                print("Exact solution confirmed. Terminating.")
                break
        except Exception:
            pass
        time.sleep(2)

    pool.terminate()
    pool.join()
    result_queue.put(('DONE',))
    listen_proc.join(timeout=5)

    elapsed = time.time() - start
    try:
        gb = shared_best.value
    except Exception:
        gb = np.inf

    print(f"\nFinal: best ε = {gb:.6f}  (√3={SQRT3:.6f}, ratio={gb/SQRT3:.4f})")
    print(f"Elapsed: {elapsed:.1f}s")
    if gb <= TARGET_EXACT:
        print("CONCLUSION: ε*(22) = 0  — exact rank IS 22")
    elif gb < 0.01:
        print("CONCLUSION: very close to 0 — likely exact rank 22, need more time")
    else:
        print(f"CONCLUSION: ε*(22) ≈ {gb:.4f} — exact rank > 22")
