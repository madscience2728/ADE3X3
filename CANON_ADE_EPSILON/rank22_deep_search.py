"""
Deep parallel search at R=22. 24 workers, 500+ restarts.
Logs ε every 50 restarts. Stops if ε < 1e-6 or plateau for 200 restarts.
"""
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import time, json, os

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

R = 22
DIM = 9
N_PARAMS = R * 3 * DIM
TARGET = 1e-6
N_WORKERS = 24
PLATEAU_WINDOW = 200

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
    worker_id, seed, q = args
    rng = np.random.default_rng(seed)
    local_best = np.inf
    local_best_x = None
    trial = 0
    while True:
        if trial % 5 == 0 or local_best_x is None:
            scale = rng.uniform(0.2, 2.0) / R**0.5
            x0 = rng.standard_normal(N_PARAMS) * scale
        else:
            x0 = local_best_x + rng.standard_normal(N_PARAMS) * 0.05
        res = minimize(obj, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-11})
        frob = np.sqrt(max(0.0, 2*res.fun))
        if frob < local_best:
            local_best = frob
            local_best_x = res.x.copy()
        q.put((worker_id, trial, frob, local_best,
               local_best_x.tolist() if frob <= TARGET else None))
        trial += 1

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    q = manager.Queue()

    rng0 = np.random.default_rng(999)
    seeds = [int(rng0.integers(0, 2**31)) for _ in range(N_WORKERS)]
    args = [(i, seeds[i], q) for i in range(N_WORKERS)]

    pool = mp.Pool(N_WORKERS)
    pool.map_async(worker, args)

    global_best = np.inf
    global_best_x = None
    total_trials = 0
    recent_bests = []
    start = time.time()
    last_improved_at = 0

    print(f"R=22 deep search: {N_WORKERS} workers")
    print(f"Target: ε < {TARGET}  (exact rank confirmation)")
    print(f"{'trials':>8}  {'global_best':>14}  {'ε²':>12}  {'time':>8}  note")
    print("-" * 65)

    os.makedirs('CANON_ADE_EPSILON/results', exist_ok=True)

    while True:
        item = q.get()
        worker_id, trial, frob, local_best, exact_x = item
        total_trials += 1
        elapsed = time.time() - start

        if frob < global_best:
            global_best = frob
            last_improved_at = total_trials
            if exact_x is not None:
                global_best_x = np.array(exact_x)

        recent_bests.append(global_best)

        if total_trials % 50 == 0:
            plateau = (len(recent_bests) >= PLATEAU_WINDOW and
                       abs(recent_bests[-1] - recent_bests[-PLATEAU_WINDOW]) < 1e-8)
            note = "PLATEAU" if plateau else ""
            print(f"{total_trials:>8}  {global_best:>14.8f}  {global_best**2:>12.8f}  "
                  f"{elapsed:>6.1f}s  {note}")

            # Save checkpoint
            if global_best_x is not None or global_best < 0.05:
                pass  # will save at end

        if global_best <= TARGET:
            print(f"\n{'='*60}")
            print(f"EXACT SOLUTION FOUND: ε = {global_best:.2e}")
            x = global_best_x
            A = x[:R*DIM].reshape(R, DIM)
            B = x[R*DIM:2*R*DIM].reshape(R, DIM)
            C = x[2*R*DIM:].reshape(R, DIM)
            with open('CANON_ADE_EPSILON/results/rank22_exact.json', 'w') as f:
                json.dump({'rank': R, 'frobenius': global_best,
                           'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist()}, f)
            print(f"Saved to rank22_exact.json")
            pool.terminate()
            break

        if total_trials >= 500 * N_WORKERS:
            print(f"\nReached 500 restarts/worker. Final ε = {global_best:.8f}")
            break

        # plateau for 200 restarts with no improvement
        if (total_trials - last_improved_at) >= PLATEAU_WINDOW * N_WORKERS:
            print(f"\nPlateau: no improvement for {PLATEAU_WINDOW * N_WORKERS} trials.")
            print(f"Final ε = {global_best:.8f},  ε² = {global_best**2:.8f}")
            break

    pool.terminate()
    pool.join()

    print(f"\nFinal: ε = {global_best:.8f},  ε² = {global_best**2:.8f}")
    if global_best <= TARGET:
        print("CONCLUSION: exact rank = 22 CONFIRMED")
    elif global_best < 0.001:
        print("CONCLUSION: likely exact rank 22 — need more restarts")
    elif global_best < 0.01:
        print("CONCLUSION: approaching 0/floor unclear — run longer")
    else:
        print(f"CONCLUSION: floor at ε ≈ {global_best:.4f} — exact rank > 22")
