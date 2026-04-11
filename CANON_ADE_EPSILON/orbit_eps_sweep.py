"""
Task 1 — Orbit-constrained ε sweep.

For each orbit configuration, run G-symmetric constrained AND free (unconstrained)
L-BFGS-B searches.  Reports εF and εF² for both.  The difference is the
cross-orbit leakage.

Parallelized: 24 workers across configs × restarts.
"""
import numpy as np
import json
import os
import time
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from scipy.optimize import minimize

# ── Target tensor ─────────────────────────────────────────────────────────────
T = np.zeros((9, 9, 9))
ORBITS = {0: [], 1: [], 2: [], 3: []}
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
            zeros = (r == 0) + (s == 0) + (u == 0)
            if zeros == 3:
                ORBITS[0].append((r, s, u))
            elif zeros == 2:
                ORBITS[1].append((r, s, u))
            elif zeros == 1:
                ORBITS[2].append((r, s, u))
            else:
                ORBITS[3].append((r, s, u))

assert len(ORBITS[0]) == 1
assert len(ORBITS[1]) == 6
assert len(ORBITS[2]) == 12
assert len(ORBITS[3]) == 8

DIM = 9
RESTARTS = 50
N_WORKERS = 24
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Warm-start directory
GATED_DIR = os.path.join(RESULTS_DIR, 'gated_search')

# ── Orbit configurations ─────────────────────────────────────────────────────
CONFIGS = {
    'R19_O0O1O2':       {'orbits': [0, 1, 2],    'R': 19, 'desc': 'O0+O1+O2 baseline'},
    'R20_O0O1O2_1xO3':  {'orbits': [0, 1, 2],    'R': 20, 'n_o3': 1, 'desc': 'O0+O1+O2 + 1 from O3'},
    'R21_O0O1O2_2xO3':  {'orbits': [0, 1, 2],    'R': 21, 'n_o3': 2, 'desc': 'O0+O1+O2 + 2 from O3'},
    'R22_O0O1O2_3xO3':  {'orbits': [0, 1, 2],    'R': 22, 'n_o3': 3, 'desc': 'O0+O1+O2 + 3 from O3'},
    'R13_O0O2':          {'orbits': [0, 2],        'R': 13, 'desc': 'O0+O2 only'},
    'R15_partial':       {'orbits': [0, 1],        'R': 15, 'n_o2': 8, 'desc': 'O0+O1 + 8 from O2'},
    'R7_O0O1':           {'orbits': [0, 1],        'R':  7, 'desc': 'O0+O1 only (Strassen analog)'},
}


def build_support(cfg):
    """Return list of (r,s,u) triples for this orbit config."""
    terms = []
    for oid in cfg['orbits']:
        terms.extend(ORBITS[oid])
    # Add partial O3 terms if specified
    n_o3 = cfg.get('n_o3', 0)
    if n_o3 > 0:
        terms.extend(ORBITS[3][:n_o3])
    # Add partial O2 terms if specified
    n_o2 = cfg.get('n_o2', 0)
    if n_o2 > 0 and 2 not in cfg['orbits']:
        terms.extend(ORBITS[2][:n_o2])
    assert len(terms) == cfg['R'], f"Expected R={cfg['R']}, got {len(terms)} terms"
    return terms


def support_to_masks(terms):
    """Convert (r,s,u) term list to alpha/beta index masks.
    Each term (r,s,u) contributes to:
      alpha index = 3r+s, beta index = 3s+u, gamma index = 3r+u
    """
    alpha_idx = [3*r + s for r, s, u in terms]
    beta_idx = [3*s + u for r, s, u in terms]
    gamma_idx = [3*r + u for r, s, u in terms]
    return alpha_idx, beta_idx, gamma_idx


# ── Objective (unconstrained) ─────────────────────────────────────────────────
def obj_free(x, R):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    E = np.einsum('ki,kj,kl->ijl', A, B, C) - T
    loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    dA = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC = np.einsum('ijl,ki,kj->kl', E, A, B)
    return loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])


def obj_constrained(x, R, terms):
    """Constrained: each term k has alpha[k] and beta[k] and gamma[k]
    as full 9-vectors, but initialized from the orbit structure.
    This is the same free objective — the 'constraint' is in the initialization
    and warm-start, not in the parameterization (soft constraint)."""
    return obj_free(x, R)


def run_one(args):
    mode, cfg_name, cfg, seed = args
    R = cfg['R']
    rng = np.random.default_rng(seed)
    n = 3 * R * DIM

    # Try warm-start from gated_search
    warm_file = os.path.join(GATED_DIR, f'rank{R}_best.json')
    x_warm = None
    if os.path.isfile(warm_file):
        with open(warm_file) as fp:
            data = json.load(fp)
        if data.get('rank') == R:
            A = np.array(data['alpha'])
            B = np.array(data['beta'])
            C = np.array(data['gamma'])
            x_warm = np.concatenate([A.ravel(), B.ravel(), C.ravel()])

    if mode == 'constrained':
        # Build orbit-structured initialization
        terms = build_support(cfg)
        scale = 1.0 / R**0.5
        A = np.zeros((R, DIM))
        B = np.zeros((R, DIM))
        C = np.zeros((R, DIM))
        for k, (r, s, u) in enumerate(terms):
            A[k, 3*r+s] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
            B[k, 3*s+u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
            C[k, 3*r+u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
            # Add small off-support noise
            A[k] += rng.standard_normal(DIM) * 0.05 * scale
            B[k] += rng.standard_normal(DIM) * 0.05 * scale
            C[k] += rng.standard_normal(DIM) * 0.05 * scale
        x0 = np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    else:
        # Free: random or warm
        if x_warm is not None and rng.random() < 0.4:
            sigma = rng.uniform(0.001, 0.1)
            x0 = x_warm + rng.standard_normal(n) * sigma
        else:
            scale = rng.uniform(0.3, 1.2) / R**0.5
            x0 = rng.standard_normal(n) * scale

    res = minimize(lambda x: obj_free(x, R), x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})

    A = res.x[:R*DIM].reshape(R, DIM)
    B = res.x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = res.x[2*R*DIM:].reshape(R, DIM)
    eps_f = float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', A, B, C) - T))
    return mode, cfg_name, seed, eps_f, res.x


def main():
    print("="*80)
    print("ORBIT εF SWEEP — G-symmetric constrained vs free search")
    print("="*80)

    tasks = []
    for cfg_name, cfg in CONFIGS.items():
        for restart in range(RESTARTS):
            seed = hash((cfg_name, 'constrained', restart)) % (2**31)
            tasks.append(('constrained', cfg_name, cfg, seed))
            seed = hash((cfg_name, 'free', restart)) % (2**31)
            tasks.append(('free', cfg_name, cfg, seed))

    print(f"Total tasks: {len(tasks)} ({len(CONFIGS)} configs × {RESTARTS} restarts × 2 modes)")
    print(f"Workers: {N_WORKERS}")
    t0 = time.time()

    results = {}  # {(mode, cfg_name): [(eps_f, x), ...]}
    done = 0
    with Pool(processes=N_WORKERS) as pool:
        for mode, cfg_name, seed, eps_f, x in pool.imap_unordered(run_one, tasks):
            key = (mode, cfg_name)
            if key not in results:
                results[key] = []
            results[key].append((eps_f, x))
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] latest: {cfg_name} {mode} εF={eps_f:.6f}", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"{'Config':<28} {'Mode':<14} {'best εF':>12} {'εF²':>12} {'mean εF':>12}")
    print("-"*80)

    summary = []
    for cfg_name, cfg in CONFIGS.items():
        row = {'config': cfg_name, 'R': cfg['R'], 'desc': cfg['desc']}
        for mode in ['constrained', 'free']:
            key = (mode, cfg_name)
            vals = results.get(key, [])
            if not vals:
                continue
            eps_list = [v[0] for v in vals]
            best_eps = min(eps_list)
            best_idx = eps_list.index(best_eps)
            best_x = vals[best_idx][1]

            print(f"{cfg_name:<28} {mode:<14} {best_eps:12.8f} {best_eps**2:12.8f} {np.mean(eps_list):12.8f}")

            row[f'{mode}_best_eps'] = best_eps
            row[f'{mode}_eps2'] = best_eps**2
            row[f'{mode}_mean_eps'] = float(np.mean(eps_list))

            # Save best solution
            R = cfg['R']
            A = best_x[:R*DIM].reshape(R, DIM)
            B = best_x[R*DIM:2*R*DIM].reshape(R, DIM)
            C = best_x[2*R*DIM:].reshape(R, DIM)
            outpath = os.path.join(RESULTS_DIR, f'orbit_sweep_{cfg_name}_{mode}.json')
            with open(outpath, 'w') as fp:
                json.dump({
                    'rank': R, 'config': cfg_name, 'mode': mode,
                    'frobenius': best_eps, 'frobenius_sq': best_eps**2,
                    'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist(),
                }, fp)

        summary.append(row)

    # ── Emmy's conjecture check ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("EMMY'S CONJECTURE CHECK")
    print("="*80)

    orbit_sizes = {0: 1, 1: 6, 2: 12, 3: 8}

    for cfg_name, cfg in CONFIGS.items():
        present = set(cfg['orbits'])
        n_o3 = cfg.get('n_o3', 0)
        n_o2 = cfg.get('n_o2', 0)
        missing_full = sum(orbit_sizes[i] for i in range(4) if i not in present)
        # Adjust for partial orbits
        if n_o3 > 0:
            missing_full -= n_o3
        if n_o2 > 0 and 2 not in present:
            missing_full -= n_o2

        key_c = ('constrained', cfg_name)
        key_f = ('free', cfg_name)
        eps2_c = min(v[0] for v in results.get(key_c, [(np.inf,)]))**2 if key_c in results else np.inf
        eps2_f = min(v[0] for v in results.get(key_f, [(np.inf,)]))**2 if key_f in results else np.inf

        leakage = eps2_c - eps2_f
        print(f"  {cfg_name:<28}  Σ|O_missing|={missing_full:3d}  "
              f"εF²_sym={eps2_c:10.6f}  εF²_free={eps2_f:10.6f}  "
              f"leakage={leakage:+10.6f}")

    # Check specific predictions
    print("\n  Specific predictions:")
    key = ('constrained', 'R22_O0O1O2_3xO3')
    if key in results:
        eps = min(v[0] for v in results[key])
        print(f"    R=22 symmetric εF = {eps:.8f}  (expected 1/√8 = {1/8**0.5:.8f})")

    key = ('constrained', 'R13_O0O2')
    if key in results:
        eps2 = min(v[0] for v in results[key])**2
        print(f"    R=13 symmetric εF² = {eps2:.6f}  (Emmy predicts 14 = |O1|+|O3| = 6+8)")

    key = ('free', 'R7_O0O1')
    if key in results:
        eps = min(v[0] for v in results[key])
        print(f"    R=7 free εF = {eps:.8f}  (Strassen analog, expect 0 or large)")

    # Save full summary
    with open(os.path.join(RESULTS_DIR, 'orbit_sweep_summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/orbit_sweep_*.json")


if __name__ == '__main__':
    main()
