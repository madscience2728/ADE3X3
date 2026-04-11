"""
Face Geometry Verification — Three experiments in one script.

Experiment 1: All 24 face-triples at R=22, confirm εF ≈ 0.032 uniformly.
Experiment 2: Free search warm-started from best of each orbit type.
Experiment 3: R=21 face-PAIRS — does Hamming signature predict εF for pairs too?
"""
import numpy as np
import json
import os
import time
from itertools import combinations
from multiprocessing import Pool
from scipy.optimize import minimize

# ── Target tensor ─────────────────────────────────────────────────────────────
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

DIM = 9
N_WORKERS = 24
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── O3 terms and base terms ──────────────────────────────────────────────────
O3_TERMS = [(r, s, u) for r in range(1, 3) for s in range(1, 3) for u in range(1, 3)]
BASE_TERMS = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            zeros = (r == 0) + (s == 0) + (u == 0)
            if zeros >= 1:
                BASE_TERMS.append((r, s, u))
assert len(BASE_TERMS) == 19
assert len(O3_TERMS) == 8

# ── Group action (Z2 ≀ S3) ───────────────────────────────────────────────────
S3_PERMS = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

def apply_g(triple, perm, flips):
    coords = [triple[perm[i]] for i in range(3)]
    return tuple(3 - c if flips[i] and c != 0 else c for i, c in enumerate(coords))

def triple_canonical(tri):
    best = None
    for p in S3_PERMS:
        for f0 in [False, True]:
            for f1 in [False, True]:
                for f2 in [False, True]:
                    t = tuple(sorted(apply_g(x, p, (f0, f1, f2)) for x in tri))
                    if best is None or t < best:
                        best = t
    return best

def pair_canonical(pair):
    best = None
    for p in S3_PERMS:
        for f0 in [False, True]:
            for f1 in [False, True]:
                for f2 in [False, True]:
                    t = tuple(sorted(apply_g(x, p, (f0, f1, f2)) for x in pair))
                    if best is None or t < best:
                        best = t
    return best

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def dist_sig(elems):
    return tuple(sorted(hamming(elems[i], elems[j])
                        for i in range(len(elems)) for j in range(i+1, len(elems))))


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


def run_constrained(args):
    """Run orbit-structured constrained optimization."""
    task_id, R, terms, seed = args
    rng = np.random.default_rng(seed)
    A = np.zeros((R, DIM))
    B = np.zeros((R, DIM))
    C = np.zeros((R, DIM))
    scale = 1.0 / R**0.5
    for k, (r, s, u) in enumerate(terms):
        A[k, 3*r+s] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
        B[k, 3*s+u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
        C[k, 3*r+u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.2 * rng.standard_normal())
        A[k] += rng.standard_normal(DIM) * 0.05 * scale
        B[k] += rng.standard_normal(DIM) * 0.05 * scale
        C[k] += rng.standard_normal(DIM) * 0.05 * scale
    x0 = np.concatenate([A.ravel(), B.ravel(), C.ravel()])
    res = minimize(lambda x: obj(x, R), x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
    Ar = res.x[:R*DIM].reshape(R, DIM)
    Br = res.x[R*DIM:2*R*DIM].reshape(R, DIM)
    Cr = res.x[2*R*DIM:].reshape(R, DIM)
    eps_f = float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', Ar, Br, Cr) - T))
    return task_id, eps_f, res.x


def run_warmstart(args):
    """Free search warm-started from a given x0 with perturbation."""
    task_id, R, x_warm, sigma, seed = args
    rng = np.random.default_rng(seed)
    x0 = x_warm + rng.standard_normal(len(x_warm)) * sigma
    res = minimize(lambda x: obj(x, R), x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
    Ar = res.x[:R*DIM].reshape(R, DIM)
    Br = res.x[R*DIM:2*R*DIM].reshape(R, DIM)
    Cr = res.x[2*R*DIM:].reshape(R, DIM)
    eps_f = float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', Ar, Br, Cr) - T))
    return task_id, eps_f, res.x


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: All 24 face-triples at R=22
# ══════════════════════════════════════════════════════════════════════════════
def experiment_1(pool):
    print("=" * 80)
    print("EXPERIMENT 1: All 24 face-triples — verify εF ≈ 0.032 uniformly")
    print("=" * 80)

    # Enumerate all face-triples (Orbit 0): Hamming sig = (1,1,2)
    face_triples = [tri for tri in combinations(O3_TERMS, 3)
                    if dist_sig(tri) == (1, 1, 2)]
    assert len(face_triples) == 24, f"Expected 24, got {len(face_triples)}"

    RESTARTS = 20
    tasks = []
    for i, tri in enumerate(face_triples):
        terms = BASE_TERMS + list(tri)
        for restart in range(RESTARTS):
            seed = i * 10000 + restart
            tasks.append((i * RESTARTS + restart, 22, terms, seed))

    print(f"  {len(face_triples)} face-triples × {RESTARTS} restarts = {len(tasks)} tasks")
    t0 = time.time()

    results = {}  # triple_idx -> [eps_f, ...]
    best_x = {}   # triple_idx -> (best_eps, x)
    done = 0
    for task_id, eps_f, x in pool.imap_unordered(run_constrained, tasks):
        tri_idx = task_id // RESTARTS
        results.setdefault(tri_idx, []).append(eps_f)
        if tri_idx not in best_x or eps_f < best_x[tri_idx][0]:
            best_x[tri_idx] = (eps_f, x)
        done += 1
        if done % 48 == 0 or done == len(tasks):
            print(f"    [{done}/{len(tasks)}]", flush=True)

    print(f"\n  Completed in {time.time() - t0:.1f}s\n")

    print(f"  {'Triple':<40} {'best εF':>12} {'mean εF':>12}")
    print("  " + "-" * 66)
    all_best = []
    for i, tri in enumerate(face_triples):
        eps_list = results[i]
        best = min(eps_list)
        all_best.append(best)
        print(f"  {str(tri):<40} {best:12.8f} {np.mean(eps_list):12.8f}")

    print(f"\n  Overall: min={min(all_best):.8f}  max={max(all_best):.8f}  "
          f"std={np.std(all_best):.8f}  mean={np.mean(all_best):.8f}")

    uniform = np.std(all_best) < 0.01
    print(f"  {'*** UNIFORM CONFIRMED ***' if uniform else '*** NOT UNIFORM ***'}")

    # Return best overall for Experiment 2
    overall_best_idx = int(np.argmin(all_best))
    return best_x, face_triples, all_best


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Warm-start free search from each orbit type
# ══════════════════════════════════════════════════════════════════════════════
def experiment_2(pool, exp1_best_x, exp1_triples):
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Free warm-start from each orbit type's best solution")
    print("=" * 80)

    # We need best solutions from each orbit type.
    # Orbit 0 (face): we have from Experiment 1.
    # Orbits 1, 2: run constrained first to get warm-starts.
    all_triples = list(combinations(O3_TERMS, 3))
    orbit_map = {}
    for tri in all_triples:
        sig = dist_sig(tri)
        orbit_map.setdefault(sig, []).append(tri)

    orbit_info = {
        (1, 1, 2): {'name': 'Orbit 0 (face)', 'idx': 0},
        (1, 2, 3): {'name': 'Orbit 1 (scalene)', 'idx': 1},
        (2, 2, 2): {'name': 'Orbit 2 (equilateral)', 'idx': 2},
    }

    # Get warm-starts for orbits 1 and 2
    print("  Phase A: Getting constrained warm-starts for Orbits 1 & 2...")
    RESTARTS_INIT = 20
    init_tasks = []
    orbit_reps = {}
    task_to_orbit = {}
    tid = 0
    for sig, members in orbit_map.items():
        if sig == (1, 1, 2):
            continue  # already have from Exp 1
        rep = members[0]
        orbit_reps[sig] = rep
        terms = BASE_TERMS + list(rep)
        for restart in range(RESTARTS_INIT):
            seed = hash((sig, restart)) % (2**31)
            init_tasks.append((tid, 22, terms, seed))
            task_to_orbit[tid] = sig
            tid += 1

    init_results = {}
    for task_id, eps_f, x in pool.imap_unordered(run_constrained, init_tasks):
        sig = task_to_orbit[task_id]
        if sig not in init_results or eps_f < init_results[sig][0]:
            init_results[sig] = (eps_f, x)

    # Collect warm-starts
    warm_starts = {}
    # Orbit 0: best from Experiment 1
    best_idx = min(exp1_best_x, key=lambda k: exp1_best_x[k][0])
    warm_starts[(1, 1, 2)] = exp1_best_x[best_idx]
    for sig in [(1, 2, 3), (2, 2, 2)]:
        warm_starts[sig] = init_results[sig]

    print("  Constrained warm-start εF:")
    for sig in [(1, 1, 2), (1, 2, 3), (2, 2, 2)]:
        print(f"    {orbit_info[sig]['name']}: εF = {warm_starts[sig][0]:.8f}")

    # Phase B: Free warm-starts with perturbation
    print("\n  Phase B: Free warm-start from each orbit's basin...")
    RESTARTS_FREE = 50
    SIGMAS = [0.001, 0.005, 0.01, 0.05, 0.1]

    free_tasks = []
    task_to_orbit2 = {}
    tid = 0
    for sig in [(1, 1, 2), (1, 2, 3), (2, 2, 2)]:
        x_warm = warm_starts[sig][1]
        for restart in range(RESTARTS_FREE):
            sigma = SIGMAS[restart % len(SIGMAS)]
            seed = hash((sig, 'free', restart)) % (2**31)
            free_tasks.append((tid, 22, x_warm, sigma, seed))
            task_to_orbit2[tid] = sig
            tid += 1

    free_results = {}
    done = 0
    for task_id, eps_f, x in pool.imap_unordered(run_warmstart, free_tasks):
        sig = task_to_orbit2[task_id]
        free_results.setdefault(sig, []).append(eps_f)
        done += 1
        if done % 30 == 0 or done == len(free_tasks):
            print(f"    [{done}/{len(free_tasks)}]", flush=True)

    print(f"\n  Free warm-start results:")
    print(f"  {'Orbit':<30} {'constrained εF':>15} {'free best εF':>15} {'free mean εF':>15}")
    print("  " + "-" * 77)
    for sig in [(1, 1, 2), (1, 2, 3), (2, 2, 2)]:
        eps_list = free_results[sig]
        print(f"  {orbit_info[sig]['name']:<30} {warm_starts[sig][0]:15.8f} "
              f"{min(eps_list):15.8f} {np.mean(eps_list):15.8f}")

    return free_results


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: R=21 face-PAIRS — orbit structure for 2 O3 terms
# ══════════════════════════════════════════════════════════════════════════════
def experiment_3(pool):
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: R=21 face-PAIRS — does Hamming signature predict εF?")
    print("=" * 80)

    all_pairs = list(combinations(O3_TERMS, 2))
    print(f"  Total O3 pairs: {len(all_pairs)}")

    # Classify by Hamming distance
    pair_classes = {}
    for pair in all_pairs:
        d = hamming(pair[0], pair[1])
        pair_classes.setdefault(d, []).append(pair)

    # Also classify by G-orbit
    pair_orbits = {}
    for pair in all_pairs:
        c = pair_canonical(pair)
        pair_orbits.setdefault(c, []).append(pair)

    print(f"  Hamming distance classes: {sorted(pair_classes.keys())}")
    for d in sorted(pair_classes.keys()):
        print(f"    d={d}: {len(pair_classes[d])} pairs")
    print(f"  G-orbits of pairs: {len(pair_orbits)}")
    for c, members in sorted(pair_orbits.items()):
        d = hamming(c[0], c[1])
        print(f"    canon={c}  d={d}  size={len(members)}")

    # Test one representative per G-orbit of pairs
    RESTARTS = 30
    R = 21
    tasks = []
    task_to_orbit = {}
    orbit_list = sorted(pair_orbits.items())
    tid = 0
    for orbit_idx, (canon_pair, members) in enumerate(orbit_list):
        rep = members[0]
        terms = BASE_TERMS + list(rep)
        assert len(terms) == R
        for restart in range(RESTARTS):
            seed = hash(('pair', canon_pair, restart)) % (2**31)
            tasks.append((tid, R, terms, seed))
            task_to_orbit[tid] = orbit_idx
            tid += 1

    print(f"\n  {len(orbit_list)} pair-orbits × {RESTARTS} restarts = {len(tasks)} tasks")
    t0 = time.time()

    pair_results = {i: [] for i in range(len(orbit_list))}
    done = 0
    for task_id, eps_f, x in pool.imap_unordered(run_constrained, tasks):
        oi = task_to_orbit[task_id]
        pair_results[oi].append(eps_f)
        done += 1
        if done % 30 == 0 or done == len(tasks):
            print(f"    [{done}/{len(tasks)}]", flush=True)

    print(f"\n  Completed in {time.time() - t0:.1f}s\n")

    print(f"  {'Pair canon':<35} {'d_H':>3} {'size':>4} {'best εF':>12} {'εF²':>12} {'mean εF':>12}")
    print("  " + "-" * 82)

    for orbit_idx, (canon_pair, members) in enumerate(orbit_list):
        d = hamming(canon_pair[0], canon_pair[1])
        eps_list = pair_results[orbit_idx]
        best = min(eps_list)
        print(f"  {str(canon_pair):<35} {d:3d} {len(members):4d} "
              f"{best:12.8f} {best**2:12.8f} {np.mean(eps_list):12.8f}")

    # Check: is face-pair (d=1) better?
    print("\n  Summary by Hamming distance:")
    by_d = {}
    for orbit_idx, (canon_pair, members) in enumerate(orbit_list):
        d = hamming(canon_pair[0], canon_pair[1])
        best = min(pair_results[orbit_idx])
        by_d.setdefault(d, []).append(best)
    for d in sorted(by_d):
        vals = by_d[d]
        print(f"    d={d}: best εF = {min(vals):.8f}, all bests = {[f'{v:.6f}' for v in vals]}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("FACE GEOMETRY VERIFICATION — Three experiments")
    print("=" * 80)
    t_total = time.time()

    with Pool(processes=N_WORKERS) as pool:
        best_x, face_triples, all_best = experiment_1(pool)
        exp2_results = experiment_2(pool, best_x, face_triples)
        experiment_3(pool)

    print(f"\n{'=' * 80}")
    print(f"Total elapsed: {time.time() - t_total:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
