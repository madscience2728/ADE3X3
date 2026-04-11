"""
Task 4 — O3 generator test.

For R=22 = 19 (O0+O1+O2) + 3 from O3, enumerate all C(8,3)=56 triples
of O3 terms, identify distinct G-orbits, and run optimization for each.

Prediction: all give εF = 1/√8 (conservation under G-transitivity on O3).
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
O3_TERMS = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
            if r != 0 and s != 0 and u != 0:
                O3_TERMS.append((r, s, u))

assert len(O3_TERMS) == 8
DIM = 9
R = 22
RESTARTS_PER_TRIPLE = 30
N_WORKERS = 24
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Base terms: O0 + O1 + O2 = 19 terms
BASE_TERMS = []
for r in range(3):
    for s in range(3):
        for u in range(3):
            zeros = (r == 0) + (s == 0) + (u == 0)
            if zeros >= 1:  # O0, O1, O2
                BASE_TERMS.append((r, s, u))
assert len(BASE_TERMS) == 19

# ── G = Z2 ≀ S3 action on (r,s,u) ∈ {0,1,2}³ ──────────────────────────────
# Z2≀S3 acts by permuting {0,1,2} independently on each coordinate? No —
# it acts on the 3×3×3 structure. For O3 (all nonzero), the group acts by
# permuting the nonzero labels {1,2} in each coordinate independently,
# combined with S3 permutations of the three coordinates.
# Z2≀S3 = (Z2)³ ⋊ S3, order 8×6 = 48.

def apply_z2_wr_s3(triple, perm, flips):
    """Apply group element to (r,s,u).
    perm: permutation of (r,s,u) coordinates (element of S3)
    flips: 3-tuple of bools, whether to flip 1↔2 in each coordinate (after perm)
    Returns the transformed triple.
    """
    coords = [triple[perm[i]] for i in range(3)]
    out = []
    for i, c in enumerate(coords):
        if flips[i] and c != 0:
            out.append(3 - c)  # flip 1↔2
        else:
            out.append(c)
    return tuple(out)


def triple_canonical(triple_of_triples):
    """Canonical form of a set of 3 O3 terms under G action."""
    S3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    min_form = None
    for perm in S3:
        for f0 in [False, True]:
            for f1 in [False, True]:
                for f2 in [False, True]:
                    flips = (f0, f1, f2)
                    transformed = tuple(sorted(
                        apply_z2_wr_s3(t, perm, flips)
                        for t in triple_of_triples
                    ))
                    if min_form is None or transformed < min_form:
                        min_form = transformed
    return min_form


# Enumerate all 56 triples and find G-orbits
ALL_TRIPLES = list(combinations(O3_TERMS, 3))
orbit_map = {}  # canonical -> [list of triples]
for tri in ALL_TRIPLES:
    canon = triple_canonical(tri)
    if canon not in orbit_map:
        orbit_map[canon] = []
    orbit_map[canon].append(tri)

print(f"O3 triples: {len(ALL_TRIPLES)} total, {len(orbit_map)} G-orbits")
for i, (canon, members) in enumerate(sorted(orbit_map.items())):
    print(f"  Orbit {i}: canon={canon}  size={len(members)}")


# ── Objective ─────────────────────────────────────────────────────────────────
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


def run_one_restart(args):
    triple_idx, restart_idx, o3_triple = args
    seed = triple_idx * 10000 + restart_idx
    rng = np.random.default_rng(seed)

    terms = BASE_TERMS + list(o3_triple)
    assert len(terms) == R

    # Orbit-structured initialization
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
    res = minimize(obj, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})

    Ar = res.x[:R*DIM].reshape(R, DIM)
    Br = res.x[R*DIM:2*R*DIM].reshape(R, DIM)
    Cr = res.x[2*R*DIM:].reshape(R, DIM)
    eps_f = float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', Ar, Br, Cr) - T))
    return triple_idx, restart_idx, eps_f, res.x


def main():
    print("\n" + "="*80)
    print("O3 GENERATOR TEST — R=22, all C(8,3)=56 triples of O3 terms")
    print("="*80)

    # Build tasks: one representative per G-orbit, all restarts
    tasks = []
    orbit_reps = []  # (orbit_idx, canonical, representative_triple, orbit_size)
    for orbit_idx, (canon, members) in enumerate(sorted(orbit_map.items())):
        rep = members[0]
        orbit_reps.append((orbit_idx, canon, rep, len(members)))
        for restart in range(RESTARTS_PER_TRIPLE):
            tasks.append((orbit_idx, restart, rep))

    print(f"Testing {len(orbit_reps)} G-orbit representatives × {RESTARTS_PER_TRIPLE} restarts = {len(tasks)} tasks")
    print(f"Workers: {N_WORKERS}")
    t0 = time.time()

    results_by_orbit = {i: [] for i in range(len(orbit_reps))}
    best_by_orbit = {}

    with Pool(processes=N_WORKERS) as pool:
        for triple_idx, restart_idx, eps_f, x in pool.imap_unordered(run_one_restart, tasks):
            results_by_orbit[triple_idx].append(eps_f)
            if triple_idx not in best_by_orbit or eps_f < best_by_orbit[triple_idx][0]:
                best_by_orbit[triple_idx] = (eps_f, x)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # ── Report ────────────────────────────────────────────────────────────────
    expected = 1.0 / 8**0.5
    print(f"Expected εF = 1/√8 = {expected:.8f}")
    print(f"\n{'Orbit':>5} {'Size':>4} {'Canon':<30} {'best εF':>12} {'εF²':>12} {'mean εF':>12} {'Match?':>7}")
    print("-"*90)

    all_match = True
    summary = []
    for orbit_idx, canon, rep, size in orbit_reps:
        eps_list = results_by_orbit[orbit_idx]
        best = min(eps_list)
        mean = np.mean(eps_list)
        match = abs(best - expected) < 0.01
        if not match:
            all_match = False
        print(f"{orbit_idx:5d} {size:4d} {str(canon):<30} {best:12.8f} {best**2:12.8f} {mean:12.8f} {'  YES' if match else '  *** NO ***':>7}")
        summary.append({
            'orbit_idx': orbit_idx, 'canonical': str(canon),
            'orbit_size': size, 'representative': list(rep),
            'best_eps': best, 'eps_sq': best**2, 'mean_eps': float(mean),
            'matches_prediction': match,
        })

    print()
    if all_match:
        print("*** ALL ORBITS MATCH εF = 1/√8 — G-symmetry conservation CONFIRMED ***")
    else:
        print("*** SOME ORBITS DEVIATE — check for symmetry breaking ***")

    outpath = os.path.join(RESULTS_DIR, 'o3_generator_test.json')
    with open(outpath, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == '__main__':
    main()
