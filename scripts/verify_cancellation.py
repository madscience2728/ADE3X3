#!/usr/bin/env python
"""Verify leakage cancellation between pairs, triplets, and quartets of rank-1 terms.

For each group of terms {k1, k2, ...}, the combined dead leakage is:
  ||Σ T_ki||²_dead = Σ||T_ki||²_dead + 2·Σ_{i<j} <T_ki, T_kj>_dead

If cross-terms <T_ki, T_kj>_dead are negative → destructive interference → cancellation.

This script computes:
1. Per-term dead leakage energy
2. All pairwise cross-terms on dead entries
3. Best cancelling pairs, triplets, quartets
4. The "cancellation ratio": combined_leakage / sum_of_individual_leakages
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR

DEAD_MASK = (TARGET_TENSOR == 0)  # (9,9,9) bool, 702 entries
LIVE_MASK = (TARGET_TENSOR != 0)  # (9,9,9) bool, 27 entries


def load_factors(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["alpha"]), np.array(d["beta"]), np.array(d["gamma"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--max-quartet", action="store_true",
                        help="Also search quartets (slow: C(19,4)=3876)")
    args = parser.parse_args()

    if args.input:
        src = args.input
    else:
        candidates = []
        for p in [Path("shotgun_best.json"), Path("chain_best.json")]:
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                candidates.append((d["fitness"], p))
        candidates.sort()
        src = candidates[0][1]

    alpha, beta, gamma = load_factors(src)
    R_full = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    fit = float(np.max(np.abs(R_full)))
    print(f"Loaded: {src}  fitness={fit:.10f}")

    # ── 1. Compute per-term contributions on dead and live entries ──
    print(f"\n{'='*70}")
    print(f"  1. PER-TERM DEAD/LIVE DECOMPOSITION")
    print(f"{'='*70}")

    # T_k[a,b,c] = alpha_k[a] * beta_k[b] * gamma_k[c]
    terms = []  # (R, 729) flattened rank-1 tensors
    for k in range(RANK):
        tk = np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])
        terms.append(tk)
    terms = np.array(terms)  # (R, 9, 9, 9)

    # Dead/live projections
    dead_flat = []  # (R, 702)
    live_flat = []  # (R, 27)
    for k in range(RANK):
        dead_flat.append(terms[k][DEAD_MASK])
        live_flat.append(terms[k][LIVE_MASK])
    dead_flat = np.array(dead_flat)  # (R, 702)
    live_flat = np.array(live_flat)  # (R, 27)

    dead_energy = np.sum(dead_flat**2, axis=1)  # (R,)
    live_energy = np.sum(live_flat**2, axis=1)   # (R,)
    total_energy = dead_energy + live_energy

    print(f"\n  {'Term':>4} {'Dead E':>10} {'Live E':>10} {'Total E':>10} {'Dead%':>7}")
    print(f"  {'─'*50}")
    for k in range(RANK):
        pct = dead_energy[k] / total_energy[k] * 100 if total_energy[k] > 0 else 0
        print(f"  {k:4d} {dead_energy[k]:10.4f} {live_energy[k]:10.4f} "
              f"{total_energy[k]:10.4f} {pct:6.1f}%")

    print(f"\n  Total dead energy (sum of individual): {dead_energy.sum():.4f}")
    print(f"  Actual combined dead energy ||Σ T_k||²_dead: "
          f"{np.sum(R_full[DEAD_MASK]**2) + np.sum(TARGET_TENSOR[DEAD_MASK]**2):.4f}")
    # Wait — R = T_hat - T, and T[dead]=0, so R[dead] = T_hat[dead] = (Σ T_k)[dead]
    combined_dead = np.sum(dead_flat.sum(axis=0)**2)
    print(f"  Combined dead energy ||Σ T_k||²_dead:        {combined_dead:.4f}")
    print(f"  Sum of individual dead energies:              {dead_energy.sum():.4f}")
    cross_total = combined_dead - dead_energy.sum()
    print(f"  Total cross-term energy (2·Σ <Ti,Tj>_dead):  {cross_total:.4f}")
    if cross_total < 0:
        print(f"  *** NET DESTRUCTIVE INTERFERENCE: cross-terms cancel {-cross_total:.4f} of leakage")
    else:
        print(f"  *** NET CONSTRUCTIVE INTERFERENCE: cross-terms ADD {cross_total:.4f} of leakage")

    # ── 2. Pairwise cross-terms on dead entries ──
    print(f"\n{'='*70}")
    print(f"  2. PAIRWISE DEAD-ENTRY CROSS-TERMS")
    print(f"{'='*70}")

    # G[i,j] = <T_i, T_j>_dead = dead_flat[i] · dead_flat[j]
    G_dead = dead_flat @ dead_flat.T  # (R, R) Gram matrix on dead entries
    G_live = live_flat @ live_flat.T  # (R, R) Gram matrix on live entries

    # Diagonal = individual dead energies (sanity check)
    assert np.allclose(np.diag(G_dead), dead_energy), "Gram diagonal mismatch"

    # Off-diagonal = cross-terms
    n_negative = 0
    n_positive = 0
    pairs_data = []
    for i in range(RANK):
        for j in range(i+1, RANK):
            cross = G_dead[i, j]
            if cross < 0:
                n_negative += 1
            else:
                n_positive += 1
            # Combined dead energy for pair
            pair_dead = dead_energy[i] + dead_energy[j] + 2 * cross
            pair_sum = dead_energy[i] + dead_energy[j]
            ratio = pair_dead / pair_sum if pair_sum > 0 else 1.0
            # Also check live: does the pair preserve signal?
            pair_live = live_energy[i] + live_energy[j] + 2 * G_live[i, j]
            pairs_data.append((ratio, cross, i, j, pair_dead, pair_sum, pair_live))

    pairs_data.sort()  # best (lowest ratio) first

    print(f"\n  Pairwise cross-terms: {n_negative} negative (cancel), {n_positive} positive (reinforce)")
    print(f"  Total pairs: {n_negative + n_positive}")

    print(f"\n  Top-10 CANCELLING pairs (lowest combined/individual ratio):")
    print(f"  {'Pair':>10} {'Cross':>10} {'Combined':>10} {'Sum_indiv':>10} {'Ratio':>7} {'Live_comb':>10}")
    print(f"  {'─'*65}")
    for ratio, cross, i, j, pair_dead, pair_sum, pair_live in pairs_data[:10]:
        print(f"  ({i:2d},{j:2d})  {cross:10.2f} {pair_dead:10.2f} {pair_sum:10.2f} "
              f"{ratio:7.3f} {pair_live:10.4f}")

    print(f"\n  Top-5 REINFORCING pairs (worst ratio):")
    for ratio, cross, i, j, pair_dead, pair_sum, pair_live in pairs_data[-5:]:
        print(f"  ({i:2d},{j:2d})  {cross:10.2f} {pair_dead:10.2f} {pair_sum:10.2f} "
              f"{ratio:7.3f} {pair_live:10.4f}")

    # ── 3. Triplets ──
    print(f"\n{'='*70}")
    print(f"  3. BEST CANCELLING TRIPLETS")
    print(f"{'='*70}")

    t0 = time.time()
    trip_data = []
    for combo in itertools.combinations(range(RANK), 3):
        i, j, k = combo
        # Combined dead = Σ d_i + 2·Σ_{i<j} G[i,j]
        trip_dead = (dead_energy[i] + dead_energy[j] + dead_energy[k]
                     + 2*(G_dead[i,j] + G_dead[i,k] + G_dead[j,k]))
        trip_sum = dead_energy[i] + dead_energy[j] + dead_energy[k]
        ratio = trip_dead / trip_sum if trip_sum > 0 else 1.0
        trip_data.append((ratio, combo, trip_dead, trip_sum))

    trip_data.sort()
    dt = time.time() - t0
    print(f"  Searched {len(trip_data)} triplets in {dt:.2f}s")

    print(f"\n  Top-10 cancelling triplets:")
    print(f"  {'Triplet':>15} {'Combined':>10} {'Sum_indiv':>10} {'Ratio':>7} {'Cancel%':>8}")
    print(f"  {'─'*55}")
    for ratio, combo, trip_dead, trip_sum in trip_data[:10]:
        cancel_pct = (1 - ratio) * 100
        print(f"  {str(combo):>15} {trip_dead:10.2f} {trip_sum:10.2f} {ratio:7.3f} {cancel_pct:7.1f}%")

    print(f"\n  Worst-5 reinforcing triplets:")
    for ratio, combo, trip_dead, trip_sum in trip_data[-5:]:
        cancel_pct = (1 - ratio) * 100
        print(f"  {str(combo):>15} {trip_dead:10.2f} {trip_sum:10.2f} {ratio:7.3f} {cancel_pct:7.1f}%")

    # ── 4. Quartets ──
    if args.max_quartet:
        print(f"\n{'='*70}")
        print(f"  4. BEST CANCELLING QUARTETS")
        print(f"{'='*70}")

        t0 = time.time()
        quad_data = []
        for combo in itertools.combinations(range(RANK), 4):
            a, b, c, d = combo
            quad_dead = (dead_energy[a] + dead_energy[b] + dead_energy[c] + dead_energy[d]
                         + 2*(G_dead[a,b] + G_dead[a,c] + G_dead[a,d]
                              + G_dead[b,c] + G_dead[b,d] + G_dead[c,d]))
            quad_sum = dead_energy[a] + dead_energy[b] + dead_energy[c] + dead_energy[d]
            ratio = quad_dead / quad_sum if quad_sum > 0 else 1.0
            quad_data.append((ratio, combo, quad_dead, quad_sum))

        quad_data.sort()
        dt = time.time() - t0
        print(f"  Searched {len(quad_data)} quartets in {dt:.2f}s")

        print(f"\n  Top-10 cancelling quartets:")
        print(f"  {'Quartet':>20} {'Combined':>10} {'Sum_indiv':>10} {'Ratio':>7} {'Cancel%':>8}")
        print(f"  {'─'*60}")
        for ratio, combo, q_dead, q_sum in quad_data[:10]:
            cancel_pct = (1 - ratio) * 100
            print(f"  {str(combo):>20} {q_dead:10.2f} {q_sum:10.2f} {ratio:7.3f} {cancel_pct:7.1f}%")
    else:
        # Quick: build quartets from best triplet + each remaining term
        print(f"\n{'='*70}")
        print(f"  4. BEST QUARTETS (extending top triplets)")
        print(f"{'='*70}")

        t0 = time.time()
        quad_data = []
        # Take top 20 triplets, extend each with remaining 16 terms
        for _, trip_combo, _, _ in trip_data[:20]:
            remaining = [k for k in range(RANK) if k not in trip_combo]
            for d in remaining:
                combo = tuple(sorted(trip_combo + (d,)))
                a, b, c, e = combo
                q_dead = (dead_energy[a] + dead_energy[b] + dead_energy[c] + dead_energy[e]
                          + 2*(G_dead[a,b] + G_dead[a,c] + G_dead[a,e]
                               + G_dead[b,c] + G_dead[b,e] + G_dead[c,e]))
                q_sum = dead_energy[a] + dead_energy[b] + dead_energy[c] + dead_energy[e]
                ratio = q_dead / q_sum if q_sum > 0 else 1.0
                quad_data.append((ratio, combo, q_dead, q_sum))

        # Deduplicate
        seen = set()
        unique = []
        for item in quad_data:
            if item[1] not in seen:
                seen.add(item[1])
                unique.append(item)
        unique.sort()
        dt = time.time() - t0
        print(f"  Searched {len(unique)} quartets in {dt:.2f}s")

        print(f"\n  Top-10 cancelling quartets:")
        print(f"  {'Quartet':>20} {'Combined':>10} {'Sum_indiv':>10} {'Ratio':>7} {'Cancel%':>8}")
        print(f"  {'─'*60}")
        for ratio, combo, q_dead, q_sum in unique[:10]:
            cancel_pct = (1 - ratio) * 100
            print(f"  {str(combo):>20} {q_dead:10.2f} {q_sum:10.2f} {ratio:7.3f} {cancel_pct:7.1f}%")

    # ── 5. Theoretical limits ──
    print(f"\n{'='*70}")
    print(f"  5. IS FULL CANCELLATION POSSIBLE?")
    print(f"{'='*70}")

    print(f"\n  All 19 terms combined:")
    print(f"    Sum of individual dead energies: {dead_energy.sum():.2f}")
    print(f"    Actual combined dead energy:     {combined_dead:.4f}")
    ratio_all = combined_dead / dead_energy.sum()
    cancel_all = (1 - ratio_all) * 100
    print(f"    Cancellation ratio:              {ratio_all:.6f}")
    print(f"    Cross-term cancellation:         {cancel_all:.2f}%")

    # Eigendecomposition of dead Gram matrix → theoretical min
    eigvals = np.linalg.eigvalsh(G_dead)
    print(f"\n  Dead Gram matrix eigenvalues:")
    for i, ev in enumerate(sorted(eigvals)):
        print(f"    λ_{i} = {ev:12.2f}{'  ← negative!' if ev < -0.01 else ''}")

    n_neg = np.sum(eigvals < -0.01)
    print(f"\n  Negative eigenvalues: {n_neg}/{RANK}")
    if n_neg > 0:
        print(f"  *** Negative eigenvalues confirm destructive interference is possible")
        print(f"  *** These define the subspace where term combinations cancel leakage")
    else:
        print(f"  All eigenvalues positive → dead Gram is PSD")
        print(f"  Cancellation only through off-diagonal cross-terms being negative")

    # Can we zero out dead entries with a linear combination of existing terms?
    # dead_flat is (R, 702). We want w such that dead_flat.T @ w = 0
    # This is the null space of dead_flat.T (702 × R)
    U, S, Vt = np.linalg.svd(dead_flat, full_matrices=False)
    print(f"\n  SVD of dead projection matrix ({RANK} × 702):")
    print(f"    Singular values: {', '.join(f'{s:.2f}' for s in S)}")
    n_zero = np.sum(S < 1e-10)
    print(f"    Near-zero singular values: {n_zero}")
    print(f"    Rank of dead projection: {RANK - n_zero}")

    # The null space of dead_flat (left null of dead_flat.T) would give
    # weight vectors w where Σ w_k * T_k has zero dead entries
    # But dead_flat is (R, 702) with R=19 < 702, so it's full row rank
    # unless some terms are linearly dependent on dead entries
    if n_zero > 0:
        print(f"\n  *** {n_zero} zero singular values → {n_zero} weight vectors that")
        print(f"      zero out all dead entries simultaneously!")
    else:
        # Check: is there a w that minimizes ||dead_flat.T @ w||² subject to
        # live_flat.T @ w = target_live?
        # This is a constrained least squares problem
        target_live = TARGET_TENSOR[LIVE_MASK]  # (27,) = the 27 ones
        # We need: live_flat.T @ w = target_live  (signal constraint)
        # Minimize: ||dead_flat.T @ w||²  (leakage minimization)
        # This is a standard constrained QP
        from scipy.optimize import minimize as sp_minimize

        def obj(w):
            return float(np.sum((dead_flat.T @ w)**2))

        def obj_grad(w):
            return 2 * dead_flat @ (dead_flat.T @ w)

        # Solve via Lagrange: minimize w'Gd w subject to Gl w = t
        # where Gd = dead_flat @ dead_flat.T, Gl = live_flat
        Gd = G_dead  # (R, R)
        # Constraint: live_flat.T @ w = target_live →  (27 × R).T @ (R,) = (27,)
        # Actually live_flat is (R, 27), so live_flat.T is (27, R)
        # live_flat.T @ w should give the live entries of T_hat = target_live

        # KKT system:
        # [2*Gd  live_flat] [w]      = [0          ]
        # [live_flat.T  0 ] [lambda]    [target_live]
        n_eq = 27
        KKT = np.zeros((RANK + n_eq, RANK + n_eq))
        KKT[:RANK, :RANK] = 2 * Gd
        KKT[:RANK, RANK:] = live_flat
        KKT[RANK:, :RANK] = live_flat.T
        rhs = np.zeros(RANK + n_eq)
        rhs[RANK:] = target_live

        try:
            sol = np.linalg.solve(KKT, rhs)
            w_opt = sol[:RANK]
            residual_dead = dead_flat.T @ w_opt
            min_dead_energy = float(np.sum(residual_dead**2))
            max_dead_abs = float(np.max(np.abs(residual_dead)))
            residual_live = live_flat.T @ w_opt - target_live
            live_err = float(np.max(np.abs(residual_live)))

            print(f"\n  Optimal weight vector (min dead leakage, exact live signal):")
            print(f"    Weights: [{', '.join(f'{w:.3f}' for w in w_opt)}]")
            print(f"    Dead energy with optimal w: {min_dead_energy:.6f}")
            print(f"    Dead max-abs with optimal w: {max_dead_abs:.6f}")
            print(f"    Live constraint error: {live_err:.2e}")
            print(f"    Current dead energy:    {combined_dead:.4f}")
            print(f"    Theoretical minimum:    {min_dead_energy:.6f}")
            if min_dead_energy < 1e-6:
                print(f"\n  *** EXACT CANCELLATION IS POSSIBLE ***")
                print(f"  *** A linear combination of existing terms can zero dead entries")
                print(f"  *** while exactly reproducing the target on live entries!")
            else:
                print(f"\n  Remaining dead energy {min_dead_energy:.4f} is the irreducible minimum")
                print(f"  given the current term structure (factor shapes).")
                print(f"  To reduce further, the factors themselves must change shape.")
        except np.linalg.LinAlgError:
            print(f"\n  KKT system is singular — constraint might be redundant or infeasible")


if __name__ == "__main__":
    main()
