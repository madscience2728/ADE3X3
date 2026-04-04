"""Kernel-family incidence mapper.

This is a mapping-mode tool, not an optimizer. It does three things:

1. Rebuild the Z2 wr S3 term-space decomposition used by irrep_kernel_enum.py.
2. Enumerate all R=13 and R=20 kernel candidates from the captured irrep dimensions.
3. Evaluate each candidate against deterministic family samples from canon-driven
   support families, measuring how close span(H) and Delta are to that kernel.

Output is a JSON report ranking every candidate by its best family compatibility.

Usage:
  python kernel_family_map.py --workers 24 --samples 8 --out kernel_family_map.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, permutations, product as iproduct

import numpy as np


def swap12(x: int) -> int:
    return x if x == 0 else 3 - x


S3 = list(permutations(range(3)))
GROUP = [(pi, eps) for pi in S3 for eps in iproduct([False, True], repeat=3)]
ALL27 = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
KEPT19 = sorted([t for t in ALL27 if 0 in t])
KEPT19_IDX = {t: i for i, t in enumerate(KEPT19)}
ALL_FIBERS = [(r, u) for r in range(3) for u in range(3)]

HYBRID_ORBIT_6 = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]


def act_on_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(swap12(x[i]) if eps[i] else x[i] for i in range(3))


def group_perm_matrix(pi, eps):
    matrix = np.zeros((19, 19))
    for i, t in enumerate(KEPT19):
        t2 = act_on_triple(pi, eps, t)
        j = KEPT19_IDX[t2]
        matrix[j, i] = 1.0
    return matrix


PERM_MATRICES = [group_perm_matrix(pi, eps) for pi, eps in GROUP]


def decompose_irreps():
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(len(PERM_MATRICES))
    commutant = sum(c * p for c, p in zip(coeffs, PERM_MATRICES)) / len(PERM_MATRICES)
    commutant = (commutant + commutant.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(commutant)

    irreps = []
    used = set()
    tol = 1e-8
    for i in range(19):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, 19):
            if j not in used and abs(eigenvalues[i] - eigenvalues[j]) < tol:
                group.append(j)
                used.add(j)
        irreps.append(eigenvectors[:, group])
    return irreps


IRREPS19 = decompose_irreps()
IRREP_DIMS = [basis.shape[1] for basis in IRREPS19]


def enumerate_candidates(kernel_dim: int):
    combos = []
    for r in range(1, len(IRREPS19) + 1):
        for combo in combinations(range(len(IRREPS19)), r):
            if sum(IRREP_DIMS[i] for i in combo) == kernel_dim:
                combos.append(combo)
    return combos


def term_matrix_mask():
    return np.zeros((3, 3), dtype=bool)


def single_fiber_mask(r, u, s):
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    alpha_mask[r, s] = True
    beta_mask[s, u] = True
    return alpha_mask, beta_mask


def row_col_mask(r, u):
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    alpha_mask[r, :] = True
    beta_mask[:, u] = True
    return alpha_mask, beta_mask


def rectangle_mask(r, u, row_size, col_size):
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    rows = [((r + offset) % 3) for offset in range(row_size)]
    cols = [((u + offset) % 3) for offset in range(col_size)]
    for rr in rows:
        alpha_mask[rr, :] = True
    for uu in cols:
        beta_mask[:, uu] = True
    return alpha_mask, beta_mask


def build_partition_family(R, counts, seed):
    rng = np.random.default_rng(seed)
    assignment = []
    for fiber, count in zip(ALL_FIBERS, counts):
        assignment.extend([fiber] * count)
    alpha = []
    beta = []
    local_seen = {fiber: 0 for fiber in ALL_FIBERS}
    for idx, (r, u) in enumerate(assignment):
        local_seen[(r, u)] += 1
        visit = local_seen[(r, u)]
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        if visit == 1:
            s = idx % 3
            a[r, s] = rng.choice([-1.0, 1.0]) * (1.0 + 0.08 * rng.standard_normal())
            b[s, u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.08 * rng.standard_normal())
        else:
            if visit == 2:
                row_size, col_size = 2, 2
            else:
                row_size, col_size = 3, 3
            rows = [((r + offset) % 3) for offset in range(row_size)]
            cols = [((u + offset) % 3) for offset in range(col_size)]
            for rr in rows:
                a[rr, :] = rng.standard_normal(3) * 0.20
            for uu in cols:
                b[:, uu] = rng.standard_normal(3) * 0.20
        alpha.append(a)
        beta.append(b)
    return np.array(alpha), np.array(beta)


def build_hybrid_family(R, residual, seed):
    rng = np.random.default_rng(seed)
    alpha = []
    beta = []
    fourier = [fiber for fiber in ALL_FIBERS if fiber not in residual]
    for (r, u) in fourier:
        for s in range(3):
            a = np.zeros((3, 3))
            b = np.zeros((3, 3))
            a[r, s] = 1.0 + 0.05 * rng.standard_normal()
            b[s, u] = 1.0 + 0.05 * rng.standard_normal()
            alpha.append(a)
            beta.append(b)
    for idx in range(R - 9):
        r, u = residual[idx % len(residual)]
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        a[r, :] = rng.standard_normal(3) * 0.25
        b[:, u] = rng.standard_normal(3) * 0.25
        alpha.append(a)
        beta.append(b)
    return np.array(alpha), np.array(beta)


def build_family_samples(rank_label: int, samples: int):
    rows = []
    if rank_label == 13:
        families = [
            ("exact3_partition_r13", lambda seed: build_partition_family(13, [3, 3, 3, 3, 1, 0, 0, 0, 0], seed)),
            ("samefiber_core_r13", lambda seed: build_partition_family(13, [3, 2, 2, 2, 2, 1, 1, 0, 0], seed)),
        ]
    elif rank_label == 20:
        families = [
            ("exact3_partition_r20", lambda seed: build_partition_family(20, [3, 3, 3, 3, 3, 3, 2, 0, 0], seed)),
            ("hybrid_orbit6_r20", lambda seed: build_hybrid_family(20, HYBRID_ORBIT_6, seed)),
        ]
    else:
        raise ValueError(rank_label)

    for family_name, builder in families:
        for sample_idx in range(samples):
            seed = 20260403 + 1000 * rank_label + 100 * sample_idx + sum(ord(c) for c in family_name)
            alpha, beta = builder(seed)
            rows.append({
                "family": family_name,
                "sample": sample_idx,
                "alpha": alpha,
                "beta": beta,
            })
    return rows


def compute_fiber_coords(alpha, beta):
    R = alpha.shape[0]
    sigma = np.zeros((R, 9))
    eta1 = np.zeros((R, 9))
    eta2 = np.zeros((R, 9))
    for k in range(R):
        for r in range(3):
            for u in range(3):
                idx = r * 3 + u
                s0 = alpha[k, r, 0] * beta[k, 0, u]
                s1 = alpha[k, r, 1] * beta[k, 1, u]
                s2 = alpha[k, r, 2] * beta[k, 2, u]
                sigma[k, idx] = s0 + s1 + s2
                eta1[k, idx] = s0 - s1
                eta2[k, idx] = s1 - s2
    h = np.hstack([eta1, eta2])
    dead_pairs = [(s, t) for s in range(3) for t in range(3) if s != t]
    delta = np.zeros((R, 54))
    for k in range(R):
        col = 0
        for s, t in dead_pairs:
            for r in range(3):
                for u in range(3):
                    delta[k, col] = alpha[k, r, s] * beta[k, t, u]
                    col += 1
    return sigma, h, delta


def evaluate_candidate(rank_label, combo, family_samples):
    R = rank_label
    kernel_basis = np.hstack([IRREPS19[i] for i in combo])
    kernel_proj = kernel_basis @ kernel_basis.T
    comp_proj = np.eye(19) - kernel_proj

    evaluations = []
    best_key = None
    best_eval = None
    for row in family_samples:
        alpha = row["alpha"]
        beta = row["beta"]
        sigma, h, delta = compute_fiber_coords(alpha, beta)

        h_norm = np.linalg.norm(h, ord="fro")
        delta_norm = np.linalg.norm(delta, ord="fro")
        sigma_comp = comp_proj[:R, :R] @ sigma if R < 19 else comp_proj @ sigma

        if R < 19:
            # Embed smaller-R family term space into the first R coordinates of the 19-dim ambient space.
            embed = np.zeros((19, R))
            embed[:R, :R] = np.eye(R)
            h_ambient = embed @ h
            delta_ambient = embed @ delta
            sigma_ambient = embed @ sigma
            h_leak = np.linalg.norm(comp_proj @ h_ambient, ord="fro") / (np.linalg.norm(h_ambient, ord="fro") + 1e-30)
            delta_leak = np.linalg.norm(comp_proj @ delta_ambient, ord="fro") / (np.linalg.norm(delta_ambient, ord="fro") + 1e-30)
            sigma_rank = int(np.linalg.matrix_rank((np.eye(19) - kernel_proj) @ sigma_ambient, tol=1e-10))
        else:
            h_leak = np.linalg.norm(comp_proj @ h, ord="fro") / (h_norm + 1e-30)
            delta_leak = np.linalg.norm(comp_proj @ delta, ord="fro") / (delta_norm + 1e-30)
            sigma_rank = int(np.linalg.matrix_rank(comp_proj @ sigma, tol=1e-10))

        rank_h = int(np.linalg.matrix_rank(h, tol=1e-10))
        rank_nuis = int(np.linalg.matrix_rank(np.hstack([h, delta]), tol=1e-10))
        evaluation = {
            "family": row["family"],
            "sample": row["sample"],
            "h_leak": float(h_leak),
            "delta_leak": float(delta_leak),
            "sigma_rank": sigma_rank,
            "rank_H": rank_h,
            "rank_nuis": rank_nuis,
            "conservation": int(R + (18 - rank_h)),
        }
        evaluations.append(evaluation)
        key = (evaluation["h_leak"], evaluation["delta_leak"], -evaluation["sigma_rank"], evaluation["rank_H"])
        if best_key is None or key < best_key:
            best_key = key
            best_eval = evaluation

    return {
        "combo": list(combo),
        "dims": [IRREP_DIMS[i] for i in combo],
        "best": best_eval,
        "evaluations": evaluations,
    }


def summarize_results(results):
    by_family = {}
    for row in results:
        fam = row["best"]["family"]
        by_family[fam] = by_family.get(fam, 0) + 1
    return by_family


def main():
    parser = argparse.ArgumentParser(description="Map all irrep candidates against canon family samples")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--ranks", type=str, default="13,20")
    parser.add_argument("--out", type=str, default="kernel_family_map.json")
    args = parser.parse_args()

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    rank_to_kernel_dim = {13: 4, 20: 11}

    payload = {
        "irrep_dims": IRREP_DIMS,
        "ranks": {},
    }

    print("=" * 72)
    print("KERNEL-FAMILY INCIDENCE MAPPER")
    print("=" * 72)
    print(f"workers={args.workers} samples={args.samples} ranks={ranks}")

    for rank_label in ranks:
        kernel_dim = rank_to_kernel_dim[rank_label]
        combos = enumerate_candidates(kernel_dim)
        family_samples = build_family_samples(rank_label, args.samples)
        print(f"\nR={rank_label}: {len(combos)} candidates, {len(family_samples)} family samples")

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(evaluate_candidate, rank_label, combo, family_samples): combo
                for combo in combos
            }
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                done += 1
                result = future.result()
                results.append(result)
                if done % 25 == 0 or done == total:
                    print(f"  [{done}/{total}]", flush=True)

        results.sort(key=lambda row: (row["best"]["h_leak"], row["best"]["delta_leak"], -row["best"]["sigma_rank"]))
        payload["ranks"][str(rank_label)] = {
            "kernel_dim": kernel_dim,
            "candidate_count": len(combos),
            "family_sample_count": len(family_samples),
            "best_family_histogram": summarize_results(results),
            "top20": results[:20],
            "all_results": results,
        }

        print("  top candidates:")
        for idx, row in enumerate(results[:10], start=1):
            dims_str = "+".join(str(d) for d in row["dims"])
            best = row["best"]
            print(
                f"    {idx:02d}. [{dims_str}] fam={best['family']} h={best['h_leak']:.3e} "
                f"d={best['delta_leak']:.3e} sigma={best['sigma_rank']} rkH={best['rank_H']}"
            )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()