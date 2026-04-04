#!/usr/bin/env python3
"""Canon-guided family scan for overnight Gate 1 / Gate 2 search.

Families implemented here are intentionally narrow. They are not generic coefficient
hunts; they are structured probes derived from the canon:

- Hybrid Fourier families: 3 exact same-fiber blocks (9 terms) plus spreaders on one of
  the surviving six-fiber residual orbit classes from Step 60.
- Same-fiber core family: all 19 terms are dead-free same-fiber terms, probing the tightest
  same-fiber core suggested by the canon.

The script optimizes alpha/beta only and solves gamma by least squares at each step.
Overnight use should focus on Gate 1 first; Gate 2 is measured on the resulting iterates.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize


def build_matmul_tensor() -> np.ndarray:
    tensor = np.zeros((9, 9, 9))
    for r in range(3):
        for u in range(3):
            for s in range(3):
                tensor[r * 3 + u, r * 3 + s, s * 3 + u] = 1.0
    return tensor


TENSOR = build_matmul_tensor()
ALL_FIBERS = [(r, u) for r in range(3) for u in range(3)]
DEFAULT_FAMILIES = [
    "exact3_partition_r13",
    "samefiber_core_r13",
    "exact3_partition_r20",
    "hybrid_orbit6_r20",
    "hybrid_orbit36_r19",
    "hybrid_orbit6_r19",
]

HYBRID_ORBIT_36 = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
HYBRID_ORBIT_18A = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)]
HYBRID_ORBIT_18B = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 2)]
HYBRID_ORBIT_6 = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]

FAMILY_CONFIGS = {
    "exact3_partition_r13": {
        "R": 13,
        "mode": "partition_samefiber",
        "counts": [3, 3, 3, 3, 1, 0, 0, 0, 0],
        "note": "Step 57 first exact-3 survivor type for R=13",
    },
    "samefiber_core_r13": {
        "R": 13,
        "mode": "partition_samefiber",
        "counts": [3, 2, 2, 2, 2, 1, 1, 0, 0],
        "note": "R=13 same-fiber control family with concentrated orbit budget",
    },
    "hybrid_orbit36_r19": {
        "R": 19,
        "mode": "hybrid",
        "residual": HYBRID_ORBIT_36,
        "note": "Step 60 six-fiber non-rectangular orbit size 36",
    },
    "hybrid_orbit18a_r19": {
        "R": 19,
        "mode": "hybrid",
        "residual": HYBRID_ORBIT_18A,
        "note": "Step 60 six-fiber non-rectangular orbit size 18A",
    },
    "hybrid_orbit18b_r19": {
        "R": 19,
        "mode": "hybrid",
        "residual": HYBRID_ORBIT_18B,
        "note": "Step 60 six-fiber non-rectangular orbit size 18B",
    },
    "hybrid_orbit6_r19": {
        "R": 19,
        "mode": "hybrid",
        "residual": HYBRID_ORBIT_6,
        "note": "Step 60 six-fiber non-rectangular orbit size 6",
    },
    "samefiber_core_r19": {
        "R": 19,
        "mode": "samefiber_core",
        "note": "Step 35 same-fiber core style dead-free family",
    },
    "hybrid_orbit6_r20": {
        "R": 20,
        "mode": "hybrid",
        "residual": HYBRID_ORBIT_6,
        "note": "R=20 slack control on the sharpest surviving Step 60 orbit",
    },
    "exact3_partition_r20": {
        "R": 20,
        "mode": "partition_samefiber",
        "counts": [3, 3, 3, 3, 3, 3, 2, 0, 0],
        "note": "Step 57 first exact-3 survivor type for R=20",
    },
}


def term_matrix_mask() -> np.ndarray:
    return np.zeros((3, 3), dtype=bool)


def single_fiber_mask(r: int, u: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    alpha_mask[r, s] = True
    beta_mask[s, u] = True
    return alpha_mask, beta_mask


def row_col_mask(r: int, u: int) -> tuple[np.ndarray, np.ndarray]:
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    alpha_mask[r, :] = True
    beta_mask[:, u] = True
    return alpha_mask, beta_mask


def rectangle_mask(r: int, u: int, row_size: int, col_size: int) -> tuple[np.ndarray, np.ndarray]:
    alpha_mask = term_matrix_mask()
    beta_mask = term_matrix_mask()
    rows = [((r + offset) % 3) for offset in range(row_size)]
    cols = [((u + offset) % 3) for offset in range(col_size)]
    for rr in rows:
        alpha_mask[rr, :] = True
    for uu in cols:
        beta_mask[:, uu] = True
    return alpha_mask, beta_mask


def build_family_structure(family_name: str, seed: int) -> dict:
    cfg = FAMILY_CONFIGS[family_name]
    rng = np.random.default_rng(seed)
    alpha_masks = []
    beta_masks = []
    alpha_init = []
    beta_init = []
    term_labels = []

    if cfg["mode"] == "hybrid":
        residual = list(cfg["residual"])
        fourier = [fiber for fiber in ALL_FIBERS if fiber not in residual]

        for (r, u) in fourier:
            for s in range(3):
                alpha_mask, beta_mask = single_fiber_mask(r, u, s)
                alpha0 = np.zeros((3, 3))
                beta0 = np.zeros((3, 3))
                alpha0[r, s] = 1.0 + 0.05 * rng.standard_normal()
                beta0[s, u] = 1.0 + 0.05 * rng.standard_normal()
                alpha_masks.append(alpha_mask)
                beta_masks.append(beta_mask)
                alpha_init.append(alpha0)
                beta_init.append(beta0)
                term_labels.append(f"fourier({r},{u};s={s})")

        spreader_terms = cfg["R"] - 9
        for idx in range(spreader_terms):
            r, u = residual[idx % len(residual)]
            alpha_mask, beta_mask = row_col_mask(r, u)
            alpha0 = np.zeros((3, 3))
            beta0 = np.zeros((3, 3))
            alpha0[r, :] = rng.standard_normal(3) * 0.25
            beta0[:, u] = rng.standard_normal(3) * 0.25
            alpha_masks.append(alpha_mask)
            beta_masks.append(beta_mask)
            alpha_init.append(alpha0)
            beta_init.append(beta0)
            term_labels.append(f"spreader({r},{u})")

    elif cfg["mode"] == "samefiber_core":
        assignment = []
        heavy = ALL_FIBERS[0]
        assignment.extend([heavy, heavy, heavy])
        for fiber in ALL_FIBERS[1:]:
            assignment.extend([fiber, fiber])
        assert len(assignment) == cfg["R"]

        for idx, (r, u) in enumerate(assignment):
            s = idx % 3
            alpha_mask, beta_mask = single_fiber_mask(r, u, s)
            alpha0 = np.zeros((3, 3))
            beta0 = np.zeros((3, 3))
            alpha0[r, s] = rng.choice([-1.0, 1.0]) * (1.0 + 0.1 * rng.standard_normal())
            beta0[s, u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.1 * rng.standard_normal())
            alpha_masks.append(alpha_mask)
            beta_masks.append(beta_mask)
            alpha_init.append(alpha0)
            beta_init.append(beta0)
            term_labels.append(f"samefiber({r},{u};s={s})")

    elif cfg["mode"] == "partition_samefiber":
        assignment = []
        for fiber, count in zip(ALL_FIBERS, cfg["counts"]):
            assignment.extend([fiber] * count)
        assert len(assignment) == cfg["R"]

        local_seen = {fiber: 0 for fiber in ALL_FIBERS}
        for idx, (r, u) in enumerate(assignment):
            local_seen[(r, u)] += 1
            visit = local_seen[(r, u)]
            alpha0 = np.zeros((3, 3))
            beta0 = np.zeros((3, 3))

            if visit == 1:
                s = idx % 3
                alpha_mask, beta_mask = single_fiber_mask(r, u, s)
                alpha0[r, s] = rng.choice([-1.0, 1.0]) * (1.0 + 0.08 * rng.standard_normal())
                beta0[s, u] = rng.choice([-1.0, 1.0]) * (1.0 + 0.08 * rng.standard_normal())
                term_labels.append(f"partition_anchor({r},{u};s={s})")
            else:
                if visit == 2:
                    row_size, col_size = 2, 2
                elif visit == 3:
                    row_size, col_size = 3, 3
                else:
                    row_size, col_size = 2, 3 if (visit % 2 == 0) else 3, 2
                alpha_mask, beta_mask = rectangle_mask(r, u, row_size, col_size)
                rows = [((r + offset) % 3) for offset in range(row_size)]
                cols = [((u + offset) % 3) for offset in range(col_size)]
                for rr in rows:
                    alpha0[rr, :] = rng.standard_normal(3) * 0.20
                for uu in cols:
                    beta0[:, uu] = rng.standard_normal(3) * 0.20
                term_labels.append(f"partition_rect({r},{u};{row_size}x{col_size})")

            alpha_masks.append(alpha_mask)
            beta_masks.append(beta_mask)
            alpha_init.append(alpha0)
            beta_init.append(beta0)

    else:
        raise ValueError(f"Unknown family mode: {cfg['mode']}")

    alpha_init = np.array(alpha_init)
    beta_init = np.array(beta_init)
    return {
        "R": cfg["R"],
        "mode": cfg["mode"],
        "note": cfg["note"],
        "alpha_masks": alpha_masks,
        "beta_masks": beta_masks,
        "alpha_init": alpha_init,
        "beta_init": beta_init,
        "term_labels": term_labels,
    }


def pack_masked(alpha: np.ndarray, beta: np.ndarray, alpha_masks: list[np.ndarray], beta_masks: list[np.ndarray]) -> np.ndarray:
    out = []
    for term_idx in range(alpha.shape[0]):
        out.extend(alpha[term_idx][alpha_masks[term_idx]].tolist())
    for term_idx in range(beta.shape[0]):
        out.extend(beta[term_idx][beta_masks[term_idx]].tolist())
    return np.array(out, dtype=float)


def unpack_masked(params: np.ndarray, alpha_masks: list[np.ndarray], beta_masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    R = len(alpha_masks)
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    offset = 0
    for term_idx in range(R):
        count = int(np.sum(alpha_masks[term_idx]))
        alpha[term_idx][alpha_masks[term_idx]] = params[offset:offset + count]
        offset += count
    for term_idx in range(R):
        count = int(np.sum(beta_masks[term_idx]))
        beta[term_idx][beta_masks[term_idx]] = params[offset:offset + count]
        offset += count
    return alpha, beta


def solve_gamma(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    R = alpha.shape[0]
    steering = np.zeros((R, 81))
    alpha_flat = alpha.reshape(R, 9)
    beta_flat = beta.reshape(R, 9)
    for k in range(R):
        steering[k] = np.outer(alpha_flat[k], beta_flat[k]).ravel()

    gamma = np.zeros((R, 9))
    for g in range(9):
        target_slice = TENSOR[g, :, :].ravel()
        gamma[:, g] = np.linalg.lstsq(steering.T, target_slice, rcond=None)[0]
    return gamma.reshape(R, 3, 3)


def compute_fiber_coords(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def full_diagnostics(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    R = alpha.shape[0]
    sigma, h, delta = compute_fiber_coords(alpha, beta)
    nuisance = np.hstack([h, delta])

    alpha_flat = alpha.reshape(R, 9)
    beta_flat = beta.reshape(R, 9)
    gamma_flat = gamma.reshape(R, 9)
    that = np.einsum("kg,ka,kb->gab", gamma_flat, alpha_flat, beta_flat)

    fit_inf = float(np.max(np.abs(TENSOR - that)))
    fit_fro = float(np.sum((TENSOR - that) ** 2))
    rank_h = int(np.linalg.matrix_rank(h, tol=1e-10))
    rank_nuis = int(np.linalg.matrix_rank(nuisance, tol=1e-10))
    aug_rank = int(np.linalg.matrix_rank(np.hstack([sigma, nuisance]), tol=1e-10))
    target_rank = R - 9
    sv = np.linalg.svd(h, compute_uv=False)
    rank_barrier = float(sv[target_rank]) if target_rank < len(sv) else 0.0

    if np.linalg.matrix_rank(h, tol=1e-12) > 0:
        coeff = np.linalg.lstsq(h, delta, rcond=None)[0]
        delta_resid = float(np.linalg.norm(delta - h @ coeff, ord="fro"))
    else:
        delta_resid = float(np.linalg.norm(delta, ord="fro"))

    gamma_op = gamma.reshape(R, 9).T
    gamma_delta = float(np.linalg.norm(gamma_op @ delta, ord="fro"))
    gamma_sigma_err = float(np.linalg.norm(gamma_op @ sigma - 3 * np.eye(9), ord="fro"))
    conservation = int(R + (18 - rank_h))

    return {
        "fitness_inf": fit_inf,
        "fitness_fro": fit_fro,
        "rank_H": rank_h,
        "rank_nuis": rank_nuis,
        "delta_resid": delta_resid,
        "gamma_delta": gamma_delta,
        "gamma_sigma_err": gamma_sigma_err,
        "aug_rank": aug_rank,
        "conservation": conservation,
        "rank_barrier": rank_barrier,
    }


def objective_masked(params: np.ndarray, alpha_masks: list[np.ndarray], beta_masks: list[np.ndarray], R: int, w_fit: float, w_rank: float, w_delta: float) -> float:
    alpha, beta = unpack_masked(params, alpha_masks, beta_masks)
    gamma = solve_gamma(alpha, beta)
    sigma, h, delta = compute_fiber_coords(alpha, beta)

    alpha_flat = alpha.reshape(R, 9)
    beta_flat = beta.reshape(R, 9)
    gamma_flat = gamma.reshape(R, 9)
    that = np.einsum("kg,ka,kb->gab", gamma_flat, alpha_flat, beta_flat)
    loss_fit = np.sum((TENSOR - that) ** 2)

    target_rank = R - 9
    sv = np.linalg.svd(h, compute_uv=False)
    loss_rank = np.sum(sv[target_rank:] ** 2)

    if np.linalg.matrix_rank(h, tol=1e-12) > 0:
        coeff = np.linalg.lstsq(h, delta, rcond=None)[0]
        loss_delta = np.sum((delta - h @ coeff) ** 2)
    else:
        loss_delta = np.sum(delta ** 2)

    return float(w_fit * loss_fit + w_rank * loss_rank + w_delta * loss_delta)


def run_trial(task: tuple[str, int, int]) -> dict:
    family_name, trial_index, maxiter = task
    trial_seed = 20260403 + trial_index * 1000 + sum(ord(c) for c in family_name)
    family = build_family_structure(family_name, trial_seed)
    x0 = pack_masked(family["alpha_init"], family["beta_init"], family["alpha_masks"], family["beta_masks"])

    stage1 = minimize(
        objective_masked,
        x0,
        args=(family["alpha_masks"], family["beta_masks"], family["R"], 0.02, 150.0, 5.0),
        method="L-BFGS-B",
        options={"maxiter": maxiter // 2, "ftol": 1e-20, "gtol": 1e-10},
    )

    stage2 = minimize(
        objective_masked,
        stage1.x,
        args=(family["alpha_masks"], family["beta_masks"], family["R"], 1.0, 80.0, 20.0),
        method="L-BFGS-B",
        options={"maxiter": maxiter // 2, "ftol": 1e-20, "gtol": 1e-10},
    )

    alpha, beta = unpack_masked(stage2.x, family["alpha_masks"], family["beta_masks"])
    gamma = solve_gamma(alpha, beta)
    diag = full_diagnostics(alpha, beta, gamma)
    diag.update(
        {
            "family": family_name,
            "trial": trial_index,
            "R": family["R"],
            "mode": family["mode"],
            "note": family["note"],
            "stage1_loss": float(stage1.fun),
            "stage2_loss": float(stage2.fun),
        }
    )
    return diag


def family_summary(results: list[dict]) -> dict:
    best = min(results, key=lambda item: (item["rank_barrier"], item["fitness_inf"], item["delta_resid"]))
    rank_counts = {}
    conservation_counts = {}
    gate1 = 0
    gate2 = 0
    for row in results:
        rank_counts[row["rank_H"]] = rank_counts.get(row["rank_H"], 0) + 1
        conservation_counts[row["conservation"]] = conservation_counts.get(row["conservation"], 0) + 1
        if row["rank_H"] == row["R"] - 9:
            gate1 += 1
            if row["rank_nuis"] == row["rank_H"]:
                gate2 += 1
    return {
        "count": len(results),
        "gate1": gate1,
        "gate2": gate2,
        "best": best,
        "rank_counts": rank_counts,
        "conservation_counts": conservation_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canon-guided family gate scan")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--out", type=str, default="family_gate_scan.json")
    args = parser.parse_args()

    selected = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown = [name for name in selected if name not in FAMILY_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown families: {unknown}")

    tasks = []
    for family_name in selected:
        for trial in range(args.trials):
            tasks.append((family_name, trial, args.maxiter))

    print("═" * 78)
    print("  CANON-GUIDED FAMILY GATE SCAN")
    print("═" * 78)
    print(f"  Families: {selected}")
    print(f"  Trials per family: {args.trials}")
    print(f"  Workers: {args.workers}")
    print(f"  MaxIter: {args.maxiter}")
    print("  Objective: Stage 1 rank barrier, Stage 2 balanced rank + fit + delta")
    print("═" * 78)

    t0 = time.time()
    grouped = {name: [] for name in selected}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_trial, task): task for task in tasks}
        done = 0
        total = len(tasks)
        for future in as_completed(futures):
            done += 1
            result = future.result()
            grouped[result["family"]].append(result)
            print(
                f"  [{done}/{total}] {result['family']}: "
                f"rk(H)={result['rank_H']} barrier={result['rank_barrier']:.4e} "
                f"fit∞={result['fitness_inf']:.4f} delta={result['delta_resid']:.4e}",
                flush=True,
            )

    elapsed = time.time() - t0
    print("\n" + "═" * 78)
    print("  FAMILY RESULTS")
    print("═" * 78)

    payload = {"elapsed": elapsed, "families": {}}
    for family_name in selected:
        rows = grouped[family_name]
        if not rows:
            continue
        summary = family_summary(rows)
        best = summary["best"]
        payload["families"][family_name] = {"summary": summary, "trials": rows}
        print(f"\n  {family_name}:")
        print(f"    trials={summary['count']} gate1={summary['gate1']} gate2={summary['gate2']}")
        print(
            f"    best: rk(H)={best['rank_H']} barrier={best['rank_barrier']:.4e} "
            f"fit∞={best['fitness_inf']:.6f} delta={best['delta_resid']:.4e} C={best['conservation']}"
        )
        print(f"    rank dist: {summary['rank_counts']}")
        print(f"    conserv dist: {summary['conservation_counts']}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n" + "═" * 78)
    print(f"  Done in {elapsed:.1f}s -> {args.out}")
    print("═" * 78)


if __name__ == "__main__":
    main()