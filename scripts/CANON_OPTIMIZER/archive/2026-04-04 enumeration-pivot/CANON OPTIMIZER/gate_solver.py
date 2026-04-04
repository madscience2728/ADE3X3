"""
gate_solver.py - Structure-first R=19 solver.

Philosophy: Do NOT minimize tensor residual. Instead, find (alpha, beta) satisfying:
  Gate 1: rank(H) = 10  (equivalently, H has exactly 8 zero singular values)
  Gate 2: Delta in span(H)  (delta containment)

Once Gates 1+2 are satisfied, Gamma falls out of a linear solve and residual=0 is automatic.

The loss function is purely structural:
  L = w1 * sum(sigma_{11..18}^2)      # push last 8 singular values of H to zero
    + w2 * ||Delta - H @ lstsq(H, Delta)||_F^2  # delta containment residual

When L=0, we check the Gamma solvability (Gate 3) as a linear system.
"""

from __future__ import annotations

import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from symmetry_solver import (
    GROUP, SUPER_SEEDS, ORBIT_DATA,
    unpack_seed_params, pack_seed_params,
    build_symmetric_decomposition,
    project_seed_to_stabilizer,
    verify_symmetry_setup,
    _jsonify,
)
from tensor import build_fiber_coordinates, build_target_tensor, build_decomposition


TARGET = build_target_tensor()


def structural_loss(params: np.ndarray) -> float:
    """Pure structural loss: push rank(H) to 10 and Delta into span(H).

    Uses log-barrier on tail singular values for stronger gradient signal
    when SVs are small but nonzero.
    """
    built = build_symmetric_decomposition(unpack_seed_params(params))
    alpha, beta = built["alpha"], built["beta"]
    coords = build_fiber_coordinates(alpha, beta)
    h_block = coords["H"]       # (19, 18)
    delta = coords["Delta"]     # (19, 54)

    # Gate 1: H should have rank 10 => SVs 10..17 should be 0
    sv_h = np.linalg.svd(h_block, compute_uv=False)
    tail = sv_h[10:]
    # Use sum of squares + log(1 + sv^2/eps) for stronger gradient on small SVs
    gate1 = float(np.sum(tail ** 2) + 0.1 * np.sum(np.log1p(tail ** 2 / 1e-12)))

    # Gate 2: Delta in span(H) — projection residual
    M, _, _, _ = np.linalg.lstsq(h_block, delta, rcond=None)
    gate2 = float(np.sum((delta - h_block @ M) ** 2))

    return gate1 + gate2


def structural_loss_detailed(params: np.ndarray) -> dict[str, Any]:
    """Return detailed breakdown of structural metrics."""
    built = build_symmetric_decomposition(unpack_seed_params(params))
    alpha, beta, gamma = built["alpha"], built["beta"], built["gamma"]
    coords = build_fiber_coordinates(alpha, beta)
    h_block = coords["H"]
    delta = coords["Delta"]
    sigma = coords["Sigma"]
    nuisance = coords["Nuisance"]

    sv_h = np.linalg.svd(h_block, compute_uv=False)
    M, _, _, _ = np.linalg.lstsq(h_block, delta, rcond=None)
    delta_residual = float(np.sum((delta - h_block @ M) ** 2))

    gate1 = float(np.sum(sv_h[10:] ** 2))
    rank_h = int(np.sum(sv_h > 1e-10))
    rank_nuisance = int(np.linalg.matrix_rank(nuisance, tol=1e-10))

    return {
        "gate1_loss": gate1,
        "gate2_loss": delta_residual,
        "total_loss": gate1 + delta_residual,
        "rank_H": rank_h,
        "rank_nuisance": rank_nuisance,
        "sv_H": sv_h.tolist(),
        "conservation": 19 + (18 - rank_h),
    }


def solve_gamma(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Given (alpha, beta) satisfying Gates 1+2, solve for Gamma via linear system.

    Returns (gamma, info) where gamma is None if Gate 3 fails.
    """
    coords = build_fiber_coordinates(alpha, beta)
    sigma = coords["Sigma"]       # (19, 9)
    nuisance = coords["Nuisance"] # (19, 72)

    # We need Gamma (9 x 19) such that:
    #   Gamma @ Sigma = 3 * I_9      (81 equations)
    #   Gamma @ Nuisance = 0         (648 equations)
    #
    # Rewrite as: for each of the 9 rows gamma_i of Gamma:
    #   gamma_i @ Sigma[:,j] = 3 * delta_{ij}   for j=0..8
    #   gamma_i @ Nuisance = 0                    (72 equations)
    #
    # Combined: gamma_i @ [Sigma | Nuisance] = [3*e_i | 0]
    # This is a 19-variable linear system with 9+72 = 81 equations per row.

    design = np.hstack([sigma, nuisance])  # (19, 81)

    # Check solvability: rank([Sigma | Nuisance]) should be 19
    rank_aug = np.linalg.matrix_rank(design, tol=1e-10)

    info = {
        "rank_augmented": rank_aug,
        "rank_nuisance": int(np.linalg.matrix_rank(nuisance, tol=1e-10)),
        "solvable": rank_aug == 19,
    }

    if rank_aug < 19:
        info["reason"] = f"rank([Sigma|Nuisance]) = {rank_aug} < 19"
        return None, info

    # Solve: Gamma @ design = target_rhs
    # target_rhs[i, :] = [3*e_i | 0_{72}]
    target_rhs = np.zeros((9, 81), dtype=np.float64)
    target_rhs[:, :9] = 3.0 * np.eye(9)

    # Gamma @ design = target_rhs  =>  design.T @ Gamma.T = target_rhs.T
    # Gamma.T shape (19, 9), design.T shape (81, 19), target_rhs.T shape (81, 9)
    gamma_T, res, _, _ = np.linalg.lstsq(design.T, target_rhs.T, rcond=None)
    gamma_flat = gamma_T.T  # (9, 19)

    # Verify
    sigma_check = gamma_flat @ sigma  # should be 3*I_9
    nuisance_check = gamma_flat @ nuisance  # should be 0
    sigma_error = float(np.max(np.abs(sigma_check - 3.0 * np.eye(9))))
    nuisance_error = float(np.max(np.abs(nuisance_check)))

    info["sigma_error"] = sigma_error
    info["nuisance_error"] = nuisance_error
    info["gamma_exact"] = sigma_error < 1e-8 and nuisance_error < 1e-8

    if not info["gamma_exact"]:
        return None, info

    gamma = gamma_flat.T.reshape(19, 3, 3)  # transpose: each row of gamma_flat is a 9-vec

    # Final verification: reconstruct tensor
    tensor_hat = build_decomposition(alpha, beta, gamma)
    tensor_residual = float(np.max(np.abs(TARGET - tensor_hat)))
    info["tensor_residual"] = tensor_residual
    info["exact_decomposition"] = tensor_residual < 1e-8

    return gamma, info


@dataclass
class GateSearchConfig:
    runs: int = 1
    seed: int = 0
    seed_scale: float = 0.3
    maxiter: int = 2000
    log_every: int = 50
    method: str = "L-BFGS-B"
    workers: int | None = None


def _run_single_gate_search(run_idx: int, config: GateSearchConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + run_idx)
    x0 = rng.standard_normal(81) * config.seed_scale
    history: list[dict[str, Any]] = []
    call_counter = {"count": 0}
    accepted = {"count": 0}
    start_time = time.time()

    def callback(xk: np.ndarray) -> None:
        accepted["count"] += 1
        if accepted["count"] % config.log_every != 0:
            return
        detail = structural_loss_detailed(xk)
        history.append({
            "iter": accepted["count"],
            "gate1": detail["gate1_loss"],
            "gate2": detail["gate2_loss"],
            "total": detail["total_loss"],
            "rank_H": detail["rank_H"],
            "elapsed": time.time() - start_time,
        })

    def objective(params: np.ndarray) -> float:
        call_counter["count"] += 1
        return structural_loss(params)

    initial = structural_loss_detailed(x0)
    history.append({
        "iter": 0,
        "gate1": initial["gate1_loss"],
        "gate2": initial["gate2_loss"],
        "total": initial["total_loss"],
        "rank_H": initial["rank_H"],
        "elapsed": 0.0,
    })

    result = minimize(
        objective,
        x0,
        method=config.method,
        callback=callback,
        options={"maxiter": config.maxiter, "disp": False},
    )

    final_params = np.array(result.x, dtype=np.float64)
    detail = structural_loss_detailed(final_params)

    # Attempt Gamma solve if structural loss is small
    built = build_symmetric_decomposition(unpack_seed_params(final_params))
    gamma_result, gamma_info = solve_gamma(built["alpha"], built["beta"])

    elapsed = time.time() - start_time

    return {
        "run": run_idx,
        "seed": config.seed + run_idx,
        "params": final_params,
        "structural": detail,
        "gamma_info": gamma_info,
        "gamma_solved": gamma_result is not None,
        "history": history,
        "optimizer_calls": call_counter["count"],
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "elapsed_seconds": elapsed,
    }


def run_gate_search(config: GateSearchConfig) -> dict[str, Any]:
    verification = verify_symmetry_setup()
    if not verification["verification_passed"]:
        raise RuntimeError("Symmetry verification failed.")

    worker_count = max(1, min(config.runs, config.workers or os.cpu_count() or 1))
    print(f"Gate solver: runs={config.runs} workers={worker_count} maxiter={config.maxiter}")

    best: dict[str, Any] | None = None
    completed = 0

    if worker_count == 1:
        for run_idx in range(config.runs):
            r = _run_single_gate_search(run_idx, config)
            completed += 1
            _print_gate_summary(r, completed, config.runs)
            if best is None or r["structural"]["total_loss"] < best["structural"]["total_loss"]:
                best = r
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_single_gate_search, run_idx, config): run_idx
                for run_idx in range(config.runs)
            }
            for future in as_completed(futures):
                r = future.result()
                completed += 1
                _print_gate_summary(r, completed, config.runs)
                if best is None or r["structural"]["total_loss"] < best["structural"]["total_loss"]:
                    best = r

    print(f"\n{'='*70}")
    print(f"BEST: total={best['structural']['total_loss']:.6e} "
          f"gate1={best['structural']['gate1_loss']:.6e} "
          f"gate2={best['structural']['gate2_loss']:.6e} "
          f"rank_H={best['structural']['rank_H']} "
          f"conservation={best['structural']['conservation']} "
          f"gamma_solved={best['gamma_solved']}")

    if best["gamma_solved"]:
        print("*** EXACT DECOMPOSITION FOUND ***")

    return best


def _print_gate_summary(r: dict[str, Any], completed: int, total: int) -> None:
    s = r["structural"]
    print(
        f"  [{completed}/{total}] seed={r['seed']} "
        f"total={s['total_loss']:.4e} "
        f"G1={s['gate1_loss']:.4e} G2={s['gate2_loss']:.4e} "
        f"rk(H)={s['rank_H']} cons={s['conservation']} "
        f"Γ_ok={r['gamma_solved']} "
        f"{r['elapsed_seconds']:.1f}s"
        f"  sv_tail={[f'{v:.1e}' for v in r['structural']['sv_H'][10:]]}"
    )


def _jsonify_gate(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: _jsonify_gate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify_gate(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify_gate(item) for item in value]
    return value


def save_gate_result(result: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(_jsonify_gate(result), f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Structure-first R=19 gate solver")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-scale", type=float, default=0.3)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--method", type=str, default="L-BFGS-B")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=str, default="CANON OPTIMIZER/result_gates.json")
    args = parser.parse_args()

    config = GateSearchConfig(
        runs=args.runs,
        seed=args.seed,
        seed_scale=args.seed_scale,
        maxiter=args.maxiter,
        log_every=args.log_every,
        method=args.method,
        workers=args.workers,
    )
    best = run_gate_search(config)
    save_gate_result(best, args.out)
