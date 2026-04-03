"""
symmetry_solver.py - Symmetry-reduced R=19 solver for 3x3 matrix multiplication.

The symmetry group is Z2 wr S3, realized as:
  - pi in S3 permuting the three roles (row, summation, column)
  - epsilon in {0,1}^3 applying the involution 1 <-> 2 independently on each role

The solver uses three super-seeds for the orbits of:
  S0 = (0,0,0)  size 1
  S1 = (0,0,1)  size 6
  S2 = (0,1,1)  size 12

Each raw seed has 27 parameters. Before orbit expansion, each seed is projected to
its stabilizer-invariant component, so the orbit construction is symmetry-consistent
while still exposing an 81-parameter interface to the optimizer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from gates import compute_diagnostics
from tensor import build_decomposition, build_target_tensor


ROLE_PAIRS = {
    (0, 1): "alpha",
    (1, 2): "beta",
    (0, 2): "gamma",
}
TARGET = build_target_tensor()
S3 = list(permutations(range(3)))
Z2_FLAGS = list(product((0, 1), repeat=3))
GROUP = [(tuple(pi), tuple(eps)) for pi in S3 for eps in Z2_FLAGS]
Z2_GROUP = [((0, 1, 2), tuple(eps)) for eps in Z2_FLAGS]

SUPER_SEEDS = {
    "S0": (0, 0, 0),
    "S1": (0, 0, 1),
    "S2": (0, 1, 1),
}


def _swap_perm(flag: int) -> tuple[int, int, int]:
    return (0, 2, 1) if flag else (0, 1, 2)


def _basis_matrix(flat_index: int) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[flat_index // 3, flat_index % 3] = 1.0
    return matrix


def _flat_index_from_basis(matrix: np.ndarray) -> int:
    nz = np.flatnonzero(np.abs(matrix.ravel()) > 0.5)
    if len(nz) != 1:
        raise ValueError("Expected a one-hot 3x3 basis matrix.")
    return int(nz[0])


def _factor_from_pair(
    pair: tuple[int, int],
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    if pair == (0, 1):
        return alpha
    if pair == (1, 2):
        return beta
    if pair == (0, 2):
        return gamma
    if pair == (1, 0):
        return alpha.T
    if pair == (2, 1):
        return beta.T
    if pair == (2, 0):
        return gamma.T
    raise ValueError(f"Unsupported role pair {pair}.")


def apply_group_element(
    pi: tuple[int, int, int],
    epsilon: tuple[int, int, int],
    term: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Apply (pi, epsilon) to the index triple (r, s, u)."""
    coords = [term[0], term[1], term[2]]
    permuted = [coords[pi[0]], coords[pi[1]], coords[pi[2]]]
    out = []
    for axis, value in enumerate(permuted):
        if value > 0 and epsilon[axis]:
            out.append(3 - value)
        else:
            out.append(value)
    return tuple(out)


def apply_to_factors(
    pi: tuple[int, int, int],
    epsilon: tuple[int, int, int],
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the group element to the factor triple.

    The role permutation acts on the triangle of role-pairs. For the new edge (a, b),
    we pull back the old edge (pi[a], pi[b]) and transpose when the induced orientation
    reverses. After that, the per-role 1 <-> 2 involutions relabel the row and column
    coordinates of the new matrix.
    """
    role_perms = [_swap_perm(flag) for flag in epsilon]
    out: dict[str, np.ndarray] = {}
    for new_pair, name in ROLE_PAIRS.items():
        old_pair = (pi[new_pair[0]], pi[new_pair[1]])
        matrix = _factor_from_pair(old_pair, alpha, beta, gamma)
        row_perm = role_perms[new_pair[0]]
        col_perm = role_perms[new_pair[1]]
        out[name] = matrix[np.ix_(row_perm, col_perm)].copy()
    return out["alpha"], out["beta"], out["gamma"]


def basis_factors_for_term(term: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row, summation, col = term
    alpha = np.zeros((3, 3), dtype=np.float64)
    beta = np.zeros((3, 3), dtype=np.float64)
    gamma = np.zeros((3, 3), dtype=np.float64)
    alpha[row, summation] = 1.0
    beta[summation, col] = 1.0
    gamma[row, col] = 1.0
    return alpha, beta, gamma


def extract_term_from_basis_factors(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[int, int, int]:
    alpha_idx = _flat_index_from_basis(alpha)
    beta_idx = _flat_index_from_basis(beta)
    gamma_idx = _flat_index_from_basis(gamma)

    alpha_row, alpha_col = divmod(alpha_idx, 3)
    beta_row, beta_col = divmod(beta_idx, 3)
    gamma_row, gamma_col = divmod(gamma_idx, 3)

    if alpha_row != gamma_row or alpha_col != beta_row or beta_col != gamma_col:
        raise ValueError("Factor triple is not a standard basis matrix-multiplication term.")
    return alpha_row, alpha_col, beta_col


def term_orbit(seed: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return sorted({apply_group_element(pi, eps, seed) for pi, eps in GROUP})


def stabilizer(seed: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    return [(pi, eps) for pi, eps in GROUP if apply_group_element(pi, eps, seed) == seed]


def orbit_transporters(
    seed: tuple[int, int, int],
) -> dict[tuple[int, int, int], tuple[tuple[int, int, int], tuple[int, int, int]]]:
    transporters: dict[tuple[int, int, int], tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
    for group_element in GROUP:
        image = apply_group_element(group_element[0], group_element[1], seed)
        transporters.setdefault(image, group_element)
    return {term: transporters[term] for term in sorted(transporters)}


ORBIT_DATA = {
    name: {
        "seed": seed,
        "orbit": term_orbit(seed),
        "stabilizer": stabilizer(seed),
        "transporters": orbit_transporters(seed),
    }
    for name, seed in SUPER_SEEDS.items()
}


def project_seed_to_stabilizer(
    seed_index: tuple[int, int, int],
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a raw seed over its stabilizer to make transporter choice irrelevant."""
    stabs = stabilizer(seed_index)
    alpha_sum = np.zeros((3, 3), dtype=np.float64)
    beta_sum = np.zeros((3, 3), dtype=np.float64)
    gamma_sum = np.zeros((3, 3), dtype=np.float64)
    for pi, eps in stabs:
        a_new, b_new, g_new = apply_to_factors(pi, eps, alpha, beta, gamma)
        alpha_sum += a_new
        beta_sum += b_new
        gamma_sum += g_new
    scale = 1.0 / len(stabs)
    return alpha_sum * scale, beta_sum * scale, gamma_sum * scale


def pack_seed_params(seed_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    pieces = []
    for name in ("S0", "S1", "S2"):
        alpha, beta, gamma = seed_dict[name]
        pieces.extend([alpha.ravel(), beta.ravel(), gamma.ravel()])
    return np.concatenate(pieces).astype(np.float64)


def unpack_seed_params(params: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if params.shape != (81,):
        raise ValueError(f"Expected 81 parameters, received shape {params.shape}.")
    offset = 0
    seeds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("S0", "S1", "S2"):
        alpha = params[offset:offset + 9].reshape(3, 3)
        beta = params[offset + 9:offset + 18].reshape(3, 3)
        gamma = params[offset + 18:offset + 27].reshape(3, 3)
        seeds[name] = (alpha, beta, gamma)
        offset += 27
    return seeds


def build_symmetric_decomposition(
    raw_seed_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Build the 19-term decomposition from the three projected super-seeds."""
    projected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, seed_index in SUPER_SEEDS.items():
        projected[name] = project_seed_to_stabilizer(seed_index, *raw_seed_dict[name])

    terms: list[tuple[tuple[int, int, int], str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name in ("S0", "S1", "S2"):
        seed_index = SUPER_SEEDS[name]
        alpha_seed, beta_seed, gamma_seed = projected[name]
        for term_index, group_element in ORBIT_DATA[name]["transporters"].items():
            alpha_term, beta_term, gamma_term = apply_to_factors(
                group_element[0], group_element[1], alpha_seed, beta_seed, gamma_seed
            )
            terms.append((term_index, name, alpha_term, beta_term, gamma_term))

    terms.sort(key=lambda item: item[0])
    alpha = np.array([item[2] for item in terms], dtype=np.float64)
    beta = np.array([item[3] for item in terms], dtype=np.float64)
    gamma = np.array([item[4] for item in terms], dtype=np.float64)
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "term_indices": [item[0] for item in terms],
        "orbit_labels": [item[1] for item in terms],
        "projected_seeds": projected,
    }


def objective_from_params(params: np.ndarray) -> float:
    built = build_symmetric_decomposition(unpack_seed_params(params))
    residual = TARGET - build_decomposition(built["alpha"], built["beta"], built["gamma"])
    return float(np.sum(residual * residual))


def apply_group_to_tensor_coordinate(
    pi: tuple[int, int, int],
    epsilon: tuple[int, int, int],
    coordinate: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Act on a tensor basis coordinate (gamma_idx, alpha_idx, beta_idx)."""
    gamma_basis = _basis_matrix(coordinate[0])
    alpha_basis = _basis_matrix(coordinate[1])
    beta_basis = _basis_matrix(coordinate[2])
    alpha_new, beta_new, gamma_new = apply_to_factors(pi, epsilon, alpha_basis, beta_basis, gamma_basis)
    return (
        _flat_index_from_basis(gamma_new),
        _flat_index_from_basis(alpha_new),
        _flat_index_from_basis(beta_new),
    )


def tensor_equation_orbits(
    group_elements: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> list[list[tuple[int, int, int]]]:
    remaining = {
        (gamma_idx, alpha_idx, beta_idx)
        for gamma_idx in range(9)
        for alpha_idx in range(9)
        for beta_idx in range(9)
    }
    orbits: list[list[tuple[int, int, int]]] = []
    while remaining:
        start = min(remaining)
        orbit = {
            apply_group_to_tensor_coordinate(pi, eps, start)
            for pi, eps in group_elements
        }
        expanded = set(orbit)
        frontier = list(orbit)
        while frontier:
            current = frontier.pop()
            for pi, eps in group_elements:
                image = apply_group_to_tensor_coordinate(pi, eps, current)
                if image not in expanded:
                    expanded.add(image)
                    frontier.append(image)
        remaining -= expanded
        orbits.append(sorted(expanded))
    orbits.sort(key=lambda orbit: orbit[0])
    return orbits


def verify_symmetry_setup(tol: float = 1e-10) -> dict[str, Any]:
    """Run the exact structural checks for the symmetry action."""
    standard_terms = [basis_factors_for_term(term) for term in product(range(3), repeat=3)]
    standard_alpha = np.array([term[0] for term in standard_terms], dtype=np.float64)
    standard_beta = np.array([term[1] for term in standard_terms], dtype=np.float64)
    standard_gamma = np.array([term[2] for term in standard_terms], dtype=np.float64)
    standard_tensor = build_decomposition(standard_alpha, standard_beta, standard_gamma)

    equivariance_errors = []
    term_map_errors = []
    for pi, eps in GROUP:
        transformed_alpha = []
        transformed_beta = []
        transformed_gamma = []
        for term in product(range(3), repeat=3):
            alpha_term, beta_term, gamma_term = basis_factors_for_term(term)
            alpha_new, beta_new, gamma_new = apply_to_factors(pi, eps, alpha_term, beta_term, gamma_term)
            term_map_errors.append(
                extract_term_from_basis_factors(alpha_new, beta_new, gamma_new)
                != apply_group_element(pi, eps, term)
            )
            transformed_alpha.append(alpha_new)
            transformed_beta.append(beta_new)
            transformed_gamma.append(gamma_new)
        tensor_new = build_decomposition(
            np.array(transformed_alpha, dtype=np.float64),
            np.array(transformed_beta, dtype=np.float64),
            np.array(transformed_gamma, dtype=np.float64),
        )
        equivariance_errors.append(float(np.max(np.abs(standard_tensor - tensor_new))))

    orbit_sizes = {
        "corner": len(term_orbit((0, 0, 0))),
        "edge": len(term_orbit((0, 0, 1))),
        "face": len(term_orbit((0, 1, 1))),
        "interior": len(term_orbit((1, 1, 1))),
    }
    kept_terms = set(term_orbit((0, 0, 0))) | set(term_orbit((0, 0, 1))) | set(term_orbit((0, 1, 1)))
    deleted_terms = set(term_orbit((1, 1, 1)))
    z2_equation_orbits = tensor_equation_orbits(Z2_GROUP)
    full_equation_orbits = tensor_equation_orbits(GROUP)

    transporter_consistency = {}
    for name in ("S0", "S1", "S2"):
        seed_idx = SUPER_SEEDS[name]
        raw = tuple(np.random.default_rng(0).standard_normal((3, 3)) for _ in range(3))
        projected = project_seed_to_stabilizer(seed_idx, *raw)
        max_gap = 0.0
        by_target: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
        for group_element in GROUP:
            by_target.setdefault(apply_group_element(group_element[0], group_element[1], seed_idx), []).append(group_element)
        for target, candidates in by_target.items():
            first = apply_to_factors(candidates[0][0], candidates[0][1], *projected)
            for group_element in candidates[1:]:
                other = apply_to_factors(group_element[0], group_element[1], *projected)
                max_gap = max(
                    max_gap,
                    float(np.max(np.abs(first[0] - other[0]))),
                    float(np.max(np.abs(first[1] - other[1]))),
                    float(np.max(np.abs(first[2] - other[2]))),
                )
        transporter_consistency[name] = max_gap

    return {
        "group_size": len(GROUP),
        "orbit_sizes": orbit_sizes,
        "stabilizer_sizes": {name: len(ORBIT_DATA[name]["stabilizer"]) for name in ("S0", "S1", "S2")},
        "kept_term_count": len(kept_terms),
        "deleted_term_count": len(deleted_terms),
        "kept_terms": sorted(kept_terms),
        "deleted_terms": sorted(deleted_terms),
        "all_kept_generated": len(kept_terms) == 19,
        "zero_mixed_orbits": len(kept_terms & deleted_terms) == 0,
        "max_standard_equivariance_error": max(equivariance_errors),
        "basis_term_maps_correctly": not any(term_map_errors),
        "tensor_equation_orbit_count": len(z2_equation_orbits),
        "tensor_equation_orbit_sizes": sorted(len(orbit) for orbit in z2_equation_orbits),
        "tensor_equation_orbit_count_full_group": len(full_equation_orbits),
        "tensor_equation_orbit_sizes_full_group": sorted(len(orbit) for orbit in full_equation_orbits),
        "transport_consistency_after_projection": transporter_consistency,
        "verification_passed": (
            len(GROUP) == 48
            and orbit_sizes == {"corner": 1, "edge": 6, "face": 12, "interior": 8}
            and len(kept_terms) == 19
            and len(deleted_terms) == 8
            and max(equivariance_errors) <= tol
            and not any(term_map_errors)
            and len(z2_equation_orbits) == 125
            and all(value <= tol for value in transporter_consistency.values())
        ),
    }


@dataclass
class SearchConfig:
    runs: int = 1
    seed: int = 0
    seed_scale: float = 0.3
    maxiter: int = 500
    log_every: int = 10
    method: str = "L-BFGS-B"


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    return value


def run_search(config: SearchConfig) -> dict[str, Any]:
    verification = verify_symmetry_setup()
    if not verification["verification_passed"]:
        raise RuntimeError("Symmetry verification failed; refusing to optimize.")

    best: dict[str, Any] | None = None
    for run_idx in range(config.runs):
        rng = np.random.default_rng(config.seed + run_idx)
        x0 = rng.standard_normal(81) * config.seed_scale
        history: list[dict[str, float | int]] = []
        call_counter = {"count": 0}
        accepted_iterations = {"count": 0}
        best_run_value = float("inf")
        start_time = time.time()

        def callback(xk: np.ndarray) -> None:
            accepted_iterations["count"] += 1
            if accepted_iterations["count"] % config.log_every != 0:
                return
            current_loss = objective_from_params(xk)
            history.append({
                "iter": accepted_iterations["count"],
                "loss_fro": float(current_loss),
                "elapsed": float(time.time() - start_time),
            })

        def objective(params: np.ndarray) -> float:
            call_counter["count"] += 1
            return objective_from_params(params)

        initial_loss = objective(x0)
        history.append({"iter": 0, "loss_fro": float(initial_loss), "elapsed": 0.0})
        result = minimize(
            objective,
            x0,
            method=config.method,
            callback=callback,
            options={"maxiter": config.maxiter, "disp": False},
        )

        final_params = np.array(result.x, dtype=np.float64)
        built = build_symmetric_decomposition(unpack_seed_params(final_params))
        diagnostics = compute_diagnostics(built["alpha"], built["beta"], built["gamma"])
        diagnostics["objective_fro"] = float(result.fun)
        diagnostics["objective_history_monotone"] = all(
            history[idx + 1]["loss_fro"] <= history[idx]["loss_fro"] + 1e-12
            for idx in range(len(history) - 1)
        )
        diagnostics["optimizer_calls"] = call_counter["count"]
        diagnostics["optimizer_success"] = bool(result.success)
        diagnostics["optimizer_message"] = str(result.message)

        run_result = {
            "run": run_idx,
            "seed": config.seed + run_idx,
            "raw_seed_params": final_params,
            "raw_seeds": unpack_seed_params(final_params),
            "projected_seeds": built["projected_seeds"],
            "alpha": built["alpha"],
            "beta": built["beta"],
            "gamma": built["gamma"],
            "term_indices": built["term_indices"],
            "orbit_labels": built["orbit_labels"],
            "verification": verification,
            "diagnostics": diagnostics,
            "history": history,
        }

        best_run_value = diagnostics["max_abs_residual"]
        if best is None or best_run_value < best["diagnostics"]["max_abs_residual"]:
            best = run_result

    if best is None:
        raise RuntimeError("Search produced no result.")
    return best


def save_result(result: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(result), handle, indent=2)