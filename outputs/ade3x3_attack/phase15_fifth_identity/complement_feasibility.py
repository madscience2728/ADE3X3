from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from matrix_core import alpha_nullspace_and_complement, q_matrix_from_identity


OUT_DIR = Path(__file__).resolve().parent


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def build_tensor_unfold() -> np.ndarray:
    from outputs.ade3x3_attack.attack_common import target_output_unfolding_9x81

    return target_output_unfolding_9x81()


def matrix_rank(matrix: np.ndarray, tol: float = 1e-8) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def null_space_basis(matrix: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, int]:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return vh[rank:].T, rank


def sample_point(
    q_matrices: list[np.ndarray],
    rng: np.random.Generator,
    max_attempts: int = 4096,
    rank_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray] | None:
    target_rank = len(q_matrices)
    for _ in range(max_attempts):
        a_vec = rng.standard_normal(9)
        norm_a = np.linalg.norm(a_vec)
        if norm_a < 1e-12:
            continue
        a_vec = a_vec / norm_a
        constraint = np.stack([q_matrix.T @ a_vec for q_matrix in q_matrices], axis=0)
        null_basis, rank = null_space_basis(constraint, tol=rank_tol)
        if rank != target_rank or null_basis.shape[1] != 9 - target_rank:
            continue
        coeffs = rng.standard_normal(null_basis.shape[1])
        b_vec = null_basis @ coeffs
        norm_b = np.linalg.norm(b_vec)
        if norm_b < 1e-12:
            continue
        return a_vec, b_vec / norm_b
    return None


def projection_residual(target_unfold: np.ndarray, w_matrix: np.ndarray, tol: float = 1e-8) -> tuple[float, float, int]:
    if w_matrix.size == 0:
        return float('inf'), float('inf'), 0
    _, singular_values, vt = np.linalg.svd(w_matrix, full_matrices=False)
    rank = int(np.sum(singular_values > tol))
    if rank == 0:
        return float('inf'), float('inf'), 0
    row_basis = vt[:rank]
    projected = target_unfold @ row_basis.T @ row_basis
    residual = target_unfold - projected
    return float(np.sum(residual * residual)), float(np.max(np.abs(residual))), rank


def feasibility_score(q_matrices: list[np.ndarray], sample_count: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    w_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    failed_samples = 0
    for _ in range(sample_count):
        point = sample_point(q_matrices, rng)
        if point is None:
            failed_samples += 1
            continue
        a_vec, b_vec = point
        w_rows.append(np.kron(a_vec, b_vec))
        sigma_rows.append((a_vec.reshape(3, 3) @ b_vec.reshape(3, 3)).reshape(9))

    if not w_rows:
        return {
            'sampled_points': 0,
            'failed_samples': failed_samples,
            'rank_W': 0,
            'rank_sigma': 0,
            'projection_loss': float('inf'),
            'projection_max_abs': float('inf'),
        }

    w_matrix = np.stack(w_rows, axis=0)
    sigma_matrix = np.stack(sigma_rows, axis=0)
    projection_loss, projection_max_abs, w_rank = projection_residual(build_tensor_unfold(), w_matrix)
    return {
        'sampled_points': int(w_matrix.shape[0]),
        'failed_samples': failed_samples,
        'rank_W': w_rank,
        'rank_sigma': matrix_rank(sigma_matrix, tol=1e-7),
        'projection_loss': projection_loss,
        'projection_max_abs': projection_max_abs,
    }


def scan_basis_directions(sample_count: int, seed: int) -> tuple[list[dict[str, object]], np.ndarray]:
    _, complement_basis, alpha_q_matrices, anisotropy_mats, _ = alpha_nullspace_and_complement()
    alpha_q_list = [matrix for matrix in alpha_q_matrices]
    rows: list[dict[str, object]] = []
    for basis_idx, basis_vector in enumerate(complement_basis, start=1):
        q5 = q_matrix_from_identity(basis_vector, anisotropy_mats)
        score = feasibility_score(alpha_q_list + [q5], sample_count=sample_count, seed=seed + basis_idx)
        rows.append(
            {
                'basis_id': basis_idx,
                'sampled_points': score['sampled_points'],
                'failed_samples': score['failed_samples'],
                'rank_W': score['rank_W'],
                'rank_sigma': score['rank_sigma'],
                'projection_loss': score['projection_loss'],
                'projection_max_abs': score['projection_max_abs'],
            }
        )
    rows.sort(key=lambda row: float(row['projection_loss']))
    return rows, complement_basis


def main() -> None:
    sample_count = env_int('ADE3X3_PHASE15_SCAN_POINTS', 200)
    seed = env_int('ADE3X3_PHASE15_SCAN_SEED', 20260329)
    rows, complement_basis = scan_basis_directions(sample_count=sample_count, seed=seed)

    with open(OUT_DIR / 'direction_scores.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['basis_id', 'sampled_points', 'failed_samples', 'rank_W', 'rank_sigma', 'projection_loss', 'projection_max_abs']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'sample_count': sample_count,
        'seed': seed,
        'best_basis_id': rows[0]['basis_id'] if rows else None,
        'best_row': rows[0] if rows else None,
        'rows': rows,
        'complement_basis': complement_basis.tolist(),
    }
    (OUT_DIR / 'complement_feasibility_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()