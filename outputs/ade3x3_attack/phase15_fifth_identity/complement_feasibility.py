from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from construct_matrices import alpha_nullspace_and_complement


OUT_DIR = Path(__file__).resolve().parent


def build_tensor_unfold() -> np.ndarray:
    from outputs.ade3x3_attack.phase12_working_optimizer.cp_als import build_cp_tensor_cab

    return build_cp_tensor_cab().reshape(9, 81)


def matrix_rank(matrix: np.ndarray, tol: float = 1e-8) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def sample_point(q_matrices: list[np.ndarray], rng: np.random.Generator, max_attempts: int = 4096) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(max_attempts):
        a_vec = rng.standard_normal(9)
        norm_a = np.linalg.norm(a_vec)
        if norm_a < 1e-12:
            continue
        a_vec = a_vec / norm_a
        constraint = np.stack([q_matrix.T @ a_vec for q_matrix in q_matrices], axis=0)
        rank = matrix_rank(constraint)
        if rank >= 9:
            continue
        _, _, vh = np.linalg.svd(constraint, full_matrices=True)
        null_basis = vh[rank:].T
        coeffs = rng.standard_normal(null_basis.shape[1])
        b_vec = null_basis @ coeffs
        norm_b = np.linalg.norm(b_vec)
        if norm_b < 1e-12:
            continue
        return a_vec, b_vec / norm_b
    return None


def feasibility_score(q_matrices: list[np.ndarray], sample_count: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    for _ in range(sample_count):
        point = sample_point(q_matrices, rng)
        if point is None:
            continue
        a_vec, b_vec = point
        rows.append(np.kron(a_vec, b_vec))
        sigma_rows.append((a_vec.reshape(3, 3) @ b_vec.reshape(3, 3)).reshape(9))
    if not rows:
        return {'sampled_points': 0, 'rank_W': 0, 'rank_sigma': 0, 'projection_loss': float('inf'), 'projection_max_abs': float('inf')}

    w_matrix = np.stack(rows, axis=0)
    sigma_matrix = np.stack(sigma_rows, axis=0)
    target = build_tensor_unfold()
    gamma = np.linalg.lstsq(w_matrix.T, target.T, rcond=None)[0]
    residual = w_matrix.T @ gamma - target.T
    return {
        'sampled_points': w_matrix.shape[0],
        'rank_W': matrix_rank(w_matrix, tol=1e-7),
        'rank_sigma': matrix_rank(sigma_matrix, tol=1e-7),
        'projection_loss': float(np.sum(residual * residual)),
        'projection_max_abs': float(np.max(np.abs(residual))),
    }


def main() -> None:
    _, complement_basis, alpha_q_matrices, _, complement_q_matrices = alpha_nullspace_and_complement()
    rows: list[dict[str, object]] = []
    for basis_idx, q_matrix in enumerate(complement_q_matrices, start=1):
        score = feasibility_score(alpha_q_matrices + [q_matrix], sample_count=100, seed=20261000 + basis_idx)
        rows.append(
            {
                'basis_id': basis_idx,
                'sampled_points': score['sampled_points'],
                'rank_W': score['rank_W'],
                'rank_sigma': score['rank_sigma'],
                'projection_loss': score['projection_loss'],
                'projection_max_abs': score['projection_max_abs'],
            }
        )

    rows.sort(key=lambda row: float(row['projection_loss']))
    with open(OUT_DIR / 'complement_feasibility.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['basis_id', 'sampled_points', 'rank_W', 'rank_sigma', 'projection_loss', 'projection_max_abs']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'best_basis_id': rows[0]['basis_id'] if rows else None,
        'best_row': rows[0] if rows else None,
        'rows': rows,
        'complement_basis': complement_basis.tolist(),
    }
    (OUT_DIR / 'complement_feasibility_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()