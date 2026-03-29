from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (  # noqa: E402
    build_mode_matrices_numeric,
    load_public_terms,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term  # noqa: E402


RANK_TOL = 1e-10


def standard_rank27_terms() -> list[Term]:
    terms: list[Term] = []
    term_idx = 1
    for row_idx in range(3):
        for shared_idx in range(3):
            for col_idx in range(3):
                alpha = np.zeros((3, 3), dtype=np.int64)
                beta = np.zeros((3, 3), dtype=np.int64)
                gamma = np.zeros((3, 3), dtype=np.int64)
                alpha[row_idx, shared_idx] = 1
                beta[shared_idx, col_idx] = 1
                gamma[row_idx, col_idx] = 1
                terms.append(Term(f's{term_idx:02d}', f'standard_rank27_{term_idx:02d}', alpha, beta, gamma))
                term_idx += 1
    return terms


def stacked_alpha_beta(terms: list[Term]) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def live_q_vectors(alpha: np.ndarray, beta: np.ndarray, shared_idx: int) -> np.ndarray:
    cols: list[np.ndarray] = []
    for row_idx in range(3):
        for col_idx in range(3):
            cols.append(alpha[:, 3 * row_idx + shared_idx] * beta[:, 3 * shared_idx + col_idx])
    return np.column_stack(cols)


def compute_channel_scalings(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, float]:
    rank_value = alpha.shape[0]
    q0 = live_q_vectors(alpha, beta, 0)
    q1 = live_q_vectors(alpha, beta, 1)
    q2 = live_q_vectors(alpha, beta, 2)

    a_matrix = np.zeros((18, 3 * rank_value), dtype=np.float64)
    for k in range(rank_value):
        a_matrix[0:9, k] = lam[k] * q0[k, :]
        a_matrix[0:9, rank_value + k] = -lam[k] * q1[k, :]
        a_matrix[9:18, rank_value + k] = lam[k] * q1[k, :]
        a_matrix[9:18, 2 * rank_value + k] = -lam[k] * q2[k, :]

    ones = np.ones(3 * rank_value, dtype=np.float64)
    gram = a_matrix @ a_matrix.T
    correction = a_matrix.T @ np.linalg.pinv(gram, rcond=RANK_TOL) @ (a_matrix @ ones)
    scales = ones - correction
    projection_residual = float(np.max(np.abs(a_matrix @ scales)))
    return scales, projection_residual


def apply_channel_scales(alpha: np.ndarray, scales: np.ndarray, t_value: float) -> np.ndarray:
    rank_value = alpha.shape[0]
    blended = (1.0 - t_value) * np.ones_like(scales) + t_value * scales
    alpha_scaled = alpha.copy().reshape(rank_value, 3, 3)
    for shared_idx in range(3):
        channel_scale = blended[shared_idx * rank_value:(shared_idx + 1) * rank_value]
        alpha_scaled[:, :, shared_idx] *= channel_scale[:, None]
    return alpha_scaled.reshape(rank_value, 9)


def choose_lambda_candidates(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, count: int = 3) -> list[tuple[str, np.ndarray]]:
    gamma_num = gamma.T
    ker_basis = nullspace_basis_numeric(gamma_num)
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix='base_')
    _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
    restricted = ker_basis.T @ h_matrix
    _, _, vh = np.linalg.svd(restricted, full_matrices=False)
    candidates: list[tuple[str, np.ndarray]] = []

    # Softest kernel direction against H.
    left_u, _, _ = np.linalg.svd(restricted, full_matrices=False)
    lam_soft = ker_basis @ left_u[:, -1]
    lam_soft = lam_soft / max(np.linalg.norm(lam_soft), 1e-12)
    candidates.append(('softest_kernel_direction', lam_soft))

    # Keep one deterministic wildcard direction in every future scan.
    rng = np.random.default_rng(20260329 + alpha.shape[0])
    if count >= 2 and ker_basis.shape[1] > 0:
        best_lambda = lam_soft
        best_score = -math.inf
        for _ in range(32):
            coeff = rng.standard_normal(ker_basis.shape[1])
            coeff /= max(np.linalg.norm(coeff), 1e-12)
            lam_trial = ker_basis @ coeff
            lam_trial /= max(np.linalg.norm(lam_trial), 1e-12)
            novelty = 1.0 - abs(float(lam_trial @ lam_soft))
            h_support = float(np.linalg.norm(lam_trial @ h_matrix))
            score = novelty * h_support
            if score > best_score:
                best_lambda = lam_trial
                best_score = score
        candidates.append(('wildcard_kernel_direction', best_lambda))

    # Fill the remaining slots with random kernel directions for contrast.
    for idx in range(max(0, count - len(candidates))):
        coeff = rng.standard_normal(ker_basis.shape[1])
        lam = ker_basis @ coeff
        lam /= max(np.linalg.norm(lam), 1e-12)
        candidates.append((f'random_kernel_direction_{idx + 1}', lam))
    return candidates


def principal_angle_degrees(vector: np.ndarray, subspace_basis: np.ndarray) -> float:
    if subspace_basis.size == 0:
        return 90.0
    coeff = subspace_basis.T @ vector
    proj_norm = float(np.linalg.norm(coeff))
    vec_norm = float(np.linalg.norm(vector))
    cosine = 0.0 if vec_norm == 0.0 else max(-1.0, min(1.0, proj_norm / vec_norm))
    return float(np.degrees(np.arccos(cosine)))


def evaluate_path(base_label: str, alpha: np.ndarray, beta: np.ndarray, candidate_name: str, lam: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_gamma = solve_gamma_least_squares(alpha, beta)
    scales, annihilator_projection_residual = compute_channel_scalings(alpha, beta, lam)

    rows: list[dict[str, object]] = []
    for t_value in [idx / 10 for idx in range(11)]:
        alpha_t = apply_channel_scales(alpha, scales, t_value)
        gamma_t = solve_gamma_least_squares(alpha_t, beta)
        terms_t = terms_from_stacked_factors(alpha_t, beta, gamma_t, prefix=f'{candidate_name}_{t_value:.1f}_')
        _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms_t)
        gamma_num = gamma_t.T
        ker_basis_t = nullspace_basis_numeric(gamma_num)
        restricted_t = ker_basis_t.T @ h_matrix
        lambda_h = lam @ h_matrix
        angle_deg = principal_angle_degrees(lam, ker_basis_t)
        max_abs_residual, loss_value = tensor_residual_stats(terms_t)
        rows.append(
            {
                'base_label': base_label,
                'candidate': candidate_name,
                't_value': t_value,
                'R': alpha.shape[0],
                'annihilator_projection_residual': annihilator_projection_residual,
                'lambda_H_max_abs': float(np.max(np.abs(lambda_h))),
                'lambda_to_ker_angle_deg': angle_deg,
                'rank_H_numeric': matrix_rank_numeric(h_matrix),
                'restricted_rank_numeric': matrix_rank_numeric(restricted_t),
                'functional_defect_dim_numeric': ker_basis_t.shape[1] - matrix_rank_numeric(restricted_t),
                'tensor_max_abs_residual': max_abs_residual,
                'tensor_loss': loss_value,
            }
        )

    thresholds = [1.0, 5.0, 15.0, 30.0]
    threshold_rows: dict[str, dict[str, object] | None] = {}
    for threshold in thresholds:
        admissible = [row for row in rows if float(row['lambda_to_ker_angle_deg']) <= threshold]
        threshold_rows[str(threshold)] = min(admissible, key=lambda row: (float(row['tensor_max_abs_residual']), float(row['tensor_loss']))) if admissible else None

    summary = {
        'base_label': base_label,
        'candidate': candidate_name,
        'min_angle_deg': min(float(row['lambda_to_ker_angle_deg']) for row in rows),
        'residual_at_min_angle': min(rows, key=lambda row: float(row['lambda_to_ker_angle_deg']))['tensor_max_abs_residual'],
        'residual_at_t1': next(row['tensor_max_abs_residual'] for row in rows if float(row['t_value']) == 1.0),
        'best_rows_by_angle_threshold': {
            key: value for key, value in threshold_rows.items()
        },
    }
    return rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    base_cases = [
        (f'alphatensor_rank23_{orientation}',) + stacked_alpha_beta(alpha_terms),
        ('standard_rank27',) + stacked_alpha_beta(standard_terms),
    ]

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for base_label, alpha, beta in base_cases:
        gamma = solve_gamma_least_squares(alpha, beta)
        for candidate_name, lam in choose_lambda_candidates(alpha, beta, gamma, count=3):
            path_rows, summary = evaluate_path(base_label, alpha, beta, candidate_name, lam)
            rows.extend(path_rows)
            summaries.append(summary)

    write_csv(
        OUT_DIR / 'kernel_linked_annihilator_homotopy.csv',
        rows,
        [
            'base_label', 'candidate', 't_value', 'R', 'annihilator_projection_residual', 'lambda_H_max_abs',
            'lambda_to_ker_angle_deg', 'rank_H_numeric', 'restricted_rank_numeric', 'functional_defect_dim_numeric',
            'tensor_max_abs_residual', 'tensor_loss',
        ],
    )
    write_json(
        OUT_DIR / 'kernel_linked_annihilator_homotopy.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'summaries': summaries,
        },
    )

    print('Phase 22 complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']} / {summary['candidate']}: min angle={summary['min_angle_deg']:.6g}, "
            f"residual@min-angle={summary['residual_at_min_angle']:.6g}, residual@t=1={summary['residual_at_t1']:.6g}"
        )


if __name__ == '__main__':
    main()