from __future__ import annotations

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
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
    load_public_terms,
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


def project_to_annihilator_family(alpha: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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
    t = ones - correction
    projection_residual = float(np.max(np.abs(a_matrix @ t)))

    alpha_scaled = alpha.copy().reshape(rank_value, 3, 3)
    for shared_idx in range(3):
        scale = t[shared_idx * rank_value:(shared_idx + 1) * rank_value]
        alpha_scaled[:, :, shared_idx] *= scale[:, None]
    return alpha_scaled.reshape(rank_value, 9), beta.copy(), projection_residual


def evaluate_family(base_label: str, base_alpha: np.ndarray, base_beta: np.ndarray, trial_count: int = 80) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(abs(hash(base_label)) % (2**32))
    rows: list[dict[str, object]] = []
    rank_value = base_alpha.shape[0]
    for trial_index in range(1, trial_count + 1):
        lam = rng.standard_normal(rank_value)
        lam /= max(np.linalg.norm(lam), 1e-12)
        alpha_proj, beta_proj, ann_res = project_to_annihilator_family(base_alpha, base_beta, lam)
        gamma = solve_gamma_least_squares(alpha_proj, beta_proj)
        terms = terms_from_stacked_factors(alpha_proj, beta_proj, gamma, prefix=f'ann_{trial_index:03d}_')
        _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
        gamma_num = gamma.T
        ker_basis = nullspace_basis_numeric(gamma_num)
        restricted = ker_basis.T @ h_matrix
        lambda_kernel_dist = float(np.min(np.linalg.norm(ker_basis - lam[:, None], axis=0))) if ker_basis.size else float('inf')
        max_abs_residual, loss_value = tensor_residual_stats(terms)
        rows.append(
            {
                'base_label': base_label,
                'trial_index': trial_index,
                'R': rank_value,
                'annihilator_projection_residual': ann_res,
                'rank_H_numeric': matrix_rank_numeric(h_matrix),
                'restricted_rank_numeric': matrix_rank_numeric(restricted),
                'functional_defect_dim_numeric': ker_basis.shape[1] - matrix_rank_numeric(restricted),
                'lambda_kernel_distance': lambda_kernel_dist,
                'tensor_max_abs_residual': max_abs_residual,
                'tensor_loss': loss_value,
            }
        )

    best_row = min(rows, key=lambda row: (float(row['tensor_max_abs_residual']), float(row['tensor_loss'])))
    summary = {
        'base_label': base_label,
        'trial_count': len(rows),
        'best_tensor_max_abs_residual': best_row['tensor_max_abs_residual'],
        'best_tensor_loss': best_row['tensor_loss'],
        'best_rank_H_numeric': best_row['rank_H_numeric'],
        'best_functional_defect_dim_numeric': best_row['functional_defect_dim_numeric'],
        'best_lambda_kernel_distance': best_row['lambda_kernel_distance'],
        'best_trial_index': best_row['trial_index'],
    }
    return rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    alpha_alpha, alpha_beta = stacked_alpha_beta(alpha_terms)
    standard_alpha, standard_beta = stacked_alpha_beta(standard_terms)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for base_label, base_alpha, base_beta in [
        (f'alphatensor_rank23_{orientation}', alpha_alpha, alpha_beta),
        ('standard_rank27', standard_alpha, standard_beta),
    ]:
        family_rows, summary = evaluate_family(base_label, base_alpha, base_beta, trial_count=80)
        rows.extend(family_rows)
        summaries.append(summary)

    write_csv(
        OUT_DIR / 'weighted_annihilator_family_trials.csv',
        rows,
        [
            'base_label', 'trial_index', 'R', 'annihilator_projection_residual', 'rank_H_numeric',
            'restricted_rank_numeric', 'functional_defect_dim_numeric', 'lambda_kernel_distance',
            'tensor_max_abs_residual', 'tensor_loss',
        ],
    )
    write_json(
        OUT_DIR / 'weighted_annihilator_family_summary.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'summaries': summaries,
        },
    )
    print('Phase 21B complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']}: best max-abs residual={summary['best_tensor_max_abs_residual']:.6g}, "
            f"best functional defect={summary['best_functional_defect_dim_numeric']}, "
            f"best lambda-kernel distance={summary['best_lambda_kernel_distance']:.6g}"
        )


if __name__ == '__main__':
    main()