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
    load_public_terms,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term  # noqa: E402


RANK_TOL = 1e-10


def stacked_alpha_beta(terms: list[Term]) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta


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


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def max_abs_numeric(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def project_family(alpha: np.ndarray, beta: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    alpha_proj = alpha.copy().reshape(-1, 3, 3)
    beta_proj = beta.copy().reshape(-1, 3, 3)

    if family == 'pair_equal_01':
        avg_a = 0.5 * (alpha_proj[:, :, 0] + alpha_proj[:, :, 1])
        avg_b = 0.5 * (beta_proj[:, 0, :] + beta_proj[:, 1, :])
        alpha_proj[:, :, 0] = avg_a
        alpha_proj[:, :, 1] = avg_a
        beta_proj[:, 0, :] = avg_b
        beta_proj[:, 1, :] = avg_b
    elif family == 'pair_equal_12':
        avg_a = 0.5 * (alpha_proj[:, :, 1] + alpha_proj[:, :, 2])
        avg_b = 0.5 * (beta_proj[:, 1, :] + beta_proj[:, 2, :])
        alpha_proj[:, :, 1] = avg_a
        alpha_proj[:, :, 2] = avg_a
        beta_proj[:, 1, :] = avg_b
        beta_proj[:, 2, :] = avg_b
    elif family == 'all_equal':
        mean_a = np.mean(alpha_proj, axis=2)
        mean_b = np.mean(beta_proj, axis=1)
        for idx in range(3):
            alpha_proj[:, :, idx] = mean_a
            beta_proj[:, idx, :] = mean_b
    elif family == 'collinear':
        for term_idx in range(alpha_proj.shape[0]):
            u_a, s_a, vh_a = np.linalg.svd(alpha_proj[term_idx], full_matrices=False)
            alpha_proj[term_idx] = (s_a[0] * np.outer(u_a[:, 0], vh_a[0, :]))
            u_b, s_b, vh_b = np.linalg.svd(beta_proj[term_idx], full_matrices=False)
            beta_proj[term_idx] = (s_b[0] * np.outer(u_b[:, 0], vh_b[0, :]))
    else:
        raise ValueError(f'Unknown family: {family}')

    return alpha_proj.reshape(alpha.shape[0], 9), beta_proj.reshape(beta.shape[0], 9)


def evaluate_family(base_label: str, family: str, base_alpha: np.ndarray, base_beta: np.ndarray, trial_count: int = 30) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(abs(hash((base_label, family))) % (2**32))
    rows: list[dict[str, object]] = []

    for trial_idx in range(trial_count):
        noise_scale = 0.0 if trial_idx == 0 else 0.02
        alpha_noisy = base_alpha + noise_scale * rng.standard_normal(base_alpha.shape)
        beta_noisy = base_beta + noise_scale * rng.standard_normal(base_beta.shape)
        alpha_proj, beta_proj = project_family(alpha_noisy, beta_noisy, family)
        gamma = solve_gamma_least_squares(alpha_proj, beta_proj)
        terms = terms_from_stacked_factors(alpha_proj, beta_proj, gamma, prefix=f'{family}_{trial_idx + 1:02d}_')
        sigma, eta1, eta2, delta, h_matrix, _ = build_mode_matrices_numeric(terms)
        gamma_num = gamma.T
        p0 = eta1 * 0.0
        p1 = eta1 * 0.0
        p2 = eta1 * 0.0
        sigma_col = 0
        for row_idx in range(3):
            for col_idx in range(3):
                p0[:, sigma_col] = alpha_proj[:, 3 * row_idx + 0] * beta_proj[:, 0 * 3 + col_idx]
                p1[:, sigma_col] = alpha_proj[:, 3 * row_idx + 1] * beta_proj[:, 1 * 3 + col_idx]
                p2[:, sigma_col] = alpha_proj[:, 3 * row_idx + 2] * beta_proj[:, 2 * 3 + col_idx]
                sigma_col += 1
        max_abs_residual, loss_value = tensor_residual_stats(terms)
        rows.append(
            {
                'base_label': base_label,
                'family': family,
                'trial_index': trial_idx + 1,
                'noise_scale': noise_scale,
                'R': alpha_proj.shape[0],
                'rank_H_numeric': matrix_rank_numeric(h_matrix),
                'rank_HDelta_numeric': matrix_rank_numeric(np.hstack([h_matrix, delta])),
                'channel0_identity_max_abs': max_abs_numeric(gamma_num @ p0 - np.eye(9)),
                'channel1_identity_max_abs': max_abs_numeric(gamma_num @ p1 - np.eye(9)),
                'channel2_identity_max_abs': max_abs_numeric(gamma_num @ p2 - np.eye(9)),
                'pair01_live_gap_max_abs': max_abs_numeric(p0 - p1),
                'pair12_live_gap_max_abs': max_abs_numeric(p1 - p2),
                'pair02_live_gap_max_abs': max_abs_numeric(p0 - p2),
                'tensor_max_abs_residual': max_abs_residual,
                'tensor_loss': loss_value,
            }
        )

    best_row = min(rows, key=lambda row: (float(row['tensor_max_abs_residual']), float(row['tensor_loss'])))
    summary = {
        'base_label': base_label,
        'family': family,
        'trial_count': len(rows),
        'best_tensor_max_abs_residual': best_row['tensor_max_abs_residual'],
        'best_tensor_loss': best_row['tensor_loss'],
        'best_rank_H_numeric': best_row['rank_H_numeric'],
        'best_rank_HDelta_numeric': best_row['rank_HDelta_numeric'],
        'best_channel_identity_max_abs': max(
            float(best_row['channel0_identity_max_abs']),
            float(best_row['channel1_identity_max_abs']),
            float(best_row['channel2_identity_max_abs']),
        ),
        'best_trial_index': best_row['trial_index'],
    }
    return rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    alpha_alpha, alpha_beta = stacked_alpha_beta(alpha_terms)
    standard_alpha, standard_beta = stacked_alpha_beta(standard_terms)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    families = ['pair_equal_01', 'pair_equal_12', 'all_equal', 'collinear']
    for base_label, base_alpha, base_beta in [
        (f'alphatensor_rank23_{orientation}', alpha_alpha, alpha_beta),
        ('standard_rank27', standard_alpha, standard_beta),
    ]:
        for family in families:
            rows, summary = evaluate_family(base_label, family, base_alpha, base_beta, trial_count=30)
            all_rows.extend(rows)
            summaries.append(summary)

    write_csv(
        OUT_DIR / 'constrained_defect_family_trials.csv',
        all_rows,
        [
            'base_label', 'family', 'trial_index', 'noise_scale', 'R', 'rank_H_numeric', 'rank_HDelta_numeric',
            'channel0_identity_max_abs', 'channel1_identity_max_abs', 'channel2_identity_max_abs',
            'pair01_live_gap_max_abs', 'pair12_live_gap_max_abs', 'pair02_live_gap_max_abs',
            'tensor_max_abs_residual', 'tensor_loss',
        ],
    )
    write_json(
        OUT_DIR / 'constrained_defect_family_summary.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'summaries': summaries,
        },
    )

    print('Phase 19B complete.')
    for summary in summaries:
        print(
            f"  {summary['base_label']} / {summary['family']}: "
            f"best max-abs residual={summary['best_tensor_max_abs_residual']:.6g}, "
            f"best rank(H)={summary['best_rank_H_numeric']}"
        )


if __name__ == '__main__':
    main()