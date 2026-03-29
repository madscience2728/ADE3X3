from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (  # noqa: E402
    stacked_factors_from_terms,
    load_public_terms,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    gamma_matrix,
)


RANK_TOL = 1e-10


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def all_integer_terms(terms: list[Term]) -> bool:
    for term in terms:
        for factor in (term.alpha, term.beta, term.gamma):
            as_float = np.asarray(factor, dtype=np.float64)
            if not np.allclose(as_float, np.rint(as_float)):
                return False
    return True


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


def gamma_matrix_numeric(terms: list[Term]) -> np.ndarray:
    _, _, gamma = stacked_factors_from_terms(terms)
    return gamma.T


def live_product_matrices_numeric(terms: list[Term]) -> list[np.ndarray]:
    alpha, beta, _ = stacked_factors_from_terms(terms)
    live_mats: list[np.ndarray] = []
    for shared_idx in range(3):
        columns: list[np.ndarray] = []
        for row_idx in range(3):
            for col_idx in range(3):
                columns.append(alpha[:, 3 * row_idx + shared_idx] * beta[:, 3 * shared_idx + col_idx])
        live_mats.append(np.column_stack(columns))
    return live_mats


def live_product_matrices_exact(terms: list[Term]) -> list[sp.Matrix]:
    mats: list[sp.Matrix] = []
    for shared_idx in range(3):
        rows: list[list[int]] = []
        for term in terms:
            alpha = np.asarray(term.alpha, dtype=np.int64)
            beta = np.asarray(term.beta, dtype=np.int64)
            rows.append([int(alpha[row_idx, shared_idx] * beta[shared_idx, col_idx]) for row_idx in range(3) for col_idx in range(3)])
        mats.append(sp.Matrix(rows))
    return mats


def max_abs_numeric(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def analyze_known_case(label: str, terms: list[Term]) -> dict[str, object]:
    gamma_num = gamma_matrix_numeric(terms)
    live_num = live_product_matrices_numeric(terms)
    h_num = np.hstack([live_num[0] - live_num[1], live_num[1] - live_num[2]])
    ker_basis = nullspace_basis_numeric(gamma_num)
    ker_dim = ker_basis.shape[1]
    kernel_coords = ker_basis.T @ h_num
    row: dict[str, object] = {
        'label': label,
        'R': len(terms),
        'rank_gamma_numeric': matrix_rank_numeric(gamma_num),
        'ker_gamma_dim_numeric': ker_dim,
        'rank_H_numeric': matrix_rank_numeric(h_num),
        'rank_kernel_coords_numeric': matrix_rank_numeric(kernel_coords),
        'kernel_saturation_numeric': matrix_rank_numeric(h_num) == ker_dim,
        'channel0_identity_max_abs': max_abs_numeric(gamma_num @ live_num[0] - np.eye(9)),
        'channel1_identity_max_abs': max_abs_numeric(gamma_num @ live_num[1] - np.eye(9)),
        'channel2_identity_max_abs': max_abs_numeric(gamma_num @ live_num[2] - np.eye(9)),
        'H_kernel_residual_max_abs': max_abs_numeric(gamma_num @ h_num),
    }

    if all_integer_terms(terms):
        gamma_exact = gamma_matrix(terms, 3)
        live_exact = live_product_matrices_exact(terms)
        h_exact = sp.Matrix.hstack(live_exact[0] - live_exact[1], live_exact[1] - live_exact[2])
        row.update(
            {
                'rank_gamma_exact': int(gamma_exact.rank()),
                'ker_gamma_dim_exact': len(terms) - int(gamma_exact.rank()),
                'rank_H_exact': int(h_exact.rank()),
                'kernel_saturation_exact': int(h_exact.rank()) == (len(terms) - int(gamma_exact.rank())),
                'channel0_identity_exact': (gamma_exact * live_exact[0]) == sp.eye(9),
                'channel1_identity_exact': (gamma_exact * live_exact[1]) == sp.eye(9),
                'channel2_identity_exact': (gamma_exact * live_exact[2]) == sp.eye(9),
                'H_kernel_exact': (gamma_exact * h_exact) == sp.zeros(9, h_exact.cols),
            }
        )

    return row


def synthetic_trial_rows(label: str, gamma_num: np.ndarray, reference_live: list[np.ndarray], trial_count: int = 250) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(abs(hash(label)) % (2**32))
    ker_basis = nullspace_basis_numeric(gamma_num)
    ker_dim = ker_basis.shape[1]
    x0 = reference_live[0]
    rows: list[dict[str, object]] = []
    saturation_count = 0

    for trial_idx in range(trial_count):
        k0 = ker_basis @ rng.standard_normal((ker_dim, 9))
        k1 = ker_basis @ rng.standard_normal((ker_dim, 9))
        k2 = ker_basis @ rng.standard_normal((ker_dim, 9))
        p0 = x0 + k0
        p1 = x0 + k1
        p2 = x0 + k2
        h_num = np.hstack([p0 - p1, p1 - p2])
        rank_h = matrix_rank_numeric(h_num)
        saturated = rank_h == ker_dim
        saturation_count += int(saturated)
        rows.append(
            {
                'family': label,
                'trial_index': trial_idx + 1,
                'ker_gamma_dim_numeric': ker_dim,
                'rank_H_numeric': rank_h,
                'kernel_saturation_numeric': saturated,
                'channel0_identity_max_abs': max_abs_numeric(gamma_num @ p0 - np.eye(9)),
                'channel1_identity_max_abs': max_abs_numeric(gamma_num @ p1 - np.eye(9)),
                'channel2_identity_max_abs': max_abs_numeric(gamma_num @ p2 - np.eye(9)),
                'H_kernel_residual_max_abs': max_abs_numeric(gamma_num @ h_num),
            }
        )

    summary = {
        'family': label,
        'trial_count': trial_count,
        'ker_gamma_dim_numeric': ker_dim,
        'saturation_count': saturation_count,
        'saturation_rate': saturation_count / trial_count,
        'min_rank_H_numeric': min(int(row['rank_H_numeric']) for row in rows),
        'max_rank_H_numeric': max(int(row['rank_H_numeric']) for row in rows),
    }
    return rows, summary


def forced_defect_rows(label: str, gamma_num: np.ndarray, reference_live: list[np.ndarray]) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260329 + len(label))
    ker_basis = nullspace_basis_numeric(gamma_num)
    ker_dim = ker_basis.shape[1]
    x0 = reference_live[0]
    base = ker_basis @ rng.standard_normal((ker_dim, 9))
    alt = ker_basis @ rng.standard_normal((ker_dim, 9))
    cases = [
        ('all_equal', base, base, base),
        ('k0_equals_k1', base, base, alt),
        ('all_collinear', base, 2.0 * base, -1.0 * base),
    ]
    rows: list[dict[str, object]] = []
    for case_name, k0, k1, k2 in cases:
        p0 = x0 + k0
        p1 = x0 + k1
        p2 = x0 + k2
        h_num = np.hstack([p0 - p1, p1 - p2])
        rows.append(
            {
                'family': label,
                'case': case_name,
                'ker_gamma_dim_numeric': ker_dim,
                'rank_H_numeric': matrix_rank_numeric(h_num),
                'deficiency_numeric': ker_dim - matrix_rank_numeric(h_num),
                'channel0_identity_max_abs': max_abs_numeric(gamma_num @ p0 - np.eye(9)),
                'channel1_identity_max_abs': max_abs_numeric(gamma_num @ p1 - np.eye(9)),
                'channel2_identity_max_abs': max_abs_numeric(gamma_num @ p2 - np.eye(9)),
                'H_kernel_residual_max_abs': max_abs_numeric(gamma_num @ h_num),
            }
        )
    return rows


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()

    known_rows = [
        analyze_known_case(f'alphatensor_rank23_{orientation}', alpha_terms),
        analyze_known_case('standard_rank27', standard_terms),
    ]
    write_csv(
        OUT_DIR / 'known_case_kernel_saturation.csv',
        known_rows,
        [
            'label', 'R', 'rank_gamma_numeric', 'ker_gamma_dim_numeric', 'rank_H_numeric', 'rank_kernel_coords_numeric',
            'kernel_saturation_numeric', 'channel0_identity_max_abs', 'channel1_identity_max_abs', 'channel2_identity_max_abs',
            'H_kernel_residual_max_abs', 'rank_gamma_exact', 'ker_gamma_dim_exact', 'rank_H_exact', 'kernel_saturation_exact',
            'channel0_identity_exact', 'channel1_identity_exact', 'channel2_identity_exact', 'H_kernel_exact',
        ],
    )

    alpha_gamma = gamma_matrix_numeric(alpha_terms)
    alpha_live = live_product_matrices_numeric(alpha_terms)
    standard_gamma = gamma_matrix_numeric(standard_terms)
    standard_live = live_product_matrices_numeric(standard_terms)

    alpha_synth_rows, alpha_synth_summary = synthetic_trial_rows('alphatensor_gamma', alpha_gamma, alpha_live, trial_count=250)
    standard_synth_rows, standard_synth_summary = synthetic_trial_rows('standard_gamma', standard_gamma, standard_live, trial_count=250)
    write_csv(
        OUT_DIR / 'synthetic_right_inverse_trials.csv',
        alpha_synth_rows + standard_synth_rows,
        [
            'family', 'trial_index', 'ker_gamma_dim_numeric', 'rank_H_numeric', 'kernel_saturation_numeric',
            'channel0_identity_max_abs', 'channel1_identity_max_abs', 'channel2_identity_max_abs', 'H_kernel_residual_max_abs',
        ],
    )

    defect_rows = forced_defect_rows('alphatensor_gamma', alpha_gamma, alpha_live) + forced_defect_rows('standard_gamma', standard_gamma, standard_live)
    write_csv(
        OUT_DIR / 'forced_defect_examples.csv',
        defect_rows,
        [
            'family', 'case', 'ker_gamma_dim_numeric', 'rank_H_numeric', 'deficiency_numeric',
            'channel0_identity_max_abs', 'channel1_identity_max_abs', 'channel2_identity_max_abs', 'H_kernel_residual_max_abs',
        ],
    )

    payload = {
        'reference_decompositions': {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rows': known_rows,
        },
        'synthetic_right_inverse_trials': {
            'alphatensor_gamma': alpha_synth_summary,
            'standard_gamma': standard_synth_summary,
        },
        'forced_defect_examples': defect_rows,
    }
    write_json(OUT_DIR / 'kernel_saturation_summary.json', payload)

    print('Phase 18 complete.')
    print(f"  AlphaTensor kernel saturation: {known_rows[0]['kernel_saturation_numeric']}")
    print(f"  Standard kernel saturation: {known_rows[1]['kernel_saturation_numeric']}")
    print(f"  Synthetic AlphaTensor-gamma saturation rate: {alpha_synth_summary['saturation_rate']:.6f}")
    print(f"  Synthetic standard-gamma saturation rate: {standard_synth_summary['saturation_rate']:.6f}")


if __name__ == '__main__':
    main()