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

from outputs.ade3x3_attack.attack_common import load_public_terms, write_csv, write_json  # noqa: E402
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


def stacked_factors(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta, gamma


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


def live_product_matrices_numeric(terms: list[Term]) -> list[np.ndarray]:
    alpha, beta, _ = stacked_factors(terms)
    live_mats: list[np.ndarray] = []
    for shared_idx in range(3):
        columns: list[np.ndarray] = []
        for row_idx in range(3):
            for col_idx in range(3):
                columns.append(alpha[:, 3 * row_idx + shared_idx] * beta[:, 3 * shared_idx + col_idx])
        live_mats.append(np.column_stack(columns))
    return live_mats


def live_product_matrices_exact(terms: list[Term]) -> list[sp.Matrix]:
    matrices: list[sp.Matrix] = []
    for shared_idx in range(3):
        rows: list[list[int]] = []
        for term in terms:
            alpha = np.asarray(term.alpha, dtype=np.int64)
            beta = np.asarray(term.beta, dtype=np.int64)
            rows.append([int(alpha[row_idx, shared_idx] * beta[shared_idx, col_idx]) for row_idx in range(3) for col_idx in range(3)])
        matrices.append(sp.Matrix(rows))
    return matrices


def max_abs_numeric(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def centered_lifts_numeric(live_mats: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    x0 = (live_mats[0] + live_mats[1] + live_mats[2]) / 3.0
    return x0, [live - x0 for live in live_mats]


def analyze_case(label: str, terms: list[Term]) -> dict[str, object]:
    gamma_num = np.asarray(stacked_factors(terms)[2], dtype=np.float64).T
    live_num = live_product_matrices_numeric(terms)
    x0_num, lifts_num = centered_lifts_numeric(live_num)
    h_num = np.hstack([live_num[0] - live_num[1], live_num[1] - live_num[2]])
    ker_basis = nullspace_basis_numeric(gamma_num)
    ker_coords = ker_basis.T @ h_num
    defect_dim = ker_basis.shape[1] - matrix_rank_numeric(ker_coords)
    lift_vec_matrix = np.column_stack([lift.reshape(-1) for lift in lifts_num])

    row: dict[str, object] = {
        'label': label,
        'R': len(terms),
        'rank_gamma_numeric': matrix_rank_numeric(gamma_num),
        'ker_gamma_dim_numeric': ker_basis.shape[1],
        'rank_H_numeric': matrix_rank_numeric(h_num),
        'defect_dim_numeric': defect_dim,
        'center_identity_max_abs': max_abs_numeric(gamma_num @ x0_num - np.eye(9)),
        'lift0_kernel_residual_max_abs': max_abs_numeric(gamma_num @ lifts_num[0]),
        'lift1_kernel_residual_max_abs': max_abs_numeric(gamma_num @ lifts_num[1]),
        'lift2_kernel_residual_max_abs': max_abs_numeric(gamma_num @ lifts_num[2]),
        'sum_lifts_max_abs': max_abs_numeric(lifts_num[0] + lifts_num[1] + lifts_num[2]),
        'pair01_equal_max_abs': max_abs_numeric(live_num[0] - live_num[1]),
        'pair12_equal_max_abs': max_abs_numeric(live_num[1] - live_num[2]),
        'pair02_equal_max_abs': max_abs_numeric(live_num[0] - live_num[2]),
        'lift_span_rank_numeric': matrix_rank_numeric(lift_vec_matrix),
    }

    if all_integer_terms(terms):
        gamma_exact = gamma_matrix(terms, 3)
        live_exact = live_product_matrices_exact(terms)
        x0_exact = (live_exact[0] + live_exact[1] + live_exact[2]) / 3
        lifts_exact = [live - x0_exact for live in live_exact]
        h_exact = sp.Matrix.hstack(live_exact[0] - live_exact[1], live_exact[1] - live_exact[2])
        row.update(
            {
                'rank_gamma_exact': int(gamma_exact.rank()),
                'ker_gamma_dim_exact': len(terms) - int(gamma_exact.rank()),
                'rank_H_exact': int(h_exact.rank()),
                'defect_dim_exact': (len(terms) - int(gamma_exact.rank())) - int(h_exact.rank()),
                'center_identity_exact': (gamma_exact * x0_exact) == sp.eye(9),
                'lift0_kernel_exact': (gamma_exact * lifts_exact[0]) == sp.zeros(9, 9),
                'lift1_kernel_exact': (gamma_exact * lifts_exact[1]) == sp.zeros(9, 9),
                'lift2_kernel_exact': (gamma_exact * lifts_exact[2]) == sp.zeros(9, 9),
                'sum_lifts_exact': (lifts_exact[0] + lifts_exact[1] + lifts_exact[2]) == sp.zeros(len(terms), 9),
            }
        )

    return row


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    rows = [
        analyze_case(f'alphatensor_rank23_{orientation}', alpha_terms),
        analyze_case('standard_rank27', standard_terms),
    ]
    write_csv(
        OUT_DIR / 'factorized_defect_diagnostics.csv',
        rows,
        [
            'label', 'R', 'rank_gamma_numeric', 'ker_gamma_dim_numeric', 'rank_H_numeric', 'defect_dim_numeric',
            'center_identity_max_abs', 'lift0_kernel_residual_max_abs', 'lift1_kernel_residual_max_abs', 'lift2_kernel_residual_max_abs',
            'sum_lifts_max_abs', 'pair01_equal_max_abs', 'pair12_equal_max_abs', 'pair02_equal_max_abs', 'lift_span_rank_numeric',
            'rank_gamma_exact', 'ker_gamma_dim_exact', 'rank_H_exact', 'defect_dim_exact', 'center_identity_exact',
            'lift0_kernel_exact', 'lift1_kernel_exact', 'lift2_kernel_exact', 'sum_lifts_exact',
        ],
    )
    write_json(
        OUT_DIR / 'factorized_defect_diagnostics.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rows': rows,
        },
    )
    print('Phase 19A complete.')
    for row in rows:
        print(f"  {row['label']}: defect_dim={row['defect_dim_numeric']}, lift_span_rank={row['lift_span_rank_numeric']}")


if __name__ == '__main__':
    main()