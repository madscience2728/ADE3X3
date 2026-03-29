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
)


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


def stacked_factors(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta, gamma


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


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def max_abs_numeric(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def best_scalar_multiple(reference: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    denom = float(np.sum(reference * reference))
    if denom <= 0.0:
        return 0.0, max_abs_numeric(target)
    scalar = float(np.sum(reference * target) / denom)
    residual = target - scalar * reference
    return scalar, max_abs_numeric(residual)


def analyze_case(label: str, terms: list[Term]) -> dict[str, object]:
    live = live_product_matrices_numeric(terms)
    center = (live[0] + live[1] + live[2]) / 3.0
    lifts = [matrix - center for matrix in live]
    flattened = np.column_stack([lift.reshape(-1) for lift in lifts])
    span_rank = matrix_rank_numeric(flattened)

    lambda10, res10 = best_scalar_multiple(lifts[0], lifts[1])
    lambda20, res20 = best_scalar_multiple(lifts[0], lifts[2])
    lambda21, res21 = best_scalar_multiple(lifts[1], lifts[2])

    row = {
        'label': label,
        'R': len(terms),
        'lift_span_rank_numeric': span_rank,
        'sum_lifts_max_abs': max_abs_numeric(lifts[0] + lifts[1] + lifts[2]),
        'lambda_10_best': lambda10,
        'lambda_20_best': lambda20,
        'lambda_21_best': lambda21,
        'residual_10_max_abs': res10,
        'residual_20_max_abs': res20,
        'residual_21_max_abs': res21,
        'affine_line_defect_numeric': span_rank - 1,
    }
    return row


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()
    rows = [
        analyze_case(f'alphatensor_rank23_{orientation}', alpha_terms),
        analyze_case('standard_rank27', standard_terms),
    ]
    write_csv(
        OUT_DIR / 'affine_line_lift_analysis.csv',
        rows,
        [
            'label', 'R', 'lift_span_rank_numeric', 'sum_lifts_max_abs',
            'lambda_10_best', 'lambda_20_best', 'lambda_21_best',
            'residual_10_max_abs', 'residual_20_max_abs', 'residual_21_max_abs',
            'affine_line_defect_numeric',
        ],
    )
    write_json(
        OUT_DIR / 'affine_line_lift_analysis.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rows': rows,
        },
    )
    print('Phase 20A complete.')
    for row in rows:
        print(f"  {row['label']}: lift-span rank={row['lift_span_rank_numeric']}, best residuals=({row['residual_10_max_abs']:.6g}, {row['residual_20_max_abs']:.6g}, {row['residual_21_max_abs']:.6g})")


if __name__ == '__main__':
    main()