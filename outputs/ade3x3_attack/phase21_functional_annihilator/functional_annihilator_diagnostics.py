from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import build_mode_matrices_numeric, load_public_terms, write_csv, write_json  # noqa: E402
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


def stacked_factors(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta, gamma


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def analyze_case(label: str, terms: list[Term]) -> dict[str, object]:
    _, _, gamma = stacked_factors(terms)
    gamma_num = gamma.T
    _, _, _, _, h_matrix, _ = build_mode_matrices_numeric(terms)
    ker_basis = nullspace_basis_numeric(gamma_num)
    restricted = ker_basis.T @ h_matrix
    singular_values = np.linalg.svd(restricted, compute_uv=False)
    row = {
        'label': label,
        'R': len(terms),
        'rank_gamma_numeric': matrix_rank_numeric(gamma_num),
        'ker_gamma_dim_numeric': ker_basis.shape[1],
        'rank_H_numeric': matrix_rank_numeric(h_matrix),
        'restricted_rank_numeric': matrix_rank_numeric(restricted),
        'restricted_min_singular_value': float(singular_values[-1]) if singular_values.size else 0.0,
        'restricted_max_singular_value': float(singular_values[0]) if singular_values.size else 0.0,
        'functional_defect_dim_numeric': ker_basis.shape[1] - matrix_rank_numeric(restricted),
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
        OUT_DIR / 'functional_annihilator_diagnostics.csv',
        rows,
        [
            'label', 'R', 'rank_gamma_numeric', 'ker_gamma_dim_numeric', 'rank_H_numeric',
            'restricted_rank_numeric', 'restricted_min_singular_value', 'restricted_max_singular_value',
            'functional_defect_dim_numeric',
        ],
    )
    write_json(
        OUT_DIR / 'functional_annihilator_diagnostics.json',
        {
            'alphatensor_orientation': orientation,
            'alphatensor_reconstruction_max_abs': residual,
            'rows': rows,
        },
    )
    print('Phase 21A complete.')
    for row in rows:
        print(f"  {row['label']}: defect_dim={row['functional_defect_dim_numeric']}, min_sv={row['restricted_min_singular_value']:.6g}")


if __name__ == '__main__':
    main()