from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = OUT_DIR.parent / 'phase1_delta_containment'
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import delta_coordinate_labels, eta_coordinate_labels, primitive_integer_vector, write_csv, write_json  # noqa: E402


def load_projection_matrix() -> tuple[np.ndarray, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    with open(PHASE1_DIR / 'delta_projection_matrix.csv', 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    eta_labels = eta_coordinate_labels()
    matrix = np.zeros((len(eta_labels), len(rows)), dtype=np.int64)
    for col_idx, row in enumerate(rows):
        for eta_idx, label in enumerate(eta_labels):
            matrix[eta_idx, col_idx] = int(sp.sympify(row[label]))
    return matrix, rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix, raw_rows = load_projection_matrix()
    eta_labels = eta_coordinate_labels()
    delta_labels = delta_coordinate_labels()
    support_rows: list[dict] = []
    eta_incidence_rows: list[dict] = []

    for col_idx, row in enumerate(raw_rows):
        coeffs = matrix[:, col_idx]
        support_rows.append(
            {
                'delta_coordinate': row['delta_coordinate'],
                'support_size': int(np.count_nonzero(coeffs)),
                'support': row['support'],
                'plus_count': int(np.sum(coeffs == 1)),
                'minus_count': int(np.sum(coeffs == -1)),
            }
        )

    for eta_idx, label in enumerate(eta_labels):
        coeffs = matrix[eta_idx, :]
        eta_incidence_rows.append(
            {
                'eta_coordinate': label,
                'incident_delta_count': int(np.count_nonzero(coeffs)),
                'positive_incident_delta_count': int(np.sum(coeffs == 1)),
                'negative_incident_delta_count': int(np.sum(coeffs == -1)),
            }
        )

    m_sym = sp.Matrix(matrix)
    null_basis = [primitive_integer_vector(vector) for vector in m_sym.nullspace()]
    unfold_left = np.reshape(matrix, (2, 9 * 54))
    unfold_right = np.reshape(matrix, (18 * 3, 18))

    write_csv(OUT_DIR / 'projection_supports.csv', support_rows, ['delta_coordinate', 'support_size', 'support', 'plus_count', 'minus_count'])
    write_csv(
        OUT_DIR / 'projection_eta_incidence.csv',
        eta_incidence_rows,
        ['eta_coordinate', 'incident_delta_count', 'positive_incident_delta_count', 'negative_incident_delta_count'],
    )
    write_csv(
        OUT_DIR / 'projection_nullspace_basis.csv',
        [
            {
                'basis_id': basis_idx + 1,
                'support_size': sum(int(value != 0) for value in vector),
                **{delta_labels[col_idx]: vector[col_idx] for col_idx in range(len(delta_labels))},
            }
            for basis_idx, vector in enumerate(null_basis)
        ],
        ['basis_id', 'support_size', *delta_labels],
    )
    write_json(
        OUT_DIR / 'projection_analysis_summary.json',
        {
            'shape': list(matrix.shape),
            'rank_exact': int(m_sym.rank()),
            'nullity_exact': matrix.shape[1] - int(m_sym.rank()),
            'support_size_range': [int(min(np.count_nonzero(matrix[:, col_idx]) for col_idx in range(matrix.shape[1]))), int(max(np.count_nonzero(matrix[:, col_idx]) for col_idx in range(matrix.shape[1])))],
            'eta_incidence_range': [int(min(np.count_nonzero(matrix[row_idx, :]) for row_idx in range(matrix.shape[0]))), int(max(np.count_nonzero(matrix[row_idx, :]) for row_idx in range(matrix.shape[0])))],
            'coefficient_alphabet': sorted(int(value) for value in np.unique(matrix) if value != 0),
            'simple_block_unfold_ranks': {
                'eta_block_vs_rest': int(np.linalg.matrix_rank(unfold_left)),
                'reshaped_right_rank': int(np.linalg.matrix_rank(unfold_right)),
            },
        },
    )

    print('Phase 5a projection analysis complete')
    print(f'  rank(M) = {int(m_sym.rank())}')
    print(f'  nullity(M) = {matrix.shape[1] - int(m_sym.rank())}')


if __name__ == '__main__':
    main()