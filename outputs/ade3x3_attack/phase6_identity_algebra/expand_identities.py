from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import (  # noqa: E402
    eta_coordinate_labels,
    eta_identity_to_q_matrix,
    factor_coordinate_labels,
    load_public_terms,
    phase5_identity_basis,
    read_phase5_null_vectors,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    build_mode_matrices,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def integer_factor_vector(matrix: np.ndarray) -> sp.Matrix:
    return sp.Matrix([int(value) for value in np.rint(np.asarray(matrix, dtype=np.float64)).reshape(-1)])


def random_nontrivial_witness(q_matrix: sp.Matrix, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    while True:
        a_entries = rng.integers(-2, 3, size=9)
        b_entries = rng.integers(-2, 3, size=9)
        if not np.any(a_entries) or not np.any(b_entries):
            continue
        a_vec = sp.Matrix([int(value) for value in a_entries])
        b_vec = sp.Matrix([int(value) for value in b_entries])
        value = sp.simplify((a_vec.T * q_matrix * b_vec)[0])
        if value != 0:
            return {
                'a': [int(value) for value in a_entries.tolist()],
                'b': [int(value) for value in b_entries.tolist()],
                'value': sp.sstr(value),
            }


def main() -> None:
    eta_labels = eta_coordinate_labels()
    alpha_labels = factor_coordinate_labels('a')
    beta_labels = factor_coordinate_labels('b')
    null_rows = read_phase5_null_vectors()
    _, identity_vectors = phase5_identity_basis()
    terms, _, _ = load_public_terms()
    _, eta1_exact, eta2_exact, _, _ = build_mode_matrices(terms, 3)

    q_rows: list[dict[str, object]] = []
    q_payload: list[dict[str, object]] = []
    identity_payload: list[dict[str, object]] = []

    for basis_idx, (row, identity_vector) in enumerate(zip(null_rows, identity_vectors, strict=True), start=1):
        q_matrix = eta_identity_to_q_matrix(identity_vector)
        full_matrix = [[sp.sstr(q_matrix[row_idx, col_idx]) for col_idx in range(9)] for row_idx in range(9)]
        for row_idx, alpha_label in enumerate(alpha_labels):
            for col_idx, beta_label in enumerate(beta_labels):
                entry = sp.simplify(q_matrix[row_idx, col_idx])
                if entry == 0:
                    continue
                q_rows.append(
                    {
                        'identity_id': basis_idx,
                        'formula': row['formula'],
                        'a_coordinate': alpha_label,
                        'b_coordinate': beta_label,
                        'coefficient': sp.sstr(entry),
                    }
                )

        term_evaluations: list[dict[str, str]] = []
        all_terms_zero = True
        for term_idx, term in enumerate(terms):
            a_vec = integer_factor_vector(term.alpha)
            b_vec = integer_factor_vector(term.beta)
            eta_eval = sp.Integer(0)
            for label_idx, label in enumerate(eta_labels[:9]):
                eta_eval += sp.nsimplify(identity_vector[label_idx, 0]) * eta1_exact[term_idx, label_idx]
            for label_idx in range(9):
                eta_eval += sp.nsimplify(identity_vector[9 + label_idx, 0]) * eta2_exact[term_idx, label_idx]
            bilinear_eval = sp.simplify((a_vec.T * q_matrix * b_vec)[0])
            if bilinear_eval != 0 or eta_eval != 0:
                all_terms_zero = False
            term_evaluations.append(
                {
                    'term_id': term.term_id,
                    'eta_evaluation': sp.sstr(eta_eval),
                    'bilinear_evaluation': sp.sstr(bilinear_eval),
                }
            )

        witness = random_nontrivial_witness(q_matrix, seed=1000 + basis_idx)
        identity_payload.append(
            {
                'identity_id': basis_idx,
                'support_size': int(row['support_size']),
                'formula': row['formula'],
                'eta_coefficients': {label: int(row[label]) for label in eta_labels},
                'q_matrix': full_matrix,
                'all_public_terms_zero': all_terms_zero,
                'term_evaluations': term_evaluations,
                'nontrivial_witness': witness,
            }
        )
        q_payload.append(
            {
                'identity_id': basis_idx,
                'formula': row['formula'],
                'q_matrix': full_matrix,
            }
        )

    write_csv(
        OUTPUT_DIR / 'Q_matrices.csv',
        q_rows,
        ['identity_id', 'formula', 'a_coordinate', 'b_coordinate', 'coefficient'],
    )
    write_json(
        OUTPUT_DIR / 'identity_expansion_summary.json',
        {
            'eta_coordinate_order': eta_labels,
            'alpha_coordinate_order': alpha_labels,
            'beta_coordinate_order': beta_labels,
            'identities': identity_payload,
            'q_matrices': q_payload,
        },
    )


if __name__ == '__main__':
    main()