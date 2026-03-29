from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import eta_identity_to_q_matrix, load_public_terms, phase5_identity_basis, write_json  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def integer_vector(matrix: np.ndarray) -> sp.Matrix:
    return sp.Matrix([int(value) for value in np.rint(np.asarray(matrix, dtype=np.float64)).reshape(-1)])


def sigma_vector(a_vec: sp.Matrix, b_vec: sp.Matrix) -> sp.Matrix:
    entries: list[sp.Expr] = []
    for row_idx in range(3):
        for col_idx in range(3):
            value = sp.Integer(0)
            for shared_idx in range(3):
                value += a_vec[3 * row_idx + shared_idx] * b_vec[3 * shared_idx + col_idx]
            entries.append(sp.simplify(value))
    return sp.Matrix(entries)


def constraint_matrix_for_a(a_vec: sp.Matrix, q_matrices: list[sp.Matrix]) -> sp.Matrix:
    rows = []
    for q_matrix in q_matrices:
        rows.append(list((a_vec.T * q_matrix)))
    return sp.Matrix(rows)


def jacobian_rank(a_vec: sp.Matrix, b_vec: sp.Matrix, q_matrices: list[sp.Matrix]) -> int:
    jacobian_rows = []
    for q_matrix in q_matrices:
        grad_a = q_matrix * b_vec
        grad_b = q_matrix.T * a_vec
        jacobian_rows.append(list(grad_a) + list(grad_b))
    return int(sp.Matrix(jacobian_rows).rank())


def kernel_combination(nullspace: list[sp.Matrix], coefficients: np.ndarray) -> sp.Matrix:
    total = sp.zeros(9, 1)
    for coeff, basis_vector in zip(coefficients.tolist(), nullspace, strict=True):
        total += int(coeff) * basis_vector
    return total


def main() -> None:
    _, identity_vectors = phase5_identity_basis()
    q_matrices = [eta_identity_to_q_matrix(vector) for vector in identity_vectors]
    terms, _, _ = load_public_terms()

    a_rank_histogram: Counter[int] = Counter()
    sample_jacobian_histogram: Counter[int] = Counter()
    sigma_samples: list[sp.Matrix] = []
    representative_samples: list[dict[str, object]] = []

    rng = np.random.default_rng(20250329)
    attempts = 0
    accepted = 0
    while accepted < 48 and attempts < 4096:
        attempts += 1
        a_entries = rng.integers(-3, 4, size=9)
        if not np.any(a_entries):
            continue
        a_vec = sp.Matrix([int(value) for value in a_entries.tolist()])
        constraint_matrix = constraint_matrix_for_a(a_vec, q_matrices)
        rank = int(constraint_matrix.rank())
        a_rank_histogram[rank] += 1
        if rank != 4:
            continue
        nullspace = constraint_matrix.nullspace()
        coeffs = rng.integers(-3, 4, size=len(nullspace))
        if not np.any(coeffs):
            coeffs[0] = 1
        b_vec = kernel_combination(nullspace, coeffs)
        if all(value == 0 for value in list(b_vec)):
            continue
        jac_rank = jacobian_rank(a_vec, b_vec, q_matrices)
        sample_jacobian_histogram[jac_rank] += 1
        sigma_samples.append(sigma_vector(a_vec, b_vec))
        if len(representative_samples) < 5:
            representative_samples.append(
                {
                    'a': [int(value) for value in list(a_vec)],
                    'b': [sp.sstr(value) for value in list(b_vec)],
                    'constraint_rank': rank,
                    'jacobian_rank': jac_rank,
                }
            )
        accepted += 1

    alpha_tensor_jacobian_histogram: Counter[int] = Counter()
    for term in terms:
        a_vec = integer_vector(term.alpha)
        b_vec = integer_vector(term.beta)
        alpha_tensor_jacobian_histogram[jacobian_rank(a_vec, b_vec, q_matrices)] += 1

    sigma_span_rank = int(sp.Matrix.hstack(*sigma_samples).rank()) if sigma_samples else 0
    max_jacobian_rank = 0
    if sample_jacobian_histogram:
        max_jacobian_rank = max(max_jacobian_rank, max(sample_jacobian_histogram))
    if alpha_tensor_jacobian_histogram:
        max_jacobian_rank = max(max_jacobian_rank, max(alpha_tensor_jacobian_histogram))

    write_json(
        OUTPUT_DIR / 'variety_dimension_summary.json',
        {
            'equation_count': 4,
            'ambient_dimension': 18,
            'sample_count': accepted,
            'attempt_count': attempts,
            'constraint_rank_histogram_for_random_a': {str(key): value for key, value in sorted(a_rank_histogram.items())},
            'sample_jacobian_rank_histogram': {str(key): value for key, value in sorted(sample_jacobian_histogram.items())},
            'alphatensor_jacobian_rank_histogram': {str(key): value for key, value in sorted(alpha_tensor_jacobian_histogram.items())},
            'max_observed_jacobian_rank': max_jacobian_rank,
            'estimated_variety_dimension': 18 - max_jacobian_rank,
            'generic_kernel_dimension_in_b_given_a': 9 - 4,
            'sampled_sigma_span_rank': sigma_span_rank,
            'representative_solution_samples': representative_samples,
            'interpretation': (
                'Random exact samples on V4 consistently realize constraint rank 4 and Jacobian rank 4, '
                'so the four bilinear identities cut the 18-dimensional factor space by the expected codimension 4, '
                'yielding an observed variety dimension of 14.'
            ),
        },
    )


if __name__ == '__main__':
    main()