from __future__ import annotations

import csv
import json
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def eta_coordinate_labels() -> list[str]:
    labels: list[str] = []
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'eta1[{row_idx},{col_idx}]')
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'eta2[{row_idx},{col_idx}]')
    return labels


def delta_coordinate_labels() -> list[str]:
    labels: list[str] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    labels.append(f'delta[{row_idx},{sum_left},{sum_right},{col_idx}]')
    return labels


def monomial_labels() -> list[str]:
    labels: list[str] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                for col_idx in range(3):
                    labels.append(f'a[{row_idx},{sum_left}]*b[{sum_right},{col_idx}]')
    return labels


def monomial_index(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> int:
    return ((row_idx * 3 + sum_left) * 3 + sum_right) * 3 + col_idx


def eta_basis_vectors() -> tuple[list[str], list[sp.Matrix]]:
    labels = eta_coordinate_labels()
    vectors: list[sp.Matrix] = []
    for row_idx in range(3):
        for col_idx in range(3):
            vec = [sp.Integer(0)] * 81
            vec[monomial_index(row_idx, 0, 0, col_idx)] = 1
            vec[monomial_index(row_idx, 1, 1, col_idx)] = -1
            vectors.append(sp.Matrix(vec))
    for row_idx in range(3):
        for col_idx in range(3):
            vec = [sp.Integer(0)] * 81
            vec[monomial_index(row_idx, 1, 1, col_idx)] = 1
            vec[monomial_index(row_idx, 2, 2, col_idx)] = -1
            vectors.append(sp.Matrix(vec))
    return labels, vectors


def delta_basis_vectors() -> tuple[list[str], list[sp.Matrix]]:
    labels = delta_coordinate_labels()
    vectors: list[sp.Matrix] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    vec = [sp.Integer(0)] * 81
                    vec[monomial_index(row_idx, sum_left, sum_right, col_idx)] = 1
                    vectors.append(sp.Matrix(vec))
    return labels, vectors


def exact_constant_containment_rows() -> list[dict]:
    eta_labels, eta_vectors = eta_basis_vectors()
    delta_labels, delta_vectors = delta_basis_vectors()
    eta_matrix = sp.Matrix.hstack(*eta_vectors)
    rows: list[dict] = []
    for delta_label, delta_vector in zip(delta_labels, delta_vectors):
        try:
            eta_matrix.gauss_jordan_solve(delta_vector)
            solvable = True
        except ValueError:
            solvable = False
        rows.append({
            'delta_coordinate': delta_label,
            'constant_linear_combination_of_eta_exists': solvable,
        })
    return rows


def structural_dependency_rows() -> list[dict]:
    rows: list[dict] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    rows.append({
                        'delta_coordinate': f'delta[{row_idx},{sum_left},{sum_right},{col_idx}]',
                        'dead_product': f'a[{row_idx},{sum_left}]*b[{sum_right},{col_idx}]',
                        'matching_eta_coordinates': f'eta1[{row_idx},{col_idx}];eta2[{row_idx},{col_idx}]',
                        'shared_live_products': ';'.join(
                            f'a[{row_idx},{sum_idx}]*b[{sum_idx},{col_idx}]'
                            for sum_idx in range(3)
                        ),
                        'shares_full_monomial_with_eta': False,
                        'shares_a_row_with_eta': True,
                        'shares_output_fiber_with_eta': True,
                    })
    return rows


def jacobian_rank_rows() -> tuple[list[dict], dict[str, int]]:
    a_symbols = sp.symbols('a0:3_0:3')
    b_symbols = sp.symbols('b0:3_0:3')
    a_matrix = sp.Matrix(3, 3, a_symbols)
    b_matrix = sp.Matrix(3, 3, b_symbols)
    variables = list(a_symbols) + list(b_symbols)

    eta_expressions: list[sp.Expr] = []
    for row_idx in range(3):
        for col_idx in range(3):
            eta_expressions.append(sp.expand(a_matrix[row_idx, 0] * b_matrix[0, col_idx] - a_matrix[row_idx, 1] * b_matrix[1, col_idx]))
    for row_idx in range(3):
        for col_idx in range(3):
            eta_expressions.append(sp.expand(a_matrix[row_idx, 1] * b_matrix[1, col_idx] - a_matrix[row_idx, 2] * b_matrix[2, col_idx]))

    joint_expressions = list(eta_expressions)
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    joint_expressions.append(sp.expand(a_matrix[row_idx, sum_left] * b_matrix[sum_right, col_idx]))

    eta_jacobian = sp.Matrix(eta_expressions).jacobian(variables)
    joint_jacobian = sp.Matrix(joint_expressions).jacobian(variables)

    samples = [
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61],
        [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67],
        [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71],
        [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73],
        [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79],
    ]

    rows: list[dict] = []
    eta_ranks: list[int] = []
    joint_ranks: list[int] = []
    for sample_idx, values in enumerate(samples, start=1):
        substitution = dict(zip(variables, values))
        eta_rank = int(eta_jacobian.subs(substitution).rank())
        joint_rank = int(joint_jacobian.subs(substitution).rank())
        eta_ranks.append(eta_rank)
        joint_ranks.append(joint_rank)
        rows.append({
            'sample_id': sample_idx,
            'assignment': ','.join(str(value) for value in values),
            'eta_jacobian_rank': eta_rank,
            'joint_jacobian_rank': joint_rank,
        })

    summary = {
        'eta_rank_sample_min': min(eta_ranks),
        'eta_rank_sample_max': max(eta_ranks),
        'joint_rank_sample_min': min(joint_ranks),
        'joint_rank_sample_max': max(joint_ranks),
        'eta_upper_bound': 15,
        'joint_upper_bound': 17,
        'eta_generic_rank_if_sample_hits_upper_bound': max(eta_ranks),
        'joint_generic_rank_if_sample_hits_upper_bound': max(joint_ranks),
    }
    return rows, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    containment_rows = exact_constant_containment_rows()
    dependency_rows = structural_dependency_rows()
    jacobian_rows, jacobian_summary = jacobian_rank_rows()

    write_csv(
        OUT_DIR / 'single_term_constant_containment.csv',
        containment_rows,
        ['delta_coordinate', 'constant_linear_combination_of_eta_exists'],
    )
    write_csv(
        OUT_DIR / 'structural_dependency_map.csv',
        dependency_rows,
        [
            'delta_coordinate',
            'dead_product',
            'matching_eta_coordinates',
            'shared_live_products',
            'shares_full_monomial_with_eta',
            'shares_a_row_with_eta',
            'shares_output_fiber_with_eta',
        ],
    )
    write_csv(
        OUT_DIR / 'jacobian_rank_samples.csv',
        jacobian_rows,
        ['sample_id', 'assignment', 'eta_jacobian_rank', 'joint_jacobian_rank'],
    )
    write_json(
        OUT_DIR / 'symbolic_summary.json',
        {
            'single_term_constant_containment_holds': False,
            'delta_coordinates_tested': len(containment_rows),
            'solvable_delta_coordinates': [row['delta_coordinate'] for row in containment_rows if row['constant_linear_combination_of_eta_exists']],
            'jacobian_summary': jacobian_summary,
        },
    )

    print('Phase 1 symbolic analysis summary')
    print(f"  constant containment holds for any delta coordinate = {any(row['constant_linear_combination_of_eta_exists'] for row in containment_rows)}")
    print(f"  eta Jacobian rank samples = {jacobian_summary['eta_rank_sample_min']}..{jacobian_summary['eta_rank_sample_max']}")
    print(f"  joint Jacobian rank samples = {jacobian_summary['joint_rank_sample_min']}..{jacobian_summary['joint_rank_sample_max']}")


if __name__ == '__main__':
    main()