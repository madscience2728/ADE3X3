from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    build_mode_matrices,
    load_public_rank23_terms,
)


SVD_TOL = 1e-10


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def eta_coordinates() -> list[str]:
    coords: list[str] = []
    for row_idx in range(3):
        for col_idx in range(3):
            coords.append(f'eta1[{row_idx},{col_idx}]')
    for row_idx in range(3):
        for col_idx in range(3):
            coords.append(f'eta2[{row_idx},{col_idx}]')
    return coords


def delta_coordinates() -> list[str]:
    coords: list[str] = []
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    coords.append(f'delta[{row_idx},{sum_left},{sum_right},{col_idx}]')
    return coords


def numeric_rank_report(matrix: np.ndarray) -> dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > SVD_TOL))
    boundary_sv = float(singular_values[rank - 1]) if rank else 0.0
    next_sv = float(singular_values[rank]) if rank < singular_values.size else 0.0
    gap_ratio = float('inf') if next_sv == 0.0 else boundary_sv / next_sv
    return {
        'rank': rank,
        'threshold': SVD_TOL,
        'boundary_singular_value': boundary_sv,
        'next_singular_value': next_sv,
        'gap_ratio': gap_ratio,
        'singular_values': [float(value) for value in singular_values],
    }


def solve_delta_projection(h_matrix: sp.Matrix, delta_matrix: sp.Matrix) -> tuple[tuple[int, ...], sp.Matrix, list[dict]]:
    _, pivots = h_matrix.rref()
    h_basis = h_matrix[:, pivots]
    eta_coords = eta_coordinates()
    delta_coords = delta_coordinates()

    projection_columns: list[list[sp.Expr]] = []
    rows: list[dict] = []
    for delta_idx, delta_coord in enumerate(delta_coords):
        basis_solution = list(h_basis.gauss_jordan_solve(delta_matrix[:, delta_idx])[0])
        full_solution = [sp.Integer(0)] * h_matrix.cols
        for basis_idx, pivot in enumerate(pivots):
            full_solution[pivot] = sp.simplify(basis_solution[basis_idx])
        projection_columns.append(full_solution)

        row: dict[str, object] = {
            'delta_coordinate': delta_coord,
            'support_size': sum(1 for value in full_solution if value != 0),
            'support': ';'.join(
                eta_coords[eta_idx]
                for eta_idx, value in enumerate(full_solution)
                if value != 0
            ),
        }
        for eta_idx, eta_coord in enumerate(eta_coords):
            row[eta_coord] = sp.sstr(full_solution[eta_idx])
        rows.append(row)

    projection_matrix = sp.Matrix(
        h_matrix.cols,
        delta_matrix.cols,
        lambda row_idx, col_idx: projection_columns[col_idx][row_idx],
    )
    if h_matrix * projection_matrix != delta_matrix:
        raise RuntimeError('Exact projection verification failed: H * M != Delta')
    return tuple(int(pivot) for pivot in pivots), projection_matrix, rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terms, orientation, residual = load_public_rank23_terms()
    _, eta1, eta2, delta, _ = build_mode_matrices(terms, 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    hd_matrix = sp.Matrix.hstack(h_matrix, delta)

    h_np = np.array(h_matrix.tolist(), dtype=float)
    delta_np = np.array(delta.tolist(), dtype=float)
    hd_np = np.hstack([h_np, delta_np])

    pivots, projection_matrix, projection_rows = solve_delta_projection(h_matrix, delta)
    nonzero_projection_values = [projection_matrix[row_idx, col_idx] for row_idx in range(projection_matrix.rows) for col_idx in range(projection_matrix.cols) if projection_matrix[row_idx, col_idx] != 0]
    unique_coefficients = sorted({sp.sstr(value) for value in nonzero_projection_values})
    zero_delta_columns = [
        delta_coordinates()[col_idx]
        for col_idx in range(delta.cols)
        if all(delta[row_idx, col_idx] == 0 for row_idx in range(delta.rows))
    ]

    rank_summary_rows = [
        {
            'matrix_name': 'H',
            'rows': h_matrix.rows,
            'cols': h_matrix.cols,
            'exact_rank': int(h_matrix.rank()),
            'numeric_rank': numeric_rank_report(h_np)['rank'],
            'boundary_singular_value': numeric_rank_report(h_np)['boundary_singular_value'],
            'next_singular_value': numeric_rank_report(h_np)['next_singular_value'],
            'gap_ratio': numeric_rank_report(h_np)['gap_ratio'],
        },
        {
            'matrix_name': 'Delta',
            'rows': delta.rows,
            'cols': delta.cols,
            'exact_rank': int(delta.rank()),
            'numeric_rank': numeric_rank_report(delta_np)['rank'],
            'boundary_singular_value': numeric_rank_report(delta_np)['boundary_singular_value'],
            'next_singular_value': numeric_rank_report(delta_np)['next_singular_value'],
            'gap_ratio': numeric_rank_report(delta_np)['gap_ratio'],
        },
        {
            'matrix_name': '[H|Delta]',
            'rows': hd_matrix.rows,
            'cols': hd_matrix.cols,
            'exact_rank': int(hd_matrix.rank()),
            'numeric_rank': numeric_rank_report(hd_np)['rank'],
            'boundary_singular_value': numeric_rank_report(hd_np)['boundary_singular_value'],
            'next_singular_value': numeric_rank_report(hd_np)['next_singular_value'],
            'gap_ratio': numeric_rank_report(hd_np)['gap_ratio'],
        },
    ]
    write_csv(
        OUT_DIR / 'rank_summary.csv',
        rank_summary_rows,
        ['matrix_name', 'rows', 'cols', 'exact_rank', 'numeric_rank', 'boundary_singular_value', 'next_singular_value', 'gap_ratio'],
    )
    write_csv(
        OUT_DIR / 'delta_projection_matrix.csv',
        projection_rows,
        ['delta_coordinate', 'support_size', 'support', *eta_coordinates()],
    )

    summary = {
        'source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
        'gamma_orientation': orientation,
        'exact_reconstruction_residual': int(residual),
        'rank_H_exact': int(h_matrix.rank()),
        'rank_Delta_exact': int(delta.rank()),
        'rank_HDelta_exact': int(hd_matrix.rank()),
        'collection_level_delta_in_eta': bool(hd_matrix.rank() == h_matrix.rank()),
        'eta_pivot_columns': list(pivots),
        'projection_exact_verification': bool(h_matrix * projection_matrix == delta),
        'projection_unique_nonzero_coefficients': unique_coefficients,
        'projection_nonzero_count': len(nonzero_projection_values),
        'zero_delta_column_count': len(zero_delta_columns),
        'zero_delta_columns': zero_delta_columns,
        'numeric_rank_reports': {
            'H': numeric_rank_report(h_np),
            'Delta': numeric_rank_report(delta_np),
            'H_and_Delta': numeric_rank_report(hd_np),
        },
    }
    write_json(OUT_DIR / 'containment_summary.json', summary)

    print('Phase 1 containment summary')
    print(f'  gamma orientation = {orientation}')
    print(f'  exact ranks: H={h_matrix.rank()} Delta={delta.rank()} [H|Delta]={hd_matrix.rank()}')
    print(f'  projection exact = {h_matrix * projection_matrix == delta}')
    print(f'  unique nonzero projection coefficients = {unique_coefficients}')


if __name__ == '__main__':
    main()