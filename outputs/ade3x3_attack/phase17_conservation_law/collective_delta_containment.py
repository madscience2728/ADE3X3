from __future__ import annotations

import ast
import csv
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
    Term,
    delta_in_eta_numeric,
    eta_coordinate_labels,
    load_public_terms,
    profile_terms,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    build_mode_matrices,
    gamma_matrix,
)


RANK_TOL = 1e-10
PHASE1_PROJECTION = ATTACK_ROOT / 'phase1_delta_containment' / 'delta_projection_matrix.csv'
STEP75_FIBERMODE = REPO_ROOT / 'outputs' / 'exports' / 'step75_anticommutator_rank19_fibermode.csv'


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def load_projection_matrix() -> tuple[list[str], list[str], sp.Matrix]:
    labels = eta_coordinate_labels()
    delta_labels: list[str] = []
    columns: list[list[int]] = []
    with open(PHASE1_PROJECTION, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            delta_labels.append(row['delta_coordinate'])
            columns.append([int(row[label]) for label in labels])
    matrix = sp.Matrix(18, len(columns), lambda row_idx, col_idx: columns[col_idx][row_idx])
    return labels, delta_labels, matrix


def verify_projection_identity(terms: list[Term]) -> dict[str, object]:
    _, delta_labels, projection = load_projection_matrix()
    _, eta1_exact, eta2_exact, delta_exact, _ = build_mode_matrices(terms, 3)
    h_exact = sp.Matrix.hstack(eta1_exact, eta2_exact)
    gamma_exact = gamma_matrix(terms, 3)

    reconstructed = h_exact * projection
    exact_match = reconstructed == delta_exact

    weighted_max_abs = 0.0
    nonzero_weighted_columns = 0
    for col_idx, delta_label in enumerate(delta_labels):
        lhs = gamma_exact * delta_exact[:, col_idx]
        rhs = gamma_exact * reconstructed[:, col_idx]
        residual = lhs - rhs
        entry_max = max(float(abs(value)) for value in residual) if residual.rows else 0.0
        weighted_max_abs = max(weighted_max_abs, entry_max)
        if any(value != 0 for value in lhs):
            nonzero_weighted_columns += 1
    return {
        'projection_source': str(PHASE1_PROJECTION.relative_to(REPO_ROOT)).replace('\\', '/'),
        'tested_delta_columns': len(delta_labels),
        'tested_output_rows': gamma_exact.rows,
        'exact_projection_verified': exact_match,
        'weighted_zero_verification_passed': weighted_max_abs == 0.0,
        'weighted_projection_max_abs': weighted_max_abs,
        'nonzero_weighted_columns': nonzero_weighted_columns,
        'projection_nonzero_count': int(sum(1 for value in projection if value != 0)),
    }


def parse_matrix_field(raw: str) -> np.ndarray:
    return np.array(ast.literal_eval(raw), dtype=np.float64)


def anticommutator_profile() -> dict[str, object]:
    with open(STEP75_FIBERMODE, 'r', encoding='utf-8', newline='') as handle:
        row = next(csv.DictReader(handle))
    eta1 = parse_matrix_field(row['eta1_matrix'])
    eta2 = parse_matrix_field(row['eta2_matrix'])
    delta = parse_matrix_field(row['delta_matrix'])
    h_matrix = np.hstack([eta1, eta2])
    profile = delta_in_eta_numeric(h_matrix, delta, rank_tol=RANK_TOL, projection_tol=1e-8)
    return {
        'available': True,
        'label': row['label'],
        'source': str(STEP75_FIBERMODE.relative_to(REPO_ROOT)).replace('\\', '/'),
        'rank_H_numeric': profile.rank_h,
        'rank_Delta_numeric': profile.rank_delta,
        'rank_HDelta_numeric': profile.rank_joint,
        'delta_in_eta_numeric': profile.delta_in_eta,
        'projection_max_residual_numeric': profile.projection_max_residual,
        'projection_nonzero_count_numeric': profile.projection_nonzero_count,
        'projection_unique_nonzero_coefficients_numeric': profile.projection_unique_nonzero_coefficients,
    }


def random_rank23_rows(trial_count: int = 1000) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(20260329)
    rows: list[dict[str, object]] = []
    containment_count = 0
    for trial_idx in range(trial_count):
        alpha = rng.standard_normal((23, 9))
        beta = rng.standard_normal((23, 9))
        sigma = np.zeros((23, 9), dtype=np.float64)
        eta1 = np.zeros((23, 9), dtype=np.float64)
        eta2 = np.zeros((23, 9), dtype=np.float64)
        delta = np.zeros((23, 54), dtype=np.float64)

        sigma_col = 0
        for row_idx in range(3):
            for col_idx in range(3):
                lambda0 = alpha[:, 3 * row_idx + 0] * beta[:, 0 * 3 + col_idx]
                lambda1 = alpha[:, 3 * row_idx + 1] * beta[:, 1 * 3 + col_idx]
                lambda2 = alpha[:, 3 * row_idx + 2] * beta[:, 2 * 3 + col_idx]
                sigma[:, sigma_col] = lambda0 + lambda1 + lambda2
                eta1[:, sigma_col] = lambda0 - lambda1
                eta2[:, sigma_col] = lambda1 - lambda2
                sigma_col += 1

        delta_col = 0
        for row_idx in range(3):
            for sum_left in range(3):
                for sum_right in range(3):
                    if sum_left == sum_right:
                        continue
                    for col_idx in range(3):
                        delta[:, delta_col] = alpha[:, 3 * row_idx + sum_left] * beta[:, 3 * sum_right + col_idx]
                        delta_col += 1

        h_matrix = np.hstack([eta1, eta2])
        profile = delta_in_eta_numeric(h_matrix, delta, rank_tol=RANK_TOL, projection_tol=1e-8)
        containment_count += int(profile.delta_in_eta)
        rows.append(
            {
                'trial_index': trial_idx + 1,
                'rank_H_numeric': profile.rank_h,
                'rank_Delta_numeric': profile.rank_delta,
                'rank_HDelta_numeric': profile.rank_joint,
                'delta_in_eta_numeric': profile.delta_in_eta,
                'projection_max_residual_numeric': profile.projection_max_residual,
            }
        )

    summary = {
        'trial_count': trial_count,
        'containment_count': containment_count,
        'containment_rate': containment_count / trial_count,
        'min_rank_H_numeric': min(int(row['rank_H_numeric']) for row in rows),
        'max_rank_H_numeric': max(int(row['rank_H_numeric']) for row in rows),
        'min_rank_HDelta_numeric': min(int(row['rank_HDelta_numeric']) for row in rows),
        'max_rank_HDelta_numeric': max(int(row['rank_HDelta_numeric']) for row in rows),
    }
    return rows, summary


def main() -> None:
    alpha_terms, orientation, residual = load_public_terms()
    projection_summary = verify_projection_identity(alpha_terms)

    deletion_rows: list[dict[str, object]] = []
    for term in alpha_terms:
        subset = [candidate for candidate in alpha_terms if candidate.term_id != term.term_id]
        profile = profile_terms(subset)
        deletion_rows.append(
            {
                'removed_term': term.term_id,
                'remaining_R': len(subset),
                'rank_H_numeric': profile['rank_H_numeric'],
                'rank_HDelta_numeric': profile['rank_HDelta_numeric'],
                'delta_in_eta_numeric': profile['delta_in_eta_numeric'],
                'projection_max_residual_numeric': profile['projection_max_residual_numeric'],
                'rank_H_exact': profile.get('rank_H_exact', ''),
                'rank_HDelta_exact': profile.get('rank_HDelta_exact', ''),
                'delta_in_eta_exact': profile.get('delta_in_eta_exact', ''),
            }
        )

    write_csv(
        OUT_DIR / 'single_deletion_containment.csv',
        deletion_rows,
        [
            'removed_term', 'remaining_R', 'rank_H_numeric', 'rank_HDelta_numeric', 'delta_in_eta_numeric',
            'projection_max_residual_numeric', 'rank_H_exact', 'rank_HDelta_exact', 'delta_in_eta_exact',
        ],
    )

    anticommutator = anticommutator_profile()
    random_rows, random_summary = random_rank23_rows(trial_count=1000)
    write_csv(
        OUT_DIR / 'random_containment_rate.csv',
        random_rows,
        ['trial_index', 'rank_H_numeric', 'rank_Delta_numeric', 'rank_HDelta_numeric', 'delta_in_eta_numeric', 'projection_max_residual_numeric'],
    )

    deletion_failures = [row['removed_term'] for row in deletion_rows if not bool(row['delta_in_eta_exact'])]
    payload = {
        'alphatensor_reference': {
            'orientation': orientation,
            'reconstruction_max_abs': residual,
        },
        'alpha_tensor_projection': projection_summary,
        'single_deletion_containment': {
            'tested': len(deletion_rows),
            'containment_numeric_count': sum(bool(row['delta_in_eta_numeric']) for row in deletion_rows),
            'containment_exact_count': sum(bool(row['delta_in_eta_exact']) for row in deletion_rows),
            'all_exact': len(deletion_failures) == 0,
            'failing_terms': deletion_failures,
        },
        'anticommutator_rank19': anticommutator,
        'random_rank23_collections': random_summary,
    }
    write_json(OUT_DIR / 'collective_containment_summary.json', payload)

    print('Phase 17C complete.')
    print(f"  AlphaTensor exact projection verified: {projection_summary['exact_projection_verified']}")
    print(f"  Single-deletion containment exact for all 23 removals: {payload['single_deletion_containment']['all_exact']}")
    print(f"  Anticommutator containment holds: {anticommutator['delta_in_eta_numeric']}")
    print(f"  Random containment rate: {random_summary['containment_rate']:.6f}")


if __name__ == '__main__':
    main()