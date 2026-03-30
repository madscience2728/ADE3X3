"""
ade3x3_step78_hamilton_term_sharing_audit.py

Step 78: Hamilton term-sharing audit.

This step turns the checklist Route 1 into a reproducible analysis:

1. Load the exact Step 75 anticommutator rank-19 decomposition.
2. Reconstruct the Step 75 commutator rank-20 witness from its recorded seed.
3. Test direct shared rank-1 terms between the two decompositions.
4. Measure linear-span overlap between the 19-term and 20-term families.
5. Export the combined 39-term Hamilton split and summary diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    matrix_multiplication_tensor,
)
from src.ade3x3.steps.ade3x3_step74_commutator_anticommutator_rank_scan import (
    build_tensor_specs,
)
from src.ade3x3.steps.ade3x3_step68_layered_correction_tiling import flattening_ranks
from src.ade3x3.steps.ade3x3_step75_anticommutator_rank19_extraction_hamilton_split import (
    factors_to_terms,
    matrix_to_json_fixed,
    read_csv,
    read_step74_best_seed,
    scalar_to_str,
    scan_explicit_seeds,
    term_tensor,
    write_csv,
)


EXPORTS = Path('outputs/exports')
ANTI_COEFFS = EXPORTS / 'step75_anticommutator_rank19_coefficients.csv'
COMM_VERIFICATION = EXPORTS / 'step75_commutator_rank20_verification.csv'
STEP74_COMM_SCAN = EXPORTS / 'step74_commutator_rank_scan.csv'

PAIR_MATCH_TOL = 1e-8
SPAN_TOL = 1e-8


def json_to_matrix(payload: str) -> np.ndarray:
    return np.array(json.loads(payload), dtype=np.float64)


def load_anticommutator_terms() -> list[Term]:
    rows = read_csv(ANTI_COEFFS)
    if not rows:
        raise RuntimeError(f'Missing Step 75 anticommutator coefficients at {ANTI_COEFFS}.')
    terms: list[Term] = []
    for row in rows:
        term_id = str(row['term'])
        terms.append(
            Term(
                term_id=term_id,
                source_label='step75_anticommutator_rank19',
                alpha=json_to_matrix(row['alpha']),
                beta=json_to_matrix(row['beta']),
                gamma=json_to_matrix(row['gamma']),
            )
        )
    return terms


def reconstruct_commutator_terms() -> tuple[list[Term], dict]:
    verification_rows = read_csv(COMM_VERIFICATION)
    if verification_rows:
        best_seed = int(verification_rows[0]['best_seed'])
    else:
        best_seed = read_step74_best_seed(STEP74_COMM_SCAN, 20)

    specs = {spec.tensor_name: spec for spec in build_tensor_specs()}
    comm_tensor = specs['commutator'].tensor
    result = scan_explicit_seeds('commutator', comm_tensor, 20, [best_seed], 1)
    terms = factors_to_terms('comm', result['best_alpha'], result['best_beta'], result['best_gamma'])
    renamed: list[Term] = []
    for term_idx, term in enumerate(terms, start=1):
        renamed.append(Term(f'c{term_idx:02d}', term.source_label, term.alpha, term.beta, term.gamma))
    return renamed, result


def canonical_term_tensor(term: Term) -> tuple[np.ndarray, float]:
    tensor = term_tensor(term).reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(tensor))
    if norm == 0.0:
        return tensor, 0.0
    canonical = tensor / norm
    nz = np.flatnonzero(np.abs(canonical) > 1e-12)
    signed_norm = norm
    if nz.size and canonical[nz[0]] < 0:
        canonical = -canonical
        signed_norm = -norm
    return canonical, signed_norm


def coefficient_rows(label: str, terms: list[Term]) -> list[dict]:
    rows: list[dict] = []
    for term in terms:
        rows.append(
            {
                'label': label,
                'term': term.term_id,
                'alpha': matrix_to_json_fixed(term.alpha, decimals=6),
                'beta': matrix_to_json_fixed(term.beta, decimals=6),
                'gamma': matrix_to_json_fixed(term.gamma, decimals=6),
            }
        )
    return rows


def pairwise_overlap_rows(comm_terms: list[Term], anti_terms: list[Term]) -> tuple[list[dict], list[dict], int]:
    rows: list[dict] = []
    matches: list[dict] = []
    comm_canon = [canonical_term_tensor(term) for term in comm_terms]
    anti_canon = [canonical_term_tensor(term) for term in anti_terms]

    for comm_idx, (comm_vec, comm_scale) in enumerate(comm_canon):
        for anti_idx, (anti_vec, anti_scale) in enumerate(anti_canon):
            diff = float(np.max(np.abs(comm_vec - anti_vec)))
            dot = float(np.dot(comm_vec, anti_vec))
            abs_dot = abs(dot)
            matched = diff <= PAIR_MATCH_TOL
            combined_half_sum_scale = 0.5 * (comm_scale + anti_scale)
            row = {
                'comm_term': comm_terms[comm_idx].term_id,
                'anti_term': anti_terms[anti_idx].term_id,
                'max_abs_difference_after_normalization': scalar_to_str(diff),
                'normalized_dot_product': scalar_to_str(dot),
                'abs_normalized_dot_product': scalar_to_str(abs_dot),
                'comm_scale': scalar_to_str(comm_scale),
                'anti_scale': scalar_to_str(anti_scale),
                'matched_rank1_term': str(matched),
                'cancels_in_half_sum_if_matched': str(matched and abs(combined_half_sum_scale) <= 1e-12),
                'provenance': 'EXACT_DERIVED',
            }
            rows.append(row)
            if matched:
                matches.append(
                    {
                        'comm_term': comm_terms[comm_idx].term_id,
                        'anti_term': anti_terms[anti_idx].term_id,
                        'comm_scale': scalar_to_str(comm_scale),
                        'anti_scale': scalar_to_str(anti_scale),
                        'combined_half_sum_scale': scalar_to_str(combined_half_sum_scale),
                        'cancels_in_T': str(abs(combined_half_sum_scale) <= 1e-12),
                        'provenance': 'EXACT_DERIVED',
                    }
                )

    top_rows = sorted(rows, key=lambda row: float(row['abs_normalized_dot_product']), reverse=True)[:25]
    return rows, top_rows, len(matches)


def stack_term_columns(terms: list[Term]) -> np.ndarray:
    if not terms:
        return np.zeros((729, 0), dtype=np.float64)
    return np.column_stack([term_tensor(term).reshape(-1).astype(np.float64) for term in terms])


def stable_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=SPAN_TOL))


def principal_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros(0, dtype=np.float64)
    left_q, _ = np.linalg.qr(left, mode='reduced')
    right_q, _ = np.linalg.qr(right, mode='reduced')
    singular_values = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return singular_values


def cross_span_membership_rows(comm_matrix: np.ndarray, anti_matrix: np.ndarray, comm_terms: list[Term], anti_terms: list[Term]) -> list[dict]:
    rows: list[dict] = []

    for idx, term in enumerate(comm_terms):
        target = comm_matrix[:, idx]
        coeffs, _, _, _ = np.linalg.lstsq(anti_matrix, target, rcond=None)
        residual = target - anti_matrix @ coeffs
        target_norm = float(np.linalg.norm(target))
        rel_residual = float(np.linalg.norm(residual) / max(target_norm, 1e-18))
        rows.append(
            {
                'term_family': 'commutator',
                'term': term.term_id,
                'other_family': 'anticommutator',
                'relative_span_residual': scalar_to_str(rel_residual),
                'lies_in_other_span': str(rel_residual <= SPAN_TOL),
                'provenance': 'EXACT_DERIVED',
            }
        )

    for idx, term in enumerate(anti_terms):
        target = anti_matrix[:, idx]
        coeffs, _, _, _ = np.linalg.lstsq(comm_matrix, target, rcond=None)
        residual = target - comm_matrix @ coeffs
        target_norm = float(np.linalg.norm(target))
        rel_residual = float(np.linalg.norm(residual) / max(target_norm, 1e-18))
        rows.append(
            {
                'term_family': 'anticommutator',
                'term': term.term_id,
                'other_family': 'commutator',
                'relative_span_residual': scalar_to_str(rel_residual),
                'lies_in_other_span': str(rel_residual <= SPAN_TOL),
                'provenance': 'EXACT_DERIVED',
            }
        )

    return rows


def combined_hamilton_terms(comm_terms: list[Term], anti_terms: list[Term]) -> list[Term]:
    combined: list[Term] = []
    for term in anti_terms:
        combined.append(
            Term(
                term_id=f'h_{len(combined) + 1:02d}',
                source_label='step78_hamilton_split',
                alpha=term.alpha.copy(),
                beta=term.beta.copy(),
                gamma=0.5 * term.gamma.copy(),
            )
        )
    for term in comm_terms:
        combined.append(
            Term(
                term_id=f'h_{len(combined) + 1:02d}',
                source_label='step78_hamilton_split',
                alpha=term.alpha.copy(),
                beta=term.beta.copy(),
                gamma=0.5 * term.gamma.copy(),
            )
        )
    return combined


def summary_rows(
    anti_terms: list[Term],
    comm_terms: list[Term],
    pair_match_count: int,
    anti_matrix: np.ndarray,
    comm_matrix: np.ndarray,
    principal_values: np.ndarray,
    membership_rows: list[dict],
    combined_terms: list[Term],
    target: np.ndarray,
    exact_anti_tensor: np.ndarray,
    exact_comm_tensor: np.ndarray,
    elapsed: float,
) -> list[dict]:
    union_matrix = np.column_stack([anti_matrix, comm_matrix])
    anti_rank = stable_rank(anti_matrix)
    comm_rank = stable_rank(comm_matrix)
    union_rank = stable_rank(union_matrix)
    intersection_dim = anti_rank + comm_rank - union_rank
    combined_tensor = sum((term_tensor(term) for term in combined_terms), start=np.zeros_like(target))
    combined_residual = float(np.max(np.abs(combined_tensor - target)))
    residual_after_half_anti = 0.5 * exact_comm_tensor
    residual_after_half_comm = 0.5 * exact_anti_tensor
    res_half_anti_ranks = flattening_ranks(residual_after_half_anti)
    res_half_comm_ranks = flattening_ranks(residual_after_half_comm)
    best_membership = min(float(row['relative_span_residual']) for row in membership_rows)
    exact_membership = sum(1 for row in membership_rows if row['lies_in_other_span'] == 'True')
    top_principal = principal_values[: min(5, principal_values.size)] if principal_values.size else np.zeros(0, dtype=np.float64)

    return [
        {
            'summary_name': 'step78_anticommutator_term_count',
            'summary_value': str(len(anti_terms)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Exact anticommutator term count imported from Step 75.',
        },
        {
            'summary_name': 'step78_commutator_term_count',
            'summary_value': str(len(comm_terms)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Exact commutator term count reconstructed from the recorded Step 75 seed.',
        },
        {
            'summary_name': 'step78_direct_shared_rank1_term_count',
            'summary_value': str(pair_match_count),
            'provenance': 'EXACT_DERIVED',
            'note': 'Count of directly shared normalized rank-1 terms across the commutator and anticommutator decompositions.',
        },
        {
            'summary_name': 'step78_anticommutator_span_rank',
            'summary_value': str(anti_rank),
            'provenance': 'EXACT_DERIVED',
            'note': 'Linear span dimension of the 19 anticommutator term tensors in ambient dimension 729.',
        },
        {
            'summary_name': 'step78_commutator_span_rank',
            'summary_value': str(comm_rank),
            'provenance': 'EXACT_DERIVED',
            'note': 'Linear span dimension of the 20 commutator term tensors in ambient dimension 729.',
        },
        {
            'summary_name': 'step78_union_span_rank',
            'summary_value': str(union_rank),
            'provenance': 'EXACT_DERIVED',
            'note': 'Linear span dimension of the full 39-term Hamilton union.',
        },
        {
            'summary_name': 'step78_span_intersection_dimension',
            'summary_value': str(intersection_dim),
            'provenance': 'EXACT_DERIVED',
            'note': 'dim(span(commutator_terms) ∩ span(anticommutator_terms)).',
        },
        {
            'summary_name': 'step78_best_cross_span_membership_residual',
            'summary_value': scalar_to_str(best_membership),
            'provenance': 'EXACT_DERIVED',
            'note': 'Best relative residual for a single term projected into the opposite family span.',
        },
        {
            'summary_name': 'step78_exact_cross_span_membership_count',
            'summary_value': str(exact_membership),
            'provenance': 'EXACT_DERIVED',
            'note': 'Number of individual terms that lie in the opposite family span at tolerance 1e-8.',
        },
        {
            'summary_name': 'step78_top5_principal_cosines',
            'summary_value': json.dumps([float(value) for value in top_principal], separators=(',', ':')),
            'provenance': 'EXACT_DERIVED',
            'note': 'Largest principal cosines between the commutator and anticommutator term spans.',
        },
        {
            'summary_name': 'step78_combined_hamilton_term_count',
            'summary_value': str(len(combined_terms)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Naive term count for T = 0.5*T_anti + 0.5*T_comm with no nonlinear recompression.',
        },
        {
            'summary_name': 'step78_combined_hamilton_max_abs_residual',
            'summary_value': scalar_to_str(combined_residual),
            'provenance': 'EXACT_DERIVED',
            'note': 'Verification residual for the explicit 39-term Hamilton split.',
        },
        {
            'summary_name': 'step78_residual_after_half_anti_flattening_lb',
            'summary_value': str(res_half_anti_ranks[3]),
            'provenance': 'EXACT_DERIVED',
            'note': 'Flattening lower bound for T - 0.5*T_anti = 0.5*T_comm.',
        },
        {
            'summary_name': 'step78_residual_after_half_anti_flattening_ranks',
            'summary_value': json.dumps(list(res_half_anti_ranks[:3]), separators=(',', ':')),
            'provenance': 'EXACT_DERIVED',
            'note': 'Mode flattening ranks for T - 0.5*T_anti.',
        },
        {
            'summary_name': 'step78_residual_after_half_comm_flattening_lb',
            'summary_value': str(res_half_comm_ranks[3]),
            'provenance': 'EXACT_DERIVED',
            'note': 'Flattening lower bound for T - 0.5*T_comm = 0.5*T_anti.',
        },
        {
            'summary_name': 'step78_residual_after_half_comm_flattening_ranks',
            'summary_value': json.dumps(list(res_half_comm_ranks[:3]), separators=(',', ':')),
            'provenance': 'EXACT_DERIVED',
            'note': 'Mode flattening ranks for T - 0.5*T_comm.',
        },
        {
            'summary_name': 'step78_runtime_seconds',
            'summary_value': f'{elapsed:.6f}',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Wall-clock runtime for the Hamilton-sharing audit.',
        },
    ]


def main() -> None:
    start = time.time()
    print('=== Step 78: Hamilton term-sharing audit ===', flush=True)

    specs = {spec.tensor_name: spec for spec in build_tensor_specs()}
    anti_terms = load_anticommutator_terms()
    comm_terms, comm_result = reconstruct_commutator_terms()

    anti_tensor = sum((term_tensor(term) for term in anti_terms), start=np.zeros((9, 9, 9), dtype=np.float64))
    comm_tensor = sum((term_tensor(term) for term in comm_terms), start=np.zeros((9, 9, 9), dtype=np.float64))
    target = matrix_multiplication_tensor(3).astype(np.float64)

    anti_matrix = stack_term_columns(anti_terms)
    comm_matrix = stack_term_columns(comm_terms)

    overlap_rows, overlap_top_rows, pair_match_count = pairwise_overlap_rows(comm_terms, anti_terms)
    membership_rows = cross_span_membership_rows(comm_matrix, anti_matrix, comm_terms, anti_terms)
    principal_values = principal_cosines(comm_matrix, anti_matrix)
    principal_rows = [
        {
            'index': idx + 1,
            'principal_cosine': scalar_to_str(float(value)),
            'principal_angle_radians': scalar_to_str(float(math.acos(min(1.0, max(0.0, float(value)))))),
            'provenance': 'EXACT_DERIVED',
        }
        for idx, value in enumerate(principal_values)
    ]

    combined_terms = combined_hamilton_terms(comm_terms, anti_terms)
    combined_rows = coefficient_rows('combined_hamilton', combined_terms)
    summary = summary_rows(
        anti_terms,
        comm_terms,
        pair_match_count,
        anti_matrix,
        comm_matrix,
        principal_values,
        membership_rows,
        combined_terms,
        target,
        specs['anticommutator'].tensor,
        specs['commutator'].tensor,
        time.time() - start,
    )

    write_csv(
        EXPORTS / 'step78_commutator_rank20_coefficients.csv',
        coefficient_rows('commutator_rank20', comm_terms),
        ['label', 'term', 'alpha', 'beta', 'gamma'],
    )
    write_csv(
        EXPORTS / 'step78_hamilton_pairwise_overlap.csv',
        overlap_rows,
        ['comm_term', 'anti_term', 'max_abs_difference_after_normalization', 'normalized_dot_product', 'abs_normalized_dot_product', 'comm_scale', 'anti_scale', 'matched_rank1_term', 'cancels_in_half_sum_if_matched', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step78_hamilton_top_pairwise_overlap.csv',
        overlap_top_rows,
        ['comm_term', 'anti_term', 'max_abs_difference_after_normalization', 'normalized_dot_product', 'abs_normalized_dot_product', 'comm_scale', 'anti_scale', 'matched_rank1_term', 'cancels_in_half_sum_if_matched', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step78_hamilton_cross_span_membership.csv',
        membership_rows,
        ['term_family', 'term', 'other_family', 'relative_span_residual', 'lies_in_other_span', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step78_hamilton_principal_angles.csv',
        principal_rows,
        ['index', 'principal_cosine', 'principal_angle_radians', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step78_hamilton_combined_term_table.csv',
        combined_rows,
        ['label', 'term', 'alpha', 'beta', 'gamma'],
    )
    write_csv(
        EXPORTS / 'step78_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )

    print(f"Reconstructed commutator from seed {comm_result['best_seed']}", flush=True)
    print(f'Completed Step 78 in {time.time() - start:.2f}s', flush=True)


if __name__ == '__main__':
    main()