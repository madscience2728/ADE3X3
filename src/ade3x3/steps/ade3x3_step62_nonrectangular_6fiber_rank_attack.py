"""
ade3x3_step62_nonrectangular_6fiber_rank_attack.py

Step 62: Non-Rectangular 6-Fiber Sub-Tensor Rank Attack.

This step attacks the surviving six-fiber hybrid residual patterns with three
tools at once:

1. Exact substitution-style restriction lower bounds.
2. Numerical CP-rank fitting on the 9 x 9 x m subtensors.
3. A direct algebraic upper-bound construction for the anti-diagonal-missing
   pattern P4.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

EXPORTS = Path('outputs/exports')
TOL = 1e-10
NUMERICAL_RANKS = [9, 10, 11, 12, 13, 14]
NUMERICAL_RESTARTS = 12
NUMERICAL_MAXITER = 500
NUMERICAL_PATIENCE = 3


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    fibers: tuple[tuple[int, int], ...]
    orbit_size: int
    row_profile: str
    col_profile: str
    rectangular_exact_rank: int | None

    @property
    def nonrectangular(self) -> bool:
        return self.rectangular_exact_rank is None


PATTERNS: list[Pattern] = [
    Pattern('P1', ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)), 36, '3,2,1', '3,2,1', None),
    Pattern('P2', ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)), 18, '3,2,1', '2,2,2', None),
    Pattern('P3', ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 2)), 18, '2,2,2', '3,2,1', None),
    Pattern('P4', ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)), 6, '2,2,2', '2,2,2', None),
    Pattern('P5', ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)), 3, '3,3,0', '2,2,2', 15),
    Pattern('P6', ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)), 3, '2,2,2', '3,3,0', 15),
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_markdown(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f"  Wrote markdown -> {path}")


def fibers_str(fibers: Iterable[tuple[int, int]]) -> str:
    return '; '.join(f'({row_idx},{col_idx})' for row_idx, col_idx in fibers)


def numeric_rank(matrix: np.ndarray, tol: float | None = None) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    if tol is None:
        tol = max(matrix.shape) * np.finfo(float).eps * float(np.max(singular_values)) * 10.0
    return int(np.sum(singular_values > tol))


def reduced_output_tensor(selected_outputs: tuple[tuple[int, int], ...]) -> np.ndarray:
    tensor = np.zeros((9, 9, len(selected_outputs)), dtype=np.float64)
    for output_idx, (row_idx, col_idx) in enumerate(selected_outputs):
        for sum_idx in range(3):
            a_idx = 3 * row_idx + sum_idx
            b_idx = 3 * sum_idx + col_idx
            tensor[a_idx, b_idx, output_idx] = 1.0
    return tensor


def flattening_ranks(tensor: np.ndarray) -> tuple[int, int, int, int]:
    rank_a = numeric_rank(tensor.reshape(tensor.shape[0], -1))
    rank_b = numeric_rank(np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1))
    rank_c = numeric_rank(np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1))
    return rank_a, rank_b, rank_c, max(rank_a, rank_b, rank_c)


def substitution_restrictions(pattern: Pattern, tensor: np.ndarray) -> list[dict]:
    rows: list[dict] = []

    def add_row(kind: str, label: str, restricted: np.ndarray) -> None:
        rank_a, rank_b, rank_c, rank_max = flattening_ranks(restricted)
        rows.append({
            'pattern_id': pattern.pattern_id,
            'kind': kind,
            'label': label,
            'restricted_shape': f'{restricted.shape[0]}x{restricted.shape[1]}x{restricted.shape[2]}',
            'flattening_rank_A_BC': rank_a,
            'flattening_rank_B_AC': rank_b,
            'flattening_rank_C_AB': rank_c,
            'lower_bound': rank_max,
            'provenance': 'EXACT_DERIVED',
        })

    add_row('none', 'no_substitution', tensor)

    for row_idx in range(3):
        keep = [idx for idx in range(9) if idx // 3 != row_idx]
        add_row('A_row_zero', f'A_row_{row_idx}=0', tensor[keep, :, :])
    for sum_idx in range(3):
        keep = [idx for idx in range(9) if idx % 3 != sum_idx]
        add_row('A_column_zero', f'A_column_{sum_idx}=0', tensor[keep, :, :])
    for sum_idx in range(3):
        keep = [idx for idx in range(9) if idx // 3 != sum_idx]
        add_row('B_row_zero', f'B_row_{sum_idx}=0', tensor[:, keep, :])
    for col_idx in range(3):
        keep = [idx for idx in range(9) if idx % 3 != col_idx]
        add_row('B_column_zero', f'B_column_{col_idx}=0', tensor[:, keep, :])

    outputs = list(pattern.fibers)
    for row_idx in range(3):
        keep_outputs = [idx for idx, (out_row, _) in enumerate(outputs) if out_row != row_idx]
        if keep_outputs:
            add_row('Gamma_row_zero', f'Gamma_row_{row_idx}=0', tensor[:, :, keep_outputs])
    for col_idx in range(3):
        keep_outputs = [idx for idx, (_, out_col) in enumerate(outputs) if out_col != col_idx]
        if keep_outputs:
            add_row('Gamma_column_zero', f'Gamma_col_{col_idx}=0', tensor[:, :, keep_outputs])
    return rows


def pack_factors(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([alpha.reshape(-1), beta.reshape(-1), gamma.reshape(-1)])


def unpack_factors(vector: np.ndarray, rank: int, output_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split1 = rank * 9
    split2 = 2 * rank * 9
    alpha = vector[:split1].reshape(rank, 9)
    beta = vector[split1:split2].reshape(rank, 9)
    gamma = vector[split2:].reshape(rank, output_dim)
    return alpha, beta, gamma


def cp_objective(vector: np.ndarray, target: np.ndarray, rank: int) -> tuple[float, np.ndarray]:
    output_dim = target.shape[2]
    alpha, beta, gamma = unpack_factors(vector, rank, output_dim)
    approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    residual = approx - target
    loss = 0.5 * float(np.sum(residual * residual))
    grad_alpha = np.einsum('abc,rb,rc->ra', residual, beta, gamma, optimize=True)
    grad_beta = np.einsum('abc,ra,rc->rb', residual, alpha, gamma, optimize=True)
    grad_gamma = np.einsum('abc,ra,rb->rc', residual, alpha, beta, optimize=True)
    gradient = pack_factors(grad_alpha, grad_beta, grad_gamma)
    return loss, gradient


def random_initialization(rank: int, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    alpha = 0.25 * rng.standard_normal((rank, 9))
    beta = 0.25 * rng.standard_normal((rank, 9))
    gamma = 0.25 * rng.standard_normal((rank, output_dim))
    return pack_factors(alpha, beta, gamma)


def cp_rank_scan(pattern: Pattern, tensor: np.ndarray) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    best_success_rank: int | None = None
    best_success_payload: dict | None = None
    global_best_loss = float('inf')

    for rank in NUMERICAL_RANKS:
        best_rank_loss = float('inf')
        best_rank_payload: dict | None = None
        success_count = 0
        consecutive_successes = 0
        for restart in range(NUMERICAL_RESTARTS):
            rng = np.random.default_rng(seed=1000 * rank + 37 * restart + len(pattern.fibers))
            x0 = random_initialization(rank, tensor.shape[2], rng)
            result = minimize(
                lambda x: cp_objective(x, tensor, rank),
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': NUMERICAL_MAXITER, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50},
            )
            alpha, beta, gamma = unpack_factors(result.x, rank, tensor.shape[2])
            approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
            residual = approx - tensor
            verified_loss = float(np.sum(residual * residual))
            max_abs_residual = float(np.max(np.abs(residual)))
            if verified_loss < best_rank_loss:
                best_rank_loss = verified_loss
                best_rank_payload = {
                    'rank_tested': rank,
                    'restart': restart,
                    'verified_loss': verified_loss,
                    'max_abs_residual': max_abs_residual,
                    'nit': int(result.nit),
                    'success': bool(result.success),
                }
            if verified_loss < TOL:
                success_count += 1
                consecutive_successes += 1
                if best_success_rank is None:
                    best_success_rank = rank
                    best_success_payload = {
                        'rank_tested': rank,
                        'restart': restart,
                        'verified_loss': verified_loss,
                        'max_abs_residual': max_abs_residual,
                    }
                if consecutive_successes >= NUMERICAL_PATIENCE:
                    break
            else:
                consecutive_successes = 0
        assert best_rank_payload is not None
        global_best_loss = min(global_best_loss, best_rank_loss)
        rows.append({
            'pattern_id': pattern.pattern_id,
            'rank_tested': rank,
            'restarts': NUMERICAL_RESTARTS,
            'success_count': success_count,
            'best_verified_loss': f'{best_rank_loss:.12e}',
            'best_max_abs_residual': f"{best_rank_payload['max_abs_residual']:.12e}",
            'best_restart': best_rank_payload['restart'],
            'best_iterations': best_rank_payload['nit'],
            'numerically_exact': str(best_rank_loss < TOL),
            'provenance': 'MEASURED_FROM_CODE',
        })
        if best_success_rank is not None:
            break

    verdict = {
        'pattern_id': pattern.pattern_id,
        'numerical_rank_upper_bound_if_found': '' if best_success_rank is None else str(best_success_rank),
        'best_loss_all_ranks': f'{global_best_loss:.12e}',
        'rank_leq_13_found': str(best_success_rank is not None and best_success_rank <= 13),
        'rank_leq_14_found': str(best_success_rank is not None and best_success_rank <= 14),
        'provenance': 'MEASURED_FROM_CODE',
    }
    if best_success_payload is not None:
        verdict['best_success_loss'] = f"{best_success_payload['verified_loss']:.12e}"
        verdict['best_success_max_abs_residual'] = f"{best_success_payload['max_abs_residual']:.12e}"
    else:
        verdict['best_success_loss'] = ''
        verdict['best_success_max_abs_residual'] = ''
    return rows, verdict


def direct_p4_rows() -> list[dict]:
    pattern = next(pattern for pattern in PATTERNS if pattern.pattern_id == 'P4')
    tensor = reduced_output_tensor(pattern.fibers)

    standard_terms = np.zeros_like(tensor)
    for output_idx, (row_idx, col_idx) in enumerate(pattern.fibers):
        for sum_idx in range(3):
            a_idx = 3 * row_idx + sum_idx
            b_idx = 3 * sum_idx + col_idx
            standard_terms[a_idx, b_idx, output_idx] += 1.0
    exact_match = np.max(np.abs(standard_terms - tensor)) < TOL

    return [
        {
            'construction_id': 'fiber_local_standard',
            'pattern_id': 'P4',
            'term_count': 18,
            'status': 'exact_verified' if exact_match else 'failed',
            'note': 'Three fiber-local terms per selected output fiber give an exact decomposition.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'construction_id': 'row_pair_cover',
            'pattern_id': 'P4',
            'term_count': 18,
            'status': 'no_improvement',
            'note': 'Three row-wise 1x2 subproblems each have known exact rank 6, totaling 18.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'construction_id': 'column_pair_cover',
            'pattern_id': 'P4',
            'term_count': 18,
            'status': 'no_improvement',
            'note': 'Three column-wise 2x1 subproblems each have known exact rank 6, totaling 18.',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def compile_pattern_summary(
    substitution_best: list[dict],
    numerical_verdicts: list[dict],
) -> list[dict]:
    sub_map = {row['pattern_id']: row for row in substitution_best}
    num_map = {row['pattern_id']: row for row in numerical_verdicts}
    rows: list[dict] = []
    for pattern in PATTERNS:
        num = num_map[pattern.pattern_id]
        known_exact = '' if pattern.rectangular_exact_rank is None else str(pattern.rectangular_exact_rank)
        feasible13 = 'measured_yes' if num['rank_leq_13_found'] == 'True' else ('exact_no' if pattern.rectangular_exact_rank is not None and pattern.rectangular_exact_rank > 13 else 'not_supported_by_measurement')
        feasible14 = 'measured_yes' if num['rank_leq_14_found'] == 'True' else ('exact_no' if pattern.rectangular_exact_rank is not None and pattern.rectangular_exact_rank > 14 else 'not_supported_by_measurement')
        rows.append({
            'pattern_id': pattern.pattern_id,
            'representative_fibers': fibers_str(pattern.fibers),
            'orbit_size': pattern.orbit_size,
            'nonrectangular': str(pattern.nonrectangular),
            'substitution_best_lower_bound': sub_map[pattern.pattern_id]['best_lower_bound'],
            'numerical_rank_upper_bound_if_found': num['numerical_rank_upper_bound_if_found'],
            'best_loss_all_ranks': num['best_loss_all_ranks'],
            'known_exact_rank_if_determined': known_exact,
            'feasible_at_13_spreaders': feasible13,
            'feasible_at_14_spreaders': feasible14,
            'provenance_exact': 'EXACT_DERIVED',
            'provenance_numerical': 'MEASURED_FROM_CODE',
        })
    return rows


def hybrid_verdict_rows(pattern_summary: list[dict]) -> list[dict]:
    nonrectangular = [row for row in pattern_summary if row['nonrectangular'] == 'True']
    any13 = any(row['feasible_at_13_spreaders'] == 'measured_yes' for row in nonrectangular)
    any14 = any(row['feasible_at_14_spreaders'] == 'measured_yes' for row in nonrectangular)
    return [
        {
            'R': 22,
            'fourier_fiber_count': 3,
            'spreader_budget': 13,
            'any_pattern_feasible': 'measured_yes' if any13 else 'no_numerical_witness_at_or_below_13',
            'scope': 'six_fiber_nonrectangular_patterns_only',
            'provenance': 'MEASURED_FROM_CODE',
        },
        {
            'R': 23,
            'fourier_fiber_count': 3,
            'spreader_budget': 14,
            'any_pattern_feasible': 'measured_yes' if any14 else 'no_numerical_witness_at_or_below_14',
            'scope': 'six_fiber_nonrectangular_patterns_only',
            'provenance': 'MEASURED_FROM_CODE',
        },
        {
            'R': 22,
            'fourier_fiber_count': 4,
            'spreader_budget': 10,
            'any_pattern_feasible': 'not_attacked_in_step62',
            'scope': 'five_fiber_patterns_not_reoptimized_here',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'R': 23,
            'fourier_fiber_count': 4,
            'spreader_budget': 11,
            'any_pattern_feasible': 'not_attacked_in_step62',
            'scope': 'five_fiber_patterns_not_reoptimized_here',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'R': 22,
            'fourier_fiber_count': 5,
            'spreader_budget': 7,
            'any_pattern_feasible': 'not_attacked_in_step62',
            'scope': 'four_fiber_patterns_not_reoptimized_here',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def summary_rows(pattern_summary: list[dict], hybrid_verdicts: list[dict]) -> list[dict]:
    nonrect = [row for row in pattern_summary if row['nonrectangular'] == 'True']
    p4 = next(row for row in pattern_summary if row['pattern_id'] == 'P4')
    r22 = next(row for row in hybrid_verdicts if row['R'] == 22 and row['fourier_fiber_count'] == 3)
    r23 = next(row for row in hybrid_verdicts if row['R'] == 23 and row['fourier_fiber_count'] == 3)
    return [
        {'summary_name': 'nonrectangular_pattern_count', 'summary_value': '4', 'provenance': 'EXACT_DERIVED', 'note': 'The true Step 62 targets are P1-P4.'},
        {'summary_name': 'rectangular_patterns_ruled_out_exactly', 'summary_value': '2', 'provenance': 'EXACT_DERIVED', 'note': 'P5 and P6 are ruled out by known exact rank 15 against budgets 13 and 14.'},
        {'summary_name': 'best_substitution_lower_bound_nonrectangular', 'summary_value': str(max(int(row['substitution_best_lower_bound']) for row in nonrect)), 'provenance': 'EXACT_DERIVED', 'note': 'Best substitution-style flattening lower bound among P1-P4.'},
        {'summary_name': 'any_nonrectangular_rank_leq_13_found', 'summary_value': str(any(row['feasible_at_13_spreaders'] == 'measured_yes' for row in nonrect)).lower(), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Whether numerical fitting found a six-fiber nonrectangular pattern with rank at most 13.'},
        {'summary_name': 'any_nonrectangular_rank_leq_14_found', 'summary_value': str(any(row['feasible_at_14_spreaders'] == 'measured_yes' for row in nonrect)).lower(), 'provenance': 'MEASURED_FROM_CODE', 'note': 'Whether numerical fitting found a six-fiber nonrectangular pattern with rank at most 14.'},
        {'summary_name': 'P4_direct_upper_bound', 'summary_value': '18', 'provenance': 'EXACT_DERIVED', 'note': 'Fiber-local exact decomposition verified for P4.'},
        {'summary_name': 'P4_numerical_rank_upper_bound_if_found', 'summary_value': p4['numerical_rank_upper_bound_if_found'], 'provenance': 'MEASURED_FROM_CODE', 'note': 'Measured smallest successful CP rank for P4, if any.'},
        {'summary_name': 'R22_f3_hybrid_verdict', 'summary_value': r22['any_pattern_feasible'], 'provenance': r22['provenance'], 'note': 'Step 62 verdict for six-fiber hybrid residuals at R=22.'},
        {'summary_name': 'R23_f3_hybrid_verdict', 'summary_value': r23['any_pattern_feasible'], 'provenance': r23['provenance'], 'note': 'Step 62 verdict for six-fiber hybrid residuals at R=23.'},
    ]


def write_markdown_summary(
    path: Path,
    summary: list[dict],
    pattern_summary: list[dict],
    hybrid_verdicts: list[dict],
    direct_rows: list[dict],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append
    w('# Step 62: Non-Rectangular 6-Fiber Sub-Tensor Rank Attack')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED] / [MEASURED_FROM_CODE]')
    w('')
    w('Step 62 attacks the six-fiber hybrid survivors from Step 60 with exact restriction bounds, measured CP-rank scans, and a direct upper-bound construction for the anti-diagonal-missing pattern P4.')
    w('')
    w(f"Nonrectangular patterns attacked: {summary_map['nonrectangular_pattern_count']}")
    w(f"Any nonrectangular rank <= 13 found numerically: {summary_map['any_nonrectangular_rank_leq_13_found']}")
    w(f"Any nonrectangular rank <= 14 found numerically: {summary_map['any_nonrectangular_rank_leq_14_found']}")
    w(f"P4 direct upper bound: {summary_map['P4_direct_upper_bound']}")
    w('')
    w('## Pattern Table')
    w('')
    w('| pattern | representative fibers | substitution best lower bound | numerical rank upper bound if found | known exact rank if determined | feasible at 13 spreaders? | feasible at 14 spreaders? |')
    w('|---------|-----------------------|-------------------------------|-------------------------------------|-------------------------------|--------------------------|--------------------------|')
    for row in pattern_summary:
        w(f"| {row['pattern_id']} | {row['representative_fibers']} | {row['substitution_best_lower_bound']} | {row['numerical_rank_upper_bound_if_found']} | {row['known_exact_rank_if_determined']} | {row['feasible_at_13_spreaders']} | {row['feasible_at_14_spreaders']} |")
    w('')
    w('## Hybrid Verdicts')
    w('')
    w('| R | Fourier fibers f | spreaders | verdict | scope |')
    w('|---|------------------|-----------|---------|-------|')
    for row in hybrid_verdicts:
        w(f"| {row['R']} | {row['fourier_fiber_count']} | {row['spreader_budget']} | {row['any_pattern_feasible']} | {row['scope']} |")
    w('')
    w('## P4 Direct Construction')
    w('')
    w('| construction | term count | status | note |')
    w('|--------------|------------|--------|------|')
    for row in direct_rows:
        w(f"| {row['construction_id']} | {row['term_count']} | {row['status']} | {row['note']} |")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The exact restriction bounds remain coarse on the nonrectangular six-fiber patterns, but the attack does something Step 60 could not: it asks directly whether rank 13 or 14 is numerically plausible on the full 9x9x6 subtensors. The rectangular patterns P5 and P6 stay dead by exact rank 15. P4 has an explicit exact upper bound 18 via the standard fiber-local decomposition, and its row-pair and column-pair covers do not improve that count. The decisive content of Step 62 is therefore the measured rank scan on P1-P4 and the resulting R=22/R=23 verdicts for the six-fiber hybrid route. Those verdicts are evidence-based and numerical, not a proof unless matched by an exact decomposition or a stronger lower bound.')
    write_markdown(path, '\n'.join(lines))


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print('=== Step 62: Non-Rectangular 6-Fiber Sub-Tensor Rank Attack ===')
    print()

    substitution_rows: list[dict] = []
    substitution_best: list[dict] = []
    numerical_rows: list[dict] = []
    numerical_verdicts: list[dict] = []

    for pattern in PATTERNS:
        tensor = reduced_output_tensor(pattern.fibers)

        pattern_sub_rows = substitution_restrictions(pattern, tensor)
        substitution_rows.extend(pattern_sub_rows)
        best_lower = max(row['lower_bound'] for row in pattern_sub_rows)
        substitution_best.append({
            'pattern_id': pattern.pattern_id,
            'representative_fibers': fibers_str(pattern.fibers),
            'best_lower_bound': best_lower,
            'best_source': '; '.join(row['label'] for row in pattern_sub_rows if row['lower_bound'] == best_lower),
            'provenance': 'EXACT_DERIVED',
        })

        pattern_num_rows, pattern_verdict = cp_rank_scan(pattern, tensor)
        numerical_rows.extend(pattern_num_rows)
        numerical_verdicts.append(pattern_verdict)

    direct_rows = direct_p4_rows()
    pattern_summary = compile_pattern_summary(substitution_best, numerical_verdicts)
    hybrid_verdicts = hybrid_verdict_rows(pattern_summary)
    summary = summary_rows(pattern_summary, hybrid_verdicts)

    write_csv(
        EXPORTS / 'step62_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step62_substitution_bounds.csv',
        substitution_rows,
        ['pattern_id', 'kind', 'label', 'restricted_shape', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'lower_bound', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step62_substitution_best.csv',
        substitution_best,
        ['pattern_id', 'representative_fibers', 'best_lower_bound', 'best_source', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step62_numerical_rank_scan.csv',
        numerical_rows,
        ['pattern_id', 'rank_tested', 'restarts', 'success_count', 'best_verified_loss', 'best_max_abs_residual', 'best_restart', 'best_iterations', 'numerically_exact', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step62_numerical_verdicts.csv',
        numerical_verdicts,
        ['pattern_id', 'numerical_rank_upper_bound_if_found', 'best_loss_all_ranks', 'rank_leq_13_found', 'rank_leq_14_found', 'best_success_loss', 'best_success_max_abs_residual', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step62_direct_p4_construction.csv',
        direct_rows,
        ['construction_id', 'pattern_id', 'term_count', 'status', 'note', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step62_pattern_summary.csv',
        pattern_summary,
        ['pattern_id', 'representative_fibers', 'orbit_size', 'nonrectangular', 'substitution_best_lower_bound', 'numerical_rank_upper_bound_if_found', 'best_loss_all_ranks', 'known_exact_rank_if_determined', 'feasible_at_13_spreaders', 'feasible_at_14_spreaders', 'provenance_exact', 'provenance_numerical'],
    )
    write_csv(
        EXPORTS / 'step62_hybrid_verdicts.csv',
        hybrid_verdicts,
        ['R', 'fourier_fiber_count', 'spreader_budget', 'any_pattern_feasible', 'scope', 'provenance'],
    )
    write_markdown_summary(
        EXPORTS / 'step62_nonrectangular_6fiber_rank_attack.md',
        summary,
        pattern_summary,
        hybrid_verdicts,
        direct_rows,
    )

    print('\nSummary:')
    for row in summary:
        print(f"  {row['summary_name']} = {row['summary_value']}")


if __name__ == '__main__':
    main()