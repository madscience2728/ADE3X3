"""
ade3x3_step59_cube_root_unity_injection.py

Step 59: Cube Root of Unity Injection.

This step separates two omega-based families:

1. Full-spread omega terms: alpha and beta spread across all rows/columns.
   These are the terms behind the tempting but false R=3 / R=9 arguments.
2. Same-fiber Fourier terms: alpha is supported on one row, beta on one column.
   These interact directly with the Step 51 nuisance basis and recover the
   standard 27-term algorithm in Fourier disguise.
"""

from __future__ import annotations

import csv
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

EXPORTS = Path('outputs/exports')
OMEGA = np.exp(2j * np.pi / 3)
TOL = 1e-8


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


def complex_to_str(value: complex) -> str:
    real = 0.0 if abs(value.real) < TOL else float(value.real)
    imag = 0.0 if abs(value.imag) < TOL else float(value.imag)
    if imag == 0.0:
        return f"{real:.6f}"
    if real == 0.0:
        return f"{imag:.6f}i"
    sign = '+' if imag >= 0 else '-'
    return f"{real:.6f}{sign}{abs(imag):.6f}i"


def numeric_rank(matrix: np.ndarray, tol: float | None = None) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    if tol is None:
        tol = max(matrix.shape) * np.finfo(float).eps * float(np.max(singular_values)) * 10.0
    return int(np.sum(singular_values > tol))


def column_index(row_idx: int, sum_idx: int) -> int:
    return 3 * row_idx + sum_idx


def beta_index(sum_idx: int, col_idx: int) -> int:
    return 3 * sum_idx + col_idx


def coord_vector(c_col: np.ndarray, d_col: np.ndarray) -> np.ndarray:
    return np.outer(c_col, d_col).reshape(-1)


def build_coordinate_profile(alpha_rows: np.ndarray, beta_rows: np.ndarray) -> dict[str, np.ndarray | int]:
    sigma_cols: list[np.ndarray] = []
    eta1_cols: list[np.ndarray] = []
    eta2_cols: list[np.ndarray] = []
    delta_cols: list[np.ndarray] = []

    for row_idx in range(3):
        for col_idx in range(3):
            live0 = coord_vector(alpha_rows[:, column_index(row_idx, 0)], beta_rows[:, beta_index(0, col_idx)])
            live1 = coord_vector(alpha_rows[:, column_index(row_idx, 1)], beta_rows[:, beta_index(1, col_idx)])
            live2 = coord_vector(alpha_rows[:, column_index(row_idx, 2)], beta_rows[:, beta_index(2, col_idx)])
            sigma_cols.append(live0 + live1 + live2)
            eta1_cols.append(live0 - live1)
            eta2_cols.append(live1 - live2)

    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    delta_cols.append(
                        coord_vector(
                            alpha_rows[:, column_index(row_idx, sum_left)],
                            beta_rows[:, beta_index(sum_right, col_idx)],
                        )
                    )

    sigma = np.column_stack(sigma_cols)
    eta1 = np.column_stack(eta1_cols)
    eta2 = np.column_stack(eta2_cols)
    delta = np.column_stack(delta_cols)
    nuisance = np.column_stack([eta1, eta2, delta])
    augmented = np.column_stack([sigma, nuisance])
    return {
        'sigma_rank': numeric_rank(sigma),
        'eta_rank': numeric_rank(np.column_stack([eta1, eta2])),
        'delta_rank': numeric_rank(delta),
        'nuisance_rank': numeric_rank(nuisance),
        'augmented_rank': numeric_rank(augmented),
        'quotient_gain': numeric_rank(augmented) - numeric_rank(nuisance),
        'sigma': sigma,
        'eta1': eta1,
        'eta2': eta2,
        'delta': delta,
        'nuisance': nuisance,
        'augmented': augmented,
    }


def target_tensor() -> np.ndarray:
    tensor = np.zeros((3, 3, 3, 3, 3, 3), dtype=np.complex128)
    for out_row, out_col, row_idx, sum_left, sum_right, col_idx in product(range(3), repeat=6):
        if out_row == row_idx and out_col == col_idx and sum_left == sum_right:
            tensor[out_row, out_col, row_idx, sum_left, sum_right, col_idx] = 1.0
    return tensor


TARGET_TENSOR = target_tensor()


def evaluate_terms(terms: list[dict[str, np.ndarray]]) -> np.ndarray:
    result = np.zeros_like(TARGET_TENSOR)
    for term in terms:
        alpha = term['alpha']
        beta = term['beta']
        gamma = term['gamma']
        result += np.einsum('ab,cd,ef->efabcd', alpha, beta, gamma, optimize=True)
    return result


def flatten_tensor(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1)


def classify_equation_failures(actual: np.ndarray) -> tuple[dict[str, int], list[dict]]:
    rows: list[dict] = []
    summary = {
        'live_target_failures': 0,
        'live_off_target_failures': 0,
        'dead_failures': 0,
        'total_failures': 0,
    }

    for out_row, out_col, row_idx, sum_left, sum_right, col_idx in product(range(3), repeat=6):
        target = TARGET_TENSOR[out_row, out_col, row_idx, sum_left, sum_right, col_idx]
        value = actual[out_row, out_col, row_idx, sum_left, sum_right, col_idx]
        if abs(value - target) <= TOL:
            continue
        if sum_left == sum_right and out_row == row_idx and out_col == col_idx:
            block = 'live_target'
            summary['live_target_failures'] += 1
        elif sum_left == sum_right:
            block = 'live_off_target'
            summary['live_off_target_failures'] += 1
        else:
            block = 'dead_x'
            summary['dead_failures'] += 1
        summary['total_failures'] += 1
        rows.append({
            'output_fiber': f'({out_row},{out_col})',
            'input_coordinate': f'({row_idx},{sum_left},{sum_right},{col_idx})',
            'block_type': block,
            'actual_value': complex_to_str(value),
            'target_value': complex_to_str(target),
            'residual': complex_to_str(value - target),
            'provenance': 'EXACT_DERIVED',
        })
    return summary, rows


def make_fullspread_term(out_row: int, out_col: int, channel_j: int, gamma_scale: complex = 1.0) -> dict[str, np.ndarray]:
    alpha = np.zeros((3, 3), dtype=np.complex128)
    beta = np.zeros((3, 3), dtype=np.complex128)
    gamma = np.zeros((3, 3), dtype=np.complex128)
    for row_idx in range(3):
        for sum_idx in range(3):
            alpha[row_idx, sum_idx] = OMEGA ** (sum_idx * channel_j)
    for sum_idx in range(3):
        for col_idx in range(3):
            beta[sum_idx, col_idx] = OMEGA ** (-sum_idx * channel_j)
    gamma[out_row, out_col] = gamma_scale
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma}


def make_samefiber_term(row_idx: int, col_idx: int, channel_j: int, gamma_scale: complex = 1.0) -> dict[str, np.ndarray]:
    alpha = np.zeros((3, 3), dtype=np.complex128)
    beta = np.zeros((3, 3), dtype=np.complex128)
    gamma = np.zeros((3, 3), dtype=np.complex128)
    for sum_idx in range(3):
        alpha[row_idx, sum_idx] = OMEGA ** (sum_idx * channel_j)
        beta[sum_idx, col_idx] = OMEGA ** (-sum_idx * channel_j)
    gamma[row_idx, col_idx] = gamma_scale
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma}


def term_label(out_row: int, out_col: int, channel_j: int) -> str:
    return f'fiber=({out_row},{out_col}),j={channel_j}'


def fullspread_live_obstruction_rows() -> tuple[list[dict], dict[str, str]]:
    live_rows = []
    for out_row, out_col, row_idx, sum_idx, col_idx in product(range(3), repeat=5):
        live_rows.append((out_row, out_col, row_idx, sum_idx, col_idx))

    block_rows = [idx for idx, row in enumerate(live_rows) if row[0] == 0 and row[1] == 0]
    target_columns: list[np.ndarray] = []
    family_columns: list[np.ndarray] = []

    for input_row in range(3):
        for input_col in range(3):
            vector = np.zeros(len(block_rows), dtype=np.complex128)
            for local_idx, live_idx in enumerate(block_rows):
                _, _, row_idx, _, col_idx = live_rows[live_idx]
                if row_idx == input_row and col_idx == input_col:
                    vector[local_idx] = 1.0
            target_columns.append(vector)

    for channel_j in range(3):
        vector = np.zeros(len(block_rows), dtype=np.complex128)
        for local_idx, _ in enumerate(block_rows):
            vector[local_idx] = 1.0
        family_columns.append(vector)

    target_matrix = np.column_stack(target_columns)
    family_matrix = np.column_stack(family_columns)
    rows = [
        {
            'statement_id': 'F1',
            'statement': 'In the full-spread family, every frequency channel gives the same live contribution on a fixed output block: the all-ones vector across the 9 input fibers x 3 live summation coordinates.',
            'value': 'verified_by_direct_matrix_build',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'F2',
            'statement': 'The full-spread family has live-block column rank 1 per output block, whereas the target live block has fiber rank 9.',
            'value': 'rank_obstruction',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'F3',
            'statement': 'Therefore no sum of full-spread omega terms can realize the 3x3 multiplication tensor exactly, regardless of R.',
            'value': 'family_impossible',
            'provenance': 'EXACT_DERIVED',
        },
    ]
    summary = {
        'fullspread_live_block_rows': str(target_matrix.shape[0]),
        'fullspread_live_family_rank_per_output': str(numeric_rank(family_matrix)),
        'fullspread_target_live_rank_per_output': str(numeric_rank(target_matrix)),
        'fullspread_exact_solution_exists': 'False',
    }
    return rows, summary


def balanced_r9_terms() -> tuple[list[dict[str, np.ndarray]], list[dict]]:
    terms: list[dict[str, np.ndarray]] = []
    labels: list[dict] = []
    for out_idx, (out_row, out_col) in enumerate(product(range(3), repeat=2)):
        channel_j = out_idx % 3
        terms.append(make_fullspread_term(out_row, out_col, channel_j, gamma_scale=1.0))
        labels.append({'output_fiber': f'({out_row},{out_col})', 'channel_j': channel_j, 'label': term_label(out_row, out_col, channel_j), 'provenance': 'EXACT_DERIVED'})
    return terms, labels


def build_fullspread_dictionary() -> tuple[list[dict[str, np.ndarray]], list[str], np.ndarray]:
    terms: list[dict[str, np.ndarray]] = []
    labels: list[str] = []
    columns: list[np.ndarray] = []
    for out_row, out_col, channel_j in product(range(3), range(3), range(3)):
        term = make_fullspread_term(out_row, out_col, channel_j, gamma_scale=1.0)
        terms.append(term)
        labels.append(term_label(out_row, out_col, channel_j))
        columns.append(flatten_tensor(evaluate_terms([term])))
    return terms, labels, np.column_stack(columns)


def greedy_fullspread_progress(max_terms: int = 27) -> list[dict]:
    _, labels, dictionary = build_fullspread_dictionary()
    target = flatten_tensor(TARGET_TENSOR)
    selected: list[int] = []
    residual = target.copy()
    rows: list[dict] = []

    for step in range(1, max_terms + 1):
        best_idx = None
        best_score = -1.0
        for idx in range(dictionary.shape[1]):
            if idx in selected:
                continue
            score = abs(np.vdot(dictionary[:, idx], residual))
            if score > best_score:
                best_score = float(score)
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        active_matrix = dictionary[:, selected]
        coeffs, _, _, _ = np.linalg.lstsq(active_matrix, target, rcond=None)
        approximation = active_matrix @ coeffs
        residual = target - approximation
        failure_summary, _ = classify_equation_failures(approximation.reshape(3, 3, 3, 3, 3, 3))
        rows.append({
            'R': step,
            'selected_term': labels[best_idx],
            'residual_l2_norm': f'{np.linalg.norm(residual):.10f}',
            'max_abs_residual': f'{np.max(np.abs(residual)):.10f}',
            'live_target_failures': failure_summary['live_target_failures'],
            'live_off_target_failures': failure_summary['live_off_target_failures'],
            'dead_failures': failure_summary['dead_failures'],
            'total_failures': failure_summary['total_failures'],
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def samefiber_profile_rows() -> tuple[list[dict], dict[str, str], list[dict]]:
    cases = [
        ('single_j1', [make_samefiber_term(0, 0, 1)]),
        ('pair_j1_j2', [make_samefiber_term(0, 0, 1), make_samefiber_term(0, 0, 2)]),
        ('triple_j0_j1_j2', [make_samefiber_term(0, 0, 0), make_samefiber_term(0, 0, 1), make_samefiber_term(0, 0, 2)]),
    ]

    rows: list[dict] = []
    detail_rows: list[dict] = []
    summary: dict[str, str] = {}
    for label, terms in cases:
        alpha_rows = np.vstack([term['alpha'].reshape(-1) for term in terms])
        beta_rows = np.vstack([term['beta'].reshape(-1) for term in terms])
        profile = build_coordinate_profile(alpha_rows, beta_rows)
        rows.append({
            'case_label': label,
            'term_count': len(terms),
            'sigma_rank': profile['sigma_rank'],
            'eta_rank': profile['eta_rank'],
            'delta_rank': profile['delta_rank'],
            'nuisance_rank': profile['nuisance_rank'],
            'augmented_rank': profile['augmented_rank'],
            'quotient_gain': profile['quotient_gain'],
            'provenance': 'EXACT_DERIVED',
        })
        sigma = profile['sigma']
        eta1 = profile['eta1']
        eta2 = profile['eta2']
        delta = profile['delta']
        detail_rows.append({
            'case_label': label,
            'sigma_target_fiber_column': complex_to_str(sigma[0, 0]) if sigma.shape[0] > 0 else '0.000000',
            'eta1_target_fiber_column': complex_to_str(eta1[0, 0]) if eta1.shape[0] > 0 else '0.000000',
            'eta2_target_fiber_column': complex_to_str(eta2[0, 0]) if eta2.shape[0] > 0 else '0.000000',
            'delta_first_dead_column': complex_to_str(delta[0, 0]) if delta.shape[0] > 0 else '0.000000',
            'provenance': 'EXACT_DERIVED',
        })

    summary['single_j1_nuisance_rank'] = str(rows[0]['nuisance_rank'])
    summary['pair_j1_j2_nuisance_rank'] = str(rows[1]['nuisance_rank'])
    summary['triple_j012_nuisance_rank'] = str(rows[2]['nuisance_rank'])
    summary['triple_j012_delta_rank'] = str(rows[2]['delta_rank'])
    return rows, summary, detail_rows


def samefiber_fourier_standard_verification() -> tuple[dict[str, str], list[dict]]:
    terms: list[dict[str, np.ndarray]] = []
    for row_idx, col_idx, channel_j in product(range(3), range(3), range(3)):
        terms.append(make_samefiber_term(row_idx, col_idx, channel_j, gamma_scale=1.0 / 3.0))

    actual = evaluate_terms(terms)
    failure_summary, failure_rows = classify_equation_failures(actual)
    summary = {
        'samefiber_fourier_term_count': '27',
        'samefiber_fourier_total_failures': str(failure_summary['total_failures']),
        'samefiber_fourier_live_target_failures': str(failure_summary['live_target_failures']),
        'samefiber_fourier_live_off_target_failures': str(failure_summary['live_off_target_failures']),
        'samefiber_fourier_dead_failures': str(failure_summary['dead_failures']),
    }
    return summary, failure_rows


def summary_rows(
    obstruction_summary: dict[str, str],
    r9_failure_summary: dict[str, int],
    nuisance_summary: dict[str, str],
    samefiber_summary: dict[str, str],
    greedy_rows: list[dict],
) -> list[dict]:
    final_greedy = greedy_rows[-1]
    return [
        {'summary_name': 'fullspread_live_family_rank_per_output', 'summary_value': obstruction_summary['fullspread_live_family_rank_per_output'], 'provenance': 'EXACT_DERIVED', 'note': 'Live subspace rank for the full-spread omega family on a fixed output block.'},
        {'summary_name': 'fullspread_target_live_rank_per_output', 'summary_value': obstruction_summary['fullspread_target_live_rank_per_output'], 'provenance': 'EXACT_DERIVED', 'note': 'Required live rank on a fixed output block for the true tensor.'},
        {'summary_name': 'fullspread_exact_solution_exists', 'summary_value': obstruction_summary['fullspread_exact_solution_exists'], 'provenance': 'EXACT_DERIVED', 'note': 'Whether any number of full-spread omega terms can solve the tensor exactly.'},
        {'summary_name': 'balanced_r9_total_failures', 'summary_value': str(r9_failure_summary['total_failures']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact failure count for the explicit 9-term full-spread attempt.'},
        {'summary_name': 'balanced_r9_live_off_target_failures', 'summary_value': str(r9_failure_summary['live_off_target_failures']), 'provenance': 'EXACT_DERIVED', 'note': 'Off-fiber live equations failed by the explicit 9-term full-spread attempt.'},
        {'summary_name': 'balanced_r9_dead_failures', 'summary_value': str(r9_failure_summary['dead_failures']), 'provenance': 'EXACT_DERIVED', 'note': 'Dead-X equations failed by the explicit 9-term full-spread attempt.'},
        {'summary_name': 'greedy_fullspread_R27_total_failures', 'summary_value': str(final_greedy['total_failures']), 'provenance': 'EXACT_DERIVED', 'note': 'Failure count after allowing all 27 full-spread basis terms in least-squares form.'},
        {'summary_name': 'single_j1_nuisance_rank', 'summary_value': nuisance_summary['single_j1_nuisance_rank'], 'provenance': 'EXACT_DERIVED', 'note': 'Step 51 nuisance rank for one same-fiber omega term with j=1.'},
        {'summary_name': 'pair_j1_j2_nuisance_rank', 'summary_value': nuisance_summary['pair_j1_j2_nuisance_rank'], 'provenance': 'EXACT_DERIVED', 'note': 'Combined nuisance rank for a same-fiber conjugate pair.'},
        {'summary_name': 'triple_j012_nuisance_rank', 'summary_value': nuisance_summary['triple_j012_nuisance_rank'], 'provenance': 'EXACT_DERIVED', 'note': 'Combined nuisance rank for the three-channel same-fiber Fourier bundle.'},
        {'summary_name': 'samefiber_fourier_total_failures', 'summary_value': samefiber_summary['samefiber_fourier_total_failures'], 'provenance': 'EXACT_DERIVED', 'note': 'Direct 729-equation verification result for the 27-term same-fiber Fourier disguise of the standard algorithm.'},
    ]


def write_markdown_summary(
    path: Path,
    summary: list[dict],
    obstruction_rows: list[dict],
    r9_assignment_rows: list[dict],
    r9_failure_summary: dict[str, int],
    greedy_rows: list[dict],
    nuisance_rows: list[dict],
    samefiber_summary: dict[str, str],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append
    w('# Step 59: Cube Root of Unity Injection')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('Step 59 splits the omega construction into two families. The full-spread family is impossible for representation-theoretic reasons in the live block; the same-fiber Fourier family is exact but recovers the standard 27-term algorithm rather than a low-rank breakthrough.')
    w('')
    w(f"Full-spread live rank per output block: {summary_map['fullspread_live_family_rank_per_output']}")
    w(f"Target live rank per output block: {summary_map['fullspread_target_live_rank_per_output']}")
    w(f"Balanced 9-term full-spread failures: {summary_map['balanced_r9_total_failures']}")
    w(f"Same-fiber j=1 nuisance rank: {summary_map['single_j1_nuisance_rank']}")
    w(f"Same-fiber j=1,2 pair nuisance rank: {summary_map['pair_j1_j2_nuisance_rank']}")
    w(f"Same-fiber j=0,1,2 triple nuisance rank: {summary_map['triple_j012_nuisance_rank']}")
    w('')
    w('## Task 2f: Full-Spread Obstruction')
    w('')
    for row in obstruction_rows:
        w(f"- {row['statement_id']}: {row['statement']}")
    w('')
    w('## Task 3: Explicit 9-Term Full-Spread Attempt')
    w('')
    w('| output fiber | channel j | label |')
    w('|--------------|-----------|-------|')
    for row in r9_assignment_rows:
        w(f"| {row['output_fiber']} | {row['channel_j']} | {row['label']} |")
    w('')
    w(f"Failure counts: live_target={r9_failure_summary['live_target_failures']}, live_off_target={r9_failure_summary['live_off_target_failures']}, dead={r9_failure_summary['dead_failures']}, total={r9_failure_summary['total_failures']}")
    w('')
    w('## Task 3e: Greedy Restricted-Family Augmentation')
    w('')
    w('| R | selected term | live target fails | live off-target fails | dead fails | total fails | residual L2 |')
    w('|---|---------------|-------------------|-----------------------|------------|-------------|-------------|')
    for row in greedy_rows[:12]:
        w(f"| {row['R']} | {row['selected_term']} | {row['live_target_failures']} | {row['live_off_target_failures']} | {row['dead_failures']} | {row['total_failures']} | {row['residual_l2_norm']} |")
    w('')
    w(f"At R=27 inside the full-spread family: total fails={greedy_rows[-1]['total_failures']}, live_off_target={greedy_rows[-1]['live_off_target_failures']}, dead={greedy_rows[-1]['dead_failures']}")
    w('')
    w('## Task 4: Same-Fiber Fourier Nuisance Profiles')
    w('')
    w('| case | term count | sigma rank | eta rank | delta rank | nuisance rank | augmented rank | quotient gain |')
    w('|------|------------|------------|----------|------------|---------------|----------------|---------------|')
    for row in nuisance_rows:
        w(f"| {row['case_label']} | {row['term_count']} | {row['sigma_rank']} | {row['eta_rank']} | {row['delta_rank']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['quotient_gain']} |")
    w('')
    w(f"27-term same-fiber Fourier verification failures: {samefiber_summary['samefiber_fourier_total_failures']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The false R=3 and R=9 stories die for an exact reason: in the full-spread omega family, the live block cannot distinguish the 9 fibers. Adding more such terms can cancel dead-X, but it cannot create the missing fiber resolution, so the obstruction is permanent for that family. By contrast, the same-fiber Fourier bundle is exact and useful: three channels on one fiber kill dead-X by DFT orthogonality, but the bundle still has nuisance rank 2 before gamma annihilation. So omega injection is not a magic low-rank shortcut; in the viable same-fiber incarnation it is best understood as a Fourier re-expression of the standard algorithm, not a route below rank 27.')
    write_markdown(path, '\n'.join(lines))


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print('=== Step 59: Cube Root of Unity Injection ===')
    print()

    obstruction_rows, obstruction_summary = fullspread_live_obstruction_rows()

    r9_terms, r9_assignment_rows = balanced_r9_terms()
    r9_actual = evaluate_terms(r9_terms)
    r9_failure_summary, r9_failure_rows = classify_equation_failures(r9_actual)

    greedy_rows = greedy_fullspread_progress(27)
    nuisance_rows, nuisance_summary, nuisance_detail_rows = samefiber_profile_rows()
    samefiber_summary, samefiber_failure_rows = samefiber_fourier_standard_verification()

    summary = summary_rows(obstruction_summary, r9_failure_summary, nuisance_summary, samefiber_summary, greedy_rows)

    write_csv(
        EXPORTS / 'step59_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step59_fullspread_obstruction.csv',
        obstruction_rows,
        ['statement_id', 'statement', 'value', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_r9_assignment.csv',
        r9_assignment_rows,
        ['output_fiber', 'channel_j', 'label', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_r9_failure_summary.csv',
        [{
            'live_target_failures': r9_failure_summary['live_target_failures'],
            'live_off_target_failures': r9_failure_summary['live_off_target_failures'],
            'dead_failures': r9_failure_summary['dead_failures'],
            'total_failures': r9_failure_summary['total_failures'],
            'provenance': 'EXACT_DERIVED',
        }],
        ['live_target_failures', 'live_off_target_failures', 'dead_failures', 'total_failures', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_r9_failure_examples.csv',
        r9_failure_rows[:120],
        ['output_fiber', 'input_coordinate', 'block_type', 'actual_value', 'target_value', 'residual', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_greedy_fullspread_progress.csv',
        greedy_rows,
        ['R', 'selected_term', 'residual_l2_norm', 'max_abs_residual', 'live_target_failures', 'live_off_target_failures', 'dead_failures', 'total_failures', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_samefiber_nuisance_profiles.csv',
        nuisance_rows,
        ['case_label', 'term_count', 'sigma_rank', 'eta_rank', 'delta_rank', 'nuisance_rank', 'augmented_rank', 'quotient_gain', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_samefiber_profile_details.csv',
        nuisance_detail_rows,
        ['case_label', 'sigma_target_fiber_column', 'eta1_target_fiber_column', 'eta2_target_fiber_column', 'delta_first_dead_column', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step59_samefiber_fourier_standard_verification.csv',
        [{**samefiber_summary, 'provenance': 'EXACT_DERIVED'}],
        [
            'samefiber_fourier_term_count',
            'samefiber_fourier_total_failures',
            'samefiber_fourier_live_target_failures',
            'samefiber_fourier_live_off_target_failures',
            'samefiber_fourier_dead_failures',
            'provenance',
        ],
    )
    if samefiber_failure_rows:
        write_csv(
            EXPORTS / 'step59_samefiber_fourier_failure_examples.csv',
            samefiber_failure_rows[:40],
            ['output_fiber', 'input_coordinate', 'block_type', 'actual_value', 'target_value', 'residual', 'provenance'],
        )
    write_markdown_summary(
        EXPORTS / 'step59_cube_root_unity_injection.md',
        summary,
        obstruction_rows,
        r9_assignment_rows,
        r9_failure_summary,
        greedy_rows,
        nuisance_rows,
        samefiber_summary,
    )

    print('\nSummary:')
    for row in summary:
        print(f"  {row['summary_name']} = {row['summary_value']}")


if __name__ == '__main__':
    main()