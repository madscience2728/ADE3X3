"""
ade3x3_step51_symbolic_fiber_mode_decomposition.py

Step 51: Symbolic Fiber-Mode Decomposition of the Tensor Equations.

This step replaces numeric restart search with an exact symbolic reformulation.
For each rank-1 term k, the 81 A x B coefficients split into three blocks:

1. Fiber sums: one scalar for each output fiber (r,u)
2. Fiber anisotropy: two independent live-direction differences per fiber
3. Dead-X coordinates: all 54 coordinates with s != t

The full 729 trilinear tensor equations then decompose exactly into:

  81  fiber-sum equations
  162 live-anisotropy equations
  486 dead-X equations

This is a symbolic derivation step, not a search step.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

EXPORTS = Path('outputs/exports')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def a_name(row_idx: int, sum_idx: int) -> str:
    return f"A[{row_idx},{sum_idx}]"


def b_name(sum_idx: int, col_idx: int) -> str:
    return f"B[{sum_idx},{col_idx}]"


def c_name(row_idx: int, col_idx: int) -> str:
    return f"C[{row_idx},{col_idx}]"


def lambda_name(rank_idx: int, row_idx: int, col_idx: int, sum_idx: int) -> str:
    return f"lambda_{rank_idx}[{row_idx},{col_idx},{sum_idx}]"


def fiber_sum_name(rank_idx: int, row_idx: int, col_idx: int) -> str:
    return f"sigma_{rank_idx}[{row_idx},{col_idx}]"


def anisotropy_name(rank_idx: int, row_idx: int, col_idx: int, which: int) -> str:
    return f"eta{which}_{rank_idx}[{row_idx},{col_idx}]"


def dead_name(rank_idx: int, row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> str:
    return f"delta_{rank_idx}[{row_idx},{sum_left},{sum_right},{col_idx}]"


def target_rhs(row_idx: int, col_idx: int, out_row: int, out_col: int) -> int:
    return int(row_idx == out_row and col_idx == out_col)


def build_mode_formulas() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    live_rows: list[dict] = []
    sum_rows: list[dict] = []
    anisotropy_rows: list[dict] = []
    dead_rows: list[dict] = []

    for row_idx in range(3):
        for col_idx in range(3):
            lambdas = [f"a_k[{row_idx},{sum_idx}] * b_k[{sum_idx},{col_idx}]" for sum_idx in range(3)]
            live_rows.append({
                'fiber': f"({row_idx},{col_idx})",
                'live_coordinates': str([f"X[{row_idx},{sum_idx}|{sum_idx},{col_idx}]" for sum_idx in range(3)]),
                'lambda_coordinates': str(lambdas),
                'provenance': 'EXACT_DERIVED',
            })
            sum_rows.append({
                'fiber': f"({row_idx},{col_idx})",
                'fiber_sum_symbol': f"sigma_k[{row_idx},{col_idx}]",
                'formula': (
                    f"sigma_k[{row_idx},{col_idx}] = "
                    f"a_k[{row_idx},0]*b_k[0,{col_idx}] + a_k[{row_idx},1]*b_k[1,{col_idx}] + a_k[{row_idx},2]*b_k[2,{col_idx}]"
                ),
                'target_equation_template': (
                    f"sum_k gamma_k[r',u'] * sigma_k[{row_idx},{col_idx}] = 3*delta_(({row_idx},{col_idx})=(r',u'))"
                ),
                'provenance': 'EXACT_DERIVED',
            })
            anisotropy_rows.append({
                'fiber': f"({row_idx},{col_idx})",
                'anisotropy_symbol': f"eta1_k[{row_idx},{col_idx}]",
                'formula': (
                    f"eta1_k[{row_idx},{col_idx}] = "
                    f"a_k[{row_idx},0]*b_k[0,{col_idx}] - a_k[{row_idx},1]*b_k[1,{col_idx}]"
                ),
                'zero_equation_template': f"sum_k gamma_k[r',u'] * eta1_k[{row_idx},{col_idx}] = 0",
                'provenance': 'EXACT_DERIVED',
            })
            anisotropy_rows.append({
                'fiber': f"({row_idx},{col_idx})",
                'anisotropy_symbol': f"eta2_k[{row_idx},{col_idx}]",
                'formula': (
                    f"eta2_k[{row_idx},{col_idx}] = "
                    f"a_k[{row_idx},1]*b_k[1,{col_idx}] - a_k[{row_idx},2]*b_k[2,{col_idx}]"
                ),
                'zero_equation_template': f"sum_k gamma_k[r',u'] * eta2_k[{row_idx},{col_idx}] = 0",
                'provenance': 'EXACT_DERIVED',
            })

    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    dead_rows.append({
                        'dead_coordinate': f"({row_idx},{sum_left},{sum_right},{col_idx})",
                        'dead_symbol': f"delta_k[{row_idx},{sum_left},{sum_right},{col_idx}]",
                        'formula': f"delta_k[{row_idx},{sum_left},{sum_right},{col_idx}] = a_k[{row_idx},{sum_left}] * b_k[{sum_right},{col_idx}]",
                        'zero_equation_template': f"sum_k gamma_k[r',u'] * delta_k[{row_idx},{sum_left},{sum_right},{col_idx}] = 0",
                        'provenance': 'EXACT_DERIVED',
                    })

    return live_rows, sum_rows, anisotropy_rows, dead_rows


def build_equation_split_rows() -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []

    for out_row in range(3):
        for out_col in range(3):
            for row_idx in range(3):
                for col_idx in range(3):
                    rows.append({
                        'output_c': c_name(out_row, out_col),
                        'block_type': 'fiber_sum',
                        'coordinate': f"({row_idx},{col_idx})",
                        'equation': (
                            f"sum_k gamma_k[{out_row},{out_col}] * sigma_k[{row_idx},{col_idx}] "
                            f"= {3 * target_rhs(row_idx, col_idx, out_row, out_col)}"
                        ),
                        'rhs': 3 * target_rhs(row_idx, col_idx, out_row, out_col),
                        'provenance': 'EXACT_DERIVED',
                    })
                    rows.append({
                        'output_c': c_name(out_row, out_col),
                        'block_type': 'fiber_anisotropy_eta1',
                        'coordinate': f"({row_idx},{col_idx})",
                        'equation': f"sum_k gamma_k[{out_row},{out_col}] * eta1_k[{row_idx},{col_idx}] = 0",
                        'rhs': 0,
                        'provenance': 'EXACT_DERIVED',
                    })
                    rows.append({
                        'output_c': c_name(out_row, out_col),
                        'block_type': 'fiber_anisotropy_eta2',
                        'coordinate': f"({row_idx},{col_idx})",
                        'equation': f"sum_k gamma_k[{out_row},{out_col}] * eta2_k[{row_idx},{col_idx}] = 0",
                        'rhs': 0,
                        'provenance': 'EXACT_DERIVED',
                    })

            for row_idx in range(3):
                for sum_left in range(3):
                    for sum_right in range(3):
                        if sum_left == sum_right:
                            continue
                        for col_idx in range(3):
                            rows.append({
                                'output_c': c_name(out_row, out_col),
                                'block_type': 'dead_x',
                                'coordinate': f"({row_idx},{sum_left},{sum_right},{col_idx})",
                                'equation': (
                                    f"sum_k gamma_k[{out_row},{out_col}] * delta_k[{row_idx},{sum_left},{sum_right},{col_idx}] = 0"
                                ),
                                'rhs': 0,
                                'provenance': 'EXACT_DERIVED',
                            })

    counts = Counter(row['block_type'] for row in rows)
    summary = {
        'fiber_sum_equations': str(counts['fiber_sum']),
        'fiber_anisotropy_equations': str(counts['fiber_anisotropy_eta1'] + counts['fiber_anisotropy_eta2']),
        'dead_x_equations': str(counts['dead_x']),
        'total_equations': str(len(rows)),
    }
    assert len(rows) == 729
    assert counts['fiber_sum'] == 81
    assert counts['fiber_anisotropy_eta1'] + counts['fiber_anisotropy_eta2'] == 162
    assert counts['dead_x'] == 486
    return rows, summary


def build_matrix_form_rows() -> list[dict]:
    rows: list[dict] = []
    rows.append({
        'matrix_name': 'Gamma',
        'shape': '9 x R',
        'entry_definition': 'Gamma[(r\',u\'), k] = gamma_k[r\',u\']',
        'meaning': 'Output weights attached to term k for each C coordinate',
        'provenance': 'EXACT_DERIVED',
    })
    rows.append({
        'matrix_name': 'Sigma',
        'shape': 'R x 9',
        'entry_definition': 'Sigma[k, (r,u)] = sigma_k[r,u]',
        'meaning': 'Fiber-sum coordinates of the A x B rank-1 profile',
        'provenance': 'EXACT_DERIVED',
    })
    rows.append({
        'matrix_name': 'Eta1',
        'shape': 'R x 9',
        'entry_definition': 'Eta1[k, (r,u)] = eta1_k[r,u]',
        'meaning': 'First live-fiber anisotropy coordinate',
        'provenance': 'EXACT_DERIVED',
    })
    rows.append({
        'matrix_name': 'Eta2',
        'shape': 'R x 9',
        'entry_definition': 'Eta2[k, (r,u)] = eta2_k[r,u]',
        'meaning': 'Second live-fiber anisotropy coordinate',
        'provenance': 'EXACT_DERIVED',
    })
    rows.append({
        'matrix_name': 'Delta',
        'shape': 'R x 54',
        'entry_definition': 'Delta[k, (r,s,t,u)] = delta_k[r,s,t,u] for s!=t',
        'meaning': 'Dead-X coordinates',
        'provenance': 'EXACT_DERIVED',
    })
    rows.append({
        'matrix_name': 'Constraint',
        'shape': 'symbolic',
        'entry_definition': 'Gamma * Sigma = 3 I_9, Gamma * Eta1 = 0, Gamma * Eta2 = 0, Gamma * Delta = 0',
        'meaning': 'Exact matrix-form restatement of all 729 tensor equations',
        'provenance': 'EXACT_DERIVED',
    })
    return rows


def standard_algorithm_terms() -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for row_idx in range(3):
        for sum_idx in range(3):
            for col_idx in range(3):
                alpha = [[0 for _ in range(3)] for _ in range(3)]
                beta = [[0 for _ in range(3)] for _ in range(3)]
                gamma = [[0 for _ in range(3)] for _ in range(3)]
                alpha[row_idx][sum_idx] = 1
                beta[sum_idx][col_idx] = 1
                gamma[row_idx][col_idx] = 1
                terms.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'row': row_idx, 'sum': sum_idx, 'col': col_idx})
    return terms


def standard_algorithm_verify() -> tuple[list[dict], dict[str, str]]:
    terms = standard_algorithm_terms()
    rows: list[dict] = []
    fiber_sum_failures = 0
    anisotropy_failures = 0
    dead_failures = 0

    for out_row in range(3):
        for out_col in range(3):
            for row_idx in range(3):
                for col_idx in range(3):
                    sigma_total = 0
                    eta1_total = 0
                    eta2_total = 0
                    for term in terms:
                        gamma_weight = term['gamma'][out_row][out_col]
                        lambda0 = term['alpha'][row_idx][0] * term['beta'][0][col_idx]
                        lambda1 = term['alpha'][row_idx][1] * term['beta'][1][col_idx]
                        lambda2 = term['alpha'][row_idx][2] * term['beta'][2][col_idx]
                        sigma_total += gamma_weight * (lambda0 + lambda1 + lambda2)
                        eta1_total += gamma_weight * (lambda0 - lambda1)
                        eta2_total += gamma_weight * (lambda1 - lambda2)

                    expected_sigma = 3 * target_rhs(row_idx, col_idx, out_row, out_col)
                    if sigma_total != expected_sigma:
                        fiber_sum_failures += 1
                    if eta1_total != 0 or eta2_total != 0:
                        anisotropy_failures += int(eta1_total != 0) + int(eta2_total != 0)

                    rows.append({
                        'output_c': c_name(out_row, out_col),
                        'fiber': f"({row_idx},{col_idx})",
                        'fiber_sum_total': sigma_total,
                        'fiber_sum_expected': expected_sigma,
                        'eta1_total': eta1_total,
                        'eta2_total': eta2_total,
                        'status': 'PASS' if sigma_total == expected_sigma and eta1_total == 0 and eta2_total == 0 else 'FAIL',
                        'provenance': 'EXACT_DERIVED',
                    })

            for row_idx in range(3):
                for sum_left in range(3):
                    for sum_right in range(3):
                        if sum_left == sum_right:
                            continue
                        for col_idx in range(3):
                            dead_total = 0
                            for term in terms:
                                gamma_weight = term['gamma'][out_row][out_col]
                                dead_total += gamma_weight * term['alpha'][row_idx][sum_left] * term['beta'][sum_right][col_idx]
                            if dead_total != 0:
                                dead_failures += 1

    summary = {
        'standard_fiber_sum_failures': str(fiber_sum_failures),
        'standard_live_anisotropy_failures': str(anisotropy_failures),
        'standard_dead_x_failures': str(dead_failures),
    }
    assert fiber_sum_failures == 0
    assert anisotropy_failures == 0
    assert dead_failures == 0
    return rows, summary


def write_markdown_summary(
    path: Path,
    live_rows: list[dict],
    sum_rows: list[dict],
    anisotropy_rows: list[dict],
    dead_rows: list[dict],
    equation_rows: list[dict],
    matrix_rows: list[dict],
    standard_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append

    w('# Step 51: Symbolic Fiber-Mode Decomposition')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('This step rewrites the 729 tensor equations as an exact 81 + 162 + 486 block decomposition.')
    w('')
    w('## Exact Block Counts')
    w('')
    w(f"- Fiber-sum equations: {summary['fiber_sum_equations']}")
    w(f"- Live-anisotropy equations: {summary['fiber_anisotropy_equations']}")
    w(f"- Dead-X equations: {summary['dead_x_equations']}")
    w(f"- Total: {summary['total_equations']}")
    w('')
    w('## Symbolic Coordinates per Fiber')
    w('')
    w('| fiber | sigma formula | eta1 formula | eta2 formula |')
    w('|-------|---------------|--------------|--------------|')
    for sum_row, eta1_row, eta2_row in zip(sum_rows, anisotropy_rows[0::2], anisotropy_rows[1::2], strict=True):
        w(f"| {sum_row['fiber']} | {sum_row['formula']} | {eta1_row['formula']} | {eta2_row['formula']} |")
    w('')
    w('## Matrix Form')
    w('')
    w('| matrix | shape | definition | meaning |')
    w('|--------|-------|------------|---------|')
    for row in matrix_rows:
        w(f"| {row['matrix_name']} | {row['shape']} | {row['entry_definition']} | {row['meaning']} |")
    w('')
    w('The exact tensor system becomes:')
    w('- Gamma * Sigma = 3 I_9')
    w('- Gamma * Eta1 = 0')
    w('- Gamma * Eta2 = 0')
    w('- Gamma * Delta = 0')
    w('')
    w('## Standard 27-Term Verification')
    w('')
    w(f"- Fiber-sum failures: {summary['standard_fiber_sum_failures']}")
    w(f"- Live-anisotropy failures: {summary['standard_live_anisotropy_failures']}")
    w(f"- Dead-X failures: {summary['standard_dead_x_failures']}")
    w('')
    w('| output_c | fiber | fiber_sum_total | fiber_sum_expected | eta1_total | eta2_total | status |')
    w('|----------|-------|-----------------|--------------------|------------|------------|--------|')
    for row in standard_rows[:18]:
        w(f"| {row['output_c']} | {row['fiber']} | {row['fiber_sum_total']} | {row['fiber_sum_expected']} | {row['eta1_total']} | {row['eta2_total']} | {row['status']} |")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('This derivation isolates what any exact algorithm must do symbolically. The target tensor lives only in the 9-dimensional fiber-sum block, while every candidate rank-1 term also produces live anisotropy and dead-X mass that must cancel exactly after gamma weighting. That is a structured elimination problem, not a random search problem.')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 51: Symbolic Fiber-Mode Decomposition ===')
    print()

    print('Building symbolic fiber-mode formulas...')
    live_rows, sum_rows, anisotropy_rows, dead_rows = build_mode_formulas()
    print(f"  fibers={len(sum_rows)} anisotropy coordinates={len(anisotropy_rows)} dead coordinates={len(dead_rows)}")
    print()

    print('Building the exact 729-equation block split...')
    equation_rows, equation_summary = build_equation_split_rows()
    print(
        f"  split=({equation_summary['fiber_sum_equations']}, "
        f"{equation_summary['fiber_anisotropy_equations']}, {equation_summary['dead_x_equations']})"
    )
    print()

    print('Verifying the standard 27-term algorithm on the symbolic split...')
    standard_rows, standard_summary = standard_algorithm_verify()
    print(
        f"  failures=(fiber_sum={standard_summary['standard_fiber_sum_failures']}, "
        f"anisotropy={standard_summary['standard_live_anisotropy_failures']}, dead={standard_summary['standard_dead_x_failures']})"
    )
    print()

    matrix_rows = build_matrix_form_rows()
    summary_rows = [
        {'summary_name': 'fiber_sum_equations', 'summary_value': equation_summary['fiber_sum_equations'], 'provenance': 'EXACT_DERIVED', 'note': 'One equation for each output C atom and each output fiber.'},
        {'summary_name': 'fiber_anisotropy_equations', 'summary_value': equation_summary['fiber_anisotropy_equations'], 'provenance': 'EXACT_DERIVED', 'note': 'Two independent live-direction difference equations per fiber and output.'},
        {'summary_name': 'dead_x_equations', 'summary_value': equation_summary['dead_x_equations'], 'provenance': 'EXACT_DERIVED', 'note': 'All dead coordinates must cancel after gamma weighting.'},
        {'summary_name': 'total_equations', 'summary_value': equation_summary['total_equations'], 'provenance': 'EXACT_DERIVED', 'note': 'The full tensor system size.'},
        {'summary_name': 'standard_fiber_sum_failures', 'summary_value': standard_summary['standard_fiber_sum_failures'], 'provenance': 'EXACT_DERIVED', 'note': 'Standard algorithm check on the 81 fiber-sum equations.'},
        {'summary_name': 'standard_live_anisotropy_failures', 'summary_value': standard_summary['standard_live_anisotropy_failures'], 'provenance': 'EXACT_DERIVED', 'note': 'Standard algorithm check on the 162 anisotropy equations.'},
        {'summary_name': 'standard_dead_x_failures', 'summary_value': standard_summary['standard_dead_x_failures'], 'provenance': 'EXACT_DERIVED', 'note': 'Standard algorithm check on the 486 dead-X equations.'},
        {'summary_name': 'matrix_form_constraint', 'summary_value': 'Gamma * Sigma = 3 I_9; Gamma * Eta1 = 0; Gamma * Eta2 = 0; Gamma * Delta = 0', 'provenance': 'EXACT_DERIVED', 'note': 'Exact matrix reformulation of the symbolic split.'},
    ]

    print('Writing outputs...')
    write_csv(EXPORTS / 'step51_live_fiber_coordinates.csv', live_rows, ['fiber', 'live_coordinates', 'lambda_coordinates', 'provenance'])
    write_csv(EXPORTS / 'step51_fiber_sum_formulas.csv', sum_rows, ['fiber', 'fiber_sum_symbol', 'formula', 'target_equation_template', 'provenance'])
    write_csv(EXPORTS / 'step51_live_anisotropy_formulas.csv', anisotropy_rows, ['fiber', 'anisotropy_symbol', 'formula', 'zero_equation_template', 'provenance'])
    write_csv(EXPORTS / 'step51_dead_x_formulas.csv', dead_rows, ['dead_coordinate', 'dead_symbol', 'formula', 'zero_equation_template', 'provenance'])
    write_csv(EXPORTS / 'step51_equation_block_split.csv', equation_rows, ['output_c', 'block_type', 'coordinate', 'equation', 'rhs', 'provenance'])
    write_csv(EXPORTS / 'step51_matrix_form.csv', matrix_rows, ['matrix_name', 'shape', 'entry_definition', 'meaning', 'provenance'])
    write_csv(EXPORTS / 'step51_standard_algorithm_symbolic_verification.csv', standard_rows, ['output_c', 'fiber', 'fiber_sum_total', 'fiber_sum_expected', 'eta1_total', 'eta2_total', 'status', 'provenance'])
    write_csv(EXPORTS / 'step51_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_markdown_summary(
        EXPORTS / 'step51_symbolic_fiber_mode_decomposition.md',
        live_rows,
        sum_rows,
        anisotropy_rows,
        dead_rows,
        equation_rows,
        matrix_rows,
        standard_rows,
        summary_rows,
    )
    print()
    print('=== SUMMARY ===')
    print(f"Equation split: {equation_summary['fiber_sum_equations']} + {equation_summary['fiber_anisotropy_equations']} + {equation_summary['dead_x_equations']} = {equation_summary['total_equations']}")
    print(f"Matrix form: {summary_rows[-1]['summary_value']}")


if __name__ == '__main__':
    main()