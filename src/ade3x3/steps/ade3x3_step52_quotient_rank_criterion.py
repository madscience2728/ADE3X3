"""
ade3x3_step52_quotient_rank_criterion.py

Step 52: Quotient-Space Rank Criterion.

Starting from Step 51's exact matrix form

  Gamma * Sigma = 3 I
  Gamma * Nuisance = 0

this step derives the exact quotient-space criterion for solvability and the
resulting necessary bound

  R >= base_dim + rank(Nuisance)

where base_dim = n^2 for n x n matrix multiplication. The criterion is checked
exactly on the standard 3x3 algorithm and on Strassen's 2x2 algorithm.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import sympy as sp

EXPORTS = Path('outputs/exports')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def standard_terms_3x3() -> list[dict[str, list[list[int]]]]:
    terms: list[dict[str, list[list[int]]]] = []
    for row_idx in range(3):
        for sum_idx in range(3):
            for col_idx in range(3):
                alpha = [[0 for _ in range(3)] for _ in range(3)]
                beta = [[0 for _ in range(3)] for _ in range(3)]
                gamma = [[0 for _ in range(3)] for _ in range(3)]
                alpha[row_idx][sum_idx] = 1
                beta[sum_idx][col_idx] = 1
                gamma[row_idx][col_idx] = 1
                terms.append({'alpha': alpha, 'beta': beta, 'gamma': gamma})
    return terms


def strassen_terms_2x2() -> list[dict[str, list[list[int]]]]:
    return [
        {'alpha': [[1, 0], [0, 1]], 'beta': [[1, 0], [0, 1]], 'gamma': [[1, 0], [0, 1]]},
        {'alpha': [[0, 0], [1, 1]], 'beta': [[1, 0], [0, 0]], 'gamma': [[0, 0], [1, -1]]},
        {'alpha': [[1, 0], [0, 0]], 'beta': [[0, 1], [0, -1]], 'gamma': [[0, 1], [0, 1]]},
        {'alpha': [[0, 0], [0, 1]], 'beta': [[-1, 0], [1, 0]], 'gamma': [[1, 0], [1, 0]]},
        {'alpha': [[1, 1], [0, 0]], 'beta': [[0, 0], [0, 1]], 'gamma': [[-1, 1], [0, 0]]},
        {'alpha': [[-1, 0], [1, 0]], 'beta': [[1, 1], [0, 0]], 'gamma': [[0, 0], [0, 1]]},
        {'alpha': [[0, 1], [0, -1]], 'beta': [[0, 0], [1, 1]], 'gamma': [[1, 0], [0, 0]]},
    ]


def build_mode_matrices(terms: list[dict[str, list[list[int]]]], n: int) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    sigma_columns: list[list[int]] = []
    eta_columns: list[list[int]] = []
    dead_columns: list[list[int]] = []

    for row_idx in range(n):
        for col_idx in range(n):
            sigma_vec: list[int] = []
            eta_basis: list[list[int]] = [[] for _ in range(n - 1)]
            for term in terms:
                lambdas = [term['alpha'][row_idx][sum_idx] * term['beta'][sum_idx][col_idx] for sum_idx in range(n)]
                sigma_vec.append(sum(lambdas))
                for basis_idx in range(n - 1):
                    eta_basis[basis_idx].append(lambdas[basis_idx] - lambdas[basis_idx + 1])
            sigma_columns.append(sigma_vec)
            eta_columns.extend(eta_basis)

    for row_idx in range(n):
        for sum_left in range(n):
            for sum_right in range(n):
                if sum_left == sum_right:
                    continue
                for col_idx in range(n):
                    dead_vec = [term['alpha'][row_idx][sum_left] * term['beta'][sum_right][col_idx] for term in terms]
                    dead_columns.append(dead_vec)

    sigma = sp.Matrix.hstack(*[sp.Matrix(column) for column in sigma_columns])
    eta = sp.Matrix.hstack(*[sp.Matrix(column) for column in eta_columns]) if eta_columns else sp.zeros(len(terms), 0)
    dead = sp.Matrix.hstack(*[sp.Matrix(column) for column in dead_columns]) if dead_columns else sp.zeros(len(terms), 0)
    nuisance = sp.Matrix.hstack(eta, dead)
    augmented = sp.Matrix.hstack(sigma, nuisance)
    return sigma, eta, dead, augmented


def rank_profile(name: str, n: int, terms: list[dict[str, list[list[int]]]]) -> dict[str, object]:
    sigma, eta, dead, augmented = build_mode_matrices(terms, n)
    nuisance = sp.Matrix.hstack(eta, dead)
    sigma_rank = sigma.rank()
    eta_rank = eta.rank()
    dead_rank = dead.rank()
    nuisance_rank = nuisance.rank()
    augmented_rank = augmented.rank()
    base_dim = n * n
    criterion_holds = augmented_rank == nuisance_rank + base_dim
    lower_bound = base_dim + nuisance_rank
    return {
        'algorithm': name,
        'matrix_size': f'{n}x{n}',
        'R': len(terms),
        'base_dim': base_dim,
        'sigma_columns': sigma.cols,
        'eta_columns': eta.cols,
        'dead_columns': dead.cols,
        'nuisance_columns': nuisance.cols,
        'sigma_rank': sigma_rank,
        'eta_rank': eta_rank,
        'dead_rank': dead_rank,
        'nuisance_rank': nuisance_rank,
        'augmented_rank': augmented_rank,
        'criterion_holds': criterion_holds,
        'lower_bound_from_nuisance': lower_bound,
        'saturates_lower_bound': len(terms) == lower_bound,
        'rank_gap_R_minus_bound': len(terms) - lower_bound,
        'provenance': 'EXACT_DERIVED',
    }


def theorem_rows() -> list[dict]:
    return [
        {
            'statement_id': 'T1',
            'statement': 'Fix a particular rank-R decomposition with term space V = R^R and columns Sigma, Eta, Delta built from that decomposition\'s alpha and beta factors. Let N be the nuisance span generated by Eta and Delta.',
            'status': 'exact_linear_algebra',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'T2',
            'statement': 'For that fixed decomposition, there exists Gamma with Gamma*Sigma = 3 I_9 and Gamma*Nuisance = 0 iff the 9 sigma columns are linearly independent modulo N.',
            'status': 'equivalent_condition',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'T3',
            'statement': 'Equivalently, for 3x3 one has rank([Sigma Nuisance]) = rank(Nuisance) + 9, and for 2x2 the analog is rank([Sigma Nuisance]) = rank(Nuisance) + 4.',
            'status': 'rank_form',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'T4',
            'statement': 'Hence this specific 3x3 decomposition satisfies R >= 9 + rank(Nuisance). This is a per-algorithm necessary bound, not yet a universal lower bound.',
            'status': 'per_algorithm_bound_3x3',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'T5',
            'statement': 'For a fixed 2x2 decomposition, the exact analog is R >= 4 + rank(Nuisance).',
            'status': 'per_algorithm_bound_2x2',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'T6',
            'statement': 'A universal bound would require proving a decomposition-independent lower bound on rank(Nuisance) for every admissible choice of alpha,beta whose Sigma columns have the required quotient-space independence. Step 52 does not prove such a theorem.',
            'status': 'scope_caveat',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def nuisance_bound_rows() -> list[dict]:
    rows: list[dict] = []
    for rank in [23, 22, 21, 20, 19, 18]:
        rows.append({
            'matrix_size': '3x3',
            'target_R': rank,
            'base_dim': 9,
            'max_allowed_nuisance_rank': rank - 9,
            'necessary_condition': f'rank(Nuisance) <= {rank - 9}',
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def summary_rows(algorithm_rows: list[dict]) -> list[dict]:
    by_name = {row['algorithm']: row for row in algorithm_rows}
    return [
        {'summary_name': 'step52_bound_scope', 'summary_value': 'per_algorithm_not_universal', 'provenance': 'EXACT_DERIVED', 'note': 'The Step 52 rank bound depends on the nuisance span of the specific decomposition under study.'},
        {'summary_name': 'standard_3x3_nuisance_rank', 'summary_value': str(by_name['standard_3x3']['nuisance_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact nuisance span dimension for the standard 27-term algorithm.'},
        {'summary_name': 'standard_3x3_lower_bound_from_nuisance', 'summary_value': str(by_name['standard_3x3']['lower_bound_from_nuisance']), 'provenance': 'EXACT_DERIVED', 'note': 'Step 52 lower bound recovered from the standard algorithm nuisance rank.'},
        {'summary_name': 'standard_3x3_saturates_bound', 'summary_value': str(by_name['standard_3x3']['saturates_lower_bound']), 'provenance': 'EXACT_DERIVED', 'note': 'Whether the standard algorithm exactly saturates R >= 9 + rank(Nuisance).'},
        {'summary_name': 'strassen_2x2_nuisance_rank', 'summary_value': str(by_name['strassen_2x2']['nuisance_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Exact nuisance span dimension for Strassen 2x2.'},
        {'summary_name': 'strassen_2x2_lower_bound_from_nuisance', 'summary_value': str(by_name['strassen_2x2']['lower_bound_from_nuisance']), 'provenance': 'EXACT_DERIVED', 'note': 'Step 52 lower bound recovered from Strassen nuisance rank.'},
        {'summary_name': 'strassen_2x2_saturates_bound', 'summary_value': str(by_name['strassen_2x2']['saturates_lower_bound']), 'provenance': 'EXACT_DERIVED', 'note': 'Whether Strassen exactly saturates R >= 4 + rank(Nuisance).'},
    ]


def write_markdown_summary(
    path: Path,
    theorem: list[dict],
    algorithm_rows: list[dict],
    bound_rows: list[dict],
    summary: list[dict],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append
    w('# Step 52: Quotient-Space Rank Criterion')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('Starting from Step 51, solvability is equivalent to the 9 fiber-sum columns being linearly independent modulo the nuisance subspace generated by live anisotropy and dead-X columns.')
    w('')
    w('## Exact Criterion')
    w('')
    for row in theorem:
        w(f"- {row['statement_id']}: {row['statement']}")
    w('')
    w('## Algorithm Profiles')
    w('')
    w('| algorithm | size | R | nuisance shape | sigma_rank | eta_rank | dead_rank | nuisance_rank | augmented_rank | lower_bound | saturates |')
    w('|-----------|------|---|----------------|------------|----------|-----------|---------------|----------------|-------------|-----------|')
    for row in algorithm_rows:
        w(f"| {row['algorithm']} | {row['matrix_size']} | {row['R']} | {row['R']}x{row['nuisance_columns']} | {row['sigma_rank']} | {row['eta_rank']} | {row['dead_rank']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['lower_bound_from_nuisance']} | {row['saturates_lower_bound']} |")
    w('')
    w('## 3x3 Nuisance-Rank Targets')
    w('')
    w('| target_R | max_allowed_nuisance_rank | necessary_condition |')
    w('|----------|---------------------------|---------------------|')
    for row in bound_rows:
        w(f"| {row['target_R']} | {row['max_allowed_nuisance_rank']} | {row['necessary_condition']} |")
    w('')
    w(f"Bound scope: {summary_map['step52_bound_scope']}")
    w('')
    w(f"Standard 3x3 nuisance rank: {summary_map['standard_3x3_nuisance_rank']}")
    w(f"Strassen 2x2 nuisance rank: {summary_map['strassen_2x2_nuisance_rank']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('This is the first exact linear-algebra obstruction beyond raw equation counting, but it is a per-decomposition statement. It does not yet prove a universal lower bound because rank(Nuisance) still depends on the chosen alpha,beta factors. What Step 52 does prove is that every candidate 3x3 algorithm must fit its own nuisance span inside dimension R-9. In particular, an R=22 algorithm would need nuisance rank at most 13. Strassen 2x2 achieves a 7x12 nuisance matrix of rank exactly 3, so it is tight against the 2x2 criterion R = 4 + rank(Nuisance).')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 52: Quotient-Space Rank Criterion ===')
    print()
    print('Deriving exact rank profiles...')
    standard_profile = rank_profile('standard_3x3', 3, standard_terms_3x3())
    strassen_profile = rank_profile('strassen_2x2', 2, strassen_terms_2x2())
    algorithm_rows = [standard_profile, strassen_profile]
    print(f"  standard_3x3 nuisance_rank={standard_profile['nuisance_rank']} lower_bound={standard_profile['lower_bound_from_nuisance']}")
    print(f"  strassen_2x2 nuisance_rank={strassen_profile['nuisance_rank']} lower_bound={strassen_profile['lower_bound_from_nuisance']}")
    print()

    theorem = theorem_rows()
    bound_rows = nuisance_bound_rows()
    summary = summary_rows(algorithm_rows)

    print('Writing outputs...')
    write_csv(EXPORTS / 'step52_quotient_rank_theorem.csv', theorem, ['statement_id', 'statement', 'status', 'provenance'])
    write_csv(EXPORTS / 'step52_algorithm_rank_profiles.csv', algorithm_rows, ['algorithm', 'matrix_size', 'R', 'base_dim', 'sigma_columns', 'eta_columns', 'dead_columns', 'nuisance_columns', 'sigma_rank', 'eta_rank', 'dead_rank', 'nuisance_rank', 'augmented_rank', 'criterion_holds', 'lower_bound_from_nuisance', 'saturates_lower_bound', 'rank_gap_R_minus_bound', 'provenance'])
    write_csv(EXPORTS / 'step52_required_nuisance_bounds.csv', bound_rows, ['matrix_size', 'target_R', 'base_dim', 'max_allowed_nuisance_rank', 'necessary_condition', 'provenance'])
    write_csv(EXPORTS / 'step52_summary.csv', summary, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_markdown_summary(EXPORTS / 'step52_quotient_rank_criterion.md', theorem, algorithm_rows, bound_rows, summary)
    print()
    print('=== SUMMARY ===')
    print(f"standard_3x3: R >= 9 + {standard_profile['nuisance_rank']} = {standard_profile['lower_bound_from_nuisance']}")
    print(f"strassen_2x2: R >= 4 + {strassen_profile['nuisance_rank']} = {strassen_profile['lower_bound_from_nuisance']}")


if __name__ == '__main__':
    main()