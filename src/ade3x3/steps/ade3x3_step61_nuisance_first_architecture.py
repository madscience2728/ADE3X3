"""
ade3x3_step61_nuisance_first_architecture.py

Step 61: Nuisance-First Architecture.

This step flips the perspective of Steps 51-60. Instead of asking how to
avoid nuisance, it treats nuisance as the required carrier of any saving
below rank 27.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

EXPORTS = Path('outputs/exports')
TARGET_RANKS = list(range(9, 28))
HIGHLIGHT_RANKS = [9, 18, 19, 20, 21, 22, 23, 27]


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


def nuisance_budget_rows() -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    highlight_rows: list[dict] = []
    for total_rank in TARGET_RANKS:
        gamma_nullity = total_rank - 9
        max_nuisance_rank = gamma_nullity
        max_dead_rank = gamma_nullity
        min_total_nuisance_dependencies = 72 - max_nuisance_rank
        min_dead_dependencies = 54 - max_dead_rank
        deadfree_all_terms_possible = total_rank >= 27
        row = {
            'R': total_rank,
            'gamma_rank': 9,
            'gamma_nullity': gamma_nullity,
            'max_nuisance_rank': max_nuisance_rank,
            'max_dead_rank': max_dead_rank,
            'min_total_nuisance_dependencies': min_total_nuisance_dependencies,
            'min_dead_dependencies': min_dead_dependencies,
            'all_terms_deadfree_possible': str(deadfree_all_terms_possible),
            'all_terms_deadfree_ruled_out': str(not deadfree_all_terms_possible),
            'provenance': 'EXACT_DERIVED',
        }
        all_rows.append(row)
        if total_rank in HIGHLIGHT_RANKS:
            highlight_rows.append(row)
    return all_rows, highlight_rows


def r9_theorem_rows() -> list[dict]:
    return [
        {
            'statement_id': 'N1',
            'statement': 'In any exact 3x3 decomposition, Gamma*Sigma = 3I_9 forces rank(Gamma)=9.',
            'value': 'rank_Gamma_is_9',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N2',
            'statement': 'For R=9, Gamma is a 9x9 invertible matrix, so ker(Gamma) = {0}.',
            'value': 'nullity_0',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N3',
            'statement': 'Therefore Eta1 = 0, Eta2 = 0, and Delta = 0 as full matrices, not just after summation.',
            'value': 'all_nuisance_zero',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N4',
            'statement': 'By Step 54 D1, Delta = 0 implies every nonzero term is dead-free and therefore supported on a unique summation index s*.',
            'value': 'singleton_sum_index_support',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N5',
            'statement': 'By Step 54 D2, a dead-free term at s*=0 has (Sigma,Eta1,Eta2)=(v,v,0), at s*=1 has (v,-v,v), and at s*=2 has (v,0,-v). Hence Eta1=Eta2=0 implies v=0 and therefore Sigma=0 for that term.',
            'value': 'no_nonzero_nuisance_free_term',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N6',
            'statement': 'So an exact R=9 decomposition would force every term to have zero Sigma row, contradicting rank(Sigma)=9 in Gamma*Sigma = 3I_9.',
            'value': 'R9_impossible',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N7',
            'statement': 'Weaker corollary: if only Delta = 0 is imposed, then terms split by summation index and each summation channel must span the full 9-dimensional output-matrix space, requiring at least 9 terms per channel and therefore R >= 27.',
            'value': 'deadfree_implies_R_at_least_27',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'N8',
            'statement': 'Hence any exact algorithm with R < 27 must use nonzero nuisance. Nuisance is structurally necessary, not an accidental by-product.',
            'value': 'nuisance_is_load_bearing',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def deadfree_channel_rows() -> list[dict]:
    rows: list[dict] = []
    for sum_index in range(3):
        rows.append({
            'summation_index': sum_index,
            'active_term_shape': 'p_k q_k^T in M_3x3',
            'target_output_space_dimension': 9,
            'minimum_terms_needed_for_channel': 9,
            'channel_statement': 'Dead-free terms at fixed summation index reconstruct only that live channel, so their output-matrix span must equal all 3x3 matrices.',
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def rank18_pressure_rows() -> list[dict]:
    rows: list[dict] = []
    for total_rank in [18, 19, 20, 21, 22, 23]:
        gamma_nullity = total_rank - 9
        rows.append({
            'R': total_rank,
            'gamma_nullity': gamma_nullity,
            'max_nuisance_rank': gamma_nullity,
            'max_dead_rank': gamma_nullity,
            'dead_columns': 54,
            'min_dead_linear_dependencies': 54 - gamma_nullity,
            'total_nuisance_columns': 72,
            'min_total_nuisance_linear_dependencies': 72 - gamma_nullity,
            'deadfree_all_terms_ruled_out': str(total_rank < 27),
            'interpretation': (
                'R=18 is the tight nuisance-first budget: dead rank must fit inside 9 dimensions, forcing at least 45 dead-column dependencies.'
                if total_rank == 18
                else f'At R={total_rank}, nuisance must fit inside a {gamma_nullity}-dimensional Gamma-nullspace.'
            ),
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def summary_rows() -> list[dict]:
    return [
        {
            'summary_name': 'R9_status',
            'summary_value': 'impossible_by_zero_nuisance_contradiction',
            'provenance': 'EXACT_DERIVED',
            'note': 'R=9 forces Gamma invertible, which forces all nuisance zero and therefore every term zero.',
        },
        {
            'summary_name': 'deadfree_global_lower_bound',
            'summary_value': '27',
            'provenance': 'EXACT_DERIVED',
            'note': 'If every term is dead-free, per-channel reconstruction requires at least 9 terms for each of the three summation channels.',
        },
        {
            'summary_name': 'R18_gamma_nullity',
            'summary_value': '9',
            'provenance': 'EXACT_DERIVED',
            'note': 'At R=18 the Gamma-nullspace has dimension 9 and must contain the entire nuisance span.',
        },
        {
            'summary_name': 'R18_min_dead_dependencies',
            'summary_value': '45',
            'provenance': 'EXACT_DERIVED',
            'note': 'The 54 dead columns must fit inside at most 9 dimensions, forcing at least 45 exact linear dependencies.',
        },
        {
            'summary_name': 'R18_min_total_nuisance_dependencies',
            'summary_value': '63',
            'provenance': 'EXACT_DERIVED',
            'note': 'The 72 nuisance columns must fit inside at most 9 dimensions, forcing at least 63 exact linear dependencies.',
        },
        {
            'summary_name': 'sub27_requires_nuisance',
            'summary_value': 'true',
            'provenance': 'EXACT_DERIVED',
            'note': 'Any exact algorithm below rank 27 must use nonzero nuisance columns.',
        },
    ]


def write_markdown_summary(
    path: Path,
    summary: list[dict],
    theorem_rows: list[dict],
    channel_rows: list[dict],
    budget_rows: list[dict],
    pressure_rows: list[dict],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append
    w('# Step 61: Nuisance-First Architecture')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('Step 61 reverses the usual perspective. Instead of trying to eliminate nuisance, it shows exactly what the Step 51-52 framework forces at small ranks: zero nuisance is impossible below rank 27, and low-rank algorithms must engineer a highly compressed nuisance span inside the Gamma-nullspace.')
    w('')
    w(f"R=9 status: {summary_map['R9_status']}")
    w(f"Dead-free global lower bound: {summary_map['deadfree_global_lower_bound']}")
    w(f"R=18 Gamma-nullity: {summary_map['R18_gamma_nullity']}")
    w(f"R=18 minimum dead dependencies: {summary_map['R18_min_dead_dependencies']}")
    w('')
    w('## R=9 Theorem Chain')
    w('')
    w('| id | statement | value |')
    w('|----|-----------|-------|')
    for row in theorem_rows:
        w(f"| {row['statement_id']} | {row['statement']} | {row['value']} |")
    w('')
    w('## Dead-Free Per-Channel Lower Bound')
    w('')
    w('| summation index | active term shape | target dimension | minimum terms needed |')
    w('|-----------------|-------------------|------------------|----------------------|')
    for row in channel_rows:
        w(f"| {row['summation_index']} | {row['active_term_shape']} | {row['target_output_space_dimension']} | {row['minimum_terms_needed_for_channel']} |")
    w('')
    w('## Nuisance Budget Table')
    w('')
    w('| R | Gamma nullity | max nuisance rank | max dead rank | min dead dependencies | min total nuisance dependencies | all terms dead-free possible? |')
    w('|---|---------------|-------------------|---------------|-----------------------|--------------------------------|-------------------------------|')
    for row in budget_rows:
        w(f"| {row['R']} | {row['gamma_nullity']} | {row['max_nuisance_rank']} | {row['max_dead_rank']} | {row['min_dead_dependencies']} | {row['min_total_nuisance_dependencies']} | {row['all_terms_deadfree_possible']} |")
    w('')
    w('## R=18..23 Pressure Rows')
    w('')
    w('| R | Gamma nullity | max nuisance rank | min dead dependencies | min total nuisance dependencies |')
    w('|---|---------------|-------------------|-----------------------|--------------------------------|')
    for row in pressure_rows:
        w(f"| {row['R']} | {row['gamma_nullity']} | {row['max_nuisance_rank']} | {row['min_dead_linear_dependencies']} | {row['min_total_nuisance_linear_dependencies']} |")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The exact message is not that nuisance is unfortunate; it is that nuisance is mandatory. At R=9 the framework collapses immediately: Gamma has no nullspace, so all nuisance would have to vanish term-by-term, but Step 54 shows a nonzero dead-free term always carries anisotropy nuisance. More generally, if every term were dead-free then the three summation channels decouple and each channel would need 9 terms to span the full 3x3 output-matrix space, forcing R>=27. So every savings below 27 must come from channel-mixing terms with load-bearing nuisance. At R=18 the burden is already extreme: the full 72 nuisance columns must live in a 9-dimensional annihilator, and the 54 dead columns alone need at least 45 exact linear dependencies. That does not prove R=18 impossible, but it recasts the problem precisely: success below 27 requires deliberately engineered nuisance identities, not nuisance avoidance.')
    write_markdown(path, '\n'.join(lines))


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print('=== Step 61: Nuisance-First Architecture ===')
    print()

    all_budget_rows, highlight_budget_rows = nuisance_budget_rows()
    theorem_rows = r9_theorem_rows()
    channel_rows = deadfree_channel_rows()
    pressure_rows = rank18_pressure_rows()
    summary = summary_rows()

    write_csv(
        EXPORTS / 'step61_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_csv(
        EXPORTS / 'step61_r9_theorem_chain.csv',
        theorem_rows,
        ['statement_id', 'statement', 'value', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step61_deadfree_channel_bound.csv',
        channel_rows,
        ['summation_index', 'active_term_shape', 'target_output_space_dimension', 'minimum_terms_needed_for_channel', 'channel_statement', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step61_nuisance_budget_table.csv',
        all_budget_rows,
        ['R', 'gamma_rank', 'gamma_nullity', 'max_nuisance_rank', 'max_dead_rank', 'min_total_nuisance_dependencies', 'min_dead_dependencies', 'all_terms_deadfree_possible', 'all_terms_deadfree_ruled_out', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step61_rank18_23_pressure.csv',
        pressure_rows,
        ['R', 'gamma_nullity', 'max_nuisance_rank', 'max_dead_rank', 'dead_columns', 'min_dead_linear_dependencies', 'total_nuisance_columns', 'min_total_nuisance_linear_dependencies', 'deadfree_all_terms_ruled_out', 'interpretation', 'provenance'],
    )
    write_markdown_summary(
        EXPORTS / 'step61_nuisance_first_architecture.md',
        summary,
        theorem_rows,
        channel_rows,
        highlight_budget_rows,
        pressure_rows,
    )

    print('\nSummary:')
    for row in summary:
        print(f"  {row['summary_name']} = {row['summary_value']}")


if __name__ == '__main__':
    main()