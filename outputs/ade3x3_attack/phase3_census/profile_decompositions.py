from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import load_serialized_decompositions, profile_terms, terms_from_decomposition, write_csv, write_json  # noqa: E402


def format_range(values: list[object]) -> str:
    if not values:
        return 'n/a'
    ordered = sorted(values)
    return f'{ordered[0]}..{ordered[-1]}'


def format_mode(values: list[object]) -> str:
    if not values:
        return 'n/a'
    counts = Counter(values)
    best_value, _ = max(counts.items(), key=lambda item: (item[1], -float(item[0]) if isinstance(item[0], (int, float)) else 0.0))
    return str(best_value)


def build_results_markdown(summary_rows: list[dict], verdict: str, profile_counts: dict[str, int], attempt_counts: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append('# Phase 3 Results')
    lines.append('')
    lines.append('## Summary Table')
    lines.append('')
    lines.append('| source | count | attempts | delta-in-eta rate | rank(H) range | rank(H) mode | M sparsity range | M integrality rate |')
    lines.append('|--------|-------|----------|-------------------|---------------|--------------|------------------|--------------------|')
    for row in summary_rows:
        source = row['source']
        lines.append(
            f"| {source} | {profile_counts.get(source, 0)} | {attempt_counts.get(source, 0)} | {row['delta_in_eta_rate']} | {row['rank_h_range']} | {row['rank_h_mode']} | {row['m_sparsity_range']} | {row['m_integrality_rate']} |"
        )
    lines.append('')
    lines.append('## Verdict')
    lines.append('')
    lines.append(verdict)
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- The symmetry-orbit census is exact and exhaustive over the 216 compatible S3 x S3 x S3 actions.')
    lines.append('- The perturbation and random-search rows count only exact kept decompositions written to the JSONL archive; all attempts are still recorded in generation_attempts.csv.')
    lines.append('- A zero count outside the symmetry orbit means no exact decomposition met the configured residual tolerance in this run, not that nearby numerical minima were absent.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    decompositions = load_serialized_decompositions(OUT_DIR / 'generated_decompositions.jsonl')
    attempt_rows_path = OUT_DIR / 'generation_attempts.csv'
    profile_rows: list[dict] = []
    by_source: defaultdict[str, list[dict]] = defaultdict(list)
    attempt_counts: Counter[str] = Counter()

    if attempt_rows_path.exists():
        import csv

        with open(attempt_rows_path, 'r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                attempt_counts[str(row['source'])] += 1

    for payload in decompositions:
        terms = terms_from_decomposition(payload)
        profile = profile_terms(terms)
        row = {
            'decomposition_id': payload['decomposition_id'],
            'source': payload['source'],
            'source_detail': payload['source_detail'],
            **profile,
        }
        profile_rows.append(row)
        by_source[str(payload['source'])].append(row)

    fieldnames: list[str] = []
    if profile_rows:
        fieldnames = list(profile_rows[0].keys())
    write_csv(OUT_DIR / 'decomposition_profiles.csv', profile_rows, fieldnames or ['decomposition_id', 'source', 'source_detail'])

    summary_rows: list[dict] = []
    profile_counts: dict[str, int] = {}
    for source in sorted(set(attempt_counts) | set(by_source)):
        rows = by_source.get(source, [])
        profile_counts[source] = len(rows)
        delta_hits = sum(int(bool(row.get('delta_in_eta_exact', row.get('delta_in_eta_numeric', False)))) for row in rows)
        rank_h_values = [int(row.get('rank_H_exact', row['rank_H_numeric'])) for row in rows]
        sparsity_values = [int(row.get('projection_nonzero_count', row.get('projection_nonzero_count_numeric', 0))) for row in rows if bool(row.get('delta_in_eta_exact', row.get('delta_in_eta_numeric', False)))]
        integrality_hits = sum(int(bool(row.get('projection_integral', row.get('projection_allclose_to_integer_numeric', False)))) for row in rows if bool(row.get('delta_in_eta_exact', row.get('delta_in_eta_numeric', False))))
        integrality_total = sum(int(bool(row.get('delta_in_eta_exact', row.get('delta_in_eta_numeric', False)))) for row in rows)
        summary_rows.append(
            {
                'source': source,
                'count': len(rows),
                'attempts': attempt_counts.get(source, 0),
                'delta_in_eta_rate': 'n/a' if not rows else f'{delta_hits}/{len(rows)}',
                'rank_h_range': format_range(rank_h_values),
                'rank_h_mode': format_mode(rank_h_values),
                'm_sparsity_range': format_range(sparsity_values),
                'm_integrality_rate': 'n/a' if integrality_total == 0 else f'{integrality_hits}/{integrality_total}',
            }
        )

    write_csv(
        OUT_DIR / 'profile_summary.csv',
        summary_rows,
        ['source', 'count', 'attempts', 'delta_in_eta_rate', 'rank_h_range', 'rank_h_mode', 'm_sparsity_range', 'm_integrality_rate'],
    )

    if by_source.get('symmetry_orbit') and all(bool(row.get('delta_in_eta_exact', False)) for row in by_source['symmetry_orbit']) and sum(profile_counts.get(source, 0) for source in profile_counts if source != 'symmetry_orbit') == 0:
        verdict = 'delta subset eta is universal across the entire 216-element AlphaTensor symmetry orbit, but this run found no exact non-orbit rank-23 decompositions. Universality beyond the AlphaTensor equivalence class remains unclear.'
    elif all(bool(row.get('delta_in_eta_exact', row.get('delta_in_eta_numeric', False))) for source, rows in by_source.items() for row in rows):
        verdict = 'delta subset eta appears universal across every exact decomposition found in this census.'
    else:
        verdict = 'delta subset eta is not universal across the exact decompositions found in this census.'

    results_md = build_results_markdown(summary_rows, verdict, profile_counts, attempt_counts)
    (OUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')
    write_json(
        OUT_DIR / 'profile_summary.json',
        {
            'profile_counts': profile_counts,
            'attempt_counts': dict(attempt_counts),
            'verdict': verdict,
        },
    )

    print('Phase 3 profiling complete')
    print(f'  profiled exact decompositions = {len(profile_rows)}')
    print(f'  verdict = {verdict}')


if __name__ == '__main__':
    main()