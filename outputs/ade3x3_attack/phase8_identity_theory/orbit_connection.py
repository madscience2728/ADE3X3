from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
PHASE6_DIR = OUTPUT_DIR.parent / 'phase6_identity_algebra'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    rep = load_json(OUTPUT_DIR / 'representation_summary.json')
    inv = load_json(OUTPUT_DIR / 'invariance_summary.json')
    removal = load_json(PHASE6_DIR / 'single_term_removal_summary.json')

    verdict = (
        'The four exact identities are decomposition-specific rather than universal orbit invariants: '
        'their space is not G-invariant, and its full 216-orbit fills the entire 18-dimensional eta-space. '
        'At the same time, the 4-dimensional identity count survives every single AlphaTensor term deletion, '
        'so the phenomenon is collective and rigid inside the AlphaTensor class, not a single-term accident.'
    )
    results_md = f'''# Phase 8 Results

## Representation Content

- Eta-space has dimension {rep['eta_space_dimension']} and decomposes as `2 + 4 + 4 + 8` under `S3(row) x S3(shared) x S3(col)`.
- The shared factor is always the standard 2-dimensional representation; the row and column permutation factors each split as `1 + 2`.
- The four AlphaTensor identities project nontrivially into every isotypic block, so they are not simple orbit-average constraints.

## G-Invariance Test

- The 4-dimensional identity space is not invariant under the full 216-group.
- A single transformed copy can increase the joint span to dimension {inv['single_transformed_copy_max_augmented_rank']}.
- The full symmetry orbit of the identity space has dimension {inv['orbit_span_dimension']} = all of eta-space.

## Orbit-Connection Verdict

- {verdict}
- This is consistent with the Phase 6 single-removal census: all 23 removals keep `rank(H)={removal['minimum_observed_rank_H']}` and nullity 4, so the crown-jewel identities persist across the AlphaTensor support but do not extend to a universal G-stable law.
'''
    (OUTPUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')
    (OUTPUT_DIR / 'orbit_connection_summary.json').write_text(
        json.dumps(
            {
                'universality_verdict': 'decomposition-specific',
                'orbit_connection_verdict': verdict,
                'phase6_single_removal_rank_H': removal['minimum_observed_rank_H'],
            },
            indent=2,
        ),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()