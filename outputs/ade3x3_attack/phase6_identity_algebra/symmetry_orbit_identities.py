from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import (  # noqa: E402
    ACTIONS,
    action_label,
    eta_action_matrix,
    eta_coordinate_labels,
    phase5_identity_basis,
    primitive_integer_vector,
    write_csv,
    write_json,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def vector_to_formula(vector: sp.Matrix, labels: list[str]) -> str:
    terms: list[str] = []
    for idx, value in enumerate(list(vector)):
        coeff = int(sp.Integer(value))
        if coeff == 0:
            continue
        sign = '+' if coeff > 0 else '-'
        terms.append(f'{sign}{abs(coeff)}*{labels[idx]}')
    return ' '.join(terms) if terms else '0'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    eta_labels, identity_vectors = phase5_identity_basis()
    nullspace = sp.Matrix.hstack(*identity_vectors)
    intersection_histogram: Counter[int] = Counter()
    rows: list[dict[str, object]] = []
    orbit_columns: list[sp.Matrix] = []
    new_identity_examples: list[dict[str, object]] = []
    seen_examples: set[tuple[int, ...]] = set()

    for action in ACTIONS:
        eta_map = eta_action_matrix(action)
        transformed_space = eta_map.inv().T * nullspace
        augmented = sp.Matrix.hstack(nullspace, transformed_space)
        augmented_rank = int(augmented.rank())
        intersection_dimension = 8 - augmented_rank
        intersection_histogram[intersection_dimension] += 1
        rows.append(
            {
                'action': action_label(action),
                'augmented_rank': augmented_rank,
                'intersection_dimension': intersection_dimension,
                'preserves_identity_space': augmented_rank == 4,
            }
        )
        orbit_columns.extend(transformed_space.columnspace())
        for column in transformed_space.columnspace():
            if int(sp.Matrix.hstack(nullspace, column).rank()) == 4:
                continue
            normalized = tuple(primitive_integer_vector(column))
            if normalized in seen_examples:
                continue
            seen_examples.add(normalized)
            if len(new_identity_examples) < 8:
                new_identity_examples.append(
                    {
                        'action': action_label(action),
                        'formula': vector_to_formula(sp.Matrix(normalized), eta_labels),
                    }
                )

    orbit_span = sp.Matrix.hstack(*orbit_columns)
    orbit_span_rank = int(sp.Matrix.hstack(*orbit_span.columnspace()).rank()) if orbit_columns else 0

    write_csv(
        OUTPUT_DIR / 'symmetry_orbit_summary.csv',
        rows,
        ['action', 'augmented_rank', 'intersection_dimension', 'preserves_identity_space'],
    )
    write_json(
        OUTPUT_DIR / 'symmetry_orbit_summary.json',
        {
            'base_nullspace_dimension': 4,
            'single_transformed_copy_max_augmented_rank': max(int(row['augmented_rank']) for row in rows),
            'full_orbit_span_rank': orbit_span_rank,
            'stabilizer_actions': [row['action'] for row in rows if bool(row['preserves_identity_space'])],
            'intersection_histogram': {str(key): value for key, value in sorted(intersection_histogram.items())},
            'new_identity_examples': new_identity_examples,
            'interpretation': (
                'Symmetry transport does not preserve the original 4-dimensional identity space except for the identity action. '
                'A single transformed copy can enlarge the span to dimension 8, and the full 216-orbit spans all 18 eta coordinates. '
                'So symmetry generates identities for transformed decompositions, not new compatible identities for the fixed AlphaTensor decomposition.'
            ),
        },
    )

    expansion = load_json(OUTPUT_DIR / 'identity_expansion_summary.json')
    variety = load_json(OUTPUT_DIR / 'variety_dimension_summary.json')
    removal = load_json(OUTPUT_DIR / 'single_term_removal_summary.json')
    symmetry = load_json(OUTPUT_DIR / 'symmetry_orbit_summary.json')

    results_md = f'''# Phase 6 Results

## 6a. Exact Bilinear Expansion

- Four exact null-space identities were expanded into 9x9 bilinear matrices `Q_1,...,Q_4` and exported in `Q_matrices.csv`.
- All 23 public AlphaTensor terms satisfy all four bilinear equations exactly.
- Each identity is non-tautological: a small integer witness `(a,b)` was found for every `Q_j` with `a^T Q_j b != 0`.

## 6b-6c. Variety Defined by the Four Identities

- Random exact sampling on `V_4 = {{(a,b) : a^T Q_j b = 0, j=1..4}}` gave constraint rank 4 for generic `a` and Jacobian rank 4 at all sampled points.
- The observed dimension is therefore `18 - 4 = {variety['estimated_variety_dimension']}`.
- Sampled `sigma` vectors from `V_4` still span dimension {variety['sampled_sigma_span_rank']}, so the four identities do not collapse the 9-dimensional signal block.

## 6d. Search for a Fifth Compatible Identity

- Direct null-space extension on every 22-term single-removal subset failed: the minimum observed `rank(H)` is {removal['minimum_observed_rank_H']}, so no subset reaches the required `rank(H) <= 13` for an immediate `R=22` opening.
- Symmetry-guided transport also does not produce a fifth compatible identity for the fixed decomposition.
- The original 4-dimensional identity space is preserved only by the identity action, a single transformed copy can enlarge the span to dimension {symmetry['single_transformed_copy_max_augmented_rank']}, and the full orbit spans dimension {symmetry['full_orbit_span_rank']} in eta-space.

## 6e. Single-Term Removal Verdict

- All 23 single-removal subsets preserve `rank(H)=14`, `eta`-nullity 4, `rank([H|Delta])=14`, and quotient gain 8.
- Verdict: no single removed AlphaTensor term reveals a fifth exact identity.
'''
    (OUTPUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')


if __name__ == '__main__':
    main()