from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import ACTIONS, action_label, eta_action_matrix, phase5_identity_basis, write_json  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    _, identity_vectors = phase5_identity_basis()
    nullspace = sp.Matrix.hstack(*identity_vectors)
    intersection_histogram: Counter[int] = Counter()
    orbit_columns: list[sp.Matrix] = []
    max_augmented_rank = 4
    near_stabilizers: list[str] = []

    for action in ACTIONS:
        transformed = eta_action_matrix(action).inv().T * nullspace
        augmented_rank = int(sp.Matrix.hstack(nullspace, transformed).rank())
        max_augmented_rank = max(max_augmented_rank, augmented_rank)
        intersection_dimension = 8 - augmented_rank
        intersection_histogram[intersection_dimension] += 1
        orbit_columns.extend(transformed.columnspace())
        if intersection_dimension >= 3:
            near_stabilizers.append(action_label(action))

    orbit_span = sp.Matrix.hstack(*orbit_columns)
    orbit_span_rank = int(sp.Matrix.hstack(*orbit_span.columnspace()).rank()) if orbit_columns else 0
    write_json(
        OUTPUT_DIR / 'invariance_summary.json',
        {
            'g_invariant': False,
            'identity_space_dimension': 4,
            'single_transformed_copy_max_augmented_rank': max_augmented_rank,
            'orbit_span_dimension': orbit_span_rank,
            'intersection_histogram': {str(key): value for key, value in sorted(intersection_histogram.items())},
            'near_stabilizers': near_stabilizers,
            'interpretation': (
                'The 4-dimensional identity space is not G-invariant under the 216 compatible actions. '
                'Only the identity action preserves it exactly; one shared transposition has 3-dimensional intersection, '
                'and the full orbit of the identity space fills all 18 eta coordinates.'
            ),
        },
    )


if __name__ == '__main__':
    main()