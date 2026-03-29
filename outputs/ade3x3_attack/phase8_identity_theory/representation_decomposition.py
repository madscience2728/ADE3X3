from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import phase5_identity_basis, write_json  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    _, identity_vectors = phase5_identity_basis()
    nullspace = sp.Matrix.hstack(*identity_vectors)

    avg = sp.ones(3, 3) / 3
    row_trivial_9 = sp.kronecker_product(avg, sp.eye(3))
    row_standard_9 = sp.eye(9) - row_trivial_9
    col_trivial_9 = sp.kronecker_product(sp.eye(3), avg)
    col_standard_9 = sp.eye(9) - col_trivial_9

    projectors = {
        'trivial_row__standard_shared__trivial_col': sp.diag(row_trivial_9 * col_trivial_9, row_trivial_9 * col_trivial_9),
        'trivial_row__standard_shared__standard_col': sp.diag(row_trivial_9 * col_standard_9, row_trivial_9 * col_standard_9),
        'standard_row__standard_shared__trivial_col': sp.diag(row_standard_9 * col_trivial_9, row_standard_9 * col_trivial_9),
        'standard_row__standard_shared__standard_col': sp.diag(row_standard_9 * col_standard_9, row_standard_9 * col_standard_9),
    }
    component_dimensions = {
        'trivial_row__standard_shared__trivial_col': 2,
        'trivial_row__standard_shared__standard_col': 4,
        'standard_row__standard_shared__trivial_col': 4,
        'standard_row__standard_shared__standard_col': 8,
    }

    projection_ranks = {
        name: int((projector * nullspace).rank()) for name, projector in projectors.items()
    }
    write_json(
        OUTPUT_DIR / 'representation_summary.json',
        {
            'eta_space_dimension': 18,
            'decomposition': [
                {'component': name, 'dimension': dimension} for name, dimension in component_dimensions.items()
            ],
            'identity_projection_ranks': projection_ranks,
            'interpretation': (
                'As a representation of S3(row) x S3(shared) x S3(col), eta-space decomposes as '
                '(1+2)_row x 2_shared x (1+2)_col = 2 + 4 + 4 + 8. '
                'The four AlphaTensor identities are not confined to a single low-dimensional orbit-average block; '
                'their projections are nonzero across all four isotypic pieces.'
            ),
        },
    )


if __name__ == '__main__':
    main()