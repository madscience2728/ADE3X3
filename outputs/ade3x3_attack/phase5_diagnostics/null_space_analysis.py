from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import eta_coordinate_labels, load_public_terms, primitive_integer_vector, write_csv, write_json  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import build_mode_matrices  # noqa: E402


def support_string(labels: list[str], coeffs: list[int]) -> str:
    parts = [f'{coeff:+d}*{label}' for coeff, label in zip(coeffs, labels) if coeff != 0]
    return ' '.join(parts) if parts else '0'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terms, orientation, residual = load_public_terms()
    _, eta1, eta2, _, _ = build_mode_matrices(terms, 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    labels = eta_coordinate_labels()
    null_basis = h_matrix.nullspace()

    rows: list[dict] = []
    formulas: list[str] = []
    for basis_idx, vector in enumerate(null_basis, start=1):
        primitive = primitive_integer_vector(vector)
        formulas.append(support_string(labels, primitive))
        row = {
            'basis_id': basis_idx,
            'support_size': sum(int(value != 0) for value in primitive),
            'formula': support_string(labels, primitive),
        }
        for label, coeff in zip(labels, primitive):
            row[label] = coeff
        rows.append(row)

    write_csv(OUT_DIR / 'null_vectors.csv', rows, ['basis_id', 'support_size', 'formula', *labels])
    write_json(
        OUT_DIR / 'null_space_summary.json',
        {
            'source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
            'gamma_orientation': orientation,
            'exact_reconstruction_residual': int(residual),
            'rank_H_exact': int(h_matrix.rank()),
            'eta_coordinate_count': h_matrix.cols,
            'nullity_exact': h_matrix.cols - int(h_matrix.rank()),
            'basis_formulas': formulas,
        },
    )

    print('Phase 5e null-space analysis complete')
    print(f'  nullity = {h_matrix.cols - int(h_matrix.rank())}')


if __name__ == '__main__':
    main()