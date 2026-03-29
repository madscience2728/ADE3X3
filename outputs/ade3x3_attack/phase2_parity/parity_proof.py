from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    build_mode_matrices,
    load_public_rank23_terms,
)


OMEGA = -sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2
OMEGA_SQ = -sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def xi_plus_minus(lambda0: sp.Expr, lambda1: sp.Expr, lambda2: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    xi_plus = sp.expand(lambda0 + OMEGA * lambda1 + OMEGA_SQ * lambda2)
    xi_minus = sp.expand(lambda0 + OMEGA_SQ * lambda1 + OMEGA * lambda2)
    return xi_plus, xi_minus


def alpha_tensor_xi_matrix(terms: list) -> sp.Matrix:
    rows: list[list[sp.Expr]] = []
    for term in terms:
        row: list[sp.Expr] = []
        for row_idx in range(3):
            for col_idx in range(3):
                lambda0 = int(term.alpha[row_idx, 0] * term.beta[0, col_idx])
                lambda1 = int(term.alpha[row_idx, 1] * term.beta[1, col_idx])
                lambda2 = int(term.alpha[row_idx, 2] * term.beta[2, col_idx])
                xi_plus, _ = xi_plus_minus(lambda0, lambda1, lambda2)
                row.append(xi_plus)
        rows.append(row)
    return sp.Matrix(rows)


def alpha_tensor_verification_rows(terms: list) -> list[dict]:
    rows: list[dict] = []
    for term in terms:
        for row_idx in range(3):
            for col_idx in range(3):
                lambda0 = int(term.alpha[row_idx, 0] * term.beta[0, col_idx])
                lambda1 = int(term.alpha[row_idx, 1] * term.beta[1, col_idx])
                lambda2 = int(term.alpha[row_idx, 2] * term.beta[2, col_idx])
                eta1 = lambda0 - lambda1
                eta2 = lambda1 - lambda2
                xi_plus, xi_minus = xi_plus_minus(lambda0, lambda1, lambda2)
                rows.append({
                    'term_id': term.term_id,
                    'fiber': f'({row_idx},{col_idx})',
                    'eta1': str(eta1),
                    'eta2': str(eta2),
                    'xi_plus': sp.sstr(sp.expand(xi_plus)),
                    'xi_minus': sp.sstr(sp.expand(xi_minus)),
                    'xi_plus_formula_holds': bool(sp.simplify(xi_plus - (eta1 - OMEGA_SQ * eta2)) == 0),
                    'xi_minus_formula_holds': bool(sp.simplify(xi_minus - (eta1 - OMEGA * eta2)) == 0),
                    'conjugacy_holds': bool(sp.simplify(sp.conjugate(xi_plus) - xi_minus) == 0),
                })
    return rows


def dead_free_counterexample_row() -> dict[str, object]:
    lambda0 = sp.Integer(1)
    lambda1 = sp.Integer(0)
    lambda2 = sp.Integer(0)
    eta1 = lambda0 - lambda1
    eta2 = lambda1 - lambda2
    xi_plus, xi_minus = xi_plus_minus(lambda0, lambda1, lambda2)
    return {
        'example': 'single_dead_free_term_s_star_0',
        'eta1': str(eta1),
        'eta2': str(eta2),
        'eta_rank': 1,
        'xi_plus': sp.sstr(xi_plus),
        'xi_minus': sp.sstr(xi_minus),
        'xi_plus_complex_rank': 1,
        'claimed_even_rank_theorem_fails': True,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lambda0, lambda1, lambda2 = sp.symbols('lambda0 lambda1 lambda2', real=True)
    eta1 = sp.expand(lambda0 - lambda1)
    eta2 = sp.expand(lambda1 - lambda2)
    xi_plus, xi_minus = xi_plus_minus(lambda0, lambda1, lambda2)
    eta1_sym, eta2_sym = sp.symbols('eta1 eta2')
    xi_plus_sym, xi_minus_sym = sp.symbols('xi_plus xi_minus')

    direct_transform_rows = [
        {
            'coordinate': 'xi_plus',
            'formula_in_eta': sp.sstr(sp.expand(eta1_sym - OMEGA_SQ * eta2_sym)),
        },
        {
            'coordinate': 'xi_minus',
            'formula_in_eta': sp.sstr(sp.expand(eta1_sym - OMEGA * eta2_sym)),
        },
        {
            'coordinate': 'eta1',
            'formula_in_xi': sp.sstr(sp.simplify((OMEGA * xi_plus_sym - OMEGA_SQ * xi_minus_sym) / (OMEGA - OMEGA_SQ))),
        },
        {
            'coordinate': 'eta2',
            'formula_in_xi': sp.sstr(sp.simplify((xi_plus_sym - xi_minus_sym) / (OMEGA - OMEGA_SQ))),
        },
    ]

    terms, orientation, residual = load_public_rank23_terms()
    _, eta1_matrix, eta2_matrix, _, _ = build_mode_matrices(terms, 3)
    h_matrix = sp.Matrix.hstack(eta1_matrix, eta2_matrix)
    xi_matrix = alpha_tensor_xi_matrix(terms)

    xi_numeric = np.array([[complex(sp.N(value)) for value in row] for row in xi_matrix.tolist()], dtype=np.complex128)
    xi_realification = np.hstack([xi_numeric.real, xi_numeric.imag])
    verification_rows = alpha_tensor_verification_rows(terms)
    counterexample = dead_free_counterexample_row()
    budget_rows = []
    for rank_value in [23, 22, 21, 20, 19]:
        budget_rows.append({
            'R': rank_value,
            'step52_max_nuisance_rank': rank_value - 9,
            'claimed_even_eta_cap_if_theorem_held': 2 * ((rank_value - 9) // 2),
            'alpha_tensor_observed_eta_rank_even': bool(int(h_matrix.rank()) % 2 == 0),
            'usable_as_theorem': False,
        })

    write_csv(
        OUT_DIR / 'fourier_transform_formulas.csv',
        direct_transform_rows,
        ['coordinate', 'formula_in_eta', 'formula_in_xi'],
    )
    write_csv(
        OUT_DIR / 'alpha_tensor_fourier_verification.csv',
        verification_rows,
        ['term_id', 'fiber', 'eta1', 'eta2', 'xi_plus', 'xi_minus', 'xi_plus_formula_holds', 'xi_minus_formula_holds', 'conjugacy_holds'],
    )
    write_csv(
        OUT_DIR / 'budget_table.csv',
        budget_rows,
        ['R', 'step52_max_nuisance_rank', 'claimed_even_eta_cap_if_theorem_held', 'alpha_tensor_observed_eta_rank_even', 'usable_as_theorem'],
    )
    write_json(
        OUT_DIR / 'parity_summary.json',
        {
            'gamma_orientation': orientation,
            'exact_reconstruction_residual': int(residual),
            'symbolic_transform_checks': {
                'xi_plus_equals_eta1_minus_omega_sq_eta2': bool(sp.simplify(xi_plus - (eta1 - OMEGA_SQ * eta2)) == 0),
                'xi_minus_equals_eta1_minus_omega_eta2': bool(sp.simplify(xi_minus - (eta1 - OMEGA * eta2)) == 0),
                'real_factor_conjugacy': bool(sp.simplify(sp.conjugate(xi_plus) - xi_minus) == 0),
            },
            'alpha_tensor_eta_rank_exact': int(h_matrix.rank()),
            'alpha_tensor_xi_plus_complex_rank_exact': int(xi_matrix.rank()),
            'alpha_tensor_xi_realification_rank_numeric': int(np.linalg.matrix_rank(xi_realification)),
            'claimed_even_rank_theorem_holds': False,
            'dead_free_counterexample': counterexample,
        },
    )

    print('Phase 2 parity summary')
    print(f'  transform checks = {all(bool(value) for value in [sp.simplify(xi_plus - (eta1 - OMEGA_SQ * eta2)) == 0, sp.simplify(xi_minus - (eta1 - OMEGA * eta2)) == 0, sp.simplify(sp.conjugate(xi_plus) - xi_minus) == 0])}')
    print(f'  AlphaTensor eta rank = {h_matrix.rank()}')
    print(f'  AlphaTensor xi+ complex rank = {xi_matrix.rank()}')
    print('  claimed even-rank theorem = False')


if __name__ == '__main__':
    main()