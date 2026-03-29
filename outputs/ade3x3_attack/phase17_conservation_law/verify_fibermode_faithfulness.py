from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (  # noqa: E402
    build_mode_matrices_numeric,
    load_public_terms,
    stacked_factors_from_terms,
    write_csv,
    write_json,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    build_mode_matrices,
    gamma_matrix,
)


RANK_TOL = 1e-10


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def bilinear_profile_matrix_numeric(terms: list[Term]) -> np.ndarray:
    alpha, beta, _ = stacked_factors_from_terms(terms)
    return np.einsum('ri,rj->rij', alpha, beta, optimize=True).reshape(alpha.shape[0], 81)


def gamma_matrix_numeric(terms: list[Term]) -> np.ndarray:
    _, _, gamma = stacked_factors_from_terms(terms)
    return gamma.T


def all_integer_terms(terms: list[Term]) -> bool:
    for term in terms:
        for factor in (term.alpha, term.beta, term.gamma):
            rounded = np.rint(np.asarray(factor, dtype=np.float64))
            if not np.allclose(rounded, np.asarray(factor, dtype=np.float64)):
                return False
    return True


def bilinear_profile_matrix_exact(terms: list[Term]) -> sp.Matrix:
    rows: list[list[int]] = []
    for term in terms:
        alpha = np.asarray(term.alpha, dtype=np.int64).reshape(-1)
        beta = np.asarray(term.beta, dtype=np.int64).reshape(-1)
        rows.append([int(alpha[i] * beta[j]) for i in range(9) for j in range(9)])
    return sp.Matrix(rows)


def standard_rank27_terms() -> list[Term]:
    terms: list[Term] = []
    term_idx = 1
    for row_idx in range(3):
        for shared_idx in range(3):
            for col_idx in range(3):
                alpha = np.zeros((3, 3), dtype=np.int64)
                beta = np.zeros((3, 3), dtype=np.int64)
                gamma = np.zeros((3, 3), dtype=np.int64)
                alpha[row_idx, shared_idx] = 1
                beta[shared_idx, col_idx] = 1
                gamma[row_idx, col_idx] = 1
                terms.append(Term(f's{term_idx:02d}', f'standard_rank27_{term_idx:02d}', alpha, beta, gamma))
                term_idx += 1
    return terms


def perturb_terms(base_terms: list[Term], seed: int, sigma: float = 0.01) -> list[Term]:
    rng = np.random.default_rng(seed)
    terms: list[Term] = []
    for term in base_terms:
        alpha = np.asarray(term.alpha, dtype=np.float64) + sigma * rng.standard_normal((3, 3))
        beta = np.asarray(term.beta, dtype=np.float64) + sigma * rng.standard_normal((3, 3))
        gamma = np.asarray(term.gamma, dtype=np.float64)
        terms.append(Term(term.term_id, term.source_label, alpha, beta, gamma))
    return terms


def random_rank1_terms(rank_value: int, seed: int) -> list[Term]:
    rng = np.random.default_rng(seed)
    terms: list[Term] = []
    for term_idx in range(rank_value):
        alpha = rng.standard_normal((3, 3))
        beta = rng.standard_normal((3, 3))
        gamma = rng.standard_normal((3, 3))
        terms.append(Term(f'r{rank_value}_{term_idx + 1:02d}', f'random_rank{rank_value}_{seed}_{term_idx + 1:02d}', alpha, beta, gamma))
    return terms


def evaluate_collection(label: str, category: str, terms: list[Term], valid_decomposition: bool) -> dict[str, object]:
    sigma_num, eta1_num, eta2_num, delta_num, h_num, _ = build_mode_matrices_numeric(terms)
    stacked_num = np.hstack([sigma_num, h_num, delta_num])
    bilinear_num = bilinear_profile_matrix_numeric(terms)
    gamma_num = gamma_matrix_numeric(terms)
    rank_stacked_num = matrix_rank_numeric(stacked_num)
    rank_bilinear_num = matrix_rank_numeric(bilinear_num)
    rank_h_num = matrix_rank_numeric(h_num)
    rank_hdelta_num = matrix_rank_numeric(np.hstack([h_num, delta_num]))
    rank_gamma_num = matrix_rank_numeric(gamma_num)
    row: dict[str, object] = {
        'label': label,
        'category': category,
        'valid_decomposition': valid_decomposition,
        'R': len(terms),
        'rank_sigma_numeric': matrix_rank_numeric(sigma_num),
        'rank_H_numeric': rank_h_num,
        'rank_HDelta_numeric': rank_hdelta_num,
        'rank_stacked_numeric': rank_stacked_num,
        'rank_bilinear_numeric': rank_bilinear_num,
        'faithfulness_numeric': rank_stacked_num == len(terms) and rank_bilinear_num == len(terms),
        'rank_gamma_numeric': rank_gamma_num,
        'ker_gamma_dim_numeric': len(terms) - rank_gamma_num,
        'corollary_numeric': (len(terms) - rank_gamma_num) == rank_hdelta_num if valid_decomposition else '',
        'eta_nullity_numeric': 18 - rank_h_num,
        'conservation_sum_numeric': len(terms) + (18 - rank_h_num) if valid_decomposition else '',
    }

    if all_integer_terms(terms):
        sigma_exact, eta1_exact, eta2_exact, delta_exact, _ = build_mode_matrices(terms, 3)
        h_exact = sp.Matrix.hstack(eta1_exact, eta2_exact)
        stacked_exact = sp.Matrix.hstack(sigma_exact, h_exact, delta_exact)
        bilinear_exact = bilinear_profile_matrix_exact(terms)
        gamma_exact = gamma_matrix(terms, 3)
        rank_h_exact = int(h_exact.rank())
        rank_hdelta_exact = int(sp.Matrix.hstack(h_exact, delta_exact).rank())
        rank_gamma_exact = int(gamma_exact.rank())
        row.update(
            {
                'rank_sigma_exact': int(sigma_exact.rank()),
                'rank_H_exact': rank_h_exact,
                'rank_HDelta_exact': rank_hdelta_exact,
                'rank_stacked_exact': int(stacked_exact.rank()),
                'rank_bilinear_exact': int(bilinear_exact.rank()),
                'faithfulness_exact': int(stacked_exact.rank()) == len(terms) and int(bilinear_exact.rank()) == len(terms),
                'rank_gamma_exact': rank_gamma_exact,
                'ker_gamma_dim_exact': len(terms) - rank_gamma_exact,
                'corollary_exact': (len(terms) - rank_gamma_exact) == rank_hdelta_exact if valid_decomposition else '',
                'eta_nullity_exact': 18 - rank_h_exact,
                'conservation_sum_exact': len(terms) + (18 - rank_h_exact) if valid_decomposition else '',
            }
        )

    return row


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    valid_rows = [row for row in rows if bool(row['valid_decomposition'])]
    perturb_rows = [row for row in rows if row['category'] == 'perturbed_alphatensor']
    random_rows = [row for row in rows if row['category'] == 'random_rank1_collection']

    by_rank: dict[str, dict[str, object]] = {}
    for rank_value in (9, 14, 18, 23, 27):
        rank_rows = [row for row in random_rows if int(row['R']) == rank_value]
        by_rank[str(rank_value)] = {
            'trial_count': len(rank_rows),
            'faithful_trial_count': sum(bool(row['faithfulness_numeric']) for row in rank_rows),
        }

    return {
        'valid_decompositions': {
            'labels': [str(row['label']) for row in valid_rows],
            'all_faithful_numeric': all(bool(row['faithfulness_numeric']) for row in valid_rows),
            'all_corollary_holds_numeric': all(bool(row['corollary_numeric']) for row in valid_rows),
            'rows': valid_rows,
        },
        'perturbed_alphatensor': {
            'trial_count': len(perturb_rows),
            'all_faithful_numeric': all(bool(row['faithfulness_numeric']) for row in perturb_rows),
            'min_rank_stacked_numeric': min(int(row['rank_stacked_numeric']) for row in perturb_rows),
            'max_rank_stacked_numeric': max(int(row['rank_stacked_numeric']) for row in perturb_rows),
        },
        'random_rank1_collections': {
            'trial_count': len(random_rows),
            'all_faithful_numeric': all(bool(row['faithfulness_numeric']) for row in random_rows),
            'by_rank': by_rank,
        },
    }


def main() -> None:
    rows: list[dict[str, object]] = []

    alpha_terms, orientation, residual = load_public_terms()
    standard_terms = standard_rank27_terms()

    rows.append(evaluate_collection(f'alphatensor_rank23_valid_{orientation}', 'valid_decomposition', alpha_terms, True))
    rows[-1]['orientation'] = orientation
    rows[-1]['reconstruction_max_abs'] = residual

    rows.append(evaluate_collection('standard_rank27_valid', 'valid_decomposition', standard_terms, True))
    rows[-1]['orientation'] = 'direct'
    rows[-1]['reconstruction_max_abs'] = 0

    for seed in range(42, 62):
        perturbed = perturb_terms(alpha_terms, seed=seed, sigma=0.01)
        row = evaluate_collection(f'perturbed_alphatensor_seed_{seed}', 'perturbed_alphatensor', perturbed, False)
        row['seed'] = seed
        row['sigma'] = 0.01
        rows.append(row)

    for rank_value in (9, 14, 18, 23, 27):
        for trial_idx in range(5):
            seed = 20260329 + 100 * rank_value + trial_idx
            random_terms = random_rank1_terms(rank_value, seed=seed)
            row = evaluate_collection(
                f'random_rank{rank_value}_trial_{trial_idx + 1}',
                'random_rank1_collection',
                random_terms,
                False,
            )
            row['seed'] = seed
            row['trial_index'] = trial_idx + 1
            rows.append(row)

    fieldnames = [
        'label', 'category', 'valid_decomposition', 'R', 'seed', 'sigma', 'trial_index',
        'orientation', 'reconstruction_max_abs',
        'rank_sigma_numeric', 'rank_H_numeric', 'rank_HDelta_numeric', 'rank_stacked_numeric',
        'rank_bilinear_numeric', 'faithfulness_numeric', 'rank_gamma_numeric', 'ker_gamma_dim_numeric',
        'corollary_numeric', 'eta_nullity_numeric', 'conservation_sum_numeric',
        'rank_sigma_exact', 'rank_H_exact', 'rank_HDelta_exact', 'rank_stacked_exact',
        'rank_bilinear_exact', 'faithfulness_exact', 'rank_gamma_exact', 'ker_gamma_dim_exact',
        'corollary_exact', 'eta_nullity_exact', 'conservation_sum_exact',
    ]
    write_csv(OUT_DIR / 'faithfulness_summary.csv', rows, fieldnames)
    payload = summarize(rows)
    write_json(OUT_DIR / 'faithfulness_summary.json', payload)

    print('Phase 17A complete.')
    print(f"  Valid decompositions faithful: {payload['valid_decompositions']['all_faithful_numeric']}")
    print(f"  Valid decomposition corollary holds: {payload['valid_decompositions']['all_corollary_holds_numeric']}")
    print(f"  Perturbed AlphaTensor trials faithful: {payload['perturbed_alphatensor']['all_faithful_numeric']}")
    print(f"  Random generic trials faithful: {payload['random_rank1_collections']['all_faithful_numeric']}")


if __name__ == '__main__':
    main()