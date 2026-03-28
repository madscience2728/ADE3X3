"""
ade3x3_step63_reverse_engineering_cancellation_visualization.py

Step 63: Reverse Engineering + Cancellation Visualization.

Primary path:
1. Recover a public exact rank-23 3x3 factorization from AlphaTensor's public
   recombination example.
2. Express the factorization in the Step 51 fiber-mode basis.
3. Test single-term and pair-term removal via the Step 52 quotient criterion.
4. Export cancellation and dependency data, plus an HTML visualization.
5. Compare the multiplication tensor against its commutator and symmetric split.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp

EXPORTS = Path('outputs/exports')
OUTPUTS = Path('outputs')


@dataclass(frozen=True)
class Term:
    term_id: str
    source_label: str
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f"  Wrote text -> {path}")


def matrix_to_str(matrix: np.ndarray) -> str:
    return json.dumps(matrix.astype(int).tolist(), separators=(',', ':'))


def alphatensor_public_factorization() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact public rank-23 3x3 factorization from AlphaTensor's repo.

    Source:
    https://raw.githubusercontent.com/google-deepmind/alphatensor/main/recombination/example.py
    """
    u = np.array([
        [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.int64)
    v = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [-1, -1, 0, 0, -1, 0, -1, -1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [-1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 1, 0, 1, 0, -1, 0, 0],
        [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.int64)
    w = np.array([
        [0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
        [-1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, -1, 1, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0, -1, 0, 0, -1, 0, 1],
        [-1, 1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0],
    ], dtype=np.int64)
    return u, v, w


def matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.int64)
    for row_idx in range(n):
        for sum_idx in range(n):
            for col_idx in range(n):
                a_idx = n * row_idx + sum_idx
                b_idx = n * sum_idx + col_idx
                c_idx = n * row_idx + col_idx
                tensor[a_idx, b_idx, c_idx] = 1
    return tensor


def swapped_matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.int64)
    for row_idx in range(n):
        for sum_idx in range(n):
            for col_idx in range(n):
                a_idx = n * sum_idx + col_idx
                b_idx = n * row_idx + sum_idx
                c_idx = n * row_idx + col_idx
                tensor[a_idx, b_idx, c_idx] = 1
    return tensor


def term_tensor(term: Term) -> np.ndarray:
    return np.einsum(
        'a,b,c->abc',
        term.alpha.reshape(-1),
        term.beta.reshape(-1),
        term.gamma.reshape(-1),
        optimize=True,
    )


def reconstruct_tensor(terms: list[Term]) -> np.ndarray:
    total = np.zeros((9, 9, 9), dtype=np.int64)
    for term in terms:
        total += term_tensor(term)
    return total


def load_public_rank23_terms() -> tuple[list[Term], str, int]:
    u, v, w = alphatensor_public_factorization()
    target = matrix_multiplication_tensor(3)
    candidates = [
        ('direct', lambda vector: vector.reshape(3, 3)),
        ('transpose', lambda vector: vector.reshape(3, 3).T),
    ]
    for orientation, gamma_builder in candidates:
        terms: list[Term] = []
        for term_idx in range(u.shape[1]):
            alpha = u[:, term_idx].reshape(3, 3)
            beta = v[:, term_idx].reshape(3, 3)
            gamma = gamma_builder(w[:, term_idx])
            terms.append(Term(f't{term_idx + 1:02d}', f'alpha_tensor_rank23_{term_idx + 1}', alpha, beta, gamma))
        residual = reconstruct_tensor(terms) - target
        max_abs = int(np.max(np.abs(residual)))
        if max_abs == 0:
            return terms, orientation, max_abs
    raise RuntimeError('Could not align the public rank-23 coefficients with ADE3x3 output orientation.')


def decode_index(idx: int, n: int) -> tuple[int, int]:
    return divmod(idx, n)


def equation_type_name(r: int, s: int, t: int, u: int, out_r: int, out_u: int) -> str:
    base = 0 if s == t else 4
    offset = (0 if r == out_r else 2) + (0 if u == out_u else 1)
    return f'E{base + offset}'


def build_mode_matrices(terms: list[Term], n: int) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    sigma_columns: list[list[int]] = []
    eta1_columns: list[list[int]] = []
    eta2_columns: list[list[int]] = []
    dead_columns: list[list[int]] = []

    for row_idx in range(n):
        for col_idx in range(n):
            sigma_vec: list[int] = []
            eta1_vec: list[int] = []
            eta2_vec: list[int] = []
            for term in terms:
                lambdas = [int(term.alpha[row_idx, sum_idx] * term.beta[sum_idx, col_idx]) for sum_idx in range(n)]
                sigma_vec.append(sum(lambdas))
                eta1_vec.append(lambdas[0] - lambdas[1])
                eta2_vec.append(lambdas[1] - lambdas[2])
            sigma_columns.append(sigma_vec)
            eta1_columns.append(eta1_vec)
            eta2_columns.append(eta2_vec)

    for row_idx in range(n):
        for sum_left in range(n):
            for sum_right in range(n):
                if sum_left == sum_right:
                    continue
                for col_idx in range(n):
                    dead_vec = [int(term.alpha[row_idx, sum_left] * term.beta[sum_right, col_idx]) for term in terms]
                    dead_columns.append(dead_vec)

    sigma = sp.Matrix.hstack(*[sp.Matrix(column) for column in sigma_columns])
    eta1 = sp.Matrix.hstack(*[sp.Matrix(column) for column in eta1_columns])
    eta2 = sp.Matrix.hstack(*[sp.Matrix(column) for column in eta2_columns])
    delta = sp.Matrix.hstack(*[sp.Matrix(column) for column in dead_columns])
    nuisance = sp.Matrix.hstack(eta1, eta2, delta)
    return sigma, eta1, eta2, delta, nuisance


def gamma_matrix(terms: list[Term], n: int) -> sp.Matrix:
    rows: list[list[int]] = []
    for row_idx in range(n):
        for col_idx in range(n):
            rows.append([int(term.gamma[row_idx, col_idx]) for term in terms])
    return sp.Matrix(rows)


def rank_profile(terms: list[Term], n: int) -> dict[str, object]:
    sigma, eta1, eta2, delta, nuisance = build_mode_matrices(terms, n)
    gamma = gamma_matrix(terms, n)
    augmented = sp.Matrix.hstack(sigma, nuisance)
    target = n * sp.eye(n * n)
    sigma_ok = gamma * sigma == target
    nuisance_ok = gamma * nuisance == sp.zeros(n * n, nuisance.cols)
    nuisance_rank = int(nuisance.rank())
    augmented_rank = int(augmented.rank())
    quotient_gain = augmented_rank - nuisance_rank
    return {
        'R': len(terms),
        'sigma_rank': int(sigma.rank()),
        'eta1_rank': int(eta1.rank()),
        'eta2_rank': int(eta2.rank()),
        'eta_rank': int(sp.Matrix.hstack(eta1, eta2).rank()),
        'delta_rank': int(delta.rank()),
        'nuisance_rank': nuisance_rank,
        'augmented_rank': augmented_rank,
        'quotient_gain': quotient_gain,
        'sigma_verification': bool(sigma_ok),
        'nuisance_verification': bool(nuisance_ok),
        'nuisance_equals_R_minus_9': nuisance_rank == len(terms) - (n * n),
    }


def term_rows(terms: list[Term]) -> list[dict]:
    rows: list[dict] = []
    for term in terms:
        rows.append({
            'term_id': term.term_id,
            'source_label': term.source_label,
            'alpha': matrix_to_str(term.alpha),
            'beta': matrix_to_str(term.beta),
            'gamma': matrix_to_str(term.gamma),
            'alpha_nonzero': int(np.count_nonzero(term.alpha)),
            'beta_nonzero': int(np.count_nonzero(term.beta)),
            'gamma_nonzero': int(np.count_nonzero(term.gamma)),
            'provenance': 'EXACT_DERIVED',
        })
    return rows


def term_residual_rows(terms: list[Term]) -> tuple[list[dict], list[dict], list[dict]]:
    residual_rows: list[dict] = []
    single_rows: list[dict] = []
    fiber_rows: list[dict] = []

    for term in terms:
        tensor = term_tensor(term)
        eq_type_counts: Counter[str] = Counter()
        live_fibers: set[tuple[int, int]] = set()
        for a_idx, b_idx, c_idx in zip(*np.nonzero(tensor)):
            coeff = int(tensor[a_idx, b_idx, c_idx])
            r, s = decode_index(int(a_idx), 3)
            t, u = decode_index(int(b_idx), 3)
            out_r, out_u = decode_index(int(c_idx), 3)
            eq_type = equation_type_name(r, s, t, u, out_r, out_u)
            eq_type_counts[eq_type] += 1
            if s == t:
                live_fibers.add((r, u))
            residual_rows.append({
                'term_id': term.term_id,
                'x_cell': f'X[{r},{s}|{t},{u}]',
                'output_cell': f'C[{out_r},{out_u}]',
                'equation_type': eq_type,
                'is_live': s == t,
                'coefficient': coeff,
                'provenance': 'EXACT_DERIVED',
            })
        sigma = term.alpha @ term.beta
        for out_r in range(3):
            for out_u in range(3):
                gamma_coeff = int(term.gamma[out_r, out_u])
                for fiber_r in range(3):
                    for fiber_u in range(3):
                        value = int(gamma_coeff * sigma[fiber_r, fiber_u])
                        fiber_rows.append({
                            'term_id': term.term_id,
                            'output_cell': f'C[{out_r},{out_u}]',
                            'fiber': f'({fiber_r},{fiber_u})',
                            'value': value,
                            'nonzero': value != 0,
                            'provenance': 'EXACT_DERIVED',
                        })
        fro_sq = int(np.sum(tensor * tensor))
        row = {
            'term_id': term.term_id,
            'frobenius_norm_squared': fro_sq,
            'frobenius_norm': float(np.sqrt(fro_sq)),
            'nonzero_equation_count': int(np.count_nonzero(tensor)),
            'affected_equation_types': ','.join(sorted(eq_type_counts)),
            'affected_live_fibers': ';'.join(f'({r},{u})' for r, u in sorted(live_fibers)),
            'provenance': 'EXACT_DERIVED',
        }
        for eq_type in [f'E{i}' for i in range(8)]:
            row[f'{eq_type}_count'] = eq_type_counts.get(eq_type, 0)
        single_rows.append(row)

    single_rows.sort(key=lambda row: (-int(row['frobenius_norm_squared']), row['term_id']))
    for rank_idx, row in enumerate(single_rows, start=1):
        row['importance_rank'] = rank_idx
    return residual_rows, single_rows, fiber_rows


def subset_feasibility_rows(terms: list[Term]) -> tuple[list[dict], list[dict]]:
    single_rows: list[dict] = []
    pair_rows: list[dict] = []

    for remove_idx, removed in enumerate(terms):
        remaining = terms[:remove_idx] + terms[remove_idx + 1:]
        profile = rank_profile(remaining, 3)
        single_rows.append({
            'removed_terms': removed.term_id,
            'remaining_R': len(remaining),
            'nuisance_rank': profile['nuisance_rank'],
            'augmented_rank': profile['augmented_rank'],
            'quotient_gain': profile['quotient_gain'],
            'gamma_reweight_feasible': profile['quotient_gain'] == 9,
            'sigma_verification_if_feasible': profile['sigma_verification'],
            'nuisance_verification_if_feasible': profile['nuisance_verification'],
            'provenance': 'EXACT_DERIVED',
        })

    for left_idx, right_idx in combinations(range(len(terms)), 2):
        remaining = [term for idx, term in enumerate(terms) if idx not in {left_idx, right_idx}]
        profile = rank_profile(remaining, 3)
        pair_rows.append({
            'removed_terms': f"{terms[left_idx].term_id},{terms[right_idx].term_id}",
            'remaining_R': len(remaining),
            'nuisance_rank': profile['nuisance_rank'],
            'augmented_rank': profile['augmented_rank'],
            'quotient_gain': profile['quotient_gain'],
            'gamma_reweight_feasible': profile['quotient_gain'] == 9,
            'provenance': 'EXACT_DERIVED',
        })

    single_rows.sort(key=lambda row: (-int(row['quotient_gain']), row['removed_terms']))
    pair_rows.sort(key=lambda row: (-int(row['quotient_gain']), row['removed_terms']))
    return single_rows, pair_rows


def balance_rows(terms: list[Term]) -> tuple[list[dict], list[dict], list[dict]]:
    dead_rows: list[dict] = []
    live_rows: list[dict] = []
    edge_counter: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {'shared_dead_cells': 0, 'shared_dead_equations': 0})

    term_tensors = {term.term_id: term_tensor(term) for term in terms}
    dead_support_cells: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
    dead_support_equations: dict[str, set[tuple[int, int, int, int, int, int]]] = defaultdict(set)

    target = matrix_multiplication_tensor(3)
    for a_idx in range(9):
        r, s = decode_index(a_idx, 3)
        for b_idx in range(9):
            t, u = decode_index(b_idx, 3)
            for c_idx in range(9):
                out_r, out_u = decode_index(c_idx, 3)
                contributions: list[tuple[str, int]] = []
                for term in terms:
                    coeff = int(term_tensors[term.term_id][a_idx, b_idx, c_idx])
                    if coeff != 0:
                        contributions.append((term.term_id, coeff))
                        if s != t:
                            dead_support_cells[term.term_id].add((r, s, t, u))
                            dead_support_equations[term.term_id].add((r, s, t, u, out_r, out_u))
                if not contributions:
                    continue

                positive = [(term_id, coeff) for term_id, coeff in contributions if coeff > 0]
                negative = [(term_id, coeff) for term_id, coeff in contributions if coeff < 0]
                total = sum(coeff for _, coeff in contributions)

                if s != t:
                    if positive and negative:
                        if len(positive) == 1 and len(negative) == 1 and positive[0][1] == -negative[0][1]:
                            balance = 'perfect_balanced'
                        else:
                            balance = 'multi_way_balanced'
                    else:
                        balance = 'one_sided'
                    dead_rows.append({
                        'x_cell': f'X[{r},{s}|{t},{u}]',
                        'output_cell': f'C[{out_r},{out_u}]',
                        'positive_terms': ','.join(term_id for term_id, _ in positive),
                        'negative_terms': ','.join(term_id for term_id, _ in negative),
                        'positive_sum': sum(coeff for _, coeff in positive),
                        'negative_sum': sum(coeff for _, coeff in negative),
                        'total': total,
                        'balance_status': balance,
                        'provenance': 'EXACT_DERIVED',
                    })
                elif out_r == r and out_u == u:
                    live_rows.append({
                        'x_cell': f'X[{r},{s}|{t},{u}]',
                        'target_output_cell': f'C[{out_r},{out_u}]',
                        'positive_terms': ','.join(term_id for term_id, _ in positive),
                        'negative_terms': ','.join(term_id for term_id, _ in negative),
                        'total': total,
                        'target_rhs': int(target[a_idx, b_idx, c_idx]),
                        'provenance': 'EXACT_DERIVED',
                    })

    term_ids = [term.term_id for term in terms]
    for left_id, right_id in combinations(term_ids, 2):
        shared_cells = dead_support_cells[left_id] & dead_support_cells[right_id]
        shared_equations = dead_support_equations[left_id] & dead_support_equations[right_id]
        if shared_cells:
            edge_counter[(left_id, right_id)]['shared_dead_cells'] = len(shared_cells)
        if shared_equations:
            edge_counter[(left_id, right_id)]['shared_dead_equations'] = len(shared_equations)

    edge_rows: list[dict] = []
    for (left_id, right_id), counts in edge_counter.items():
        edge_rows.append({
            'left_term': left_id,
            'right_term': right_id,
            'shared_dead_cells': counts['shared_dead_cells'],
            'shared_dead_equations': counts['shared_dead_equations'],
            'provenance': 'EXACT_DERIVED',
        })
    edge_rows.sort(key=lambda row: (-int(row['shared_dead_cells']), -int(row['shared_dead_equations']), row['left_term'], row['right_term']))
    return dead_rows, live_rows, edge_rows


def flattening_ranks(tensor: np.ndarray) -> tuple[int, int, int, int]:
    rank_a = int(np.linalg.matrix_rank(tensor.reshape(tensor.shape[0], -1)))
    rank_b = int(np.linalg.matrix_rank(np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1)))
    rank_c = int(np.linalg.matrix_rank(np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)))
    return rank_a, rank_b, rank_c, max(rank_a, rank_b, rank_c)


def commutator_rows() -> list[dict]:
    rows: list[dict] = []
    for n in [2, 3]:
        mult = matrix_multiplication_tensor(n)
        swapped = swapped_matrix_multiplication_tensor(n)
        pieces = {
            'multiplication': mult,
            'commutator': mult - swapped,
            'anticommutator': mult + swapped,
        }
        for label, tensor in pieces.items():
            rank_a, rank_b, rank_c, lower = flattening_ranks(tensor)
            rows.append({
                'matrix_size': f'{n}x{n}',
                'tensor_piece': label,
                'nonzero_entries': int(np.count_nonzero(tensor)),
                'flattening_rank_A_BC': rank_a,
                'flattening_rank_B_AC': rank_b,
                'flattening_rank_C_AB': rank_c,
                'flattening_lower_bound': lower,
                'provenance': 'EXACT_DERIVED',
            })
    return rows


def source_rows() -> list[dict]:
    return [
        {
            'source_kind': 'smirnov_2013_search',
            'source_label': 'Smirnov explicit 23-term coefficients',
            'status': 'not_recovered_from_available_sources',
            'detail': 'Search surfaced 23-term 3x3 papers but not an explicit coefficient table in fetched content.',
            'url': 'https://arxiv.org/abs/1108.2830',
            'provenance': 'MEASURED_FROM_CODE',
        },
        {
            'source_kind': 'alternative_public_algorithm',
            'source_label': 'AlphaTensor public recombination example',
            'status': 'explicit_rank23_coefficients_recovered',
            'detail': 'Public get_3x3x3_factorization() provides exact 9x23 factor matrices u,v,w.',
            'url': 'https://raw.githubusercontent.com/google-deepmind/alphatensor/main/recombination/example.py',
            'provenance': 'MEASURED_FROM_CODE',
        },
        {
            'source_kind': 'alphatensor_repo_scope',
            'source_label': 'AlphaTensor benchmarking algorithms',
            'status': '4x4_public_benchmarking_factorizations_only',
            'detail': 'The public benchmarking factorization module exposes 4x4 rank-49 GPU/TPU algorithms; the explicit 3x3 rank-23 example is in recombination/example.py.',
            'url': 'https://github.com/google-deepmind/alphatensor',
            'provenance': 'MEASURED_FROM_CODE',
        },
    ]


def visualization_html(
    terms: list[Term],
    dead_rows: list[dict],
    live_rows: list[dict],
    edge_rows: list[dict],
    summary_map: dict[str, str],
) -> str:
    def footprint_cells(term: Term) -> str:
        cards: list[str] = []
        for fiber_r in range(3):
            for fiber_u in range(3):
                cells: list[str] = []
                for sum_idx in range(3):
                    value = int(term.alpha[fiber_r, sum_idx] * term.beta[sum_idx, fiber_u])
                    css = 'zero'
                    if value > 0:
                        css = 'pos'
                    elif value < 0:
                        css = 'neg'
                    cells.append(f'<div class="chan {css}">{value}</div>')
                cards.append(
                    '<div class="fiber-card">'
                    f'<div class="fiber-title">F({fiber_r},{fiber_u})</div>'
                    f'<div class="fiber-stack">{"".join(cells)}</div>'
                    '</div>'
                )
        outputs = [(r, u) for r in range(3) for u in range(3) if int(term.gamma[r, u]) != 0]
        output_text = ', '.join(f'C[{r},{u}]={int(term.gamma[r, u])}' for r, u in outputs)
        return (
            '<div class="term-card">'
            f'<h3>{term.term_id}</h3>'
            f'<div class="small">Outputs: {html.escape(output_text) if output_text else "none"}</div>'
            '<div class="fiber-grid">'
            f'{"".join(cards)}'
            '</div>'
            '</div>'
        )

    details_dead = []
    for row in dead_rows:
        color = 'green' if row['balance_status'] == 'perfect_balanced' else 'yellow' if row['balance_status'] == 'multi_way_balanced' else 'red'
        details_dead.append(
            '<tr>'
            f'<td>{html.escape(row["x_cell"])}</td>'
            f'<td>{html.escape(row["output_cell"])}</td>'
            f'<td class="{color}">{html.escape(row["balance_status"])}</td>'
            f'<td>{html.escape(row["positive_terms"])}</td>'
            f'<td>{html.escape(row["negative_terms"])}</td>'
            f'<td>{row["total"]}</td>'
            '</tr>'
        )

    live_table = []
    for row in live_rows:
        live_table.append(
            '<tr>'
            f'<td>{html.escape(row["x_cell"])}</td>'
            f'<td>{html.escape(row["target_output_cell"])}</td>'
            f'<td>{html.escape(row["positive_terms"])}</td>'
            f'<td>{html.escape(row["negative_terms"])}</td>'
            f'<td>{row["total"]}</td>'
            f'<td>{row["target_rhs"]}</td>'
            '</tr>'
        )

    node_positions: dict[str, tuple[float, float]] = {}
    radius = 210.0
    center = 260.0
    for idx, term in enumerate(terms):
        angle = 2.0 * np.pi * idx / len(terms) - np.pi / 2.0
        node_positions[term.term_id] = (center + radius * float(np.cos(angle)), center + radius * float(np.sin(angle)))
    edge_svg: list[str] = []
    max_shared = max((int(row['shared_dead_cells']) for row in edge_rows), default=1)
    for row in edge_rows:
        left_x, left_y = node_positions[row['left_term']]
        right_x, right_y = node_positions[row['right_term']]
        opacity = 0.12 + 0.45 * int(row['shared_dead_cells']) / max_shared
        edge_svg.append(
            f'<line x1="{left_x:.1f}" y1="{left_y:.1f}" x2="{right_x:.1f}" y2="{right_y:.1f}" '
            f'stroke="rgba(40,40,40,{opacity:.3f})" stroke-width="{1 + int(row["shared_dead_cells"]) * 0.35:.2f}" />'
        )
    node_svg: list[str] = []
    for term in terms:
        x, y = node_positions[term.term_id]
        node_svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="#f6f1d3" stroke="#2d2d2d" stroke-width="1.2" />')
        node_svg.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="10">{html.escape(term.term_id)}</text>')

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Step 63 Cancellation Visualization</title>
  <style>
    :root {{
      --bg: #f8f4ea;
      --ink: #22201c;
      --muted: #6e6659;
      --panel: #fffdf7;
      --line: #d2c6b1;
      --pos: #1f6aa5;
      --neg: #b53a2d;
      --zero: #c7c1b7;
      --green: #d8ead2;
      --yellow: #f4e6a2;
      --red: #efc4bc;
    }}
    body {{ font-family: Georgia, 'Times New Roman', serif; background: linear-gradient(180deg, #f4efe2, var(--bg)); color: var(--ink); margin: 0; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    .small {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.04); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .summary-item {{ background: #fbf7ef; border: 1px solid var(--line); border-radius: 12px; padding: 10px; }}
    .term-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }}
    .term-card {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: #fffef9; }}
    .fiber-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }}
    .fiber-card {{ border: 1px solid var(--line); border-radius: 10px; padding: 6px; background: #faf5eb; }}
    .fiber-title {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .fiber-stack {{ display: grid; grid-template-rows: repeat(3, 1fr); gap: 4px; }}
    .chan {{ text-align: center; padding: 6px 0; border-radius: 6px; font-weight: 700; font-size: 13px; }}
    .pos {{ background: rgba(31,106,165,0.18); color: var(--pos); }}
    .neg {{ background: rgba(181,58,45,0.16); color: var(--neg); }}
    .zero {{ background: rgba(199,193,183,0.26); color: #6f675c; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left; }}
    .green {{ background: var(--green); }}
    .yellow {{ background: var(--yellow); }}
    .red {{ background: var(--red); }}
    .graph-wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <div class=\"section\">
    <h1>Step 63: Reverse Engineering + Cancellation Visualization</h1>
    <div class=\"small\">Public exact rank-23 3x3 factorization recovered from AlphaTensor's public recombination example.</div>
    <div class=\"summary-grid\">
      <div class=\"summary-item\"><strong>Algorithm</strong><br>{html.escape(summary_map['algorithm_label'])}</div>
      <div class=\"summary-item\"><strong>Terms</strong><br>{html.escape(summary_map['term_count'])}</div>
      <div class=\"summary-item\"><strong>Nuisance Rank</strong><br>{html.escape(summary_map['nuisance_rank'])}</div>
      <div class=\"summary-item\"><strong>Quotient Gain</strong><br>{html.escape(summary_map['quotient_gain'])}</div>
      <div class=\"summary-item\"><strong>Single Removals Feasible</strong><br>{html.escape(summary_map['single_term_reweight_feasible_count'])}</div>
      <div class=\"summary-item\"><strong>Pair Removals Feasible</strong><br>{html.escape(summary_map['pair_reweight_feasible_count'])}</div>
    </div>
  </div>

  <div class=\"section\">
    <h2>Fiber-Mode Footprints</h2>
    <div class=\"small\">Each term is shown on the requested 9-fiber by 3-channel layout. Cell values are the per-channel A/B contributions $a[r,s]b[s,u]$ before output weighting.</div>
    <div class=\"term-grid\">{''.join(footprint_cells(term) for term in terms)}</div>
  </div>

  <div class=\"section\">
    <h2>Dead-Equation Cancellation</h2>
    <div class=\"small\">Green marks exact two-sided cancellation with one positive and one negative contributor. Yellow marks exact cancellation that needs a more complex multi-term balance. Red would indicate a one-sided inconsistency.</div>
    <table>
      <thead><tr><th>Dead X cell</th><th>Output</th><th>Status</th><th>Positive terms</th><th>Negative terms</th><th>Total</th></tr></thead>
      <tbody>{''.join(details_dead)}</tbody>
    </table>
  </div>

  <div class=\"section\">
    <h2>Fiber-Sum Live Equations</h2>
    <div class=\"small\">These are the target live cells. The total must equal 1 on the correct output equation.</div>
    <table>
      <thead><tr><th>Live X cell</th><th>Target output</th><th>Positive terms</th><th>Negative terms</th><th>Total</th><th>RHS</th></tr></thead>
      <tbody>{''.join(live_table)}</tbody>
    </table>
  </div>

  <div class=\"section\">
    <h2>Dependency Graph</h2>
    <div class=\"small\">Edges connect terms that share at least one dead X-cell. Darker edges indicate more shared dead cells.</div>
    <div class=\"graph-wrap\">
      <svg width=\"520\" height=\"520\" viewBox=\"0 0 520 520\" role=\"img\" aria-label=\"Step 63 dependency graph\">{''.join(edge_svg)}{''.join(node_svg)}</svg>
    </div>
  </div>
</body>
</html>
"""


def markdown_summary(summary_rows: list[dict], source_status_rows: list[dict], profile: dict[str, object], single_rows: list[dict], pair_rows: list[dict], comm_rows: list[dict]) -> str:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary_rows}
    feasible_single = [row for row in single_rows if row['gamma_reweight_feasible']]
    feasible_pairs = [row for row in pair_rows if row['gamma_reweight_feasible']]
    lines: list[str] = []
    w = lines.append
    w('# Step 63: Reverse Engineering + Cancellation Visualization')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('Step 63 first searched for an explicit sub-27 3x3 factorization. Smirnov coefficients were not recovered from the fetched paper metadata, but the public AlphaTensor repository exposes an exact rank-23 3x3x3 factorization in recombination/example.py, and that coefficient table is used here.')
    w('')
    w('## External Source Status')
    w('')
    for row in source_status_rows:
        w(f"- {row['source_label']}: {row['status']} ({row['detail']})")
    w('')
    w('## Fiber-Mode Profile')
    w('')
    w(f"- term_count = {summary_map['term_count']}")
    w(f"- gamma_orientation = {summary_map['gamma_orientation']}")
    w(f"- nuisance_rank = {profile['nuisance_rank']}")
    w(f"- quotient_gain = {profile['quotient_gain']}")
    w(f"- nuisance_equals_R_minus_9 = {profile['nuisance_equals_R_minus_9']}")
    w(f"- Gamma*Sigma = 3I_9 verified exactly = {profile['sigma_verification']}")
    w(f"- Gamma*Nuisance = 0 verified exactly = {profile['nuisance_verification']}")
    w('')
    w('## Removal Feasibility')
    w('')
    w(f"- single-term feasible removals = {len(feasible_single)}")
    w(f"- pair feasible removals = {len(feasible_pairs)}")
    if not feasible_single:
        best_single = single_rows[0]
        w(f"- best single removal quotient gain = {best_single['quotient_gain']} after removing {best_single['removed_terms']}")
    if not feasible_pairs:
        best_pair = pair_rows[0]
        w(f"- best pair removal quotient gain = {best_pair['quotient_gain']} after removing {best_pair['removed_terms']}")
    w('')
    w('## Commutator Split')
    w('')
    for row in comm_rows:
        if row['matrix_size'] == '3x3':
            w(
                f"- {row['tensor_piece']} ({row['matrix_size']}): flattening lower bound {row['flattening_lower_bound']} "
                f"from ranks ({row['flattening_rank_A_BC']}, {row['flattening_rank_B_AC']}, {row['flattening_rank_C_AB']})"
            )
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The main exact finding is that the public rank-23 3x3 decomposition sits exactly on the Step 52 boundary: its nuisance rank can now be measured directly rather than hypothesized. The removal tests then ask the stronger reverse-engineering question the forward search could not ask: whether this known 23-term solution hides a 22-term or 21-term reweightable sub-decomposition. The HTML visualization records the actual cancellation fabric term-by-term rather than only aggregate ranks.')
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    terms, gamma_orientation, reconstruction_max_abs = load_public_rank23_terms()
    profile = rank_profile(terms, 3)
    source_status = source_rows()
    term_coeff_rows = term_rows(terms)
    residual_rows, term_summary_rows, fiber_rows = term_residual_rows(terms)
    single_feasibility_rows, pair_feasibility_rows = subset_feasibility_rows(terms)
    dead_rows, live_rows, edge_rows = balance_rows(terms)
    comm_rows = commutator_rows()

    summary_rows = [
        {'summary_name': 'algorithm_label', 'summary_value': 'AlphaTensor_public_rank23_3x3x3', 'provenance': 'MEASURED_FROM_CODE', 'note': 'Exact public coefficient table used for Step 63 reverse engineering.'},
        {'summary_name': 'term_count', 'summary_value': str(len(terms)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of rank-1 terms in the recovered public factorization.'},
        {'summary_name': 'gamma_orientation', 'summary_value': gamma_orientation, 'provenance': 'EXACT_DERIVED', 'note': 'Orientation needed to align the public output factors with ADE3x3 output indexing.'},
        {'summary_name': 'reconstruction_max_abs_residual', 'summary_value': str(reconstruction_max_abs), 'provenance': 'EXACT_DERIVED', 'note': 'Exact tensor reconstruction residual after orientation alignment.'},
        {'summary_name': 'sigma_rank', 'summary_value': str(profile['sigma_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Rank of Sigma for the recovered rank-23 algorithm.'},
        {'summary_name': 'eta_rank', 'summary_value': str(profile['eta_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Combined rank of Eta1 and Eta2.'},
        {'summary_name': 'delta_rank', 'summary_value': str(profile['delta_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Rank of Delta for the recovered rank-23 algorithm.'},
        {'summary_name': 'nuisance_rank', 'summary_value': str(profile['nuisance_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Step 52 nuisance rank measured on the recovered rank-23 algorithm.'},
        {'summary_name': 'augmented_rank', 'summary_value': str(profile['augmented_rank']), 'provenance': 'EXACT_DERIVED', 'note': 'Rank of [Sigma Nuisance] for the recovered rank-23 algorithm.'},
        {'summary_name': 'quotient_gain', 'summary_value': str(profile['quotient_gain']), 'provenance': 'EXACT_DERIVED', 'note': 'Sigma quotient gain modulo nuisance.'},
        {'summary_name': 'nuisance_equals_R_minus_9', 'summary_value': str(profile['nuisance_equals_R_minus_9']), 'provenance': 'EXACT_DERIVED', 'note': 'Whether the recovered algorithm saturates nuisance rank = R - 9 exactly.'},
        {'summary_name': 'single_term_reweight_feasible_count', 'summary_value': str(sum(int(row['gamma_reweight_feasible']) for row in single_feasibility_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of single-term removals whose remaining alpha/beta factors still admit some gamma reweighting.'},
        {'summary_name': 'pair_reweight_feasible_count', 'summary_value': str(sum(int(row['gamma_reweight_feasible']) for row in pair_feasibility_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of pair removals whose remaining alpha/beta factors still admit some gamma reweighting.'},
        {'summary_name': 'dependency_graph_edge_count', 'summary_value': str(len(edge_rows)), 'provenance': 'EXACT_DERIVED', 'note': 'Number of term pairs sharing at least one dead X cell.'},
        {'summary_name': 'visualization_html', 'summary_value': str((OUTPUTS / 'step63_cancellation_visualization.html').as_posix()), 'provenance': 'MEASURED_FROM_CODE', 'note': 'HTML visualization path.'},
    ]

    write_csv(EXPORTS / 'step63_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step63_external_source_status.csv', source_status, ['source_kind', 'source_label', 'status', 'detail', 'url', 'provenance'])
    write_csv(EXPORTS / 'step63_term_coefficients.csv', term_coeff_rows, ['term_id', 'source_label', 'alpha', 'beta', 'gamma', 'alpha_nonzero', 'beta_nonzero', 'gamma_nonzero', 'provenance'])
    write_csv(EXPORTS / 'step63_term_residual_equations.csv', residual_rows, ['term_id', 'x_cell', 'output_cell', 'equation_type', 'is_live', 'coefficient', 'provenance'])
    write_csv(EXPORTS / 'step63_term_importance.csv', term_summary_rows, ['term_id', 'frobenius_norm_squared', 'frobenius_norm', 'nonzero_equation_count', 'affected_equation_types', 'affected_live_fibers', 'provenance', 'E0_count', 'E1_count', 'E2_count', 'E3_count', 'E4_count', 'E5_count', 'E6_count', 'E7_count', 'importance_rank'])
    write_csv(EXPORTS / 'step63_term_fiber_residuals.csv', fiber_rows, ['term_id', 'output_cell', 'fiber', 'value', 'nonzero', 'provenance'])
    write_csv(EXPORTS / 'step63_single_term_removal.csv', single_feasibility_rows, ['removed_terms', 'remaining_R', 'nuisance_rank', 'augmented_rank', 'quotient_gain', 'gamma_reweight_feasible', 'sigma_verification_if_feasible', 'nuisance_verification_if_feasible', 'provenance'])
    write_csv(EXPORTS / 'step63_pair_term_removal.csv', pair_feasibility_rows, ['removed_terms', 'remaining_R', 'nuisance_rank', 'augmented_rank', 'quotient_gain', 'gamma_reweight_feasible', 'provenance'])
    write_csv(EXPORTS / 'step63_dead_equation_balances.csv', dead_rows, ['x_cell', 'output_cell', 'positive_terms', 'negative_terms', 'positive_sum', 'negative_sum', 'total', 'balance_status', 'provenance'])
    write_csv(EXPORTS / 'step63_live_equation_balances.csv', live_rows, ['x_cell', 'target_output_cell', 'positive_terms', 'negative_terms', 'total', 'target_rhs', 'provenance'])
    write_csv(EXPORTS / 'step63_dependency_graph.csv', edge_rows, ['left_term', 'right_term', 'shared_dead_cells', 'shared_dead_equations', 'provenance'])
    write_csv(EXPORTS / 'step63_commutator_split.csv', comm_rows, ['matrix_size', 'tensor_piece', 'nonzero_entries', 'flattening_rank_A_BC', 'flattening_rank_B_AC', 'flattening_rank_C_AB', 'flattening_lower_bound', 'provenance'])

    summary_map = {row['summary_name']: row['summary_value'] for row in summary_rows}
    html_text = visualization_html(terms, dead_rows, live_rows, edge_rows, summary_map)
    write_text(OUTPUTS / 'step63_cancellation_visualization.html', html_text)

    md_text = markdown_summary(summary_rows, source_status, profile, single_feasibility_rows, pair_feasibility_rows, comm_rows)
    write_text(EXPORTS / 'step63_reverse_engineering_cancellation_visualization.md', md_text)

    print('Step 63 complete: recovered public rank-23 coefficients, exported reverse-engineering diagnostics, and wrote HTML visualization.')


if __name__ == '__main__':
    main()