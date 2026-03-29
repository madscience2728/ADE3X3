"""
ade3x3_step74_commutator_anticommutator_rank_scan.py

Step 74: Commutator / Anticommutator Definitive Rank Scan.

This step upgrades Step 71 Shot 3 from a lightweight loss-only probe into a
factor-preserving rank scan. It runs the requested wide phase on the 3x3
commutator and anticommutator tensors, reruns the transition ranks with a finer
scan, exports the best decompositions, performs shared-term analysis, and then
profiles the resulting exact decompositions in the Step 51 fiber-mode basis.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
    swapped_matrix_multiplication_tensor,
)
from src.ade3x3.steps.ade3x3_step68_layered_correction_tiling import flattening_ranks


EXPORTS = Path('outputs/exports')

PAIR_MATCH_TOL = 1e-8


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def env_rank_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    ranks: list[int] = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if '..' in token:
            start_str, end_str = token.split('..', 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if start <= end else -1
            ranks.extend(list(range(start, end + step, step)))
        else:
            ranks.append(int(token))
    if not ranks:
        raise ValueError(f'{name} did not contain any valid ranks.')
    return sorted(set(ranks))


PHASE1_RANKS = env_rank_list('STEP74_PHASE1_RANKS', list(range(8, 24)))
PHASE1_RESTARTS = env_int('STEP74_PHASE1_RESTARTS', 2000)
PHASE2_RESTARTS = env_int('STEP74_PHASE2_RESTARTS', 5000)
CHUNK_SIZE = env_int('STEP74_CHUNK_SIZE', 25)
VERIFY_TOL = float(os.environ.get('STEP74_VERIFY_TOL', '1e-15'))
LOSS_TOL = float(os.environ.get('STEP74_LOSS_TOL', '1e-24'))
MAXITER = env_int('STEP74_MAXITER', 500)
INIT_SCALE = float(os.environ.get('STEP74_INIT_SCALE', '0.25'))
DEFAULT_WORKERS = env_int('STEP74_WORKERS', max(1, os.cpu_count() or 1))


@dataclass(frozen=True)
class TensorSpec:
    tensor_name: str
    display_name: str
    tensor: np.ndarray
    flattening_lb: int


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Wrote {len(rows)} rows -> {path}', flush=True)


def write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f'  Wrote text -> {path}', flush=True)


def matrix_to_json(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=np.float64).tolist(), separators=(',', ':'))


def scalar_to_str(value: float) -> str:
    return f'{value:.16e}'


def build_tensor_specs() -> list[TensorSpec]:
    mult = matrix_multiplication_tensor(3).astype(np.float64)
    swapped = swapped_matrix_multiplication_tensor(3).astype(np.float64)
    comm = mult - swapped
    anti = mult + swapped
    return [
        TensorSpec('commutator', 'T_comm = T - T_swapped', comm, flattening_ranks(comm)[3]),
        TensorSpec('anticommutator', 'T_anti = T + T_swapped', anti, flattening_ranks(anti)[3]),
    ]


def pack_factors(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([alpha.reshape(-1), beta.reshape(-1), gamma.reshape(-1)])


def unpack_factors(vector: np.ndarray, rank: int, output_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split1 = rank * 9
    split2 = 2 * rank * 9
    alpha = vector[:split1].reshape(rank, 9)
    beta = vector[split1:split2].reshape(rank, 9)
    gamma = vector[split2:].reshape(rank, output_dim)
    return alpha, beta, gamma


def cp_objective(vector: np.ndarray, target: np.ndarray, rank: int) -> tuple[float, np.ndarray]:
    output_dim = target.shape[2]
    alpha, beta, gamma = unpack_factors(vector, rank, output_dim)
    approx = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)
    residual = approx - target
    loss = 0.5 * float(np.sum(residual * residual))
    grad_alpha = np.einsum('abc,rb,rc->ra', residual, beta, gamma, optimize=True)
    grad_beta = np.einsum('abc,ra,rc->rb', residual, alpha, gamma, optimize=True)
    grad_gamma = np.einsum('abc,ra,rb->rc', residual, alpha, beta, optimize=True)
    return loss, pack_factors(grad_alpha, grad_beta, grad_gamma)


def random_initialization(rank: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = INIT_SCALE * rng.standard_normal((rank, 9))
    beta = INIT_SCALE * rng.standard_normal((rank, 9))
    gamma = INIT_SCALE * rng.standard_normal((rank, output_dim))
    return pack_factors(alpha, beta, gamma)


def refine_gamma(alpha: np.ndarray, beta: np.ndarray, target: np.ndarray) -> np.ndarray:
    features = np.einsum('ra,rb->abr', alpha, beta, optimize=True).reshape(81, alpha.shape[0])
    target_flat = target.reshape(81, target.shape[2])
    gamma, *_ = np.linalg.lstsq(features, target_flat, rcond=None)
    return gamma.astype(np.float64)


def reconstruct_tensor(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)


def max_abs_residual(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, target: np.ndarray) -> float:
    residual = reconstruct_tensor(alpha, beta, gamma) - target
    return float(np.max(np.abs(residual)))


def loss_value(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, target: np.ndarray) -> float:
    residual = reconstruct_tensor(alpha, beta, gamma) - target
    return float(np.sum(residual * residual))


def restart_seed(tensor_name: str, rank: int, restart_idx: int) -> int:
    base = sum((idx + 1) * ord(ch) for idx, ch in enumerate(tensor_name))
    return 7400000 + 100000 * rank + 97 * restart_idx + base


def chunk_worker(payload: dict) -> dict:
    tensor = np.array(payload['tensor'], dtype=np.float64)
    tensor_name = str(payload['tensor_name'])
    phase = str(payload['phase'])
    rank = int(payload['rank'])
    seeds = list(payload['seeds'])

    best_loss = math.inf
    best_max_abs = math.inf
    best_seed = -1
    best_alpha = None
    best_beta = None
    best_gamma = None
    exact_hit_count = 0

    for seed in seeds:
        x0 = random_initialization(rank, tensor.shape[2], int(seed))
        result = minimize(
            lambda vector: cp_objective(vector, tensor, rank),
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': MAXITER, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 50},
        )
        alpha, beta, gamma = unpack_factors(result.x, rank, tensor.shape[2])
        gamma = refine_gamma(alpha, beta, tensor)
        current_loss = loss_value(alpha, beta, gamma, tensor)
        current_max_abs = max_abs_residual(alpha, beta, gamma, tensor)
        if current_max_abs <= VERIFY_TOL:
            exact_hit_count += 1
        if current_max_abs < best_max_abs or (abs(current_max_abs - best_max_abs) <= 1e-18 and current_loss < best_loss):
            best_loss = current_loss
            best_max_abs = current_max_abs
            best_seed = int(seed)
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma

    return {
        'tensor_name': tensor_name,
        'phase': phase,
        'rank': rank,
        'restart_count': len(seeds),
        'exact_hit_count': exact_hit_count,
        'best_loss': best_loss,
        'best_max_abs': best_max_abs,
        'best_seed': best_seed,
        'best_alpha': best_alpha.tolist() if best_alpha is not None else [],
        'best_beta': best_beta.tolist() if best_beta is not None else [],
        'best_gamma': best_gamma.tolist() if best_gamma is not None else [],
    }


def scan_rank_jobs(specs: list[TensorSpec], jobs: list[tuple[str, str, int, int]], workers: int) -> dict[tuple[str, str, int], dict]:
    tensors = {spec.tensor_name: spec.tensor for spec in specs}
    aggregates: dict[tuple[str, str, int], dict] = {}
    payloads: list[dict] = []
    for tensor_name, phase, rank, restarts in jobs:
        seeds = [restart_seed(tensor_name, rank, restart_idx) for restart_idx in range(restarts)]
        for start_idx in range(0, len(seeds), CHUNK_SIZE):
            payloads.append(
                {
                    'tensor_name': tensor_name,
                    'phase': phase,
                    'rank': rank,
                    'tensor': tensors[tensor_name].tolist(),
                    'seeds': seeds[start_idx:start_idx + CHUNK_SIZE],
                }
            )
        aggregates[(tensor_name, phase, rank)] = {
            'tensor_name': tensor_name,
            'phase': phase,
            'rank': rank,
            'restart_count': 0,
            'exact_hit_count': 0,
            'best_loss': math.inf,
            'best_max_abs': math.inf,
            'best_seed': -1,
            'best_alpha': None,
            'best_beta': None,
            'best_gamma': None,
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(chunk_worker, payload) for payload in payloads]
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            key = (result['tensor_name'], result['phase'], result['rank'])
            if key not in aggregates:
                raise RuntimeError('Chunk result could not be matched to an aggregate job.')
            aggregate = aggregates[key]
            aggregate['restart_count'] += int(result['restart_count'])
            aggregate['exact_hit_count'] += int(result['exact_hit_count'])
            if float(result['best_max_abs']) < aggregate['best_max_abs'] or (
                abs(float(result['best_max_abs']) - aggregate['best_max_abs']) <= 1e-18
                and float(result['best_loss']) < aggregate['best_loss']
            ):
                aggregate['best_loss'] = float(result['best_loss'])
                aggregate['best_max_abs'] = float(result['best_max_abs'])
                aggregate['best_seed'] = int(result['best_seed'])
                aggregate['best_alpha'] = np.array(result['best_alpha'], dtype=np.float64)
                aggregate['best_beta'] = np.array(result['best_beta'], dtype=np.float64)
                aggregate['best_gamma'] = np.array(result['best_gamma'], dtype=np.float64)
            if completed == 1 or completed % 50 == 0 or completed == total:
                print(f'  completed {completed}/{total} chunk jobs', flush=True)

    return aggregates


def collect_rows(aggregates: dict[tuple[str, str, int], dict], tensor_name: str) -> list[dict]:
    rows: list[dict] = []
    for key, aggregate in sorted(aggregates.items(), key=lambda item: (item[0][1], item[0][2])):
        if key[0] != tensor_name:
            continue
        rows.append(
            {
                'tensor_name': aggregate['tensor_name'],
                'phase': aggregate['phase'],
                'rank_tested': aggregate['rank'],
                'restarts': aggregate['restart_count'],
                'best_seed': aggregate['best_seed'],
                'best_verified_loss': scalar_to_str(aggregate['best_loss']),
                'best_max_abs_residual': scalar_to_str(aggregate['best_max_abs']),
                'exact_hit_count': aggregate['exact_hit_count'],
                'verified_exact': str(aggregate['best_max_abs'] <= VERIFY_TOL),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )
    return rows


def first_exact_rank(rows: list[dict], phase: str = 'phase1') -> str:
    for row in rows:
        if row['phase'] != phase:
            continue
        if row['verified_exact'] == 'True' or int(row['exact_hit_count']) > 0:
            return str(row['rank_tested'])
    return 'none'


def aggregate_by_rank(aggregates: dict[tuple[str, str, int], dict], tensor_name: str, phase: str, rank: int) -> dict | None:
    return aggregates.get((tensor_name, phase, rank))


def factors_to_terms(tensor_name: str, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> list[Term]:
    terms: list[Term] = []
    for term_idx in range(alpha.shape[0]):
        terms.append(
            Term(
                term_id=f'{tensor_name}_r{alpha.shape[0]:02d}_t{term_idx + 1:02d}',
                source_label=f'step74_{tensor_name}_rank_{alpha.shape[0]}',
                alpha=alpha[term_idx].reshape(3, 3),
                beta=beta[term_idx].reshape(3, 3),
                gamma=gamma[term_idx].reshape(3, 3),
            )
        )
    return terms


def decomposition_residual(terms: list[Term], target: np.ndarray) -> float:
    total = np.zeros_like(target)
    for term in terms:
        total += np.einsum('a,b,c->abc', term.alpha.reshape(-1), term.beta.reshape(-1), term.gamma.reshape(-1), optimize=True)
    return float(np.max(np.abs(total - target)))


def coefficient_rows(tensor_name: str, rank_value: int, phase: str, terms: list[Term]) -> list[dict]:
    rows: list[dict] = []
    for term in terms:
        flat = np.einsum('a,b,c->abc', term.alpha.reshape(-1), term.beta.reshape(-1), term.gamma.reshape(-1), optimize=True)
        rows.append(
            {
                'tensor_name': tensor_name,
                'phase': phase,
                'rank_value': rank_value,
                'term_id': term.term_id,
                'alpha': matrix_to_json(term.alpha),
                'beta': matrix_to_json(term.beta),
                'gamma': matrix_to_json(term.gamma),
                'term_frobenius_norm': f'{float(np.linalg.norm(flat)):.16e}',
                'term_max_abs': f'{float(np.max(np.abs(flat))):.16e}',
                'provenance': 'MEASURED_FROM_CODE',
            }
        )
    return rows


def build_numeric_mode_matrices(terms: list[Term], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    term_count = len(terms)
    sigma_columns: list[list[float]] = []
    eta1_columns: list[list[float]] = []
    eta2_columns: list[list[float]] = []
    delta_columns: list[list[float]] = []

    for row_idx in range(n):
        for col_idx in range(n):
            sigma_vec: list[float] = []
            eta1_vec: list[float] = []
            eta2_vec: list[float] = []
            for term in terms:
                lambdas = [float(term.alpha[row_idx, sum_idx] * term.beta[sum_idx, col_idx]) for sum_idx in range(n)]
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
                    delta_columns.append(
                        [float(term.alpha[row_idx, sum_left] * term.beta[sum_right, col_idx]) for term in terms]
                    )

    sigma = np.column_stack(sigma_columns) if sigma_columns else np.zeros((term_count, 0), dtype=np.float64)
    eta1 = np.column_stack(eta1_columns) if eta1_columns else np.zeros((term_count, 0), dtype=np.float64)
    eta2 = np.column_stack(eta2_columns) if eta2_columns else np.zeros((term_count, 0), dtype=np.float64)
    delta = np.column_stack(delta_columns) if delta_columns else np.zeros((term_count, 0), dtype=np.float64)
    nuisance = np.hstack([eta1, eta2, delta]) if term_count else np.zeros((0, 0), dtype=np.float64)
    return sigma, eta1, eta2, delta, nuisance


def profile_rows_for_terms(label: str, terms: list[Term]) -> tuple[dict, dict]:
    sigma, eta1, eta2, delta, nuisance = build_numeric_mode_matrices(terms, 3)
    augmented = np.hstack([
        sigma,
        eta1,
        eta2,
        delta,
    ])
    summary = {
        'profile_name': label,
        'term_count': len(terms),
        'sigma_rank': int(np.linalg.matrix_rank(sigma)),
        'eta1_rank': int(np.linalg.matrix_rank(eta1)),
        'eta2_rank': int(np.linalg.matrix_rank(eta2)),
        'eta_rank': int(np.linalg.matrix_rank(np.hstack([eta1, eta2]))),
        'delta_rank': int(np.linalg.matrix_rank(delta)),
        'nuisance_rank': int(np.linalg.matrix_rank(nuisance)),
        'augmented_rank': int(np.linalg.matrix_rank(augmented)),
        'quotient_gain': int(np.linalg.matrix_rank(augmented)) - int(np.linalg.matrix_rank(nuisance)),
        'provenance': 'EXACT_DERIVED',
    }
    matrices = {
        'profile_name': label,
        'sigma_matrix': matrix_to_json(sigma),
        'eta1_matrix': matrix_to_json(eta1),
        'eta2_matrix': matrix_to_json(eta2),
        'delta_matrix': matrix_to_json(delta),
        'provenance': 'EXACT_DERIVED',
    }
    return summary, matrices


def canonical_term_tensor(term: Term) -> tuple[np.ndarray, float]:
    tensor = np.einsum('a,b,c->abc', term.alpha.reshape(-1), term.beta.reshape(-1), term.gamma.reshape(-1), optimize=True).reshape(-1)
    norm = float(np.linalg.norm(tensor))
    if norm == 0.0:
        return tensor, 0.0
    canonical = tensor / norm
    nz = np.flatnonzero(np.abs(canonical) > 1e-12)
    if nz.size and canonical[nz[0]] < 0:
        canonical = -canonical
        norm = -norm
    return canonical, norm


def build_match_graph(comm_terms: list[Term], anti_terms: list[Term]) -> tuple[list[list[int]], list[dict]]:
    graph: list[list[int]] = [[] for _ in comm_terms]
    comparisons: list[dict] = []
    comm_canon = [canonical_term_tensor(term) for term in comm_terms]
    anti_canon = [canonical_term_tensor(term) for term in anti_terms]
    for comm_idx, (comm_vec, comm_scale) in enumerate(comm_canon):
        for anti_idx, (anti_vec, anti_scale) in enumerate(anti_canon):
            diff = float(np.max(np.abs(comm_vec - anti_vec)))
            dot = float(np.dot(comm_vec, anti_vec)) if np.linalg.norm(comm_vec) and np.linalg.norm(anti_vec) else 0.0
            matched = diff <= PAIR_MATCH_TOL
            if matched:
                graph[comm_idx].append(anti_idx)
            comparisons.append(
                {
                    'comm_term_id': comm_terms[comm_idx].term_id,
                    'anti_term_id': anti_terms[anti_idx].term_id,
                    'max_abs_difference_after_normalization': scalar_to_str(diff),
                    'normalized_dot_product': scalar_to_str(dot),
                    'comm_scale': scalar_to_str(comm_scale),
                    'anti_scale': scalar_to_str(anti_scale),
                    'matched': str(matched),
                    'provenance': 'EXACT_DERIVED',
                }
            )
    return graph, comparisons


def maximum_matching(graph: list[list[int]], right_size: int) -> dict[int, int]:
    match_right: dict[int, int] = {}

    def dfs(left_idx: int, seen: set[int]) -> bool:
        for right_idx in graph[left_idx]:
            if right_idx in seen:
                continue
            seen.add(right_idx)
            if right_idx not in match_right or dfs(match_right[right_idx], seen):
                match_right[right_idx] = left_idx
                return True
        return False

    for left_idx in range(len(graph)):
        dfs(left_idx, set())
    return {left_idx: right_idx for right_idx, left_idx in match_right.items()}


def sharing_rows(comm_terms: list[Term], anti_terms: list[Term]) -> tuple[list[dict], list[dict], dict, list[Term] | None, float | None]:
    graph, comparisons = build_match_graph(comm_terms, anti_terms)
    matches = maximum_matching(graph, len(anti_terms))
    matched_left = set(matches)
    matched_right = {matches[left_idx] for left_idx in matches}

    comm_canon = [canonical_term_tensor(term) for term in comm_terms]
    anti_canon = [canonical_term_tensor(term) for term in anti_terms]
    combined_terms: list[Term] = []
    match_rows: list[dict] = []

    for left_idx, right_idx in sorted(matches.items()):
        comm_scale = comm_canon[left_idx][1]
        anti_scale = anti_canon[right_idx][1]
        combined_scale = 0.5 * (comm_scale + anti_scale)
        scale_ratio = 0.0 if abs(comm_scale) <= 1e-18 else combined_scale / comm_scale
        representative = comm_terms[left_idx]
        if abs(combined_scale) > 1e-12:
            combined_terms.append(
                Term(
                    term_id=f'combined_shared_{len(combined_terms) + 1:02d}',
                    source_label='step74_shared_term',
                    alpha=representative.alpha.copy(),
                    beta=representative.beta.copy(),
                    gamma=representative.gamma.copy() * scale_ratio,
                )
            )
        match_rows.append(
            {
                'comm_term_id': representative.term_id,
                'anti_term_id': anti_terms[right_idx].term_id,
                'comm_scale': scalar_to_str(comm_scale),
                'anti_scale': scalar_to_str(anti_scale),
                'combined_half_sum_scale': scalar_to_str(combined_scale),
                'cancels_in_T': str(abs(combined_scale) <= 1e-12),
                'provenance': 'EXACT_DERIVED',
            }
        )

    for idx, term in enumerate(comm_terms):
        if idx in matched_left:
            continue
        combined_terms.append(
            Term(
                term_id=f'combined_comm_only_{len(combined_terms) + 1:02d}',
                source_label='step74_comm_only',
                alpha=term.alpha.copy(),
                beta=term.beta.copy(),
                gamma=0.5 * term.gamma.copy(),
            )
        )

    for idx, term in enumerate(anti_terms):
        if idx in matched_right:
            continue
        combined_terms.append(
            Term(
                term_id=f'combined_anti_only_{len(combined_terms) + 1:02d}',
                source_label='step74_anti_only',
                alpha=term.alpha.copy(),
                beta=term.beta.copy(),
                gamma=0.5 * term.gamma.copy(),
            )
        )

    union_count = len(comm_terms) + len(anti_terms) - len(matches)
    combined_count = len(combined_terms)
    combined_residual = decomposition_residual(combined_terms, matrix_multiplication_tensor(3).astype(np.float64)) if combined_terms else None
    summary = {
        'shared_term_count': len(matches),
        'union_distinct_term_count': union_count,
        'combined_nonzero_term_count': combined_count,
        'combined_tensor_max_abs_residual': combined_residual,
        'beats_public_rank23': combined_count < 23,
    }
    return comparisons, match_rows, summary, combined_terms, combined_residual


def summary_rows(
    specs: list[TensorSpec],
    rows_by_tensor: dict[str, list[dict]],
    exact_terms: dict[str, list[Term]],
    profile_summaries: list[dict],
    sharing_summary: dict | None,
    elapsed_seconds: float,
) -> list[dict]:
    output: list[dict] = []
    spec_map = {spec.tensor_name: spec for spec in specs}
    for tensor_name in ('commutator', 'anticommutator'):
        rows = rows_by_tensor[tensor_name]
        exact_rank = first_exact_rank(rows, 'phase1')
        phase2_rank = first_exact_rank(rows, 'phase2')
        best_overall = min(rows, key=lambda row: float(row['best_max_abs_residual'])) if rows else None
        output.extend(
            [
                {
                    'summary_name': f'step74_{tensor_name}_flattening_lb',
                    'summary_value': str(spec_map[tensor_name].flattening_lb),
                    'provenance': 'EXACT_DERIVED',
                    'note': f'Flattening lower bound for {tensor_name}.',
                },
                {
                    'summary_name': f'step74_{tensor_name}_phase1_first_exact_rank',
                    'summary_value': exact_rank,
                    'provenance': 'MEASURED_FROM_CODE',
                    'note': f'First exact rank hit in the 2000-restart wide scan for {tensor_name}.',
                },
                {
                    'summary_name': f'step74_{tensor_name}_phase2_exact_rank',
                    'summary_value': phase2_rank,
                    'provenance': 'MEASURED_FROM_CODE',
                    'note': f'Exact rank confirmed after the 5000-restart fine scan for {tensor_name}.',
                },
                {
                    'summary_name': f'step74_{tensor_name}_best_overall_max_abs_residual',
                    'summary_value': best_overall['best_max_abs_residual'] if best_overall else '',
                    'provenance': 'MEASURED_FROM_CODE',
                    'note': f'Best max-abs residual seen across all Step 74 scans for {tensor_name}.',
                },
            ]
        )
    if sharing_summary is not None:
        output.extend(
            [
                {
                    'summary_name': 'step74_shared_term_count',
                    'summary_value': str(sharing_summary['shared_term_count']),
                    'provenance': 'EXACT_DERIVED',
                    'note': 'Maximum matched shared rank-1 terms between the commutator and anticommutator decompositions.',
                },
                {
                    'summary_name': 'step74_union_distinct_term_count',
                    'summary_value': str(sharing_summary['union_distinct_term_count']),
                    'provenance': 'EXACT_DERIVED',
                    'note': 'Union size rank(T_comm) + rank(T_anti) - shared.',
                },
                {
                    'summary_name': 'step74_combined_nonzero_term_count',
                    'summary_value': str(sharing_summary['combined_nonzero_term_count']),
                    'provenance': 'EXACT_DERIVED',
                    'note': 'Nonzero term count after forming T = (T_comm + T_anti)/2 and canceling matched terms.',
                },
                {
                    'summary_name': 'step74_combined_beats_23',
                    'summary_value': str(sharing_summary['beats_public_rank23']),
                    'provenance': 'EXACT_DERIVED',
                    'note': 'Whether the combined commutator/anticommutator term set beats the public rank-23 factorization.',
                },
            ]
        )
        if sharing_summary['combined_tensor_max_abs_residual'] is not None:
            output.append(
                {
                    'summary_name': 'step74_combined_tensor_max_abs_residual',
                    'summary_value': scalar_to_str(float(sharing_summary['combined_tensor_max_abs_residual'])),
                    'provenance': 'EXACT_DERIVED',
                    'note': 'Verification residual of the combined T = (T_comm + T_anti)/2 term list.',
                }
            )
    for row in profile_summaries:
        output.append(
            {
                'summary_name': f"step74_profile_{row['profile_name']}_nuisance_rank",
                'summary_value': str(row['nuisance_rank']),
                'provenance': row['provenance'],
                'note': f"Nuisance rank for {row['profile_name']}.",
            }
        )
    output.append(
        {
            'summary_name': 'step74_runtime_seconds',
            'summary_value': f'{elapsed_seconds:.6f}',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Wall-clock runtime for Step 74.',
        }
    )
    return output


def markdown_report(
    specs: list[TensorSpec],
    rows_by_tensor: dict[str, list[dict]],
    profile_summaries: list[dict],
    sharing_summary: dict | None,
    summary_map: dict[str, dict],
) -> str:
    lines: list[str] = []
    w = lines.append

    w('# Step 74: Commutator / Anticommutator Definitive Rank Scan')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[MEASURED_FROM_CODE] + [EXACT_DERIVED]')
    w('')
    w('## Wide Scan')
    w('')
    w('| tensor | flattening lb | phase-1 first exact rank | phase-2 exact rank | best overall max-abs residual |')
    w('|--------|---------------|--------------------------|--------------------|-------------------------------|')
    for spec in specs:
        w(
            f"| {spec.tensor_name} | {spec.flattening_lb} | {summary_map[f'step74_{spec.tensor_name}_phase1_first_exact_rank']['summary_value']} | {summary_map[f'step74_{spec.tensor_name}_phase2_exact_rank']['summary_value']} | {summary_map[f'step74_{spec.tensor_name}_best_overall_max_abs_residual']['summary_value']} |"
        )
    w('')
    for spec in specs:
        w(f"### {spec.display_name}")
        w('')
        w('| phase | rank | restarts | best loss | best max-abs residual | exact hit count | verified exact |')
        w('|-------|------|----------|-----------|------------------------|-----------------|----------------|')
        for row in rows_by_tensor[spec.tensor_name]:
            w(
                f"| {row['phase']} | {row['rank_tested']} | {row['restarts']} | {row['best_verified_loss']} | {row['best_max_abs_residual']} | {row['exact_hit_count']} | {row['verified_exact']} |"
            )
        w('')
    if sharing_summary is not None:
        w('## Sharing Analysis')
        w('')
        w(f"- Shared term count: {sharing_summary['shared_term_count']}")
        w(f"- Union distinct term count: {sharing_summary['union_distinct_term_count']}")
        w(f"- Combined nonzero term count in T = (T_comm + T_anti)/2: {sharing_summary['combined_nonzero_term_count']}")
        w(f"- Combined tensor max-abs residual: {scalar_to_str(float(sharing_summary['combined_tensor_max_abs_residual'])) if sharing_summary['combined_tensor_max_abs_residual'] is not None else 'n/a'}")
        w(f"- Beats public rank-23 factorization: {sharing_summary['beats_public_rank23']}")
        w('')
    w('## Fiber-Mode Profiles')
    w('')
    w('| profile | terms | sigma rank | eta1 rank | eta2 rank | eta rank | delta rank | nuisance rank | quotient gain |')
    w('|---------|-------|------------|-----------|-----------|----------|------------|---------------|---------------|')
    for row in profile_summaries:
        w(
            f"| {row['profile_name']} | {row['term_count']} | {row['sigma_rank']} | {row['eta1_rank']} | {row['eta2_rank']} | {row['eta_rank']} | {row['delta_rank']} | {row['nuisance_rank']} | {row['quotient_gain']} |"
        )
    return '\n'.join(lines)


def main() -> None:
    start = time.time()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    workers = DEFAULT_WORKERS
    specs = build_tensor_specs()

    print('=== Step 74: Commutator / Anticommutator Definitive Rank Scan ===', flush=True)
    print(f'Using {workers} worker processes across both tensors.', flush=True)
    print(f'Phase-1 ranks: {PHASE1_RANKS}', flush=True)
    print(f'Phase-1 restarts per rank: {PHASE1_RESTARTS}', flush=True)
    print(f'Phase-2 restarts per rank: {PHASE2_RESTARTS}', flush=True)
    print(f'Chunk size: {CHUNK_SIZE}; optimizer maxiter: {MAXITER}', flush=True)

    phase1_jobs = [(spec.tensor_name, 'phase1', rank, PHASE1_RESTARTS) for spec in specs for rank in PHASE1_RANKS]
    phase1_aggregates = scan_rank_jobs(specs, phase1_jobs, workers)
    rows_by_tensor = {spec.tensor_name: collect_rows(phase1_aggregates, spec.tensor_name) for spec in specs}

    phase2_jobs: list[tuple[str, str, int, int]] = []
    for spec in specs:
        exact_rank = first_exact_rank(rows_by_tensor[spec.tensor_name], 'phase1')
        if exact_rank == 'none':
            continue
        exact_rank_value = int(exact_rank)
        if exact_rank_value - 1 >= PHASE1_RANKS[0]:
            phase2_jobs.append((spec.tensor_name, 'phase2', exact_rank_value - 1, PHASE2_RESTARTS))
        phase2_jobs.append((spec.tensor_name, 'phase2', exact_rank_value, PHASE2_RESTARTS))

    phase2_aggregates = scan_rank_jobs(specs, phase2_jobs, workers) if phase2_jobs else {}
    combined_aggregates = {**phase1_aggregates, **phase2_aggregates}
    rows_by_tensor = {spec.tensor_name: collect_rows(combined_aggregates, spec.tensor_name) for spec in specs}

    exact_terms: dict[str, list[Term]] = {}
    coefficient_tables: list[dict] = []
    verification_rows: list[dict] = []
    for spec in specs:
        exact_rank = first_exact_rank(rows_by_tensor[spec.tensor_name], 'phase2')
        chosen_phase = 'phase2'
        if exact_rank == 'none':
            exact_rank = first_exact_rank(rows_by_tensor[spec.tensor_name], 'phase1')
            chosen_phase = 'phase1'
        if exact_rank == 'none':
            continue
        aggregate = aggregate_by_rank(combined_aggregates, spec.tensor_name, chosen_phase, int(exact_rank))
        if aggregate is None or aggregate['best_alpha'] is None:
            continue
        terms = factors_to_terms(spec.tensor_name, aggregate['best_alpha'], aggregate['best_beta'], aggregate['best_gamma'])
        exact_terms[spec.tensor_name] = terms
        coefficient_tables.extend(coefficient_rows(spec.tensor_name, int(exact_rank), chosen_phase, terms))
        max_abs = decomposition_residual(terms, spec.tensor)
        verification_rows.append(
            {
                'tensor_name': spec.tensor_name,
                'phase_used': chosen_phase,
                'rank_value': int(exact_rank),
                'max_abs_residual_all_entries': scalar_to_str(max_abs),
                'verified_entrywise_below_1e-15': str(max_abs <= VERIFY_TOL),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )

    sharing_comparison_rows: list[dict] = []
    sharing_match_rows: list[dict] = []
    combined_term_rows: list[dict] = []
    sharing_summary = None
    if 'commutator' in exact_terms and 'anticommutator' in exact_terms:
        sharing_comparison_rows, sharing_match_rows, sharing_summary, combined_terms, combined_residual = sharing_rows(exact_terms['commutator'], exact_terms['anticommutator'])
        if combined_terms is not None:
            combined_term_rows = coefficient_rows('combined_T', len(combined_terms), 'derived', combined_terms)
            verification_rows.append(
                {
                    'tensor_name': 'combined_T',
                    'phase_used': 'derived',
                    'rank_value': len(combined_terms),
                    'max_abs_residual_all_entries': scalar_to_str(float(combined_residual) if combined_residual is not None else math.inf),
                    'verified_entrywise_below_1e-15': str((combined_residual is not None) and combined_residual <= VERIFY_TOL),
                    'provenance': 'EXACT_DERIVED',
                }
            )

    profile_summaries: list[dict] = []
    profile_matrix_rows: list[dict] = []
    if 'commutator' in exact_terms:
        summary, matrices = profile_rows_for_terms('commutator_exact', exact_terms['commutator'])
        profile_summaries.append(summary)
        profile_matrix_rows.append(matrices)
    if 'anticommutator' in exact_terms:
        summary, matrices = profile_rows_for_terms('anticommutator_exact', exact_terms['anticommutator'])
        profile_summaries.append(summary)
        profile_matrix_rows.append(matrices)
    public_terms, _, _ = load_public_rank23_terms()
    public_summary, public_matrices = profile_rows_for_terms('public_rank23_multiplication', public_terms)
    profile_summaries.append(public_summary)
    profile_matrix_rows.append(public_matrices)

    elapsed = time.time() - start
    summary = summary_rows(specs, rows_by_tensor, exact_terms, profile_summaries, sharing_summary, elapsed)
    summary_map = {row['summary_name']: row for row in summary}
    report = markdown_report(specs, rows_by_tensor, profile_summaries, sharing_summary, summary_map)

    for spec in specs:
        write_csv(
            EXPORTS / f'step74_{spec.tensor_name}_rank_scan.csv',
            rows_by_tensor[spec.tensor_name],
            ['tensor_name', 'phase', 'rank_tested', 'restarts', 'best_seed', 'best_verified_loss', 'best_max_abs_residual', 'exact_hit_count', 'verified_exact', 'provenance'],
        )
    write_csv(
        EXPORTS / 'step74_exact_decomposition_terms.csv',
        coefficient_tables,
        ['tensor_name', 'phase', 'rank_value', 'term_id', 'alpha', 'beta', 'gamma', 'term_frobenius_norm', 'term_max_abs', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_exact_verification.csv',
        verification_rows,
        ['tensor_name', 'phase_used', 'rank_value', 'max_abs_residual_all_entries', 'verified_entrywise_below_1e-15', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_shared_term_comparisons.csv',
        sharing_comparison_rows,
        ['comm_term_id', 'anti_term_id', 'max_abs_difference_after_normalization', 'normalized_dot_product', 'comm_scale', 'anti_scale', 'matched', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_shared_term_matches.csv',
        sharing_match_rows,
        ['comm_term_id', 'anti_term_id', 'comm_scale', 'anti_scale', 'combined_half_sum_scale', 'cancels_in_T', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_combined_term_table.csv',
        combined_term_rows,
        ['tensor_name', 'phase', 'rank_value', 'term_id', 'alpha', 'beta', 'gamma', 'term_frobenius_norm', 'term_max_abs', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_fiber_mode_profiles.csv',
        profile_summaries,
        ['profile_name', 'term_count', 'sigma_rank', 'eta1_rank', 'eta2_rank', 'eta_rank', 'delta_rank', 'nuisance_rank', 'augmented_rank', 'quotient_gain', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_fiber_mode_matrices.csv',
        profile_matrix_rows,
        ['profile_name', 'sigma_matrix', 'eta1_matrix', 'eta2_matrix', 'delta_matrix', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step74_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_text(EXPORTS / 'step74_commutator_anticommutator_rank_scan.md', report)
    print(f'Step 74 complete in {elapsed:.2f}s', flush=True)


if __name__ == '__main__':
    main()