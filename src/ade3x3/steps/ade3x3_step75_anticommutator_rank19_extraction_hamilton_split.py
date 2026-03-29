"""
ade3x3_step75_anticommutator_rank19_extraction_hamilton_split.py

Step 75 Phase 1: Anticommutator Rank-19 Extraction + Hamilton Split.

This step performs the requested data extraction only:

1. Rerun the anticommutator rank-19 scan with 5000 restarts.
2. Extract the best 19-term decomposition and verify it with relaxed tolerances.
3. Compute the Step 51 fiber-mode profile and the A/B symmetry pairing counts.
4. Reconstruct the Step 74 commutator rank-20 best witness from its recorded best seed,
   then run the same verification, fiber-mode, and antisymmetry checks.
5. Write the requested Phase 1 CSV/Markdown exports without touching the dossier.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term
from src.ade3x3.steps.ade3x3_step74_commutator_anticommutator_rank_scan import (
    build_numeric_mode_matrices,
    build_tensor_specs,
    cp_objective,
    factors_to_terms,
    loss_value,
    max_abs_residual,
    random_initialization,
    reconstruct_tensor,
    refine_gamma,
    restart_seed,
    unpack_factors,
)


EXPORTS = Path('outputs/exports')
STEP74_COMM_SCAN = EXPORTS / 'step74_commutator_rank_scan.csv'

ANTI_RANK = 19
ANTI_RESTARTS = 5000
COMM_RANK = 20
CHUNK_SIZE = 25
MAXITER = 500
WORKERS = max(1, os.cpu_count() or 1)
NUMERICAL_EXACT_TOL = 1e-6
SYMMETRY_TOL = 1e-6


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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def matrix_to_json_fixed(matrix: np.ndarray, decimals: int = 6) -> str:
    rounded = np.round(np.asarray(matrix, dtype=np.float64), decimals=decimals)
    return json.dumps(rounded.tolist(), separators=(',', ':'))


def scalar_to_str(value: float) -> str:
    return f'{float(value):.16e}'


def term_tensor(term: Term) -> np.ndarray:
    return np.einsum(
        'a,b,c->abc',
        term.alpha.reshape(-1),
        term.beta.reshape(-1),
        term.gamma.reshape(-1),
        optimize=True,
    )


def swapped_term_tensor(term: Term, gamma_sign: float = 1.0) -> np.ndarray:
    return np.einsum(
        'a,b,c->abc',
        term.beta.reshape(-1),
        term.alpha.reshape(-1),
        (gamma_sign * term.gamma).reshape(-1),
        optimize=True,
    )


def normalized_tensor_vector(tensor: np.ndarray) -> np.ndarray:
    flat = tensor.reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(flat))
    if norm <= 0.0:
        return flat
    return flat / norm


def chunk_worker(payload: dict) -> dict:
    tensor = np.array(payload['tensor'], dtype=np.float64)
    tensor_name = str(payload['tensor_name'])
    rank = int(payload['rank'])
    seeds = list(payload['seeds'])

    best_loss = math.inf
    best_max_abs = math.inf
    best_seed = -1
    best_alpha = None
    best_beta = None
    best_gamma = None

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
        if current_loss < best_loss or (abs(current_loss - best_loss) <= 1e-24 and current_max_abs < best_max_abs):
            best_loss = current_loss
            best_max_abs = current_max_abs
            best_seed = int(seed)
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma

    return {
        'tensor_name': tensor_name,
        'rank': rank,
        'restart_count': len(seeds),
        'best_loss': best_loss,
        'best_max_abs': best_max_abs,
        'best_seed': best_seed,
        'best_alpha': best_alpha.tolist() if best_alpha is not None else [],
        'best_beta': best_beta.tolist() if best_beta is not None else [],
        'best_gamma': best_gamma.tolist() if best_gamma is not None else [],
    }


def scan_explicit_seeds(tensor_name: str, tensor: np.ndarray, rank: int, seeds: list[int], workers: int) -> dict:
    payloads: list[dict] = []
    for start_idx in range(0, len(seeds), CHUNK_SIZE):
        payloads.append(
            {
                'tensor_name': tensor_name,
                'rank': rank,
                'tensor': tensor.tolist(),
                'seeds': seeds[start_idx:start_idx + CHUNK_SIZE],
            }
        )

    aggregate = {
        'tensor_name': tensor_name,
        'rank': rank,
        'restart_count': 0,
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
            aggregate['restart_count'] += int(result['restart_count'])
            if float(result['best_loss']) < aggregate['best_loss'] or (
                abs(float(result['best_loss']) - aggregate['best_loss']) <= 1e-24
                and float(result['best_max_abs']) < aggregate['best_max_abs']
            ):
                aggregate['best_loss'] = float(result['best_loss'])
                aggregate['best_max_abs'] = float(result['best_max_abs'])
                aggregate['best_seed'] = int(result['best_seed'])
                aggregate['best_alpha'] = np.array(result['best_alpha'], dtype=np.float64)
                aggregate['best_beta'] = np.array(result['best_beta'], dtype=np.float64)
                aggregate['best_gamma'] = np.array(result['best_gamma'], dtype=np.float64)
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(f'  completed {completed}/{total} chunk jobs for {tensor_name} rank {rank}', flush=True)

    return aggregate


def read_step74_best_seed(path: Path, rank: int) -> int:
    rows = read_csv(path)
    for row in rows:
        if int(row['rank_tested']) == rank:
            return int(row['best_seed'])
    raise RuntimeError(f'Could not find rank {rank} in {path}.')


def verification_stats(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, target: np.ndarray) -> dict:
    residual = target - reconstruct_tensor(alpha, beta, gamma)
    abs_residual = np.abs(residual)
    return {
        'max_abs_residual': float(np.max(abs_residual)),
        'mean_abs_residual': float(np.mean(abs_residual)),
        'entries_gt_1e8': int(np.count_nonzero(abs_residual > 1e-8)),
        'entries_gt_1e6': int(np.count_nonzero(abs_residual > 1e-6)),
        'verified_exact_under_1e6': bool(float(np.max(abs_residual)) < NUMERICAL_EXACT_TOL),
    }


def profile_for_terms(terms: list[Term]) -> tuple[dict, dict[str, np.ndarray]]:
    sigma, eta1, eta2, delta, nuisance = build_numeric_mode_matrices(terms, 3)
    summary = {
        'sigma_rank': int(np.linalg.matrix_rank(sigma)),
        'eta_rank': int(np.linalg.matrix_rank(np.hstack([eta1, eta2]))),
        'delta_rank': int(np.linalg.matrix_rank(delta)),
        'nuisance_rank': int(np.linalg.matrix_rank(nuisance)),
        'quotient_gain': int(np.linalg.matrix_rank(np.hstack([sigma, nuisance]))) - int(np.linalg.matrix_rank(nuisance)),
    }
    matrices = {
        'sigma': sigma,
        'eta1': eta1,
        'eta2': eta2,
        'delta': delta,
        'nuisance': nuisance,
    }
    return summary, matrices


def symmetry_counts(terms: list[Term], gamma_sign: float) -> tuple[dict, list[dict]]:
    base_vectors = [normalized_tensor_vector(term_tensor(term)) for term in terms]
    partner_vectors = [normalized_tensor_vector(swapped_term_tensor(term, gamma_sign=gamma_sign)) for term in terms]
    pair_rows: list[dict] = []
    paired_terms: set[int] = set()
    self_count = 0
    pair_count = 0

    for term_idx in range(len(terms)):
        self_diff = float(np.max(np.abs(base_vectors[term_idx] - partner_vectors[term_idx])))
        if self_diff <= SYMMETRY_TOL:
            self_count += 1
            paired_terms.add(term_idx)
            pair_rows.append(
                {
                    'term_id': terms[term_idx].term_id,
                    'partner_term_id': terms[term_idx].term_id,
                    'pair_type': 'self',
                    'normalized_max_abs_difference': scalar_to_str(self_diff),
                }
            )

    used: set[int] = set()
    for left_idx in range(len(terms)):
        if left_idx in used or left_idx in paired_terms:
            continue
        best_idx = None
        best_diff = math.inf
        for right_idx in range(len(terms)):
            if right_idx == left_idx or right_idx in used or right_idx in paired_terms:
                continue
            diff = float(np.max(np.abs(partner_vectors[left_idx] - base_vectors[right_idx])))
            if diff < best_diff:
                best_diff = diff
                best_idx = right_idx
        if best_idx is not None and best_diff <= SYMMETRY_TOL:
            used.add(left_idx)
            used.add(best_idx)
            paired_terms.add(left_idx)
            paired_terms.add(best_idx)
            pair_count += 1
            pair_rows.append(
                {
                    'term_id': terms[left_idx].term_id,
                    'partner_term_id': terms[best_idx].term_id,
                    'pair_type': 'paired',
                    'normalized_max_abs_difference': scalar_to_str(best_diff),
                }
            )

    unpaired_count = len(terms) - len(paired_terms)
    return {
        'self_count': self_count,
        'pair_count': pair_count,
        'unpaired_count': unpaired_count,
    }, pair_rows


def anti_coefficient_rows(terms: list[Term]) -> list[dict]:
    rows: list[dict] = []
    for term_idx, term in enumerate(terms, start=1):
        rows.append(
            {
                'term': f's{term_idx:02d}',
                'alpha': matrix_to_json_fixed(term.alpha, decimals=6),
                'beta': matrix_to_json_fixed(term.beta, decimals=6),
                'gamma': matrix_to_json_fixed(term.gamma, decimals=6),
            }
        )
    return rows


def fibermode_row(label: str, profile: dict, matrices: dict[str, np.ndarray]) -> dict:
    return {
        'label': label,
        'sigma_rank': profile['sigma_rank'],
        'eta_rank': profile['eta_rank'],
        'delta_rank': profile['delta_rank'],
        'nuisance_rank': profile['nuisance_rank'],
        'quotient_gain': profile['quotient_gain'],
        'sigma_matrix': json.dumps(np.round(matrices['sigma'], 12).tolist(), separators=(',', ':')),
        'eta1_matrix': json.dumps(np.round(matrices['eta1'], 12).tolist(), separators=(',', ':')),
        'eta2_matrix': json.dumps(np.round(matrices['eta2'], 12).tolist(), separators=(',', ':')),
        'delta_matrix': json.dumps(np.round(matrices['delta'], 12).tolist(), separators=(',', ':')),
        'nuisance_matrix': json.dumps(np.round(matrices['nuisance'], 12).tolist(), separators=(',', ':')),
        'provenance': 'MEASURED_FROM_CODE',
    }


def verification_row(
    label: str,
    rank_value: int,
    best_seed: int,
    best_loss: float,
    verification: dict,
    profile: dict,
    symmetry: dict,
) -> dict:
    return {
        'label': label,
        'rank_value': rank_value,
        'best_seed': best_seed,
        'best_loss': scalar_to_str(best_loss),
        'max_abs_residual': scalar_to_str(verification['max_abs_residual']),
        'mean_abs_residual': scalar_to_str(verification['mean_abs_residual']),
        'entries_gt_1e-8': verification['entries_gt_1e8'],
        'entries_gt_1e-6': verification['entries_gt_1e6'],
        'verified_exact_under_1e-6': str(verification['verified_exact_under_1e6']),
        'sigma_rank': profile['sigma_rank'],
        'eta_rank': profile['eta_rank'],
        'delta_rank': profile['delta_rank'],
        'nuisance_rank': profile['nuisance_rank'],
        'quotient_gain': profile['quotient_gain'],
        'self_count': symmetry['self_count'],
        'pair_count': symmetry['pair_count'],
        'unpaired_count': symmetry['unpaired_count'],
        'provenance': 'MEASURED_FROM_CODE',
    }


def anti_table_markdown(rows: list[dict]) -> list[str]:
    lines = [
        '| term | alpha | beta | gamma |',
        '|------|-------|------|-------|',
    ]
    for row in rows:
        lines.append(f"| {row['term']} | {row['alpha']} | {row['beta']} | {row['gamma']} |")
    return lines


def summary_markdown(
    anti_result: dict,
    anti_verification: dict,
    anti_profile: dict,
    anti_symmetry: dict,
    anti_coeff_rows: list[dict],
    comm_result: dict,
    comm_verification: dict,
    comm_profile: dict,
    comm_symmetry: dict,
) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Step 75 Phase 1: Anticommutator Rank-19 Extraction + Hamilton Split')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[MEASURED_FROM_CODE]')
    w('')
    w('## Anticommutator {A,B} = AB + BA')
    w('')
    w(f"- Rank-19 restarts: {anti_result['restart_count']}")
    w(f"- Best seed: {anti_result['best_seed']}")
    w(f"- Best loss: {scalar_to_str(anti_result['best_loss'])}")
    w(f"- Max residual: {scalar_to_str(anti_verification['max_abs_residual'])}")
    w(f"- Mean residual: {scalar_to_str(anti_verification['mean_abs_residual'])}")
    w(f"- Entries with |residual| > 1e-8: {anti_verification['entries_gt_1e8']}")
    w(f"- Entries with |residual| > 1e-6: {anti_verification['entries_gt_1e6']}")
    w(f"- Verified exact under 1e-6: {anti_verification['verified_exact_under_1e6']}")
    w(f"- Fiber-mode: sigma_rank={anti_profile['sigma_rank']}, eta_rank={anti_profile['eta_rank']}, delta_rank={anti_profile['delta_rank']}, nuisance_rank={anti_profile['nuisance_rank']}, quotient_gain={anti_profile['quotient_gain']}")
    w(f"- Self-symmetric terms: {anti_symmetry['self_count']}")
    w(f"- Symmetric pairs: {anti_symmetry['pair_count']}")
    w(f"- Unpaired terms: {anti_symmetry['unpaired_count']}")
    w('')
    w('### Full 19-Term Coefficient Table')
    w('')
    lines.extend(anti_table_markdown(anti_coeff_rows))
    w('')
    w('## Commutator [A,B] = AB - BA')
    w('')
    w(f"- Rank-20 source seed: {comm_result['best_seed']}")
    w(f"- Best loss: {scalar_to_str(comm_result['best_loss'])}")
    w(f"- Max residual: {scalar_to_str(comm_verification['max_abs_residual'])}")
    w(f"- Mean residual: {scalar_to_str(comm_verification['mean_abs_residual'])}")
    w(f"- Entries with |residual| > 1e-8: {comm_verification['entries_gt_1e8']}")
    w(f"- Entries with |residual| > 1e-6: {comm_verification['entries_gt_1e6']}")
    w(f"- Verified exact under 1e-6: {comm_verification['verified_exact_under_1e6']}")
    w(f"- Fiber-mode: sigma_rank={comm_profile['sigma_rank']}, eta_rank={comm_profile['eta_rank']}, delta_rank={comm_profile['delta_rank']}, nuisance_rank={comm_profile['nuisance_rank']}, quotient_gain={comm_profile['quotient_gain']}")
    w(f"- Self-antisymmetric terms: {comm_symmetry['self_count']}")
    w(f"- Antisymmetric pairs: {comm_symmetry['pair_count']}")
    w(f"- Unpaired terms: {comm_symmetry['unpaired_count']}")
    w('')
    w('## Hamilton Split')
    w('')
    w('- T = ({A,B} + [A,B]) / 2')
    w(f"- rank(T_anti) = {ANTI_RANK if anti_verification['verified_exact_under_1e6'] else 'unverified'}")
    w(f"- rank(T_comm) = {COMM_RANK if comm_verification['verified_exact_under_1e6'] else 'unverified'}")
    naive_combined = (ANTI_RANK if anti_verification['verified_exact_under_1e6'] else ANTI_RANK) + (COMM_RANK if comm_verification['verified_exact_under_1e6'] else COMM_RANK)
    w(f'- Naive combined: {naive_combined}')
    w('- Awaiting derivation for sharing analysis.')
    return '\n'.join(lines)


def main() -> None:
    start = time.time()
    EXPORTS.mkdir(parents=True, exist_ok=True)

    specs = {spec.tensor_name: spec for spec in build_tensor_specs()}
    anti_tensor = specs['anticommutator'].tensor
    comm_tensor = specs['commutator'].tensor

    print('=== Step 75 Phase 1: Anticommutator Rank-19 Extraction + Hamilton Split ===', flush=True)
    print(f'Using {WORKERS} worker processes.', flush=True)

    anti_seeds = [restart_seed('anticommutator', ANTI_RANK, restart_idx) for restart_idx in range(ANTI_RESTARTS)]
    anti_result = scan_explicit_seeds('anticommutator', anti_tensor, ANTI_RANK, anti_seeds, WORKERS)
    anti_terms = factors_to_terms('anti', anti_result['best_alpha'], anti_result['best_beta'], anti_result['best_gamma'])
    for term_idx, term in enumerate(anti_terms, start=1):
        anti_terms[term_idx - 1] = Term(f's{term_idx:02d}', term.source_label, term.alpha, term.beta, term.gamma)

    anti_verification = verification_stats(anti_result['best_alpha'], anti_result['best_beta'], anti_result['best_gamma'], anti_tensor)
    anti_profile, anti_matrices = profile_for_terms(anti_terms)
    anti_symmetry, anti_pairs = symmetry_counts(anti_terms, gamma_sign=1.0)
    anti_coeff_rows = anti_coefficient_rows(anti_terms)

    comm_best_seed = read_step74_best_seed(STEP74_COMM_SCAN, COMM_RANK)
    comm_result = scan_explicit_seeds('commutator', comm_tensor, COMM_RANK, [comm_best_seed], 1)
    comm_terms = factors_to_terms('comm', comm_result['best_alpha'], comm_result['best_beta'], comm_result['best_gamma'])
    for term_idx, term in enumerate(comm_terms, start=1):
        comm_terms[term_idx - 1] = Term(f'c{term_idx:02d}', term.source_label, term.alpha, term.beta, term.gamma)

    comm_verification = verification_stats(comm_result['best_alpha'], comm_result['best_beta'], comm_result['best_gamma'], comm_tensor)
    comm_profile, comm_matrices = profile_for_terms(comm_terms)
    comm_symmetry, comm_pairs = symmetry_counts(comm_terms, gamma_sign=-1.0)

    anti_verification_rows = [
        verification_row('anticommutator_rank19', ANTI_RANK, anti_result['best_seed'], anti_result['best_loss'], anti_verification, anti_profile, anti_symmetry)
    ]
    comm_verification_rows = [
        verification_row('commutator_rank20', COMM_RANK, comm_result['best_seed'], comm_result['best_loss'], comm_verification, comm_profile, comm_symmetry)
    ]
    anti_fibermode_rows = [
        fibermode_row('anticommutator_rank19', anti_profile, anti_matrices)
    ]

    summary_md = summary_markdown(
        anti_result,
        anti_verification,
        anti_profile,
        anti_symmetry,
        anti_coeff_rows,
        comm_result,
        comm_verification,
        comm_profile,
        comm_symmetry,
    )

    write_csv(
        EXPORTS / 'step75_anticommutator_rank19_coefficients.csv',
        anti_coeff_rows,
        ['term', 'alpha', 'beta', 'gamma'],
    )
    write_csv(
        EXPORTS / 'step75_anticommutator_rank19_verification.csv',
        anti_verification_rows,
        ['label', 'rank_value', 'best_seed', 'best_loss', 'max_abs_residual', 'mean_abs_residual', 'entries_gt_1e-8', 'entries_gt_1e-6', 'verified_exact_under_1e-6', 'sigma_rank', 'eta_rank', 'delta_rank', 'nuisance_rank', 'quotient_gain', 'self_count', 'pair_count', 'unpaired_count', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step75_anticommutator_rank19_fibermode.csv',
        anti_fibermode_rows,
        ['label', 'sigma_rank', 'eta_rank', 'delta_rank', 'nuisance_rank', 'quotient_gain', 'sigma_matrix', 'eta1_matrix', 'eta2_matrix', 'delta_matrix', 'nuisance_matrix', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step75_commutator_rank20_verification.csv',
        comm_verification_rows,
        ['label', 'rank_value', 'best_seed', 'best_loss', 'max_abs_residual', 'mean_abs_residual', 'entries_gt_1e-8', 'entries_gt_1e-6', 'verified_exact_under_1e-6', 'sigma_rank', 'eta_rank', 'delta_rank', 'nuisance_rank', 'quotient_gain', 'self_count', 'pair_count', 'unpaired_count', 'provenance'],
    )
    write_text(EXPORTS / 'step75_phase1_summary.md', summary_md)

    elapsed = time.time() - start
    print(f'Completed Step 75 Phase 1 in {elapsed:.2f}s', flush=True)
    print(f'Anticommutator best loss: {scalar_to_str(anti_result["best_loss"])}', flush=True)
    print(f'Anticommutator max residual: {scalar_to_str(anti_verification["max_abs_residual"])}', flush=True)
    print(f'Commutator best loss: {scalar_to_str(comm_result["best_loss"])}', flush=True)
    print(f'Commutator max residual: {scalar_to_str(comm_verification["max_abs_residual"])}', flush=True)
    print(f'Anticommutator symmetry pairs: self={anti_symmetry["self_count"]}, paired={anti_symmetry["pair_count"]}, unpaired={anti_symmetry["unpaired_count"]}', flush=True)
    print(f'Commutator antisymmetry pairs: self={comm_symmetry["self_count"]}, paired={comm_symmetry["pair_count"]}, unpaired={comm_symmetry["unpaired_count"]}', flush=True)
    if anti_pairs:
        print(f'Anticommutator matched pair rows: {len(anti_pairs)}', flush=True)
    if comm_pairs:
        print(f'Commutator matched pair rows: {len(comm_pairs)}', flush=True)


if __name__ == '__main__':
    main()