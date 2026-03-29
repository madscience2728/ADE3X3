"""
ade3x3_step71_five_shots_at_the_wall.py

Step 71: Five Shots at the Wall.

Runs five independent probes around the public AlphaTensor rank-23 3x3 algorithm:
1. Slotwise term replacement over an explicit replacement pool.
2. Exact and approximate pair-merging checks.
3. Numerical CP-rank scans for the commutator and anticommutator tensors.
4. Detailed analysis of the three nonconstant-overlap AlphaTensor pairs from Step 70.
5. Random neighborhood feasibility search plus a full-729 greedy probe on a local pool.

Important scope note:
The repository exports the exact Step 64 orbit count (570,521) and the top-100 shortlist,
but not a materialized full 570,521-orbit representative table. Shot 1 therefore runs on the
largest explicit pool currently available in-repo plus additional random ternary samples.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from src.ade3x3.steps.ade3x3_step68_layered_correction_tiling import flattening_ranks, rank_scan_target
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
    swapped_matrix_multiplication_tensor,
)
from src.ade3x3.steps.ade3x3_step70_depth2_arithmetic_circuit_attack import classify_pair_ratio, profile_from_term


EXPORTS = Path('outputs/exports')
STEP64_TOP100 = EXPORTS / 'step64_top100_usefulness_profiles.csv'
STEP70_PAIRS = EXPORTS / 'step70_alphatensor_pair_ratios.csv'

SHOT1_RANDOM_CANDIDATES = 5000
SHOT5_RANDOM_POOL = 100
SHOT5_RANDOM_SUBSETS = 10000
SHOT5_GREEDY_MAX_RANK = 22
NUMERIC_TOL = 1e-9
EXACT_TOL = 1e-8


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    source: str
    alpha: np.ndarray
    beta: np.ndarray
    augmented_row: np.ndarray


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
    if np.issubdtype(matrix.dtype, np.integer):
        return json.dumps(matrix.astype(int).tolist(), separators=(',', ':'))
    return json.dumps(matrix.astype(float).tolist(), separators=(',', ':'))


def compute_term_augmented_row(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    sigma: list[float] = []
    eta1: list[float] = []
    eta2: list[float] = []
    delta: list[float] = []
    for row_idx in range(3):
        for col_idx in range(3):
            lambdas = [float(alpha[row_idx, sum_idx] * beta[sum_idx, col_idx]) for sum_idx in range(3)]
            sigma.append(sum(lambdas))
            eta1.append(lambdas[0] - lambdas[1])
            eta2.append(lambdas[1] - lambdas[2])
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    delta.append(float(alpha[row_idx, sum_left] * beta[sum_right, col_idx]))
    return np.array(sigma + eta1 + eta2 + delta, dtype=np.float64)


def numeric_rank(matrix: np.ndarray, tol: float = NUMERIC_TOL) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tol))


def quotient_metrics(augmented_rows: np.ndarray) -> tuple[int, int, int]:
    nuisance_rank = numeric_rank(augmented_rows[:, 9:])
    augmented_rank = numeric_rank(augmented_rows)
    return nuisance_rank, augmented_rank, augmented_rank - nuisance_rank


def target_augmented_matrix() -> np.ndarray:
    target = np.zeros((9, 81), dtype=np.float64)
    target[:, :9] = 3.0 * np.eye(9)
    return target


def solve_gamma_columns(augmented_rows: np.ndarray) -> tuple[np.ndarray, float]:
    target = target_augmented_matrix()
    gamma_columns = target @ np.linalg.pinv(augmented_rows)
    residual = float(np.max(np.abs(gamma_columns @ augmented_rows - target)))
    return gamma_columns, residual


def reconstruct_tensor(alphas: np.ndarray, betas: np.ndarray, gamma_columns: np.ndarray) -> np.ndarray:
    total = np.zeros((9, 9, 9), dtype=np.float64)
    for term_idx in range(alphas.shape[0]):
        total += np.einsum(
            'a,b,c->abc',
            alphas[term_idx].reshape(-1),
            betas[term_idx].reshape(-1),
            gamma_columns[:, term_idx],
            optimize=True,
        )
    return total


def verify_terms(alphas: np.ndarray, betas: np.ndarray, gamma_columns: np.ndarray) -> tuple[bool, float]:
    target = matrix_multiplication_tensor(3).astype(np.float64)
    residual = reconstruct_tensor(alphas, betas, gamma_columns) - target
    max_abs = float(np.max(np.abs(residual)))
    return max_abs <= EXACT_TOL, max_abs


def read_step64_top100() -> list[CandidateProfile]:
    rows: list[CandidateProfile] = []
    if not STEP64_TOP100.exists():
        return rows
    with open(STEP64_TOP100, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            alpha = np.array(ast.literal_eval(record['alpha']), dtype=np.float64)
            beta = np.array(ast.literal_eval(record['beta']), dtype=np.float64)
            rows.append(
                CandidateProfile(
                    candidate_id=f"step64_top_{int(record['rank']):03d}",
                    source='step64_top100_usefulness_profiles',
                    alpha=alpha,
                    beta=beta,
                    augmented_row=compute_term_augmented_row(alpha, beta),
                )
            )
    return rows


def canonicalize_vector(vector: np.ndarray) -> np.ndarray:
    flat = vector.reshape(-1).astype(np.int8)
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        raise ValueError('Zero vector is not allowed.')
    if flat[nz[0]] < 0:
        flat = -flat
    return flat.reshape(vector.shape)


def random_nonzero_ternary_matrix(rng: np.random.Generator) -> np.ndarray:
    while True:
        mat = rng.integers(-1, 2, size=(3, 3), dtype=np.int8)
        if np.any(mat):
            return canonicalize_vector(mat).astype(np.float64)


def build_replacement_pool(base_terms: list[Term], random_candidates: int) -> tuple[list[CandidateProfile], str]:
    pool: list[CandidateProfile] = []
    seen: set[tuple[bytes, bytes]] = set()

    for term in base_terms:
        key = (term.alpha.astype(np.int8).tobytes(), term.beta.astype(np.int8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        pool.append(
            CandidateProfile(
                candidate_id=term.term_id,
                source='alphatensor23',
                alpha=term.alpha.astype(np.float64),
                beta=term.beta.astype(np.float64),
                augmented_row=compute_term_augmented_row(term.alpha, term.beta),
            )
        )

    for candidate in read_step64_top100():
        key = (candidate.alpha.astype(np.int8).tobytes(), candidate.beta.astype(np.int8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        pool.append(candidate)

    rng = np.random.default_rng(710071)
    added = 0
    while added < random_candidates:
        alpha = random_nonzero_ternary_matrix(rng)
        beta = random_nonzero_ternary_matrix(rng)
        key = (alpha.astype(np.int8).tobytes(), beta.astype(np.int8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        added += 1
        pool.append(
            CandidateProfile(
                candidate_id=f'random_{added:05d}',
                source='random_ternary_local_pool',
                alpha=alpha,
                beta=beta,
                augmented_row=compute_term_augmented_row(alpha, beta),
            )
        )

    scope_note = (
        'Step 64 exports the exact symmetry-orbit count but not a materialized 570521-orbit '
        f'representative table. Shot 1 scanned the explicit AlphaTensor+Step64-top100 pool plus {random_candidates} '
        'unique random ternary profiles instead of the unattached full orbit table.'
    )
    return pool, scope_note


def shot1_worker(payload: dict) -> tuple[list[dict], dict]:
    slot_index = int(payload['slot_index'])
    slot_id = str(payload['slot_id'])
    base_alphas = np.array(payload['base_alphas'], dtype=np.float64)
    base_betas = np.array(payload['base_betas'], dtype=np.float64)
    base_rows = np.array(payload['base_rows'], dtype=np.float64)
    candidates = payload['candidates']
    removed_alpha = np.array(payload['removed_alpha'], dtype=np.float64)
    removed_beta = np.array(payload['removed_beta'], dtype=np.float64)

    found_rows: list[dict] = []
    scanned = 0
    feasible_count = 0

    for candidate in candidates:
        candidate_alpha = np.array(candidate['alpha'], dtype=np.float64)
        candidate_beta = np.array(candidate['beta'], dtype=np.float64)
        if np.array_equal(candidate_alpha, removed_alpha) and np.array_equal(candidate_beta, removed_beta):
            continue

        scanned += 1
        augmented = np.vstack([base_rows, np.array(candidate['augmented_row'], dtype=np.float64)])
        nuisance_rank, augmented_rank, quotient_gain = quotient_metrics(augmented)
        if quotient_gain != 9:
            continue

        alphas = np.concatenate([base_alphas, candidate_alpha[None, :, :]], axis=0)
        betas = np.concatenate([base_betas, candidate_beta[None, :, :]], axis=0)
        gamma_columns, gamma_residual = solve_gamma_columns(augmented)
        verified, max_abs = verify_terms(alphas, betas, gamma_columns)
        if not verified:
            continue

        feasible_count += 1
        found_rows.append({
            'removed_term': slot_id,
            'replacement_id': candidate['candidate_id'],
            'replacement_source': candidate['source'],
            'replacement_alpha': matrix_to_json(candidate_alpha),
            'replacement_beta': matrix_to_json(candidate_beta),
            'candidate_nuisance_rank': nuisance_rank,
            'candidate_augmented_rank': augmented_rank,
            'candidate_quotient_gain': quotient_gain,
            'gamma_solver_max_abs_residual': gamma_residual,
            'full_tensor_verified': verified,
            'full_tensor_max_abs_residual': max_abs,
            'replacement_gamma_column': matrix_to_json(gamma_columns[:, -1].reshape(3, 3)),
            'provenance': 'MEASURED_FROM_CODE',
        })

    summary = {
        'removed_term': slot_id,
        'slot_index': slot_index,
        'candidates_scanned': scanned,
        'valid_replacements': feasible_count,
        'provenance': 'MEASURED_FROM_CODE',
    }
    return found_rows, summary


def run_shot1(base_terms: list[Term], pool: list[CandidateProfile], scope_note: str) -> tuple[str, dict]:
    original_alphas = np.stack([term.alpha.astype(np.float64) for term in base_terms], axis=0)
    original_betas = np.stack([term.beta.astype(np.float64) for term in base_terms], axis=0)
    original_rows = np.stack([compute_term_augmented_row(term.alpha, term.beta) for term in base_terms], axis=0)

    serializable_candidates = [
        {
            'candidate_id': candidate.candidate_id,
            'source': candidate.source,
            'alpha': candidate.alpha.tolist(),
            'beta': candidate.beta.tolist(),
            'augmented_row': candidate.augmented_row.tolist(),
        }
        for candidate in pool
    ]

    payloads = []
    for slot_index, term in enumerate(base_terms):
        keep_mask = np.array([idx != slot_index for idx in range(len(base_terms))])
        payloads.append({
            'slot_index': slot_index,
            'slot_id': term.term_id,
            'base_alphas': original_alphas[keep_mask].tolist(),
            'base_betas': original_betas[keep_mask].tolist(),
            'base_rows': original_rows[keep_mask].tolist(),
            'removed_alpha': term.alpha.tolist(),
            'removed_beta': term.beta.tolist(),
            'candidates': serializable_candidates,
        })

    replacement_rows: list[dict] = []
    slot_rows: list[dict] = []
    worker_count = min(8, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=worker_count) as pool_exec:
        futures = [pool_exec.submit(shot1_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            found_rows, summary = future.result()
            replacement_rows.extend(found_rows)
            slot_rows.append(summary)
            print(
                f"Shot 1 completed slot {summary['removed_term']}: {summary['valid_replacements']} valid replacements after {summary['candidates_scanned']} scans.",
                flush=True,
            )

    slot_rows.sort(key=lambda row: int(row['slot_index']))
    for row in slot_rows:
        del row['slot_index']
        row['pool_scope_note'] = scope_note

    total_valid = sum(int(row['valid_replacements']) for row in slot_rows)
    slots_with_replacements = sum(int(row['valid_replacements']) > 0 for row in slot_rows)
    summary_rows = [
        {
            'summary_name': 'shot1_pool_size',
            'summary_value': str(len(pool)),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Explicit local replacement pool size used for Step 71 Shot 1.',
        },
        {
            'summary_name': 'shot1_total_valid_replacements',
            'summary_value': str(total_valid),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Total exact slot replacements found across all 23 removal slots.',
        },
        {
            'summary_name': 'shot1_slots_with_any_replacement',
            'summary_value': str(slots_with_replacements),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'How many of the 23 removal slots admit at least one exact replacement in the scanned pool.',
        },
        {
            'summary_name': 'shot1_orbit_scan_status',
            'summary_value': 'full_step64_orbit_table_not_materialized_in_repo',
            'provenance': 'EXACT_DERIVED',
            'note': scope_note,
        },
    ]
    write_csv(
        EXPORTS / 'step71_shot1_slot_summary.csv',
        slot_rows,
        ['removed_term', 'candidates_scanned', 'valid_replacements', 'pool_scope_note', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step71_shot1_replacements.csv',
        replacement_rows,
        [
            'removed_term',
            'replacement_id',
            'replacement_source',
            'replacement_alpha',
            'replacement_beta',
            'candidate_nuisance_rank',
            'candidate_augmented_rank',
            'candidate_quotient_gain',
            'gamma_solver_max_abs_residual',
            'full_tensor_verified',
            'full_tensor_max_abs_residual',
            'replacement_gamma_column',
            'provenance',
        ],
    )
    verdict = 'worked' if total_valid > 0 else 'did_not_work'
    line = f'Shot 1: {verdict}; {total_valid} exact replacements across {slots_with_replacements}/23 slots in pool size {len(pool)}.'
    return line, {'summary_rows': summary_rows}


def factor_rank1_profile(profile_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    profile_matrix = profile_vec.reshape(9, 9)
    singular_values = np.linalg.svd(profile_matrix, compute_uv=False)
    bilinear_rank = int(np.sum(singular_values > NUMERIC_TOL))
    bilinear_error = float(np.sum(singular_values[1:] ** 2)) if singular_values.size > 1 else 0.0
    if bilinear_rank != 1:
        return np.zeros((3, 3), dtype=np.float64), np.zeros((3, 3), dtype=np.float64), bilinear_rank, bilinear_error
    u, s, vh = np.linalg.svd(profile_matrix, full_matrices=False)
    alpha_vec = u[:, 0] * np.sqrt(s[0])
    beta_vec = vh[0, :] * np.sqrt(s[0])
    return alpha_vec.reshape(3, 3), beta_vec.reshape(3, 3), bilinear_rank, bilinear_error


def run_shot2(base_terms: list[Term]) -> tuple[str, dict]:
    pair_rows: list[dict] = []
    merge_rows: list[dict] = []
    best_pair = None
    best_error = float('inf')

    for idx, jdx in combinations(range(len(base_terms)), 2):
        term_i = base_terms[idx]
        term_j = base_terms[jdx]
        profile_i = profile_from_term(term_i.alpha, term_i.beta).reshape(-1)
        profile_j = profile_from_term(term_j.alpha, term_j.beta).reshape(-1)
        matrix = np.outer(term_i.gamma.reshape(-1), profile_i) + np.outer(term_j.gamma.reshape(-1), profile_j)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        output_rank = int(np.sum(singular_values > NUMERIC_TOL))
        rank1_error = float(np.sum(singular_values[1:] ** 2)) if singular_values.size > 1 else 0.0
        merged_profile = vh[0, :] * singular_values[0] if singular_values.size else np.zeros(81, dtype=np.float64)
        alpha_prime, beta_prime, bilinear_rank, bilinear_error = factor_rank1_profile(merged_profile)
        mergeable = output_rank == 1 and bilinear_rank == 1
        verified = False
        max_abs = ''
        merged_gamma = np.zeros((3, 3), dtype=np.float64)
        if mergeable:
            merged_gamma = u[:, 0].reshape(3, 3)
            kept_alphas = [base_terms[k].alpha.astype(np.float64) for k in range(len(base_terms)) if k not in {idx, jdx}]
            kept_betas = [base_terms[k].beta.astype(np.float64) for k in range(len(base_terms)) if k not in {idx, jdx}]
            kept_gamma_cols = [base_terms[k].gamma.reshape(-1).astype(np.float64) for k in range(len(base_terms)) if k not in {idx, jdx}]
            kept_alphas.append(alpha_prime)
            kept_betas.append(beta_prime)
            kept_gamma_cols.append(merged_gamma.reshape(-1))
            residual = reconstruct_tensor(
                np.stack(kept_alphas, axis=0),
                np.stack(kept_betas, axis=0),
                np.stack(kept_gamma_cols, axis=1),
            ) - matrix_multiplication_tensor(3).astype(np.float64)
            max_abs = float(np.max(np.abs(residual)))
            verified = max_abs <= EXACT_TOL
            if verified:
                merge_rows.append({
                    'term_i': term_i.term_id,
                    'term_j': term_j.term_id,
                    'merged_alpha': matrix_to_json(alpha_prime),
                    'merged_beta': matrix_to_json(beta_prime),
                    'merged_gamma': matrix_to_json(merged_gamma),
                    'full_tensor_max_abs_residual': max_abs,
                    'provenance': 'MEASURED_FROM_CODE',
                })

        if rank1_error < best_error:
            best_error = rank1_error
            best_pair = (term_i.term_id, term_j.term_id, output_rank, bilinear_rank, rank1_error, bilinear_error)

        pair_rows.append({
            'term_i': term_i.term_id,
            'term_j': term_j.term_id,
            'output_flatten_rank': output_rank,
            'rank1_frobenius_error_squared': rank1_error,
            'best_profile_bilinear_rank': bilinear_rank,
            'best_profile_bilinear_rank1_error_squared': bilinear_error,
            'mergeable_to_single_rank1_term': mergeable,
            'verified_r22_algorithm': verified,
            'provenance': 'MEASURED_FROM_CODE',
        })

    summary_rows = [
        {
            'summary_name': 'shot2_exact_mergeable_pair_count',
            'summary_value': str(len(merge_rows)),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Exact term pairs whose combined contribution is a single rank-1 tensor and verifies a 22-term algorithm.',
        },
        {
            'summary_name': 'shot2_best_near_merge_pair',
            'summary_value': f'{best_pair[0]},{best_pair[1]}' if best_pair else 'none',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Pair with smallest rank-1 approximation error in the output-vs-profile flattening.',
        },
        {
            'summary_name': 'shot2_best_near_merge_error_squared',
            'summary_value': f'{best_error:.12e}',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Best Frobenius squared tail after the leading singular value.',
        },
    ]
    write_csv(
        EXPORTS / 'step71_shot2_pair_merge_scan.csv',
        pair_rows,
        [
            'term_i',
            'term_j',
            'output_flatten_rank',
            'rank1_frobenius_error_squared',
            'best_profile_bilinear_rank',
            'best_profile_bilinear_rank1_error_squared',
            'mergeable_to_single_rank1_term',
            'verified_r22_algorithm',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step71_shot2_exact_merged_algorithms.csv',
        merge_rows,
        ['term_i', 'term_j', 'merged_alpha', 'merged_beta', 'merged_gamma', 'full_tensor_max_abs_residual', 'provenance'],
    )
    verdict = 'worked' if merge_rows else 'did_not_work'
    best_pair_text = 'none' if best_pair is None else f'{best_pair[0]}+{best_pair[1]}'
    line = f'Shot 2: {verdict}; exact mergeable pairs={len(merge_rows)}, best near-merge={best_pair_text} with error^2={best_error:.3e}.'
    return line, {'summary_rows': summary_rows}


def build_commutator_tensors() -> tuple[np.ndarray, np.ndarray]:
    mult = matrix_multiplication_tensor(3).astype(np.float64)
    swapped = swapped_matrix_multiplication_tensor(3).astype(np.float64)
    return mult - swapped, mult + swapped


def best_exact_rank(rows: list[dict]) -> str:
    for row in rows:
        if float(row['best_verified_loss']) <= EXACT_TOL:
            return str(row['rank_tested'])
    return 'none'


def run_shot3() -> tuple[str, dict]:
    workers = max(1, min(3, (os.cpu_count() or 1) // 4 or 1))
    comm_tensor, anti_tensor = build_commutator_tensors()
    ranks = list(range(8, 21))

    comm_rows, comm_verdict = rank_scan_target('step71_commutator_3x3', comm_tensor, ranks=ranks, restarts=200, workers=workers)
    anti_rows, anti_verdict = rank_scan_target('step71_anticommutator_3x3', anti_tensor, ranks=ranks, restarts=200, workers=workers)

    comm_exact = best_exact_rank(comm_rows)
    anti_exact = best_exact_rank(anti_rows)
    comm_lb = flattening_ranks(comm_tensor)[3]
    anti_lb = flattening_ranks(anti_tensor)[3]
    shared_status = 'not_triggered'
    if comm_exact != 'none' and anti_exact != 'none' and int(comm_exact) + int(anti_exact) < 46:
        shared_status = 'upper_bounds_below_46_but_no_explicit_factor_lists_exported_from_rank_scan'

    summary_rows = [
        {
            'summary_name': 'shot3_commutator_flattening_lb',
            'summary_value': str(comm_lb),
            'provenance': 'EXACT_DERIVED',
            'note': 'Flattening lower bound for the 3x3 commutator tensor.',
        },
        {
            'summary_name': 'shot3_anticommutator_flattening_lb',
            'summary_value': str(anti_lb),
            'provenance': 'EXACT_DERIVED',
            'note': 'Flattening lower bound for the 3x3 anticommutator tensor.',
        },
        {
            'summary_name': 'shot3_commutator_first_exact_rank_in_scan',
            'summary_value': comm_exact,
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'First exact rank hit, if any, in the requested numeric scan.',
        },
        {
            'summary_name': 'shot3_anticommutator_first_exact_rank_in_scan',
            'summary_value': anti_exact,
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'First exact rank hit, if any, in the requested numeric scan.',
        },
        {
            'summary_name': 'shot3_shared_term_status',
            'summary_value': shared_status,
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Whether the shared-term follow-up was triggered.',
        },
    ]
    write_csv(
        EXPORTS / 'step71_shot3_commutator_rank_scan.csv',
        comm_rows,
        list(comm_rows[0].keys()) if comm_rows else ['piece_id', 'rank'],
    )
    write_csv(
        EXPORTS / 'step71_shot3_anticommutator_rank_scan.csv',
        anti_rows,
        list(anti_rows[0].keys()) if anti_rows else ['piece_id', 'rank'],
    )
    write_csv(
        EXPORTS / 'step71_shot3_summary.csv',
        summary_rows,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    verdict = 'promising' if comm_exact != 'none' or anti_exact != 'none' else 'did_not_work'
    line = f'Shot 3: {verdict}; commutator exact rank in scan={comm_exact}, anticommutator exact rank in scan={anti_exact}, flattening LBs=({comm_lb},{anti_lb}).'
    return line, {'summary_rows': summary_rows, 'comm_verdict': comm_verdict, 'anti_verdict': anti_verdict}


def overlap_positions_and_ratios(profile_i: np.ndarray, profile_j: np.ndarray) -> tuple[list[dict], bool, bool, bool]:
    positions: list[dict] = []
    ratio_matrix = np.full(profile_i.shape, np.nan, dtype=np.float64)
    for a_idx in range(profile_i.shape[0]):
        for b_idx in range(profile_i.shape[1]):
            if profile_i[a_idx, b_idx] != 0 and profile_j[a_idx, b_idx] != 0:
                ratio = float(profile_i[a_idx, b_idx] / profile_j[a_idx, b_idx])
                ratio_matrix[a_idx, b_idx] = ratio
                positions.append({
                    'a_idx': a_idx,
                    'b_idx': b_idx,
                    'ratio': ratio,
                    'a_coord': f'({a_idx // 3},{a_idx % 3})',
                    'b_coord': f'({b_idx // 3},{b_idx % 3})',
                })

    row_constant = True
    row_values = {}
    for a_idx in range(profile_i.shape[0]):
        vals = [entry['ratio'] for entry in positions if entry['a_idx'] == a_idx]
        if vals and len(set(vals)) != 1:
            row_constant = False
            break
        if vals:
            row_values[a_idx] = vals[0]

    col_constant = True
    col_values = {}
    for b_idx in range(profile_i.shape[1]):
        vals = [entry['ratio'] for entry in positions if entry['b_idx'] == b_idx]
        if vals and len(set(vals)) != 1:
            col_constant = False
            break
        if vals:
            col_values[b_idx] = vals[0]

    separable = True
    if positions:
        left = {}
        right = {}
        first = positions[0]
        left[first['a_idx']] = 1.0
        right[first['b_idx']] = first['ratio']
        changed = True
        while changed:
            changed = False
            for entry in positions:
                a_idx = entry['a_idx']
                b_idx = entry['b_idx']
                ratio = entry['ratio']
                if a_idx in left and b_idx not in right:
                    right[b_idx] = ratio / left[a_idx]
                    changed = True
                elif b_idx in right and a_idx not in left:
                    left[a_idx] = ratio / right[b_idx]
                    changed = True
        for entry in positions:
            a_idx = entry['a_idx']
            b_idx = entry['b_idx']
            if a_idx not in left or b_idx not in right or abs(left[a_idx] * right[b_idx] - entry['ratio']) > 1e-9:
                separable = False
                break

    return positions, row_constant, col_constant, separable


def run_shot4(base_terms: list[Term]) -> tuple[str, dict]:
    if not STEP70_PAIRS.exists():
        raise RuntimeError('Step 70 pair-ratio export is required before running Shot 4.')

    unstructured_pairs: list[tuple[str, str]] = []
    with open(STEP70_PAIRS, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row['classification'] == 'unstructured_overlap':
                unstructured_pairs.append((row['term_i'], row['term_j']))

    by_id = {term.term_id: term for term in base_terms}
    rows: list[dict] = []
    promising = 0

    for term_i_id, term_j_id in unstructured_pairs:
        term_i = by_id[term_i_id]
        term_j = by_id[term_j_id]
        profile_i = profile_from_term(term_i.alpha, term_i.beta)
        profile_j = profile_from_term(term_j.alpha, term_j.beta)
        positions, row_constant, col_constant, separable = overlap_positions_and_ratios(profile_i, profile_j)
        matrix = np.outer(term_i.gamma.reshape(-1), profile_i.reshape(-1)) + np.outer(term_j.gamma.reshape(-1), profile_j.reshape(-1))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        merge_error = float(np.sum(singular_values[1:] ** 2)) if singular_values.size > 1 else 0.0
        if merge_error < 1e-6:
            promising += 1
        rows.append({
            'term_i': term_i_id,
            'term_j': term_j_id,
            'overlap_positions_json': json.dumps(positions, separators=(',', ':')),
            'ratio_values_json': json.dumps(sorted({entry['ratio'] for entry in positions}), separators=(',', ':')),
            'ratio_depends_only_on_alpha_index': row_constant,
            'ratio_depends_only_on_beta_index': col_constant,
            'overlap_ratio_separable': separable,
            'pair_merge_rank1_error_squared': merge_error,
            'provenance': 'MEASURED_FROM_CODE',
        })

    summary_rows = [
        {
            'summary_name': 'shot4_nonconstant_overlap_pair_count',
            'summary_value': str(len(rows)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Count inherited from the Step 70 pair-ratio table.',
        },
        {
            'summary_name': 'shot4_near_merge_pairs_below_1e-6',
            'summary_value': str(promising),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Pairs whose Shot 2 merge error falls below 1e-6.',
        },
    ]
    write_csv(
        EXPORTS / 'step71_shot4_overlap_pairs.csv',
        rows,
        [
            'term_i',
            'term_j',
            'overlap_positions_json',
            'ratio_values_json',
            'ratio_depends_only_on_alpha_index',
            'ratio_depends_only_on_beta_index',
            'overlap_ratio_separable',
            'pair_merge_rank1_error_squared',
            'provenance',
        ],
    )
    verdict = 'promising' if promising > 0 else 'did_not_work'
    line = f'Shot 4: {verdict}; analyzed {len(rows)} nonconstant-overlap pairs, near-merge count below 1e-6 = {promising}.'
    return line, {'summary_rows': summary_rows}


def random_pool_for_shot5(base_terms: list[Term]) -> list[CandidateProfile]:
    pool: list[CandidateProfile] = []
    seen: set[tuple[bytes, bytes]] = set()
    for term in base_terms:
        alpha = term.alpha.astype(np.float64)
        beta = term.beta.astype(np.float64)
        key = (alpha.astype(np.int8).tobytes(), beta.astype(np.int8).tobytes())
        seen.add(key)
        pool.append(CandidateProfile(term.term_id, 'alphatensor23', alpha, beta, compute_term_augmented_row(alpha, beta)))

    rng = np.random.default_rng(710571)
    added = 0
    while added < SHOT5_RANDOM_POOL:
        alpha = random_nonzero_ternary_matrix(rng)
        beta = random_nonzero_ternary_matrix(rng)
        key = (alpha.astype(np.int8).tobytes(), beta.astype(np.int8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        added += 1
        pool.append(CandidateProfile(f'shot5_random_{added:03d}', 'random_ternary_neighborhood', alpha, beta, compute_term_augmented_row(alpha, beta)))
    return pool


def subset_worker(payload: dict) -> dict:
    rows = np.array(payload['rows'], dtype=np.float64)
    sizes = payload['sizes']
    seed = int(payload['seed'])
    pool_size = rows.shape[0]
    per_size = int(payload['per_size'])
    rng = np.random.default_rng(seed)
    hits = {size: 0 for size in sizes}
    best = {size: {'quotient_gain': -1, 'subset': ''} for size in sizes}
    witness = None

    for size in sizes:
        for _ in range(per_size):
            subset = np.sort(rng.choice(pool_size, size=size, replace=False))
            nuisance_rank, augmented_rank, quotient_gain = quotient_metrics(rows[subset])
            if quotient_gain > best[size]['quotient_gain']:
                best[size] = {'quotient_gain': quotient_gain, 'subset': ','.join(str(int(idx)) for idx in subset)}
            if quotient_gain == 9:
                hits[size] += 1
                witness = {'size': size, 'subset': subset.tolist(), 'nuisance_rank': nuisance_rank, 'augmented_rank': augmented_rank}
                break
        if witness is not None:
            break

    return {'hits': hits, 'best': best, 'witness': witness}


def candidate_basis_columns(profile: CandidateProfile) -> np.ndarray:
    profile_flat = np.outer(profile.alpha.reshape(-1), profile.beta.reshape(-1)).reshape(-1)
    basis = np.zeros((729, 9), dtype=np.float64)
    for col_idx in range(9):
        basis[col_idx::9, col_idx] = profile_flat
    return basis


def full_tensor_greedy(pool: list[CandidateProfile]) -> list[dict]:
    target = matrix_multiplication_tensor(3).astype(np.float64).reshape(-1)
    basis_cache = [candidate_basis_columns(candidate) for candidate in pool]
    selected: list[int] = []
    rows: list[dict] = []
    current_matrix = np.zeros((729, 0), dtype=np.float64)

    for rank_step in range(1, SHOT5_GREEDY_MAX_RANK + 1):
        best_idx = -1
        best_residual = float('inf')
        best_gamma = None
        for idx, basis in enumerate(basis_cache):
            if idx in selected:
                continue
            trial_matrix = basis if current_matrix.size == 0 else np.hstack([current_matrix, basis])
            gamma_vec, _, _, _ = np.linalg.lstsq(trial_matrix, target, rcond=None)
            residual = target - trial_matrix @ gamma_vec
            residual_norm_sq = float(np.dot(residual, residual))
            if residual_norm_sq < best_residual:
                best_residual = residual_norm_sq
                best_idx = idx
                best_gamma = gamma_vec
        if best_idx < 0:
            break
        selected.append(best_idx)
        current_matrix = basis_cache[best_idx] if current_matrix.size == 0 else np.hstack([current_matrix, basis_cache[best_idx]])
        chosen_rows = np.stack([pool[idx].augmented_row for idx in selected], axis=0)
        nuisance_rank, augmented_rank, quotient_gain = quotient_metrics(chosen_rows)
        rows.append({
            'rank_step': rank_step,
            'selected_candidate_id': pool[best_idx].candidate_id,
            'selected_source': pool[best_idx].source,
            'residual_norm_squared': best_residual,
            'quotient_gain': quotient_gain,
            'nuisance_rank': nuisance_rank,
            'augmented_rank': augmented_rank,
            'exact_zero': best_residual <= EXACT_TOL,
            'gamma_vector_json': json.dumps(best_gamma.tolist(), separators=(',', ':')) if best_gamma is not None else '[]',
            'provenance': 'MEASURED_FROM_CODE',
        })
        if best_residual <= EXACT_TOL:
            break
    return rows


def run_shot5(base_terms: list[Term]) -> tuple[str, dict]:
    pool = random_pool_for_shot5(base_terms)
    rows = np.stack([candidate.augmented_row for candidate in pool], axis=0)
    sizes = [20, 21, 22]
    worker_count = min(6, max(1, os.cpu_count() or 1))
    per_worker = SHOT5_RANDOM_SUBSETS // worker_count
    remainder = SHOT5_RANDOM_SUBSETS % worker_count
    subset_rows: list[dict] = []
    feasible_witness = None
    best_by_size = {size: {'quotient_gain': -1, 'subset': ''} for size in sizes}
    hit_counts = {size: 0 for size in sizes}

    payloads = []
    for worker_idx in range(worker_count):
        payloads.append({
            'rows': rows.tolist(),
            'sizes': sizes,
            'seed': 900000 + worker_idx,
            'per_size': per_worker + (1 if worker_idx < remainder else 0),
        })

    with ProcessPoolExecutor(max_workers=worker_count) as pool_exec:
        futures = [pool_exec.submit(subset_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            for size in sizes:
                hit_counts[size] += int(result['hits'][size])
                if result['best'][size]['quotient_gain'] > best_by_size[size]['quotient_gain']:
                    best_by_size[size] = result['best'][size]
            if feasible_witness is None and result['witness'] is not None:
                feasible_witness = result['witness']

    for size in sizes:
        subset_rows.append({
            'subset_size': size,
            'samples_tested': SHOT5_RANDOM_SUBSETS,
            'feasible_subset_hits': hit_counts[size],
            'best_quotient_gain': best_by_size[size]['quotient_gain'],
            'best_subset_pool_indices': best_by_size[size]['subset'],
            'provenance': 'MEASURED_FROM_CODE',
        })

    witness_rows: list[dict] = []
    if feasible_witness is not None:
        subset = feasible_witness['subset']
        witness_rows.append({
            'subset_size': feasible_witness['size'],
            'pool_indices': ','.join(str(idx) for idx in subset),
            'candidate_ids': ';'.join(pool[idx].candidate_id for idx in subset),
            'nuisance_rank': feasible_witness['nuisance_rank'],
            'augmented_rank': feasible_witness['augmented_rank'],
            'provenance': 'MEASURED_FROM_CODE',
        })

    greedy_rows = full_tensor_greedy(pool)
    best_greedy = greedy_rows[-1] if greedy_rows else None
    summary_rows = [
        {
            'summary_name': 'shot5_pool_size',
            'summary_value': str(len(pool)),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Pool size for the random-neighborhood subset and greedy search.',
        },
        {
            'summary_name': 'shot5_found_feasible_subset_below_23',
            'summary_value': str(feasible_witness is not None),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Whether random subset search found a feasible R=20,21, or 22 subset.',
        },
        {
            'summary_name': 'shot5_greedy_final_rank',
            'summary_value': str(best_greedy['rank_step']) if best_greedy else '0',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Final rank reached by the full-729 greedy on the local pool.',
        },
        {
            'summary_name': 'shot5_greedy_final_residual_norm_squared',
            'summary_value': f"{float(best_greedy['residual_norm_squared']):.12e}" if best_greedy else '0.0',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Best residual achieved by the greedy trace.',
        },
    ]
    write_csv(
        EXPORTS / 'step71_shot5_random_subset_scan.csv',
        subset_rows,
        ['subset_size', 'samples_tested', 'feasible_subset_hits', 'best_quotient_gain', 'best_subset_pool_indices', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step71_shot5_subset_witnesses.csv',
        witness_rows,
        ['subset_size', 'pool_indices', 'candidate_ids', 'nuisance_rank', 'augmented_rank', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step71_shot5_greedy_trace.csv',
        greedy_rows,
        ['rank_step', 'selected_candidate_id', 'selected_source', 'residual_norm_squared', 'quotient_gain', 'nuisance_rank', 'augmented_rank', 'exact_zero', 'gamma_vector_json', 'provenance'],
    )
    verdict = 'worked' if feasible_witness is not None else 'did_not_work'
    best_gain = max(int(row['best_quotient_gain']) for row in subset_rows) if subset_rows else 0
    greedy_text = 'none' if best_greedy is None else f"R={best_greedy['rank_step']} residual^2={float(best_greedy['residual_norm_squared']):.3e}"
    line = f'Shot 5: {verdict}; random subsets found_feasible={feasible_witness is not None}, best quotient gain={best_gain}, greedy={greedy_text}.'
    return line, {'summary_rows': summary_rows}


def markdown_report(lines: list[str], summary_rows: list[dict]) -> str:
    summary = {row['summary_name']: row['summary_value'] for row in summary_rows}
    shot1_total = summary.get('shot1_total_valid_replacements', '0')
    shot2_merges = summary.get('shot2_exact_mergeable_pair_count', '0')
    shot3_comm = summary.get('shot3_commutator_first_exact_rank_in_scan', 'none')
    shot3_anti = summary.get('shot3_anticommutator_first_exact_rank_in_scan', 'none')
    shot4_count = summary.get('shot4_near_merge_pairs_below_1e-6', '0')
    shot5_feasible = summary.get('shot5_found_feasible_subset_below_23', 'False')
    out: list[str] = []
    out.append('# Step 71: Five Shots at the Wall')
    out.append('')
    out.append('## One-Line Results')
    for line in lines:
        out.append(f'- {line}')
    out.append('')
    out.append('## Aggregate Verdict')
    out.append(
        f'The slot-replacement scan found {shot1_total} exact replacements in the scanned explicit pool, '
        f'the pair-merge scan found {shot2_merges} exact 22-term merges, the commutator/anticommutator numeric '
        f'scan first hit exact ranks ({shot3_comm}, {shot3_anti}), the three Step 70 overlap pairs yielded {shot4_count} '
        f'near-merges below 1e-6, and the random-neighborhood search found an R<23 feasible subset = {shot5_feasible}.'
    )
    out.append('')
    out.append('The requested full Step 64 570,521-orbit replacement scan could not be run exactly because the repository ') 
    out.append('does not contain a materialized orbit-representative table, only the exact orbit count and the exported top-100 shortlist.')
    return '\n'.join(out) + '\n'


def main() -> None:
    start = time.time()
    base_terms, _, _ = load_public_rank23_terms()
    replacement_pool, scope_note = build_replacement_pool(base_terms, SHOT1_RANDOM_CANDIDATES)

    shots = [
        ('shot1', run_shot1, (base_terms, replacement_pool, scope_note)),
        ('shot2', run_shot2, (base_terms,)),
        ('shot3', run_shot3, ()),
        ('shot4', run_shot4, (base_terms,)),
        ('shot5', run_shot5, (base_terms,)),
    ]

    ordered_lines: dict[str, str] = {}
    summary_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as pool_exec:
        future_map = {pool_exec.submit(func, *args): key for key, func, args in shots}
        for future in as_completed(future_map):
            key = future_map[future]
            line, payload = future.result()
            ordered_lines[key] = line
            summary_rows.extend(payload.get('summary_rows', []))
            print(line, flush=True)

    major_result = any(
        row['summary_name'] in {'shot2_exact_mergeable_pair_count', 'shot5_found_feasible_subset_below_23'} and row['summary_value'] not in {'0', 'False'}
        for row in summary_rows
    )
    summary_rows.extend([
        {
            'summary_name': 'step71_major_result_flag',
            'summary_value': str(major_result),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Immediate stop flag requested by the user if any verified R < 23 witness appears.',
        },
        {
            'summary_name': 'step71_runtime_seconds',
            'summary_value': f'{time.time() - start:.3f}',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Wall-clock runtime for Step 71.',
        },
    ])

    ordered = [ordered_lines[key] for key in sorted(ordered_lines)]
    write_csv(
        EXPORTS / 'step71_summary.csv',
        summary_rows,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_text(EXPORTS / 'step71_five_shots_at_the_wall.md', markdown_report(ordered, summary_rows))
    print(f'Step 71 complete in {time.time() - start:.2f}s', flush=True)


if __name__ == '__main__':
    main()