"""
ade3x3_step83b_support_expansion_sparse_meta_analysis.py

Step 83b: Support expansion + heuristic sparse meta-analysis.

This step runs two linked campaigns around the Step 81/83 homotopy machinery.

Track 1 starts from the strongest Step 83 singleton-drop subsets and expands the
remaining 19 terms' sparse supports by one or two extra nonzeros per factor.

Track 2 stops anchoring itself to AlphaTensor entirely. It screens many random
rank-19 sparse support patterns to map out which sparsity regimes give square
subsystems under the 200-variable cap with acceptable conditioning, then tracks
the best cases in Julia.

The goal is not only to hunt for a rank-19 endpoint, but also to characterize
the tractable sparse-support region well enough to guide the next campaign.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term, load_public_rank23_terms, matrix_multiplication_tensor
from src.ade3x3.steps.ade3x3_step81_homotopy_bridge_pilot import (
    SupportModel,
    build_support_model,
    coordinate_polynomial,
    evaluate_coordinates,
    full_coordinate_list,
    full_tensor_from_factors,
    julia_complex_literal,
    julia_executable,
    julia_float_literal,
    reduce_model_to_independent_variables,
    scalar_to_str,
    select_square_subsystem,
    target_values_for_coordinates,
    unpack_free_vector,
    witness_free_vector,
    write_csv,
    write_json,
    write_text,
)


EXPORTS = Path('outputs/exports')
STEP83_CONFIGS = EXPORTS / 'step83_rank19_sparse_case_configs.csv'
VARIABLE_CAP = int(os.environ.get('STEP83B_VARIABLE_CAP', '200'))
TRACK2_SCREEN_COUNT = int(os.environ.get('STEP83B_TRACK2_SCREEN_COUNT', '1000'))
TRACK2_TOPK = int(os.environ.get('STEP83B_TRACK2_TOPK', '24'))
JULIA_TIMEOUT_SECONDS = int(os.environ.get('STEP83B_JULIA_TIMEOUT_SECONDS', '120'))
CPU_COUNT = int(os.environ.get('STEP83B_CPU_COUNT', str(os.cpu_count() or 24)))
SCREEN_WORKERS = int(os.environ.get('STEP83B_SCREEN_WORKERS', str(max(1, math.ceil(CPU_COUNT * 1.5)))))
TRACK_WORKERS = int(os.environ.get('STEP83B_TRACK_WORKERS', str(max(1, math.ceil(CPU_COUNT * 1.5)))))
TRACK1_BASE_SEED = int(os.environ.get('STEP83B_TRACK1_SEED', '8301001'))
TRACK2_BASE_SEED = int(os.environ.get('STEP83B_TRACK2_SEED', '8302001'))
TRACK2_VIABLE_COND_CAP = float(os.environ.get('STEP83B_TRACK2_VIABLE_COND_CAP', '1e8'))
JULIA_PREFIX = 'STEP83B'


def set_single_thread_blas_env() -> None:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def term_flat_arrays(term: Term) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(term.alpha, dtype=np.float64).reshape(9).copy(),
        np.asarray(term.beta, dtype=np.float64).reshape(9).copy(),
        np.asarray(term.gamma, dtype=np.float64).reshape(9).copy(),
    )


def flat_to_matrix(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64).reshape(3, 3)


def support_indices(vector: np.ndarray) -> set[int]:
    return {int(index) for index in np.flatnonzero(np.abs(vector) > 0)}


def random_signed_values(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=count)
    magnitudes = rng.uniform(low, high, size=count)
    return signs * magnitudes


def choose_new_positions(
    existing_support: set[int],
    count: int,
    rng: np.random.Generator,
    preferred_positions: list[int] | None = None,
) -> list[int]:
    available = [index for index in range(9) if index not in existing_support]
    if count > len(available):
        raise RuntimeError(f'Requested {count} new positions from only {len(available)} available zeros.')
    preferred = []
    if preferred_positions:
        preferred = [index for index in preferred_positions if index in available]
    chosen: list[int] = []
    if preferred:
        take = min(count, len(preferred))
        chosen.extend(sorted(int(index) for index in rng.choice(np.array(preferred, dtype=np.int64), size=take, replace=False).tolist()))
    remaining = count - len(chosen)
    if remaining > 0:
        residue = [index for index in available if index not in chosen]
        chosen.extend(sorted(int(index) for index in rng.choice(np.array(residue, dtype=np.int64), size=remaining, replace=False).tolist()))
    return chosen


def build_output_fiber_preferences(dropped_terms: list[Term]) -> tuple[list[int], list[int], list[int]]:
    gamma_positions = sorted({int(index) for term in dropped_terms for index in np.flatnonzero(np.abs(np.asarray(term.gamma).reshape(9)) > 0)})
    output_cells = [(index // 3, index % 3) for index in gamma_positions]
    alpha_positions = sorted({3 * row_idx + shared_idx for row_idx, _ in output_cells for shared_idx in range(3)})
    beta_positions = sorted({3 * shared_idx + col_idx for _, col_idx in output_cells for shared_idx in range(3)})
    return alpha_positions, beta_positions, gamma_positions


def retained_terms_from_drop_ids(terms: list[Term], drop_ids: set[str]) -> list[Term]:
    return [term for term in terms if term.term_id not in drop_ids]


def expand_alpha_tensor_case(
    retained_terms: list[Term],
    dropped_terms: list[Term],
    expand_alpha: int,
    expand_beta: int,
    expand_gamma: int,
    targeted: bool,
    seed: int,
) -> list[Term]:
    rng = np.random.default_rng(seed)
    preferred_alpha, preferred_beta, preferred_gamma = build_output_fiber_preferences(dropped_terms)
    expanded_terms: list[Term] = []
    for term in retained_terms:
        alpha, beta, gamma = term_flat_arrays(term)
        alpha_support = support_indices(alpha)
        beta_support = support_indices(beta)
        gamma_support = support_indices(gamma)
        alpha_new = choose_new_positions(alpha_support, expand_alpha, rng, preferred_alpha if targeted else None)
        beta_new = choose_new_positions(beta_support, expand_beta, rng, preferred_beta if targeted else None)
        gamma_new = choose_new_positions(gamma_support, expand_gamma, rng, preferred_gamma if targeted else None) if expand_gamma else []

        for index, value in zip(alpha_new, random_signed_values(rng, len(alpha_new), 0.01, 0.10), strict=False):
            alpha[index] = value
        for index, value in zip(beta_new, random_signed_values(rng, len(beta_new), 0.01, 0.10), strict=False):
            beta[index] = value
        for index, value in zip(gamma_new, random_signed_values(rng, len(gamma_new), 0.01, 0.10), strict=False):
            gamma[index] = value

        expanded_terms.append(
            Term(
                term.term_id,
                'step83b_track1_support_expansion',
                flat_to_matrix(alpha),
                flat_to_matrix(beta),
                flat_to_matrix(gamma),
            )
        )
    return expanded_terms


def build_track1_case_specs(public_terms: list[Term]) -> list[dict]:
    config_rows = read_csv_rows(STEP83_CONFIGS)
    if len(config_rows) < 4:
        raise RuntimeError(f'Step 83 configs missing or too short at {STEP83_CONFIGS}.')
    top4_drop_sets = [tuple(row['dropped_terms'].split(';')) for row in config_rows[:4]]
    term_map = {term.term_id: term for term in public_terms}

    specs: list[dict] = []
    next_seed = TRACK1_BASE_SEED
    for case_index, drop_set in enumerate(top4_drop_sets, start=1):
        specs.append({
            'track': 'track1',
            'case_id': f'step83b_track1_case_{case_index:02d}',
            'drop_terms': drop_set,
            'expand_alpha': 1,
            'expand_beta': 1,
            'expand_gamma': 0,
            'targeted': False,
            'seed': next_seed,
            'case_family': 'uniform_expand_ab_1',
        })
        next_seed += 1
    for offset, drop_set in enumerate(top4_drop_sets, start=5):
        specs.append({
            'track': 'track1',
            'case_id': f'step83b_track1_case_{offset:02d}',
            'drop_terms': drop_set,
            'expand_alpha': 2,
            'expand_beta': 2,
            'expand_gamma': 0,
            'targeted': False,
            'seed': next_seed,
            'case_family': 'uniform_expand_ab_2',
        })
        next_seed += 1
    for offset, drop_set in enumerate(top4_drop_sets[:2], start=9):
        specs.append({
            'track': 'track1',
            'case_id': f'step83b_track1_case_{offset:02d}',
            'drop_terms': drop_set,
            'expand_alpha': 1,
            'expand_beta': 1,
            'expand_gamma': 1,
            'targeted': False,
            'seed': next_seed,
            'case_family': 'uniform_expand_abg_1',
        })
        next_seed += 1
    for offset, drop_set in enumerate(top4_drop_sets[:2], start=11):
        specs.append({
            'track': 'track1',
            'case_id': f'step83b_track1_case_{offset:02d}',
            'drop_terms': drop_set,
            'expand_alpha': 1,
            'expand_beta': 1,
            'expand_gamma': 0,
            'targeted': True,
            'seed': next_seed,
            'case_family': 'targeted_expand_ab_1',
        })
        next_seed += 1

    for spec in specs:
        drop_ids = set(spec['drop_terms'])
        retained_terms = retained_terms_from_drop_ids(public_terms, drop_ids)
        dropped_terms = [term_map[term_id] for term_id in spec['drop_terms']]
        spec['terms'] = expand_alpha_tensor_case(
            retained_terms,
            dropped_terms,
            expand_alpha=spec['expand_alpha'],
            expand_beta=spec['expand_beta'],
            expand_gamma=spec['expand_gamma'],
            targeted=bool(spec['targeted']),
            seed=int(spec['seed']),
        )
    return specs


def build_random_sparse_terms(seed: int, nnz_alpha: int, nnz_beta: int, nnz_gamma: int, rank_value: int = 19) -> list[Term]:
    rng = np.random.default_rng(seed)
    terms: list[Term] = []
    for term_index in range(rank_value):
        alpha = np.zeros(9, dtype=np.float64)
        beta = np.zeros(9, dtype=np.float64)
        gamma = np.zeros(9, dtype=np.float64)

        alpha_support = sorted(int(index) for index in rng.choice(np.arange(9), size=nnz_alpha, replace=False).tolist())
        beta_support = sorted(int(index) for index in rng.choice(np.arange(9), size=nnz_beta, replace=False).tolist())
        gamma_support = sorted(int(index) for index in rng.choice(np.arange(9), size=nnz_gamma, replace=False).tolist())

        alpha[alpha_support] = random_signed_values(rng, len(alpha_support), 0.5, 2.0)
        beta[beta_support] = random_signed_values(rng, len(beta_support), 0.5, 2.0)
        gamma[gamma_support] = random_signed_values(rng, len(gamma_support), 0.5, 2.0)

        terms.append(
            Term(
                f'r{term_index + 1:02d}',
                'step83b_track2_random_sparse',
                flat_to_matrix(alpha),
                flat_to_matrix(beta),
                flat_to_matrix(gamma),
            )
        )
    return terms


def screen_terms(case_id: str, track: str, terms: list[Term], target_tensor: np.ndarray, metadata: dict | None = None) -> dict:
    metadata = {} if metadata is None else dict(metadata)
    try:
        model = build_support_model(terms)
        initial_variable_count = len(model.variable_specs)
        witness_vector = witness_free_vector(model)
        model, independent_variable_count, dropped_variable_count = reduce_model_to_independent_variables(model, witness_vector)
        witness_vector = witness_free_vector(model)
        reduced_variable_count = len(model.variable_specs)
        if reduced_variable_count > VARIABLE_CAP:
            return {
                'track': track,
                'case_id': case_id,
                'status': 'variable_cap_exceeded',
                'initial_variable_count': initial_variable_count,
                'reduced_variable_count': reduced_variable_count,
                'selected_equation_count': '',
                'square_jacobian_condition_number': '',
                'start_full_tensor_max_abs_residual': '',
                'start_full_tensor_fro_residual': '',
                'independent_variable_rank': independent_variable_count,
                'dropped_variable_count': dropped_variable_count,
                'fixed_gauge_count': len(model.fixed_lookup),
                'error': f'reduced_variable_count={reduced_variable_count} exceeds cap {VARIABLE_CAP}',
                **metadata,
            }

        selected_rows, _square_jacobian, subsystem_condition = select_square_subsystem(model, witness_vector)
        coordinates = full_coordinate_list()
        selected_coordinates = [coordinates[row_index] for row_index in selected_rows]
        start_alpha, start_beta, start_gamma = unpack_free_vector(model, witness_vector)
        start_full_tensor = full_tensor_from_factors(start_alpha, start_beta, start_gamma)

        gamma_union = set()
        alpha_union = set()
        beta_union = set()
        unique_gamma_supports = set()
        for term in terms:
            alpha_flat, beta_flat, gamma_flat = term_flat_arrays(term)
            alpha_support = tuple(sorted(support_indices(alpha_flat)))
            beta_support = tuple(sorted(support_indices(beta_flat)))
            gamma_support = tuple(sorted(support_indices(gamma_flat)))
            alpha_union.update(alpha_support)
            beta_union.update(beta_support)
            gamma_union.update(gamma_support)
            unique_gamma_supports.add(gamma_support)

        return {
            'track': track,
            'case_id': case_id,
            'status': 'screened_ok',
            'terms': terms,
            'model': model,
            'witness_vector': witness_vector,
            'selected_coordinates': selected_coordinates,
            'start_parameters': evaluate_coordinates(start_alpha, start_beta, start_gamma, selected_coordinates),
            'initial_variable_count': initial_variable_count,
            'reduced_variable_count': reduced_variable_count,
            'selected_equation_count': len(selected_coordinates),
            'square_jacobian_condition_number': subsystem_condition,
            'start_full_tensor_max_abs_residual': float(np.max(np.abs(start_full_tensor - target_tensor))),
            'start_full_tensor_fro_residual': float(np.linalg.norm(start_full_tensor - target_tensor)),
            'independent_variable_rank': independent_variable_count,
            'dropped_variable_count': dropped_variable_count,
            'fixed_gauge_count': len(model.fixed_lookup),
            'gamma_union_size': len(gamma_union),
            'alpha_union_size': len(alpha_union),
            'beta_union_size': len(beta_union),
            'full_gamma_coverage': len(gamma_union) == 9,
            'unique_gamma_support_count': len(unique_gamma_supports),
            **metadata,
        }
    except Exception as exc:
        return {
            'track': track,
            'case_id': case_id,
            'status': 'screen_failed',
            'initial_variable_count': '',
            'reduced_variable_count': '',
            'selected_equation_count': '',
            'square_jacobian_condition_number': '',
            'start_full_tensor_max_abs_residual': '',
            'start_full_tensor_fro_residual': '',
            'error': str(exc),
            **metadata,
        }


TRACK2_SIGNATURES = [
    (3, 3, 3),
    (4, 4, 3),
    (4, 3, 3),
    (3, 4, 3),
    (4, 4, 4),
    (4, 4, 2),
]


def screen_random_pattern_worker(payload: dict) -> dict:
    set_single_thread_blas_env()
    target_tensor = matrix_multiplication_tensor(3).astype(np.float64)
    terms = build_random_sparse_terms(
        seed=int(payload['seed']),
        nnz_alpha=int(payload['nnz_alpha']),
        nnz_beta=int(payload['nnz_beta']),
        nnz_gamma=int(payload['nnz_gamma']),
        rank_value=19,
    )
    metadata = {
        'seed': int(payload['seed']),
        'nnz_alpha': int(payload['nnz_alpha']),
        'nnz_beta': int(payload['nnz_beta']),
        'nnz_gamma': int(payload['nnz_gamma']),
        'support_signature': f"({payload['nnz_alpha']},{payload['nnz_beta']},{payload['nnz_gamma']})",
    }
    screen = screen_terms(payload['case_id'], 'track2_screen', terms, target_tensor, metadata)
    if screen['status'] == 'screened_ok':
        cond = float(screen['square_jacobian_condition_number'])
        screen['viable'] = bool(screen['reduced_variable_count'] <= VARIABLE_CAP and cond < TRACK2_VIABLE_COND_CAP)
    else:
        screen['viable'] = False
    screen.pop('terms', None)
    screen.pop('model', None)
    screen.pop('witness_vector', None)
    screen.pop('selected_coordinates', None)
    screen.pop('start_parameters', None)
    return screen


def build_track2_payloads() -> list[dict]:
    payloads: list[dict] = []
    for index in range(TRACK2_SCREEN_COUNT):
        nnz_alpha, nnz_beta, nnz_gamma = TRACK2_SIGNATURES[index % len(TRACK2_SIGNATURES)]
        payloads.append(
            {
                'case_id': f'step83b_track2_screen_{index + 1:04d}',
                'seed': TRACK2_BASE_SEED + index,
                'nnz_alpha': nnz_alpha,
                'nnz_beta': nnz_beta,
                'nnz_gamma': nnz_gamma,
            }
        )
    return payloads


def build_julia_script_text(
    model: SupportModel,
    selected_coordinates: list[tuple[int, int, int]],
    start_vector: np.ndarray,
    start_parameters: np.ndarray,
    target_parameters: np.ndarray,
) -> str:
    equation_lines = [f'    {coordinate_polynomial(model, coordinate, idx)},' for idx, coordinate in enumerate(selected_coordinates)]
    lines: list[str] = []
    write_line = lines.append
    write_line('using HomotopyContinuation')
    write_line('')
    write_line(f'@var x[1:{len(model.variable_specs)}] p[1:{len(selected_coordinates)}]')
    write_line('')
    write_line('equations = [')
    lines.extend(equation_lines)
    write_line(']')
    write_line('F = System(equations; variables=x, parameters=p)')
    write_line('')
    write_line('start_solution = [')
    for value in start_vector:
        write_line(f'    {julia_complex_literal(float(value))},')
    write_line(']')
    write_line('start_parameters = [')
    for value in start_parameters:
        write_line(f'    {julia_float_literal(float(value))},')
    write_line(']')
    write_line('target_parameters = [')
    for value in target_parameters:
        write_line(f'    {julia_float_literal(float(value))},')
    write_line(']')
    write_line('')
    write_line('result = solve(F, [start_solution]; start_parameters=start_parameters, target_parameters=target_parameters, show_progress=false, threading=false)')
    write_line('solutions_found = solutions(result)')
    write_line(f'println("{JULIA_PREFIX}_NUM_SOLUTIONS=" * string(length(solutions_found)))')
    write_line('if !isempty(solutions_found)')
    write_line('    endpoint = solutions_found[1]')
    write_line(f'    println("{JULIA_PREFIX}_ENDPOINT_REAL=" * join(string.(real.(endpoint)), ","))')
    write_line(f'    println("{JULIA_PREFIX}_ENDPOINT_IMAG=" * join(string.(imag.(endpoint)), ","))')
    write_line(f'    println("{JULIA_PREFIX}_MAX_IMAG=" * string(maximum(abs.(imag.(endpoint)))))')
    write_line('end')
    return '\n'.join(lines) + '\n'


def parse_prefixed_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith(f'{JULIA_PREFIX}_') and '=' in line:
            key, value = line.split('=', 1)
            parsed[key.strip()] = value.strip()
    return parsed


def endpoint_vector_from_output(parsed: dict[str, str]) -> np.ndarray:
    values = [float(token) for token in parsed.get(f'{JULIA_PREFIX}_ENDPOINT_REAL', '').split(',') if token.strip()]
    if not values:
        raise RuntimeError('Julia output did not include an endpoint vector.')
    imag_values = [float(token) for token in parsed.get(f'{JULIA_PREFIX}_ENDPOINT_IMAG', '').split(',') if token.strip()]
    if imag_values and max(abs(value) for value in imag_values) > 1e-7:
        raise RuntimeError(f'Endpoint has large imaginary part: {max(abs(value) for value in imag_values)}')
    return np.array(values, dtype=np.float64)


def run_prepared_case(prepared: dict, target_tensor: np.ndarray, julia_path: Path) -> dict:
    case_start = time.time()
    case_id = str(prepared['case_id'])
    model: SupportModel = prepared['model']
    witness_vector: np.ndarray = prepared['witness_vector']
    selected_coordinates: list[tuple[int, int, int]] = prepared['selected_coordinates']
    start_parameters: np.ndarray = prepared['start_parameters']
    target_parameters = target_values_for_coordinates(target_tensor, selected_coordinates)

    script_path = EXPORTS / f'{case_id}.jl'
    stdout_path = EXPORTS / f'{case_id}_stdout.txt'
    write_text(
        script_path,
        build_julia_script_text(model, selected_coordinates, witness_vector, start_parameters, target_parameters),
    )

    env = os.environ.copy()
    env['JULIA_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    try:
        process = subprocess.run(
            [str(julia_path), str(script_path)],
            capture_output=True,
            text=True,
            timeout=JULIA_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        write_text(
            stdout_path,
            f'Timeout after {JULIA_TIMEOUT_SECONDS} seconds while tracking {case_id}.\n'
            f'Condition number: {prepared["square_jacobian_condition_number"]:.16e}\n',
        )
        return {
            'case_id': case_id,
            'track': prepared['track'],
            'status': 'timeout',
            'runtime_seconds': scalar_to_str(time.time() - case_start),
            'initial_variable_count': prepared['initial_variable_count'],
            'reduced_variable_count': prepared['reduced_variable_count'],
            'selected_equation_count': prepared['selected_equation_count'],
            'square_jacobian_condition_number': scalar_to_str(prepared['square_jacobian_condition_number']),
            'start_full_tensor_max_abs_residual': scalar_to_str(prepared['start_full_tensor_max_abs_residual']),
            'stdout_path': str(stdout_path),
            'provenance': 'MEASURED_FROM_CODE',
        }

    write_text(stdout_path, process.stdout + ('\nSTDERR\n' + process.stderr if process.stderr else ''))
    row = {
        'case_id': case_id,
        'track': prepared['track'],
        'status': 'julia_error' if process.returncode != 0 else 'julia_ok',
        'runtime_seconds': scalar_to_str(time.time() - case_start),
        'initial_variable_count': prepared['initial_variable_count'],
        'reduced_variable_count': prepared['reduced_variable_count'],
        'selected_equation_count': prepared['selected_equation_count'],
        'square_jacobian_condition_number': scalar_to_str(prepared['square_jacobian_condition_number']),
        'start_full_tensor_max_abs_residual': scalar_to_str(prepared['start_full_tensor_max_abs_residual']),
        'stdout_path': str(stdout_path),
        'julia_return_code': process.returncode,
        'provenance': 'MEASURED_FROM_CODE',
    }
    if process.returncode != 0:
        return row

    parsed = parse_prefixed_output(process.stdout)
    row['num_solutions_returned'] = int(parsed.get(f'{JULIA_PREFIX}_NUM_SOLUTIONS', '0'))
    row['endpoint_max_imag'] = scalar_to_str(float(parsed.get(f'{JULIA_PREFIX}_MAX_IMAG', '0.0')))
    if int(row['num_solutions_returned']) == 0:
        row['status'] = 'no_solution_returned'
        return row

    endpoint_vector = endpoint_vector_from_output(parsed)
    endpoint_alpha, endpoint_beta, endpoint_gamma = unpack_free_vector(model, endpoint_vector)
    endpoint_selected_values = evaluate_coordinates(endpoint_alpha, endpoint_beta, endpoint_gamma, selected_coordinates)
    endpoint_full_tensor = full_tensor_from_factors(endpoint_alpha, endpoint_beta, endpoint_gamma)
    full_residual = float(np.max(np.abs(endpoint_full_tensor - target_tensor)))

    row['endpoint_selected_residual_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_selected_values - target_parameters))))
    row['endpoint_full_tensor_max_abs_residual'] = scalar_to_str(full_residual)
    row['endpoint_to_start_chart_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_vector - witness_vector))))
    row['status'] = 'complex_endpoint' if float(parsed.get(f'{JULIA_PREFIX}_MAX_IMAG', '0.0')) > 1e-7 else 'real_endpoint'
    row['exact_hit'] = bool(row['status'] == 'real_endpoint' and full_residual < 1e-10)
    return row


def build_track2_terms_from_row(row: dict) -> list[Term]:
    return build_random_sparse_terms(
        seed=int(row['seed']),
        nnz_alpha=int(row['nnz_alpha']),
        nnz_beta=int(row['nnz_beta']),
        nnz_gamma=int(row['nnz_gamma']),
        rank_value=19,
    )


def compute_meta_analysis(screening_rows: list[dict], tracked_rows: list[dict]) -> dict:
    viable_rows = [row for row in screening_rows if row.get('status') == 'screened_ok' and str(row.get('viable', '')).lower() == 'true']
    viable_conditions = [float(row['square_jacobian_condition_number']) for row in viable_rows]
    viable_var_counts = [int(row['reduced_variable_count']) for row in viable_rows]
    signature_stats: dict[str, dict] = {}
    by_signature: dict[str, list[dict]] = defaultdict(list)
    for row in screening_rows:
        by_signature[str(row['support_signature'])].append(row)
    tracked_by_case = {row['case_id']: row for row in tracked_rows}
    for signature, rows in sorted(by_signature.items()):
        sig_viable = [row for row in rows if row.get('status') == 'screened_ok' and str(row.get('viable', '')).lower() == 'true']
        conds = [float(row['square_jacobian_condition_number']) for row in sig_viable]
        tracked = [tracked_by_case[row['case_id']] for row in rows if row['case_id'] in tracked_by_case]
        real = [row for row in tracked if row.get('status') == 'real_endpoint']
        signature_stats[signature] = {
            'screened_count': len(rows),
            'viable_count': len(sig_viable),
            'viable_fraction': 0.0 if not rows else len(sig_viable) / len(rows),
            'median_condition': None if not conds else float(np.median(conds)),
            'best_condition': None if not conds else float(min(conds)),
            'median_reduced_variable_count': None if not sig_viable else float(np.median([int(row['reduced_variable_count']) for row in sig_viable])),
            'tracked_count': len(tracked),
            'real_endpoint_count': len(real),
        }
    optimal_signature = None
    if signature_stats:
        optimal_signature = min(
            signature_stats.items(),
            key=lambda item: (
                -item[1]['viable_fraction'],
                float('inf') if item[1]['median_condition'] is None else item[1]['median_condition'],
                float('inf') if item[1]['median_reduced_variable_count'] is None else item[1]['median_reduced_variable_count'],
            ),
        )[0]

    corr_var_logcond = None
    corr_gamma_logcond = None
    if len(viable_rows) >= 2:
        log_conditions = np.log10(np.array(viable_conditions, dtype=np.float64))
        corr_var_logcond = float(np.corrcoef(np.array(viable_var_counts, dtype=np.float64), log_conditions)[0, 1])
        corr_gamma_logcond = float(np.corrcoef(np.array([int(row['gamma_union_size']) for row in viable_rows], dtype=np.float64), log_conditions)[0, 1])

    return {
        'screened_count': len(screening_rows),
        'viable_count': len(viable_rows),
        'median_condition': None if not viable_conditions else float(np.median(viable_conditions)),
        'best_condition': None if not viable_conditions else float(min(viable_conditions)),
        'median_reduced_variable_count': None if not viable_var_counts else float(np.median(viable_var_counts)),
        'optimal_support_signature': optimal_signature,
        'signature_stats': signature_stats,
        'correlation_reduced_vars_vs_log10_condition': corr_var_logcond,
        'correlation_gamma_union_vs_log10_condition': corr_gamma_logcond,
    }


def main() -> None:
    set_single_thread_blas_env()
    start_time = time.time()
    julia_path = julia_executable()
    target_tensor = matrix_multiplication_tensor(3).astype(np.float64)
    public_terms, orientation, reconstruction_max_abs = load_public_rank23_terms()

    print(f'Step 83b using screen_workers={SCREEN_WORKERS}, track_workers={TRACK_WORKERS}, variable_cap={VARIABLE_CAP}', flush=True)

    track2_payloads = build_track2_payloads()
    screening_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=SCREEN_WORKERS) as executor:
        futures = [executor.submit(screen_random_pattern_worker, payload) for payload in track2_payloads]
        for future in as_completed(futures):
            screening_rows.append(future.result())
    screening_rows.sort(key=lambda row: row['case_id'])

    viable_screening_rows = [
        row for row in screening_rows
        if row.get('status') == 'screened_ok' and str(row.get('viable', '')).lower() == 'true'
    ]
    viable_screening_rows.sort(key=lambda row: (float(row['square_jacobian_condition_number']), int(row['reduced_variable_count']), row['case_id']))
    track2_top_rows = viable_screening_rows[:TRACK2_TOPK]

    track1_specs = build_track1_case_specs(public_terms)
    track1_screened: list[dict] = []
    prepared_cases: list[dict] = []

    for spec in track1_specs:
        metadata = {
            'case_family': spec['case_family'],
            'drop_terms': ';'.join(spec['drop_terms']),
            'expand_alpha': spec['expand_alpha'],
            'expand_beta': spec['expand_beta'],
            'expand_gamma': spec['expand_gamma'],
            'targeted': bool(spec['targeted']),
            'seed': spec['seed'],
        }
        screened = screen_terms(spec['case_id'], 'track1', spec['terms'], target_tensor, metadata)
        track1_screened.append(screened)
        if screened['status'] == 'screened_ok':
            prepared_cases.append(screened)

    track1_config_rows = []
    for screen in track1_screened:
        track1_config_rows.append(
            {
                'case_id': screen['case_id'],
                'case_family': screen.get('case_family', ''),
                'drop_terms': screen.get('drop_terms', ''),
                'expand_alpha': screen.get('expand_alpha', ''),
                'expand_beta': screen.get('expand_beta', ''),
                'expand_gamma': screen.get('expand_gamma', ''),
                'targeted': screen.get('targeted', ''),
                'seed': screen.get('seed', ''),
                'status': screen['status'],
                'initial_variable_count': screen.get('initial_variable_count', ''),
                'reduced_variable_count': screen.get('reduced_variable_count', ''),
                'selected_equation_count': screen.get('selected_equation_count', ''),
                'square_jacobian_condition_number': '' if screen.get('square_jacobian_condition_number', '') == '' else scalar_to_str(screen['square_jacobian_condition_number']),
                'start_full_tensor_max_abs_residual': '' if screen.get('start_full_tensor_max_abs_residual', '') == '' else scalar_to_str(screen['start_full_tensor_max_abs_residual']),
                'start_full_tensor_fro_residual': '' if screen.get('start_full_tensor_fro_residual', '') == '' else scalar_to_str(screen['start_full_tensor_fro_residual']),
                'error': screen.get('error', ''),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )

    track2_prepared: list[dict] = []
    for row in track2_top_rows:
        terms = build_track2_terms_from_row(row)
        prepared = screen_terms(
            case_id=row['case_id'],
            track='track2',
            terms=terms,
            target_tensor=target_tensor,
            metadata={
                'seed': int(row['seed']),
                'nnz_alpha': int(row['nnz_alpha']),
                'nnz_beta': int(row['nnz_beta']),
                'nnz_gamma': int(row['nnz_gamma']),
                'support_signature': row['support_signature'],
            },
        )
        if prepared['status'] == 'screened_ok':
            track2_prepared.append(prepared)

    tracking_queue = prepared_cases + track2_prepared
    tracked_results: list[dict] = []
    exact_hit_row = None
    with ThreadPoolExecutor(max_workers=TRACK_WORKERS) as executor:
        future_map = {executor.submit(run_prepared_case, prepared, target_tensor, julia_path): prepared['case_id'] for prepared in tracking_queue}
        for future in as_completed(future_map):
            row = future.result()
            tracked_results.append(row)
            if row.get('status') == 'real_endpoint' and float(row.get('endpoint_full_tensor_max_abs_residual', '1.0')) < 1e-10:
                exact_hit_row = row
                break
        if exact_hit_row is not None:
            for future in future_map:
                future.cancel()

    track1_results = [row for row in tracked_results if row['track'] == 'track1']
    track2_results = [row for row in tracked_results if row['track'] == 'track2']

    meta_analysis = compute_meta_analysis(screening_rows, track2_results)
    track1_real = [row for row in track1_results if row.get('status') == 'real_endpoint']
    track2_real = [row for row in track2_results if row.get('status') == 'real_endpoint']
    track1_best = min(track1_real, key=lambda row: float(row['endpoint_full_tensor_max_abs_residual'])) if track1_real else None
    track2_best = min(track2_real, key=lambda row: float(row['endpoint_full_tensor_max_abs_residual'])) if track2_real else None

    write_csv(
        EXPORTS / 'step83b_track1_configs.csv',
        track1_config_rows,
        [
            'case_id',
            'case_family',
            'drop_terms',
            'expand_alpha',
            'expand_beta',
            'expand_gamma',
            'targeted',
            'seed',
            'status',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_full_tensor_max_abs_residual',
            'start_full_tensor_fro_residual',
            'error',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step83b_track1_results.csv',
        track1_results,
        [
            'case_id',
            'track',
            'status',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_full_tensor_max_abs_residual',
            'num_solutions_returned',
            'endpoint_max_imag',
            'endpoint_selected_residual_max_abs',
            'endpoint_full_tensor_max_abs_residual',
            'endpoint_to_start_chart_max_abs',
            'exact_hit',
            'julia_return_code',
            'stdout_path',
            'runtime_seconds',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step83b_track2_screening.csv',
        screening_rows,
        [
            'case_id',
            'track',
            'seed',
            'support_signature',
            'nnz_alpha',
            'nnz_beta',
            'nnz_gamma',
            'status',
            'viable',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_full_tensor_max_abs_residual',
            'start_full_tensor_fro_residual',
            'independent_variable_rank',
            'dropped_variable_count',
            'fixed_gauge_count',
            'gamma_union_size',
            'alpha_union_size',
            'beta_union_size',
            'full_gamma_coverage',
            'unique_gamma_support_count',
            'error',
        ],
    )
    write_csv(
        EXPORTS / 'step83b_track2_tracked.csv',
        track2_results,
        [
            'case_id',
            'track',
            'status',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_full_tensor_max_abs_residual',
            'num_solutions_returned',
            'endpoint_max_imag',
            'endpoint_selected_residual_max_abs',
            'endpoint_full_tensor_max_abs_residual',
            'endpoint_to_start_chart_max_abs',
            'exact_hit',
            'julia_return_code',
            'stdout_path',
            'runtime_seconds',
            'provenance',
        ],
    )
    write_json(EXPORTS / 'step83b_meta_analysis.json', meta_analysis)

    summary = {
        'track1_cases': len(track1_config_rows),
        'track1_completed': len(track1_results),
        'track1_real_endpoints': len(track1_real),
        'track1_best_residual': None if track1_best is None else float(track1_best['endpoint_full_tensor_max_abs_residual']),
        'track2_screened': len(screening_rows),
        'track2_viable': len(viable_screening_rows),
        'track2_tracked': len(track2_results),
        'track2_real_endpoints': len(track2_real),
        'track2_best_residual': None if track2_best is None else float(track2_best['endpoint_full_tensor_max_abs_residual']),
        'track2_median_condition': meta_analysis['median_condition'],
        'track2_best_condition': meta_analysis['best_condition'],
        'track2_optimal_nnz': meta_analysis['optimal_support_signature'],
        'any_exact_hit': bool(exact_hit_row is not None),
        'step83b_orientation': orientation,
        'step83b_public_reconstruction_max_abs': reconstruction_max_abs,
        'screen_workers': SCREEN_WORKERS,
        'track_workers': TRACK_WORKERS,
        'variable_cap': VARIABLE_CAP,
        'runtime_seconds': float(time.time() - start_time),
        'provenance': 'MEASURED_FROM_CODE',
    }
    write_json(EXPORTS / 'step83b_summary.json', summary)

    print('=== Step 83b: Support Expansion + Heuristic Sparse Campaign ===', flush=True)
    print(f"Track 1 completed: {summary['track1_completed']} / {summary['track1_cases']}", flush=True)
    print(f"Track 2 screened: {summary['track2_screened']}, viable: {summary['track2_viable']}, tracked: {summary['track2_tracked']}", flush=True)
    print(f"Track 2 optimal nnz signature: {summary['track2_optimal_nnz']}", flush=True)
    print(f"Any exact hit: {summary['any_exact_hit']}", flush=True)


if __name__ == '__main__':
    main()