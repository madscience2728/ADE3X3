"""
ade3x3_step83_rank19_sparse_rank23_chart_homotopy.py

Step 83: Sparse rank-19 charts inherited from the public rank-23 witness.

Step 82 showed that the exact rank-19 anticommutator witness is too dense for a
practical direct continuation chart: even after gauge fixing it leaves 475 free
coordinates with square-Jacobian condition numbers around 1e11. Step 83 changes
the chart, not the backend.

The experiment here is conservative and intentionally low-variable:
1. Start from the exact public rank-23 decomposition used in Step 81.
2. Build rank-19 support templates by dropping 4 terms from the 9 singleton-
   gamma terms. This keeps the remaining 19 terms as sparse as possible.
3. For each candidate 19-term support chart, freeze the exact retained witness
   values as the start point and ask whether that sparse chart can absorb the
   dropped residual under a parameter homotopy from the retained-term partial
   tensor to the full 3x3 multiplication tensor.
4. Screen all candidates cheaply via the Step 81 Jacobian reduction / square-
   subsystem selection.
5. Track only the best-conditioned surviving cases in Julia.

This does not prove anything about rank 19 globally. It is a targeted search for
small, structurally plausible rank-19 charts that inherit real cancellation
structure from the known exact rank-23 witness.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import time
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
    parse_julia_output,
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
SCREEN_LIMIT = int(os.environ.get('STEP83_SCREEN_LIMIT', '24'))
TRACK_CASES = int(os.environ.get('STEP83_TRACK_CASES', '4'))
JULIA_TIMEOUT_SECONDS = int(os.environ.get('STEP83_JULIA_TIMEOUT_SECONDS', '180'))


def gamma_support_size(term: Term) -> int:
    return int(np.count_nonzero(np.asarray(term.gamma)))


def retained_terms_from_drop_set(terms: list[Term], drop_ids: set[str]) -> list[Term]:
    return [term for term in terms if term.term_id not in drop_ids]


def candidate_drop_sets(terms: list[Term]) -> list[tuple[str, ...]]:
    singleton_terms = sorted(term.term_id for term in terms if gamma_support_size(term) == 1)
    if len(singleton_terms) < 4:
        raise RuntimeError('Need at least four singleton-gamma terms to build Step 83 cases.')
    return list(itertools.combinations(singleton_terms, 4))


def julia_script_text_step83(
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
    write_line('println("STEP83_NUM_SOLUTIONS=" * string(length(solutions_found)))')
    write_line('if !isempty(solutions_found)')
    write_line('    endpoint = solutions_found[1]')
    write_line('    println("STEP83_ENDPOINT_REAL=" * join(string.(real.(endpoint)), ","))')
    write_line('    println("STEP83_ENDPOINT_IMAG=" * join(string.(imag.(endpoint)), ","))')
    write_line('    println("STEP83_MAX_IMAG=" * string(maximum(abs.(imag.(endpoint)))))')
    write_line('end')
    return '\n'.join(lines) + '\n'


def parse_step83_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith('STEP83_') and '=' in line:
            key, value = line.split('=', 1)
            parsed[key.strip()] = value.strip()
    return parsed


def endpoint_vector_from_step83(parsed: dict[str, str]) -> np.ndarray:
    values = [float(token) for token in parsed.get('STEP83_ENDPOINT_REAL', '').split(',') if token.strip()]
    if not values:
        raise RuntimeError('Julia output did not include a Step 83 endpoint vector.')
    imag_values = [float(token) for token in parsed.get('STEP83_ENDPOINT_IMAG', '').split(',') if token.strip()]
    if imag_values and max(abs(value) for value in imag_values) > 1e-7:
        raise RuntimeError(f'Step 83 endpoint has large imaginary part: {max(abs(value) for value in imag_values)}')
    return np.array(values, dtype=np.float64)


def screen_case(case_id: str, retained_terms: list[Term], target_tensor: np.ndarray) -> dict:
    try:
        model = build_support_model(retained_terms)
        initial_variable_count = len(model.variable_specs)
        witness_vector = witness_free_vector(model)
        model, independent_variable_count, dropped_variable_count = reduce_model_to_independent_variables(model, witness_vector)
        witness_vector = witness_free_vector(model)
        selected_rows, _square_jacobian, subsystem_condition = select_square_subsystem(model, witness_vector)
        coordinates = full_coordinate_list()
        selected_coordinates = [coordinates[row_index] for row_index in selected_rows]

        start_alpha, start_beta, start_gamma = unpack_free_vector(model, witness_vector)
        start_parameters = evaluate_coordinates(start_alpha, start_beta, start_gamma, selected_coordinates)
        start_full_tensor = full_tensor_from_factors(start_alpha, start_beta, start_gamma)

        return {
            'case_id': case_id,
            'status': 'screened_ok',
            'model': model,
            'witness_vector': witness_vector,
            'selected_coordinates': selected_coordinates,
            'start_parameters': start_parameters,
            'initial_variable_count': initial_variable_count,
            'reduced_variable_count': len(model.variable_specs),
            'dropped_variable_count': dropped_variable_count,
            'independent_variable_rank': independent_variable_count,
            'fixed_gauge_count': len(model.fixed_lookup),
            'selected_equation_count': len(selected_coordinates),
            'square_jacobian_condition_number': subsystem_condition,
            'start_full_tensor_max_abs_residual': float(np.max(np.abs(start_full_tensor - target_tensor))),
            'start_full_tensor_fro_residual': float(np.linalg.norm(start_full_tensor - target_tensor)),
        }
    except Exception as exc:
        return {
            'case_id': case_id,
            'status': 'screen_failed',
            'error': str(exc),
        }


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
        julia_script_text_step83(model, selected_coordinates, witness_vector, start_parameters, target_parameters),
    )

    try:
        process = subprocess.run(
            [str(julia_path), str(script_path)],
            capture_output=True,
            text=True,
            timeout=JULIA_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        write_text(
            stdout_path,
            f'Timeout after {JULIA_TIMEOUT_SECONDS} seconds while tracking {case_id}.\n'
            f'Condition number: {prepared["square_jacobian_condition_number"]:.16e}\n',
        )
        return {
            'case_id': case_id,
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

    parsed = parse_step83_output(process.stdout)
    row['num_solutions_returned'] = int(parsed.get('STEP83_NUM_SOLUTIONS', '0'))
    row['endpoint_max_imag'] = scalar_to_str(float(parsed.get('STEP83_MAX_IMAG', '0.0')))
    if int(row['num_solutions_returned']) == 0:
        row['status'] = 'no_solution_returned'
        return row

    endpoint_vector = endpoint_vector_from_step83(parsed)
    endpoint_alpha, endpoint_beta, endpoint_gamma = unpack_free_vector(model, endpoint_vector)
    endpoint_selected_values = evaluate_coordinates(endpoint_alpha, endpoint_beta, endpoint_gamma, selected_coordinates)
    endpoint_full_tensor = full_tensor_from_factors(endpoint_alpha, endpoint_beta, endpoint_gamma)

    row['endpoint_selected_residual_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_selected_values - target_parameters))))
    row['endpoint_full_tensor_max_abs_residual'] = scalar_to_str(float(np.max(np.abs(endpoint_full_tensor - target_tensor))))
    row['endpoint_to_start_chart_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_vector - witness_vector))))
    row['status'] = 'complex_endpoint' if float(parsed.get('STEP83_MAX_IMAG', '0.0')) > 1e-7 else 'real_endpoint'
    return row


def main() -> None:
    start_time = time.time()
    julia_path = julia_executable()
    full_tensor = matrix_multiplication_tensor(3).astype(np.float64)
    terms, orientation, reconstruction_max_abs = load_public_rank23_terms()

    all_drop_sets = candidate_drop_sets(terms)
    screened_rows: list[dict] = []
    prepared_cases: list[dict] = []

    for drop_set in all_drop_sets:
        drop_ids = set(drop_set)
        retained_terms = retained_terms_from_drop_set(terms, drop_ids)
        case_id = 'step83_' + '_'.join(drop_set)
        screened = screen_case(case_id, retained_terms, full_tensor)
        screened_rows.append(
            {
                'case_id': case_id,
                'dropped_terms': ';'.join(drop_set),
                'retained_term_count': len(retained_terms),
                'status': screened['status'],
                'initial_variable_count': screened.get('initial_variable_count', ''),
                'reduced_variable_count': screened.get('reduced_variable_count', ''),
                'selected_equation_count': screened.get('selected_equation_count', ''),
                'square_jacobian_condition_number': '' if 'square_jacobian_condition_number' not in screened else scalar_to_str(screened['square_jacobian_condition_number']),
                'start_full_tensor_max_abs_residual': '' if 'start_full_tensor_max_abs_residual' not in screened else scalar_to_str(screened['start_full_tensor_max_abs_residual']),
                'start_full_tensor_fro_residual': '' if 'start_full_tensor_fro_residual' not in screened else scalar_to_str(screened['start_full_tensor_fro_residual']),
                'error': screened.get('error', ''),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )
        if screened['status'] == 'screened_ok':
            screened['dropped_terms'] = tuple(drop_set)
            prepared_cases.append(screened)

    prepared_cases.sort(
        key=lambda row: (
            float(row['square_jacobian_condition_number']),
            int(row['reduced_variable_count']),
            float(row['start_full_tensor_max_abs_residual']),
        )
    )
    case_configs = prepared_cases[:SCREEN_LIMIT]

    config_rows = [
        {
            'config_rank': rank,
            'case_id': prepared['case_id'],
            'dropped_terms': ';'.join(prepared['dropped_terms']),
            'initial_variable_count': prepared['initial_variable_count'],
            'reduced_variable_count': prepared['reduced_variable_count'],
            'selected_equation_count': prepared['selected_equation_count'],
            'square_jacobian_condition_number': scalar_to_str(prepared['square_jacobian_condition_number']),
            'start_full_tensor_max_abs_residual': scalar_to_str(prepared['start_full_tensor_max_abs_residual']),
            'start_full_tensor_fro_residual': scalar_to_str(prepared['start_full_tensor_fro_residual']),
            'provenance': 'MEASURED_FROM_CODE',
        }
        for rank, prepared in enumerate(case_configs, start=1)
    ]

    tracked_cases = case_configs[:TRACK_CASES]
    case_rows: list[dict] = []
    for case_index, prepared in enumerate(tracked_cases, start=1):
        print(
            f"Running Step 83 track {case_index}/{len(tracked_cases)} on {prepared['case_id']} with cond={prepared['square_jacobian_condition_number']:.3e}...",
            flush=True,
        )
        case_rows.append(run_prepared_case(prepared, full_tensor, julia_path))

    real_rows = [row for row in case_rows if row.get('status') == 'real_endpoint']
    best_row = min(real_rows, key=lambda row: float(row['endpoint_full_tensor_max_abs_residual'])) if real_rows else None

    summary = {
        'step83_orientation': orientation,
        'step83_public_reconstruction_max_abs': str(reconstruction_max_abs),
        'step83_singleton_term_count': sum(1 for term in terms if gamma_support_size(term) == 1),
        'step83_total_candidate_drop_sets': len(all_drop_sets),
        'step83_screened_ok_count': sum(1 for row in screened_rows if row['status'] == 'screened_ok'),
        'step83_screen_failed_count': sum(1 for row in screened_rows if row['status'] != 'screened_ok'),
        'step83_case_config_count': len(case_configs),
        'step83_tracked_case_count': len(case_rows),
        'step83_real_endpoint_count': len(real_rows),
        'step83_timeout_count': sum(1 for row in case_rows if row.get('status') == 'timeout'),
        'step83_no_solution_count': sum(1 for row in case_rows if row.get('status') == 'no_solution_returned'),
        'step83_complex_endpoint_count': sum(1 for row in case_rows if row.get('status') == 'complex_endpoint'),
        'step83_julia_error_count': sum(1 for row in case_rows if row.get('status') == 'julia_error'),
        'step83_best_full_tensor_max_abs_residual': '' if best_row is None else best_row['endpoint_full_tensor_max_abs_residual'],
        'step83_best_selected_residual_max_abs': '' if best_row is None else best_row['endpoint_selected_residual_max_abs'],
        'step83_best_case_id': '' if best_row is None else best_row['case_id'],
        'step83_runtime_seconds': scalar_to_str(time.time() - start_time),
        'provenance': 'MEASURED_FROM_CODE',
    }

    write_csv(
        EXPORTS / 'step83_rank19_sparse_screening.csv',
        screened_rows,
        [
            'case_id',
            'dropped_terms',
            'retained_term_count',
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
        EXPORTS / 'step83_rank19_sparse_case_configs.csv',
        config_rows,
        [
            'config_rank',
            'case_id',
            'dropped_terms',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_full_tensor_max_abs_residual',
            'start_full_tensor_fro_residual',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step83_rank19_sparse_cases.csv',
        case_rows,
        [
            'case_id',
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
            'julia_return_code',
            'stdout_path',
            'runtime_seconds',
            'provenance',
        ],
    )
    write_json(EXPORTS / 'step83_rank19_sparse_summary.json', summary)

    print('=== Step 83: Sparse rank-19 charts from the public rank-23 witness ===', flush=True)
    print(f"Singleton-gamma terms: {summary['step83_singleton_term_count']}", flush=True)
    print(f"Total candidate drop sets: {summary['step83_total_candidate_drop_sets']}", flush=True)
    print(f"Screened OK cases: {summary['step83_screened_ok_count']}", flush=True)
    print(f"Tracked cases: {summary['step83_tracked_case_count']}", flush=True)
    print(f"Real endpoints: {summary['step83_real_endpoint_count']}", flush=True)
    print(f"Best full tensor residual: {summary['step83_best_full_tensor_max_abs_residual'] or 'none'}", flush=True)


if __name__ == '__main__':
    main()