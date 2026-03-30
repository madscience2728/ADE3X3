"""
ade3x3_step82_rank19_anticommutator_homotopy.py

Step 82: Direct rank-19 anticommutator-to-full homotopy.

This step follows the exact rank-19 anticommutator witness from Step 75 under

    T(t) = (1 - t) * T_anti + t * T_full.

The witness factors are first reconstructed from the recorded best seed rather
than from the rounded CSV. Because the Step 75 rank-19 witness is dense, the raw
gauge-fixed chart still has 475 coordinates. The practical strategy here is:

1. balance each term to improve coordinate scaling,
2. gauge-fix using the largest-magnitude alpha/beta entries per term,
3. screen many gauge-equivalent charts cheaply by Jacobian conditioning,
4. track only the best-conditioned charts in Julia, and
5. record whether any real endpoint on the selected subsystem also solves the
   full 729-entry tensor.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term, matrix_multiplication_tensor
from src.ade3x3.steps.ade3x3_step74_commutator_anticommutator_rank_scan import build_tensor_specs, factors_to_terms
from src.ade3x3.steps.ade3x3_step75_anticommutator_rank19_extraction_hamilton_split import read_csv, scan_explicit_seeds
from src.ade3x3.steps.ade3x3_step81_homotopy_bridge_pilot import (
    SupportModel,
    VariableSpec,
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
STEP75_VERIFICATION = EXPORTS / 'step75_anticommutator_rank19_verification.csv'
CHART_CANDIDATES = int(os.environ.get('STEP82_CHART_CANDIDATES', '24'))
TRACK_CASES = int(os.environ.get('STEP82_TRACK_CASES', '4'))
GAUGE_LOG_SCALE = float(os.environ.get('STEP82_GAUGE_LOG_SCALE', '0.35'))
CASE_SEED = int(os.environ.get('STEP82_SEED', '820019'))
JULIA_TIMEOUT_SECONDS = int(os.environ.get('STEP82_JULIA_TIMEOUT_SECONDS', '180'))


def load_exact_step75_anticommutator_terms() -> tuple[list[Term], float, int]:
    verification_rows = read_csv(STEP75_VERIFICATION)
    if not verification_rows:
        raise RuntimeError(f'Missing Step 75 verification CSV at {STEP75_VERIFICATION}.')
    best_seed = int(verification_rows[0]['best_seed'])
    specs = {spec.tensor_name: spec for spec in build_tensor_specs()}
    anti_tensor = specs['anticommutator'].tensor
    result = scan_explicit_seeds('anticommutator', anti_tensor, 19, [best_seed], 1)
    terms = factors_to_terms('anti19', result['best_alpha'], result['best_beta'], result['best_gamma'])
    renamed = [Term(f's{term_idx + 1:02d}', 'step75_anticommutator_rank19_exact_seed', term.alpha, term.beta, term.gamma) for term_idx, term in enumerate(terms)]
    return renamed, float(result['best_max_abs']), best_seed


def balance_term_norms(terms: list[Term]) -> list[Term]:
    balanced: list[Term] = []
    for term in terms:
        alpha = np.asarray(term.alpha, dtype=np.float64)
        beta = np.asarray(term.beta, dtype=np.float64)
        gamma = np.asarray(term.gamma, dtype=np.float64)
        alpha_norm = float(np.linalg.norm(alpha))
        beta_norm = float(np.linalg.norm(beta))
        gamma_norm = float(np.linalg.norm(gamma))
        geo = float(np.exp((np.log(alpha_norm) + np.log(beta_norm) + np.log(gamma_norm)) / 3.0))
        lambda_scale = geo / alpha_norm
        mu_scale = geo / beta_norm
        gamma_scale = 1.0 / (lambda_scale * mu_scale)
        balanced.append(
            Term(
                term.term_id,
                term.source_label,
                lambda_scale * alpha,
                mu_scale * beta,
                gamma_scale * gamma,
            )
        )
    return balanced


def rescale_terms(terms: list[Term], rng: np.random.Generator) -> list[Term]:
    scaled_terms: list[Term] = []
    for term in terms:
        lambda_scale = float(np.exp(rng.uniform(-GAUGE_LOG_SCALE, GAUGE_LOG_SCALE)))
        mu_scale = float(np.exp(rng.uniform(-GAUGE_LOG_SCALE, GAUGE_LOG_SCALE)))
        gamma_scale = 1.0 / (lambda_scale * mu_scale)
        scaled_terms.append(
            Term(
                term.term_id,
                term.source_label,
                lambda_scale * np.asarray(term.alpha, dtype=np.float64),
                mu_scale * np.asarray(term.beta, dtype=np.float64),
                gamma_scale * np.asarray(term.gamma, dtype=np.float64),
            )
        )
    return scaled_terms


def largest_magnitude_index(vector: np.ndarray) -> int:
    return int(np.argmax(np.abs(vector)))


def build_support_model_step82(terms: list[Term]) -> SupportModel:
    witness_alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(9) for term in terms], axis=0)
    witness_beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(9) for term in terms], axis=0)
    witness_gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(9) for term in terms], axis=0)

    variable_specs: list[VariableSpec] = []
    variable_lookup: dict[tuple[int, str, int], int] = {}
    fixed_lookup: dict[tuple[int, str, int], float] = {}
    next_index = 0

    for term_index, term in enumerate(terms):
        gauge_alpha = largest_magnitude_index(witness_alpha[term_index])
        gauge_beta = largest_magnitude_index(witness_beta[term_index])
        for factor_name, flattened, gauge_index in (
            ('alpha', witness_alpha[term_index], gauge_alpha),
            ('beta', witness_beta[term_index], gauge_beta),
            ('gamma', witness_gamma[term_index], -1),
        ):
            for flat_index in np.flatnonzero(np.abs(flattened) > 0):
                flat_index = int(flat_index)
                value = float(flattened[flat_index])
                key = (term_index, factor_name, flat_index)
                if factor_name in {'alpha', 'beta'} and flat_index == gauge_index:
                    fixed_lookup[key] = value
                    continue
                variable_lookup[key] = next_index
                variable_specs.append(
                    VariableSpec(
                        variable_index=next_index,
                        term_index=term_index,
                        term_id=term.term_id,
                        factor_name=factor_name,
                        flat_index=flat_index,
                        row_index=flat_index // 3,
                        col_index=flat_index % 3,
                        witness_value=value,
                    )
                )
                next_index += 1

    return SupportModel(
        variable_specs=variable_specs,
        variable_lookup=variable_lookup,
        fixed_lookup=fixed_lookup,
        witness_alpha=witness_alpha,
        witness_beta=witness_beta,
        witness_gamma=witness_gamma,
    )


def julia_script_text_step82(
    model: SupportModel,
    selected_coordinates: list[tuple[int, int, int]],
    start_vector: np.ndarray,
    start_parameters: np.ndarray,
    target_parameters: np.ndarray,
) -> str:
    equation_lines = [f'    {coordinate_polynomial(model, coordinate, idx)},' for idx, coordinate in enumerate(selected_coordinates)]
    lines: list[str] = []
    w = lines.append
    w('using HomotopyContinuation')
    w('')
    w(f'@var x[1:{len(model.variable_specs)}] p[1:{len(selected_coordinates)}]')
    w('')
    w('equations = [')
    lines.extend(equation_lines)
    w(']')
    w('F = System(equations; variables=x, parameters=p)')
    w('')
    w('start_solution = [')
    for value in start_vector:
        w(f'    {julia_complex_literal(float(value))},')
    w(']')
    w('start_parameters = [')
    for value in start_parameters:
        w(f'    {julia_float_literal(float(value))},')
    w(']')
    w('target_parameters = [')
    for value in target_parameters:
        w(f'    {julia_float_literal(float(value))},')
    w(']')
    w('')
    w('result = solve(F, [start_solution]; start_parameters=start_parameters, target_parameters=target_parameters, show_progress=false, threading=false)')
    w('solutions_found = solutions(result)')
    w('println("STEP82_NUM_SOLUTIONS=" * string(length(solutions_found)))')
    w('if !isempty(solutions_found)')
    w('    endpoint = solutions_found[1]')
    w('    println("STEP82_ENDPOINT_REAL=" * join(string.(real.(endpoint)), ","))')
    w('    println("STEP82_ENDPOINT_IMAG=" * join(string.(imag.(endpoint)), ","))')
    w('    println("STEP82_MAX_IMAG=" * string(maximum(abs.(imag.(endpoint)))))')
    w('end')
    return '\n'.join(lines) + '\n'


def parse_step82_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith('STEP82_') and '=' in line:
            key, value = line.split('=', 1)
            parsed[key.strip()] = value.strip()
    return parsed


def screen_chart(chart_id: str, case_terms: list[Term], start_tensor: np.ndarray) -> dict:
    try:
        model = build_support_model_step82(case_terms)
        initial_variable_count = len(model.variable_specs)
        witness_vector = witness_free_vector(model)
        model, independent_variable_count, dropped_variable_count = reduce_model_to_independent_variables(model, witness_vector)
        witness_vector = witness_free_vector(model)
        selected_rows, _square_jacobian, subsystem_condition = select_square_subsystem(model, witness_vector)
        coordinates = full_coordinate_list()
        selected_coordinates = [coordinates[row_index] for row_index in selected_rows]
        start_alpha, start_beta, start_gamma = unpack_free_vector(model, witness_vector)
        start_parameters = evaluate_coordinates(start_alpha, start_beta, start_gamma, selected_coordinates)
        exact_start_parameters = target_values_for_coordinates(start_tensor, selected_coordinates)
        start_selected_consistency = float(np.max(np.abs(start_parameters - exact_start_parameters)))
        start_full_residual = float(np.max(np.abs(full_tensor_from_factors(start_alpha, start_beta, start_gamma) - start_tensor)))
        return {
            'chart_id': chart_id,
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
            'start_selected_consistency_max_abs': start_selected_consistency,
            'start_full_tensor_max_abs_residual': start_full_residual,
        }
    except Exception as exc:
        return {
            'chart_id': chart_id,
            'status': 'screen_failed',
            'error': str(exc),
        }


def run_prepared_case(prepared: dict, target_tensor: np.ndarray, julia_path: Path) -> dict:
    case_start = time.time()
    chart_id = str(prepared['chart_id'])
    model: SupportModel = prepared['model']
    witness_vector: np.ndarray = prepared['witness_vector']
    selected_coordinates: list[tuple[int, int, int]] = prepared['selected_coordinates']
    start_parameters: np.ndarray = prepared['start_parameters']
    target_parameters = target_values_for_coordinates(target_tensor, selected_coordinates)

    script_path = EXPORTS / f'step82_{chart_id}.jl'
    stdout_path = EXPORTS / f'step82_{chart_id}_stdout.txt'
    script_text = julia_script_text_step82(model, selected_coordinates, witness_vector, start_parameters, target_parameters)
    write_text(script_path, script_text)

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
            f'Timeout after {JULIA_TIMEOUT_SECONDS} seconds while tracking {chart_id}.\n'
            f'Condition number: {prepared["square_jacobian_condition_number"]:.16e}\n',
        )
        return {
            'chart_id': chart_id,
            'status': 'timeout',
            'runtime_seconds': scalar_to_str(time.time() - case_start),
            'square_jacobian_condition_number': scalar_to_str(prepared['square_jacobian_condition_number']),
            'initial_variable_count': prepared['initial_variable_count'],
            'reduced_variable_count': prepared['reduced_variable_count'],
            'selected_equation_count': prepared['selected_equation_count'],
            'stdout_path': str(stdout_path),
            'provenance': 'MEASURED_FROM_CODE',
        }

    write_text(stdout_path, process.stdout + ('\nSTDERR\n' + process.stderr if process.stderr else ''))
    row = {
        'chart_id': chart_id,
        'status': 'julia_error' if process.returncode != 0 else 'julia_ok',
        'runtime_seconds': scalar_to_str(time.time() - case_start),
        'square_jacobian_condition_number': scalar_to_str(prepared['square_jacobian_condition_number']),
        'initial_variable_count': prepared['initial_variable_count'],
        'reduced_variable_count': prepared['reduced_variable_count'],
        'selected_equation_count': prepared['selected_equation_count'],
        'stdout_path': str(stdout_path),
        'julia_return_code': process.returncode,
        'provenance': 'MEASURED_FROM_CODE',
    }
    if process.returncode != 0:
        return row

    parsed = parse_step82_output(process.stdout)
    row['num_solutions_returned'] = int(parsed.get('STEP82_NUM_SOLUTIONS', '0'))
    row['endpoint_max_imag'] = scalar_to_str(float(parsed.get('STEP82_MAX_IMAG', '0.0')))
    if int(row['num_solutions_returned']) == 0:
        row['status'] = 'no_solution_returned'
        return row

    endpoint_real = [float(token) for token in parsed.get('STEP82_ENDPOINT_REAL', '').split(',') if token.strip()]
    endpoint_vector = np.array(endpoint_real, dtype=np.float64)
    endpoint_alpha, endpoint_beta, endpoint_gamma = unpack_free_vector(model, endpoint_vector)
    endpoint_selected_values = evaluate_coordinates(endpoint_alpha, endpoint_beta, endpoint_gamma, selected_coordinates)
    endpoint_full_tensor = full_tensor_from_factors(endpoint_alpha, endpoint_beta, endpoint_gamma)

    row['endpoint_selected_residual_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_selected_values - target_parameters))))
    row['endpoint_full_tensor_max_abs_residual'] = scalar_to_str(float(np.max(np.abs(endpoint_full_tensor - target_tensor))))
    row['endpoint_to_start_chart_max_abs'] = scalar_to_str(float(np.max(np.abs(endpoint_vector - witness_vector))))
    row['status'] = 'complex_endpoint' if float(parsed.get('STEP82_MAX_IMAG', '0.0')) > 1e-7 else 'real_endpoint'
    return row


def main() -> None:
    start_time = time.time()
    julia_path = julia_executable()
    rng = np.random.default_rng(CASE_SEED)

    start_terms, exact_start_max_abs, best_seed = load_exact_step75_anticommutator_terms()
    balanced_terms = balance_term_norms(start_terms)
    specs = {spec.tensor_name: spec for spec in build_tensor_specs()}
    anti_tensor = specs['anticommutator'].tensor.astype(np.float64)
    full_tensor = matrix_multiplication_tensor(3).astype(np.float64)

    screened_rows: list[dict] = []
    prepared_cases: list[dict] = []

    for chart_id, case_terms in [('chart_00_balanced', balanced_terms)] + [
        (f'chart_{candidate_index:02d}', rescale_terms(balanced_terms, rng))
        for candidate_index in range(1, CHART_CANDIDATES + 1)
    ]:
        screened = screen_chart(chart_id, case_terms, anti_tensor)
        screened_rows.append(
            {
                'chart_id': screened['chart_id'],
                'status': screened['status'],
                'initial_variable_count': screened.get('initial_variable_count', ''),
                'reduced_variable_count': screened.get('reduced_variable_count', ''),
                'selected_equation_count': screened.get('selected_equation_count', ''),
                'square_jacobian_condition_number': '' if 'square_jacobian_condition_number' not in screened else scalar_to_str(screened['square_jacobian_condition_number']),
                'start_selected_consistency_max_abs': '' if 'start_selected_consistency_max_abs' not in screened else scalar_to_str(screened['start_selected_consistency_max_abs']),
                'start_full_tensor_max_abs_residual': '' if 'start_full_tensor_max_abs_residual' not in screened else scalar_to_str(screened['start_full_tensor_max_abs_residual']),
                'error': screened.get('error', ''),
                'provenance': 'MEASURED_FROM_CODE',
            }
        )
        if screened['status'] == 'screened_ok':
            prepared_cases.append(screened)

    prepared_cases.sort(key=lambda row: float(row['square_jacobian_condition_number']))
    tracked_cases = prepared_cases[:TRACK_CASES]

    case_rows: list[dict] = []
    for case_index, prepared in enumerate(tracked_cases, start=1):
        print(
            f"Running Step 82 track {case_index}/{len(tracked_cases)} on {prepared['chart_id']} with cond={prepared['square_jacobian_condition_number']:.3e}...",
            flush=True,
        )
        case_rows.append(run_prepared_case(prepared, full_tensor, julia_path))

    real_rows = [row for row in case_rows if row.get('status') == 'real_endpoint']
    best_row = min(real_rows, key=lambda row: float(row['endpoint_full_tensor_max_abs_residual'])) if real_rows else None

    summary = {
        'step82_best_seed': best_seed,
        'step82_exact_start_max_abs_residual': scalar_to_str(exact_start_max_abs),
        'step82_chart_candidates_requested': CHART_CANDIDATES,
        'step82_screened_ok_count': sum(1 for row in screened_rows if row['status'] == 'screened_ok'),
        'step82_screen_failed_count': sum(1 for row in screened_rows if row['status'] != 'screened_ok'),
        'step82_tracked_case_count': len(case_rows),
        'step82_real_endpoint_count': len(real_rows),
        'step82_timeout_count': sum(1 for row in case_rows if row.get('status') == 'timeout'),
        'step82_no_solution_count': sum(1 for row in case_rows if row.get('status') == 'no_solution_returned'),
        'step82_complex_endpoint_count': sum(1 for row in case_rows if row.get('status') == 'complex_endpoint'),
        'step82_julia_error_count': sum(1 for row in case_rows if row.get('status') == 'julia_error'),
        'step82_best_full_tensor_max_abs_residual': '' if best_row is None else best_row['endpoint_full_tensor_max_abs_residual'],
        'step82_best_selected_residual_max_abs': '' if best_row is None else best_row['endpoint_selected_residual_max_abs'],
        'step82_best_chart_id': '' if best_row is None else best_row['chart_id'],
        'step82_runtime_seconds': scalar_to_str(time.time() - start_time),
        'provenance': 'MEASURED_FROM_CODE',
    }

    write_csv(
        EXPORTS / 'step82_rank19_homotopy_screening.csv',
        screened_rows,
        [
            'chart_id',
            'status',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
            'start_selected_consistency_max_abs',
            'start_full_tensor_max_abs_residual',
            'error',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step82_rank19_homotopy_cases.csv',
        case_rows,
        [
            'chart_id',
            'status',
            'initial_variable_count',
            'reduced_variable_count',
            'selected_equation_count',
            'square_jacobian_condition_number',
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
    write_json(EXPORTS / 'step82_rank19_homotopy_summary.json', summary)

    print('=== Step 82: Rank-19 anticommutator homotopy ===', flush=True)
    print(f"Screened OK charts: {summary['step82_screened_ok_count']}", flush=True)
    print(f"Tracked charts: {summary['step82_tracked_case_count']}", flush=True)
    print(f"Real endpoints: {summary['step82_real_endpoint_count']}", flush=True)
    print(f"Best full tensor residual: {summary['step82_best_full_tensor_max_abs_residual'] or 'none'}", flush=True)


if __name__ == '__main__':
    main()