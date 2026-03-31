"""
ade3x3_step81_homotopy_bridge_pilot.py

Step 81: HomotopyContinuation bridge pilot.

This is the first real execution step after the Step 80 environment precheck.
It does not attempt the full rank-19 system yet. Instead it builds a square,
support-restricted polynomial subsystem around the known exact public rank-23
witness, exports the corresponding Julia system, and tracks one parameter-
homotopy path from a nearby start assignment back to the ADE3X3 target.

The purpose of this pilot is to verify the full bridge end-to-end:
1. reuse the exact ADE3X3 tensor orientation already established in Step 63,
2. remove the two scaling gauges per term by fixing one alpha and one beta
   support entry for each of the 23 terms,
3. select an independent square subset of tensor equations using the Jacobian at
   the exact witness,
4. run HomotopyContinuation.jl on that square subsystem, and
5. check the returned endpoint against the full 729-entry tensor.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import qr

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
)


EXPORTS = Path('outputs/exports')
PERTURB_SCALE = float(os.environ.get('STEP81_PERTURB_SCALE', '0.05'))
RNG_SEED = int(os.environ.get('STEP81_SEED', '810023'))
JULIA_TIMEOUT_SECONDS = int(os.environ.get('STEP81_JULIA_TIMEOUT_SECONDS', '600'))


@dataclass(frozen=True)
class VariableSpec:
    variable_index: int
    term_index: int
    term_id: str
    factor_name: str
    flat_index: int
    row_index: int
    col_index: int
    witness_value: float


@dataclass
class SupportModel:
    variable_specs: list[VariableSpec]
    variable_lookup: dict[tuple[int, str, int], int]
    fixed_lookup: dict[tuple[int, str, int], float]
    witness_alpha: np.ndarray
    witness_beta: np.ndarray
    witness_gamma: np.ndarray


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def scalar_to_str(value: float) -> str:
    return f'{float(value):.16e}'


def candidate_julia_paths() -> list[Path]:
    paths: list[Path] = []
    local_app_data = os.environ.get('LOCALAPPDATA')
    program_files = os.environ.get('ProgramFiles')
    if local_app_data:
        paths.extend(sorted((Path(local_app_data) / 'Programs').glob('Julia-*\\bin\\julia.exe'), reverse=True))
    if program_files:
        paths.extend(sorted(Path(program_files).glob('Julia-*\\bin\\julia.exe'), reverse=True))
    return paths


def julia_executable() -> Path:
    for candidate in candidate_julia_paths():
        if candidate.exists():
            return candidate
    raise RuntimeError('Julia executable not found. Run Step 80 refresh or install Julia first.')


def flattened_terms(terms: list[Term]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(9) for term in terms], axis=0)
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(9) for term in terms], axis=0)
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(9) for term in terms], axis=0)
    return alpha, beta, gamma


def first_nonzero_index(vector: np.ndarray) -> int:
    nz = np.flatnonzero(np.abs(vector) > 0)
    if nz.size == 0:
        raise ValueError('Expected at least one nonzero support entry.')
    return int(nz[0])


def build_support_model(terms: list[Term]) -> SupportModel:
    witness_alpha, witness_beta, witness_gamma = flattened_terms(terms)
    variable_specs: list[VariableSpec] = []
    variable_lookup: dict[tuple[int, str, int], int] = {}
    fixed_lookup: dict[tuple[int, str, int], float] = {}

    next_index = 0
    for term_index, term in enumerate(terms):
        gauge_alpha = first_nonzero_index(witness_alpha[term_index])
        gauge_beta = first_nonzero_index(witness_beta[term_index])

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


def witness_free_vector(model: SupportModel) -> np.ndarray:
    vector = np.zeros(len(model.variable_specs), dtype=np.float64)
    for spec in model.variable_specs:
        source = {
            'alpha': model.witness_alpha,
            'beta': model.witness_beta,
            'gamma': model.witness_gamma,
        }[spec.factor_name]
        vector[spec.variable_index] = source[spec.term_index, spec.flat_index]
    return vector


def unpack_free_vector(model: SupportModel, free_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.zeros_like(model.witness_alpha)
    beta = np.zeros_like(model.witness_beta)
    gamma = np.zeros_like(model.witness_gamma)
    factors = {'alpha': alpha, 'beta': beta, 'gamma': gamma}

    for (term_index, factor_name, flat_index), value in model.fixed_lookup.items():
        factors[factor_name][term_index, flat_index] = value
    for spec in model.variable_specs:
        factors[spec.factor_name][spec.term_index, spec.flat_index] = float(free_vector[spec.variable_index])
    return alpha, beta, gamma


def evaluate_coordinates(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    coordinates: list[tuple[int, int, int]],
) -> np.ndarray:
    values = np.zeros(len(coordinates), dtype=np.float64)
    for coord_index, (a_idx, b_idx, c_idx) in enumerate(coordinates):
        values[coord_index] = float(np.sum(alpha[:, a_idx] * beta[:, b_idx] * gamma[:, c_idx]))
    return values


def full_tensor_from_factors(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    tensor = np.zeros((9, 9, 9), dtype=np.float64)
    for term_index in range(alpha.shape[0]):
        tensor += np.einsum('a,b,c->abc', alpha[term_index], beta[term_index], gamma[term_index], optimize=True)
    return tensor


def full_coordinate_list() -> list[tuple[int, int, int]]:
    return [(a_idx, b_idx, c_idx) for a_idx in range(9) for b_idx in range(9) for c_idx in range(9)]


def build_jacobian(model: SupportModel, free_vector: np.ndarray, coordinates: list[tuple[int, int, int]]) -> np.ndarray:
    alpha, beta, gamma = unpack_free_vector(model, free_vector)
    n_coords = len(coordinates)
    n_vars = len(model.variable_specs)
    if n_vars == 0:
        return np.zeros((n_coords, 0), dtype=np.float64)

    coords = np.asarray(coordinates, dtype=np.intp)
    coord_a, coord_b, coord_c = coords[:, 0], coords[:, 1], coords[:, 2]

    factor_map = {'alpha': 0, 'beta': 1, 'gamma': 2}
    spec_factor = np.empty(n_vars, dtype=np.int8)
    spec_flat = np.empty(n_vars, dtype=np.intp)
    spec_term = np.empty(n_vars, dtype=np.intp)
    spec_col = np.empty(n_vars, dtype=np.intp)
    for j, spec in enumerate(model.variable_specs):
        spec_factor[j] = factor_map[spec.factor_name]
        spec_flat[j] = spec.flat_index
        spec_term[j] = spec.term_index
        spec_col[j] = spec.variable_index

    jacobian = np.zeros((n_coords, n_vars), dtype=np.float64)

    # Alpha variables: d/d(alpha[t,f]) = delta(a,f) * beta[t,b] * gamma[t,c]
    idx = np.where(spec_factor == 0)[0]
    if idx.size:
        t, f, c = spec_term[idx], spec_flat[idx], spec_col[idx]
        mask = (coord_a[None, :] == f[:, None])  # (n_group, n_coords)
        jacobian[:, c] = (mask * beta[t][:, coord_b] * gamma[t][:, coord_c]).T

    # Beta variables: d/d(beta[t,f]) = alpha[t,a] * delta(b,f) * gamma[t,c]
    idx = np.where(spec_factor == 1)[0]
    if idx.size:
        t, f, c = spec_term[idx], spec_flat[idx], spec_col[idx]
        mask = (coord_b[None, :] == f[:, None])
        jacobian[:, c] = (mask * alpha[t][:, coord_a] * gamma[t][:, coord_c]).T

    # Gamma variables: d/d(gamma[t,f]) = alpha[t,a] * beta[t,b] * delta(c,f)
    idx = np.where(spec_factor == 2)[0]
    if idx.size:
        t, f, c = spec_term[idx], spec_flat[idx], spec_col[idx]
        mask = (coord_c[None, :] == f[:, None])
        jacobian[:, c] = (mask * alpha[t][:, coord_a] * beta[t][:, coord_b]).T

    return jacobian


def reduce_model_to_independent_variables(model: SupportModel, free_vector: np.ndarray) -> tuple[SupportModel, int, int]:
    jacobian = build_jacobian(model, free_vector, full_coordinate_list())
    _, r_factor, pivots = qr(jacobian, pivoting=True, mode='economic')
    diagonal = np.abs(np.diag(r_factor))
    if diagonal.size == 0 or diagonal[0] == 0:
        raise RuntimeError('Variable Jacobian has zero rank at the witness.')
    tolerance = diagonal[0] * 1e-12
    rank = int(np.sum(diagonal > tolerance))
    active_columns = set(int(index) for index in pivots[:rank])

    reduced_variable_specs: list[VariableSpec] = []
    reduced_variable_lookup: dict[tuple[int, str, int], int] = {}
    reduced_fixed_lookup = dict(model.fixed_lookup)

    next_index = 0
    for spec in model.variable_specs:
        key = (spec.term_index, spec.factor_name, spec.flat_index)
        if spec.variable_index in active_columns:
            reduced_variable_specs.append(
                VariableSpec(
                    variable_index=next_index,
                    term_index=spec.term_index,
                    term_id=spec.term_id,
                    factor_name=spec.factor_name,
                    flat_index=spec.flat_index,
                    row_index=spec.row_index,
                    col_index=spec.col_index,
                    witness_value=spec.witness_value,
                )
            )
            reduced_variable_lookup[key] = next_index
            next_index += 1
        else:
            reduced_fixed_lookup[key] = spec.witness_value

    reduced_model = SupportModel(
        variable_specs=reduced_variable_specs,
        variable_lookup=reduced_variable_lookup,
        fixed_lookup=reduced_fixed_lookup,
        witness_alpha=model.witness_alpha,
        witness_beta=model.witness_beta,
        witness_gamma=model.witness_gamma,
    )
    return reduced_model, rank, len(model.variable_specs) - rank


def select_square_subsystem(model: SupportModel, free_vector: np.ndarray) -> tuple[list[int], np.ndarray, float]:
    coordinates = full_coordinate_list()
    jacobian = build_jacobian(model, free_vector, coordinates)
    _, r_factor, pivots = qr(jacobian.T, pivoting=True, mode='economic')
    diagonal = np.abs(np.diag(r_factor))
    if diagonal.size == 0 or diagonal[0] == 0:
        raise RuntimeError('Jacobian has zero rank at the witness; cannot build a square subsystem.')
    tolerance = diagonal[0] * 1e-12
    rank = int(np.sum(diagonal > tolerance))
    if rank != len(model.variable_specs):
        raise RuntimeError(
            f'Witness Jacobian rank {rank} does not match variable count {len(model.variable_specs)}.'
        )
    selected_rows = [int(index) for index in pivots[:rank]]
    square_jacobian = jacobian[selected_rows, :]
    condition = float(np.linalg.cond(square_jacobian))
    return selected_rows, square_jacobian, condition


def target_values_for_coordinates(target_tensor: np.ndarray, coordinates: list[tuple[int, int, int]]) -> np.ndarray:
    return np.array([float(target_tensor[a_idx, b_idx, c_idx]) for a_idx, b_idx, c_idx in coordinates], dtype=np.float64)


def perturbed_start_vector(model: SupportModel, witness_vector: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    start_vector = witness_vector.copy()
    for spec in model.variable_specs:
        start_vector[spec.variable_index] += PERTURB_SCALE * rng.standard_normal()
    return start_vector


def julia_float_literal(value: float) -> str:
    if float(value).is_integer():
        return f'{float(value):.1f}'
    return repr(float(value))


def julia_complex_literal(value: float) -> str:
    return f'ComplexF64({julia_float_literal(float(value))}, 0.0)'


def factor_reference(model: SupportModel, term_index: int, factor_name: str, flat_index: int) -> str | None:
    variable_index = model.variable_lookup.get((term_index, factor_name, flat_index))
    if variable_index is not None:
        return f'x[{variable_index + 1}]'
    fixed_value = model.fixed_lookup.get((term_index, factor_name, flat_index))
    if fixed_value is not None:
        return julia_float_literal(fixed_value)
    return None


def coordinate_polynomial(model: SupportModel, coordinate: tuple[int, int, int], parameter_index: int) -> str:
    a_idx, b_idx, c_idx = coordinate
    monomials: list[str] = []
    term_count = model.witness_alpha.shape[0]
    for term_index in range(term_count):
        alpha_ref = factor_reference(model, term_index, 'alpha', a_idx)
        beta_ref = factor_reference(model, term_index, 'beta', b_idx)
        gamma_ref = factor_reference(model, term_index, 'gamma', c_idx)
        if alpha_ref is None or beta_ref is None or gamma_ref is None:
            continue
        monomials.append(f'({alpha_ref})*({beta_ref})*({gamma_ref})')
    lhs = ' + '.join(monomials) if monomials else '0.0'
    return f'({lhs}) - p[{parameter_index + 1}]'


def julia_script_text(
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
    w('result = solve(F, [start_solution]; start_parameters=start_parameters, target_parameters=target_parameters, show_progress=false)')
    w('solutions_found = solutions(result)')
    w('println("STEP81_NUM_SOLUTIONS=" * string(length(solutions_found)))')
    w('if !isempty(solutions_found)')
    w('    endpoint = solutions_found[1]')
    w('    println("STEP81_ENDPOINT_REAL=" * join(string.(real.(endpoint)), ","))')
    w('    println("STEP81_ENDPOINT_IMAG=" * join(string.(imag.(endpoint)), ","))')
    w('    println("STEP81_MAX_IMAG=" * string(maximum(abs.(imag.(endpoint)))))')
    w('end')
    return '\n'.join(lines) + '\n'


def parse_julia_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith('STEP81_') and '=' in line:
            key, value = line.split('=', 1)
            parsed[key.strip()] = value.strip()
    return parsed


def endpoint_vector_from_output(parsed: dict[str, str]) -> np.ndarray:
    if 'STEP81_ENDPOINT_REAL' not in parsed:
        raise RuntimeError('Julia output did not include an endpoint vector.')
    real_values = [float(token) for token in parsed['STEP81_ENDPOINT_REAL'].split(',') if token.strip()]
    imag_values = [float(token) for token in parsed.get('STEP81_ENDPOINT_IMAG', '').split(',') if token.strip()]
    if imag_values and max(abs(value) for value in imag_values) > 1e-7:
        raise RuntimeError(f'Endpoint has large imaginary part: {max(abs(value) for value in imag_values)}')
    return np.array(real_values, dtype=np.float64)


def main() -> None:
    start_time = time.time()
    julia_path = julia_executable()

    terms, orientation, reconstruction_max_abs = load_public_rank23_terms()
    target_tensor = matrix_multiplication_tensor(3).astype(np.float64)
    model = build_support_model(terms)
    full_chart_variable_count = len(model.variable_specs)
    witness_vector = witness_free_vector(model)
    model, independent_variable_count, dropped_variable_count = reduce_model_to_independent_variables(model, witness_vector)
    witness_vector = witness_free_vector(model)
    selected_rows, square_jacobian, subsystem_condition = select_square_subsystem(model, witness_vector)
    all_coordinates = full_coordinate_list()
    selected_coordinates = [all_coordinates[row_index] for row_index in selected_rows]

    start_vector = perturbed_start_vector(model, witness_vector)
    start_alpha, start_beta, start_gamma = unpack_free_vector(model, start_vector)
    witness_alpha, witness_beta, witness_gamma = unpack_free_vector(model, witness_vector)
    start_parameters = evaluate_coordinates(start_alpha, start_beta, start_gamma, selected_coordinates)
    target_parameters = target_values_for_coordinates(target_tensor, selected_coordinates)
    start_selected_residual = float(np.max(np.abs(start_parameters - evaluate_coordinates(start_alpha, start_beta, start_gamma, selected_coordinates))))

    julia_script = julia_script_text(model, selected_coordinates, start_vector, start_parameters, target_parameters)
    julia_script_path = EXPORTS / 'step81_homotopy_pilot.jl'
    julia_stdout_path = EXPORTS / 'step81_homotopy_pilot_stdout.txt'
    write_text(julia_script_path, julia_script)

    process = subprocess.run(
        [str(julia_path), str(julia_script_path)],
        capture_output=True,
        text=True,
        timeout=JULIA_TIMEOUT_SECONDS,
        check=False,
    )
    write_text(julia_stdout_path, process.stdout + ('\nSTDERR\n' + process.stderr if process.stderr else ''))
    if process.returncode != 0:
        raise RuntimeError(f'Julia pilot failed with code {process.returncode}. See {julia_stdout_path}.')

    parsed = parse_julia_output(process.stdout)
    endpoint_vector = endpoint_vector_from_output(parsed)
    endpoint_alpha, endpoint_beta, endpoint_gamma = unpack_free_vector(model, endpoint_vector)
    endpoint_selected_values = evaluate_coordinates(endpoint_alpha, endpoint_beta, endpoint_gamma, selected_coordinates)
    endpoint_full_tensor = full_tensor_from_factors(endpoint_alpha, endpoint_beta, endpoint_gamma)
    witness_full_tensor = full_tensor_from_factors(witness_alpha, witness_beta, witness_gamma)

    endpoint_selected_max_abs = float(np.max(np.abs(endpoint_selected_values - target_parameters)))
    endpoint_full_max_abs = float(np.max(np.abs(endpoint_full_tensor - target_tensor)))
    endpoint_to_witness_max_abs = float(np.max(np.abs(endpoint_vector - witness_vector)))
    endpoint_max_imag = float(parsed.get('STEP81_MAX_IMAG', '0.0'))

    variable_rows = [
        {
            'variable_index': spec.variable_index,
            'term_index': spec.term_index,
            'term_id': spec.term_id,
            'factor_name': spec.factor_name,
            'flat_index': spec.flat_index,
            'row_index': spec.row_index,
            'col_index': spec.col_index,
            'witness_value': scalar_to_str(spec.witness_value),
            'start_value': scalar_to_str(start_vector[spec.variable_index]),
            'endpoint_value': scalar_to_str(endpoint_vector[spec.variable_index]),
            'provenance': 'MEASURED_FROM_CODE',
        }
        for spec in model.variable_specs
    ]
    equation_rows = [
        {
            'equation_index': equation_index,
            'full_tensor_row_index': selected_rows[equation_index],
            'a_flat_index': coordinate[0],
            'b_flat_index': coordinate[1],
            'c_flat_index': coordinate[2],
            'target_value': scalar_to_str(target_parameters[equation_index]),
            'start_parameter_value': scalar_to_str(start_parameters[equation_index]),
            'endpoint_value': scalar_to_str(endpoint_selected_values[equation_index]),
            'endpoint_abs_residual': scalar_to_str(abs(endpoint_selected_values[equation_index] - target_parameters[equation_index])),
            'provenance': 'MEASURED_FROM_CODE',
        }
        for equation_index, coordinate in enumerate(selected_coordinates)
    ]

    summary = {
        'step81_rank_value': 23,
        'step81_orientation': orientation,
        'step81_public_reconstruction_max_abs': str(reconstruction_max_abs),
        'step81_support_variable_count_initial': full_chart_variable_count,
        'step81_support_variable_count_reduced': len(model.variable_specs),
        'step81_support_variable_drop_count': dropped_variable_count,
        'step81_independent_variable_rank': independent_variable_count,
        'step81_fixed_gauge_count': len(model.fixed_lookup),
        'step81_selected_equation_count': len(selected_coordinates),
        'step81_square_jacobian_condition_number': scalar_to_str(subsystem_condition),
        'step81_julia_path': str(julia_path),
        'step81_julia_return_code': process.returncode,
        'step81_path_count_requested': 1,
        'step81_num_solutions_returned': int(parsed.get('STEP81_NUM_SOLUTIONS', '0')),
        'step81_start_selected_residual_max_abs': scalar_to_str(start_selected_residual),
        'step81_endpoint_selected_residual_max_abs': scalar_to_str(endpoint_selected_max_abs),
        'step81_endpoint_full_tensor_max_abs': scalar_to_str(endpoint_full_max_abs),
        'step81_endpoint_to_witness_max_abs': scalar_to_str(endpoint_to_witness_max_abs),
        'step81_endpoint_max_imag': scalar_to_str(endpoint_max_imag),
        'step81_witness_full_tensor_max_abs': scalar_to_str(float(np.max(np.abs(witness_full_tensor - target_tensor)))),
        'step81_runtime_seconds': scalar_to_str(time.time() - start_time),
        'provenance': 'MEASURED_FROM_CODE',
    }

    write_csv(
        EXPORTS / 'step81_homotopy_variable_layout.csv',
        variable_rows,
        ['variable_index', 'term_index', 'term_id', 'factor_name', 'flat_index', 'row_index', 'col_index', 'witness_value', 'start_value', 'endpoint_value', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step81_homotopy_selected_equations.csv',
        equation_rows,
        ['equation_index', 'full_tensor_row_index', 'a_flat_index', 'b_flat_index', 'c_flat_index', 'target_value', 'start_parameter_value', 'endpoint_value', 'endpoint_abs_residual', 'provenance'],
    )
    write_json(
        EXPORTS / 'step81_homotopy_pilot_summary.json',
        summary,
    )

    print('=== Step 81: HomotopyContinuation bridge pilot ===', flush=True)
    print(f"Selected square subsystem size: {len(selected_coordinates)}", flush=True)
    print(f"Endpoint selected max residual: {summary['step81_endpoint_selected_residual_max_abs']}", flush=True)
    print(f"Endpoint full tensor max residual: {summary['step81_endpoint_full_tensor_max_abs']}", flush=True)
    print(f"Endpoint to witness max abs difference: {summary['step81_endpoint_to_witness_max_abs']}", flush=True)


if __name__ == '__main__':
    main()