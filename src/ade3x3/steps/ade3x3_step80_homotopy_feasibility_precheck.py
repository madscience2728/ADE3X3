"""
ade3x3_step80_homotopy_feasibility_precheck.py

Step 80: Homotopy continuation feasibility precheck.

This step does the first honest pass on checklist item 3 without pretending the
 backend tooling already exists. It audits the local environment for Julia / PHC /
 Bertini / phcpy, records the raw polynomial-system sizes for rank-R CP models,
 and exports the dimensional facts needed to decide whether this route is ready
 to run or still blocked on setup.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


EXPORTS = Path('outputs/exports')
RANKS = [19, 20, 21, 22, 23]
TENSOR_ENTRY_COUNT = 9 * 9 * 9
MODE_DIM = 9
SEGRE_PROJECTIVE_DIM = 8 + 8 + 8
SEGRE_SECANT_TERM_DIM = SEGRE_PROJECTIVE_DIM + 1


def candidate_julia_paths() -> list[Path]:
    paths: list[Path] = []
    for env_var in ('LOCALAPPDATA', 'ProgramFiles'):
        base = os.environ.get(env_var)
        if not base:
            continue
        base_path = Path(base)
        if env_var == 'LOCALAPPDATA':
            paths.extend(sorted((base_path / 'Programs').glob('Julia-*\\bin\\julia.exe'), reverse=True))
        else:
            paths.extend(sorted(base_path.glob('Julia-*\\bin\\julia.exe'), reverse=True))
    return paths


def resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    if name == 'julia':
        for candidate in candidate_julia_paths():
            if candidate.exists():
                return str(candidate)
    return ''


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def scalar_to_str(value: float) -> str:
    return f'{float(value):.16e}'


def executable_row(name: str) -> dict:
    resolved = resolve_executable(name)
    return {
        'tool_name': name,
        'available': bool(resolved),
        'path': resolved,
        'provenance': 'MEASURED_FROM_CODE',
    }


def julia_package_row(julia_path: str, package_name: str) -> dict:
    if not julia_path:
        return {
            'package_name': package_name,
            'available': False,
            'julia_path': '',
            'package_path': '',
            'error': 'julia_not_found',
            'provenance': 'MEASURED_FROM_CODE',
        }

    expr = (
        'try\n'
        f'    using {package_name}\n'
        f'    print(Base.find_package("{package_name}"))\n'
        'catch err\n'
        '    print(stderr, err)\n'
        '    exit(1)\n'
        'end\n'
    )
    result = subprocess.run(
        [julia_path, '-e', expr],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return {
        'package_name': package_name,
        'available': result.returncode == 0,
        'julia_path': julia_path,
        'package_path': result.stdout.strip(),
        'error': result.stderr.strip(),
        'provenance': 'MEASURED_FROM_CODE',
    }


def python_module_row(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    origin = '' if spec is None or spec.origin is None else str(spec.origin)
    return {
        'module_name': name,
        'available': bool(spec),
        'origin': origin,
        'provenance': 'MEASURED_FROM_CODE',
    }


def rank_system_row(rank: int) -> dict:
    raw_unknowns = 3 * rank * MODE_DIM
    scaling_gauge_dim = 2 * rank
    effective_unknowns = raw_unknowns - scaling_gauge_dim
    affine_excess = TENSOR_ENTRY_COUNT - effective_unknowns
    expected_projective_secant_dim = min(rank * SEGRE_SECANT_TERM_DIM - 1, TENSOR_ENTRY_COUNT - 1)
    expected_affine_secant_dim = min(rank * SEGRE_SECANT_TERM_DIM, TENSOR_ENTRY_COUNT)
    codimension = TENSOR_ENTRY_COUNT - expected_affine_secant_dim
    return {
        'rank_value': rank,
        'tensor_entry_count': TENSOR_ENTRY_COUNT,
        'raw_unknown_count': raw_unknowns,
        'scaling_gauge_dimension': scaling_gauge_dim,
        'effective_unknown_count': effective_unknowns,
        'equation_minus_effective_unknowns': affine_excess,
        'expected_projective_secant_dimension': expected_projective_secant_dim,
        'expected_affine_secant_dimension': expected_affine_secant_dim,
        'ambient_codimension_at_expected_secant': codimension,
        'square_after_gauge_fixing': str(effective_unknowns == TENSOR_ENTRY_COUNT),
        'overdetermined_after_gauge_fixing': str(effective_unknowns < TENSOR_ENTRY_COUNT),
        'provenance': 'EXACT_DERIVED',
    }


def main() -> None:
    start = time.time()

    executable_rows = [executable_row(name) for name in ('julia', 'phc', 'bertini')]
    module_rows = [python_module_row(name) for name in ('phcpy', 'sympy', 'numpy', 'scipy')]
    julia_path = next((row['path'] for row in executable_rows if row['tool_name'] == 'julia' and row['available']), '')
    julia_package_rows = [julia_package_row(julia_path, 'HomotopyContinuation')]
    system_rows = [rank_system_row(rank) for rank in RANKS]

    julia_available = bool(julia_path)
    homotopycontinuation_available = any(
        row['package_name'] == 'HomotopyContinuation' and row['available']
        for row in julia_package_rows
    )
    phc_available = any(row['tool_name'] == 'phc' and row['available'] for row in executable_rows)
    bertini_available = any(row['tool_name'] == 'bertini' and row['available'] for row in executable_rows)
    phcpy_available = any(row['module_name'] == 'phcpy' and row['available'] for row in module_rows)

    julia_backend_ready = julia_available and homotopycontinuation_available
    backend_ready = julia_backend_ready or phc_available or bertini_available or phcpy_available
    preferred_backend = (
        'julia' if julia_backend_ready else
        'phc' if phc_available else
        'bertini' if bertini_available else
        'phcpy' if phcpy_available else
        'none'
    )

    summary = {
        'step80_backend_ready': backend_ready,
        'step80_preferred_backend': preferred_backend,
        'step80_julia_available': julia_available,
        'step80_julia_executable_path': julia_path,
        'step80_homotopycontinuation_available': homotopycontinuation_available,
        'step80_phc_available': phc_available,
        'step80_bertini_available': bertini_available,
        'step80_phcpy_available': phcpy_available,
        'step80_rank19_effective_unknown_count': next(row['effective_unknown_count'] for row in system_rows if row['rank_value'] == 19),
        'step80_rank19_equation_minus_effective_unknowns': next(row['equation_minus_effective_unknowns'] for row in system_rows if row['rank_value'] == 19),
        'step80_rank23_effective_unknown_count': next(row['effective_unknown_count'] for row in system_rows if row['rank_value'] == 23),
        'step80_runtime_seconds': scalar_to_str(time.time() - start),
    }

    notes = {
        'verdict': 'blocked_on_backend' if not backend_ready else 'backend_available',
        'interpretation': (
            'The homotopy route is not runnable yet in this workspace because no supported backend is installed and loadable.'
            if not backend_ready else
            'A homotopy backend is present and loadable locally, so the next step can move from precheck to actual system export and pilot path tracking.'
        ),
        'system_size_note': (
            'After quotienting the 2R scaling gauge, the raw rank-19 CP model still has 475 effective unknowns against 729 polynomial equations; '
            'the route is heavily overdetermined in affine coordinates and needs a careful square subsystem or parameter homotopy design.'
        ),
    }

    write_csv(
        EXPORTS / 'step80_homotopy_backend_check.csv',
        executable_rows,
        ['tool_name', 'available', 'path', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step80_homotopy_python_modules.csv',
        module_rows,
        ['module_name', 'available', 'origin', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step80_homotopy_julia_packages.csv',
        julia_package_rows,
        ['package_name', 'available', 'julia_path', 'package_path', 'error', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step80_homotopy_system_sizes.csv',
        system_rows,
        ['rank_value', 'tensor_entry_count', 'raw_unknown_count', 'scaling_gauge_dimension', 'effective_unknown_count', 'equation_minus_effective_unknowns', 'expected_projective_secant_dimension', 'expected_affine_secant_dimension', 'ambient_codimension_at_expected_secant', 'square_after_gauge_fixing', 'overdetermined_after_gauge_fixing', 'provenance'],
    )
    write_json(
        EXPORTS / 'step80_homotopy_feasibility.json',
        {
            'summary': summary,
            'notes': notes,
            'executables': executable_rows,
            'python_modules': module_rows,
            'julia_packages': julia_package_rows,
            'system_sizes': system_rows,
        },
    )

    print('=== Step 80: Homotopy continuation feasibility precheck ===', flush=True)
    print(f"Backend ready: {backend_ready} (preferred={preferred_backend})", flush=True)
    print(f"Rank-19 effective unknown count after scaling gauge: {summary['step80_rank19_effective_unknown_count']}", flush=True)
    print(f"Rank-19 equation minus effective unknowns: {summary['step80_rank19_equation_minus_effective_unknowns']}", flush=True)


if __name__ == '__main__':
    main()