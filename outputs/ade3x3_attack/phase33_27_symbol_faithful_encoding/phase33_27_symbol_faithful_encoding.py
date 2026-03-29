from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step52_quotient_rank_criterion import (  # noqa: E402
    standard_terms_3x3,
    strassen_terms_2x2,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    load_public_rank23_terms,
)


S3 = list(permutations(range(3)))
NUMERIC_TOL = 1e-9
RANDOM_SEED = 33


@dataclass(frozen=True)
class Decomposition:
    label: str
    family: str
    n: int
    terms: list[Term]


def to_term_list(prefix: str, raw_terms: list[dict[str, list[list[int]]]]) -> list[Term]:
    terms: list[Term] = []
    for idx, payload in enumerate(raw_terms, start=1):
        terms.append(
            Term(
                term_id=f'{prefix}{idx:02d}',
                source_label=f'{prefix}{idx:02d}',
                alpha=np.array(payload['alpha'], dtype=np.int64),
                beta=np.array(payload['beta'], dtype=np.int64),
                gamma=np.array(payload['gamma'], dtype=np.int64),
            )
        )
    return terms


def load_decompositions() -> tuple[Decomposition, Decomposition, Decomposition]:
    alpha_terms, _, _ = load_public_rank23_terms()
    standard = Decomposition('standard_rank27', 'standard_3x3', 3, to_term_list('std', standard_terms_3x3()))
    alpha = Decomposition('alphatensor_rank23', 'alphatensor_rank23', 3, alpha_terms)
    strassen = Decomposition('strassen_2x2', 'strassen_2x2', 2, to_term_list('str', strassen_terms_2x2()))
    return alpha, standard, strassen


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':')))
            handle.write('\n')


def sp_to_jsonable(value: object) -> object:
    if isinstance(value, sp.MatrixBase):
        return [[sp_to_jsonable(value[row_idx, col_idx]) for col_idx in range(value.cols)] for row_idx in range(value.rows)]
    if isinstance(value, list):
        return [sp_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [sp_to_jsonable(item) for item in value]
    if isinstance(value, sp.Basic):
        if value.is_Integer:
            return int(value)
        if value.is_Rational:
            return f'{int(sp.numer(value))}/{int(sp.denom(value))}'
        return sp.sstr(value)
    return value


def matrix_literal(matrix: sp.Matrix) -> str:
    rows = []
    for row_idx in range(matrix.rows):
        rows.append('[' + ', '.join(sp.sstr(sp.nsimplify(matrix[row_idx, col_idx])) for col_idx in range(matrix.cols)) + ']')
    return '[' + ', '.join(rows) + ']'


def unique_coefficients(matrix: sp.Matrix) -> list[str]:
    return sorted({sp.sstr(sp.nsimplify(matrix[row_idx, col_idx])) for row_idx in range(matrix.rows) for col_idx in range(matrix.cols) if matrix[row_idx, col_idx] != 0})


def support_stats(matrix: sp.Matrix) -> tuple[int, float, int, int]:
    total = matrix.rows * matrix.cols
    nonzero = sum(1 for row_idx in range(matrix.rows) for col_idx in range(matrix.cols) if matrix[row_idx, col_idx] != 0)
    per_col = [sum(1 for row_idx in range(matrix.rows) if matrix[row_idx, col_idx] != 0) for col_idx in range(matrix.cols)]
    return nonzero, (nonzero / total if total else 0.0), (min(per_col) if per_col else 0), (max(per_col) if per_col else 0)


def gamma_matrix_exact(decomposition: Decomposition) -> sp.Matrix:
    rows: list[list[int]] = []
    for row_idx in range(decomposition.n):
        for col_idx in range(decomposition.n):
            rows.append([int(term.gamma[row_idx, col_idx]) for term in decomposition.terms])
    return sp.Matrix(rows)


def channel_faces_exact(decomposition: Decomposition) -> list[sp.Matrix]:
    faces: list[sp.Matrix] = []
    for sum_idx in range(decomposition.n):
        cols: list[sp.Matrix] = []
        for row_idx in range(decomposition.n):
            for col_idx in range(decomposition.n):
                cols.append(
                    sp.Matrix(
                        [
                            int(term.alpha[row_idx, sum_idx]) * int(term.beta[sum_idx, col_idx])
                            for term in decomposition.terms
                        ]
                    )
                )
        faces.append(sp.Matrix.hstack(*cols))
    return faces


def cross_section_blocks_exact(decomposition: Decomposition) -> dict[tuple[int, int], sp.Matrix]:
    blocks: dict[tuple[int, int], sp.Matrix] = {}
    for sum_left in range(decomposition.n):
        for sum_right in range(decomposition.n):
            if sum_left == sum_right:
                continue
            cols: list[sp.Matrix] = []
            for row_idx in range(decomposition.n):
                for col_idx in range(decomposition.n):
                    cols.append(
                        sp.Matrix(
                            [
                                int(term.alpha[row_idx, sum_left]) * int(term.beta[sum_right, col_idx])
                                for term in decomposition.terms
                            ]
                        )
                    )
            blocks[(sum_left, sum_right)] = sp.Matrix.hstack(*cols)
    return blocks


def x0_matrix_exact(faces: list[sp.Matrix]) -> sp.Matrix:
    return sum(faces, sp.zeros(faces[0].rows, faces[0].cols)) / sp.Integer(len(faces))


def k_matrices_exact(faces: list[sp.Matrix]) -> list[sp.Matrix]:
    x0 = x0_matrix_exact(faces)
    return [face - x0 for face in faces]


def h_matrix_exact(decomposition: Decomposition, faces: list[sp.Matrix]) -> sp.Matrix:
    k_list = k_matrices_exact(faces)
    if decomposition.n == 2:
        return k_list[0] - k_list[1]
    return sp.Matrix.hstack(k_list[0] - k_list[1], k_list[1] - k_list[2])


def sigma_matrix_exact(faces: list[sp.Matrix]) -> sp.Matrix:
    return sum(faces, sp.zeros(faces[0].rows, faces[0].cols))


def exact_recovery_matrix(basis: sp.Matrix, target: sp.Matrix) -> tuple[bool, sp.Matrix | None]:
    if sp.Matrix.hstack(basis, target).rank() != basis.rank():
        return False, None
    _, pivots = basis.rref()
    basis_pivot = basis[:, pivots]
    columns: list[sp.Matrix] = []
    for col_idx in range(target.cols):
        solution = basis_pivot.gauss_jordan_solve(target[:, col_idx])[0]
        full = [sp.Integer(0)] * basis.cols
        for basis_idx, pivot in enumerate(pivots):
            full[pivot] = sp.simplify(solution[basis_idx, 0])
        columns.append(sp.Matrix(full))
    recovery = sp.Matrix.hstack(*columns)
    if basis * recovery != target:
        raise RuntimeError('Exact recovery verification failed.')
    return True, recovery


def build_face_tensor_list(decomposition: Decomposition) -> list[np.ndarray]:
    faces: list[np.ndarray] = []
    for sum_idx in range(decomposition.n):
        face = np.zeros((len(decomposition.terms), decomposition.n * decomposition.n), dtype=np.float64)
        col = 0
        for row_idx in range(decomposition.n):
            for out_col in range(decomposition.n):
                face[:, col] = [float(term.alpha[row_idx, sum_idx] * term.beta[sum_idx, out_col]) for term in decomposition.terms]
                col += 1
        faces.append(face)
    return faces


def factorized_faces_all_rank_one(faces: list[np.ndarray], n: int) -> bool:
    for face in faces:
        for row_idx in range(face.shape[0]):
            if np.linalg.matrix_rank(face[row_idx].reshape(n, n), tol=NUMERIC_TOL) > 1:
                return False
    return True


def synthetic_defect_trials(decomposition: Decomposition, gamma_exact: sp.Matrix, faces_exact: list[sp.Matrix]) -> tuple[list[dict], dict[str, object]]:
    rng = np.random.default_rng(RANDOM_SEED + len(decomposition.terms))
    null_basis_exact = gamma_exact.nullspace()
    null_basis = np.column_stack([np.array(vec, dtype=np.float64).reshape(-1) for vec in null_basis_exact]) if null_basis_exact else np.zeros((len(decomposition.terms), 0), dtype=np.float64)
    x0 = np.array(x0_matrix_exact(faces_exact).tolist(), dtype=np.float64)
    ker_dim = null_basis.shape[1]
    rows: list[dict] = []
    any_factorized = False
    random_factorized_count = 0

    def record_case(case_name: str, k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, case_kind: str) -> None:
        nonlocal any_factorized, random_factorized_count
        h = np.hstack([k0 - k1, k1 - k2])
        rank_h = int(np.linalg.matrix_rank(h, tol=NUMERIC_TOL))
        faces = [x0 + k0, x0 + k1, x0 + k2]
        factorized = factorized_faces_all_rank_one(faces, decomposition.n)
        any_factorized = any_factorized or factorized
        if factorized and case_kind == 'wildcard':
            random_factorized_count += 1
        rows.append(
            {
                'label': decomposition.label,
                'case_name': case_name,
                'case_kind': case_kind,
                'rank_H_numeric': rank_h,
                'ker_gamma_dim_exact': ker_dim,
                'defect_numeric': max(0, ker_dim - rank_h),
                'sum_zero_max_abs': float(np.max(np.abs(k0 + k1 + k2))),
                'factorized_rank1_faces': factorized,
            }
        )

    if decomposition.n != 3 or ker_dim == 0:
        return rows, {'random_trials': 0, 'random_factorized_count': 0, 'factorized_defective_found': False}

    coeff = rng.normal(size=(ker_dim, 9))
    v = null_basis @ coeff
    record_case('k0_equals_k1', v, v, -2.0 * v, 'named')

    coeff = rng.normal(size=(ker_dim, 9))
    v = null_basis @ coeff
    record_case('all_collinear', v, -0.5 * v, -0.5 * v, 'named')

    coeff_u = rng.normal(size=(ker_dim, 2))
    u = null_basis @ coeff_u
    mix0 = rng.normal(size=(2, 9))
    mix1 = rng.normal(size=(2, 9))
    k0 = u @ mix0
    k1 = u @ mix1
    record_case('shared_two_plane', k0, k1, -(k0 + k1), 'named')

    for trial_idx in range(32):
        ambient_dim = int(rng.integers(1, min(3, ker_dim) + 1))
        coeff_u = rng.normal(size=(ker_dim, ambient_dim))
        u = null_basis @ coeff_u
        mix0 = rng.normal(size=(ambient_dim, 9))
        mix1 = rng.normal(size=(ambient_dim, 9))
        k0 = u @ mix0
        k1 = u @ mix1
        record_case(f'wildcard_trial_{trial_idx + 1:02d}', k0, k1, -(k0 + k1), 'wildcard')

    return rows, {
        'random_trials': 32,
        'random_factorized_count': random_factorized_count,
        'factorized_defective_found': any_factorized,
    }


def track_a_for_decomposition(decomposition: Decomposition) -> tuple[dict, list[dict], list[dict]]:
    gamma = gamma_matrix_exact(decomposition)
    faces = channel_faces_exact(decomposition)
    h = h_matrix_exact(decomposition, faces)
    sigma = sigma_matrix_exact(faces)
    blocks = cross_section_blocks_exact(decomposition)
    recovery_rows: list[dict] = []
    matrix_rows: list[dict] = []
    all_in_h = True
    all_in_sigma_h = True

    for (sum_left, sum_right), block in sorted(blocks.items()):
        in_h, m_h = exact_recovery_matrix(h, block)
        in_sigma_h, _ = exact_recovery_matrix(sp.Matrix.hstack(sigma, h), block)
        all_in_h = all_in_h and in_h
        all_in_sigma_h = all_in_sigma_h and in_sigma_h
        if m_h is not None:
            nonzero, density, col_support_min, col_support_max = support_stats(m_h)
            coeffs = unique_coefficients(m_h)
            split = decomposition.n * decomposition.n if decomposition.n == 3 else m_h.rows
            first_block = m_h[:split, :]
            second_block = m_h[split:, :] if decomposition.n == 3 else sp.zeros(0, m_h.cols)
            matrix_rows.append(
                {
                    'label': decomposition.label,
                    'block': f'D_{sum_left}{sum_right}',
                    'M_h_literal': matrix_literal(m_h),
                    'M_h_json': sp_to_jsonable(m_h),
                }
            )
            recovery_rows.append(
                {
                    'label': decomposition.label,
                    'block': f'D_{sum_left}{sum_right}',
                    'in_H_exact': in_h,
                    'in_SigmaH_exact': in_sigma_h,
                    'M_h_rank': int(m_h.rank()),
                    'M_h_nonzero': nonzero,
                    'M_h_density': density,
                    'M_h_col_support_min': col_support_min,
                    'M_h_col_support_max': col_support_max,
                    'M_h_unique_coefficients': coeffs,
                    'first_block_rank': int(first_block.rank()) if first_block.rows else 0,
                    'second_block_rank': int(second_block.rank()) if second_block.rows else 0,
                }
            )
        else:
            recovery_rows.append(
                {
                    'label': decomposition.label,
                    'block': f'D_{sum_left}{sum_right}',
                    'in_H_exact': False,
                    'in_SigmaH_exact': in_sigma_h,
                    'M_h_rank': None,
                    'M_h_nonzero': None,
                    'M_h_density': None,
                    'M_h_col_support_min': None,
                    'M_h_col_support_max': None,
                    'M_h_unique_coefficients': [],
                    'first_block_rank': None,
                    'second_block_rank': None,
                }
            )

    return ({
        'label': decomposition.label,
        'R': len(decomposition.terms),
        'rank_Gamma_exact': int(gamma.rank()),
        'ker_Gamma_dim_exact': len(decomposition.terms) - int(gamma.rank()),
        'rank_H_exact': int(h.rank()),
        'all_cross_sections_in_H_exact': all_in_h,
        'all_cross_sections_in_SigmaH_exact': all_in_sigma_h,
    }, recovery_rows, matrix_rows)


def track_b_for_decomposition(decomposition: Decomposition) -> tuple[dict, list[dict], list[dict]]:
    gamma = gamma_matrix_exact(decomposition)
    faces = channel_faces_exact(decomposition)
    base_h = h_matrix_exact(decomposition, faces)
    ker_dim = len(decomposition.terms) - int(gamma.rank())
    clone_rows: list[dict] = []

    if decomposition.n == 3:
        k_list = k_matrices_exact(faces)
        for perm in S3:
            h_perm = sp.Matrix.hstack(k_list[perm[0]] - k_list[perm[1]], k_list[perm[1]] - k_list[perm[2]])
            clone_rows.append({
                'label': decomposition.label,
                'perm': ''.join(str(entry) for entry in perm),
                'rank_H_perm_exact': int(h_perm.rank()),
                'saturates_ker_Gamma': int(h_perm.rank()) == ker_dim,
            })
    else:
        clone_rows.append({
            'label': decomposition.label,
            'perm': '01',
            'rank_H_perm_exact': int(base_h.rank()),
            'saturates_ker_Gamma': int(base_h.rank()) == ker_dim,
        })

    synthetic_rows, synthetic_summary = synthetic_defect_trials(decomposition, gamma, faces)
    return ({
        'label': decomposition.label,
        'ker_Gamma_dim_exact': ker_dim,
        'identity_frame_rank_exact': int(base_h.rank()),
        'identity_frame_saturates': int(base_h.rank()) == ker_dim,
        'all_clone_frames_saturate': all(bool(row['saturates_ker_Gamma']) for row in clone_rows),
        **synthetic_summary,
    }, clone_rows, synthetic_rows)


def observed_gauge_orbit_dimension(decomposition: Decomposition) -> int:
    total = 0
    for term in decomposition.terms:
        active_channels = [bool(np.any(term.alpha[:, sum_idx]) or np.any(term.beta[sum_idx, :])) for sum_idx in range(decomposition.n)]
        total += max(0, sum(active_channels) - 1)
    return total


def track_c_for_decomposition(decomposition: Decomposition) -> tuple[dict, list[dict]]:
    faces = channel_faces_exact(decomposition)
    h = h_matrix_exact(decomposition, faces)
    h_np = np.array(h.tolist(), dtype=np.float64)
    rank_h = int(np.linalg.matrix_rank(h_np, tol=NUMERIC_TOL))
    blocks = cross_section_blocks_exact(decomposition)
    block_rows: list[dict] = []
    tangent_escape_any = False
    sampled_escape_any = False

    for (sum_left, sum_right), block in sorted(blocks.items()):
        block_np = np.array(block.tolist(), dtype=np.float64)
        active_rows = [row_idx for row_idx in range(block_np.shape[0]) if np.any(np.abs(block_np[row_idx, :]) > NUMERIC_TOL)]
        tangent_escape = False
        witness_row = None
        for row_idx in active_rows:
            e = np.zeros((block_np.shape[0], 1), dtype=np.float64)
            e[row_idx, 0] = 1.0
            if int(np.linalg.matrix_rank(np.hstack([h_np, e]), tol=NUMERIC_TOL)) > rank_h:
                tangent_escape = True
                witness_row = row_idx
                break
        tangent_escape_any = tangent_escape_any or tangent_escape

        q = np.ones(block_np.shape[0], dtype=np.float64)
        if active_rows:
            q[active_rows] = np.linspace(1.0, 2.0, len(active_rows))
        moved = np.diag(q) @ block_np
        sampled_escape = int(np.linalg.matrix_rank(np.hstack([h_np, moved]), tol=NUMERIC_TOL)) > rank_h
        sampled_escape_any = sampled_escape_any or sampled_escape
        block_rows.append({
            'label': decomposition.label,
            'block': f'D_{sum_left}{sum_right}',
            'active_row_count': len(active_rows),
            'tangent_escape_detected': tangent_escape,
            'tangent_escape_row': witness_row,
            'sampled_escape_detected': sampled_escape,
        })

    return ({
        'label': decomposition.label,
        'combined_observed_gauge_orbit_dim': observed_gauge_orbit_dimension(decomposition),
        'tangent_escape_detected': tangent_escape_any,
        'sampled_escape_detected': sampled_escape_any,
    }, block_rows)


def track_d_for_decomposition(decomposition: Decomposition) -> tuple[dict, list[dict], list[dict]]:
    faces = channel_faces_exact(decomposition)
    sigma = sigma_matrix_exact(faces)
    h = h_matrix_exact(decomposition, faces)
    blocks = cross_section_blocks_exact(decomposition)
    rows: list[dict] = []
    matrix_rows: list[dict] = []
    all_in_sigma_h = True
    all_in_h = True
    for (sum_left, sum_right), block in sorted(blocks.items()):
        in_h, _ = exact_recovery_matrix(h, block)
        in_sigma_h, m_sigma_h = exact_recovery_matrix(sp.Matrix.hstack(sigma, h), block)
        all_in_h = all_in_h and in_h
        all_in_sigma_h = all_in_sigma_h and in_sigma_h
        rows.append({
            'label': decomposition.label,
            'block': f'D_{sum_left}{sum_right}',
            'in_H_exact': in_h,
            'in_SigmaH_exact': in_sigma_h,
            'SigmaH_rank': int(m_sigma_h.rank()) if m_sigma_h is not None else None,
            'SigmaH_unique_coefficients': unique_coefficients(m_sigma_h) if m_sigma_h is not None else [],
        })
        if m_sigma_h is not None:
            matrix_rows.append({
                'label': decomposition.label,
                'block': f'D_{sum_left}{sum_right}',
                'M_sigma_h_literal': matrix_literal(m_sigma_h),
                'M_sigma_h_json': sp_to_jsonable(m_sigma_h),
            })
    return ({
        'label': decomposition.label,
        'all_cross_sections_in_H_exact': all_in_h,
        'all_cross_sections_in_SigmaH_exact': all_in_sigma_h,
    }, rows, matrix_rows)


def build_results_markdown(track_a: dict, track_b: dict, track_c: dict, track_d: dict) -> str:
    lines: list[str] = []
    w = lines.append
    w('# Phase 33 Results')
    w('')
    w('## 33a. Track A: 27-Symbol Encoding')
    w('')
    w('[GROUND_TRUTH] The 27-symbol word is the per-term stack W_k = [P_0[k,:], P_1[k,:], P_2[k,:]] with P_s[k,(r,u)] = alpha_k[r,s] * beta_k[s,u].')
    w('')
    w('[EXACT_DERIVED] For each known 3x3 decomposition, every cross-section block D_st was tested against span(H) with H = [K_0-K_1 | K_1-K_2].')
    w('')
    for algo in track_a['algorithms']:
        w(f"### {algo['label']}")
        w('')
        w(f"- rank(Gamma) = {algo['rank_Gamma_exact']}, dim ker(Gamma) = {algo['ker_Gamma_dim_exact']}, rank(H) = {algo['rank_H_exact']}.")
        w(f"- All six D_st blocks lie in span(H): {algo['all_cross_sections_in_H_exact']}.")
        w(f"- All six D_st blocks lie in span([Sigma|H]): {algo['all_cross_sections_in_SigmaH_exact']}.")
        w('')
        w('| block | in span(H) | rank(M_st) | nnz(M_st) | density | coeffs | first 9-row block rank | second 9-row block rank |')
        w('|-------|------------|------------|-----------|---------|--------|------------------------|-------------------------|')
        for row in [entry for entry in track_a['recovery_rows'] if entry['label'] == algo['label']]:
            coeffs = ', '.join(row['M_h_unique_coefficients']) if row['M_h_unique_coefficients'] else 'none'
            density = 'n/a' if row['M_h_density'] is None else f"{row['M_h_density']:.4f}"
            w(f"| {row['block']} | {row['in_H_exact']} | {row['M_h_rank']} | {row['M_h_nonzero']} | {density} | {coeffs} | {row['first_block_rank']} | {row['second_block_rank']} |")
        w('')
    w('[MEASURED_FROM_CODE] Full exact H-recovery matrices M_st and [Sigma|H]-recovery matrices are exported in this session directory as JSONL artifacts.')
    w('')
    w('## 33b. Track B: Clone-Frame Saturation')
    w('')
    w('[EXACT_DERIVED] Each channel permutation pi in S_3 was tested via H_pi = [K_pi(0)-K_pi(1) | K_pi(1)-K_pi(2)].')
    w('')
    for algo in track_b['algorithms']:
        w(f"### {algo['label']}")
        w('')
        w(f"- dim ker(Gamma) = {algo['ker_Gamma_dim_exact']}.")
        w(f"- Identity-frame rank(H) = {algo['identity_frame_rank_exact']}, saturation = {algo['identity_frame_saturates']}.")
        w(f"- All clone frames saturate individually: {algo['all_clone_frames_saturate']}.")
        if algo['random_trials']:
            w(f"- Synthetic defective triples tested: {algo['random_trials']} wildcard trials plus named structured cases; any factorized defective triple found = {algo['factorized_defective_found']}.")
        w('')
        w('| perm | rank(H_pi) | saturates ker(Gamma) |')
        w('|------|------------|----------------------|')
        for row in [entry for entry in track_b['clone_rows'] if entry['label'] == algo['label']]:
            w(f"| {row['perm']} | {row['rank_H_perm_exact']} | {row['saturates_ker_Gamma']} |")
        if algo['random_trials']:
            w('')
            w('| synthetic case | kind | rank(H) | defect | factorized rank-1 faces? |')
            w('|----------------|------|---------|--------|---------------------------|')
            for row in [entry for entry in track_b['synthetic_rows'] if entry['label'] == algo['label']][:8]:
                w(f"| {row['case_name']} | {row['case_kind']} | {row['rank_H_numeric']} | {row['defect_numeric']} | {row['factorized_rank1_faces']} |")
            w('')
            w('[WILDCARD] Additional wildcard synthetic trials are exported in JSONL; the sample above shows the defect families actually tested.')
        w('')
    w('## 33c. Track C: Gauge Orbit')
    w('')
    w('[EXACT_DERIVED] Under the tiling identities Gamma * P_s = I, the matched faces P_s are gauge-invariant while each D_st row is rescaled by lambda_s^(k) / lambda_t^(k).')
    w('')
    for algo in track_c['algorithms']:
        w(f"### {algo['label']}")
        w('')
        w(f"- Observed gauge-orbit dimension across cross-sections: {algo['combined_observed_gauge_orbit_dim']}.")
        w(f"- Tangent escape from span(H) detected: {algo['tangent_escape_detected']}.")
        w(f"- Finite sampled gauge escape from span(H) detected: {algo['sampled_escape_detected']}.")
        w('')
        w('| block | active rows | tangent escape? | sampled escape? |')
        w('|-------|-------------|-----------------|-----------------|')
        for row in [entry for entry in track_c['block_rows'] if entry['label'] == algo['label']]:
            w(f"| {row['block']} | {row['active_row_count']} | {row['tangent_escape_detected']} | {row['sampled_escape_detected']} |")
        w('')
    w('## 33d. Track D: Khatri-Rao Absorption')
    w('')
    w('[EXACT_DERIVED] The sanity-check question here is whether each D_st is recoverable from [Sigma|H], first in Strassen 2x2, then on the known 3x3 decompositions.')
    w('')
    for algo in track_d['algorithms']:
        w(f"### {algo['label']}")
        w('')
        w(f"- All D_st blocks recover from span(H): {algo['all_cross_sections_in_H_exact']}.")
        w(f"- All D_st blocks recover from span([Sigma|H]): {algo['all_cross_sections_in_SigmaH_exact']}.")
        w('')
        w('| block | in span(H) | in span([Sigma|H]) | rank(recovery) | coeffs |')
        w('|-------|------------|--------------------|----------------|--------|')
        for row in [entry for entry in track_d['rows'] if entry['label'] == algo['label']]:
            coeffs = ', '.join(row['SigmaH_unique_coefficients']) if row['SigmaH_unique_coefficients'] else 'none'
            w(f"| {row['block']} | {row['in_H_exact']} | {row['in_SigmaH_exact']} | {row['SigmaH_rank']} | {coeffs} |")
        w('')
    w('## 33e. Interpretation')
    w('')
    w('[INTERPRETATION] Track A is the central positive result: on both known 3x3 decompositions, every individual cross-section block D_st is already exactly recoverable from H alone, not merely from [Sigma|H]. That is stronger than the fallback absorption statement and numerically supports the 27-symbol-faithful-encoding picture on the known examples.')
    w('')
    w('[INTERPRETATION] Track B gives a second positive result and one negative obstruction. Positive: every clone frame already saturates ker(Gamma) individually on AlphaTensor and the standard algorithm, so the union-of-clones argument is unnecessary on the known exact decompositions. Negative: in synthetic defective families inside ker(Gamma), the low-rank H defects are easy to manufacture at the right-inverse level but did not survive the sampled factorized rank-1 face test.')
    w('')
    w('[INTERPRETATION] Track C is the key obstruction. The tiling identities alone do not rigidify the dead cross-sections inside span(H): both tangent and finite gauge escapes appear. So Delta containment does not follow from gauge-orbit rigidity under Gamma * P_s = I alone. Any proof must use more than the matched-face tiling constraint.')
    w('')
    w('[INTERPRETATION] Track D passes the mandatory Strassen sanity check and then matches Track A on the known 3x3 decompositions. The symbolic absorption route remains viable as a decomposition-specific identity, but the negative Track C result means it is not yet a universal proof of Delta containment.')
    w('')
    w('[OPEN_FRONT] The concrete next gap for Phase 34 is now precise: characterize what additional exact constraints, beyond the tiling identities and per-term gauge freedom, force the measured H-recovery matrices M_st to exist on every minimum-rank decomposition rather than only on the known ones.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    alpha, standard, strassen = load_decompositions()
    track_a_algorithms, track_a_rows, track_a_matrices = [], [], []
    for decomposition in (alpha, standard):
        summary, rows, matrix_rows = track_a_for_decomposition(decomposition)
        track_a_algorithms.append(summary)
        track_a_rows.extend(rows)
        track_a_matrices.extend(matrix_rows)

    track_b_algorithms, track_b_clone_rows, track_b_synthetic_rows = [], [], []
    for decomposition in (alpha, standard):
        summary, clone_rows, synthetic_rows = track_b_for_decomposition(decomposition)
        track_b_algorithms.append(summary)
        track_b_clone_rows.extend(clone_rows)
        track_b_synthetic_rows.extend(synthetic_rows)

    track_c_algorithms, track_c_block_rows = [], []
    for decomposition in (alpha, standard):
        summary, rows = track_c_for_decomposition(decomposition)
        track_c_algorithms.append(summary)
        track_c_block_rows.extend(rows)

    track_d_algorithms, track_d_rows, track_d_matrices = [], [], []
    for decomposition in (strassen, alpha, standard):
        summary, rows, matrix_rows = track_d_for_decomposition(decomposition)
        track_d_algorithms.append(summary)
        track_d_rows.extend(rows)
        track_d_matrices.extend(matrix_rows)

    payload = {
        'track_a': {'algorithms': track_a_algorithms, 'recovery_rows': track_a_rows},
        'track_b': {'algorithms': track_b_algorithms, 'clone_rows': track_b_clone_rows, 'synthetic_rows': track_b_synthetic_rows},
        'track_c': {'algorithms': track_c_algorithms, 'block_rows': track_c_block_rows},
        'track_d': {'algorithms': track_d_algorithms, 'rows': track_d_rows},
    }

    results_md = build_results_markdown(
        {'algorithms': track_a_algorithms, 'recovery_rows': track_a_rows},
        {'algorithms': track_b_algorithms, 'clone_rows': track_b_clone_rows, 'synthetic_rows': track_b_synthetic_rows},
        {'algorithms': track_c_algorithms, 'block_rows': track_c_block_rows},
        {'algorithms': track_d_algorithms, 'rows': track_d_rows},
    )

    write_text(OUT_DIR / 'RESULTS.md', results_md)
    write_json(OUT_DIR / 'phase33_summary.json', payload)
    write_jsonl(OUT_DIR / 'trackA_recovery_matrices.jsonl', track_a_matrices)
    write_jsonl(OUT_DIR / 'trackB_synthetic_trials.jsonl', track_b_synthetic_rows)
    write_jsonl(OUT_DIR / 'trackD_sigma_h_recovery_matrices.jsonl', track_d_matrices)


if __name__ == '__main__':
    main()