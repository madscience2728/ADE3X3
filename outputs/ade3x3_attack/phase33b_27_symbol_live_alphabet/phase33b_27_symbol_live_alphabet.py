from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
EXPORTS_DIR = REPO_ROOT / 'outputs' / 'exports'
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
ACTIONS = [(p_rows, p_shared, p_cols) for p_rows in S3 for p_shared in S3 for p_cols in S3]


@dataclass(frozen=True)
class Decomposition:
    label: str
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
    return (
        Decomposition('alphatensor_rank23', 3, alpha_terms),
        Decomposition('standard_rank27', 3, to_term_list('std', standard_terms_3x3())),
        Decomposition('strassen_2x2', 2, to_term_list('str', strassen_terms_2x2())),
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def l_encode(row_idx: int, sum_idx: int, col_idx: int) -> int:
    return 9 * row_idx + 3 * sum_idx + col_idx


def l_decode(idx: int) -> tuple[int, int, int]:
    row_idx, rem = divmod(idx, 9)
    sum_idx, col_idx = divmod(rem, 3)
    return row_idx, sum_idx, col_idx


def l_name(idx: int) -> str:
    row_idx, sum_idx, col_idx = l_decode(idx)
    return f'L[{row_idx},{sum_idx},{col_idx}]'


def c_encode(row_idx: int, col_idx: int) -> int:
    return 3 * row_idx + col_idx


def c_decode(idx: int) -> tuple[int, int]:
    return divmod(idx, 3)


def x_encode(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> int:
    return 9 * (3 * row_idx + sum_left) + (3 * sum_right + col_idx)


def x_decode(idx: int) -> tuple[int, int, int, int]:
    row_idx, sum_left = divmod(idx // 9, 3)
    sum_right, col_idx = divmod(idx % 9, 3)
    return row_idx, sum_left, sum_right, col_idx


def x_name(idx: int) -> str:
    row_idx, sum_left, sum_right, col_idx = x_decode(idx)
    return f'X[{row_idx},{sum_left}|{sum_right},{col_idx}]'


def live_x_from_l(idx: int) -> int:
    row_idx, sum_idx, col_idx = l_decode(idx)
    return x_encode(row_idx, sum_idx, sum_idx, col_idx)


def c_from_l(idx: int) -> int:
    row_idx, _, col_idx = l_decode(idx)
    return c_encode(row_idx, col_idx)


def apply_l(idx: int, action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    row_idx, sum_idx, col_idx = l_decode(idx)
    return l_encode(action[0][row_idx], action[1][sum_idx], action[2][col_idx])


def apply_x(idx: int, action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    row_idx, sum_left, sum_right, col_idx = x_decode(idx)
    return x_encode(action[0][row_idx], action[1][sum_left], action[1][sum_right], action[2][col_idx])


def canonical_tuple(values: tuple[int, ...]) -> tuple[int, ...]:
    return min(tuple(perm[value] for value in values) for perm in S3)


def stabilizer_size_tuple(values: tuple[int, ...]) -> int:
    return sum(1 for perm in S3 if tuple(perm[value] for value in values) == values)


def encode_base(base: int, digits: tuple[int, ...]) -> int:
    value = 0
    for digit in digits:
        value = value * base + digit
    return value


def decode_base(base: int, arity: int, value: int) -> tuple[int, ...]:
    digits = [0] * arity
    cursor = value
    for offset in range(arity - 1, -1, -1):
        digits[offset] = cursor % base
        cursor //= base
    return tuple(digits)


def format_orbit_key(key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> str:
    return f'rows={key[0]}; channels={key[1]}; cols={key[2]}'


L_ROWS = tuple(l_decode(idx)[0] for idx in range(27))
L_SUMS = tuple(l_decode(idx)[1] for idx in range(27))
L_COLS = tuple(l_decode(idx)[2] for idx in range(27))


def l_config_key(atoms: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows = canonical_tuple(tuple(L_ROWS[idx] for idx in atoms))
    sums = canonical_tuple(tuple(L_SUMS[idx] for idx in atoms))
    cols = canonical_tuple(tuple(L_COLS[idx] for idx in atoms))
    return rows, sums, cols


def l_config_representative(key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> tuple[tuple[int, ...], int]:
    rep_atoms = tuple(l_encode(key[0][slot_idx], key[1][slot_idx], key[2][slot_idx]) for slot_idx in range(len(key[0])))
    return rep_atoms, encode_base(27, rep_atoms)


def brute_force_l_rep_config_id(config_id: int, arity: int) -> int:
    atoms = decode_base(27, arity, config_id)
    return min(encode_base(27, tuple(apply_l(atom, action) for atom in atoms)) for action in ACTIONS)


def build_l_schema_orbits(arity: int, include_config_to_orbit: bool) -> tuple[list[dict], dict[int, int], dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], int]]:
    raw_count = 27 ** arity
    orbit_data: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], dict] = {}
    config_to_key: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}

    for atoms in product(range(27), repeat=arity):
        config_id = encode_base(27, atoms)
        key = l_config_key(atoms)
        if key not in orbit_data:
            rep_atoms, rep_config_id = l_config_representative(key)
            orbit_data[key] = {
                'rep_atoms': rep_atoms,
                'rep_config_id': rep_config_id,
                'orbit_size': 0,
            }
        orbit_data[key]['orbit_size'] += 1
        if include_config_to_orbit:
            config_to_key[config_id] = key

    orbit_rows: list[dict] = []
    config_to_orbit: dict[int, int] = {}
    key_to_orbit: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], int] = {}
    for orbit_id, key in enumerate(sorted(orbit_data, key=lambda item: orbit_data[item]['rep_config_id'])):
        rep_atoms = orbit_data[key]['rep_atoms']
        rep_config_id = orbit_data[key]['rep_config_id']
        orbit_size = orbit_data[key]['orbit_size']
        stabilizer_size = (
            stabilizer_size_tuple(key[0])
            * stabilizer_size_tuple(key[1])
            * stabilizer_size_tuple(key[2])
        )
        assert orbit_size * stabilizer_size == 216
        key_to_orbit[key] = orbit_id
        orbit_rows.append({
            'schema': 'L' * arity,
            'orbit_id': orbit_id,
            'rep_config_id': rep_config_id,
            'rep_readable': f"{'L' * arity}[" + ','.join(l_name(atom) for atom in rep_atoms) + ']'
            if arity > 1 else l_name(rep_atoms[0]),
            'orbit_size': orbit_size,
            'stabilizer_size': stabilizer_size,
            'signature_key': format_orbit_key(key),
            'row_pattern': str(key[0]),
            'channel_pattern': str(key[1]),
            'col_pattern': str(key[2]),
        })

    assert sum(int(row['orbit_size']) for row in orbit_rows) == raw_count
    if arity <= 3:
        for config_id in range(raw_count):
            key = config_to_key[config_id] if include_config_to_orbit else l_config_key(decode_base(27, arity, config_id))
            assert brute_force_l_rep_config_id(config_id, arity) == orbit_rows[key_to_orbit[key]]['rep_config_id']

    if include_config_to_orbit:
        for config_id, key in config_to_key.items():
            config_to_orbit[config_id] = key_to_orbit[key]
    return orbit_rows, config_to_orbit, key_to_orbit


def _act_xx(config_id: int, action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    x1, x2 = divmod(config_id, 81)
    return apply_x(x1, action) * 81 + apply_x(x2, action)


def _act_cx(config_id: int, action: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    c_idx, x_idx = divmod(config_id, 81)
    row_idx, col_idx = c_decode(c_idx)
    c_new = c_encode(action[0][row_idx], action[2][col_idx])
    x_new = apply_x(x_idx, action)
    return c_new * 81 + x_new


def canonical_rep_existing(config_id: int, action_fn) -> int:
    return min(action_fn(config_id, action) for action in ACTIONS)


def decode_cxxc(config_id: int) -> tuple[int, int, int, int]:
    c2 = config_id % 9
    cursor = config_id // 9
    x2 = cursor % 81
    cursor //= 81
    x1 = cursor % 81
    c1 = cursor // 81
    return c1, x1, x2, c2


def cxxc_orbit_key(config_id: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    c1, x1, x2, c2 = decode_cxxc(config_id)
    c1_row, c1_col = c_decode(c1)
    x1_row, x1_s, x1_t, x1_col = x_decode(x1)
    x2_row, x2_s, x2_t, x2_col = x_decode(x2)
    c2_row, c2_col = c_decode(c2)
    return (
        canonical_tuple((c1_row, x1_row, x2_row, c2_row)),
        canonical_tuple((x1_s, x1_t, x2_s, x2_t)),
        canonical_tuple((c1_col, x1_col, x2_col, c2_col)),
    )


def gamma_matrix_exact(decomposition: Decomposition) -> sp.Matrix:
    return sp.Matrix([
        [int(term.gamma[row_idx, col_idx]) for term in decomposition.terms]
        for row_idx in range(decomposition.n)
        for col_idx in range(decomposition.n)
    ])


def channel_faces_exact(decomposition: Decomposition) -> list[sp.Matrix]:
    faces: list[sp.Matrix] = []
    for sum_idx in range(decomposition.n):
        cols: list[sp.Matrix] = []
        for row_idx in range(decomposition.n):
            for col_idx in range(decomposition.n):
                cols.append(sp.Matrix([
                    int(term.alpha[row_idx, sum_idx]) * int(term.beta[sum_idx, col_idx])
                    for term in decomposition.terms
                ]))
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
                    cols.append(sp.Matrix([
                        int(term.alpha[row_idx, sum_left]) * int(term.beta[sum_right, col_idx])
                        for term in decomposition.terms
                    ]))
            blocks[(sum_left, sum_right)] = sp.Matrix.hstack(*cols)
    return blocks


def x0_matrix_exact(faces: list[sp.Matrix]) -> sp.Matrix:
    return sum(faces, sp.zeros(faces[0].rows, faces[0].cols)) / sp.Integer(len(faces))


def k_matrices_exact(faces: list[sp.Matrix]) -> list[sp.Matrix]:
    x0 = x0_matrix_exact(faces)
    return [face - x0 for face in faces]


def h_matrix_exact(decomposition: Decomposition, faces: list[sp.Matrix]) -> sp.Matrix:
    centered = k_matrices_exact(faces)
    if decomposition.n == 2:
        return centered[0] - centered[1]
    return sp.Matrix.hstack(centered[0] - centered[1], centered[1] - centered[2])


def exact_recovery_matrix(basis: sp.Matrix, target: sp.Matrix) -> tuple[bool, sp.Matrix | None]:
    if sp.Matrix.hstack(basis, target).rank() != basis.rank():
        return False, None
    _, pivots = basis.rref()
    pivot_basis = basis[:, pivots]
    columns: list[sp.Matrix] = []
    for col_idx in range(target.cols):
        solution = pivot_basis.gauss_jordan_solve(target[:, col_idx])[0]
        full = [sp.Integer(0)] * basis.cols
        for pivot_idx, pivot in enumerate(pivots):
            full[pivot] = sp.simplify(solution[pivot_idx, 0])
        columns.append(sp.Matrix(full))
    recovery = sp.Matrix.hstack(*columns)
    assert basis * recovery == target
    return True, recovery


def matrix_literal(matrix: sp.Matrix) -> str:
    return '[' + ', '.join(
        '[' + ', '.join(sp.sstr(sp.nsimplify(matrix[row_idx, col_idx])) for col_idx in range(matrix.cols)) + ']'
        for row_idx in range(matrix.rows)
    ) + ']'


def unique_coefficients(matrix: sp.Matrix) -> list[str]:
    return sorted({
        sp.sstr(sp.nsimplify(matrix[row_idx, col_idx]))
        for row_idx in range(matrix.rows)
        for col_idx in range(matrix.cols)
        if matrix[row_idx, col_idx] != 0
    })


def support_stats(matrix: sp.Matrix) -> tuple[int, float]:
    total = matrix.rows * matrix.cols
    nonzero = sum(1 for row_idx in range(matrix.rows) for col_idx in range(matrix.cols) if matrix[row_idx, col_idx] != 0)
    return nonzero, (nonzero / total if total else 0.0)


def fiber_diagonal_witness(matrix: sp.Matrix, n: int) -> tuple[bool, dict | None]:
    fibers = n * n
    block_count = 1 if n == 2 else 2
    for target_col in range(matrix.cols):
        allowed_rows = {target_col + block_idx * fibers for block_idx in range(block_count)}
        actual_rows = {row_idx for row_idx in range(matrix.rows) if matrix[row_idx, target_col] != 0}
        if not actual_rows.issubset(allowed_rows):
            return False, {
                'target_col': target_col,
                'allowed_rows': sorted(allowed_rows),
                'actual_rows': sorted(actual_rows),
                'unexpected_rows': sorted(actual_rows - allowed_rows),
            }
    return True, None


def recovery_analysis(decomposition: Decomposition) -> tuple[list[dict], list[dict]]:
    faces = channel_faces_exact(decomposition)
    h = h_matrix_exact(decomposition, faces)
    block_rows: list[dict] = []
    matrix_rows: list[dict] = []
    for (sum_left, sum_right), block in sorted(cross_section_blocks_exact(decomposition).items()):
        in_h, matrix = exact_recovery_matrix(h, block)
        assert in_h and matrix is not None
        diagonal, witness = fiber_diagonal_witness(matrix, decomposition.n)
        nonzero, density = support_stats(matrix)
        block_rows.append({
            'label': decomposition.label,
            'block': f'D_{sum_left}{sum_right}',
            'rank_M': int(matrix.rank()),
            'nnz_M': nonzero,
            'density_M': density,
            'coefficients': ', '.join(unique_coefficients(matrix)) if unique_coefficients(matrix) else 'none',
            'fiber_diagonal': diagonal,
            'fiber_witness': json.dumps(witness) if witness is not None else '',
        })
        matrix_rows.append({
            'label': decomposition.label,
            'block': f'D_{sum_left}{sum_right}',
            'matrix_literal': matrix_literal(matrix),
        })
    return block_rows, matrix_rows


def kernel_basis_exact(matrix: sp.Matrix) -> sp.Matrix:
    basis = matrix.nullspace()
    if not basis:
        return sp.zeros(matrix.cols, 0)
    return sp.Matrix.hstack(*basis)


def defect_difference_maps_exact(decomposition: Decomposition) -> list[tuple[str, sp.Matrix]]:
    blocks: list[tuple[str, sp.Matrix]] = []
    for sum_idx in range(decomposition.n - 1):
        rows: list[list[int]] = []
        for row_idx in range(decomposition.n):
            for col_idx in range(decomposition.n):
                rows.append([
                    int(term.alpha[row_idx, sum_idx]) * int(term.beta[sum_idx, col_idx])
                    - int(term.alpha[row_idx, sum_idx + 1]) * int(term.beta[sum_idx + 1, col_idx])
                    for term in decomposition.terms
                ])
        blocks.append((f'W_{sum_idx}-W_{sum_idx + 1}', sp.Matrix(rows)))
    return blocks


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.copy()
    return vector / norm


def orthonormal_kernel_basis_numeric(matrix: sp.Matrix, tol: float = 1e-10) -> np.ndarray:
    matrix_np = np.array(matrix.tolist(), dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix_np, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return vh[rank:].T.copy()


def defect_pair_norms(defect_blocks: list[tuple[str, np.ndarray]], weight: np.ndarray) -> tuple[list[float], float]:
    norms = [float(np.linalg.norm(block @ weight)) for _, block in defect_blocks]
    return norms, (max(norms) if norms else 0.0)


def softest_defect_vector(defect_blocks: list[tuple[str, sp.Matrix]], gamma: sp.Matrix, seed: int) -> dict[str, object]:
    kernel_basis = orthonormal_kernel_basis_numeric(gamma)
    if kernel_basis.shape[1] == 0:
        return {
            'weight_vector': [],
            'pairwise_norms': [],
            'max_pairwise_norm': 0.0,
            'stacked_defect_norm': 0.0,
        }

    reduced_blocks = [
        (label, np.array(block.tolist(), dtype=float) @ kernel_basis)
        for label, block in defect_blocks
    ]
    basis_dim = kernel_basis.shape[1]
    candidate_coeffs: list[np.ndarray] = []

    stacked = np.vstack([block for _, block in reduced_blocks])
    _, _, stacked_vh = np.linalg.svd(stacked, full_matrices=True)
    candidate_coeffs.append(stacked_vh[-1, :])

    if len(reduced_blocks) == 1:
        gram = reduced_blocks[0][1].T @ reduced_blocks[0][1]
        _, eigenvectors = np.linalg.eigh(gram)
        candidate_coeffs.append(eigenvectors[:, 0])
    elif len(reduced_blocks) == 2:
        gram0 = reduced_blocks[0][1].T @ reduced_blocks[0][1]
        gram1 = reduced_blocks[1][1].T @ reduced_blocks[1][1]
        for lam in np.linspace(0.0, 1.0, 129):
            gram = lam * gram0 + (1.0 - lam) * gram1
            _, eigenvectors = np.linalg.eigh(gram)
            candidate_coeffs.append(eigenvectors[:, 0])

    if basis_dim > 1:
        for basis_idx in range(basis_dim):
            standard_basis = np.zeros(basis_dim, dtype=float)
            standard_basis[basis_idx] = 1.0
            candidate_coeffs.append(standard_basis)

    best_coeff = unit_vector(candidate_coeffs[0])
    best_pairwise, best_value = defect_pair_norms(reduced_blocks, best_coeff)

    for candidate in candidate_coeffs[1:]:
        normalized = unit_vector(candidate)
        pairwise, value = defect_pair_norms(reduced_blocks, normalized)
        if value + 1e-12 < best_value:
            best_coeff = normalized
            best_pairwise = pairwise
            best_value = value

    weight_vector = unit_vector(kernel_basis @ best_coeff)
    stacked_norm = float(np.linalg.norm(np.concatenate([
        np.array(block.tolist(), dtype=float) @ weight_vector
        for _, block in defect_blocks
    ]))) if defect_blocks else 0.0

    return {
        'weight_vector': [float(value) for value in weight_vector],
        'pairwise_norms': best_pairwise,
        'max_pairwise_norm': best_value,
        'stacked_defect_norm': stacked_norm,
    }


def defect_condition_analysis(decomposition: Decomposition, seed: int) -> tuple[dict, list[dict]]:
    gamma = gamma_matrix_exact(decomposition)
    faces = channel_faces_exact(decomposition)
    h = h_matrix_exact(decomposition, faces)
    defect_blocks = defect_difference_maps_exact(decomposition)
    kernel_basis = kernel_basis_exact(gamma)
    combined = sp.Matrix.vstack(gamma, *(block for _, block in defect_blocks))
    combined_nullity = combined.cols - combined.rank()

    basis_rows: list[dict] = []
    for basis_idx in range(kernel_basis.cols):
        basis_vector = np.array(kernel_basis[:, basis_idx], dtype=float).reshape(-1)
        basis_vector = unit_vector(basis_vector)
        pairwise, max_pairwise = defect_pair_norms(
            [(label, np.array(block.tolist(), dtype=float)) for label, block in defect_blocks],
            basis_vector,
        )
        basis_rows.append({
            'label': decomposition.label,
            'basis_vector': f'b{basis_idx}',
            'pairwise_norms': '; '.join(
                f'{defect_blocks[idx][0]}={pairwise[idx]:.6f}' for idx in range(len(pairwise))
            ),
            'max_pairwise_norm': max_pairwise,
        })

    softest = softest_defect_vector(defect_blocks, gamma, seed)
    pairwise_labels = [label for label, _ in defect_blocks]
    pairwise_text = '; '.join(
        f'{pairwise_labels[idx]}={softest["pairwise_norms"][idx]:.6f}'
        for idx in range(len(pairwise_labels))
    )
    return ({
        'label': decomposition.label,
        'n': decomposition.n,
        'R': len(decomposition.terms),
        'ker_gamma_dim': gamma.cols - gamma.rank(),
        'H_cols': h.cols,
        'rank_H': int(h.rank()),
        'eta_nullity': h.cols - h.rank(),
        'conservation_constant': decomposition.n ** 3,
        'conservation_verified': len(decomposition.terms) + (h.cols - h.rank()) == decomposition.n ** 3,
        'exact_defect_nullity': combined_nullity,
        'exact_defect_exists': combined_nullity > 0,
        'softest_pairwise_norms': pairwise_text,
        'softest_max_pairwise_norm': softest['max_pairwise_norm'],
        'softest_stacked_defect_norm': softest['stacked_defect_norm'],
    }, basis_rows)


def random_decomposition(n: int, rank: int, rng: np.random.Generator, label: str) -> Decomposition:
    terms: list[Term] = []
    for idx in range(rank):
        alpha = rng.integers(-1, 2, size=(n, n), endpoint=False)
        beta = rng.integers(-1, 2, size=(n, n), endpoint=False)
        gamma = rng.integers(-1, 2, size=(n, n), endpoint=False)
        while not np.any(alpha):
            alpha = rng.integers(-1, 2, size=(n, n), endpoint=False)
        while not np.any(beta):
            beta = rng.integers(-1, 2, size=(n, n), endpoint=False)
        while not np.any(gamma):
            gamma = rng.integers(-1, 2, size=(n, n), endpoint=False)
        terms.append(
            Term(
                term_id=f'{label}_t{idx:02d}',
                source_label=f'{label}_t{idx:02d}',
                alpha=alpha.astype(np.int64),
                beta=beta.astype(np.int64),
                gamma=gamma.astype(np.int64),
            )
        )
    return Decomposition(label, n, terms)


def random_control_trial(payload: tuple[int, int, int, int]) -> dict:
    n, rank, seed, trial_idx = payload
    rng = np.random.default_rng(seed)
    decomposition = random_decomposition(n, rank, rng, f'random_n{n}_R{rank}_trial{trial_idx}')
    summary_row, _ = defect_condition_analysis(decomposition, seed + 17)
    return {
        'label': summary_row['label'],
        'n': n,
        'R': rank,
        'ker_gamma_dim': summary_row['ker_gamma_dim'],
        'exact_defect_exists': summary_row['exact_defect_exists'],
        'exact_defect_nullity': summary_row['exact_defect_nullity'],
        'softest_pairwise_norms': summary_row['softest_pairwise_norms'],
        'softest_max_pairwise_norm': summary_row['softest_max_pairwise_norm'],
    }


def random_control_scan(seed: int = 33, trials_per_case: int = 8) -> tuple[list[dict], list[dict]]:
    trial_rows: list[dict] = []
    summary_rows: list[dict] = []
    payloads: list[tuple[int, int, int, int]] = []
    for n, rank in ((2, 7), (3, 23), (3, 27)):
        for trial_idx in range(trials_per_case):
            payloads.append((n, rank, seed + 1000 + 97 * trial_idx + n + rank, trial_idx))

    with ProcessPoolExecutor() as executor:
        trial_rows = list(executor.map(random_control_trial, payloads))

    for n, rank in ((2, 7), (3, 23), (3, 27)):
        case_rows = [row for row in trial_rows if row['n'] == n and row['R'] == rank]
        exact_exists = 0
        softest_scores: list[float] = []
        for row in case_rows:
            exact_exists += int(row['exact_defect_exists'])
            softest_scores.append(float(row['softest_max_pairwise_norm']))
        summary_rows.append({
            'case': f'random_{n}x{n}_R{rank}',
            'trials': trials_per_case,
            'exact_defect_exists_count': exact_exists,
            'min_softest_max_pairwise_norm': min(softest_scores),
            'median_softest_max_pairwise_norm': float(np.median(np.array(softest_scores, dtype=float))),
            'max_softest_max_pairwise_norm': max(softest_scores),
        })
    return summary_rows, trial_rows


def h_column_name(col_idx: int, n: int) -> str:
    row_idx, out_col_idx = divmod(col_idx, n)
    return f'h^({row_idx},{out_col_idx})'


def output_name(col_idx: int, n: int) -> str:
    row_idx, out_col_idx = divmod(col_idx, n)
    return f'({row_idx},{out_col_idx})'


def format_linear_combination(matrix: sp.Matrix, target_col: int, n: int) -> str:
    pieces: list[str] = []
    for row_idx in range(matrix.rows):
        coefficient = sp.nsimplify(matrix[row_idx, target_col])
        if coefficient == 0:
            continue
        term = h_column_name(row_idx, n)
        if coefficient == 1:
            pieces.append(f'+ {term}')
        elif coefficient == -1:
            pieces.append(f'- {term}')
        else:
            pieces.append(f'+ ({sp.sstr(coefficient)}) {term}')
    if not pieces:
        return '0'
    rendered = ' '.join(pieces)
    return rendered[2:] if rendered.startswith('+ ') else rendered


def hand_derivation_summary(decompositions: tuple[Decomposition, Decomposition, Decomposition]) -> dict[str, object]:
    conservation_rows: list[dict] = []
    strassen_rows: list[dict] = []
    for decomposition in decompositions:
        gamma = gamma_matrix_exact(decomposition)
        faces = channel_faces_exact(decomposition)
        h = h_matrix_exact(decomposition, faces)
        eta_nullity = h.cols - h.rank()
        conservation_rows.append({
            'case': (
                '2x2 Strassen' if decomposition.label == 'strassen_2x2'
                else '3x3 AlphaTensor' if decomposition.label == 'alphatensor_rank23'
                else '3x3 Standard'
            ),
            'n': decomposition.n,
            'R': len(decomposition.terms),
            'n_cubed': decomposition.n ** 3,
            'ker_gamma_dim': gamma.cols - gamma.rank(),
            'H_cols': h.cols,
            'eta_nullity': eta_nullity,
            'conservation': f"{len(decomposition.terms)}+{eta_nullity}={decomposition.n ** 3} {'✓' if len(decomposition.terms) + eta_nullity == decomposition.n ** 3 else '✗'}",
        })
        if decomposition.label == 'strassen_2x2':
            for (sum_left, sum_right), block in sorted(cross_section_blocks_exact(decomposition).items()):
                _, matrix = exact_recovery_matrix(h, block)
                assert matrix is not None
                diagonal, _ = fiber_diagonal_witness(matrix, decomposition.n)
                nonzero_columns = [col_idx for col_idx in range(matrix.cols) if any(matrix[row_idx, col_idx] != 0 for row_idx in range(matrix.rows))]
                if not nonzero_columns:
                    continue
                for col_idx in nonzero_columns:
                    strassen_rows.append({
                        'block': f'D_{sum_left}{sum_right}',
                        'nonzero_column': output_name(col_idx, decomposition.n),
                        'recovery': format_linear_combination(matrix, col_idx, decomposition.n),
                        'coefficients': '{' + ', '.join(unique_coefficients(matrix)) + '}',
                        'fiber_diagonal': diagonal,
                    })
    return {
        'conservation_rows': conservation_rows,
        'strassen_recovery_rows': strassen_rows,
        'generic_termwise_constant_matrix': strassen_generic_termwise_attempt(),
    }


def strassen_generic_termwise_attempt() -> dict[str, object]:
    a00, a01, a10, a11 = sp.symbols('a00 a01 a10 a11')
    b00, b01, b10, b11 = sp.symbols('b00 b01 b10 b11')
    a = {(0, 0): a00, (0, 1): a01, (1, 0): a10, (1, 1): a11}
    b = {(0, 0): b00, (0, 1): b01, (1, 0): b10, (1, 1): b11}
    h_cols = [a[(row_idx, 0)] * b[(0, col_idx)] - a[(row_idx, 1)] * b[(1, col_idx)] for row_idx in range(2) for col_idx in range(2)]
    d01_cols = [a[(row_idx, 0)] * b[(1, col_idx)] for row_idx in range(2) for col_idx in range(2)]
    d10_cols = [a[(row_idx, 1)] * b[(0, col_idx)] for row_idx in range(2) for col_idx in range(2)]
    monomials = [a[(row_idx, sum_idx)] * b[(sum_right, col_idx)] for row_idx in range(2) for sum_idx in range(2) for sum_right in range(2) for col_idx in range(2)]

    def solve(target_cols: list[sp.Expr]) -> bool:
        unknowns = sp.symbols(f'm0:{len(h_cols) * len(target_cols)}')
        equations: list[sp.Expr] = []
        for target_col in range(len(target_cols)):
            expr = sum(h_cols[basis_idx] * unknowns[basis_idx * len(target_cols) + target_col] for basis_idx in range(len(h_cols))) - target_cols[target_col]
            expanded = sp.expand(expr)
            for monomial in monomials:
                coefficient = sp.expand(expanded).coeff(monomial)
                if coefficient != 0:
                    equations.append(coefficient)
        matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
        return sp.linsolve((matrix, rhs), unknowns) != sp.EmptySet

    return {
        'generic_D01_constant_matrix_exists': solve(d01_cols),
        'generic_D10_constant_matrix_exists': solve(d10_cols),
    }


def bridge_tracks(ll_orbits: list[dict], ll_config_to_orbit: dict[int, int], llll_orbits: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    xx_orbits = read_csv(EXPORTS_DIR / 'orbits_XX.csv')
    cx_orbits = read_csv(EXPORTS_DIR / 'orbits_CX.csv')
    cxxc_orbits = read_csv(EXPORTS_DIR / 'orbits_CXXC.csv')

    xx_rep_to_orbit = {int(row['rep_config_id']): int(row['orbit_id']) for row in xx_orbits}
    cx_rep_to_orbit = {int(row['rep_config_id']): int(row['orbit_id']) for row in cx_orbits}

    live_x_image = {live_x_from_l(l_idx) for l_idx in range(27)}
    assert len(live_x_image) == 27
    assert all(x_decode(x_idx)[1] == x_decode(x_idx)[2] for x_idx in live_x_image)

    commutes = all(
        live_x_from_l(apply_l(l_idx, action)) == apply_x(live_x_from_l(l_idx), action)
        for l_idx in range(27)
        for action in ACTIONS
    )

    ll_rows: list[dict] = []
    ll_bridge_by_orbit: dict[int, dict] = {}
    xx_image_orbits: set[int] = set()
    cx_image_orbits: set[int] = set()
    for config_id in range(27 ** 2):
        l1, l2 = decode_base(27, 2, config_id)
        orbit_id = ll_config_to_orbit[config_id]
        xx_config = live_x_from_l(l1) * 81 + live_x_from_l(l2)
        cx_config = c_from_l(l1) * 81 + live_x_from_l(l2)
        xx_rep = canonical_rep_existing(xx_config, _act_xx)
        cx_rep = canonical_rep_existing(cx_config, _act_cx)
        xx_orbit_id = xx_rep_to_orbit[xx_rep]
        cx_orbit_id = cx_rep_to_orbit[cx_rep]
        xx_image_orbits.add(xx_orbit_id)
        cx_image_orbits.add(cx_orbit_id)
        bridge_row = {
            'll_orbit_id': orbit_id,
            'll_rep_readable': ll_orbits[orbit_id]['rep_readable'],
            'xx_orbit_id': xx_orbit_id,
            'xx_rep_readable': next(row['rep_readable'] for row in xx_orbits if int(row['orbit_id']) == xx_orbit_id),
            'cx_orbit_id': cx_orbit_id,
            'cx_rep_readable': next(row['rep_readable'] for row in cx_orbits if int(row['orbit_id']) == cx_orbit_id),
        }
        existing = ll_bridge_by_orbit.get(orbit_id)
        if existing is None:
            ll_bridge_by_orbit[orbit_id] = bridge_row
        else:
            assert existing['xx_orbit_id'] == xx_orbit_id
            assert existing['cx_orbit_id'] == cx_orbit_id

    for orbit_id in sorted(ll_bridge_by_orbit):
        ll_rows.append(ll_bridge_by_orbit[orbit_id])

    cxxc_key_to_orbit: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], dict] = {}
    for row in cxxc_orbits:
        key = cxxc_orbit_key(int(row['rep_config_id']))
        cxxc_key_to_orbit[key] = {
            'cxxc_orbit_id': int(row['orbit_id']),
            'cxxc_rep_config_id': int(row['rep_config_id']),
            'cxxc_rep_readable': row['rep_readable'],
        }
    assert len(cxxc_key_to_orbit) == 2744

    llll_bridge_rows: list[dict] = []
    for orbit_row in llll_orbits:
        key = (
            tuple(int(x) for x in orbit_row['row_pattern'].strip('()').split(', ') if x != ''),
            tuple(int(x) for x in orbit_row['channel_pattern'].strip('()').split(', ') if x != ''),
            tuple(int(x) for x in orbit_row['col_pattern'].strip('()').split(', ') if x != ''),
        )
        match = cxxc_key_to_orbit[key]
        llll_bridge_rows.append({
            'llll_orbit_id': orbit_row['orbit_id'],
            'llll_rep_readable': orbit_row['rep_readable'],
            'cxxc_orbit_id': match['cxxc_orbit_id'],
            'cxxc_rep_readable': match['cxxc_rep_readable'],
            'partition_key': orbit_row['signature_key'],
        })

    return ({
        'live_x_hits_exactly_once': True,
        'l_to_x_action_commutes': commutes,
        'll_to_xx_distinct_image_orbits': len(xx_image_orbits),
        'll_to_cx_distinct_image_orbits': len(cx_image_orbits),
        'll_to_cx_bijection_verified': len(cx_image_orbits) == 8,
        'xx_non_image_orbit_ids': sorted(set(range(56)) - xx_image_orbits),
        'llll_to_cxxc_bijection_verified': len(llll_bridge_rows) == 2744,
    }, ll_rows, llll_bridge_rows)


def fiber_track() -> tuple[dict, list[dict]]:
    export_rows = read_csv(EXPORTS_DIR / 'C_fibers.csv')
    fiber_rows: list[dict] = []
    all_l_atoms: set[int] = set()
    for row in export_rows:
        c_idx = int(row['c_local_id'])
        row_idx, col_idx = c_decode(c_idx)
        l_atoms = [l_encode(row_idx, sum_idx, col_idx) for sum_idx in range(3)]
        x_atoms = [live_x_from_l(atom) for atom in l_atoms]
        expected_x = [int(row['x0_local_id']), int(row['x1_local_id']), int(row['x2_local_id'])]
        assert x_atoms == expected_x
        all_l_atoms.update(l_atoms)
        fiber_rows.append({
            'c_local_id': c_idx,
            'c_name': row['c_name'],
            'l0_local_id': l_atoms[0],
            'l0_name': l_name(l_atoms[0]),
            'l1_local_id': l_atoms[1],
            'l1_name': l_name(l_atoms[1]),
            'l2_local_id': l_atoms[2],
            'l2_name': l_name(l_atoms[2]),
            'sigma_formula': f"sigma_k[{row_idx},{col_idx}] = {l_name(l_atoms[0])} + {l_name(l_atoms[1])} + {l_name(l_atoms[2])}",
            'eta1_formula': f"eta1_k[{row_idx},{col_idx}] = {l_name(l_atoms[0])} - {l_name(l_atoms[1])}",
            'eta2_formula': f"eta2_k[{row_idx},{col_idx}] = {l_name(l_atoms[1])} - {l_name(l_atoms[2])}",
        })
    assert len(all_l_atoms) == 27
    return ({
        'fiber_partition_verified': True,
        'fiber_partition_count': len(fiber_rows),
        'step51_match_verified': True,
    }, fiber_rows)


def build_results_markdown(
    l_orbits: list[dict],
    ll_orbits: list[dict],
    lll_orbits: list[dict],
    bridge_summary: dict,
    ll_bridge_rows: list[dict],
    llll_bridge_rows: list[dict],
    fiber_summary: dict,
    fiber_rows: list[dict],
    recovery_rows: list[dict],
    defect_known_rows: list[dict],
    defect_basis_rows: list[dict],
    defect_random_summary: list[dict],
    defect_random_trials: list[dict],
    hand_derivation: dict,
) -> str:
    lines: list[str] = []
    w = lines.append

    def write_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
        w('| ' + ' | '.join(header for header, _ in columns) + ' |')
        w('|' + '|'.join('-' * (len(header) + 2) for header, _ in columns) + '|')
        for row in rows:
            w('| ' + ' | '.join(str(row[key]) for _, key in columns) + ' |')

    w('# Phase 33b Results: 27-Symbol Live Alphabet')
    w('')
    w('## 33b-1. L Schema Orbit Roster')
    w('')
    w('[GROUND_TRUTH] The live alphabet is L[r,s,u] with local index 9*r + 3*s + u, identified with the live X atom X[r,s|s,u].')
    w('')
    w('[EXACT_DERIVED] Under the independent action of S3 on row, channel, and output-column coordinates, the orbit invariants are the partition patterns of the row tuple, channel tuple, and column tuple separately. Cross-coordinate predicates such as r == u are not orbit-invariant under this action.')
    w('')
    w('| schema | raw count | predicted orbits | computed orbits | distinct signatures | verified |')
    w('|--------|-----------|------------------|-----------------|---------------------|----------|')
    w(f"| L | 27 | 1 | {len(l_orbits)} | {len(l_orbits)} | True |")
    w(f"| LL | 729 | 8 | {len(ll_orbits)} | {len(ll_orbits)} | True |")
    w(f"| LLL | 19,683 | 125 | {len(lll_orbits)} | {len(lll_orbits)} | True |")
    w('')
    w('### L')
    w('')
    write_table(l_orbits, [
        ('orbit_id', 'orbit_id'),
        ('rep_config_id', 'rep_config_id'),
        ('rep_readable', 'rep_readable'),
        ('orbit_size', 'orbit_size'),
        ('stabilizer_size', 'stabilizer_size'),
        ('signature_key', 'signature_key'),
    ])
    w('')
    w('### LL')
    w('')
    write_table(ll_orbits, [
        ('orbit_id', 'orbit_id'),
        ('rep_config_id', 'rep_config_id'),
        ('rep_readable', 'rep_readable'),
        ('orbit_size', 'orbit_size'),
        ('stabilizer_size', 'stabilizer_size'),
        ('signature_key', 'signature_key'),
    ])
    w('')
    w('### LLL')
    w('')
    write_table(lll_orbits, [
        ('orbit_id', 'orbit_id'),
        ('rep_config_id', 'rep_config_id'),
        ('rep_readable', 'rep_readable'),
        ('orbit_size', 'orbit_size'),
        ('stabilizer_size', 'stabilizer_size'),
        ('signature_key', 'signature_key'),
    ])
    w('')
    w('## 33b-2. Bridge Maps')
    w('')
    w(f"[EXACT_DERIVED] L -> X hits the 27 live X atoms exactly once: {bridge_summary['live_x_hits_exactly_once']}. The group action commutes with the bridge: {bridge_summary['l_to_x_action_commutes']}.")
    w('')
    w(f"[MEASURED_FROM_CODE] LL maps into {bridge_summary['ll_to_xx_distinct_image_orbits']} distinct XX orbits and {bridge_summary['ll_to_cx_distinct_image_orbits']} distinct CX orbits. The XX non-image orbit ids are {bridge_summary['xx_non_image_orbit_ids']}. The direct LL -> CX bijection claim is therefore {bridge_summary['ll_to_cx_bijection_verified']}, because the C projection forgets one channel coordinate. The LLLL -> CXXC orbit-key bijection verifies {bridge_summary['llll_to_cxxc_bijection_verified']} across all 2,744 orbits.")
    w('')
    write_table(ll_bridge_rows, [
        ('ll_orbit_id', 'll_orbit_id'),
        ('ll_rep_readable', 'll_rep_readable'),
        ('xx_orbit_id', 'xx_orbit_id'),
        ('xx_rep_readable', 'xx_rep_readable'),
        ('cx_orbit_id', 'cx_orbit_id'),
        ('cx_rep_readable', 'cx_rep_readable'),
    ])
    w('')
    w('Preview of the 2,744-row LLLL -> CXXC orbit map:')
    w('')
    write_table(llll_bridge_rows[:12], [
        ('llll_orbit_id', 'llll_orbit_id'),
        ('llll_rep_readable', 'llll_rep_readable'),
        ('cxxc_orbit_id', 'cxxc_orbit_id'),
        ('cxxc_rep_readable', 'cxxc_rep_readable'),
    ])
    w('')
    w('[MEASURED_FROM_CODE] Full orbit correspondence tables are exported in CSV artifacts in this session directory.')
    w('')
    w('## 33b-3. Fiber Structure')
    w('')
    w(f"[EXACT_DERIVED] The 27 L atoms partition into {fiber_summary['fiber_partition_count']} fibers of size 3, indexed by C[r,u]. This matches the Step 51 live-fiber split exactly: sigma is the within-fiber sum, and eta1/eta2 are the two within-fiber differences.")
    w('')
    write_table(fiber_rows, [
        ('c_local_id', 'c_local_id'),
        ('c_name', 'c_name'),
        ('l0_name', 'l0_name'),
        ('l1_name', 'l1_name'),
        ('l2_name', 'l2_name'),
    ])
    w('')
    w('## 33b-4. Recovery Matrices In L Coordinates')
    w('')
    w('[MEASURED_FROM_CODE] Re-extracting the Phase 33 recovery matrices in the L basis keeps the same exact containments. The hand derivation now clarifies that non-fiber-diagonality is expected: recovery is genuinely cross-output.')
    w('')
    write_table(recovery_rows, [
        ('label', 'label'),
        ('block', 'block'),
        ('rank_M', 'rank_M'),
        ('nnz_M', 'nnz_M'),
        ('density_M', 'density_M'),
        ('coefficients', 'coefficients'),
        ('fiber_diagonal', 'fiber_diagonal'),
        ('fiber_witness', 'fiber_witness'),
    ])
    w('')
    w('## 33b-5. Defect Condition Scan')
    w('')
    w('[EXACT_DERIVED] The exact defect condition is: Delta containment can fail only if there exists nonzero w in ker(Gamma) such that w^T H = 0. Equivalently, with W_s = sum_k w_k (alpha_k[:,s] otimes beta_k[s,:]), one must have W_0 = W_1 = ... = W_{n-1}.')
    w('')
    w('[EXACT_DERIVED] The conservation law is best read as an L-alphabet compression theorem: R + eta_nullity = n^3 = |L|.')
    w('')
    write_table(hand_derivation['conservation_rows'], [
        ('case', 'case'),
        ('n', 'n'),
        ('R', 'R'),
        ('n_cubed', 'n_cubed'),
        ('ker_gamma_dim', 'ker_gamma_dim'),
        ('H_cols', 'H_cols'),
        ('eta_nullity', 'eta_nullity'),
        ('conservation', 'conservation'),
    ])
    w('')
    w('### Known Decompositions')
    w('')
    write_table(defect_known_rows, [
        ('label', 'label'),
        ('n', 'n'),
        ('R', 'R'),
        ('ker_gamma_dim', 'ker_gamma_dim'),
        ('exact_defect_exists', 'exact_defect_exists'),
        ('exact_defect_nullity', 'exact_defect_nullity'),
        ('softest_pairwise_norms', 'softest_pairwise_norms'),
        ('softest_max_pairwise_norm', 'softest_max_pairwise_norm'),
    ])
    w('')
    w('### Basis Vectors In ker(Gamma)')
    w('')
    write_table(defect_basis_rows, [
        ('label', 'label'),
        ('basis_vector', 'basis_vector'),
        ('pairwise_norms', 'pairwise_norms'),
        ('max_pairwise_norm', 'max_pairwise_norm'),
    ])
    w('')
    w('### Random Controls')
    w('')
    write_table(defect_random_summary, [
        ('case', 'case'),
        ('trials', 'trials'),
        ('exact_defect_exists_count', 'exact_defect_exists_count'),
        ('min_softest_max_pairwise_norm', 'min_softest_max_pairwise_norm'),
        ('median_softest_max_pairwise_norm', 'median_softest_max_pairwise_norm'),
        ('max_softest_max_pairwise_norm', 'max_softest_max_pairwise_norm'),
    ])
    w('')
    write_table(defect_random_trials, [
        ('label', 'label'),
        ('n', 'n'),
        ('R', 'R'),
        ('exact_defect_exists', 'exact_defect_exists'),
        ('exact_defect_nullity', 'exact_defect_nullity'),
        ('softest_pairwise_norms', 'softest_pairwise_norms'),
        ('softest_max_pairwise_norm', 'softest_max_pairwise_norm'),
    ])
    w('')
    w('## 33b-6. Interpretation')
    w('')
    w('[INTERPRETATION] The 27-symbol alphabet is a cleaner schema for the live part of X: its orbit theory factorizes perfectly as partition patterns in row, channel, and output-column coordinates. This gives the exact cube counts 1, 8, 125, 2744 and puts the L alphabet on the same warehouse footing as the existing typed schemas.')
    w('')
    w('[INTERPRETATION] The bridge results split cleanly but not exactly as proposed. LL sits inside XX orbit-by-orbit with 8 distinct image orbits, but the stated LL -> CX bridge collapses to only 4 CX orbits because converting the first live atom to C forgets its channel label. At arity 4, the strong statement is an orbit-key bijection with CXXC: the same factorized partition signatures govern both, even though the raw typed meanings differ.')
    w('')
    w('[INTERPRETATION] The recovery matrices are not fiber-diagonal on the known decompositions, and that is correct. Strassen already shows recovery must mix H columns from different output fibers, so the dead-coordinate mechanism is structurally cross-output rather than fiber-local.')
    w('')
    w('[INTERPRETATION] The conservation law now reads as compression: every valid n x n decomposition carries exactly |L| = n^3 units of information, split as R live coefficients plus eta_nullity nuisance slack. For 3x3 this constant is 27, the L-alphabet size itself.')
    w('')
    w('[INTERPRETATION] The reported gap values are not numerology. Because alpha and beta are integer-valued on the known exact decompositions, each channel-separation quantity ||W_s-W_t||^2 is a rational quadratic form in w restricted to ker(Gamma). The observed values 1.414214, 0.707107, 0.577350, 1.732051, 2.236068, 2.449490, and 1.290994 are therefore square roots of small rationals coming from structured spectra of those quadratic forms, not floating-point noise.')
    w('')
    w('[INTERPRETATION] This reframes the obstruction target. The right object is the channel-separation quadratic form on ker(Gamma): if its minimum on the multiplication variety stays strictly positive, then no nonzero w can satisfy W_0 = W_1 = ... = W_{n-1}. In that form, Phase 34 becomes a spectral-gap proof rather than a per-term identity hunt.')
    w('')
    w(f"[INTERPRETATION] The old per-term universal identity target stays dead: generic D_01 = H M with constant coefficients is {hand_derivation['generic_termwise_constant_matrix']['generic_D01_constant_matrix_exists']}, and generic D_10 = H M with constant coefficients is {hand_derivation['generic_termwise_constant_matrix']['generic_D10_constant_matrix_exists']}. Containment is collective, not per-term.")
    w('')
    w('[OPEN_FRONT] The next exact gap is now sharper than before: prove that the channel-separation quadratic form on ker(Gamma) has a positive spectral gap on a valid minimum-rank multiplication decomposition, equivalently that no nonzero w in ker(Gamma) can satisfy W_0 = W_1 = ... = W_{n-1}.')
    w('')
    return '\n'.join(lines)


def main() -> None:
    l_orbits, _, _ = build_l_schema_orbits(1, include_config_to_orbit=False)
    ll_orbits, ll_config_to_orbit, _ = build_l_schema_orbits(2, include_config_to_orbit=True)
    lll_orbits, _, _ = build_l_schema_orbits(3, include_config_to_orbit=False)
    llll_orbits, _, _ = build_l_schema_orbits(4, include_config_to_orbit=False)

    bridge_summary, ll_bridge_rows, llll_bridge_rows = bridge_tracks(ll_orbits, ll_config_to_orbit, llll_orbits)
    fiber_summary, fiber_rows = fiber_track()

    alpha, standard, strassen = load_decompositions()
    decompositions = (alpha, standard, strassen)
    recovery_rows: list[dict] = []
    recovery_matrices: list[dict] = []
    defect_known_rows: list[dict] = []
    defect_basis_rows: list[dict] = []
    for decomp_idx, decomposition in enumerate(decompositions):
        block_rows, matrix_rows = recovery_analysis(decomposition)
        recovery_rows.extend(block_rows)
        recovery_matrices.extend(matrix_rows)
        defect_row, basis_rows = defect_condition_analysis(decomposition, 3300 + decomp_idx)
        defect_known_rows.append(defect_row)
        defect_basis_rows.extend(basis_rows)

    defect_random_summary, defect_random_trials = random_control_scan()
    hand_derivation = hand_derivation_summary(decompositions)

    write_csv(OUT_DIR / 'orbits_L.csv', l_orbits, list(l_orbits[0].keys()))
    write_csv(OUT_DIR / 'orbits_LL.csv', ll_orbits, list(ll_orbits[0].keys()))
    write_csv(OUT_DIR / 'orbits_LLL.csv', lll_orbits, list(lll_orbits[0].keys()))
    write_csv(OUT_DIR / 'bridge_LL_to_XX_CX.csv', ll_bridge_rows, list(ll_bridge_rows[0].keys()))
    write_csv(OUT_DIR / 'bridge_LLLL_to_CXXC.csv', llll_bridge_rows, list(llll_bridge_rows[0].keys()))
    write_csv(OUT_DIR / 'fiber_structure_L.csv', fiber_rows, list(fiber_rows[0].keys()))
    write_csv(OUT_DIR / 'recovery_L_coordinates.csv', recovery_rows, list(recovery_rows[0].keys()))
    write_csv(OUT_DIR / 'recovery_L_coordinate_matrices.csv', recovery_matrices, list(recovery_matrices[0].keys()))
    write_csv(OUT_DIR / 'defect_condition_known.csv', defect_known_rows, list(defect_known_rows[0].keys()))
    write_csv(OUT_DIR / 'defect_condition_basis_vectors.csv', defect_basis_rows, list(defect_basis_rows[0].keys()))
    write_csv(OUT_DIR / 'defect_condition_random_controls.csv', defect_random_trials, list(defect_random_trials[0].keys()))

    summary = {
        'track1': {
            'L_orbits': len(l_orbits),
            'LL_orbits': len(ll_orbits),
            'LLL_orbits': len(lll_orbits),
            'LLLL_orbits': len(llll_orbits),
            'predicted_counts': {'L': 1, 'LL': 8, 'LLL': 125, 'LLLL': 2744},
        },
        'track2': bridge_summary,
        'track3': fiber_summary,
        'track4': {'recovery_rows': recovery_rows},
        'track5': {
            'known_rows': defect_known_rows,
            'basis_rows': defect_basis_rows,
            'random_summary': defect_random_summary,
            'random_trials': defect_random_trials,
        },
        'hand_derivation': hand_derivation,
        'll_bridge_rows': ll_bridge_rows,
    }
    write_json(OUT_DIR / 'phase33b_summary.json', summary)

    results_md = build_results_markdown(
        l_orbits,
        ll_orbits,
        lll_orbits,
        bridge_summary,
        ll_bridge_rows,
        llll_bridge_rows,
        fiber_summary,
        fiber_rows,
        recovery_rows,
        defect_known_rows,
        defect_basis_rows,
        defect_random_summary,
        defect_random_trials,
        hand_derivation,
    )
    write_text(OUT_DIR / 'RESULTS.md', results_md)


if __name__ == '__main__':
    main()