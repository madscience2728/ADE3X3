from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def flat_idx(row_idx: int, col_idx: int, n: int) -> int:
    return n * row_idx + col_idx


def matrix_multiplication_tensor(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.float64)
    for row_idx in range(n):
        for shared_idx in range(n):
            for col_idx in range(n):
                c_idx = flat_idx(row_idx, col_idx, n)
                a_idx = flat_idx(row_idx, shared_idx, n)
                b_idx = flat_idx(shared_idx, col_idx, n)
                tensor[c_idx, a_idx, b_idx] = 1.0
    return tensor


def outer_term(c_vec: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    return np.einsum('c,a,b->cab', c_vec, a_vec, b_vec, optimize=True)


def strassen_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_terms = np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ],
        dtype=np.float64,
    )
    b_terms = np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    c_terms = np.array(
        [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    return a_terms, b_terms, c_terms


def standard_2x2_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_terms = []
    b_terms = []
    c_terms = []
    for row_idx in range(2):
        for shared_idx in range(2):
            for col_idx in range(2):
                a_vec = np.zeros(4, dtype=np.float64)
                b_vec = np.zeros(4, dtype=np.float64)
                c_vec = np.zeros(4, dtype=np.float64)
                a_vec[flat_idx(row_idx, shared_idx, 2)] = 1.0
                b_vec[flat_idx(shared_idx, col_idx, 2)] = 1.0
                c_vec[flat_idx(row_idx, col_idx, 2)] = 1.0
                a_terms.append(a_vec)
                b_terms.append(b_vec)
                c_terms.append(c_vec)
    return np.stack(a_terms), np.stack(b_terms), np.stack(c_terms)


def anisotropy_row(a_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    a_matrix = a_vec.reshape(2, 2)
    b_matrix = b_vec.reshape(2, 2)
    coords = []
    for row_idx in range(2):
        for col_idx in range(2):
            coords.append(a_matrix[row_idx, 0] * b_matrix[0, col_idx] - a_matrix[row_idx, 1] * b_matrix[1, col_idx])
    return np.array(coords, dtype=np.float64)


def build_h_matrix(a_terms: np.ndarray, b_terms: np.ndarray) -> np.ndarray:
    return np.stack([anisotropy_row(a_vec, b_vec) for a_vec, b_vec in zip(a_terms, b_terms, strict=True)], axis=0)


def matrix_rank(matrix: np.ndarray, tol: float = 1e-10) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def primitive_integer_vector(vector: np.ndarray, tol: float = 1e-10) -> list[int]:
    rounded = np.rint(vector).astype(int)
    nonzero = [abs(int(value)) for value in rounded if abs(int(value)) > 0]
    gcd = 0
    for value in nonzero:
        gcd = value if gcd == 0 else int(np.gcd(gcd, value))
    gcd = gcd or 1
    primitive = [int(value // gcd) for value in rounded]
    for value in primitive:
        if value < 0:
            primitive = [-entry for entry in primitive]
            break
        if value > 0:
            break
    if np.linalg.norm(np.array(primitive, dtype=np.float64) - vector) > tol * max(1.0, np.linalg.norm(vector)):
        return [int(value) for value in rounded]
    return primitive


def null_vector(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return vt[rank:].T[:, 0]


def verify_decomposition(c_terms: np.ndarray, a_terms: np.ndarray, b_terms: np.ndarray, n: int) -> dict[str, float]:
    target = matrix_multiplication_tensor(n)
    total = np.zeros_like(target)
    for c_vec, a_vec, b_vec in zip(c_terms, a_terms, b_terms, strict=True):
        total += outer_term(c_vec, a_vec, b_vec)
    residual = total - target
    return {
        'loss': float(np.sum(residual * residual)),
        'max_abs': float(np.max(np.abs(residual))),
    }


def identity_formula(vector: list[int]) -> str:
    labels = ['eta[0,0]', 'eta[0,1]', 'eta[1,0]', 'eta[1,1]']
    pieces: list[str] = []
    for coeff, label in zip(vector, labels, strict=True):
        if coeff == 0:
            continue
        sign = '+' if coeff > 0 else '-'
        magnitude = abs(coeff)
        if not pieces:
            prefix = '' if coeff > 0 else '-'
        else:
            prefix = f' {sign} '
        term = label if magnitude == 1 else f'{magnitude}*{label}'
        pieces.append(f'{prefix}{term}')
    return ''.join(pieces) + ' = 0'


def main() -> None:
    strassen_a, strassen_b, strassen_c = strassen_terms()
    h_matrix = build_h_matrix(strassen_a, strassen_b)
    rank_h = matrix_rank(h_matrix)
    identity = null_vector(h_matrix)
    primitive = primitive_integer_vector(identity)
    verification = verify_decomposition(strassen_c, strassen_a, strassen_b, n=2)

    standard_a, standard_b, _ = standard_2x2_terms()
    standard_h = build_h_matrix(standard_a, standard_b)

    payload = {
        'tensor_shape': [4, 4, 4],
        'strassen_rank': 7,
        'rank_H2': rank_h,
        'nullity_k': int(h_matrix.shape[1] - rank_h),
        'conservation_sum': 7 + int(h_matrix.shape[1] - rank_h),
        'conservation_target': 8,
        'identity_vector': primitive,
        'identity_formula': identity_formula(primitive),
        'identity_values_per_term': (h_matrix @ np.array(primitive, dtype=np.float64)).tolist(),
        'h_matrix': h_matrix.tolist(),
        'verification': verification,
        'standard_2x2_rank_H2': matrix_rank(standard_h),
        'standard_2x2_nullity_k': int(standard_h.shape[1] - matrix_rank(standard_h)),
        'identity_interpretation': 'The single Strassen anisotropy identity is an affine checkerboard relation across the four eta[r,u] coordinates.',
    }
    (OUT_DIR / 'strassen_identity.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()