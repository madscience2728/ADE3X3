from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))


def flat_idx(row_idx: int, col_idx: int) -> int:
    return 3 * row_idx + col_idx


def signal_matrix(row_idx: int, col_idx: int) -> np.ndarray:
    matrix = np.zeros((9, 9), dtype=np.float64)
    for shared_idx in range(3):
        matrix[flat_idx(row_idx, shared_idx), flat_idx(shared_idx, col_idx)] = 1.0
    return matrix


def p_matrix(level_idx: int, row_idx: int, col_idx: int) -> np.ndarray:
    matrix = np.zeros((9, 9), dtype=np.float64)
    matrix[flat_idx(row_idx, level_idx), flat_idx(level_idx, col_idx)] = 1.0
    return matrix


def anisotropy_matrix_1(row_idx: int, col_idx: int) -> np.ndarray:
    return p_matrix(0, row_idx, col_idx) - p_matrix(1, row_idx, col_idx)


def anisotropy_matrix_2(row_idx: int, col_idx: int) -> np.ndarray:
    return p_matrix(1, row_idx, col_idx) - p_matrix(2, row_idx, col_idx)


def build_signal_matrices() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'S[{row_idx},{col_idx}]')
            matrices.append(signal_matrix(row_idx, col_idx))
    return labels, np.stack(matrices, axis=0)


def build_anisotropy_matrices() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'E1[{row_idx},{col_idx}]')
            matrices.append(anisotropy_matrix_1(row_idx, col_idx))
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'E2[{row_idx},{col_idx}]')
            matrices.append(anisotropy_matrix_2(row_idx, col_idx))
    return labels, np.stack(matrices, axis=0)


def evaluate_bilinear(matrix: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
    return float(alpha.reshape(-1) @ matrix @ beta.reshape(-1))


def matrix_rank(matrix: np.ndarray, tol: float = 1e-10) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def orthonormal_complement_rows(null_basis_rows: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(null_basis_rows, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    return vh[rank:]


def q_matrix_from_identity(identity_vector: np.ndarray, anisotropy_mats: np.ndarray) -> np.ndarray:
    return np.tensordot(identity_vector, anisotropy_mats, axes=(0, 0))


def candidate_vector_from_theta(theta: np.ndarray, complement_basis: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return theta @ complement_basis


def alpha_nullspace_and_complement() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from attack_common import eta_identity_to_q_matrix, phase5_identity_basis

    _, identity_vectors = phase5_identity_basis()
    null_basis = np.stack([np.array(vector, dtype=np.float64).reshape(-1) for vector in identity_vectors], axis=0)
    _, anisotropy_mats = build_anisotropy_matrices()
    alpha_q_matrices = np.stack(
        [np.array(eta_identity_to_q_matrix(vector), dtype=np.float64) for vector in identity_vectors],
        axis=0,
    )
    complement_basis = orthonormal_complement_rows(null_basis)
    complement_q_matrices = np.stack(
        [q_matrix_from_identity(vector, anisotropy_mats) for vector in complement_basis],
        axis=0,
    )
    return null_basis, complement_basis, alpha_q_matrices, anisotropy_mats, complement_q_matrices