from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import eta_coordinate_labels, eta_identity_to_q_matrix, load_public_terms, phase5_identity_basis  # noqa: E402


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


def build_signal_matrices() -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for row_idx in range(3):
        for col_idx in range(3):
            labels.append(f'S[{row_idx},{col_idx}]')
            matrices.append(signal_matrix(row_idx, col_idx))
    return labels, matrices


def build_anisotropy_matrices() -> tuple[list[str], list[np.ndarray]]:
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
    return labels, matrices


def build_p_matrices() -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for level_idx in range(3):
        for row_idx in range(3):
            for col_idx in range(3):
                labels.append(f'P{level_idx}[{row_idx},{col_idx}]')
                matrices.append(p_matrix(level_idx, row_idx, col_idx))
    return labels, matrices


def evaluate_bilinear(matrix: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
    return float(alpha.reshape(-1) @ matrix @ beta.reshape(-1))


def alpha_nullspace_and_complement() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    _, identity_vectors = phase5_identity_basis()
    null_basis = np.column_stack([np.array(vector, dtype=np.float64).reshape(-1) for vector in identity_vectors])
    alpha_q_matrices = [np.array(eta_identity_to_q_matrix(vector), dtype=np.float64) for vector in identity_vectors]
    _, _, vh = np.linalg.svd(null_basis.T, full_matrices=True)
    complement_basis = vh[null_basis.shape[1]:].T
    anisotropy_labels, anisotropy_mats = build_anisotropy_matrices()
    complement_q_matrices: list[np.ndarray] = []
    for col_idx in range(complement_basis.shape[1]):
        q_matrix = np.zeros((9, 9), dtype=np.float64)
        for basis_idx, basis_coeff in enumerate(complement_basis[:, col_idx]):
            q_matrix += basis_coeff * anisotropy_mats[basis_idx]
        complement_q_matrices.append(q_matrix)
    return null_basis, complement_basis, alpha_q_matrices, np.array(anisotropy_mats), complement_q_matrices


def matrix_payload(labels: list[str], matrices: list[np.ndarray]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for label, matrix in zip(labels, matrices, strict=True):
        payload.append({'label': label, 'matrix': matrix.tolist()})
    return payload


def main() -> None:
    signal_labels, signal_mats = build_signal_matrices()
    anisotropy_labels, anisotropy_mats = build_anisotropy_matrices()
    p_labels, p_mats = build_p_matrices()
    null_basis, complement_basis, alpha_q_matrices, _, complement_q_matrices = alpha_nullspace_and_complement()

    relation_errors = []
    for row_idx in range(3):
        for col_idx in range(3):
            s = signal_matrix(row_idx, col_idx)
            e1 = anisotropy_matrix_1(row_idx, col_idx)
            e2 = anisotropy_matrix_2(row_idx, col_idx)
            p0 = (s + 2.0 * e1 + e2) / 3.0
            p1 = (s - e1 + e2) / 3.0
            p2 = (s - e1 - 2.0 * e2) / 3.0
            relation_errors.append(
                {
                    'fiber': f'({row_idx},{col_idx})',
                    'p0_residual': float(np.max(np.abs(p0 - p_matrix(0, row_idx, col_idx)))),
                    'p1_residual': float(np.max(np.abs(p1 - p_matrix(1, row_idx, col_idx)))),
                    'p2_residual': float(np.max(np.abs(p2 - p_matrix(2, row_idx, col_idx)))),
                }
            )

    terms, _, _ = load_public_terms()
    eta_labels = eta_coordinate_labels()
    eta_verification_rows: list[dict[str, object]] = []
    for term in terms:
        alpha_vec = term.alpha.reshape(-1)
        beta_vec = term.beta.reshape(-1)
        for label, matrix in zip(anisotropy_labels, anisotropy_mats, strict=True):
            if label.startswith('E1'):
                row_idx = int(label[3])
                col_idx = int(label[5])
                expected = term.alpha[row_idx, 0] * term.beta[0, col_idx] - term.alpha[row_idx, 1] * term.beta[1, col_idx]
            else:
                row_idx = int(label[3])
                col_idx = int(label[5])
                expected = term.alpha[row_idx, 1] * term.beta[1, col_idx] - term.alpha[row_idx, 2] * term.beta[2, col_idx]
            value = float(alpha_vec @ matrix @ beta_vec)
            eta_verification_rows.append(
                {
                    'term_id': term.term_id,
                    'coordinate': label,
                    'value': value,
                    'expected': float(expected),
                    'matches': bool(abs(value - expected) < 1e-12),
                }
            )

    with open(OUT_DIR / 'complement_basis.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['coordinate'] + [f'w{col_idx + 1:02d}' for col_idx in range(complement_basis.shape[1])]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, coordinate in enumerate(eta_labels):
            row = {'coordinate': coordinate}
            for col_idx in range(complement_basis.shape[1]):
                row[f'w{col_idx + 1:02d}'] = float(complement_basis[row_idx, col_idx])
            writer.writerow(row)

    payload = {
        'signal_matrices': matrix_payload(signal_labels, signal_mats),
        'anisotropy_matrices': matrix_payload(anisotropy_labels, anisotropy_mats),
        'p_matrices': matrix_payload(p_labels, p_mats),
        'alpha_null_basis': null_basis.tolist(),
        'alpha_q_matrices': [matrix.tolist() for matrix in alpha_q_matrices],
        'complement_basis': complement_basis.tolist(),
        'complement_q_matrices': [matrix.tolist() for matrix in complement_q_matrices],
        'relation_errors': relation_errors,
        'eta_verification_all_match': all(bool(row['matches']) for row in eta_verification_rows),
    }
    (OUT_DIR / 'matrix_construction_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with open(OUT_DIR / 'eta_coordinate_verification.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['term_id', 'coordinate', 'value', 'expected', 'matches']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eta_verification_rows)


if __name__ == '__main__':
    main()