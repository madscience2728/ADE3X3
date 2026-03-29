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

from attack_common import eta_coordinate_labels, load_public_terms  # noqa: E402
from matrix_core import (  # noqa: E402
    alpha_nullspace_and_complement,
    anisotropy_matrix_1,
    anisotropy_matrix_2,
    build_anisotropy_matrices,
    build_signal_matrices,
    evaluate_bilinear,
    matrix_rank,
    p_matrix,
    signal_matrix,
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_public_terms(
    signal_labels: list[str],
    signal_mats: np.ndarray,
    anisotropy_labels: list[str],
    anisotropy_mats: np.ndarray,
    null_basis: np.ndarray,
    q_matrices: np.ndarray,
) -> dict[str, object]:
    terms, orientation, residual = load_public_terms()
    signal_rows: list[dict[str, object]] = []
    eta_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    q_rows: list[dict[str, object]] = []

    max_signal_error = 0.0
    max_eta_error = 0.0
    max_null_error = 0.0
    max_q_error = 0.0

    for term in terms:
        sigma_expected = (term.alpha @ term.beta).reshape(-1)
        eta_expected = np.empty(18, dtype=np.float64)
        eta_col = 0
        for row_idx in range(3):
            for col_idx in range(3):
                eta_expected[eta_col] = term.alpha[row_idx, 0] * term.beta[0, col_idx] - term.alpha[row_idx, 1] * term.beta[1, col_idx]
                eta_col += 1
        for row_idx in range(3):
            for col_idx in range(3):
                eta_expected[eta_col] = term.alpha[row_idx, 1] * term.beta[1, col_idx] - term.alpha[row_idx, 2] * term.beta[2, col_idx]
                eta_col += 1

        for coord_idx, (label, matrix) in enumerate(zip(signal_labels, signal_mats, strict=True)):
            value = evaluate_bilinear(matrix, term.alpha, term.beta)
            expected = float(sigma_expected[coord_idx])
            error = abs(value - expected)
            max_signal_error = max(max_signal_error, error)
            signal_rows.append({'term_id': term.term_id, 'coordinate': label, 'value': value, 'expected': expected, 'abs_error': error, 'matches': error < 1e-12})

        for coord_idx, (label, matrix) in enumerate(zip(anisotropy_labels, anisotropy_mats, strict=True)):
            value = evaluate_bilinear(matrix, term.alpha, term.beta)
            expected = float(eta_expected[coord_idx])
            error = abs(value - expected)
            max_eta_error = max(max_eta_error, error)
            eta_rows.append({'term_id': term.term_id, 'coordinate': label, 'value': value, 'expected': expected, 'abs_error': error, 'matches': error < 1e-12})

        for basis_idx, identity_vector in enumerate(null_basis, start=1):
            value = float(eta_expected @ identity_vector)
            max_null_error = max(max_null_error, abs(value))
            null_rows.append({'term_id': term.term_id, 'basis_id': basis_idx, 'value': value, 'matches': abs(value) < 1e-12})

        for basis_idx, q_matrix in enumerate(q_matrices, start=1):
            value = evaluate_bilinear(q_matrix, term.alpha, term.beta)
            max_q_error = max(max_q_error, abs(value))
            q_rows.append({'term_id': term.term_id, 'basis_id': basis_idx, 'value': value, 'matches': abs(value) < 1e-12})

    return {
        'orientation': orientation,
        'reconstruction_max_abs': residual,
        'term_count': len(terms),
        'signal_rows': signal_rows,
        'eta_rows': eta_rows,
        'null_rows': null_rows,
        'q_rows': q_rows,
        'max_signal_error': max_signal_error,
        'max_eta_error': max_eta_error,
        'max_null_error': max_null_error,
        'max_q_error': max_q_error,
        'all_signal_match': all(bool(row['matches']) for row in signal_rows),
        'all_eta_match': all(bool(row['matches']) for row in eta_rows),
        'all_null_match': all(bool(row['matches']) for row in null_rows),
        'all_q_match': all(bool(row['matches']) for row in q_rows),
    }


def main() -> None:
    signal_labels, signal_mats = build_signal_matrices()
    anisotropy_labels, anisotropy_mats = build_anisotropy_matrices()
    null_basis, complement_basis, q_matrices, _, complement_q_matrices = alpha_nullspace_and_complement()

    relation_rows: list[dict[str, object]] = []
    for row_idx in range(3):
        for col_idx in range(3):
            s_matrix = signal_matrix(row_idx, col_idx)
            e1 = anisotropy_matrix_1(row_idx, col_idx)
            e2 = anisotropy_matrix_2(row_idx, col_idx)
            p0 = (s_matrix + 2.0 * e1 + e2) / 3.0
            p1 = (s_matrix - e1 + e2) / 3.0
            p2 = (s_matrix - e1 - 2.0 * e2) / 3.0
            relation_rows.append({'fiber': f'({row_idx},{col_idx})', 'p0_residual': float(np.max(np.abs(p0 - p_matrix(0, row_idx, col_idx)))), 'p1_residual': float(np.max(np.abs(p1 - p_matrix(1, row_idx, col_idx)))), 'p2_residual': float(np.max(np.abs(p2 - p_matrix(2, row_idx, col_idx))))})

    orthogonality_rows: list[dict[str, object]] = []
    max_trace_abs = 0.0
    for basis_idx, q_matrix in enumerate(q_matrices, start=1):
        for signal_idx, label in enumerate(signal_labels):
            value = float(np.trace(q_matrix.T @ signal_mats[signal_idx]))
            max_trace_abs = max(max_trace_abs, abs(value))
            orthogonality_rows.append({'basis_id': basis_idx, 'signal': label, 'trace': value, 'matches': abs(value) < 1e-12})

    verification = verify_public_terms(signal_labels, signal_mats, anisotropy_labels, anisotropy_mats, null_basis, q_matrices)
    complement_gram = complement_basis @ complement_basis.T
    complement_orthogonality = complement_basis @ null_basis.T

    np.save(OUT_DIR / 'S_matrices.npy', signal_mats)
    np.save(OUT_DIR / 'E_matrices.npy', anisotropy_mats)
    np.save(OUT_DIR / 'Q_matrices.npy', q_matrices)
    np.save(OUT_DIR / 'complement_basis.npy', complement_basis)

    write_csv(OUT_DIR / 'signal_coordinate_verification.csv', verification['signal_rows'], ['term_id', 'coordinate', 'value', 'expected', 'abs_error', 'matches'])
    write_csv(OUT_DIR / 'eta_coordinate_verification.csv', verification['eta_rows'], ['term_id', 'coordinate', 'value', 'expected', 'abs_error', 'matches'])
    write_csv(OUT_DIR / 'null_vector_verification.csv', verification['null_rows'], ['term_id', 'basis_id', 'value', 'matches'])
    write_csv(OUT_DIR / 'q_form_verification.csv', verification['q_rows'], ['term_id', 'basis_id', 'value', 'matches'])
    write_csv(OUT_DIR / 'q_signal_orthogonality.csv', orthogonality_rows, ['basis_id', 'signal', 'trace', 'matches'])
    write_csv(
        OUT_DIR / 'complement_basis.csv',
        [{'basis_id': basis_idx + 1, **{label: float(value) for label, value in zip(eta_coordinate_labels(), vector, strict=True)}} for basis_idx, vector in enumerate(complement_basis)],
        ['basis_id', *eta_coordinate_labels()],
    )

    summary = {
        'signal_labels': signal_labels,
        'anisotropy_labels': anisotropy_labels,
        'signal_matrix_count': int(signal_mats.shape[0]),
        'anisotropy_matrix_count': int(anisotropy_mats.shape[0]),
        'q_matrix_count': int(q_matrices.shape[0]),
        'complement_dimension': int(complement_basis.shape[0]),
        'signal_matrices': signal_mats.tolist(),
        'anisotropy_matrices': anisotropy_mats.tolist(),
        'q_matrices': q_matrices.tolist(),
        'complement_basis': complement_basis.tolist(),
        'complement_q_matrices': complement_q_matrices.tolist(),
        'relation_rows': relation_rows,
        'relation_max_residual': float(max(max(row['p0_residual'], row['p1_residual'], row['p2_residual']) for row in relation_rows)),
        'verification': {
            'term_count': verification['term_count'],
            'orientation': verification['orientation'],
            'reconstruction_max_abs': verification['reconstruction_max_abs'],
            'all_signal_match': verification['all_signal_match'],
            'all_eta_match': verification['all_eta_match'],
            'all_null_match': verification['all_null_match'],
            'all_q_match': verification['all_q_match'],
            'max_signal_error': verification['max_signal_error'],
            'max_eta_error': verification['max_eta_error'],
            'max_null_error': verification['max_null_error'],
            'max_q_error': verification['max_q_error'],
        },
        'orthogonality': {'all_match': all(bool(row['matches']) for row in orthogonality_rows), 'max_trace_abs': max_trace_abs},
        'complement_checks': {
            'gram_max_abs_error': float(np.max(np.abs(complement_gram - np.eye(complement_basis.shape[0])))),
            'null_orthogonality_max_abs': float(np.max(np.abs(complement_orthogonality))),
            'null_basis_rank': matrix_rank(null_basis),
            'complement_basis_rank': matrix_rank(complement_basis),
        },
    }
    (OUT_DIR / 'matrix_construction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()