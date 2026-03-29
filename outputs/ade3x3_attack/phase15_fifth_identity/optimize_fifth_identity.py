from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from complement_feasibility import feasibility_score
from construct_matrices import alpha_nullspace_and_complement


OUT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def normalized_candidate(theta: np.ndarray, complement_basis: np.ndarray) -> np.ndarray:
    vector = complement_basis @ theta
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError('Zero candidate direction.')
    return vector / norm


def q_from_candidate(candidate: np.ndarray, anisotropy_mats: np.ndarray) -> np.ndarray:
    q_matrix = np.zeros((9, 9), dtype=np.float64)
    for coeff, basis_matrix in zip(candidate, anisotropy_mats, strict=True):
        q_matrix += coeff * basis_matrix
    return q_matrix


def main() -> None:
    _, complement_basis, alpha_q_matrices, anisotropy_mats, _ = alpha_nullspace_and_complement()
    feasibility_summary = load_json(OUT_DIR / 'complement_feasibility_summary.json')
    start_indices = [int(feasibility_summary['rows'][idx]['basis_id']) - 1 for idx in range(min(3, len(feasibility_summary['rows'])))]
    starts = [np.eye(complement_basis.shape[1])[start_idx] for start_idx in start_indices]
    starts.extend([
        np.full(complement_basis.shape[1], 1.0 / np.sqrt(complement_basis.shape[1])),
        np.linspace(1.0, 2.0, complement_basis.shape[1]),
    ])

    history: list[dict[str, object]] = []

    def objective(theta: np.ndarray, tag: str) -> float:
        candidate = normalized_candidate(theta, complement_basis)
        q_matrix = q_from_candidate(candidate, anisotropy_mats)
        score = feasibility_score(alpha_q_matrices + [q_matrix], sample_count=200, seed=20261111)
        history.append(
            {
                'tag': tag,
                'projection_loss': score['projection_loss'],
                'projection_max_abs': score['projection_max_abs'],
                'sampled_points': score['sampled_points'],
                'rank_W': score['rank_W'],
                'rank_sigma': score['rank_sigma'],
                'theta': theta.tolist(),
                'candidate': candidate.tolist(),
            }
        )
        return float(score['projection_loss'])

    best_result: dict[str, object] | None = None
    for start_idx, theta0 in enumerate(starts, start=1):
        result = minimize(
            lambda theta, tag=f'start_{start_idx}': objective(theta, tag),
            theta0,
            method='Nelder-Mead',
            options={'maxiter': 120, 'xatol': 1e-4, 'fatol': 1e-6},
        )
        candidate = normalized_candidate(np.array(result.x, dtype=np.float64), complement_basis)
        q_matrix = q_from_candidate(candidate, anisotropy_mats)
        final_score = feasibility_score(alpha_q_matrices + [q_matrix], sample_count=200, seed=20261111)
        record = {
            'start_idx': start_idx,
            'start_theta': theta0.tolist(),
            'theta': np.array(result.x, dtype=np.float64).tolist(),
            'candidate': candidate.tolist(),
            'projection_loss': final_score['projection_loss'],
            'projection_max_abs': final_score['projection_max_abs'],
            'sampled_points': final_score['sampled_points'],
            'rank_W': final_score['rank_W'],
            'rank_sigma': final_score['rank_sigma'],
            'status': int(result.status),
            'message': str(result.message),
            'nit': int(result.nit),
            'nfev': int(result.nfev),
            'q_matrix': q_matrix.tolist(),
        }
        if best_result is None or float(record['projection_loss']) < float(best_result['projection_loss']):
            best_result = record

    with open(OUT_DIR / 'optimization_trajectory.csv', 'w', newline='', encoding='utf-8') as handle:
        fieldnames = ['tag', 'projection_loss', 'projection_max_abs', 'sampled_points', 'rank_W', 'rank_sigma', 'theta', 'candidate']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    (OUT_DIR / 'optimize_fifth_identity_summary.json').write_text(
        json.dumps({'best_result': best_result, 'history_length': len(history)}, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()