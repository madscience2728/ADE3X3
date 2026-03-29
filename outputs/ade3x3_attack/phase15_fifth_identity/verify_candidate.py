from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from complement_feasibility import feasibility_score, sample_point
from matrix_core import alpha_nullspace_and_complement
from outputs.ade3x3_attack.phase12_working_optimizer.cp_als import build_cp_tensor_cab, cp_loss_stats, cp_terms_from_factors


OUT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def best_sampled_decomposition(q_matrices: list[np.ndarray], attempts: int = 16) -> dict[str, object]:
    target = build_cp_tensor_cab().reshape(9, 81)
    best: dict[str, object] | None = None
    for attempt_idx in range(attempts):
        rng = np.random.default_rng(20261200 + attempt_idx)
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(22):
            point = sample_point(q_matrices, rng)
            if point is None:
                continue
            samples.append(point)
        if len(samples) < 22:
            continue
        a_factor = np.column_stack([sample[0] for sample in samples])
        b_factor = np.column_stack([sample[1] for sample in samples])
        w_matrix = np.stack([np.kron(sample[0], sample[1]) for sample in samples], axis=0)
        gamma = np.linalg.lstsq(w_matrix.T, target.T, rcond=None)[0]
        c_factor = gamma.T
        loss_value, max_abs = cp_loss_stats(build_cp_tensor_cab(), c_factor, a_factor, b_factor)
        candidate = {
            'attempt_idx': attempt_idx,
            'sampled_points': len(samples),
            'final_loss': loss_value,
            'final_max_abs': max_abs,
            'c_factor': c_factor.tolist(),
            'a_factor': a_factor.tolist(),
            'b_factor': b_factor.tolist(),
        }
        if best is None or float(candidate['final_loss']) < float(best['final_loss']):
            best = candidate
    return best or {'final_loss': float('inf'), 'final_max_abs': float('inf')}


def main() -> None:
    matrix_summary = load_json(OUT_DIR / 'matrix_construction_summary.json')
    feasibility_summary = load_json(OUT_DIR / 'complement_feasibility_summary.json')
    optimize_summary = load_json(OUT_DIR / 'optimize_fifth_identity_summary.json')

    _, _, alpha_q_matrices, anisotropy_mats, _ = alpha_nullspace_and_complement()
    best_result = optimize_summary['best_result']
    candidate_vector = np.array(best_result['candidate'], dtype=np.float64)
    q5 = np.zeros((9, 9), dtype=np.float64)
    for coeff, matrix in zip(candidate_vector, anisotropy_mats, strict=True):
        q5 += coeff * matrix
    q_matrices = alpha_q_matrices + [q5]
    score = feasibility_score(q_matrices, sample_count=200, seed=20261111)
    candidate = best_sampled_decomposition(q_matrices)

    profile = None
    if np.isfinite(candidate['final_loss']):
        import sys

        if str(OUT_DIR.parent) not in sys.path:
            sys.path.insert(0, str(OUT_DIR.parent))
        from attack_common import profile_terms  # noqa: E402

        terms = cp_terms_from_factors(
            np.array(candidate['c_factor'], dtype=np.float64),
            np.array(candidate['a_factor'], dtype=np.float64),
            np.array(candidate['b_factor'], dtype=np.float64),
            prefix='phase15_candidate_',
        )
        profile = profile_terms(terms, rank_tol=1e-8, projection_tol=1e-6)

    summary = {
        'best_basis_row': feasibility_summary['best_row'],
        'optimized_direction': best_result,
        'optimized_score': score,
        'candidate_decomposition': candidate,
        'candidate_profile': profile,
        'near_zero_candidate_found': bool(np.isfinite(candidate['final_max_abs']) and float(candidate['final_max_abs']) < 1e-8),
    }
    (OUT_DIR / 'verify_candidate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    results_md = f'''# Phase 15 Results

## 15a-15b. Matrix Construction and AlphaTensor Null Space

- Signal matrices constructed: {len(matrix_summary['signal_matrices'])}
- Anisotropy matrices constructed: {len(matrix_summary['anisotropy_matrices'])}
- AlphaTensor Q matrices verified from the null-space basis: {len(matrix_summary['alpha_q_matrices'])}
- Complement dimension inside anisotropy space: {len(matrix_summary['complement_basis'][0]) if matrix_summary['complement_basis'] else 0}
- All anisotropy coordinate checks on the 23 public terms matched: {matrix_summary['eta_verification_all_match']}

## 15c. Complement Basis Feasibility Ranking

- Best raw complement basis direction: w{int(feasibility_summary['best_basis_id']):02d}
- Best basis projection loss: {feasibility_summary['best_row']['projection_loss']}
- Best basis max-abs residual: {feasibility_summary['best_row']['projection_max_abs']}

## 15d. Optimization in the 14-Dimensional Complement

- Best optimized projection loss: {best_result['projection_loss']}
- Best optimized max-abs residual: {best_result['projection_max_abs']}
- Optimizer iterations: {best_result['nit']}
- Optimizer evaluations: {best_result['nfev']}

## 15e. Candidate Verification

- Best 22-point candidate final loss: {candidate['final_loss']}
- Best 22-point candidate max-abs residual: {candidate['final_max_abs']}
- Near-zero candidate found: {summary['near_zero_candidate_found']}
'''
    (OUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')


if __name__ == '__main__':
    main()