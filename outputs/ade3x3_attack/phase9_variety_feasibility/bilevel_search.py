from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import target_output_unfolding_9x81, write_csv, write_json  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def matrix_rank(matrix: np.ndarray, tol: float = 1e-8) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def sample_point(q_matrices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(512):
        a_vec = rng.standard_normal(9)
        norm_a = np.linalg.norm(a_vec)
        if norm_a < 1e-12:
            continue
        constraint = np.stack([q_matrix.T @ a_vec for q_matrix in q_matrices], axis=0)
        rank = matrix_rank(constraint)
        if rank >= 9:
            continue
        _, _, vh = np.linalg.svd(constraint, full_matrices=True)
        nullspace = vh[rank:].T
        if nullspace.size == 0:
            continue
        coeffs = rng.standard_normal(nullspace.shape[1])
        b_vec = nullspace @ coeffs
        norm_b = np.linalg.norm(b_vec)
        if norm_b < 1e-12:
            continue
        return a_vec / norm_a, b_vec / norm_b
    return None


def evaluate_q_space(q_matrices: np.ndarray, sample_count: int, target_unfold: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    sigma_rows: list[np.ndarray] = []
    w_rows: list[np.ndarray] = []
    for _ in range(sample_count):
        point = sample_point(q_matrices, rng)
        if point is None:
            continue
        a_vec, b_vec = point
        sigma_rows.append((a_vec.reshape(3, 3) @ b_vec.reshape(3, 3)).reshape(9))
        w_rows.append(np.kron(a_vec, b_vec))
    if not w_rows:
        return {
            'sampled_points': 0,
            'sigma_rank': 0,
            'rank_W': 0,
            'projection_max_abs': float('inf'),
            'projection_loss': float('inf'),
        }
    sigma_matrix = np.stack(sigma_rows, axis=0)
    w_matrix = np.stack(w_rows, axis=0)
    gamma = np.linalg.lstsq(w_matrix.T, target_unfold.T, rcond=None)[0]
    residual = w_matrix.T @ gamma - target_unfold.T
    return {
        'sampled_points': len(w_rows),
        'sigma_rank': matrix_rank(sigma_matrix, tol=1e-7),
        'rank_W': matrix_rank(w_matrix, tol=1e-7),
        'projection_max_abs': float(np.max(np.abs(residual))),
        'projection_loss': float(np.sum(residual * residual)),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    outer_trials = env_int('ADE3X3_PHASE9_BILEVEL_TRIALS', 64)
    sample_count = env_int('ADE3X3_PHASE9_BILEVEL_POINTS', 22)
    rng = np.random.default_rng(20260401)
    target_unfold = target_output_unfolding_9x81()
    rows: list[dict[str, object]] = []
    best_q_matrices: list[list[list[float]]] | None = None
    best_loss = float('inf')

    for trial_idx in range(outer_trials):
        q_matrices = rng.standard_normal((5, 9, 9))
        metrics = evaluate_q_space(q_matrices, sample_count, target_unfold, rng)
        row = {'trial': trial_idx, **metrics}
        rows.append(row)
        if float(metrics['projection_loss']) < best_loss:
            best_loss = float(metrics['projection_loss'])
            best_q_matrices = q_matrices.tolist()

    rows.sort(key=lambda row: float(row['projection_loss']))
    write_csv(
        OUTPUT_DIR / 'bilevel_random_search.csv',
        rows,
        ['trial', 'sampled_points', 'sigma_rank', 'rank_W', 'projection_max_abs', 'projection_loss'],
    )
    write_json(
        OUTPUT_DIR / 'bilevel_search_summary.json',
        {
            'outer_trials': outer_trials,
            'sample_count_per_trial': sample_count,
            'best_trial': rows[0] if rows else None,
            'best_q_matrices': best_q_matrices,
            'interpretation': (
                'This is a random-search proxy for the outer identity-subspace optimization: low projection residual means '
                'the sampled V5 variety places its rank-1 rows W closer to containing the tensor unfolding.'
            ),
        },
    )

    dim_summary = load_json(OUTPUT_DIR / 'variety_dimension_summary.json')
    sigma_summary = load_json(OUTPUT_DIR / 'sigma_span_summary.json')
    tensor_summary = load_json(OUTPUT_DIR / 'tensor_feasibility_summary.json')
    bilevel_summary = load_json(OUTPUT_DIR / 'bilevel_search_summary.json')
    dim_lookup = {int(item['k']): item for item in dim_summary['summary']}

    results_md = f'''# Phase 9 Results

## 9a. Generic Variety Dimension

- Random bilinear systems were sampled for `k=1,...,9`.
- Observed median dimensions: `k=4 -> {dim_lookup[4]['median_observed_dimension']}`, `k=5 -> {dim_lookup[5]['median_observed_dimension']}`, `k=6 -> {dim_lookup[6]['median_observed_dimension']}`, `k=9 -> {dim_lookup[9]['median_observed_dimension']}`.
- Sampled sigma-image ranks stayed at `k=4 -> {dim_lookup[4]['median_sigma_sample_rank']}`, `k=5 -> {dim_lookup[5]['median_sigma_sample_rank']}`, `k=6 -> {dim_lookup[6]['median_sigma_sample_rank']}`, `k=9 -> {dim_lookup[9]['median_sigma_sample_rank']}`.

## 9b. Sigma-Span on Random V5

- Trial count: {sigma_summary['trial_count']}.
- `rank(Sigma)=9` success rate from 22 sampled points: {sigma_summary['rank9_success_rate']:.3f}.
- Rank histogram: {sigma_summary['rank_histogram']}.

## 9c. Tensor Feasibility on Random V5

- Trial count: {tensor_summary['trial_count']}.
- Exact/near-exact tensor-feasible random V5 count: {tensor_summary['feasible_count']}.
- Best random V5 projection loss: {tensor_summary['best_trial']['projection_loss'] if tensor_summary['best_trial'] else None}.
- Best random V5 max-abs residual: {tensor_summary['best_trial']['projection_max_abs'] if tensor_summary['best_trial'] else None}.

## 9d. Outer Random Search Proxy

- A random search over 5-identity subspaces was run for {bilevel_summary['outer_trials']} outer trials.
- Best observed bilevel proxy loss: {bilevel_summary['best_trial']['projection_loss'] if bilevel_summary['best_trial'] else None}.
- Best observed proxy max-abs residual: {bilevel_summary['best_trial']['projection_max_abs'] if bilevel_summary['best_trial'] else None}.

## Phase 9 Verdict

- Generic 5-identity varieties remain geometrically large enough to support full 9-dimensional sigma span.
- Random V5 choices almost never satisfy the tensor constraint exactly, so the obstruction is not raw variety dimension but the special placement of the identity subspace relative to the tensor.
- The outer search problem is therefore highly structured: finding a useful 5-identity space is a rare-position problem, not a dimension-count problem.
'''
    (OUTPUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')


if __name__ == '__main__':
    main()