from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


OUT_DIR = Path(__file__).resolve().parent
BEST_DIR = OUT_DIR / 'best_decompositions'
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import (  # noqa: E402
    TARGET_TENSOR,
    load_public_terms,
    profile_terms,
    serialize_decomposition,
    stacked_factors_from_terms,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)


torch.set_default_dtype(torch.float64)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


BATCH_SIZE = env_int('ADE3X3_PHASE4_BATCH_SIZE', 16)
ADAM_STEPS = env_int('ADE3X3_PHASE4_ADAM_STEPS', 2500)
ADAM_LR = env_float('ADE3X3_PHASE4_ADAM_LR', 4e-3)
GAMMA_REFRESH = env_int('ADE3X3_PHASE4_GAMMA_REFRESH', 250)
EXACT_TOL = env_float('ADE3X3_PHASE4_EXACT_TOL', 1e-10)
RANDOM_INIT_SCALE = env_float('ADE3X3_PHASE4_RANDOM_INIT_SCALE', 0.2)
R22_COLD_RESTARTS = env_int('ADE3X3_PHASE4_R22_COLD_RESTARTS', 256)
R21_COLD_RESTARTS = env_int('ADE3X3_PHASE4_R21_COLD_RESTARTS', 64)
R23_COLD_RESTARTS = env_int('ADE3X3_PHASE4_R23_COLD_RESTARTS', 64)
WARM_PAIR_RESTARTS = env_int('ADE3X3_PHASE4_WARM_PAIR_RESTARTS', 16)
WARM_CORE_RESTARTS = env_int('ADE3X3_PHASE4_WARM_CORE_RESTARTS', 16)
BORDER_RESTARTS = env_int('ADE3X3_PHASE4_BORDER_RESTARTS', 32)
BORDER_PENALTY = env_float('ADE3X3_PHASE4_BORDER_PENALTY', 1e-2)
NOISE_SCALE = env_float('ADE3X3_PHASE4_NOISE_SCALE', 0.01)
SAVE_TOP_K = env_int('ADE3X3_PHASE4_SAVE_TOP_K', 10)

PAIR_REMOVALS = [('t03', 't06'), ('t09', 't12'), ('t15', 't17')]


def gamma_fit_batch(alpha: torch.Tensor, beta: torch.Tensor, target81: torch.Tensor) -> torch.Tensor:
    batch_size, rank, _ = alpha.shape
    features = torch.einsum('xra,xrb->xabr', alpha, beta).reshape(batch_size, 81, rank)
    return torch.linalg.lstsq(features, target81.expand(batch_size, -1, -1)).solution


def optimize_batch(
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    gamma_init: np.ndarray,
    rank: int,
    border_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = torch.tensor(TARGET_TENSOR, dtype=torch.float64)
    target81 = target.reshape(81, 9)
    alpha = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float64))
    gamma = torch.nn.Parameter(torch.tensor(gamma_init, dtype=torch.float64))

    optimizer = torch.optim.Adam([alpha, beta, gamma], lr=ADAM_LR)
    for step in range(ADAM_STEPS):
        if step and step % GAMMA_REFRESH == 0:
            with torch.no_grad():
                gamma.copy_(gamma_fit_batch(alpha, beta, target81))
        optimizer.zero_grad()
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        losses = (residual * residual).sum(dim=(1, 2, 3))
        if border_penalty > 0.0:
            losses = losses + border_penalty * (gamma[:, -1, :] * gamma[:, -1, :]).sum(dim=1)
        losses.mean().backward()
        optimizer.step()

    with torch.no_grad():
        gamma.copy_(gamma_fit_batch(alpha, beta, target81))
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        max_abs = residual.abs().amax(dim=(1, 2, 3)).cpu().numpy()
        losses = (residual * residual).sum(dim=(1, 2, 3)).cpu().numpy()
        vanishing_norm = np.linalg.norm(gamma[:, -1, :].cpu().numpy(), axis=1)
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy(), gamma.detach().cpu().numpy(), max_abs, losses, vanishing_norm


def candidate_row(search_id: str, rank: int, strategy: str, init_detail: str, seed: int, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, max_abs: float, loss_value: float, border_tail_norm: float | None = None) -> dict:
    terms = terms_from_stacked_factors(alpha, beta, gamma, prefix=f'{search_id}_')
    profile = profile_terms(terms, rank_tol=1e-8, projection_tol=1e-6)
    return {
        'search_id': search_id,
        'rank': rank,
        'strategy': strategy,
        'init_detail': init_detail,
        'seed': seed,
        'final_loss': loss_value,
        'final_max_abs': max_abs,
        'exact_hit': bool(max_abs <= EXACT_TOL),
        'rank_H_numeric': profile['rank_H_numeric'],
        'rank_Delta_numeric': profile['rank_Delta_numeric'],
        'rank_Nuisance_numeric': profile['rank_Nuisance_numeric'],
        'quotient_gain_numeric': profile['quotient_gain_numeric'],
        'delta_in_eta_numeric': profile['delta_in_eta_numeric'],
        'projection_max_residual_numeric': profile['projection_max_residual_numeric'],
        'border_tail_norm': '' if border_tail_norm is None else border_tail_norm,
    }


def make_random_inits(batch_size: int, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
    beta = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
    gamma = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
    return alpha, beta, gamma


def make_warm_pair_inits(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    term_ids = [f't{idx + 1:02d}' for idx in range(base_alpha.shape[0])]
    id_to_idx = {term_id: idx for idx, term_id in enumerate(term_ids)}
    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    rng = np.random.default_rng(40400)
    for left_id, right_id in PAIR_REMOVALS:
        remove = {id_to_idx[left_id], id_to_idx[right_id]}
        keep = [idx for idx in range(base_alpha.shape[0]) if idx not in remove]
        for restart_idx in range(WARM_PAIR_RESTARTS):
            alpha = np.zeros((22, 9), dtype=np.float64)
            beta = np.zeros((22, 9), dtype=np.float64)
            gamma = np.zeros((22, 9), dtype=np.float64)
            alpha[:21] = base_alpha[keep]
            beta[:21] = base_beta[keep]
            gamma[:21] = base_gamma[keep]
            alpha[21] = RANDOM_INIT_SCALE * rng.standard_normal(9)
            beta[21] = RANDOM_INIT_SCALE * rng.standard_normal(9)
            gamma[21] = RANDOM_INIT_SCALE * rng.standard_normal(9)
            alpha += NOISE_SCALE * rng.standard_normal(alpha.shape)
            beta += NOISE_SCALE * rng.standard_normal(beta.shape)
            gamma += NOISE_SCALE * rng.standard_normal(gamma.shape)
            specs.append((f'remove_{left_id}_{right_id}', alpha, beta, gamma, 40400 + restart_idx))
    return specs


def make_warm_core_inits(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    term_ids = [f't{idx + 1:02d}' for idx in range(base_alpha.shape[0])]
    id_to_idx = {term_id: idx for idx, term_id in enumerate(term_ids)}
    pair_indices = [(id_to_idx[left_id], id_to_idx[right_id]) for left_id, right_id in PAIR_REMOVALS]
    rng = np.random.default_rng(50500)
    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for mask in range(8):
        removed = set()
        bits = []
        for pair_idx, (left_idx, right_idx) in enumerate(pair_indices):
            pick = (mask >> pair_idx) & 1
            removed.add(left_idx if pick == 0 else right_idx)
            bits.append(str(pick))
        keep = [idx for idx in range(base_alpha.shape[0]) if idx not in removed]
        for restart_idx in range(WARM_CORE_RESTARTS):
            alpha = np.zeros((22, 9), dtype=np.float64)
            beta = np.zeros((22, 9), dtype=np.float64)
            gamma = np.zeros((22, 9), dtype=np.float64)
            alpha[:20] = base_alpha[keep]
            beta[:20] = base_beta[keep]
            gamma[:20] = base_gamma[keep]
            alpha[20:] = RANDOM_INIT_SCALE * rng.standard_normal((2, 9))
            beta[20:] = RANDOM_INIT_SCALE * rng.standard_normal((2, 9))
            gamma[20:] = RANDOM_INIT_SCALE * rng.standard_normal((2, 9))
            alpha += NOISE_SCALE * rng.standard_normal(alpha.shape)
            beta += NOISE_SCALE * rng.standard_normal(beta.shape)
            gamma += NOISE_SCALE * rng.standard_normal(gamma.shape)
            specs.append((f'core_mask_{"".join(bits)}', alpha, beta, gamma, 50500 + 100 * mask + restart_idx))
    return specs


def run_warm_search(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    specs = make_warm_pair_inits(base_alpha, base_beta, base_gamma) + make_warm_core_inits(base_alpha, base_beta, base_gamma)
    for spec_idx in range(0, len(specs), BATCH_SIZE):
        chunk = specs[spec_idx:spec_idx + BATCH_SIZE]
        alpha_init = np.stack([item[1] for item in chunk])
        beta_init = np.stack([item[2] for item in chunk])
        gamma_init = np.stack([item[3] for item in chunk])
        alpha_batch, beta_batch, gamma_batch, residuals, losses, _ = optimize_batch(alpha_init, beta_init, gamma_init, rank=22)
        for local_idx, (detail, _, _, _, seed) in enumerate(chunk):
            search_id = f'warm22_{spec_idx + local_idx:04d}'
            row = candidate_row(search_id, 22, 'warm_alpha_seed', detail, seed, alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], float(residuals[local_idx]), float(losses[local_idx]))
            rows.append(row)
            saved.append({'row': row, 'terms': terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=search_id)})
    return rows, saved


def run_cold_search(rank: int, restarts: int, prefix: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    completed = 0
    while completed < restarts:
        batch_size = min(BATCH_SIZE, restarts - completed)
        alpha_init, beta_init, gamma_init = make_random_inits(batch_size, rank, 601000 + 1000 * rank + completed)
        alpha_batch, beta_batch, gamma_batch, residuals, losses, _ = optimize_batch(alpha_init, beta_init, gamma_init, rank=rank)
        for local_idx in range(batch_size):
            search_id = f'{prefix}_{completed + local_idx:04d}'
            row = candidate_row(search_id, rank, 'cold_random', prefix, 601000 + 1000 * rank + completed + local_idx, alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], float(residuals[local_idx]), float(losses[local_idx]))
            rows.append(row)
            saved.append({'row': row, 'terms': terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=search_id)})
        completed += batch_size
    return rows, saved


def run_border_search() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    completed = 0
    while completed < BORDER_RESTARTS:
        batch_size = min(BATCH_SIZE, BORDER_RESTARTS - completed)
        alpha_init, beta_init, gamma_init = make_random_inits(batch_size, 23, 777000 + completed)
        alpha_batch, beta_batch, gamma_batch, residuals, losses, tail_norms = optimize_batch(alpha_init, beta_init, gamma_init, rank=23, border_penalty=BORDER_PENALTY)
        for local_idx in range(batch_size):
            truncated_terms = terms_from_stacked_factors(alpha_batch[local_idx][:-1], beta_batch[local_idx][:-1], gamma_batch[local_idx][:-1], prefix=f'border22_{completed + local_idx:04d}')
            truncated_profile = profile_terms(truncated_terms, rank_tol=1e-8, projection_tol=1e-6)
            search_id = f'border23_{completed + local_idx:04d}'
            row = {
                'search_id': search_id,
                'rank': 22,
                'strategy': 'border_vanishing_tail',
                'init_detail': 'rank23_with_penalized_last_term',
                'seed': 777000 + completed + local_idx,
                'final_loss': float(losses[local_idx]),
                'final_max_abs': float(residuals[local_idx]),
                'exact_hit': False,
                'rank_H_numeric': truncated_profile['rank_H_numeric'],
                'rank_Delta_numeric': truncated_profile['rank_Delta_numeric'],
                'rank_Nuisance_numeric': truncated_profile['rank_Nuisance_numeric'],
                'quotient_gain_numeric': truncated_profile['quotient_gain_numeric'],
                'delta_in_eta_numeric': truncated_profile['delta_in_eta_numeric'],
                'projection_max_residual_numeric': truncated_profile['projection_max_residual_numeric'],
                'border_tail_norm': float(tail_norms[local_idx]),
            }
            rows.append(row)
            saved.append({'row': row, 'terms': truncated_terms})
        completed += batch_size
    return rows, saved


def build_results_markdown(rows: list[dict], stats: dict) -> str:
    lines: list[str] = []
    lines.append('# Phase 4 Results')
    lines.append('')
    lines.append('## Search Summary')
    lines.append('')
    lines.append(f"- quotient formulation used: {stats['quotient_formulation_used']}")
    lines.append(f"- total candidate runs: {stats['total_runs']}")
    lines.append(f"- best rank-22 max-abs residual: {stats['best_r22_max_abs']}")
    lines.append(f"- best rank-22 quotient gain: {stats['best_r22_quotient_gain']}")
    lines.append(f"- exact rank-22 hits: {stats['exact_r22_hits']}")
    lines.append('')
    lines.append('## Landscape')
    lines.append('')
    for rank in [21, 22, 23]:
        rank_stats = stats['landscape_by_rank'].get(str(rank), {})
        if not rank_stats:
            continue
        lines.append(f"- R={rank}: runs={rank_stats['count']}, best max-abs={rank_stats['best_max_abs']}, median max-abs={rank_stats['median_max_abs']}")
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    if stats['exact_r22_hits'] == 0:
        lines.append('No exact 22-term decomposition was found in this pass. The script still exports the best near-solutions and the rank-21/rank-23 comparison rows for follow-up.')
    else:
        lines.append('At least one exact 22-term decomposition candidate met the configured residual tolerance. Those coefficients are exported under best_decompositions/.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    base_terms, orientation, residual = load_public_terms()
    base_alpha, base_beta, base_gamma = stacked_factors_from_terms(base_terms)

    rows: list[dict] = []
    saved_candidates: list[dict] = []

    warm_rows, warm_saved = run_warm_search(base_alpha, base_beta, base_gamma)
    rows.extend(warm_rows)
    saved_candidates.extend(warm_saved)

    r22_rows, r22_saved = run_cold_search(22, R22_COLD_RESTARTS, 'cold22')
    rows.extend(r22_rows)
    saved_candidates.extend(r22_saved)

    r21_rows, _ = run_cold_search(21, R21_COLD_RESTARTS, 'cold21')
    rows.extend(r21_rows)

    r23_rows, _ = run_cold_search(23, R23_COLD_RESTARTS, 'cold23')
    rows.extend(r23_rows)

    border_rows, border_saved = run_border_search()
    rows.extend(border_rows)
    saved_candidates.extend(border_saved)

    rows.sort(key=lambda row: (row['rank'] != 22, float(row['final_max_abs']), float(row['final_loss'])))
    write_csv(
        OUT_DIR / 'search_results.csv',
        rows,
        [
            'search_id',
            'rank',
            'strategy',
            'init_detail',
            'seed',
            'final_loss',
            'final_max_abs',
            'exact_hit',
            'rank_H_numeric',
            'rank_Delta_numeric',
            'rank_Nuisance_numeric',
            'quotient_gain_numeric',
            'delta_in_eta_numeric',
            'projection_max_residual_numeric',
            'border_tail_norm',
        ],
    )

    saved_candidates.sort(key=lambda item: (float(item['row']['final_max_abs']), float(item['row']['final_loss'])))
    for save_idx, item in enumerate(saved_candidates[:SAVE_TOP_K], start=1):
        payload = serialize_decomposition(
            decomposition_id=item['row']['search_id'],
            source='phase4_best_candidate',
            source_detail=item['row']['strategy'],
            terms=item['terms'],
            max_abs_residual=float(item['row']['final_max_abs']),
            loss_value=float(item['row']['final_loss']),
            metadata=item['row'],
        )
        (BEST_DIR / f'{save_idx:02d}_{item["row"]["search_id"]}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    landscape_by_rank: dict[str, dict] = {}
    for rank in [21, 22, 23]:
        rank_rows = [row for row in rows if int(row['rank']) == rank and row['strategy'] != 'border_vanishing_tail']
        if not rank_rows:
            continue
        max_abs_values = sorted(float(row['final_max_abs']) for row in rank_rows)
        landscape_by_rank[str(rank)] = {
            'count': len(rank_rows),
            'best_max_abs': max_abs_values[0],
            'median_max_abs': max_abs_values[len(max_abs_values) // 2],
            'worst_max_abs': max_abs_values[-1],
        }

    r22_rows_only = [row for row in rows if int(row['rank']) == 22]
    best_r22 = min(r22_rows_only, key=lambda row: (float(row['final_max_abs']), float(row['final_loss'])))
    stats = {
        'base_source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
        'gamma_orientation': orientation,
        'exact_reconstruction_residual': int(residual),
        'quotient_formulation_used': False,
        'total_runs': len(rows),
        'exact_r22_hits': sum(int(bool(row['exact_hit'])) for row in r22_rows_only),
        'best_r22_max_abs': float(best_r22['final_max_abs']),
        'best_r22_quotient_gain': int(best_r22['quotient_gain_numeric']),
        'landscape_by_rank': landscape_by_rank,
    }
    write_json(OUT_DIR / 'landscape_statistics.json', stats)
    (OUT_DIR / 'RESULTS.md').write_text(build_results_markdown(rows, stats), encoding='utf-8')

    print('Phase 4 search complete')
    print(f"  total runs = {len(rows)}")
    print(f"  best R=22 max abs = {best_r22['final_max_abs']}")


if __name__ == '__main__':
    main()