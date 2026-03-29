from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import (  # noqa: E402
    load_public_terms,
    profile_terms,
    stacked_factors_from_terms,
    target_output_unfolding_9x81,
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


RANK = 23
WARM_NOISE = env_float('ADE3X3_PHASE10_WARM_NOISE', 0.01)
WARM_ADAM_STEPS = env_int('ADE3X3_PHASE10_WARM_ADAM_STEPS', 5000)
WARM_LR = env_float('ADE3X3_PHASE10_WARM_LR', 3e-3)
COLD_RESTARTS = env_int('ADE3X3_PHASE10_COLD_RESTARTS', 200)
COLD_BATCH_SIZE = env_int('ADE3X3_PHASE10_COLD_BATCH_SIZE', 16)
COLD_ADAM_STEPS = env_int('ADE3X3_PHASE10_COLD_ADAM_STEPS', 50000)
COLD_LR = env_float('ADE3X3_PHASE10_COLD_LR', 2e-3)
INIT_SCALE = env_float('ADE3X3_PHASE10_INIT_SCALE', 1.0 / 3.0)
CLIP_NORM = env_float('ADE3X3_PHASE10_CLIP_NORM', 10.0)
BALANCE_EVERY = env_int('ADE3X3_PHASE10_BALANCE_EVERY', 100)
EXACT_TOL = env_float('ADE3X3_PHASE10_EXACT_TOL', 1e-20)
HIT_TOL = env_float('ADE3X3_PHASE10_HIT_TOL', 1e-10)


def build_target_validation(target_unfold: np.ndarray) -> dict[str, object]:
    nonzero_positions = np.argwhere(np.abs(target_unfold) > 0)
    return {
        'shape': list(target_unfold.shape),
        'nonzero_count': int(nonzero_positions.shape[0]),
        'unique_values': sorted({float(value) for value in target_unfold.reshape(-1) if value != 0}),
    }


def balance_factors(alpha: torch.Tensor, beta: torch.Tensor) -> None:
    with torch.no_grad():
        alpha_norm = alpha.norm(dim=-1).clamp_min(1e-12)
        beta_norm = beta.norm(dim=-1).clamp_min(1e-12)
        scale = torch.sqrt(beta_norm / alpha_norm)
        alpha.mul_(scale.unsqueeze(-1))
        beta.div_(scale.unsqueeze(-1))


def factorized_residual(alpha: torch.Tensor, beta: torch.Tensor, target_unfold_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_matrix = torch.einsum('...ra,...rb->...rab', alpha, beta).reshape(*alpha.shape[:-2], alpha.shape[-2], 81)
    system = w_matrix.transpose(-1, -2)
    rhs = target_unfold_t.expand(*alpha.shape[:-2], *target_unfold_t.shape)
    gamma = torch.linalg.lstsq(system, rhs).solution
    residual = system @ gamma - rhs
    loss = residual.square().sum(dim=(-2, -1))
    return loss, residual, gamma


def optimize_batch(alpha_init: np.ndarray, beta_init: np.ndarray, steps: int, learning_rate: float, target_unfold_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float64))
    optimizer = torch.optim.Adam([alpha, beta], lr=learning_rate)
    for step in range(steps):
        optimizer.zero_grad()
        loss, _, _ = factorized_residual(alpha, beta, target_unfold_t)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_([alpha, beta], max_norm=CLIP_NORM)
        optimizer.step()
        if BALANCE_EVERY > 0 and step % BALANCE_EVERY == 0:
            balance_factors(alpha, beta)

    with torch.no_grad():
        loss, residual, gamma = factorized_residual(alpha, beta, target_unfold_t)
        max_abs = residual.abs().amax(dim=(-2, -1)).cpu().numpy()
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy(), gamma.detach().cpu().numpy(), loss.detach().cpu().numpy(), max_abs


def warm_start_run(base_alpha: np.ndarray, base_beta: np.ndarray, target_unfold_t: torch.Tensor) -> dict[str, object]:
    rng = np.random.default_rng(20260402)
    alpha_init = base_alpha + WARM_NOISE * rng.standard_normal(base_alpha.shape)
    beta_init = base_beta + WARM_NOISE * rng.standard_normal(base_beta.shape)
    alpha_batch, beta_batch, gamma_batch, losses, max_abs = optimize_batch(
        alpha_init[None, ...],
        beta_init[None, ...],
        steps=WARM_ADAM_STEPS,
        learning_rate=WARM_LR,
        target_unfold_t=target_unfold_t,
    )
    terms = terms_from_stacked_factors(alpha_batch[0], beta_batch[0], gamma_batch[0], prefix='warm23_')
    profile = profile_terms(terms, rank_tol=1e-8, projection_tol=1e-6)
    return {
        'strategy': 'warm_alphatensor_plus_noise',
        'seed': 20260402,
        'adam_steps': WARM_ADAM_STEPS,
        'learning_rate': WARM_LR,
        'noise_sigma': WARM_NOISE,
        'final_loss': float(losses[0]),
        'final_max_abs': float(max_abs[0]),
        'pass_lt_1e_10': bool(max_abs[0] < HIT_TOL),
        'pass_lt_1e_20': bool(float(losses[0]) < EXACT_TOL),
        'rank_H_numeric': profile['rank_H_numeric'],
        'rank_Nuisance_numeric': profile['rank_Nuisance_numeric'],
        'quotient_gain_numeric': profile['quotient_gain_numeric'],
        'delta_in_eta_numeric': profile['delta_in_eta_numeric'],
    }


def cold_start_runs(target_unfold_t: torch.Tensor) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    saved: list[dict[str, object]] = []
    completed = 0
    while completed < COLD_RESTARTS:
        batch_size = min(COLD_BATCH_SIZE, COLD_RESTARTS - completed)
        rng = np.random.default_rng(730000 + completed)
        alpha_init = rng.normal(0.0, INIT_SCALE, size=(batch_size, RANK, 9))
        beta_init = rng.normal(0.0, INIT_SCALE, size=(batch_size, RANK, 9))
        alpha_batch, beta_batch, gamma_batch, losses, max_abs = optimize_batch(
            alpha_init,
            beta_init,
            steps=COLD_ADAM_STEPS,
            learning_rate=COLD_LR,
            target_unfold_t=target_unfold_t,
        )
        for local_idx in range(batch_size):
            run_id = completed + local_idx
            row = {
                'restart_id': run_id,
                'seed': 730000 + completed + local_idx,
                'final_loss': float(losses[local_idx]),
                'final_max_abs': float(max_abs[local_idx]),
                'pass_lt_1e_10': bool(max_abs[local_idx] < HIT_TOL),
                'pass_lt_1e_20': bool(float(losses[local_idx]) < EXACT_TOL),
            }
            rows.append(row)
            saved.append(
                {
                    'row': row,
                    'terms': terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=f'cold23_{run_id:04d}_'),
                }
            )
        completed += batch_size
    return rows, saved


def summarize_cold_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    loss_values = sorted(float(row['final_loss']) for row in rows)
    max_abs_values = sorted(float(row['final_max_abs']) for row in rows)
    hit_1e10 = sum(1 for row in rows if bool(row['pass_lt_1e_10']))
    hit_1e20 = sum(1 for row in rows if bool(row['pass_lt_1e_20']))
    median_idx = len(rows) // 2
    return {
        'count': len(rows),
        'hit_count_lt_1e_10': hit_1e10,
        'hit_rate_lt_1e_10': hit_1e10 / len(rows) if rows else 0.0,
        'hit_count_lt_1e_20': hit_1e20,
        'hit_rate_lt_1e_20': hit_1e20 / len(rows) if rows else 0.0,
        'median_final_loss': float(loss_values[median_idx]) if rows else math.nan,
        'median_final_max_abs': float(max_abs_values[median_idx]) if rows else math.nan,
        'best_final_loss': float(loss_values[0]) if rows else math.nan,
        'best_final_max_abs': float(max_abs_values[0]) if rows else math.nan,
    }


def build_results_markdown(target_validation: dict[str, object], warm_row: dict[str, object], cold_summary: dict[str, object], best_cold: dict[str, object] | None, best_profile: dict[str, object] | None) -> str:
    lines: list[str] = []
    lines.append('# Phase 10 Results Through 10c')
    lines.append('')
    lines.append('## 10a-10b. Factored Loss and Tensor Construction')
    lines.append('')
    lines.append(f"- target unfolding shape: {target_validation['shape']}")
    lines.append(f"- target unfolding nonzero count: {target_validation['nonzero_count']}")
    lines.append(f"- target unfolding nonzero values: {target_validation['unique_values']}")
    lines.append('')
    lines.append('## 10c. R=23 Baseline')
    lines.append('')
    lines.append(f"- warm start final loss: {warm_row['final_loss']}")
    lines.append(f"- warm start final max-abs residual: {warm_row['final_max_abs']}")
    lines.append(f"- warm start passes `<1e-10`: {warm_row['pass_lt_1e_10']}")
    lines.append(f"- warm start passes `<1e-20`: {warm_row['pass_lt_1e_20']}")
    lines.append(f"- cold-start runs: {cold_summary['count']}")
    lines.append(f"- cold-start success rate `<1e-10`: {cold_summary['hit_rate_lt_1e_10']:.3f}")
    lines.append(f"- cold-start success rate `<1e-20`: {cold_summary['hit_rate_lt_1e_20']:.3f}")
    lines.append(f"- cold-start median final loss: {cold_summary['median_final_loss']}")
    lines.append(f"- cold-start best final loss: {cold_summary['best_final_loss']}")
    lines.append(f"- cold-start best final max-abs residual: {cold_summary['best_final_max_abs']}")
    if best_profile is not None:
        lines.append(f"- best cold candidate rank(H)_numeric: {best_profile['rank_H_numeric']}")
        lines.append(f"- best cold candidate quotient_gain_numeric: {best_profile['quotient_gain_numeric']}")
        lines.append(f"- best cold candidate delta_in_eta_numeric: {best_profile['delta_in_eta_numeric']}")
    lines.append('')
    lines.append('## Calibration Verdict')
    lines.append('')
    if warm_row['pass_lt_1e_20'] and cold_summary['hit_rate_lt_1e_10'] >= 0.05:
        lines.append('- The factored-loss baseline is calibrated: warm start closes and cold starts hit often enough to make R=22 negatives interpretable.')
    else:
        lines.append('- The factored-loss baseline is not yet calibrated. Either the warm start did not return to exactness or the cold-start hit rate stayed below the 5% target, so R=22 negatives would still be untrustworthy under this optimizer.')
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_unfold = target_output_unfolding_9x81()
    target_validation = build_target_validation(target_unfold)
    target_unfold_t = torch.tensor(target_unfold.T, dtype=torch.float64)

    base_terms, orientation, loader_residual = load_public_terms()
    base_alpha, base_beta, _ = stacked_factors_from_terms(base_terms)
    warm_row = warm_start_run(base_alpha, base_beta, target_unfold_t)
    cold_rows, cold_saved = cold_start_runs(target_unfold_t)
    cold_summary = summarize_cold_rows(cold_rows)

    cold_rows_sorted = sorted(cold_rows, key=lambda row: (float(row['final_max_abs']), float(row['final_loss'])))
    best_profile = None
    best_cold = cold_rows_sorted[0] if cold_rows_sorted else None
    if best_cold is not None:
        best_saved = min(cold_saved, key=lambda item: (float(item['row']['final_max_abs']), float(item['row']['final_loss'])))
        best_profile = profile_terms(best_saved['terms'], rank_tol=1e-8, projection_tol=1e-6)

    write_csv(
        OUT_DIR / 'baseline_results.csv',
        cold_rows,
        ['restart_id', 'seed', 'final_loss', 'final_max_abs', 'pass_lt_1e_10', 'pass_lt_1e_20'],
    )
    write_json(
        OUT_DIR / 'baseline_summary.json',
        {
            'orientation': orientation,
            'loader_residual': loader_residual,
            'target_validation': target_validation,
            'warm_start': warm_row,
            'cold_start': cold_summary,
            'best_cold_restart': best_cold,
            'best_cold_profile': best_profile,
            'configuration': {
                'warm_noise': WARM_NOISE,
                'warm_adam_steps': WARM_ADAM_STEPS,
                'cold_restarts': COLD_RESTARTS,
                'cold_adam_steps': COLD_ADAM_STEPS,
                'cold_batch_size': COLD_BATCH_SIZE,
                'warm_learning_rate': WARM_LR,
                'cold_learning_rate': COLD_LR,
                'init_scale': INIT_SCALE,
            },
        },
    )
    (OUT_DIR / 'RESULTS.md').write_text(
        build_results_markdown(target_validation, warm_row, cold_summary, best_cold, best_profile),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()