from __future__ import annotations

import json
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
    ACTIONS,
    TARGET_TENSOR,
    apply_action_to_terms,
    load_public_terms,
    serialize_decomposition,
    stacked_factors_from_terms,
    tensor_residual_stats,
    terms_from_stacked_factors,
    write_csv,
    write_json,
    write_jsonl,
)


torch.set_default_dtype(torch.float64)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


PERTURB_SIGMAS = [0.01, 0.05, 0.1, 0.5]
PERTURB_BATCH_SIZE = env_int('ADE3X3_PHASE3_PERTURB_BATCH_SIZE', 8)
PERTURB_MAX_ATTEMPTS = env_int('ADE3X3_PHASE3_PERTURB_MAX_ATTEMPTS', 128)
PERTURB_TARGET_VALID = env_int('ADE3X3_PHASE3_PERTURB_TARGET_VALID', 100)
PERTURB_ADAM_STEPS = env_int('ADE3X3_PHASE3_PERTURB_ADAM_STEPS', 1600)
PERTURB_ADAM_LR = env_float('ADE3X3_PHASE3_PERTURB_ADAM_LR', 3e-3)
PERTURB_GAMMA_REFRESH = env_int('ADE3X3_PHASE3_PERTURB_GAMMA_REFRESH', 200)
PERTURB_EXACT_TOL = env_float('ADE3X3_PHASE3_PERTURB_EXACT_TOL', 1e-10)

RANDOM_BATCH_SIZE = env_int('ADE3X3_PHASE3_RANDOM_BATCH_SIZE', 16)
RANDOM_RESTARTS = env_int('ADE3X3_PHASE3_RANDOM_RESTARTS', 512)
RANDOM_ADAM_STEPS = env_int('ADE3X3_PHASE3_RANDOM_ADAM_STEPS', 2200)
RANDOM_ADAM_LR = env_float('ADE3X3_PHASE3_RANDOM_ADAM_LR', 5e-3)
RANDOM_GAMMA_REFRESH = env_int('ADE3X3_PHASE3_RANDOM_GAMMA_REFRESH', 250)
RANDOM_INIT_SCALE = env_float('ADE3X3_PHASE3_RANDOM_INIT_SCALE', 0.2)
RANDOM_EXACT_TOL = env_float('ADE3X3_PHASE3_RANDOM_EXACT_TOL', 1e-10)


def gamma_fit_batch(alpha: torch.Tensor, beta: torch.Tensor, target81: torch.Tensor) -> torch.Tensor:
    batch_size, rank, _ = alpha.shape
    features = torch.einsum('xra,xrb->xabr', alpha, beta).reshape(batch_size, 81, rank)
    return torch.linalg.lstsq(features, target81.expand(batch_size, -1, -1)).solution


def optimize_batch(alpha_init: np.ndarray, beta_init: np.ndarray, gamma_init: np.ndarray, steps: int, lr: float, gamma_refresh: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = torch.tensor(TARGET_TENSOR, dtype=torch.float64)
    target81 = target.reshape(81, 9)
    alpha = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float64))
    gamma = torch.nn.Parameter(torch.tensor(gamma_init, dtype=torch.float64))

    optimizer = torch.optim.Adam([alpha, beta, gamma], lr=lr)
    for step in range(steps):
        if step and step % gamma_refresh == 0:
            with torch.no_grad():
                gamma.copy_(gamma_fit_batch(alpha, beta, target81))
        optimizer.zero_grad()
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        losses = (residual * residual).sum(dim=(1, 2, 3))
        losses.mean().backward()
        optimizer.step()

    with torch.no_grad():
        gamma.copy_(gamma_fit_batch(alpha, beta, target81))
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        max_abs = residual.abs().amax(dim=(1, 2, 3)).cpu().numpy()
        losses = (residual * residual).sum(dim=(1, 2, 3)).cpu().numpy()
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy(), gamma.detach().cpu().numpy(), max_abs, losses


def generate_symmetry_orbit_rows(base_terms) -> tuple[list[dict], list[dict]]:
    exact_rows: list[dict] = []
    attempt_rows: list[dict] = []
    for action_idx, action in enumerate(ACTIONS):
        transformed_terms = apply_action_to_terms(base_terms, action)
        max_abs, loss_value = tensor_residual_stats(transformed_terms)
        decomposition_id = f'sym_{action_idx:03d}'
        exact_rows.append(
            serialize_decomposition(
                decomposition_id=decomposition_id,
                source='symmetry_orbit',
                source_detail=f'action_id={action_idx};{action}',
                terms=transformed_terms,
                max_abs_residual=max_abs,
                loss_value=loss_value,
                metadata={'action_id': action_idx, 'action': [list(part) for part in action]},
            )
        )
        attempt_rows.append(
            {
                'source': 'symmetry_orbit',
                'source_detail': f'action_id={action_idx}',
                'attempt_id': decomposition_id,
                'sigma': '',
                'restart_seed': '',
                'rank': len(transformed_terms),
                'max_abs_residual': max_abs,
                'loss_value': loss_value,
                'exact_hit': max_abs <= 0.0,
            }
        )
    return exact_rows, attempt_rows


def generate_perturbation_rows(base_terms) -> tuple[list[dict], list[dict], list[dict]]:
    exact_rows: list[dict] = []
    attempt_rows: list[dict] = []
    summary_rows: list[dict] = []
    base_alpha, base_beta, base_gamma = stacked_factors_from_terms(base_terms)
    attempt_counter = 0

    for sigma in PERTURB_SIGMAS:
        valid_count = 0
        best_residual = float('inf')
        total_attempts = 0
        rng = np.random.default_rng(7300 + int(1000 * sigma))
        while valid_count < PERTURB_TARGET_VALID and total_attempts < PERTURB_MAX_ATTEMPTS:
            batch_size = min(PERTURB_BATCH_SIZE, PERTURB_MAX_ATTEMPTS - total_attempts)
            alpha_init = np.repeat(base_alpha[None, :, :], batch_size, axis=0)
            beta_init = np.repeat(base_beta[None, :, :], batch_size, axis=0)
            gamma_init = np.repeat(base_gamma[None, :, :], batch_size, axis=0)
            alpha_init += sigma * rng.standard_normal(alpha_init.shape)
            beta_init += sigma * rng.standard_normal(beta_init.shape)
            gamma_init = np.tile(base_gamma[None, :, :], (batch_size, 1, 1))
            alpha_batch, beta_batch, gamma_batch, residuals, losses = optimize_batch(
                alpha_init,
                beta_init,
                gamma_init,
                steps=PERTURB_ADAM_STEPS,
                lr=PERTURB_ADAM_LR,
                gamma_refresh=PERTURB_GAMMA_REFRESH,
            )
            for local_idx in range(batch_size):
                decomposition_id = f'perturb_sigma_{sigma:.2f}_{attempt_counter:05d}'
                exact_hit = bool(residuals[local_idx] <= PERTURB_EXACT_TOL)
                best_residual = min(best_residual, float(residuals[local_idx]))
                attempt_rows.append(
                    {
                        'source': 'perturbation',
                        'source_detail': f'sigma={sigma}',
                        'attempt_id': decomposition_id,
                        'sigma': sigma,
                        'restart_seed': 7300 + int(1000 * sigma) + attempt_counter,
                        'rank': base_alpha.shape[0],
                        'max_abs_residual': float(residuals[local_idx]),
                        'loss_value': float(losses[local_idx]),
                        'exact_hit': exact_hit,
                    }
                )
                if exact_hit:
                    valid_count += 1
                    terms = terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=f'p{attempt_counter:02d}_')
                    exact_rows.append(
                        serialize_decomposition(
                            decomposition_id=decomposition_id,
                            source='perturbation',
                            source_detail=f'sigma={sigma}',
                            terms=terms,
                            max_abs_residual=float(residuals[local_idx]),
                            loss_value=float(losses[local_idx]),
                            metadata={'sigma': sigma, 'restart_seed': 7300 + int(1000 * sigma) + attempt_counter},
                        )
                    )
                attempt_counter += 1
            total_attempts += batch_size

        summary_rows.append(
            {
                'source': 'perturbation',
                'sigma': sigma,
                'attempts': total_attempts,
                'valid_exact_decompositions': valid_count,
                'target_valid_exact_decompositions': PERTURB_TARGET_VALID,
                'best_max_abs_residual': None if best_residual == float('inf') else best_residual,
                'met_target': valid_count >= PERTURB_TARGET_VALID,
            }
        )

    return exact_rows, attempt_rows, summary_rows


def generate_random_rows() -> tuple[list[dict], list[dict], list[dict]]:
    exact_rows: list[dict] = []
    attempt_rows: list[dict] = []
    summary_rows: list[dict] = []
    rank = 23
    rng = np.random.default_rng(910000)
    completed = 0
    hit_count = 0
    best_residual = float('inf')
    while completed < RANDOM_RESTARTS:
        batch_size = min(RANDOM_BATCH_SIZE, RANDOM_RESTARTS - completed)
        alpha_init = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
        beta_init = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
        gamma_init = RANDOM_INIT_SCALE * rng.standard_normal((batch_size, rank, 9))
        alpha_batch, beta_batch, gamma_batch, residuals, losses = optimize_batch(
            alpha_init,
            beta_init,
            gamma_init,
            steps=RANDOM_ADAM_STEPS,
            lr=RANDOM_ADAM_LR,
            gamma_refresh=RANDOM_GAMMA_REFRESH,
        )
        for local_idx in range(batch_size):
            restart_idx = completed + local_idx
            decomposition_id = f'random_rank23_{restart_idx:05d}'
            exact_hit = bool(residuals[local_idx] <= RANDOM_EXACT_TOL)
            hit_count += int(exact_hit)
            best_residual = min(best_residual, float(residuals[local_idx]))
            attempt_rows.append(
                {
                    'source': 'random_rank23',
                    'source_detail': 'cold_random',
                    'attempt_id': decomposition_id,
                    'sigma': '',
                    'restart_seed': 910000 + restart_idx,
                    'rank': rank,
                    'max_abs_residual': float(residuals[local_idx]),
                    'loss_value': float(losses[local_idx]),
                    'exact_hit': exact_hit,
                }
            )
            if exact_hit:
                terms = terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=f'r{restart_idx:02d}_')
                exact_rows.append(
                    serialize_decomposition(
                        decomposition_id=decomposition_id,
                        source='random_rank23',
                        source_detail='cold_random',
                        terms=terms,
                        max_abs_residual=float(residuals[local_idx]),
                        loss_value=float(losses[local_idx]),
                        metadata={'restart_seed': 910000 + restart_idx},
                    )
                )
        completed += batch_size

    summary_rows.append(
        {
            'source': 'random_rank23',
            'sigma': '',
            'attempts': RANDOM_RESTARTS,
            'valid_exact_decompositions': hit_count,
            'target_valid_exact_decompositions': '',
            'best_max_abs_residual': None if best_residual == float('inf') else best_residual,
            'met_target': '',
        }
    )
    return exact_rows, attempt_rows, summary_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_terms, orientation, residual = load_public_terms()

    exact_rows: list[dict] = []
    attempt_rows: list[dict] = []
    generation_summary_rows: list[dict] = []

    symmetry_exact_rows, symmetry_attempt_rows = generate_symmetry_orbit_rows(base_terms)
    exact_rows.extend(symmetry_exact_rows)
    attempt_rows.extend(symmetry_attempt_rows)
    generation_summary_rows.append(
        {
            'source': 'symmetry_orbit',
            'sigma': '',
            'attempts': len(symmetry_attempt_rows),
            'valid_exact_decompositions': len(symmetry_exact_rows),
            'target_valid_exact_decompositions': len(symmetry_exact_rows),
            'best_max_abs_residual': 0.0,
            'met_target': True,
        }
    )

    perturb_exact_rows, perturb_attempt_rows, perturb_summary_rows = generate_perturbation_rows(base_terms)
    exact_rows.extend(perturb_exact_rows)
    attempt_rows.extend(perturb_attempt_rows)
    generation_summary_rows.extend(perturb_summary_rows)

    random_exact_rows, random_attempt_rows, random_summary_rows = generate_random_rows()
    exact_rows.extend(random_exact_rows)
    attempt_rows.extend(random_attempt_rows)
    generation_summary_rows.extend(random_summary_rows)

    write_jsonl(OUT_DIR / 'generated_decompositions.jsonl', exact_rows)
    write_csv(
        OUT_DIR / 'generation_attempts.csv',
        attempt_rows,
        ['source', 'source_detail', 'attempt_id', 'sigma', 'restart_seed', 'rank', 'max_abs_residual', 'loss_value', 'exact_hit'],
    )
    write_csv(
        OUT_DIR / 'generation_summary.csv',
        generation_summary_rows,
        ['source', 'sigma', 'attempts', 'valid_exact_decompositions', 'target_valid_exact_decompositions', 'best_max_abs_residual', 'met_target'],
    )
    write_json(
        OUT_DIR / 'generation_metadata.json',
        {
            'base_source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
            'gamma_orientation': orientation,
            'exact_reconstruction_residual': int(residual),
            'symmetry_actions': len(ACTIONS),
            'perturb_sigmas': PERTURB_SIGMAS,
            'perturb_target_valid': PERTURB_TARGET_VALID,
            'perturb_max_attempts': PERTURB_MAX_ATTEMPTS,
            'random_restarts': RANDOM_RESTARTS,
            'exact_decompositions_written': len(exact_rows),
        },
    )

    print('Phase 3 generation complete')
    print(f'  exact decompositions written = {len(exact_rows)}')
    print(f'  symmetry exact rows = {len(symmetry_exact_rows)}')
    print(f'  perturbation exact rows = {len(perturb_exact_rows)}')
    print(f'  random exact rows = {len(random_exact_rows)}')


if __name__ == '__main__':
    main()