from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch


OUT_DIR = Path(__file__).resolve().parent
BEST_DIR = OUT_DIR / 'best_decompositions'
REPO_ROOT = OUT_DIR.parents[1]
PHASE4_DIR = REPO_ROOT / 'outputs' / 'ade3x3_attack' / 'phase4_gradient_search'
ATTACK_DIR = REPO_ROOT / 'outputs' / 'ade3x3_attack'
if str(PHASE4_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIR))
if str(ATTACK_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACK_DIR))

from attack_common import (  # noqa: E402
    TARGET_TENSOR,
    load_public_terms,
    serialize_decomposition,
    stacked_factors_from_terms,
    terms_from_stacked_factors,
    write_csv,
    write_json,
)
from search_r22 import candidate_row, optimize_batch, make_random_inits  # noqa: E402


torch.set_default_dtype(torch.float64)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


WARM_STRUCTURED_RESTARTS = env_int('ADE3X3_WING_IT_WARM_STRUCTURED_RESTARTS', 2)
WARM_RANDOM_MASKS = env_int('ADE3X3_WING_IT_WARM_RANDOM_MASKS', 64)
WARM_RANDOM_RESTARTS = env_int('ADE3X3_WING_IT_WARM_RANDOM_RESTARTS', 1)
BORDER23_TO_19_RESTARTS = env_int('ADE3X3_WING_IT_BORDER23_TO_19_RESTARTS', 24)
COLD19_RESTARTS = env_int('ADE3X3_WING_IT_COLD19_RESTARTS', 64)
COLD18_RESTARTS = env_int('ADE3X3_WING_IT_COLD18_RESTARTS', 24)
SAVE_TOP_K = env_int('ADE3X3_WING_IT_SAVE_TOP_K', 12)
TAIL_PENALTY = env_float('ADE3X3_WING_IT_TAIL_PENALTY', 3e-2)
RANDOM_NOISE = env_float('ADE3X3_WING_IT_WARM_NOISE', 0.02)
EXACT_TOL = env_float('ADE3X3_WING_IT_EXACT_TOL', 1e-10)

PAIR_REMOVALS = [('t03', 't06'), ('t09', 't12'), ('t15', 't17')]


def optimize_batch_multi_tail(
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    gamma_init: np.ndarray,
    tail_terms: int,
    tail_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import search_r22 as phase4  # noqa: WPS433

    target = torch.tensor(TARGET_TENSOR, dtype=torch.float64)
    target81 = target.reshape(81, 9)
    alpha = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float64))
    gamma = torch.nn.Parameter(torch.tensor(gamma_init, dtype=torch.float64))

    optimizer = torch.optim.Adam([alpha, beta, gamma], lr=phase4.ADAM_LR)
    for step in range(phase4.ADAM_STEPS):
        if step and step % phase4.GAMMA_REFRESH == 0:
            with torch.no_grad():
                gamma.copy_(phase4.gamma_fit_batch(alpha, beta, target81))
        optimizer.zero_grad()
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        losses = (residual * residual).sum(dim=(1, 2, 3))
        tail_slice = gamma[:, -tail_terms:, :]
        losses = losses + tail_penalty * (tail_slice * tail_slice).sum(dim=(1, 2))
        losses.mean().backward()
        optimizer.step()

    with torch.no_grad():
        gamma.copy_(phase4.gamma_fit_batch(alpha, beta, target81))
        approx = torch.einsum('xra,xrb,xrc->xabc', alpha, beta, gamma)
        residual = approx - target.unsqueeze(0)
        max_abs = residual.abs().amax(dim=(1, 2, 3)).cpu().numpy()
        losses = (residual * residual).sum(dim=(1, 2, 3)).cpu().numpy()
        tail_norms = np.linalg.norm(gamma[:, -tail_terms:, :].cpu().numpy().reshape(gamma.shape[0], -1), axis=1)
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy(), gamma.detach().cpu().numpy(), max_abs, losses, tail_norms


def term_id_map(rank: int) -> list[str]:
    return [f't{idx + 1:02d}' for idx in range(rank)]


def structured_removal_specs(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    ids = term_id_map(base_alpha.shape[0])
    id_to_idx = {term_id: idx for idx, term_id in enumerate(ids)}
    protected = {id_to_idx[left] for left, _ in PAIR_REMOVALS} | {id_to_idx[right] for _, right in PAIR_REMOVALS}
    extras = [idx for idx in range(base_alpha.shape[0]) if idx not in protected]
    rng = np.random.default_rng(20260330)
    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    seed_base = 910000
    mask_idx = 0
    for bits in product([0, 1], repeat=len(PAIR_REMOVALS)):
        removed = set()
        labels: list[str] = []
        for pair_bit, (left_id, right_id) in zip(bits, PAIR_REMOVALS, strict=True):
            chosen = left_id if pair_bit == 0 else right_id
            removed.add(id_to_idx[chosen])
            labels.append(chosen)
        for extra_idx in extras:
            keep = [idx for idx in range(base_alpha.shape[0]) if idx not in removed | {extra_idx}]
            for restart_idx in range(WARM_STRUCTURED_RESTARTS):
                alpha = base_alpha[keep].copy()
                beta = base_beta[keep].copy()
                gamma = base_gamma[keep].copy()
                alpha += RANDOM_NOISE * rng.standard_normal(alpha.shape)
                beta += RANDOM_NOISE * rng.standard_normal(beta.shape)
                gamma += RANDOM_NOISE * rng.standard_normal(gamma.shape)
                detail = f'structured_remove_{"_".join(labels)}_plus_t{extra_idx + 1:02d}'
                specs.append((detail, alpha, beta, gamma, seed_base + 1000 * mask_idx + restart_idx))
            mask_idx += 1
    return specs


def random_mask_specs(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(20260331)
    seen: set[tuple[int, int, int, int]] = set()
    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    trial = 0
    while len(seen) < WARM_RANDOM_MASKS:
        removed = tuple(sorted(int(x) for x in rng.choice(base_alpha.shape[0], size=4, replace=False)))
        if removed in seen:
            continue
        seen.add(removed)
        keep = [idx for idx in range(base_alpha.shape[0]) if idx not in removed]
        for restart_idx in range(WARM_RANDOM_RESTARTS):
            alpha = base_alpha[keep].copy()
            beta = base_beta[keep].copy()
            gamma = base_gamma[keep].copy()
            alpha += RANDOM_NOISE * rng.standard_normal(alpha.shape)
            beta += RANDOM_NOISE * rng.standard_normal(beta.shape)
            gamma += RANDOM_NOISE * rng.standard_normal(gamma.shape)
            detail = 'random_remove_' + '_'.join(f't{idx + 1:02d}' for idx in removed)
            specs.append((detail, alpha, beta, gamma, 920000 + 100 * trial + restart_idx))
        trial += 1
    return specs


def border23_to_19_specs(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(20260332)
    specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for restart_idx in range(BORDER23_TO_19_RESTARTS):
        perm = rng.permutation(base_alpha.shape[0])
        alpha = base_alpha[perm].copy()
        beta = base_beta[perm].copy()
        gamma = base_gamma[perm].copy()
        alpha += RANDOM_NOISE * rng.standard_normal(alpha.shape)
        beta += RANDOM_NOISE * rng.standard_normal(beta.shape)
        gamma += RANDOM_NOISE * rng.standard_normal(gamma.shape)
        tail = '_'.join(f't{idx + 1:02d}' for idx in perm[-4:])
        specs.append((f'border_tail_{tail}', alpha, beta, gamma, 930000 + restart_idx))
    return specs


def run_warm_truncation(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    specs = structured_removal_specs(base_alpha, base_beta, base_gamma) + random_mask_specs(base_alpha, base_beta, base_gamma)
    if not specs:
        return rows, saved
    import search_r22 as phase4  # noqa: WPS433
    batch_size = phase4.BATCH_SIZE
    for start in range(0, len(specs), batch_size):
        chunk = specs[start:start + batch_size]
        alpha_init = np.stack([item[1] for item in chunk])
        beta_init = np.stack([item[2] for item in chunk])
        gamma_init = np.stack([item[3] for item in chunk])
        alpha_batch, beta_batch, gamma_batch, residuals, losses, _ = optimize_batch(alpha_init, beta_init, gamma_init, rank=19)
        for local_idx, (detail, _, _, _, seed) in enumerate(chunk):
            search_id = f'wing19_warm_{start + local_idx:04d}'
            row = candidate_row(
                search_id,
                19,
                'warm_truncation',
                detail,
                seed,
                alpha_batch[local_idx],
                beta_batch[local_idx],
                gamma_batch[local_idx],
                float(residuals[local_idx]),
                float(losses[local_idx]),
            )
            rows.append(row)
            saved.append({'row': row, 'terms': terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=search_id)})
    return rows, saved


def run_border23_to_19(base_alpha: np.ndarray, base_beta: np.ndarray, base_gamma: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    specs = border23_to_19_specs(base_alpha, base_beta, base_gamma)
    if not specs:
        return rows, saved
    import search_r22 as phase4  # noqa: WPS433
    batch_size = phase4.BATCH_SIZE
    for start in range(0, len(specs), batch_size):
        chunk = specs[start:start + batch_size]
        alpha_init = np.stack([item[1] for item in chunk])
        beta_init = np.stack([item[2] for item in chunk])
        gamma_init = np.stack([item[3] for item in chunk])
        alpha_batch, beta_batch, gamma_batch, residuals, losses, tail_norms = optimize_batch_multi_tail(
            alpha_init,
            beta_init,
            gamma_init,
            tail_terms=4,
            tail_penalty=TAIL_PENALTY,
        )
        for local_idx, (detail, _, _, _, seed) in enumerate(chunk):
            truncated_alpha = alpha_batch[local_idx][:-4]
            truncated_beta = beta_batch[local_idx][:-4]
            truncated_gamma = gamma_batch[local_idx][:-4]
            search_id = f'wing19_border_{start + local_idx:04d}'
            row = candidate_row(
                search_id,
                19,
                'border23_to_19',
                detail,
                seed,
                truncated_alpha,
                truncated_beta,
                truncated_gamma,
                float(residuals[local_idx]),
                float(losses[local_idx]),
                float(tail_norms[local_idx]),
            )
            rows.append(row)
            saved.append({'row': row, 'terms': terms_from_stacked_factors(truncated_alpha, truncated_beta, truncated_gamma, prefix=search_id)})
    return rows, saved


def run_cold(rank: int, restarts: int, prefix: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    saved: list[dict] = []
    if restarts <= 0:
        return rows, saved
    import search_r22 as phase4  # noqa: WPS433
    batch_size = phase4.BATCH_SIZE
    completed = 0
    while completed < restarts:
        size = min(batch_size, restarts - completed)
        alpha_init, beta_init, gamma_init = make_random_inits(size, rank, 940000 + 1000 * rank + completed)
        alpha_batch, beta_batch, gamma_batch, residuals, losses, _ = optimize_batch(alpha_init, beta_init, gamma_init, rank=rank)
        for local_idx in range(size):
            search_id = f'{prefix}_{completed + local_idx:04d}'
            row = candidate_row(
                search_id,
                rank,
                'cold_random',
                prefix,
                940000 + 1000 * rank + completed + local_idx,
                alpha_batch[local_idx],
                beta_batch[local_idx],
                gamma_batch[local_idx],
                float(residuals[local_idx]),
                float(losses[local_idx]),
            )
            rows.append(row)
            saved.append({'row': row, 'terms': terms_from_stacked_factors(alpha_batch[local_idx], beta_batch[local_idx], gamma_batch[local_idx], prefix=search_id)})
        completed += size
    return rows, saved


def build_results_markdown(rows: list[dict], stats: dict) -> str:
    lines: list[str] = []
    lines.append('# Wing It Compression Attempt')
    lines.append('')
    lines.append('## Scope')
    lines.append('')
    lines.append('- Target ranks attempted: 19 and 18')
    lines.append('- Warm start source: public AlphaTensor rank-23 decomposition')
    lines.append('- Search families: structured warm truncation, random warm truncation, border 23->19, cold random')
    lines.append('')
    lines.append('## Best Results')
    lines.append('')
    for rank_label in ['18', '19']:
        best = stats['best_by_rank'].get(rank_label)
        if best is None:
            continue
        lines.append(f"- R={rank_label}: best max-abs residual = {best['final_max_abs']} via {best['strategy']} ({best['init_detail']})")
        lines.append(f"  quotient gain = {best['quotient_gain_numeric']}, rank(H) = {best['rank_H_numeric']}, rank(Nuisance) = {best['rank_Nuisance_numeric']}")
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    if stats['exact_hits_total'] == 0:
        lines.append('- No exact hit at rank 19 or 18 in this pass.')
        lines.append('- The exported best candidates are still useful as diagnostics for whether warm truncation or border-tail continuation is the less bad route.')
    else:
        lines.append(f"- Exact hits found: {stats['exact_hits_total']} candidates at tolerance {EXACT_TOL}.")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    base_terms, orientation, residual = load_public_terms()
    base_alpha, base_beta, base_gamma = stacked_factors_from_terms(base_terms)

    rows: list[dict] = []
    saved: list[dict] = []

    warm_rows, warm_saved = run_warm_truncation(base_alpha, base_beta, base_gamma)
    rows.extend(warm_rows)
    saved.extend(warm_saved)

    border_rows, border_saved = run_border23_to_19(base_alpha, base_beta, base_gamma)
    rows.extend(border_rows)
    saved.extend(border_saved)

    cold19_rows, cold19_saved = run_cold(19, COLD19_RESTARTS, 'wing19_cold')
    rows.extend(cold19_rows)
    saved.extend(cold19_saved)

    cold18_rows, cold18_saved = run_cold(18, COLD18_RESTARTS, 'wing18_cold')
    rows.extend(cold18_rows)
    saved.extend(cold18_saved)

    rows.sort(key=lambda row: (int(row['rank']), float(row['final_max_abs']), float(row['final_loss'])))
    write_csv(
        OUT_DIR / 'attempt_results.csv',
        rows,
        [
            'search_id', 'rank', 'strategy', 'init_detail', 'seed', 'final_loss', 'final_max_abs',
            'exact_hit', 'rank_H_numeric', 'rank_Delta_numeric', 'rank_Nuisance_numeric',
            'quotient_gain_numeric', 'delta_in_eta_numeric', 'projection_max_residual_numeric',
            'border_tail_norm',
        ],
    )

    saved.sort(key=lambda item: (float(item['row']['final_max_abs']), float(item['row']['final_loss'])))
    for save_idx, item in enumerate(saved[:SAVE_TOP_K], start=1):
        payload = serialize_decomposition(
            decomposition_id=item['row']['search_id'],
            source='experiments_wing_it',
            source_detail=item['row']['strategy'],
            terms=item['terms'],
            max_abs_residual=float(item['row']['final_max_abs']),
            loss_value=float(item['row']['final_loss']),
            metadata=item['row'],
        )
        (BEST_DIR / f'{save_idx:02d}_{item["row"]["search_id"]}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    best_by_rank: dict[str, dict] = {}
    for rank in [18, 19]:
        rank_rows = [row for row in rows if int(row['rank']) == rank]
        if rank_rows:
            best_by_rank[str(rank)] = min(rank_rows, key=lambda row: (float(row['final_max_abs']), float(row['final_loss'])))

    stats = {
        'base_source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
        'gamma_orientation': orientation,
        'exact_reconstruction_residual': int(residual),
        'warm_structured_restarts': WARM_STRUCTURED_RESTARTS,
        'warm_random_masks': WARM_RANDOM_MASKS,
        'warm_random_restarts': WARM_RANDOM_RESTARTS,
        'border23_to_19_restarts': BORDER23_TO_19_RESTARTS,
        'cold19_restarts': COLD19_RESTARTS,
        'cold18_restarts': COLD18_RESTARTS,
        'tail_penalty': TAIL_PENALTY,
        'total_runs': len(rows),
        'exact_hits_total': sum(int(bool(row['exact_hit'])) for row in rows),
        'best_by_rank': best_by_rank,
    }
    write_json(OUT_DIR / 'summary.json', stats)
    (OUT_DIR / 'RESULTS.md').write_text(build_results_markdown(rows, stats), encoding='utf-8')

    print('Wing-it compression attempt complete')
    print(f"  total runs = {len(rows)}")
    for rank in [18, 19]:
        best = best_by_rank.get(str(rank))
        if best is not None:
            print(f"  best R={rank} max abs = {best['final_max_abs']} via {best['strategy']}")


if __name__ == '__main__':
    main()