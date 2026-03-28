"""
ade3x3_step50_overnight_search.py

Step 50: Overnight Algorithm Search.

This is a search step, not a catalog step. It runs repeated float64 gradient
descent restarts against the exact 729-equation tensor system for 3x3 matrix
product decomposition. Results are logged to outputs/exports/search_results/
and are intentionally kept out of the canonical dossier until independently
verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch missing
    raise SystemExit(
        "PyTorch is required for Step 50. Install it into the repo virtual environment before running this script."
    ) from exc


EXPORTS = Path('outputs/exports/search_results')
DEFAULT_SCHEDULE = [
    (9, 5000),
    (10, 5000),
    (11, 5000),
    (12, 5000),
    (13, 5000),
    (14, 10000),
    (15, 10000),
    (16, 10000),
    (17, 10000),
    (18, 20000),
    (19, 100000),
    (20, 80000),
    (21, 100000),
    (22, 100000),
    (23, 20000),
]


def build_target() -> torch.Tensor:
    target = torch.zeros((9, 9, 9), dtype=torch.float64)
    for row_idx in range(3):
        for sum_idx in range(3):
            for col_idx in range(3):
                a_atom = 3 * row_idx + sum_idx
                b_atom = 3 * sum_idx + col_idx
                c_atom = 3 * row_idx + col_idx
                target[a_atom, b_atom, c_atom] = 1.0
    return target


def compute_loss(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    approx = torch.einsum('ka,kb,kc->abc', alpha, beta, gamma)
    return ((approx - target) ** 2).sum()


def exact_residual_stats(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    with torch.no_grad():
        approx = torch.einsum('ka,kb,kc->abc', alpha, beta, gamma)
        residual = approx - target
        abs_residual = residual.abs()
        return {
            'verified_loss': float((residual ** 2).sum().item()),
            'max_abs_residual': float(abs_residual.max().item()),
            'mean_abs_residual': float(abs_residual.mean().item()),
            'n_entries_over_1e-12': int((abs_residual > 1e-12).sum().item()),
            'n_entries_over_1e-20': int((abs_residual > 1e-20).sum().item()),
        }


def make_random_factors(rank: int, init: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if init == 'normal':
        alpha = torch.randn((rank, 9), dtype=torch.float64, requires_grad=True)
        beta = torch.randn((rank, 9), dtype=torch.float64, requires_grad=True)
        gamma = torch.randn((rank, 9), dtype=torch.float64, requires_grad=True)
    elif init == 'uniform':
        alpha = (2.0 * torch.rand((rank, 9), dtype=torch.float64) - 1.0).requires_grad_()
        beta = (2.0 * torch.rand((rank, 9), dtype=torch.float64) - 1.0).requires_grad_()
        gamma = (2.0 * torch.rand((rank, 9), dtype=torch.float64) - 1.0).requires_grad_()
    else:
        raise ValueError(f'Unsupported init mode: {init}')
    return alpha, beta, gamma


def search_run(
    rank: int,
    target: torch.Tensor,
    max_iters: int,
    lr: float,
    tol: float,
    seed: int,
    init: str,
) -> dict[str, object]:
    torch.manual_seed(seed)

    alpha, beta, gamma = make_random_factors(rank, init)
    optimizer = torch.optim.Adam([alpha, beta, gamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2000,
        factor=0.5,
        min_lr=1e-6,
    )

    best_loss = float('inf')
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_iter = -1

    for iteration in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(alpha, beta, gamma, target)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        scheduler.step(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_alpha = alpha.detach().clone()
            best_beta = beta.detach().clone()
            best_gamma = gamma.detach().clone()
            best_iter = iteration

        if loss_value < tol:
            stats = exact_residual_stats(alpha.detach(), beta.detach(), gamma.detach(), target)
            return {
                'success': True,
                'loss': loss_value,
                'best_loss': loss_value,
                'iters': iteration,
                'best_iter': iteration,
                'seed': seed,
                'alpha': alpha.detach().tolist(),
                'beta': beta.detach().tolist(),
                'gamma': gamma.detach().tolist(),
                **stats,
            }

    assert best_alpha is not None and best_beta is not None and best_gamma is not None
    stats = exact_residual_stats(best_alpha, best_beta, best_gamma, target)
    return {
        'success': False,
        'loss': best_loss,
        'best_loss': best_loss,
        'iters': max_iters,
        'best_iter': best_iter,
        'seed': seed,
        **stats,
    }


def parse_schedule(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return list(DEFAULT_SCHEDULE)

    schedule: list[tuple[int, int]] = []
    for chunk in raw.split(','):
        item = chunk.strip()
        if not item:
            continue
        rank_str, iters_str = item.split(':', maxsplit=1)
        schedule.append((int(rank_str), int(iters_str)))
    if not schedule:
        raise ValueError('Schedule cannot be empty')
    return schedule


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload) + '\n')


def append_text(path: Path, line: str) -> None:
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def write_record_summary(path: Path, result: dict[str, object], rank: int, seed: int, elapsed_hours: float, total_restarts: int) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('POTENTIAL NEW RECORD\n')
        handle.write(f'R = {rank}\n')
        handle.write(f'seed = {seed}\n')
        handle.write(f'loss = {result["loss"]}\n')
        handle.write(f'verified_loss = {result["verified_loss"]}\n')
        handle.write(f'max_abs_residual = {result["max_abs_residual"]}\n')
        handle.write(f'elapsed = {elapsed_hours:.2f}h\n')
        handle.write(f'total restarts = {total_restarts}\n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Step 50 overnight rank search for 3x3 matrix multiplication tensor.')
    parser.add_argument('--schedule', default=None, help='Comma-separated schedule like 23:20000,22:100000. Defaults to the overnight schedule.')
    parser.add_argument('--max-cycles', type=int, default=None, help='Optional finite number of cycles for smoke tests. Default is infinite.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate.')
    parser.add_argument('--tol', type=float, default=1e-20, help='Win threshold on total squared loss.')
    parser.add_argument('--near-miss-tol', type=float, default=1e-6, help='Near-miss threshold for saving failed runs.')
    parser.add_argument('--init', choices=['normal', 'uniform'], default='normal', help='Random initialization distribution.')
    parser.add_argument('--seed-offset', type=int, default=0, help='Initial seed offset for restart numbering.')
    parser.add_argument('--torch-threads', type=int, default=(os.cpu_count() or 1), help='PyTorch intra-op CPU thread count. Defaults to all logical CPUs.')
    parser.add_argument('--torch-interop-threads', type=int, default=1, help='PyTorch inter-op CPU thread count. Keep low for this single-worker search.')
    parser.add_argument('--output-dir', default=str(EXPORTS), help='Directory for search artifacts.')
    return parser


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)

    args = build_parser().parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(max(1, args.torch_interop_threads))
    schedule = parse_schedule(args.schedule)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / 'search_log.jsonl'
    summary_path = output_dir / 'search_summary.txt'
    heartbeat_path = output_dir / 'search_heartbeat.json'

    target = build_target()
    start_time = time.time()
    total_restarts = args.seed_offset
    wins = 0
    best_per_rank: dict[int, float] = {}

    print(f"Starting overnight search at {time.strftime('%H:%M:%S')}")
    print(f"Schedule: {[rank for rank, _ in schedule]}")
    print(f"Output: {output_dir}")
    print(f"Win condition: loss < {args.tol:.1e}")
    print(f"Near-miss condition: loss < {args.near_miss_tol:.1e}")
    print(f"Init mode: {args.init}")
    print(f"Torch threads: intra_op={torch.get_num_threads()} inter_op={torch.get_num_interop_threads()}")
    print()

    cycle = 0
    while args.max_cycles is None or cycle < args.max_cycles:
        cycle += 1
        for rank, max_iters in schedule:
            seed = total_restarts
            total_restarts += 1
            elapsed_hours = (time.time() - start_time) / 3600.0

            print(
                f"[{time.strftime('%H:%M:%S')}] cycle={cycle} R={rank} seed={seed} "
                f"elapsed={elapsed_hours:.2f}h restarts={total_restarts} wins={wins}"
            )

            write_json(
                heartbeat_path,
                {
                    'cycle': cycle,
                    'R': rank,
                    'seed': seed,
                    'status': 'running',
                    'elapsed_hours': elapsed_hours,
                    'restarts': total_restarts,
                    'wins': wins,
                    'best_per_R': {str(key): value for key, value in sorted(best_per_rank.items())},
                },
            )

            result = search_run(
                rank=rank,
                target=target,
                max_iters=max_iters,
                lr=args.lr,
                tol=args.tol,
                seed=seed,
                init=args.init,
            )

            elapsed_hours = (time.time() - start_time) / 3600.0
            log_entry = {
                'cycle': cycle,
                'R': rank,
                'seed': seed,
                'success': result['success'],
                'loss': result['loss'],
                'best_iter': result['best_iter'],
                'iters': result['iters'],
                'verified_loss': result['verified_loss'],
                'max_abs_residual': result['max_abs_residual'],
                'elapsed_hours': elapsed_hours,
                'init': args.init,
            }
            append_jsonl(log_path, log_entry)

            heartbeat_payload = {
                'cycle': cycle,
                'R': rank,
                'seed': seed,
                'status': 'completed',
                'success': result['success'],
                'loss': result['loss'],
                'verified_loss': result['verified_loss'],
                'elapsed_hours': elapsed_hours,
                'restarts': total_restarts,
                'wins': wins,
                'best_per_R': {str(key): value for key, value in sorted(best_per_rank.items())},
            }
            write_json(heartbeat_path, heartbeat_payload)

            if result['success']:
                wins += 1
                print(f"  *** WIN *** R={rank} loss={result['loss']:.2e} verified={result['verified_loss']:.2e} seed={seed}")
                win_path = output_dir / f'WIN_R{rank}_seed{seed}.json'
                write_json(win_path, result)

                if rank < 23:
                    print()
                    print('=' * 60)
                    print(f'  POTENTIAL NEW RECORD: R = {rank}')
                    print(f'  Saved to: {win_path}')
                    print('  VERIFY THIS RESULT CAREFULLY')
                    print('=' * 60)
                    print()
                    write_record_summary(summary_path, result, rank, seed, elapsed_hours, total_restarts)
            else:
                print(
                    f"  no win, best_loss={result['loss']:.6e} verified={result['verified_loss']:.6e} "
                    f"max_abs={result['max_abs_residual']:.3e}"
                )

            if rank not in best_per_rank or float(result['loss']) < best_per_rank[rank]:
                best_per_rank[rank] = float(result['loss'])
                if not result['success'] and float(result['loss']) < args.near_miss_tol:
                    near_miss_path = output_dir / f'NEARMISS_R{rank}_seed{seed}.json'
                    write_json(near_miss_path, result)
                    print(f"  ** NEAR MISS ** R={rank} loss={result['loss']:.2e} verified={result['verified_loss']:.2e}")

            if total_restarts % 20 == 0:
                elapsed_hours = (time.time() - start_time) / 3600.0
                rate = total_restarts / max(elapsed_hours, 1e-6)
                best_json = {key: f'{value:.2e}' for key, value in sorted(best_per_rank.items())}
                print()
                print(f"  --- Summary: {total_restarts} restarts in {elapsed_hours:.2f}h ({rate:.1f}/h), {wins} wins ---")
                print(f"  Best loss per R: {json.dumps(best_json)}")
                print()


if __name__ == '__main__':
    main()