"""
swap_optimizer.py — Coordinator for the multi-worker swap optimizer.

Spawns worker processes, collects results, maintains global best,
checkpoints, and drives the Rich dashboard.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from rich.console import Console
from rich.live import Live

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "CANON_DATABASE"))
sys.path.insert(0, str(_ROOT / "CANON_DATABASE" / "scripts"))

from swap_config   import SwapConfig
from swap_worker   import worker_main, _Msg
from swap_dashboard import SwapDashboard


def _verify_solution(indices: list[int], config: SwapConfig) -> float:
    """Independent solution verification. Returns max reconstruction error."""
    from term_db import TermDB
    db = TermDB(load_path=config.db_path, generate=False)
    idx = [int(i) for i in indices]
    H     = db.H[idx].astype(float)
    sigma = db.sigma[idx].astype(float)
    delta = db.compute_delta_batch(idx).astype(float)
    N  = __import__('numpy').hstack([H, delta])
    SN = __import__('numpy').hstack([sigma, N])
    import numpy as np
    target = np.zeros((9, 81)); target[:, :9] = 3.0 * np.eye(9)
    G, _, _, _ = np.linalg.lstsq(SN.T, target.T, rcond=None)
    return float(np.max(np.abs(target.T - SN.T @ G)))


def run(config: SwapConfig):
    console = Console()
    console.print(f"[bold cyan]ADE3x3 Phase 4 — Swap Optimizer[/bold cyan]")
    console.print(f"R={config.R}, target_rank_H={config.target_rank_H}, "
                  f"workers={config.n_workers}")
    console.print(f"DB: {config.db_path}")
    console.print(f"Stall threshold: {config.stall_threshold}")

    # Ensure output dirs exist
    Path(config.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)

    log_fh = open(config.log_file, 'a', encoding='utf-8') if config.log_improvements else None

    # Shared state
    out_queue: mp.Queue = mp.Queue(maxsize=10000)
    stop_event: mp.Event = mp.Event()

    dashboard = SwapDashboard(config.n_workers, config.R)

    global_best_score:   Optional[tuple] = None
    global_best_indices: Optional[list[int]] = None
    total_swaps = 0
    total_improv = 0
    n_workers = config.n_workers  # may be adjusted by RAM check below

    last_checkpoint = time.time()

    def _save_checkpoint():
        data = {
            'rank':                config.R,
            'global_best_score':   list(global_best_score) if global_best_score else None,
            'global_best_indices': global_best_indices,
            'total_swaps':         total_swaps,
            'total_improvements':  total_improv,
            'wall_time_seconds':   time.time() - dashboard.start_time,
        }
        Path(config.checkpoint_file).write_text(json.dumps(data, indent=2))

    def _process_message(msg: _Msg):
        nonlocal global_best_score, global_best_indices, total_swaps, total_improv

        dashboard.update(msg.worker_id, msg.kind, msg.payload)

        if msg.kind in ('update', 'solution'):
            score   = tuple(msg.payload['score'])
            swaps   = msg.payload.get('swaps', 0)
            improv  = msg.payload.get('improv', 0)

            if global_best_score is None or score < global_best_score:
                global_best_score   = score
                global_best_indices = msg.payload.get('indices')
                total_swaps  = swaps
                total_improv = improv

                if log_fh:
                    entry = json.dumps({
                        'time':    time.time() - dashboard.start_time,
                        'worker':  msg.worker_id,
                        'score':   list(score),
                        'swaps':   swaps,
                    })
                    log_fh.write(entry + '\n')
                    log_fh.flush()

                if msg.kind == 'solution':
                    stop_event.set()
                    return True  # signal solution found

        return False

    # Compute safe worker count based on available RAM
    n_workers = config.n_workers
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        # Each worker: ~11.2 GB mmap (shared read-only) + ~0.5 GB Python overhead.
        # With readonly mmap the DB pages are shared, so only overhead scales.
        # Reserve 15% of total RAM for OS/other; each worker costs ~0.5 GB overhead.
        available_gb = mem.available / 1e9
        reserve_gb   = mem.total * 0.15 / 1e9
        worker_overhead_gb = 0.6  # Python + numpy per worker, excluding shared mmap
        max_by_ram = max(1, int((available_gb - reserve_gb) / worker_overhead_gb))
        if max_by_ram < n_workers:
            console.print(f"[yellow]RAM limit: reducing workers {n_workers} -> {max_by_ram} "
                           f"(avail={available_gb:.1f}GB, overhead~{worker_overhead_gb}GB/worker)[/yellow]")
            n_workers = max_by_ram
        else:
            console.print(f"RAM check OK: {available_gb:.1f}GB free, {n_workers} workers "
                          f"(~{n_workers*worker_overhead_gb:.1f}GB overhead + shared mmap)")
    else:
        console.print("[yellow]psutil not found; skipping RAM check[/yellow]")

    # Spawn workers
    processes = []
    for wid in range(n_workers):
        p = mp.Process(
            target=worker_main,
            args=(wid, config, out_queue, stop_event),
            daemon=True,
            name=f"swap-worker-{wid}",
        )
        p.start()
        processes.append(p)
        # Brief stagger so workers don't all mmap simultaneously
        time.sleep(0.3)

    # Update dashboard to reflect actual worker count
    dashboard.n_workers = n_workers
    for i in range(n_workers):
        dashboard.worker_swaps.setdefault(i, 0)
        dashboard.worker_restarts.setdefault(i, 0)
        dashboard.worker_done.setdefault(i, False)

    console.print(f"\nSpawned {n_workers} workers. Press Ctrl+C to stop.\n")

    solution_found = False
    start_time = time.time()

    try:
        with Live(dashboard.render(), console=console, refresh_per_second=2) as live:
            while not stop_event.is_set():
                # Drain queue
                msgs_processed = 0
                while True:
                    try:
                        msg = out_queue.get_nowait()
                        found = _process_message(msg)
                        msgs_processed += 1
                        if found:
                            solution_found = True
                            break
                    except Exception:
                        break

                live.update(dashboard.render())

                # RAM guard: if usage > 90%, kill the last worker
                if _HAS_PSUTIL and len(processes) > 1:
                    ram_pct = psutil.virtual_memory().percent
                    if ram_pct > 90.0:
                        victim = processes.pop()
                        victim.terminate()
                        n_workers -= 1
                        dashboard.n_workers = n_workers
                        console.print(f"[red]RAM {ram_pct:.0f}% > 90% — killed worker {len(processes)}[/red]")

                # Check time limit
                if config.time_limit_seconds > 0:
                    if time.time() - start_time > config.time_limit_seconds:
                        console.print("\n[yellow]Time limit reached.[/yellow]")
                        stop_event.set()
                        break

                # Checkpoint
                if time.time() - last_checkpoint > config.checkpoint_interval:
                    _save_checkpoint()
                    last_checkpoint = time.time()

                # Check if all workers finished
                if all(dashboard.worker_done.values()):
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        stop_event.set()
    finally:
        stop_event.set()
        # Wait briefly for workers to finish
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        if log_fh:
            log_fh.close()
        _save_checkpoint()

    # Final report
    console.print(f"\n[bold]Search complete.[/bold]")
    console.print(f"Total wall time: {time.time()-start_time:.0f}s")
    console.print(f"Checkpoint saved to: {config.checkpoint_file}")

    if solution_found and global_best_indices is not None:
        console.print(f"\n[bold bright_green]*** SOLUTION FOUND! ***[/bold bright_green]")
        err = _verify_solution(global_best_indices, config)
        console.print(f"  Verification error: {err:.2e}")
        sol = {
            'rank':              config.R,
            'score':             list(global_best_score),
            'indices':           global_best_indices,
            'verification_err':  err,
        }
        Path(config.solution_file).write_text(json.dumps(sol, indent=2))
        console.print(f"  Solution saved to: {config.solution_file}")
    elif global_best_score is not None:
        console.print(f"\nBest score reached: {global_best_score}")
        g1, leak, resid, aug, err = global_best_score
        console.print(f"  Gate1 gap={g1}, delta_leak={leak}, resid={resid:.3f}, "
                      f"aug_gap={aug}, recon_err={err:.4e}")
