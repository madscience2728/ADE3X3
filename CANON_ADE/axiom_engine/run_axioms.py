"""
run_axioms.py — Main overnight runner with rich TUI, checkpointing, RAM throttling.

"The axioms are not filters — they are lenses through which we refract the
 solution space into canonical eigenmodes of truth." — CANON ADE

Usage:
  python -m CANON_ADE.axiom_engine.run_axioms [--resume RUN_NAME] [--ranks 13,19,20,21,22]
"""
import sys
import os
import time
import argparse
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import queue

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CANON_ADE"))

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# ═══════════════════════════════════════════════════════════════
# WORKER FUNCTIONS (top-level for pickling)
# ═══════════════════════════════════════════════════════════════

def _worker_init():
    """Process pool initializer — import once per worker, not per task."""
    global _wk_build, _wk_PERM19, _wk_decompose, _wk_evaluate_all, _wk_build_fiber, _wk_build_fiber_sym
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from CANON_ADE.axiom_engine.tensor_core import (
        build_random_factors, PERM19, decompose_irreps,
        build_fiber_constrained_factors, build_fiber_constrained_symmetric_factors,
    )
    from CANON_ADE.axiom_engine.axiom_evaluators import evaluate_all
    _wk_build = build_random_factors
    _wk_build_fiber = build_fiber_constrained_factors
    _wk_build_fiber_sym = build_fiber_constrained_symmetric_factors
    _wk_PERM19 = PERM19
    _wk_decompose = decompose_irreps
    _wk_evaluate_all = evaluate_all


def _worker_evaluate_chunk(chunk):
    """Evaluate a CHUNK of tasks in one process call — amortizes process overhead."""
    import numpy as np, traceback
    results = []
    for task in chunk:
        rank = task['rank']
        seed_id = task['seed_id']
        mode = task.get('mode', 'symmetric')
        rng = np.random.default_rng(seed_id)
        try:
            if mode == 'fiber' and rank == 19:
                alpha, beta, gamma = _wk_build_fiber(rng=rng)
            elif mode == 'fiber_sym' and rank == 19:
                alpha, beta, gamma = _wk_build_fiber_sym(rng=rng)
            else:
                alpha, beta, gamma = _wk_build(
                    rank, rng=rng, symmetric=(mode == 'symmetric' and rank == 19)
                )
            irreps = perm_mats = None
            if rank == 19:
                perm_mats = _wk_PERM19
                irreps = _wk_decompose(perm_mats, 19)

            res = _wk_evaluate_all(
                alpha, beta, gamma, rank,
                irreps=irreps, perm_matrices=perm_mats,
                skip_classical=False,
                skip_novel=False,
            )
            res['seed_id'] = seed_id
            res['mode'] = mode
            res['status'] = 'ok'
            for k in list(res.keys()):
                if isinstance(res[k], list) and len(res[k]) > 20:
                    res[k] = res[k][:20]
        except Exception as e:
            res = {
                'R': rank, 'seed_id': seed_id, 'mode': mode,
                'status': 'error', 'error': str(e),
                'traceback': traceback.format_exc(),
            }
        results.append(res)
    return results


def _gpu_worker_loop(task_queue, result_queue, done_event):
    """Dedicated GPU process — runs in its own process, no GIL contention.
    Pulls tasks from task_queue, pushes aggregate results to result_queue."""
    import sys, numpy as np
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from CANON_ADE.axiom_engine.tensor_core import T_MATMUL
    from CANON_ADE.axiom_engine.gpu_kernels import (
        batch_step51_gpu, batch_rank_H_gpu, batch_relation_module_gpu,
        batch_fitness_gpu, batch_gram_eigenvalues_gpu,
    )
    import torch

    # Use CUDA streams for pipelining
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        streams = [torch.cuda.Stream() for _ in range(3)]

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except Exception:
            if done_event.is_set():
                break
            continue
        if task is None:  # sentinel
            break

        rank = task['rank']
        seed_start = task['seed_start']
        batch_size = task['batch_size']

        # Vectorized factor generation
        master_rng = np.random.default_rng(int(seed_start))
        alphas = master_rng.standard_normal((batch_size, rank, 3, 3))
        betas = master_rng.standard_normal((batch_size, rank, 3, 3))
        gammas = master_rng.standard_normal((batch_size, rank, 3, 3))

        # GPU compute — use streams for overlap if CUDA
        if use_cuda:
            with torch.cuda.stream(streams[0]):
                Sigma, H, Delta = batch_step51_gpu(alphas, betas)
                ranks_H = batch_rank_H_gpu(H)
            with torch.cuda.stream(streams[1]):
                dim_KA, dim_KB, dim_KC = batch_relation_module_gpu(alphas, betas, gammas)
                fitness = batch_fitness_gpu(alphas, betas, gammas, T_MATMUL)
            with torch.cuda.stream(streams[2]):
                spec_A = batch_gram_eigenvalues_gpu(alphas.reshape(-1, rank, 9))
                spec_B = batch_gram_eigenvalues_gpu(betas.reshape(-1, rank, 9))
                spec_C = batch_gram_eigenvalues_gpu(gammas.reshape(-1, rank, 9))
            torch.cuda.synchronize()
        else:
            Sigma, H, Delta = batch_step51_gpu(alphas, betas)
            ranks_H = batch_rank_H_gpu(H)
            dim_KA, dim_KB, dim_KC = batch_relation_module_gpu(alphas, betas, gammas)
            fitness = batch_fitness_gpu(alphas, betas, gammas, T_MATMUL)
            spec_A = batch_gram_eigenvalues_gpu(alphas.reshape(-1, rank, 9))
            spec_B = batch_gram_eigenvalues_gpu(betas.reshape(-1, rank, 9))
            spec_C = batch_gram_eigenvalues_gpu(gammas.reshape(-1, rank, 9))

        # Vectorized post-processing
        target = rank - 9
        A1_pass = (ranks_H == target).astype(int)
        A2_pass = (fitness < 1e-6).astype(int)
        spec_sym = (np.linalg.norm(spec_A - spec_B, axis=1)
                  + np.linalg.norm(spec_A - spec_C, axis=1)
                  + np.linalg.norm(spec_B - spec_C, axis=1))

        # Batch entropy — vectorized where possible
        def _batch_entropy(ev_batch):
            out = np.zeros(ev_batch.shape[0])
            for i in range(ev_batch.shape[0]):
                ev = ev_batch[i]
                ev = ev[ev > 1e-15]
                if len(ev) == 0:
                    continue
                p = ev / ev.sum()
                out[i] = -np.sum(p * np.log(p + 1e-30))
            return out

        ent_A = _batch_entropy(spec_A)
        ent_B = _batch_entropy(spec_B)
        ent_C = _batch_entropy(spec_C)

        # Send AGGREGATE stats, not individual results — eliminates per-result queue overhead
        # Compute aggregates over the batch
        agg = {
            'type': 'gpu_batch_agg',
            'rank': rank,
            'count': batch_size,
            'metrics': {
                'A1_pass': {'sum': float(A1_pass.sum()), 'min': float(A1_pass.min()), 'max': float(A1_pass.max())},
                'A1_rank_H': {'sum': float(ranks_H.sum()), 'min': float(ranks_H.min()), 'max': float(ranks_H.max())},
                'A2_fitness': {'sum': float(fitness.sum()), 'min': float(fitness.min()), 'max': float(fitness.max())},
                'A2_pass': {'sum': float(A2_pass.sum()), 'min': float(A2_pass.min()), 'max': float(A2_pass.max())},
                'A4_dim_KA': {'sum': float(dim_KA.sum()), 'min': float(dim_KA.min()), 'max': float(dim_KA.max())},
                'A4_dim_KB': {'sum': float(dim_KB.sum()), 'min': float(dim_KB.min()), 'max': float(dim_KB.max())},
                'A4_dim_KC': {'sum': float(dim_KC.sum()), 'min': float(dim_KC.min()), 'max': float(dim_KC.max())},
                'NV_spec_entropy_A': {'sum': float(ent_A.sum()), 'min': float(ent_A.min()), 'max': float(ent_A.max())},
                'NV_spec_entropy_B': {'sum': float(ent_B.sum()), 'min': float(ent_B.min()), 'max': float(ent_B.max())},
                'NV_spec_entropy_C': {'sum': float(ent_C.sum()), 'min': float(ent_C.min()), 'max': float(ent_C.max())},
                'NV_spec_symmetry': {'sum': float(spec_sym.sum()), 'min': float(spec_sym.min()), 'max': float(spec_sym.max())},
            },
        }
        result_queue.put(agg)

    result_queue.put(None)  # signal done


# ═══════════════════════════════════════════════════════════════
# TUI DASHBOARD
# ═══════════════════════════════════════════════════════════════

class AxiomDashboard:
    """Rich terminal dashboard — axioms as ROWS, ranks as COLUMNS."""

    AXIOM_ROWS = [
        # (display_name, result_key, is_rate_based)
        # ── A1-A7 Novel ──
        ('A1 Conservation',    'A1_pass',              True),
        ('A1 R+η=27',         'A1_conservation_ok',   True),
        ('A1 rank(H)',         'A1_rank_H',            False),
        ('A2 Fitness',         'A2_fitness',           False),
        ('A2 Pass',            'A2_pass',              True),
        ('A3 Σ Achievable',    'A3_achievable',        True),
        ('A3 rank(Σ)',         'A3_rank_sigma',        False),
        ('A4 Generic Pos',     'A4_generic_position',  True),
        ('A4 Minimal Rank',    'A4_minimal_rank',      True),
        ('A4 dim(K_A∩K_B)',    'A4_dim_AB',            False),
        ('A5 Gate 2',          'A5_gate2',             True),
        ('A5 Gate 3',          'A5_gate3',             True),
        ('A5 Gates Both',      'A5_gates_compatible',  True),
        ('A5 aug_gap',         'A5_aug_gap',           False),
        ('A6 Parity Sep',      'A6_parity_separation', False),
        ('A6 Violating',       'A6_n_violating',       False),
        ('A6b Fiber Pass',     'A6b_fiber_pass',       True),
        ('A6b Fiber Err',      'A6b_fiber_sum_err',    False),
        ('A6b Max Dev',        'A6b_max_fiber_dev',    False),
        ('A7 Quantized',       'A7_quantized',         True),
        ('A7 Best Score',      'A7_best_score',        False),
        # ── Classical ──
        ('CL Associative',     'CL_assoc_pass',        True),
        ('CL Commutative',     'CL_comm_pass',         True),
        ('CL Jacobi',          'CL_jacobi_pass',       True),
        ('CL Nilpotent',       'CL_nilpotent',         True),
        ('CL Killing rk',      'CL_killing_rank',      False),
        ('CL Semisimple',      'CL_semisimple',        True),
        # ── Novel diagnostics ──
        ('NV Coherence',       'NV_coherence_max',     False),
        ('NV Welch Excess',    'NV_coherence_excess',  False),
        ('NV Coupling rk(CA)', 'NV_coupling_rank_CA',  False),
        ('NV Coupling rk(AB)', 'NV_coupling_rank_AB',  False),
        ('NV Spec Symmetry',   'NV_spec_symmetry',     False),
        ('NV Flat rk₀',        'NV_flat_rank_0',       False),
        ('NV Flat rk₁',        'NV_flat_rank_1',       False),
        ('NV Flat rk₂',        'NV_flat_rank_2',       False),
        ('NV Recon Error',     'NV_recon_err',         False),
    ]

    def __init__(self, ranks, n_cpu, n_gpu, gpu_batch_size):
        self.ranks = ranks
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.gpu_batch_size = gpu_batch_size
        total_gpu = sum(n_gpu[r] * gpu_batch_size for r in ranks)
        total_cpu = sum(n_cpu[r] for r in ranks)
        self.total = total_gpu + total_cpu
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self.throttle_count = 0

        self.rank_data = {r: {} for r in ranks}
        self.rank_done_gpu = {r: 0 for r in ranks}
        self.rank_done_cpu = {r: 0 for r in ranks}
        self.rank_errors = {r: 0 for r in ranks}
        self.rank_activity = {r: 'waiting' for r in ranks}

        self.current_phase = "Initializing"
        self.phase_detail = ""
        self.substeps = {}
        self._lock = threading.Lock()

    def set_substep(self, name, done, total):
        with self._lock:
            self.substeps[name] = (done, total)

    def clear_substep(self, name):
        with self._lock:
            self.substeps.pop(name, None)

    def update(self, result):
        with self._lock:
            r = result.get('R', result.get('rank', 0))
            if result.get('status') == 'error':
                self.completed += 1
                self.errors += 1
                if r in self.rank_errors:
                    self.rank_errors[r] += 1
                return
            if r not in self.rank_data:
                return

            mode = result.get('mode', '')

            # Handle GPU batch aggregates (one message per batch)
            if result.get('type') == 'gpu_batch_agg':
                count = result['count']
                self.completed += count
                self.rank_done_gpu[r] = self.rank_done_gpu.get(r, 0) + count
                data = self.rank_data[r]
                for key, agg in result['metrics'].items():
                    if key not in data:
                        data[key] = {'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                    data[key]['sum'] += agg['sum']
                    data[key]['count'] += count
                    data[key]['min'] = min(data[key]['min'], agg['min'])
                    data[key]['max'] = max(data[key]['max'], agg['max'])
                return

            # Handle individual CPU results
            self.completed += 1
            if mode == 'gpu_batch':
                self.rank_done_gpu[r] = self.rank_done_gpu.get(r, 0) + 1
            else:
                self.rank_done_cpu[r] = self.rank_done_cpu.get(r, 0) + 1
            data = self.rank_data[r]
            for _name, key, _is_rate in self.AXIOM_ROWS:
                val = result.get(key)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if key not in data:
                    data[key] = {'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                data[key]['sum'] += val
                data[key]['count'] += 1
                data[key]['min'] = min(data[key]['min'], val)
                data[key]['max'] = max(data[key]['max'], val)

    def update_batch(self, results_list):
        """Bulk update from a CPU chunk — fewer lock acquisitions."""
        with self._lock:
            for result in results_list:
                r = result.get('R', 0)
                if result.get('status') == 'error':
                    self.completed += 1
                    self.errors += 1
                    if r in self.rank_errors:
                        self.rank_errors[r] += 1
                    continue
                if r not in self.rank_data:
                    continue
                self.completed += 1
                self.rank_done_cpu[r] = self.rank_done_cpu.get(r, 0) + 1
                data = self.rank_data[r]
                for _name, key, _is_rate in self.AXIOM_ROWS:
                    val = result.get(key)
                    if val is None:
                        continue
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if key not in data:
                        data[key] = {'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                    data[key]['sum'] += val
                    data[key]['count'] += 1
                    data[key]['min'] = min(data[key]['min'], val)
                    data[key]['max'] = max(data[key]['max'], val)

    def set_activity(self, rank, activity):
        with self._lock:
            if rank in self.rank_activity:
                self.rank_activity[rank] = activity

    def _fmt_cell(self, key, is_rate, stats):
        if stats is None or stats['count'] == 0:
            return Text("—", style="dim")
        if is_rate:
            rate = stats['sum'] / stats['count']
            pct = rate * 100
            style = "bold green" if pct >= 50 else ("yellow" if pct > 0 else "red")
            return Text(f"{pct:.1f}%", style=style)
        else:
            if 'fitness' in key or 'err' in key.lower() or 'recon' in key.lower():
                v = stats['min']
                s = f"{v:.2e}" if abs(v) > 1e3 or (0 < abs(v) < 0.01) else f"{v:.3f}"
                style = "green" if v < 1e-3 else ("yellow" if v < 1.0 else "red")
            else:
                v = stats['sum'] / stats['count']
                s = f"{v:.2f}"
                style = "white"
            return Text(s, style=style)

    def build_display(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / max(elapsed, 0.01)
        remaining = (self.total - self.completed) / max(rate, 0.001) if rate > 0 else 0
        mem = psutil.virtual_memory()
        pct_done = self.completed / max(self.total, 1)

        # ── Compact table: axioms=rows, ranks=columns, no separator rows ──
        table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold green",
            pad_edge=False, padding=(0, 1), show_edge=False,
            title_style="bold cyan",
        )
        # First column: axiom name (narrow)
        table.add_column("Metric", style="cyan", width=18, no_wrap=True)
        for r in self.ranks:
            # Column header includes per-rank GPU/CPU progress
            gpu_done = self.rank_done_gpu.get(r, 0)
            cpu_done = self.rank_done_cpu.get(r, 0)
            gpu_target = self.n_gpu.get(r, 0) * self.gpu_batch_size
            cpu_target = self.n_cpu.get(r, 0)
            gp = int(gpu_done / max(gpu_target, 1) * 100)
            cp = int(cpu_done / max(cpu_target, 1) * 100)
            table.add_column(f"R={r}\nG{gp}% C{cp}%", justify="center", width=9)

        for name, key, is_rate in self.AXIOM_ROWS:
            row = [Text(name, style="cyan")]
            for r in self.ranks:
                stats = self.rank_data[r].get(key)
                row.append(self._fmt_cell(key, is_rate, stats))
            table.add_row(*row)

        # ── Single status line: progress + RAM + substeps ──
        bw = 40
        filled = int(bw * pct_done)
        bar = "█" * filled + "░" * (bw - filled)
        status = Text()
        status.append("ADE3×3 ", style="bold cyan")
        status.append(f"[{bar}] ", style="cyan")
        status.append(f"{pct_done*100:.1f}% ", style="bold white")
        status.append(f"{self.completed:,}/{self.total:,} ", style="white")
        status.append(f"{timedelta(seconds=int(elapsed))} ", style="dim")
        status.append(f"ETA {timedelta(seconds=int(remaining))} ", style="dim")
        status.append(f"{rate:.0f}/s ", style="green")
        ram_style = "red bold" if mem.percent > 85 else ("yellow" if mem.percent > 70 else "green")
        status.append(f"RAM {mem.percent:.0f}%", style=ram_style)
        if self.errors:
            status.append(f" err:{self.errors}", style="red")
        if self.throttle_count:
            status.append(f" thr:{self.throttle_count}", style="yellow")
        # Append substep info inline
        with self._lock:
            for name, (done, total) in self.substeps.items():
                sp = done / max(total, 1) * 100
                status.append(f"  {name}:{sp:.0f}%", style="blue")

        # Per-rank activity
        activity = Text()
        for r in self.ranks:
            act = self.rank_activity.get(r, '')
            gpu_done = self.rank_done_gpu.get(r, 0)
            cpu_done = self.rank_done_cpu.get(r, 0)
            err = self.rank_errors.get(r, 0)
            gpu_target = self.n_gpu.get(r, 0) * self.gpu_batch_size
            cpu_target = self.n_cpu.get(r, 0)

            activity.append(f"R={r}", style="bold cyan")
            activity.append(f" GPU:", style="dim")
            gp = gpu_done / max(gpu_target, 1) * 100
            activity.append(f"{gpu_done}/{gpu_target}", style="green" if gp > 99 else "white")
            activity.append(f" CPU:", style="dim")
            cp = cpu_done / max(cpu_target, 1) * 100
            activity.append(f"{cpu_done}/{cpu_target}", style="green" if cp > 99 else "white")
            if err:
                activity.append(f" ❌{err}", style="red")
            if act and act != 'waiting':
                activity.append(f" ▸{act}", style="yellow bold")
            elif act == 'waiting':
                activity.append(f" ◦idle", style="dim")
            activity.append("   ")

        return Group(status, Panel(activity, title="[bold]Per-Rank Activity[/bold]", border_style="yellow"), table)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ADE3×3 Axiom Engine v2.0")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ranks', type=str, default='13,19,20,21,22,23,27')
    parser.add_argument('--cpu-samples', type=int, default=500)
    parser.add_argument('--gpu-batches', type=int, default=500)
    parser.add_argument('--gpu-batch-size', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--ram-limit', type=float, default=85.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--mode', type=str, default='symmetric',
                        choices=['symmetric', 'random', 'fiber', 'fiber_sym'],
                        help='Factor generation mode (fiber/fiber_sym use Tower Connection constraints)')
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(',')]
    if args.quick:
        cpu_per_rank, gpu_batches, gpu_batch_size = 10, 2, 32
    else:
        cpu_per_rank = args.cpu_samples
        gpu_batches = args.gpu_batches
        gpu_batch_size = args.gpu_batch_size
        # RTX 3060 (12GB) + Ryzen 9 5900X (12C/24T) tuning:
        # Default batch_size=256 barely warms the GPU. 2048 keeps CUDA cores busy.
        # Default gpu_batches=100 → 500 gives enough work for sustained throughput.

    n_cpu = {r: cpu_per_rank for r in ranks}
    n_gpu = {r: gpu_batches for r in ranks}
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]ADE3x3 AXIOM ENGINE v2.0\n"
        "\"What no one has done, we shall do.\"\n"
        "\n"
        "34 axiom metrics: A1-A7 + Classical + Novel\n"
        "All ranks run simultaneously -- GPU||CPU overlap\n"
        f"psutil RAM throttle @ {int(args.ram_limit)}%\n"
        f"Factor mode: {args.mode}[/bold cyan]",
        border_style="cyan"
    ))

    from CANON_ADE.axiom_engine.checkpoint import CheckpointManager

    if args.resume:
        ckpt = CheckpointManager.resume(args.resume)
        completed_ids = ckpt.get_completed_ids()
        console.print(f"[yellow]Resuming: {ckpt.n_completed} results loaded[/yellow]")
    else:
        ckpt = CheckpointManager()
        completed_ids = set()
        console.print(f"[green]New run: {ckpt.run_name}[/green]")

    ckpt.set_metadata('ranks', ranks)
    ckpt.set_metadata('cpu_per_rank', cpu_per_rank)
    ckpt.set_metadata('gpu_batches', gpu_batches)
    ckpt.set_metadata('workers', args.workers)
    ckpt.set_metadata('start_time', datetime.now().isoformat())

    # Build tasks — ROUND-ROBIN interleaved across ranks so all progress in parallel
    gpu_tasks = []
    for bi in range(gpu_batches):
        for r in ranks:
            gpu_tasks.append({
                'rank': r,
                'seed_start': r * 1_000_000 + bi * gpu_batch_size,
                'batch_size': gpu_batch_size,
            })

    # CPU tasks in CHUNKS — each worker gets CPU_CHUNK_SIZE tasks at once
    CPU_CHUNK_SIZE = 50  # amortizes process overhead; each chunk ~2-5s of work
    cpu_task_list = []
    for i in range(cpu_per_rank):
        for r in ranks:
            seed = r * 100_000 + i
            if (r, seed) not in completed_ids:
                cpu_task_list.append({
                    'rank': r,
                    'seed_id': seed,
                    'mode': args.mode if r == 19 else 'random',
                })
    # Chunk the CPU tasks
    cpu_chunks = [cpu_task_list[i:i+CPU_CHUNK_SIZE]
                  for i in range(0, len(cpu_task_list), CPU_CHUNK_SIZE)]

    total_gpu_evals = len(gpu_tasks) * gpu_batch_size
    total_cpu_evals = len(cpu_task_list)
    total_tasks = total_gpu_evals + total_cpu_evals

    dashboard = AxiomDashboard(ranks, n_cpu, n_gpu, gpu_batch_size)
    dashboard.total = total_tasks

    has_cuda = torch_available()
    mem = psutil.virtual_memory()
    console.print(f"\n[bold]Task Summary:[/bold]")
    console.print(f"  Ranks: {ranks} (all simultaneous)")
    console.print(f"  GPU: {len(gpu_tasks)} batches × {gpu_batch_size} = {total_gpu_evals:,} evals  [{'CUDA' if has_cuda else 'CPU fallback'}]")
    console.print(f"  CPU: {total_cpu_evals:,} full evals in {len(cpu_chunks)} chunks × {CPU_CHUNK_SIZE}")
    console.print(f"  Workers: {args.workers} CPU ‖ 1 GPU process (no GIL)")
    console.print(f"  RAM: limit {args.ram_limit}% = {args.ram_limit/100*mem.total/1e9:.0f}GB of {mem.total/1e9:.0f}GB\n")

    # ═══════════════════════════════════════════════════════════════
    # NEW PARALLEL ENGINE: GPU=separate process, CPU=chunked ProcessPool
    # ═══════════════════════════════════════════════════════════════
    dashboard.current_phase = "GPU‖CPU Parallel"
    gpu_result_queue = mp.Queue(maxsize=256)
    gpu_task_queue = mp.Queue()  # unbounded — filled by feeder thread
    gpu_stop_event = mp.Event()

    # Start GPU process FIRST, then feed tasks via background thread
    gpu_proc = mp.Process(
        target=_gpu_worker_loop,
        args=(gpu_task_queue, gpu_result_queue, gpu_stop_event),
        daemon=True,
    )
    gpu_proc.start()

    # Feed GPU task queue in background (never blocks main thread)
    def _gpu_feeder():
        for t in gpu_tasks:
            gpu_task_queue.put(t)
        gpu_task_queue.put(None)  # sentinel

    threading.Thread(target=_gpu_feeder, daemon=True).start()

    # Drain GPU results in a background thread (transfers from mp.Queue to dashboard)
    gpu_batches_done = [0]
    gpu_finished = threading.Event()

    def _gpu_drain():
        while True:
            try:
                msg = gpu_result_queue.get(timeout=0.5)
            except Exception:
                if not gpu_proc.is_alive():
                    break
                continue
            if msg is None:  # GPU process done
                break
            dashboard.update(msg)
            gpu_batches_done[0] += 1
            dashboard.set_substep("GPU screening", gpu_batches_done[0], len(gpu_tasks))
            dashboard.set_activity(msg['rank'], f"GPU batch {gpu_batches_done[0]}/{len(gpu_tasks)}")
        dashboard.clear_substep("GPU screening")
        gpu_finished.set()

    gpu_drain_thread = threading.Thread(target=_gpu_drain, daemon=True)
    gpu_drain_thread.start()

    # CPU: chunked ProcessPoolExecutor with as_completed
    dashboard.phase_detail = f"{args.workers} CPU workers + GPU process"
    cpu_chunks_done = 0

    with Live(dashboard.build_display(), console=console, refresh_per_second=4) as live:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
        ) as executor:
            # Submit all chunks — the pool handles queuing internally
            # But limit in-flight to avoid RAM explosion
            max_in_flight = args.workers * 3
            pending = {}
            chunk_iter = iter(enumerate(cpu_chunks))

            def _submit_chunks(n):
                submitted = 0
                for _ in range(n):
                    try:
                        idx, chunk = next(chunk_iter)
                        f = executor.submit(_worker_evaluate_chunk, chunk)
                        pending[f] = (idx, chunk)
                        submitted += 1
                    except StopIteration:
                        break
                return submitted

            _submit_chunks(min(max_in_flight, len(cpu_chunks)))

            last_update = time.time()
            while pending or not gpu_finished.is_set():
                # Use as_completed with a short timeout for responsiveness
                newly_done = []
                for f in list(pending):
                    if f.done():
                        newly_done.append(f)

                for f in newly_done:
                    idx, chunk = pending.pop(f)
                    try:
                        results_list = f.result(timeout=1)
                        ckpt.save_results_batch(results_list)
                        dashboard.update_batch(results_list)
                    except Exception as e:
                        for t in chunk:
                            dashboard.update({'R': t['rank'], 'status': 'error', 'error': str(e)})
                    cpu_chunks_done += 1
                    dashboard.set_substep("CPU full eval", cpu_chunks_done, len(cpu_chunks))

                    # Refill — RAM throttle check
                    if psutil.virtual_memory().percent < args.ram_limit:
                        _submit_chunks(1)
                    else:
                        dashboard.throttle_count += 1

                # Periodic UI refresh (not every iteration to reduce overhead)
                now = time.time()
                if now - last_update > 0.25:
                    live.update(dashboard.build_display())
                    last_update = now

                if not newly_done:
                    time.sleep(0.05)  # short sleep to avoid busy-wait

                if not pending and gpu_finished.is_set():
                    break

        # Wait for GPU to finish
        gpu_proc.join(timeout=60)
        gpu_drain_thread.join(timeout=10)

        dashboard.current_phase = "Finalizing"
        dashboard.phase_detail = ""
        dashboard.clear_substep("CPU full eval")
        live.update(dashboard.build_display())

    summary = ckpt.finalize()

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Run Complete: {ckpt.run_name}[/bold green]\n"
        f"Total evaluations: {summary['n_results']:,}\n"
        f"Ranks tested: {summary.get('ranks_tested', [])}\n"
        f"Checkpoint: CANON_ADE/axiom_engine/checkpoints/{ckpt.run_name}/",
        title="[bold]Summary[/bold]", border_style="green"
    ))

    console.print("\n[bold]Per-Rank Axiom Pass Rates:[/bold]")
    rt = Table(box=box.SIMPLE)
    rt.add_column("Axiom", style="cyan")
    for r in ranks:
        rt.add_column(f"R={r}", justify="center")
    for name, key, is_rate in AxiomDashboard.AXIOM_ROWS:
        if is_rate:
            row = [name]
            for r in ranks:
                stats = dashboard.rank_data[r].get(key)
                if stats and stats['count'] > 0:
                    row.append(f"{stats['sum']/stats['count']*100:.1f}%")
                else:
                    row.append("—")
            rt.add_row(*row)
    console.print(rt)


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    main()
