"""
run_axioms.py — Main overnight runner with rich TUI, checkpointing, RAM throttling.

Usage:
  python -m CANON_ADE.axiom_engine.run_axioms [--resume RUN_NAME] [--ranks 13,19,20,21,22]
  
Or directly:
  python CANON_ADE/axiom_engine/run_axioms.py [--resume RUN_NAME]
"""
import sys
import os
import time
import argparse
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CANON_ADE"))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
)
from rich.layout import Layout
from rich.text import Text
from rich import box

# ═══════════════════════════════════════════════════════════════
# WORKER FUNCTION (must be top-level for pickling)
# ═══════════════════════════════════════════════════════════════

def _worker_evaluate(task):
    """Evaluate all axioms for a single candidate. Runs in subprocess."""
    rank = task['rank']
    seed_id = task['seed_id']
    mode = task.get('mode', 'symmetric')
    
    import sys, numpy as np
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    rng = np.random.default_rng(seed_id)
    
    from CANON_ADE.axiom_engine.tensor_core import (
        build_random_factors, PERM19, decompose_irreps, KEPT19,
        compute_step51
    )
    from CANON_ADE.axiom_engine.axiom_evaluators import evaluate_all
    
    try:
        alpha, beta, gamma = build_random_factors(rank, rng=rng, symmetric=(mode == 'symmetric' and rank == 19))
        
        # Pre-compute irreps for A7 (only for R=19 where we have G-stable kept set)
        irreps = None
        perm_mats = None
        if rank == 19:
            perm_mats = PERM19
            irreps = decompose_irreps(perm_mats, 19)
        
        results = evaluate_all(
            alpha, beta, gamma, rank,
            irreps=irreps, perm_matrices=perm_mats,
            skip_classical=True,  # classical checks are expensive and need proper structure tensor
        )
        results['seed_id'] = seed_id
        results['mode'] = mode
        results['status'] = 'ok'
        
        # Strip large arrays to save memory in checkpoint
        for k in list(results.keys()):
            if isinstance(results[k], list) and len(results[k]) > 20:
                results[k] = results[k][:20]
        
    except Exception as e:
        results = {
            'R': rank, 'seed_id': seed_id, 'mode': mode,
            'status': 'error', 'error': str(e)
        }
    
    return results


def _worker_gpu_batch(task):
    """GPU-accelerated batch evaluation."""
    rank = task['rank']
    seed_start = task['seed_start']
    batch_size = task['batch_size']
    
    import sys, numpy as np
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from CANON_ADE.axiom_engine.tensor_core import build_random_factors, T_MATMUL
    from CANON_ADE.axiom_engine.gpu_kernels import (
        batch_step51_gpu, batch_rank_H_gpu, batch_relation_module_gpu, batch_fitness_gpu
    )
    
    alphas = np.zeros((batch_size, rank, 3, 3))
    betas = np.zeros((batch_size, rank, 3, 3))
    gammas = np.zeros((batch_size, rank, 3, 3))
    
    for i in range(batch_size):
        rng = np.random.default_rng(seed_start + i)
        a, b, g = build_random_factors(rank, rng=rng, symmetric=False)
        alphas[i] = a
        betas[i] = b
        gammas[i] = g
    
    # GPU batch evaluation
    Sigma, H, Delta = batch_step51_gpu(alphas, betas)
    ranks_H = batch_rank_H_gpu(H)
    dim_KA, dim_KB, dim_KC = batch_relation_module_gpu(alphas, betas, gammas)
    fitness = batch_fitness_gpu(alphas, betas, gammas, T_MATMUL)
    
    results = []
    for i in range(batch_size):
        results.append({
            'R': rank,
            'seed_id': seed_start + i,
            'mode': 'gpu_batch',
            'status': 'ok',
            'A1_rank_H': int(ranks_H[i]),
            'A1_target': rank - 9,
            'A1_pass': int(ranks_H[i] == rank - 9),
            'A4_dim_KA': int(dim_KA[i]),
            'A4_dim_KB': int(dim_KB[i]),
            'A4_dim_KC': int(dim_KC[i]),
            'A2_fitness': float(fitness[i]),
        })
    
    return results


# ═══════════════════════════════════════════════════════════════
# TUI 
# ═══════════════════════════════════════════════════════════════

class AxiomDashboard:
    # Every axiom/metric we track, grouped by category
    AXIOM_ROWS = [
        # (display_name, result_key, is_rate_based)
        # ── Novel A1-A7 ──
        ('A1 Conservation',    'A1_pass',              True),
        ('A1 R+η=27',         'A1_conservation_ok',   True),
        ('A2 Fitness',         'A2_fitness',           False),  # min-tracked
        ('A3 Σ Achievable',    'A3_achievable',        True),
        ('A3 rank(Σ)',         'A3_rank_sigma',        False),  # avg-tracked
        ('A4 Generic Pos',     'A4_generic_position',  True),
        ('A4 Minimal Rank',    'A4_minimal_rank',      True),
        ('A4 dim(K_A∩K_B)',    'A4_dim_AB',            False),  # avg-tracked
        ('A5 Gate 2',          'A5_gate2',             True),
        ('A5 Gate 3',          'A5_gate3',             True),
        ('A5 Gates Both',      'A5_gates_compatible',  True),
        ('A5 aug_gap',         'A5_aug_gap',           False),  # avg-tracked
        ('A6 Parity Sep',      'A6_parity_separation', False),  # avg-tracked
        ('A7 Quantized',       'A7_quantized',         True),
        # ── Classical ──
        ('CL Associative',     'CL_assoc_pass',        True),
        ('CL Commutative',     'CL_comm_pass',         True),
        ('CL Jacobi',          'CL_jacobi_pass',       True),
        ('CL Nilpotent',       'CL_nilpotent',         True),
        ('CL Killing rk',      'CL_killing_rank',      False),  # avg-tracked
        ('CL Semisimple',      'CL_semisimple',        True),
        # ── Novel diagnostics ──
        ('NV Coherence',       'NV_coherence_max',     False),  # avg-tracked
        ('NV Coupling rk',     'NV_coupling_rank_CA',  False),  # avg-tracked
        ('NV Spec Symmetry',   'NV_spec_symmetry',     False),  # avg-tracked
        ('NV Flat rk₀',        'NV_flat_rank_0',       False),  # avg-tracked
        # ── Summary ──
        ('Avg rank(H)',        'A1_rank_H',            False),  # avg-tracked
    ]

    def __init__(self, ranks, n_samples_per_rank, n_gpu_batches):
        self.console = Console()
        self.ranks = ranks
        self.n_cpu = n_samples_per_rank
        self.n_gpu = n_gpu_batches
        self.total = sum(n_samples_per_rank.values()) + sum(n_gpu_batches.values())
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self.throttle_count = 0

        # Per-rank accumulators: {rank: {key: [sum, count, min]}}
        self.rank_data = {r: {} for r in ranks}
        self.rank_done = {r: 0 for r in ranks}
        self.rank_errors = {r: 0 for r in ranks}

        # Activity tracking: which axiom is currently being tested per rank
        self.rank_activity = {r: '' for r in ranks}

        self.current_phase = "Initializing"

    def update(self, result):
        self.completed += 1
        r = result.get('R', 0)

        if result.get('status') == 'error':
            self.errors += 1
            if r in self.rank_errors:
                self.rank_errors[r] += 1
            return

        if r not in self.rank_data:
            return

        self.rank_done[r] = self.rank_done.get(r, 0) + 1
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

    def set_activity(self, rank, axiom_name):
        if rank in self.rank_activity:
            self.rank_activity[rank] = axiom_name

    def _fmt_cell(self, key, is_rate, stats):
        """Format a single table cell from accumulated stats."""
        if stats is None or stats['count'] == 0:
            return Text("—", style="dim")
        if is_rate:
            rate = stats['sum'] / stats['count']
            pct = rate * 100
            if pct >= 50:
                style = "bold green"
            elif pct > 0:
                style = "yellow"
            else:
                style = "red"
            return Text(f"{pct:.1f}%", style=style)
        else:
            # For fitness/error metrics, show min; for others show avg
            if 'fitness' in key or 'err' in key.lower():
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
        remaining = (self.total - self.completed) / max(rate, 0.001)
        mem = psutil.virtual_memory()

        # ── Header
        header = Text()
        header.append("═══ ADE3×3 AXIOM ENGINE ", style="bold cyan")
        header.append(f"═══  Phase: {self.current_phase}", style="yellow")

        # ── Progress bar
        pct_done = self.completed / max(self.total, 1)
        bar_width = 50
        filled = int(bar_width * pct_done)
        bar = "█" * filled + "░" * (bar_width - filled)
        bar_text = Text()
        bar_text.append(f"[{bar}] ", style="cyan")
        bar_text.append(f"{pct_done*100:.1f}%  ", style="bold white")
        bar_text.append(f"{self.completed:,}/{self.total:,}  ", style="white")
        bar_text.append(f"⏱ {timedelta(seconds=int(elapsed))}  ", style="dim")
        bar_text.append(f"ETA {timedelta(seconds=int(remaining))}  ", style="dim")
        bar_text.append(f"⚡{rate:.0f}/s  ", style="green")
        ram_style = "red bold" if mem.percent > 85 else ("yellow" if mem.percent > 70 else "green")
        bar_text.append(f"RAM {mem.used/1e9:.1f}/{mem.total/1e9:.0f}GB", style=ram_style)
        if self.errors:
            bar_text.append(f"  ❌{self.errors}", style="red")
        if self.throttle_count:
            bar_text.append(f"  🔻{self.throttle_count}", style="yellow")

        # ── Activity indicators per rank
        activity = Text()
        for r in self.ranks:
            act = self.rank_activity.get(r, '')
            done = self.rank_done.get(r, 0)
            err = self.rank_errors.get(r, 0)
            activity.append(f"R={r}", style="bold cyan")
            activity.append(f" [{done}", style="white")
            if err:
                activity.append(f",❌{err}", style="red")
            activity.append("]", style="white")
            if act:
                activity.append(f" ▸{act}", style="yellow")
            activity.append("   ")

        # ── FLIPPED TABLE: axioms as rows, ranks as columns
        table = Table(
            title="[bold]Axiom Results[/bold]",
            box=box.ROUNDED, show_header=True, header_style="bold green",
            pad_edge=False, padding=(0, 1),
        )
        table.add_column("Axiom", style="cyan", width=18, no_wrap=True)
        for r in self.ranks:
            table.add_column(f"R={r}", justify="center", width=10)

        prev_prefix = None
        for name, key, is_rate in self.AXIOM_ROWS:
            # Add separator between categories
            prefix = name.split()[0]
            if prev_prefix and prefix != prev_prefix:
                table.add_row(*[""] * (1 + len(self.ranks)), end_section=True)
            prev_prefix = prefix

            row = [Text(name, style="cyan")]
            for r in self.ranks:
                stats = self.rank_data[r].get(key)
                row.append(self._fmt_cell(key, is_rate, stats))
            table.add_row(*row)

        # Combine layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(header, border_style="cyan"), size=3),
            Layout(Panel(bar_text, border_style="blue"), size=3),
            Layout(Panel(activity, title="[bold]Worker Activity[/bold]", border_style="yellow"), size=3),
            Layout(table),
        )
        return layout


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ADE3×3 Axiom Engine")
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--ranks', type=str, default='13,19,20,21,22',
                       help='Comma-separated ranks to test')
    parser.add_argument('--cpu-samples', type=int, default=500,
                       help='CPU samples per rank (full axiom eval)')
    parser.add_argument('--gpu-batches', type=int, default=100,
                       help='GPU batch count per rank (fast screening)')
    parser.add_argument('--gpu-batch-size', type=int, default=256,
                       help='Samples per GPU batch')
    parser.add_argument('--workers', type=int, default=24,
                       help='Max CPU workers')
    parser.add_argument('--ram-limit', type=float, default=85.0,
                       help='RAM% limit for throttling')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (10 samples each)')
    args = parser.parse_args()
    
    ranks = [int(r) for r in args.ranks.split(',')]
    
    if args.quick:
        cpu_per_rank = 10
        gpu_batches = 2
        gpu_batch_size = 32
    else:
        cpu_per_rank = args.cpu_samples
        gpu_batches = args.gpu_batches
        gpu_batch_size = args.gpu_batch_size
    
    n_cpu = {r: cpu_per_rank for r in ranks}
    n_gpu = {r: gpu_batches for r in ranks}
    
    console = Console()

    # ── Banner
    console.print(Panel.fit(
        "[bold cyan]╔══════════════════════════════════════════════════╗\n"
        "║     ADE3×3 AXIOM ENGINE v1.0                    ║\n"
        "║     \"What no one has done, we shall do.\"        ║\n"
        "║                                                  ║\n"
        "║     Mapping the solution space of T_matmul       ║\n"
        "║     across ranks 13, 19, 20, 21, 22              ║\n"
        "║     with novel + classical axiom evaluation      ║\n"
        "╚══════════════════════════════════════════════════╝[/bold cyan]",
        border_style="cyan"
    ))
    
    # ── Checkpoint
    from CANON_ADE.axiom_engine.checkpoint import CheckpointManager
    
    if args.resume:
        ckpt = CheckpointManager.resume(args.resume)
        completed_ids = ckpt.get_completed_ids()
        console.print(f"[yellow]Resuming from {args.resume}: {ckpt.n_completed} results loaded[/yellow]")
    else:
        ckpt = CheckpointManager()
        completed_ids = set()
        console.print(f"[green]New run: {ckpt.run_name}[/green]")
    
    ckpt.set_metadata('ranks', ranks)
    ckpt.set_metadata('cpu_per_rank', cpu_per_rank)
    ckpt.set_metadata('gpu_batches', gpu_batches)
    ckpt.set_metadata('start_time', datetime.now().isoformat())
    
    # ── Build task lists
    # Phase 1: GPU batch screening (fast, covers many seeds)
    gpu_tasks = []
    for r in ranks:
        for batch_idx in range(gpu_batches):
            seed_start = r * 1_000_000 + batch_idx * gpu_batch_size
            # Check if any in this batch already done
            gpu_tasks.append({
                'rank': r,
                'seed_start': seed_start,
                'batch_size': gpu_batch_size,
                'phase': 'gpu_screen',
            })
    
    # Phase 2: CPU full axiom evaluation
    cpu_tasks = []
    for r in ranks:
        for i in range(cpu_per_rank):
            seed = r * 100_000 + i
            if (r, seed) not in completed_ids:
                cpu_tasks.append({
                    'rank': r,
                    'seed_id': seed,
                    'mode': 'symmetric' if r == 19 else 'random',
                    'phase': 'cpu_full',
                })
    
    total_gpu_evals = len(gpu_tasks) * gpu_batch_size
    total_tasks = total_gpu_evals + len(cpu_tasks)
    
    dashboard = AxiomDashboard(ranks, n_cpu, n_gpu)
    dashboard.total = total_tasks
    
    console.print(f"\n[bold]Task Summary:[/bold]")
    console.print(f"  GPU screening: {len(gpu_tasks)} batches × {gpu_batch_size} = {len(gpu_tasks)*gpu_batch_size:,} evals")
    console.print(f"  CPU full eval: {len(cpu_tasks)} candidates")
    console.print(f"  Workers: {args.workers} CPU, GPU: {'CUDA' if torch_available() else 'CPU fallback'}")
    console.print(f"  RAM limit: {args.ram_limit}%\n")
    
    # ── Phase 1: GPU batch screening
    dashboard.current_phase = "Phase 1: GPU Batch Screening"
    
    with Live(dashboard.build_display(), console=console, refresh_per_second=4) as live:
        
        # GPU phase — run sequentially on GPU (GPU is already parallel internally)
        for task in gpu_tasks:
            try:
                batch_results = _worker_gpu_batch(task)
                ckpt.save_results_batch(batch_results)
                for res in batch_results:
                    dashboard.update(res)
            except Exception as e:
                dashboard.errors += gpu_batch_size
                dashboard.completed += gpu_batch_size
                console.print(f"[red]GPU batch error: {e}[/red]")
            
            live.update(dashboard.build_display())
        
        # ── Phase 2: CPU full axiom evaluation with RAM throttling
        dashboard.current_phase = "Phase 2: CPU Full Axiom Evaluation"
        live.update(dashboard.build_display())
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            task_iter = iter(cpu_tasks)
            max_in_flight = args.workers * 2
            
            def _submit():
                try:
                    t = next(task_iter)
                    f = executor.submit(_worker_evaluate, t)
                    futures[f] = t
                    return True
                except StopIteration:
                    return False
            
            # Initial batch
            for _ in range(min(max_in_flight, len(cpu_tasks))):
                if not _submit():
                    break
            
            while futures:
                # Check RAM
                mem = psutil.virtual_memory()
                if mem.percent > args.ram_limit:
                    dashboard.throttle_count += 1
                    while psutil.virtual_memory().percent > args.ram_limit - 5:
                        live.update(dashboard.build_display())
                        time.sleep(1.0)
                
                # Collect completed
                done = [f for f in futures if f.done()]
                for f in done:
                    task = futures.pop(f)
                    try:
                        result = f.result(timeout=1)
                        ckpt.save_result(result)
                        dashboard.update(result)
                    except Exception as e:
                        dashboard.errors += 1
                        dashboard.completed += 1
                    
                    # Submit more
                    if mem.percent < args.ram_limit:
                        _submit()
                
                live.update(dashboard.build_display())
                
                if not done:
                    time.sleep(0.1)
        
        # ── Finalize
        dashboard.current_phase = "Finalizing"
        live.update(dashboard.build_display())
    
    summary = ckpt.finalize()
    
    # ── Final report
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Run Complete: {ckpt.run_name}[/bold green]\n"
        f"Total evaluations: {summary['n_results']:,}\n"
        f"Ranks tested: {summary.get('ranks_tested', [])}\n"
        f"Checkpoint: CANON_ADE/axiom_engine/checkpoints/{ckpt.run_name}/",
        title="[bold]Summary[/bold]",
        border_style="green"
    ))
    
    # Print pass rates
    console.print("\n[bold]Axiom Pass Rates:[/bold]")
    rate_table = Table(box=box.SIMPLE)
    rate_table.add_column("Axiom", style="cyan")
    rate_table.add_column("Rate", style="white")
    for key, val in sorted(summary.items()):
        if key.endswith('_rate'):
            rate_table.add_row(key.replace('_rate', ''), f"{val*100:.1f}%")
    console.print(rate_table)


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


if __name__ == '__main__':
    main()
