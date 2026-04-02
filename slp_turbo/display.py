"""Rich console display for SLP Turbo."""

import time
from rich.console import Console
from rich.table import Table
from rich.live import Live

from .diagnostics import tier_fitness, dead_energy_stats, kappa_ker_diagnostic


class Display:
    """Periodically prints a status table to the console."""

    def __init__(self, pool, gpu_engine, cpu_engine):
        self.pool = pool
        self.gpu = gpu_engine
        self.cpu = cpu_engine
        self.console = Console()
        self._start_time = time.time()
        self._prev_best = None
        self._plateau_count = 0

    def print_status(self):
        snap = self.pool.snapshot()
        elapsed = snap["elapsed"]
        best = snap["global_best_fit"]
        n_alive = int(snap["alive"].sum())
        n_total = len(snap["alive"])
        n_accept = int(snap["accepts"].sum())
        n_reject = int(snap["rejects"].sum())

        # Plateau detection
        if self._prev_best is not None and abs(best - self._prev_best) < 1e-12:
            self._plateau_count += 1
        else:
            self._plateau_count = 0
        self._prev_best = best

        table = Table(title=f"SLP Turbo - {elapsed:.0f}s", show_header=False,
                      min_width=60)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Best fitness", f"{best:.10f}")
        table.add_row("Alive / Total", f"{n_alive} / {n_total}")
        table.add_row("CPU iters", f"{snap['cpu_iterations']}")
        table.add_row("CPU accepts/rejects", f"{n_accept} / {n_reject}")
        table.add_row("CPU cycle", f"{self.cpu.last_cycle_time:.1f}s  (LP {self.cpu.last_lp_time:.1f}s)")
        table.add_row("CPU state", self.cpu.state)
        table.add_row("GPU bursts", f"{self.gpu.bursts}")
        table.add_row("GPU injections", f"{snap['gpu_injections']} ({snap['gpu_improvements']} improved best)")
        table.add_row("GPU best", f"{self.gpu.best_gpu_fit:.10f}" if self.gpu.best_gpu_fit < float('inf') else "-")
        table.add_row("GPU state", self.gpu.state)

        # Structural diagnostics on global best
        best_x = self.pool.global_best_x
        if best_x is not None:
            lm, dm, _ = tier_fitness(best_x)
            table.add_row("Live maxabs", f"{lm:.10f}")
            table.add_row("Dead maxabs", f"{dm:.10f}")
            try:
                _, _, cr = dead_energy_stats(best_x)
                table.add_row("Dead cancel", f"{cr:.4%}")
            except Exception:
                pass
            try:
                kk, nr, kd = kappa_ker_diagnostic(best_x)
                table.add_row("k_ker", f"{kk:.6f}  (nuisance {nr}/{kd})")
            except Exception:
                pass

        if self._plateau_count >= 5:
            table.add_row("[yellow]! PLATEAU[/yellow]",
                          f"[yellow]No improvement for {self._plateau_count} cycles[/yellow]")

        self.console.print(table)
