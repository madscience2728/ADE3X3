"""
swap_dashboard.py — Rich TUI for the swap optimizer coordinator.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


class SwapDashboard:
    """
    Manages a Rich Live display for the swap optimizer.

    State is updated by the coordinator after processing each worker message.
    Call .render() inside a Live context to refresh the display.
    """

    def __init__(self, n_workers: int, R: int):
        self.n_workers   = n_workers
        self.R           = R
        self.start_time  = time.time()

        # Global state
        self.global_best_score: Optional[tuple] = None
        self.total_swaps    = 0
        self.total_improv   = 0
        self.total_restarts = 0

        # Per-worker state
        self.worker_scores:    dict[int, Optional[tuple]] = {}
        self.worker_diags:     dict[int, dict] = {}
        self.worker_swaps:     dict[int, int]  = {i: 0 for i in range(n_workers)}
        self.worker_restarts:  dict[int, int]  = {i: 0 for i in range(n_workers)}
        self.worker_done:      dict[int, bool] = {i: False for i in range(n_workers)}

        self.global_best_diag: Optional[dict] = None

        # Score history (last 10 global improvements)
        self.score_history: deque[tuple] = deque(maxlen=10)

        # For rate calculation
        self._last_swaps = 0
        self._last_time  = time.time()
        self._rate = 0.0

    def update(self, worker_id: int, kind: str, payload: dict):
        now = time.time()

        if kind in ('update', 'solution'):
            score   = tuple(payload['score'])
            swaps   = payload.get('swaps', 0)
            improv  = payload.get('improv', 0)
            restart = payload.get('restart', 0)

            self.worker_scores[worker_id]   = score
            self.worker_swaps[worker_id]    = swaps
            self.worker_restarts[worker_id] = restart
            if 'diag' in payload:
                self.worker_diags[worker_id] = payload['diag']

            # Update global totals (approximate -- workers are independent)
            old_total = sum(self.worker_swaps.get(i, 0) for i in range(self.n_workers))
            if old_total > self._last_swaps + 5000:
                elapsed = now - self._last_time
                if elapsed > 0:
                    self._rate = (old_total - self._last_swaps) / elapsed
                self._last_swaps = old_total
                self._last_time  = now

            if self.global_best_score is None or score < self.global_best_score:
                self.global_best_score = score
                self.score_history.append(score)
                if 'diag' in payload:
                    self.global_best_diag = payload['diag']

        elif kind == 'status':
            swaps    = payload.get('swaps', 0)
            restarts = payload.get('restarts', 0)
            self.worker_swaps[worker_id]    = swaps
            self.worker_restarts[worker_id] = restarts
            if payload.get('done'):
                self.worker_done[worker_id] = True

        # Update computed totals
        self.total_swaps    = sum(self.worker_swaps.values())
        self.total_improv   = max(self.total_improv, sum(
            self.worker_restarts.values()))  # rough proxy
        self.total_restarts = sum(self.worker_restarts.values())

    def render(self) -> Panel:
        elapsed = time.time() - self.start_time
        h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
        wall_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"

        # --- Global best panel ---
        if self.global_best_score:
            g1, combined = self.global_best_score
            g1_color = "green" if g1 == 0 else "yellow"
            best_diag = self.global_best_diag or {}
            leak = best_diag.get('delta_leak', '?')
            aug  = best_diag.get('augmented_gap', '?')
            si   = best_diag.get('sigma_innovation', '?')
            lk_color = "green" if leak == 0 else "red"
            ag_color = "green" if aug == 0  else "yellow"
            best_lines = (
                f"  Gate1 gap: [{g1_color}]{g1}[/{g1_color}]   "
                f"Delta leak: [{lk_color}]{leak}[/{lk_color}]   "
                f"Aug gap: [{ag_color}]{aug}[/{ag_color}]\n"
                f"  Sigma innov: {si}   Combined: {combined:.4f}\n"
            )
        else:
            best_lines = "  (waiting for first result...)\n"

        # --- Score history ---
        hist_str = " -> ".join(
            f"({','.join(f'{v:.2g}' if isinstance(v, float) else str(v) for v in s)})"
            for s in self.score_history
        ) or "(none)"

        # --- Rate ---
        rate_str = f"{self._rate/1000:.1f}K" if self._rate > 0 else "?"

        # --- Delta leak histogram ---
        leaks = [
            self.worker_diags.get(i, {}).get('delta_leak', None)
            for i in range(self.n_workers)
            if i in self.worker_diags
        ]
        leaks = [v for v in leaks if v is not None]
        if leaks:
            from collections import Counter
            lc = Counter(leaks)
            max_ct = max(lc.values())
            bars = []
            for v in range(min(leaks), max(leaks)+1):
                ct = lc.get(v, 0)
                bar = "#" * int(8 * ct / max_ct) if max_ct > 0 else ""
                bars.append(f"{v}:[green]{bar}[/green]" if v == 0 else f"{v}:{bar}")
            hist_str_leak = "  ".join(bars)
        else:
            hist_str_leak = "(no data)"

        # --- Worker table ---
        wt = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        wt.add_column("ID", width=3)
        wt.add_column("Best score", width=40)
        wt.add_column("Swaps", justify="right", width=10)
        wt.add_column("Restarts", justify="right", width=8)
        wt.add_column("Done", width=5)

        for i in range(self.n_workers):
            sc   = self.worker_scores.get(i)
            sw   = self.worker_swaps.get(i, 0)
            rst  = self.worker_restarts.get(i, 0)
            done = self.worker_done.get(i, False)
            diag = self.worker_diags.get(i, {})
            if sc:
                leak = diag.get('delta_leak', '?')
                aug  = diag.get('augmented_gap', '?')
                si   = diag.get('sigma_innovation', '?')
                sc_str = f"g1={sc[0]} leak={leak} aug={aug} si={si} comb={sc[1]:.2f}"
            else:
                sc_str = "..."
            wt.add_row(
                str(i),
                sc_str,
                f"{sw:,}",
                str(rst),
                "Y" if done else "",
            )

        # Build content
        content = (
            f"[bold]R={self.R} SWAP OPTIMIZER  --  {self.n_workers} workers[/bold]\n\n"
            f"[bold]GLOBAL BEST:[/bold]\n{best_lines}\n"
            f"[bold]SEARCH STATS:[/bold]\n"
            f"  Wall time: {wall_str}   Total swaps: {self.total_swaps:,}\n"
            f"  Restarts: {self.total_restarts:,}   Rate: ~{rate_str} swaps/s\n\n"
            f"[bold]SCORE HISTORY (last 10 improvements):[/bold]\n"
            f"  {hist_str}\n\n"
            f"[bold]DELTA LEAK HISTOGRAM (current worker bests):[/bold]\n"
            f"  {hist_str_leak}\n\n"
        )

        from rich.console import Group
        return Panel(
            Group(Text.from_markup(content), wt),
            title="[bold cyan]ADE3x3 Phase 4 -- Swap Optimizer[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
