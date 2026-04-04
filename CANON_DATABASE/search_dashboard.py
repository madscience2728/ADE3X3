from __future__ import annotations

import itertools
import time
from typing import Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── Stage definitions for progress tracking ──
LOAD_STAGES = [
    "load: templates",
    "load: H memmap",
    "load: sigma memmap",
    "load: factor indices",
    "load: complete",
]
RAM_STAGES = [
    "load_to_ram: H",
    "load_to_ram: sigma",
    "load_to_ram: complete",
]
DEDUP_STAGES = [
    "dedup: packing H rows into int64 keys",
    "dedup: sorting",
    "dedup: unpacking unique H rows",
    "dedup: done",
]
REP_STAGES = [
    "rep: building representative index",
    "rep: done",
]


class SearchDashboard:
    """Rich TUI with heartbeat, per-stage progress bars, and live wall time."""

    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, refresh_rate: float = 0.25):
        self.refresh_rate = refresh_rate
        self.state: dict = {}
        self.console = Console() if HAS_RICH else None
        self.live: Optional[Live] = None
        self._start_time: float = 0.0
        self._heartbeat = itertools.cycle(self.SPINNER_FRAMES)
        # Stage progress tracking
        self._stage_order: list[str] = []
        self._stage_done: int = 0

    def start(self, state: dict) -> None:
        self.state = state
        self._start_time = time.perf_counter()
        if HAS_RICH:
            # auto_refresh=False — we control refresh from the main thread
            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=10,
                auto_refresh=False,
            )
            self.live.start()

    def update(self, state: dict) -> None:
        self.state = state

    def refresh(self) -> None:
        """Call from main thread in a loop to keep the TUI alive."""
        if self.live:
            self.live.update(self._render())
            self.live.refresh()

    def stop(self) -> None:
        if self.live:
            # One final refresh to show final state
            self.live.update(self._render())
            self.live.refresh()
            self.live.stop()

    def set_stages(self, stages: list[str]) -> None:
        """Set the stage list for the current phase's progress bar."""
        self._stage_order = stages
        self._stage_done = 0

    def advance_stage(self, stage_prefix: str) -> None:
        """Mark stages matching prefix as done."""
        for i, s in enumerate(self._stage_order):
            if s.startswith(stage_prefix) and i >= self._stage_done:
                self._stage_done = i + 1
                break

    def _render(self):
        if not HAS_RICH:
            return None

        beat = next(self._heartbeat)
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0

        rank = self.state.get("current_rank")
        rank_label = f"R={rank}" if rank is not None else "LOADING"
        phase = self.state.get("phase", "idle")
        db_status = self.state.get("db_status", "-")
        basis = self.state.get("basis", {})
        assembly = self.state.get("assembly", {})
        near_miss = self.state.get("near_miss", "-")
        workers = self.state.get("workers_active", 0)

        # ── Progress bar for current stage list ──
        total = len(self._stage_order)
        done = min(self._stage_done, total)
        if total > 0:
            pct = done / total
            bar_width = 30
            filled = int(pct * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_line = f"[{bar}] {done}/{total} steps ({pct*100:.0f}%)"
        else:
            progress_line = ""

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=24, style="bold cyan")
        table.add_column()
        table.add_row(f"{beat} Phase", str(phase))
        table.add_row("  DB status", str(db_status))
        if progress_line:
            table.add_row("  Progress", progress_line)
        # ETA based on basis seed progress
        seeds_done = basis.get('seeds_processed', 0)
        seeds_total = basis.get('seeds_total', 0)
        if seeds_done > 0 and seeds_total > 0 and elapsed > 0:
            rate = seeds_done / elapsed
            remaining_seeds = seeds_total - seeds_done
            eta_s = remaining_seeds / rate if rate > 0 else 0
            eta_min = eta_s / 60
            pct = seeds_done / seeds_total * 100
            wall_str = f"{elapsed:.1f}s  ETA {eta_min:.1f}m  ({pct:.1f}% seeds)"
        else:
            wall_str = f"{elapsed:.1f}s"
        table.add_row(f"{beat} Wall time", wall_str)
        # Search progress bar (seed-level)
        if seeds_total > 0:
            pct_s = seeds_done / seeds_total
            bar_w = 30
            filled_s = int(pct_s * bar_w)
            seed_bar = "█" * filled_s + "░" * (bar_w - filled_s)
            table.add_row("  Seed progress", f"[{seed_bar}] {seeds_done}/{seeds_total} ({pct_s*100:.1f}%)")
        ram_used = self.state.get("ram_used_gb", 0.0)
        ram_avail = self.state.get("ram_avail_gb", 0.0)
        if ram_used == 0.0 and ram_avail == 0.0:
            ram_text = "[red]psutil unavailable[/red]"
        else:
            ram_style = "red" if ram_avail < 8.0 else ("yellow" if ram_avail < 16.0 else "green")
            ram_text = f"[{ram_style}]{ram_used:.1f}GB used / {ram_avail:.1f}GB free[/{ram_style}]"
        table.add_row(f"  RAM", ram_text)
        table.add_row("  Workers active", str(workers))
        table.add_row("", "")  # spacer
        table.add_row("  Bases built", f"{basis.get('built_bases', 0):,}")
        table.add_row("  Nodes explored", f"{basis.get('explored_nodes', 0):,}")
        table.add_row("  Pruned dependent", f"{basis.get('pruned_dependent', 0):,}")
        # Per-seed depth-1 progress bar
        d1_done = basis.get('depth1_done', 0)
        d1_total = basis.get('depth1_total', 0)
        if d1_total > 0:
            d1_pct = d1_done / d1_total
            d1_w = 30
            d1_filled = int(d1_pct * d1_w)
            d1_bar = "█" * d1_filled + "░" * (d1_w - d1_filled)
            table.add_row("  Seed DFS (d=1)", f"[{d1_bar}] {d1_done:,}/{d1_total:,} ({d1_pct*100:.1f}%)")
        if phase == "harvest":
            # Stage 1: show packet stats, hide assembly fields
            packets_written = basis.get('built_bases', 0) - basis.get('_skipped', 0)
            table.add_row("  Packets written", f"{packets_written:,}")
            table.add_row("  Avg hits/packet", f"{assembly.get('avg_hits', 0.0):,.0f}" if assembly.get('avg_hits') else "-")
        else:
            # Stage 2 / legacy search: show assembly stats
            table.add_row("  Assembly bases", f"{assembly.get('bases_tested', 0):,}")
            table.add_row("  Avg hits", f"{assembly.get('avg_hits', 0.0):,.1f}")
            table.add_row("  Gate 2 pass", f"{assembly.get('gate2_passes', 0):,}")
            table.add_row("  Gate 3 pass", f"{assembly.get('gate3_passes', 0):,}")
            table.add_row("  Configs tested", f"{assembly.get('configs_tested', 0):,}")
        table.add_row("  Best near-miss", str(near_miss))

        mode = "HARVEST" if phase == "harvest" else ("ASSEMBLE" if phase == "assemble" else "SEARCH")
        return Panel(
            table,
            title=Text(f" {rank_label} {mode} ", style="bold white on blue"),
            subtitle=Text(f" {beat} heartbeat alive ", style="dim"),
            border_style="blue",
        )


def create_search_dashboard(refresh_rate: float = 0.25) -> SearchDashboard:
    return SearchDashboard(refresh_rate=refresh_rate)