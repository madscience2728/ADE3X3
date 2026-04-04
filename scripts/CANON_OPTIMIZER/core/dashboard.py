"""
Real-time Rich TUI dashboard for the exhaustive search.
"""

import time
from typing import Dict, List, Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class Dashboard:
    def __init__(self, refresh_rate: float = 0.5):
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.live: Optional[Live] = None
        self.ranks: List[int] = []
        self.stats: Dict = {}
        self.elapsed: float = 0
        self.worker_messages: List[str] = []
        self.solution_found: Dict[int, bool] = {}

    def start(self, ranks: List[int], stats: Dict):
        self.ranks = sorted(ranks)
        self.stats = stats
        self.solution_found = {R: False for R in ranks}
        self.live = Live(self._render(), console=self.console,
                         refresh_per_second=1.0 / self.refresh_rate)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def update(self, stats: Dict, elapsed: float):
        self.stats = stats
        self.elapsed = elapsed
        if self.live:
            self.live.update(self._render())

    def on_progress(self, msg):
        self.worker_messages.append(str(msg.message)[:80])
        if len(self.worker_messages) > 5:
            self.worker_messages = self.worker_messages[-5:]

    def on_solution(self, msg):
        self.solution_found[msg.R] = True

    def on_worker_done(self, msg):
        pass

    def _render(self):
        # Header
        header = Text("R-RANK MATRIX MULTIPLICATION TENSOR — EXHAUSTIVE SEARCH",
                       style="bold white on blue", justify="center")

        # Gate statistics table
        gate_table = Table(title="Gate Statistics", show_header=True,
                           header_style="bold cyan")
        gate_table.add_column("R", style="bold", width=6)
        gate_table.add_column("Checked", justify="right", width=14)
        gate_table.add_column("Gate 1", justify="right", width=10, style="yellow")
        gate_table.add_column("Gate 2", justify="right", width=10, style="cyan")
        gate_table.add_column("Gate 3", justify="right", width=10, style="green")
        gate_table.add_column("Solutions", justify="right", width=10)
        gate_table.add_column("Status", width=12)

        for R in self.ranks:
            s = self.stats.get(R, {})
            checked = s.get('candidates_checked', 0)
            g1 = s.get('gate1_passes', 0)
            g2 = s.get('gate2_passes', 0)
            g3 = s.get('gate3_passes', 0)
            sols = s.get('solutions_found', 0)
            active = s.get('workers_active', 0)
            done = s.get('workers_done', 0)

            if self.solution_found.get(R):
                status = Text("SOLVED!", style="bold bright_green blink")
            elif done > 0 and active == 0:
                status = Text("DONE", style="dim")
            elif active > 0:
                status = Text("RUNNING", style="bright_white")
            else:
                status = Text("PENDING", style="dim yellow")

            gate_table.add_row(
                str(R),
                f"{checked:,}",
                f"{g1:,}" if g1 else "—",
                f"{g2:,}" if g2 else "—",
                f"{g3:,}" if g3 else "—",
                f"{sols}" if sols else "—",
                status,
            )

        # Elapsed time
        h, rem = divmod(int(self.elapsed), 3600)
        m, sec = divmod(rem, 60)
        time_str = f"Wall time: {h}h {m:02d}m {sec:02d}s"

        # Conservation law
        conservation = Text(
            "Conservation Law: R + η_null = 27 ✓",
            style="dim green"
        )

        # Recent messages
        msg_text = "\n".join(self.worker_messages[-5:]) if self.worker_messages else "(waiting...)"

        # Build layout
        content = Text()
        content.append(f"\n  {time_str}\n\n", style="magenta")

        panel = Panel(
            gate_table,
            title="[bold]EXHAUSTIVE ENUMERATION PIPELINE[/bold]",
            subtitle=f"[magenta]{time_str}[/magenta]",
            border_style="blue",
        )

        return panel


def create_dashboard(refresh_rate: float = 0.5) -> Dashboard:
    return Dashboard(refresh_rate=refresh_rate)
