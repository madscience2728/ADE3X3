"""
run_axiom_bfs.py — BFS axiom graph traversal with Rich TUI.

Traverses the axiom graph for each target rank, discovering which
axiom combinations are constructively reachable.

Usage:
  python -m CANON_ADE.axiom_engine.run_axiom_bfs [--ranks 13,19,23,27] [--max-states 50000]
"""

import sys, os, time, argparse, json
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from CANON_ADE.axiom_engine.axiom_graph import bfs_for_rank, BFSResult
from CANON_ADE.axiom_engine.operators import OPERATORS

# All axiom IDs in display order
ALL_AXIOMS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A6b', 'A7']
AXIOM_NAMES = {
    'A1': 'Conservation',  'A2': 'Parity Partition', 'A3': 'Fiber Sums',
    'A4': 'Relation Module', 'A5': 'Coord Liberation', 'A6': 'CD Bridge',
    'A6b': 'CD Fiber', 'A7': 'Grassmannian',
}


def build_display(ranks, rank_results, rank_status, explored_counts,
                  queue_counts, start_time):
    """Build the Rich display."""
    elapsed = time.time() - start_time

    # ── Status bar ──
    status = Text()
    status.append("ADE3×3 ", style="bold cyan")
    status.append("Axiom Graph BFS ", style="bold white")
    total_explored = sum(explored_counts.values())
    total_queued = sum(queue_counts.values())
    status.append(f"explored: {total_explored:,}  ", style="white")
    status.append(f"queued: {total_queued:,}  ", style="dim")
    status.append(f"{timedelta(seconds=int(elapsed))} ", style="dim")
    for R in ranks:
        st = rank_status.get(R, 'waiting')
        style = {"waiting": "dim", "running": "bold yellow", "done": "bold green"}[st]
        status.append(f"R={R}:{st} ", style=style)

    # ── Operator graph panel ──
    graph_text = Text()
    graph_text.append("Operator Graph (edges):\n", style="bold")
    for src, tgt, _, prereqs in OPERATORS:
        pre = ','.join(sorted(prereqs)) if prereqs else '∅'
        graph_text.append(f"  {src:8s} → {tgt:5s}  [prereqs: {pre}]\n", style="dim")

    # ── Results table: axiom × rank ──
    table = Table(box=box.SIMPLE_HEAVY, show_header=True,
                  header_style="bold green", pad_edge=False, padding=(0, 1))
    table.add_column("Axiom", style="cyan", width=22, no_wrap=True)
    for R in ranks:
        table.add_column(f"R={R}", justify="center", width=12)

    for aid in ALL_AXIOMS:
        name = AXIOM_NAMES.get(aid, aid)
        row = [Text(f"{aid} {name}", style="cyan")]
        for R in ranks:
            leaves = rank_results.get(R, [])
            if not leaves:
                row.append(Text("—", style="dim"))
                continue
            # How many leaves include this axiom?
            n_with = sum(1 for l in leaves if aid in l.satisfied)
            n_total = len(leaves)
            if n_with == 0:
                row.append(Text("✗", style="red"))
            elif n_with == n_total:
                row.append(Text(f"✓ {n_with}/{n_total}", style="bold green"))
            else:
                row.append(Text(f"~ {n_with}/{n_total}", style="yellow"))
        table.add_row(*row)

    # ── Best paths panel ──
    path_text = Text()
    for R in ranks:
        leaves = rank_results.get(R, [])
        if not leaves:
            continue
        best = leaves[0]  # sorted by most axioms first
        path_text.append(f"\nR={R} ", style="bold cyan")
        path_text.append(f"best: {len(best.satisfied)} axioms ", style="bold white")
        path_text.append(f"{{{','.join(sorted(best.satisfied))}}}\n", style="green")
        for src, op_desc, tgt in best.path:
            path_text.append(f"  {src} ", style="dim")
            path_text.append(f"──▶ ", style="yellow")
            path_text.append(f"{tgt} ", style="bold green")
            path_text.append(f"({op_desc})\n", style="dim")

        # Show top 3 leaf axiom sets
        shown = set()
        count = 0
        for leaf in leaves:
            key = frozenset(leaf.satisfied)
            if key in shown:
                continue
            shown.add(key)
            count += 1
            if count > 5:
                break
            axioms = ','.join(sorted(leaf.satisfied))
            path_text.append(f"    [{count}] {{{axioms}}} depth={leaf.depth}\n",
                             style="white")

    path_panel = Panel(path_text, title="[bold]Best Paths[/bold]",
                       border_style="green") if path_text.plain.strip() else \
                 Panel(Text("(waiting...)", style="dim"), title="Best Paths")

    return Group(status, Panel(graph_text, title="[bold]Axiom Graph[/bold]",
                               border_style="blue"),
                 table, path_panel)


def main():
    parser = argparse.ArgumentParser(description="ADE3×3 Axiom Graph BFS")
    parser.add_argument('--ranks', type=str, default='13,19,21,22,23,27',
                        help='Comma-separated target ranks')
    parser.add_argument('--max-states', type=int, default=50000,
                        help='Max BFS states per rank')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--no-tui', action='store_true')
    args = parser.parse_args()

    ranks = [int(r.strip()) for r in args.ranks.split(',')]
    console = Console()

    console.print(Panel.fit(
        f"[bold cyan]ADE3×3 — Axiom Graph BFS[/bold cyan]\n"
        f"[bold]Ranks:[/bold] {ranks}\n"
        f"[bold]Max states/rank:[/bold] {args.max_states:,}\n"
        f"[bold]Operators:[/bold] {len(OPERATORS)} edges in graph",
        border_style="cyan",
    ))

    rank_results = {}
    rank_status = {R: 'waiting' for R in ranks}
    explored_counts = {R: 0 for R in ranks}
    queue_counts = {R: 0 for R in ranks}
    start_time = time.time()
    current_rank = [None]

    def on_progress(explored, queued, best_depth, best_axioms):
        R = current_rank[0]
        if R is None:
            return
        if explored == -1:  # rank complete signal
            return
        explored_counts[R] = explored
        queue_counts[R] = queued

    if args.no_tui:
        for R in ranks:
            current_rank[0] = R
            rank_status[R] = 'running'
            console.print(f"\n[bold]BFS for R={R}...[/bold]")
            t0 = time.time()
            leaves = bfs_for_rank(R, max_states=args.max_states,
                                  on_progress=on_progress)
            elapsed = time.time() - t0
            rank_results[R] = leaves
            rank_status[R] = 'done'
            console.print(f"  R={R}: {len(leaves)} leaves, "
                          f"{explored_counts[R]} explored, {elapsed:.1f}s")
            if leaves:
                best = leaves[0]
                console.print(f"  Best: {len(best.satisfied)} axioms "
                              f"{{{','.join(sorted(best.satisfied))}}}")
                for src, desc, tgt in best.path:
                    console.print(f"    {src} → {tgt}: {desc}")
    else:
        with Live(build_display(ranks, rank_results, rank_status,
                                explored_counts, queue_counts, start_time),
                  console=console, refresh_per_second=4) as live:
            for R in ranks:
                current_rank[0] = R
                rank_status[R] = 'running'
                live.update(build_display(ranks, rank_results, rank_status,
                                          explored_counts, queue_counts, start_time))

                def progress_with_update(explored, queued, best_depth, best_axioms):
                    on_progress(explored, queued, best_depth, best_axioms)
                    if explored % 50 == 0:
                        live.update(build_display(ranks, rank_results, rank_status,
                                                  explored_counts, queue_counts, start_time))

                leaves = bfs_for_rank(R, max_states=args.max_states,
                                      on_progress=progress_with_update)
                rank_results[R] = leaves
                rank_status[R] = 'done'
                live.update(build_display(ranks, rank_results, rank_status,
                                          explored_counts, queue_counts, start_time))

    # Final summary
    total_time = time.time() - start_time
    console.print(f"\n[bold]Completed in {timedelta(seconds=int(total_time))}[/bold]")

    for R in ranks:
        leaves = rank_results.get(R, [])
        console.print(f"\n[bold cyan]R={R}:[/bold cyan] {len(leaves)} terminal states")
        shown = set()
        for leaf in leaves[:10]:
            key = frozenset(leaf.satisfied)
            if key in shown:
                continue
            shown.add(key)
            axioms = ','.join(sorted(leaf.satisfied))
            console.print(f"  {{{axioms}}} (depth={leaf.depth})")

    # Save results
    out_name = args.output or f"bfs_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file = PROJECT_ROOT / "outputs" / out_name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_s': total_time,
        'ranks': ranks,
        'max_states': args.max_states,
        'results': {},
    }
    for R in ranks:
        leaves = rank_results.get(R, [])
        save_data['results'][str(R)] = [
            {
                'satisfied': sorted(l.satisfied),
                'depth': l.depth,
                'path': l.path,
            }
            for l in leaves[:100]  # cap for file size
        ]

    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {out_file}[/dim]")


if __name__ == '__main__':
    main()
