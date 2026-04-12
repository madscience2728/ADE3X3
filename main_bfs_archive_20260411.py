#!/usr/bin/env python3

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
ED_DIR = ROOT / "ed_degree_computation"
RESULTS_DIR = ED_DIR / "results"
RUNNER = ED_DIR / "run_rank.jl"
DEFAULT_JULIA = Path(r"C:\Users\madsc\AppData\Local\Programs\Julia-1.12.5\bin\julia.exe")
DEFAULT_RANKS = [13, 19, 20, 21, 22, 23, 27]
@dataclass
class RankProcess:
    rank: int
    threads: int
    process: subprocess.Popen
    start_time: float
    queue: queue.Queue
    json_path: Path
    last_json_mtime: float = 0.0
    launch_index: int = 0
    recent_lines: deque = field(default_factory=lambda: deque(maxlen=50))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ED-rank solves with live JSONL checkpoints.")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in DEFAULT_RANKS), help="Comma-separated ranks to launch")
    parser.add_argument("--target-cpu", type=float, default=95.0, help="Target average CPU usage percentage during smoke tuning")
    parser.add_argument("--warmup-seconds", type=int, default=20, help="Seconds to wait before sampling CPU in a smoke-tune cycle")
    parser.add_argument("--sample-seconds", type=int, default=8, help="Seconds of CPU sampling per smoke-tune cycle")
    parser.add_argument("--initial-threads", type=int, default=0, help="Initial Julia threads per process. Defaults to logical_cpus // nranks")
    parser.add_argument("--max-threads-per-process", type=int, default=0, help="Maximum Julia threads per process. Defaults to logical CPU count")
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path. Defaults to ed_degree_computation/results/session_<timestamp>.jsonl")
    parser.add_argument("--no-autotune", action="store_true", help="Disable smoke tuning and launch once with the chosen thread count")
    parser.add_argument("--graceful-stop-seconds", type=int, default=5, help="Seconds to wait after terminate() before kill()")
    return parser.parse_args()


def parse_ranks(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def default_jsonl_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"session_{timestamp}.jsonl"


def json_dump_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def reader_thread(rank_state: RankProcess) -> None:
    assert rank_state.process.stdout is not None
    for line in rank_state.process.stdout:
        line = line.rstrip("\r\n")
        rank_state.queue.put(("line", line))
    rank_state.queue.put(("eof", ""))


def launch_rank(rank: int, threads: int, launch_index: int, jsonl_path: Path) -> RankProcess:
    command = [
        str(DEFAULT_JULIA if DEFAULT_JULIA.exists() else Path(os.environ.get("ADE_JULIA", "julia"))),
        f"--project={ED_DIR}",
        f"--threads={threads}",
        str(RUNNER),
        str(rank),
    ]
    env = os.environ.copy()
    env["ADE_JSONL_PATH"] = str(jsonl_path)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    state = RankProcess(
        rank=rank,
        threads=threads,
        process=process,
        start_time=time.time(),
        queue=queue.Queue(),
        json_path=RESULTS_DIR / f"ed_R{rank}.json",
        launch_index=launch_index,
    )
    thread = threading.Thread(target=reader_thread, args=(state,), daemon=True)
    thread.start()
    return state


def terminate_states(states: list[RankProcess], graceful_stop_seconds: int) -> None:
    for state in states:
        if state.process.poll() is None:
            state.process.terminate()
    deadline = time.time() + graceful_stop_seconds
    while time.time() < deadline:
        if all(state.process.poll() is not None for state in states):
            return
        time.sleep(0.2)
    for state in states:
        if state.process.poll() is None:
            state.process.kill()


def maybe_append_result_snapshot(state: RankProcess, jsonl_path: Path, event_name: str) -> None:
    if not state.json_path.exists():
        return
    mtime = state.json_path.stat().st_mtime
    if mtime <= state.last_json_mtime:
        return
    try:
        payload = json.loads(state.json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    state.last_json_mtime = mtime
    json_dump_line(
        jsonl_path,
        {
            "event": event_name,
            "rank": state.rank,
            "threads": state.threads,
            "launch_index": state.launch_index,
            "timestamp": time.time(),
            "payload": payload,
        },
    )


def drain_output(states: list[RankProcess], jsonl_path: Path) -> None:
    for state in states:
        while True:
            try:
                kind, line = state.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                state.recent_lines.append(line)
                print(f"[R={state.rank:02d}] {line}", flush=True)
                if "monodromy checkpoint written" in line:
                    maybe_append_result_snapshot(state, jsonl_path, "monodromy_checkpoint")
                if "tracking checkpoint written" in line:
                    maybe_append_result_snapshot(state, jsonl_path, "tracking_checkpoint")
            elif kind == "eof":
                pass
        if state.process.poll() is not None:
            maybe_append_result_snapshot(state, jsonl_path, "process_exit_snapshot")


def all_finished(states: list[RankProcess]) -> bool:
    return all(state.process.poll() is not None for state in states)


def average_cpu_percent(sample_seconds: int) -> float:
    samples = []
    for _ in range(sample_seconds):
        samples.append(psutil.cpu_percent(interval=1.0))
    return sum(samples) / len(samples) if samples else 0.0


def choose_initial_threads(ranks: list[int], logical_cpus: int, requested_initial: int) -> int:
    if requested_initial > 0:
        return requested_initial
    return max(1, logical_cpus // max(1, len(ranks)))


def thread_schedule(initial_threads: int, max_threads: int) -> list[int]:
    values = []
    current = max(1, initial_threads)
    while current < max_threads:
        values.append(current)
        current = min(max_threads, current * 2)
        if values and current == values[-1]:
            break
    if not values or values[-1] != max_threads:
        values.append(max_threads)
    return values


def smoke_tune(ranks: list[int], args: argparse.Namespace, jsonl_path: Path) -> tuple[int, list[RankProcess]]:
    logical_cpus = psutil.cpu_count(logical=True) or 1
    max_threads = args.max_threads_per_process or logical_cpus
    initial_threads = choose_initial_threads(ranks, logical_cpus, args.initial_threads)

    if args.no_autotune:
        states = [launch_rank(rank, initial_threads, 0, jsonl_path) for rank in ranks]
        return initial_threads, states

    schedule = thread_schedule(initial_threads, max_threads)
    for attempt_index, threads in enumerate(schedule):
        print(f"[launcher] smoke attempt {attempt_index + 1}/{len(schedule)} with {threads} Julia threads per rank", flush=True)
        states = [launch_rank(rank, threads, attempt_index, jsonl_path) for rank in ranks]
        warmup_deadline = time.time() + args.warmup_seconds
        while time.time() < warmup_deadline:
            drain_output(states, jsonl_path)
            time.sleep(0.5)
        cpu_avg = average_cpu_percent(args.sample_seconds)
        json_dump_line(
            jsonl_path,
            {
                "event": "smoke_profile",
                "attempt_index": attempt_index,
                "threads": threads,
                "cpu_percent_avg": round(cpu_avg, 3),
                "target_cpu": args.target_cpu,
                "timestamp": time.time(),
            },
        )
        print(f"[launcher] smoke CPU average with {threads} threads/process: {cpu_avg:.1f}%", flush=True)
        if cpu_avg >= args.target_cpu or threads == schedule[-1]:
            print(f"[launcher] keeping current launch with {threads} threads/process", flush=True)
            return threads, states
        print(f"[launcher] CPU below target, restarting with more threads", flush=True)
        terminate_states(states, args.graceful_stop_seconds)
    raise RuntimeError("unreachable smoke-tune state")


def main() -> int:
    args = parse_args()
    ranks = parse_ranks(args.ranks)
    if not RUNNER.exists():
        print(f"missing Julia runner: {RUNNER}", file=sys.stderr)
        return 1

    jsonl_path = Path(args.jsonl) if args.jsonl else default_jsonl_path()
    print(f"[launcher] writing JSONL checkpoints to {jsonl_path}", flush=True)
    print(f"[launcher] ranks = {ranks}", flush=True)

    threads, states = smoke_tune(ranks, args, jsonl_path)
    print(f"[launcher] active configuration: {threads} Julia threads/process across {len(ranks)} ranks", flush=True)

    try:
        while not all_finished(states):
            drain_output(states, jsonl_path)
            time.sleep(0.5)
        drain_output(states, jsonl_path)
    except KeyboardInterrupt:
        print("[launcher] keyboard interrupt, terminating all rank processes", flush=True)
        terminate_states(states, args.graceful_stop_seconds)
        drain_output(states, jsonl_path)
        return 130

    worst_exit = 0
    for state in states:
        returncode = state.process.poll()
        worst_exit = max(worst_exit, 0 if returncode is None else abs(returncode))
        json_dump_line(
            jsonl_path,
            {
                "event": "process_exit",
                "rank": state.rank,
                "threads": state.threads,
                "launch_index": state.launch_index,
                "timestamp": time.time(),
                "returncode": returncode,
            },
        )
        print(f"[launcher] rank {state.rank} exited with code {returncode}", flush=True)

    print(f"[launcher] session complete, JSONL checkpoints saved to {jsonl_path}", flush=True)
    return 0 if worst_exit == 0 else 1


if __name__ == "__main__":
        raise SystemExit(main())
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# 1 BLAS thread per process — parallelism is at the process level
for _env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLAS_NUM_THREADS'):
    os.environ[_env_var] = '1'
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

import psutil
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.tree import Tree
from rich.layout import Layout
from rich.spinner import Spinner
from rich import box

from CANON_ADE.axiom_engine.axiom_graph import bfs_for_rank, BFSResult
from CANON_ADE.axiom_engine.operators import OPERATORS

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALL_AXIOMS = [
    # ── Core constructive axioms (BFS graph nodes) ──
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A6b', 'A7',
    # ── Constraint-derived diagnostics ──
    'gate2', 'gate3', 'cd_parity', 'fiber_forced',
    # ── Classical algebra properties (from evaluate_all) ──
    'CL_assoc', 'CL_comm', 'CL_jacobi', 'CL_nilp', 'CL_killing',
    # ── Novel / spectral diagnostics ──
    'NV_spec', 'NV_coupling', 'NV_coherence', 'NV_flat_rank',
]
AXIOM_NAMES = {
    # ── Core constructive axioms ──
    'A1': 'Conservation',     'A2': 'Parity Partition',
    'A3': 'Fiber Sums',       'A4': 'Relation Module',
    'A5': 'Coord Liberation', 'A6': 'CD Bridge',
    'A6b': 'CD Fiber',        'A7': 'Grassmannian',
    # ── Constraint-derived diagnostics ──
    'gate2': 'Gate2 δ=0',     'gate3': 'Gate3 aug=9',
    'cd_parity': 'CD Parity', 'fiber_forced': 'All Fibers Forced',
    # ── Classical algebra properties ──
    'CL_assoc': 'Associativity',  'CL_comm': 'Commutativity',
    'CL_jacobi': 'Jacobi',        'CL_nilp': 'Nilpotency',
    'CL_killing': 'Killing Form',
    # ── Novel / spectral diagnostics ──
    'NV_spec': 'Spectral Sig',      'NV_coupling': 'Coupling',
    'NV_coherence': 'Coherence',     'NV_flat_rank': 'Flat Rank',
}
AXIOM_GLYPHS = {
    'A1': '⊕', 'A2': '◈', 'A3': '≋', 'A4': '⊗',
    'A5': '⇌', 'A6': '⊛', 'A6b': '⊙', 'A7': '∇',
    'gate2': '▓', 'gate3': '░', 'cd_parity': '±', 'fiber_forced': 'Φ',
    'CL_assoc': 'α', 'CL_comm': 'γ', 'CL_jacobi': 'J',
    'CL_nilp': 'N', 'CL_killing': 'K',
    'NV_spec': 'λ', 'NV_coupling': 'Λ', 'NV_coherence': 'μ', 'NV_flat_rank': 'ρ',
}
RANK_COLORS = {
    13: 'bright_magenta', 19: 'bright_cyan', 20: 'bright_blue',
    21: 'bright_green', 22: 'bright_yellow', 23: 'orange1', 27: 'white',
}


def rank_style(R):
    return RANK_COLORS.get(R, 'white')


def _format_duration(seconds, *, max_seconds=10 * 365 * 24 * 3600):
    """Format a duration safely for UI display without overflowing timedelta."""
    if seconds is None or not isinstance(seconds, (int, float)):
        return "—"
    if not math.isfinite(seconds):
        return "—"
    if seconds < 0:
        seconds = 0
    if seconds > max_seconds:
        return ">10y"
    return str(timedelta(seconds=int(seconds)))


# ═══════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION — show operator edges as a DAG
# ═══════════════════════════════════════════════════════════════

def build_graph_tree():
    """Build a Rich Tree showing the operator graph structure."""
    tree = Tree("[bold cyan]Axiom Graph[/bold cyan]", guide_style="dim")

    # Group operators by source
    by_source = defaultdict(list)
    for src, tgt, _, prereqs in OPERATORS:
        pre = ','.join(sorted(prereqs)) if prereqs else '∅'
        by_source[src].append((tgt, pre))

    for src in ['root', 'A2', 'A4', 'A7', 'A1', 'A3', 'A5']:
        if src not in by_source:
            continue
        if src == 'root':
            label = "[bold white]● root[/bold white]"
        else:
            glyph = AXIOM_GLYPHS.get(src, '·')
            name = AXIOM_NAMES.get(src, src)
            label = f"[bold green]{glyph} {src}[/bold green] [dim]{name}[/dim]"

        branch = tree.add(label)
        for tgt, pre in by_source[src]:
            tgt_glyph = AXIOM_GLYPHS.get(tgt, '·')
            tgt_name = AXIOM_NAMES.get(tgt, tgt)
            prereq_str = f" [dim]req:{pre}[/dim]" if pre != '∅' else ""
            branch.add(f"[yellow]→[/yellow] [green]{tgt_glyph} {tgt}[/green] "
                       f"[dim]{tgt_name}[/dim]{prereq_str}")

    return tree


# ═══════════════════════════════════════════════════════════════
# MAIN DISPLAY BUILDER
# ═══════════════════════════════════════════════════════════════

def build_display(ranks, rank_results, rank_status, explored_counts,
                  queue_counts, start_time, leaf_counts, cpu_stats=None,
                  effective_budget=None, rank_start_times=None, shard_counts=None,
                  eta_estimators=None, overall_eta_estimator=None,
                  rank_axiom_frontier=None):
    elapsed = time.time() - start_time
    now = time.time()
    if effective_budget is None:
        effective_budget = {}

    # ── Header line ──
    hdr = Text()
    hdr.append(" ADE3×3 ", style="bold bright_white on blue")
    hdr.append("  ", style="")
    hdr.append(f"⏱ {_format_duration(elapsed)}  ", style="dim")
    total_exp = sum(explored_counts.values())
    total_lf = sum(leaf_counts.values())
    total_budget = sum(effective_budget.get(R, 50000) for R in ranks)
    overall_pct = 100 * total_exp / total_budget if total_budget else 0
    n_done = sum(1 for s in rank_status.values() if s == 'done')
    # Overall ETA
    if total_exp > 0 and n_done < len(ranks):
        if overall_eta_estimator is not None:
            eta_s = overall_eta_estimator.eta_seconds(total_budget)
            if eta_s is not None:
                eta_str = _format_duration(eta_s)
            else:
                eta_str = "—"
        else:
            rate = total_exp / elapsed if elapsed > 0 else 0
            remaining = total_budget - total_exp
            eta_s = remaining / rate if rate > 0 else 0
            eta_str = _format_duration(eta_s)
    else:
        eta_str = "—"
    hdr.append(f"explored {total_exp:,}/{total_budget:,} ({overall_pct:.0f}%)  ", style="white")
    hdr.append(f"ETA {eta_str}  ", style="bold yellow")
    hdr.append(f"leaves {total_lf:,}  ", style="white")
    hdr.append(f"done {n_done}/{len(ranks)}", style="bold green" if n_done == len(ranks) else "white")

    # ── CPU bar ──
    cpu_line = Text()
    if cpu_stats:
        overall = cpu_stats.get('overall', 0.0)
        per_core = cpu_stats.get('per_core', [])
        n_workers = cpu_stats.get('n_workers', 0)
        n_cores = len(per_core)
        if overall > 80:
            ov_style = "bold bright_red"
        elif overall > 50:
            ov_style = "bold yellow"
        elif overall > 20:
            ov_style = "bold white"
        else:
            ov_style = "bold dim"
        cpu_line.append(f" CPU {overall:5.1f}%", style=ov_style)
        cpu_line.append(f" {n_workers}w/{n_cores}c ", style="dim")
        for pct in per_core:
            if pct > 80:
                cpu_line.append("█", style="bright_red")
            elif pct > 50:
                cpu_line.append("▓", style="yellow")
            elif pct > 20:
                cpu_line.append("▒", style="green")
            elif pct > 5:
                cpu_line.append("░", style="dim")
            else:
                cpu_line.append("·", style="dim")
        worker_pids = cpu_stats.get('worker_pids', {})
        if worker_pids:
            cpu_line.append("  ", style="")
            for R, pct in sorted(worker_pids.items()):
                if pct > 0:
                    cpu_line.append(f" R{R}:", style=f"{rank_style(R)}")
                    cpu_line.append(f"{pct:.0f}%", style="bold white")
        # RAM indicator
        ram_gb = cpu_stats.get('ram_used_gb', 0)
        ram_total = cpu_stats.get('ram_total_gb', 1)
        ram_pct = cpu_stats.get('ram_pct', 0)
        ram_style = "bold bright_red" if ram_pct > 85 else "bold yellow" if ram_pct > 70 else "dim"
        cpu_line.append(f"  RAM {ram_gb:.1f}/{ram_total:.0f}GB ({ram_pct:.0f}%)", style=ram_style)

    # ── Per-rank progress bars ──
    BAR_WIDTH = 30
    bars = Text()
    for R in ranks:
        st = rank_status.get(R, 'waiting')
        exp = explored_counts.get(R, 0)
        que = queue_counts.get(R, 0)
        lc = leaf_counts.get(R, 0)
        n_shards = (shard_counts or {}).get(R, 1)
        r_start = (rank_start_times or {}).get(R)
        r_budget = effective_budget.get(R, 50000)

        pct = min(100, 100 * exp / r_budget) if r_budget > 0 else 0
        filled = int(BAR_WIDTH * pct / 100)
        empty = BAR_WIDTH - filled

        bars.append(f" R={R:2d}", style=f"bold {rank_style(R)}")
        bars.append(f" ×{n_shards} ", style="dim")

        if st == 'waiting':
            bars.append("░" * BAR_WIDTH, style="dim")
            bars.append(f" waiting\n", style="dim")
        elif st == 'running':
            bars.append("█" * filled, style="yellow")
            bars.append("░" * empty, style="dim")
            bars.append(f" {pct:5.1f}%", style="bold yellow")
            bars.append(f" {exp:>7,}/{r_budget:,}", style="white")
            bars.append(f" q={que:,} lf={lc:,}", style="dim")
            # Per-rank ETA (non-linear)
            if r_start and exp > 0:
                if eta_estimators and R in eta_estimators:
                    r_eta = eta_estimators[R].eta_seconds(r_budget)
                    if r_eta is not None:
                        bars.append(f" ETA {_format_duration(r_eta)}", style="yellow")
                else:
                    r_elapsed = now - r_start
                    r_rate = exp / r_elapsed
                    r_remaining = r_budget - exp
                    r_eta = r_remaining / r_rate if r_rate > 0 else 0
                    bars.append(f" ETA {_format_duration(r_eta)}", style="yellow")
            bars.append("\n")
        else:
            bars.append("█" * BAR_WIDTH, style="green")
            bars.append(f" 100.0%", style="bold green")
            bars.append(f" {exp:>7,}/{r_budget:,}", style="green")
            bars.append(f" lf={lc:,}", style="green")
            if r_start:
                took = now - r_start
                bars.append(f" ✓ {_format_duration(took)}", style="bold green")
            bars.append("\n")

    # ── Overall progress bar ──
    overall_filled = int(BAR_WIDTH * overall_pct / 100)
    overall_empty = BAR_WIDTH - overall_filled
    overall_bar = Text()
    overall_bar.append(f" TOTAL ", style="bold white")
    overall_bar.append("    ", style="")
    if n_done == len(ranks):
        overall_bar.append("█" * BAR_WIDTH, style="bold green")
        overall_bar.append(f" 100% DONE in {_format_duration(elapsed)}", style="bold green")
    else:
        overall_bar.append("█" * overall_filled, style="bold cyan")
        overall_bar.append("░" * overall_empty, style="dim")
        overall_bar.append(f" {overall_pct:5.1f}%", style="bold cyan")
        overall_bar.append(f" ETA {eta_str}", style="bold yellow")

    # ── Axiom × rank matrix (compact) ──
    table = Table(
        box=box.SIMPLE_HEAD, show_header=True, show_edge=False,
        header_style="bold", pad_edge=False, padding=(0, 0),
        collapse_padding=True,
    )
    table.add_column("Axiom", style="cyan", width=16, no_wrap=True)
    for R in ranks:
        table.add_column(f"R{R}", justify="center", width=5,
                         header_style=f"bold {rank_style(R)}")

    for aid in ALL_AXIOMS:
        g = AXIOM_GLYPHS.get(aid, '·')
        name = AXIOM_NAMES.get(aid, aid)[:10]
        row = [Text(f"{g} {aid}", style="cyan")]
        for R in ranks:
            leaves = rank_results.get(R, [])
            if not leaves:
                row.append(Text("·" if rank_status.get(R) == 'running' else "—", style="dim"))
                continue
            n_with = sum(1 for l in leaves if aid in l.satisfied)
            if n_with == 0:
                row.append(Text("✗", style="red"))
            else:
                pct = 100 * n_with / len(leaves)
                if pct == 100:
                    row.append(Text("✓", style="bold green"))
                elif pct > 50:
                    row.append(Text(f"{pct:.0f}", style="yellow"))
                else:
                    row.append(Text(f"{pct:.0f}", style="dim yellow"))
        table.add_row(*row)

    # ── Compact best-path lines ──
    paths = Text()
    for R in ranks:
        leaves = rank_results.get(R, [])
        if not leaves:
            continue
        best = leaves[0]
        n_unique = len(set(frozenset(l.satisfied) for l in leaves))
        paths.append(f" R={R}", style=f"bold {rank_style(R)}")
        paths.append(f" {len(best.satisfied)}ax ", style="white")
        for a in sorted(best.satisfied):
            paths.append(f"{AXIOM_GLYPHS.get(a,'·')}{a} ", style="green")
        paths.append(f"({n_unique} configs) ", style="dim")
        if best.path:
            chain = '→'.join(t for _, _, t in best.path)
            paths.append(chain, style="dim")
        paths.append("\n")

    # ── Axiom frontier (what's being explored per rank) ──
    frontier_text = Text()
    if rank_axiom_frontier:
        for R in ranks:
            fr = rank_axiom_frontier.get(R) if rank_axiom_frontier else None
            if not fr or rank_status.get(R) != 'running':
                continue
            frontier_text.append(f" R={R}", style=f"bold {rank_style(R)}")
            frontier_text.append(": ", style="dim")
            top = sorted(fr.items(), key=lambda x: -x[1])[:4]
            for i, (ax_str, cnt) in enumerate(top):
                if i > 0:
                    frontier_text.append(" | ", style="dim")
                frontier_text.append(f"{{{ax_str}}}", style="cyan")
                frontier_text.append(f":{cnt}", style="white")
            frontier_text.append("\n")

    return Group(
        hdr, cpu_line,
        Panel(Group(bars, overall_bar),
              title="[bold]Progress[/bold]", border_style="yellow", padding=(0, 0)),
        Panel(frontier_text, title="[bold]Frontier[/bold]", border_style="cyan", padding=(0, 0)) if frontier_text.plain.strip() else Text(),
        table,
        Panel(paths if paths.plain.strip() else Text(" (waiting…)", style="dim"),
              title="[bold]Best[/bold]", border_style="green", padding=(0, 0)),
    )


# ═══════════════════════════════════════════════════════════════
# PARALLEL WORKER (module-level for multiprocessing spawn)
# ═══════════════════════════════════════════════════════════════

def _subtree_worker(worker_id, R, slice_start, slice_end, max_states,
                    progress_queue, result_queue):
    """
    BFS worker for a SLICE of root children for one rank.
    Receives only ints — zero pickle overhead.
    Reconstructs root children locally, takes [slice_start:slice_end],
    then runs bfs_from_states on that subtree.
    """
    import sys, traceback

    def _send_result(status, serialized, total_count, note=None):
        result_queue.put({
            'kind': 'result',
            'worker_id': worker_id,
            'rank': R,
            'status': status,
            'serialized': serialized,
            'total_count': total_count,
            'note': note,
        })

    try:
        from CANON_ADE.axiom_engine.axiom_graph import expand_roots, bfs_from_states

        all_children = expand_roots(R)
        my_states = all_children[slice_start:slice_end]

        if not my_states:
            _send_result('empty-shard', [], 0, 'No root children assigned to this worker shard')
            return

        def on_prog(explored, queued, best_depth, best_axioms, frontier):
            try:
                progress_queue.put_nowait((
                    'progress', worker_id, R, explored, queued, best_axioms, frontier
                ))
            except Exception:
                pass

        leaves = bfs_from_states(R, my_states, max_states=max_states,
                                 on_progress=on_prog)
        total_leaf_count = len(leaves)

        # Cap serialized leaves to avoid Queue/pickle overflow
        MAX_SERIALIZE = 500
        leaves_to_send = leaves[:MAX_SERIALIZE]  # already sorted by axiom count desc

        # Evaluate classical/novel axioms on top leaves via tensor sampling
        from CANON_ADE.axiom_engine.tensor_core import build_symmetric_factors, build_random_factors
        from CANON_ADE.axiom_engine.axiom_evaluators import evaluate_all
        import numpy as np

        N_EVAL_LEAVES = 20   # evaluate top N leaves per worker
        N_SAMPLES = 3        # random tensor samples per leaf

        CL_NV_KEYS = {
            # key in evaluate_all result -> axiom name, with lambda to check pass
        }

        def _check_cl_nv(res):
            """Extract which CL_*/NV_* axioms pass from evaluate_all results."""
            hits = {}
            if res.get('CL_assoc_pass'):
                hits['CL_assoc'] = True
            if res.get('CL_comm_pass'):
                hits['CL_comm'] = True
            if res.get('CL_jacobi_pass'):
                hits['CL_jacobi'] = True
            if res.get('CL_nilpotent'):
                hits['CL_nilp'] = True
            if res.get('CL_killing_rank', 0) > 0:
                hits['CL_killing'] = True
            # Novel: define reasonable thresholds
            if res.get('NV_spec_symmetry', 999) < 1.0:
                hits['NV_spec'] = True
            if res.get('NV_coupling_diag_ratio', 0) > 0.3:
                hits['NV_coupling'] = True
            if res.get('NV_coherence_excess', 999) < 0.1:
                hits['NV_coherence'] = True
            # Flat rank: all three flattenings should have rank 9
            if (res.get('NV_flat_rank_0', 0) == 9 and
                res.get('NV_flat_rank_1', 0) == 9 and
                res.get('NV_flat_rank_2', 0) == 9):
                hits['NV_flat_rank'] = True
            return hits

        def _eval_leaf(leaf):
            """Sample tensors for a leaf and evaluate classical/novel axioms."""
            state = leaf.state
            if state is None:
                return {}
            kept = state.get('kept')
            rng = np.random.default_rng()
            hits = {}
            for _ in range(N_SAMPLES):
                try:
                    if kept and R == 19:
                        params = rng.standard_normal(81)
                        alpha, beta, gamma = build_symmetric_factors(params, kept)
                    else:
                        alpha, beta, gamma = build_random_factors(R, rng=rng)
                    actual_R = alpha.shape[0]
                    res = evaluate_all(alpha, beta, gamma, actual_R,
                                       skip_classical=False, skip_novel=False)
                    for ax_name, _ in _check_cl_nv(res).items():
                        hits[ax_name] = True
                except Exception:
                    continue
            return hits

        serialized = []
        for i, l in enumerate(leaves_to_send):
            d = {
                'rank': l.rank,
                'satisfied': sorted(l.satisfied),
                'path': l.path,
                'depth': l.depth,
                'constraints': l.constraints or {},
            }
            if i < N_EVAL_LEAVES:
                d['eval_hits'] = _eval_leaf(l)
            serialized.append(d)

        _send_result('ok', serialized, total_leaf_count)
    except Exception as e:
        print(f"[worker {worker_id} R={R}] EXCEPTION: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        _send_result('exception', [], 0, f"{type(e).__name__}: {e}")


def _derive_extra_axioms(leaf):
    """Derive constraint-based axiom tags from leaf state constraints."""
    c = leaf.constraints or {}
    extra = set()
    if c.get('gate2'):
        extra.add('gate2')
    if c.get('gate3'):
        extra.add('gate3')
    if c.get('cd_even_nonzero') is not None or c.get('cd_odd_nonzero') is not None:
        extra.add('cd_parity')
    if c.get('fiber_type') == 'all_forced':
        extra.add('fiber_forced')
    return extra


def _deserialize_leaves(serialized):
    leaves = []
    for d in serialized:
        leaf = BFSResult(
            rank=d['rank'],
            satisfied=d['satisfied'],
            path=d['path'],
            state=None,
            depth=d['depth'],
        )
        leaf.constraints = d.get('constraints', {})
        # Merge constraint-derived axioms into satisfied
        extra = _derive_extra_axioms(leaf)
        # Merge tensor-evaluated classical/novel axioms
        eval_hits = d.get('eval_hits', {})
        if extra or eval_hits:
            leaf.satisfied = frozenset(set(leaf.satisfied) | extra | set(eval_hits.keys()))
        leaves.append(leaf)
    return leaves


def _sample_cpu(all_procs, worker_rank_map=None):
    """Sample CPU stats from all worker processes."""
    alive_list = [p for p in all_procs.values() if p.is_alive()]
    stats = {
        'overall': psutil.cpu_percent(interval=0),
        'per_core': psutil.cpu_percent(interval=0, percpu=True),
        'n_workers': len(alive_list),
        'worker_pids': {},
    }
    # Aggregate CPU by rank
    rank_cpu = defaultdict(float)
    for wid, proc in list(all_procs.items()):
        if not proc.is_alive():
            continue
        R = worker_rank_map.get(wid) if worker_rank_map else None
        if R is None:
            continue
        try:
            p = psutil.Process(proc.pid)
            pct = p.cpu_percent(interval=0)
            rank_cpu[R] += pct
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    stats['worker_pids'] = dict(rank_cpu)
    # RAM usage
    mem = psutil.virtual_memory()
    stats['ram_used_gb'] = mem.used / (1024**3)
    stats['ram_total_gb'] = mem.total / (1024**3)
    stats['ram_pct'] = mem.percent
    return stats


# ═══════════════════════════════════════════════════════════════
# NON-LINEAR ETA ESTIMATOR
# ═══════════════════════════════════════════════════════════════

class ETAEstimator:
    """
    Fits explored(t) to a quadratic model using a rolling window of recent
    samples, then smooths the output with an exponential moving average
    to prevent wild jumps.
    """
    def __init__(self, max_samples=30):
        self.samples = deque(maxlen=max_samples)  # (time_offset, explored)
        self._last_eta = None  # EMA-smoothed ETA
        self._ema_alpha = 0.15  # smoothing factor (lower = more stable)

    def add(self, t, explored):
        self.samples.append((t, explored))

    def eta_seconds(self, budget):
        """Return estimated seconds until explored reaches budget, or None."""
        n = len(self.samples)
        if n < 3:
            return None
        t_last, exp_last = self.samples[-1]
        if exp_last >= budget:
            return 0.0

        raw_eta = None

        # Try quadratic fit if we have enough points
        if n >= 8:
            raw_eta = self._quadratic_eta(budget)

        # Fallback: linear from recent rate
        if raw_eta is None or raw_eta <= 0:
            recent = list(self.samples)[-15:]
            t0, e0 = recent[0]
            t1, e1 = recent[-1]
            dt = t1 - t0
            de = e1 - e0
            if dt <= 0 or de <= 0:
                return self._last_eta
            rate = de / dt
            raw_eta = (budget - exp_last) / rate

        # EMA smooth to prevent jumps
        if self._last_eta is None:
            self._last_eta = raw_eta
        else:
            self._last_eta = self._ema_alpha * raw_eta + (1 - self._ema_alpha) * self._last_eta

        return self._last_eta

    def _quadratic_eta(self, budget):
        """Fit explored = a*t^2 + b*t + c, solve for t when explored = budget."""
        pts = list(self.samples)
        n = len(pts)
        s = [0.0] * 5  # sum of t^0..t^4
        se = [0.0] * 3  # sum of t^0*e .. t^2*e
        for t, e in pts:
            tp = 1.0
            for k in range(5):
                s[k] += tp
                if k < 3:
                    se[k] += tp * e
                tp *= t
        A = [
            [s[4], s[3], s[2]],
            [s[3], s[2], s[1]],
            [s[2], s[1], s[0]],
        ]
        b = [se[2], se[1], se[0]]
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                   -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                   +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        D = det3(A)
        if abs(D) < 1e-12:
            return None
        def replace_col(m, col, v):
            return [[v[r] if c == col else m[r][c] for c in range(3)] for r in range(3)]
        a = det3(replace_col(A, 0, b)) / D
        bcoef = det3(replace_col(A, 1, b)) / D
        c = det3(replace_col(A, 2, b)) / D

        t_last = pts[-1][0]
        cc = c - budget
        if abs(a) < 1e-12:
            if abs(bcoef) < 1e-12:
                return None
            t_target = -cc / bcoef
        else:
            disc = bcoef * bcoef - 4 * a * cc
            if disc < 0:
                return None
            sqrt_disc = disc ** 0.5
            t1 = (-bcoef + sqrt_disc) / (2 * a)
            t2 = (-bcoef - sqrt_disc) / (2 * a)
            candidates = [t for t in [t1, t2] if t > t_last]
            if not candidates:
                return None
            t_target = min(candidates)

        return max(0.0, t_target - t_last)


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME
# ═══════════════════════════════════════════════════════════════

CHECKPOINT_FILE = 'checkpoint.pkl'


def _checkpoint_path(checkpoint_dir):
    return Path(checkpoint_dir) / CHECKPOINT_FILE


def save_checkpoint(checkpoint_dir, rank_results, rank_status, explored_counts,
                    leaf_counts, ranks, max_states):
    """Save completed rank results to a checkpoint file."""
    path = _checkpoint_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'timestamp': datetime.now().isoformat(),
        'max_states': max_states,
        'ranks': ranks,
        'completed': {},
    }
    for R in ranks:
        if rank_status.get(R) == 'done' and R in rank_results:
            leaves = rank_results[R]
            data['completed'][R] = {
                'explored': explored_counts.get(R, 0),
                'leaf_count': len(leaves),
                'serialized': [
                    {
                        'rank': l.rank,
                        'satisfied': sorted(l.satisfied),
                        'path': l.path,
                        'depth': l.depth,
                        'constraints': l.constraints or {},
                    }
                    for l in leaves
                ],
            }
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic on Windows (same dir)


def load_checkpoint(checkpoint_dir, ranks, max_states):
    """Load checkpoint. Returns (rank_results, rank_status, explored_counts, leaf_counts) or None."""
    path = _checkpoint_path(checkpoint_dir)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return None
    if data.get('max_states') != max_states:
        return None
    rank_results = {}
    rank_status = {}
    explored_counts = {}
    leaf_counts = {}
    for R in ranks:
        if R in data.get('completed', {}):
            info = data['completed'][R]
            leaves = _deserialize_leaves(info['serialized'])
            leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))
            rank_results[R] = leaves
            rank_status[R] = 'done'
            explored_counts[R] = info['explored']
            leaf_counts[R] = info['leaf_count']
        else:
            rank_status[R] = 'waiting'
            explored_counts[R] = 0
            leaf_counts[R] = 0
    return rank_results, rank_status, explored_counts, leaf_counts


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ADE3×3 — Algebra Discovery Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                          # default: all target ranks
  python main.py --ranks 19,22            # specific ranks only
  python main.py --max-states 200000      # deeper search
  python main.py --no-tui                 # plain text mode
""",
    )
    parser.add_argument('--ranks', '--rank', type=str, default='13,19,20,21,22,23,27',
                        help='Comma-separated target ranks (default: 13,19,20,21,22,23,27)')
    parser.add_argument('--max-states', type=int, default=10000,
                        help='Max BFS states per rank (default: 10000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Explicit worker cap for BFS multiprocessing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (saved to outputs/)')
    parser.add_argument('--no-tui', action='store_true',
                        help='Disable live TUI, use plain text output')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                        help='Directory for checkpoint files')
    parser.add_argument('--max-ram-gb', type=float, default=60.0,
                        help='Target cap for ADE worker RAM usage (GB, default: 60)')
    parser.add_argument('--ram-headroom-gb', type=float, default=16.0,
                        help='System RAM headroom to leave unused when sizing workers (GB, default: 16)')
    parser.add_argument('--worker-warmup-s', type=float, default=30.0,
                        help='Seconds to wait after launching a worker before admitting another queued worker (default: 30)')
    args = parser.parse_args()

    ranks = [int(r.strip()) for r in args.ranks.split(',')]
    console = Console()

    # ── Banner ──
    console.print()
    console.print(Panel.fit(
        "[bold bright_white]ADE3×3[/bold bright_white]  "
        "[bold cyan]Algebra Discovery Engine[/bold cyan]\n\n"
        f"[bold]Target ranks:[/bold]  {', '.join(str(r) for r in ranks)}\n"
        f"[bold]Max states:[/bold]    {args.max_states:,} per rank\n"
        f"[bold]Operators:[/bold]     {len(OPERATORS)} edges in axiom graph\n"
        f"[bold]Axioms:[/bold]        {', '.join(ALL_AXIOMS)}\n\n"
        "[dim]All operations are constructive. No optimization. No regression.[/dim]",
        border_style="cyan",
        title="[bold cyan]╍╍╍ ADE ╍╍╍[/bold cyan]",
    ))
    console.print()

    rank_results = {}
    rank_status = {R: 'waiting' for R in ranks}
    explored_counts = {R: 0 for R in ranks}
    queue_counts = {R: 0 for R in ranks}
    leaf_counts = {R: 0 for R in ranks}
    start_time = time.time()
    cpu_stats = {}
    rank_start_times = {}
    shard_counts = {R: 1 for R in ranks}
    effective_budget = {R: args.max_states for R in ranks}

    # ── ETA estimators (per-rank + overall) ──
    eta_estimators = {R: ETAEstimator() for R in ranks}
    overall_eta_estimator = ETAEstimator()

    # ── Resume from checkpoint if requested ──
    ranks_to_run = list(ranks)
    if args.resume:
        ckpt = load_checkpoint(args.checkpoint_dir, ranks, args.max_states)
        if ckpt is not None:
            ck_results, ck_status, ck_explored, ck_leaves = ckpt
            rank_results.update(ck_results)
            rank_status.update(ck_status)
            explored_counts.update(ck_explored)
            leaf_counts.update(ck_leaves)
            resumed = [R for R in ranks if ck_status.get(R) == 'done']
            ranks_to_run = [R for R in ranks if ck_status.get(R) != 'done']
            if resumed:
                console.print(f"[bold green]Resumed {len(resumed)} ranks from checkpoint: "
                              f"{', '.join(str(r) for r in resumed)}[/bold green]")
            if not ranks_to_run:
                console.print("[bold green]All ranks already completed![/bold green]")
        else:
            console.print("[dim]No valid checkpoint found, starting fresh[/dim]")

    # ── Expand roots and shard across workers (RAM-scaled) ──
    n_cores = os.cpu_count() or 24
    # Each worker imports heavy numeric stacks, so be conservative on startup fan-out.
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    usable_gb = max(0.0, min(args.max_ram_gb, available_gb - args.ram_headroom_gb))
    RAM_PER_WORKER_GB = 6.0
    max_by_ram = max(1, int(usable_gb / RAM_PER_WORKER_GB)) if usable_gb > 0 else 1
    N_WORKERS = min(n_cores, max_by_ram)
    if args.workers is not None:
        N_WORKERS = max(1, min(N_WORKERS, args.workers))
    console.print(f"[dim]  RAM budget: {args.max_ram_gb:.0f}GB for ADE workers, "
                  f"{available_gb:.1f}GB available, {args.ram_headroom_gb:.1f}GB reserved → {N_WORKERS} workers "
                  f"(capped by {'RAM' if max_by_ram < n_cores else 'cores'})[/dim]")
    progress_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    all_procs = {}          # worker_id -> Process
    worker_rank_map = {}    # worker_id -> R
    workers_per_rank = defaultdict(int)  # R -> count of workers
    rank_pending = defaultdict(int)      # R -> how many workers still running
    rank_axiom_frontier = defaultdict(dict)  # R -> {axiom_set_str: count}
    worker_done = set()
    worker_failure_notes = defaultdict(list)  # R -> [note, ...]
    rank_completion_reason = {}
    quit_reason = None
    pending_rank_queue = deque()
    next_worker_id = 0
    root_counts = {}
    raw_alloc = {}
    worker_launch_times = {}

    if ranks_to_run:
        from CANON_ADE.axiom_engine.axiom_graph import expand_roots

        # Count root children per rank to allocate workers proportionally
        for R in ranks_to_run:
            n = len(expand_roots(R))
            root_counts[R] = max(n, 1)
            console.print(f"[dim]  R={R}: {n} root children[/dim]")

        # Live RAM scheduler: queue one worker per rank and only admit a new
        # worker when measured RSS and system headroom say there is room.
        raw_alloc = {R: 1 for R in ranks_to_run}

        def _running_worker_slots():
            return sum(rank_pending[R] for R in ranks if rank_status.get(R) == 'running')

        def _current_worker_ram_gb():
            total = 0.0
            alive = 0
            max_rss = 0.0
            for proc in all_procs.values():
                if not proc.is_alive():
                    continue
                try:
                    rss = psutil.Process(proc.pid).memory_info().rss / (1024**3)
                    total += rss
                    max_rss = max(max_rss, rss)
                    alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total, alive, max_rss

        def _has_room_for_next_worker():
            worker_used_gb, alive_workers, max_worker_rss_gb = _current_worker_ram_gb()
            if alive_workers >= N_WORKERS:
                return False
            now = time.time()
            if worker_launch_times:
                most_recent_launch = max(worker_launch_times.values())
                if now - most_recent_launch < args.worker_warmup_s:
                    return False
            vm = psutil.virtual_memory()
            system_room_gb = vm.available / (1024**3) - args.ram_headroom_gb
            budget_room_gb = args.max_ram_gb - worker_used_gb
            if alive_workers > 0:
                avg_worker_rss_gb = worker_used_gb / alive_workers
                estimated_next_gb = max(3.0, avg_worker_rss_gb * 1.75, max_worker_rss_gb * 1.5)
            else:
                estimated_next_gb = 3.0
            return min(system_room_gb, budget_room_gb) >= estimated_next_gb

        def _launch_rank(R):
            nonlocal next_worker_id
            n_children = root_counts[R]
            n_workers = raw_alloc[R]
            shard_counts[R] = n_workers
            rank_status[R] = 'running'
            rank_start_times[R] = time.time()
            rank_pending[R] = n_workers

            # Divide children evenly across workers
            chunk = max(1, n_children // n_workers)
            for i in range(n_workers):
                s_start = i * chunk
                s_end = n_children if i == n_workers - 1 else (i + 1) * chunk
                per_worker_budget = args.max_states // n_workers
                if i == n_workers - 1:
                    per_worker_budget = args.max_states - per_worker_budget * (n_workers - 1)

                p = multiprocessing.Process(
                    target=_subtree_worker,
                    args=(next_worker_id, R, s_start, s_end, per_worker_budget,
                          progress_queue, result_queue),
                    daemon=True,
                )
                all_procs[next_worker_id] = p
                worker_rank_map[next_worker_id] = R
                worker_launch_times[next_worker_id] = time.time()
                workers_per_rank[R] += 1
                next_worker_id += 1
                p.start()

        def _launch_more_ranks():
            if not pending_rank_queue or not _has_room_for_next_worker():
                return []
            R = pending_rank_queue.popleft()
            _launch_rank(R)
            return [R]

        pending_rank_queue.extend(ranks_to_run)
        launched_now = _launch_more_ranks()

        console.print(f"[bold green]Launched {len(all_procs)} workers across "
                      f"{len(launched_now)} active ranks[/bold green]")
        for R in launched_now:
            console.print(f"  [bold {rank_style(R)}]R={R}[/bold {rank_style(R)}]: "
                          f"{shard_counts[R]} workers, "
                          f"{root_counts[R]} root children")
        if pending_rank_queue:
            console.print(f"[dim]{len(pending_rank_queue)} ranks queued behind live RAM admission control[/dim]")
    console.print()

    # Prime psutil
    psutil.cpu_percent(interval=0, percpu=True)
    for proc in all_procs.values():
        try:
            psutil.Process(proc.pid).cpu_percent(interval=0)
        except Exception:
            pass

    # Track partial results per rank (accumulate leaves from multiple workers)
    rank_leaves_accum = defaultdict(list)  # R -> [leaf, leaf, ...]

    def _record_rank_completion(R, reason):
        if R not in rank_completion_reason:
            rank_completion_reason[R] = reason

    def _launch_more_ranks_if_possible():
        if not pending_rank_queue:
            return
        if not _has_room_for_next_worker():
            return
        launched = []
        if pending_rank_queue:
            R = pending_rank_queue.popleft()
            n_children = root_counts[R]
            n_workers = raw_alloc.get(R, 1)
            shard_counts[R] = n_workers
            rank_status[R] = 'running'
            rank_start_times[R] = time.time()
            rank_pending[R] = n_workers
            chunk = max(1, n_children // n_workers)
            nonlocal next_worker_id
            for i in range(n_workers):
                s_start = i * chunk
                s_end = n_children if i == n_workers - 1 else (i + 1) * chunk
                per_worker_budget = args.max_states // n_workers
                if i == n_workers - 1:
                    per_worker_budget = args.max_states - per_worker_budget * (n_workers - 1)
                p = multiprocessing.Process(
                    target=_subtree_worker,
                    args=(next_worker_id, R, s_start, s_end, per_worker_budget,
                          progress_queue, result_queue),
                    daemon=True,
                )
                all_procs[next_worker_id] = p
                worker_rank_map[next_worker_id] = R
                worker_launch_times[next_worker_id] = time.time()
                workers_per_rank[R] += 1
                next_worker_id += 1
                p.start()
            launched.append(R)
        for R in launched:
            console.print(f"[dim]RAM admitted queued search: R={R} with {shard_counts[R]} worker(s)[/dim]")

    def _drain_and_collect():
        """Drain progress + result queues, update per-rank stats and ETA estimators."""
        now = time.time()
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                _, wid, R, explored, queued, best_ax, frontier = msg
                # Accumulate explored/queued across workers for this rank
                # We track per-worker and sum
                _worker_explored[wid] = explored
                _worker_queued[wid] = queued
                explored_counts[R] = sum(_worker_explored.get(w, 0)
                                         for w, wr in worker_rank_map.items() if wr == R)
                queue_counts[R] = sum(_worker_queued.get(w, 0)
                                      for w, wr in worker_rank_map.items() if wr == R)
                r_start = rank_start_times.get(R, start_time)
                eta_estimators[R].add(now - r_start, explored_counts[R])
                # Update axiom frontier
                if frontier:
                    for axioms, count in frontier:
                        key = ','.join(axioms)
                        rank_axiom_frontier[R][key] = rank_axiom_frontier[R].get(key, 0) + count
            except Exception:
                break
        checkpoint_needed = False
        while not result_queue.empty():
            try:
                msg = result_queue.get_nowait()
                if isinstance(msg, dict):
                    wid = msg.get('worker_id')
                    R = msg.get('rank')
                    serialized = msg.get('serialized', [])
                    total_count = msg.get('total_count', 0)
                    status = msg.get('status', 'ok')
                    note = msg.get('note')
                else:
                    _, wid, R, serialized, total_count = msg
                    status = 'ok'
                    note = None
                worker_done.add(wid)
                if status != 'ok':
                    detail = f"worker {wid}: {status}"
                    if note:
                        detail += f" ({note})"
                    worker_failure_notes[R].append(detail)
                leaves = _deserialize_leaves(serialized)
                rank_leaves_accum[R].extend(leaves)
                # Track total (including uncapped) leaf count
                if not hasattr(_drain_and_collect, '_total_leaves'):
                    _drain_and_collect._total_leaves = defaultdict(int)
                _drain_and_collect._total_leaves[R] += total_count
                leaf_counts[R] = _drain_and_collect._total_leaves[R]
                rank_pending[R] -= 1
                # Only mark done when ALL workers for this rank are done
                if rank_pending[R] <= 0:
                    all_leaves = rank_leaves_accum[R]
                    all_leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))
                    rank_results[R] = all_leaves
                    rank_status[R] = 'done'
                    if worker_failure_notes.get(R):
                        _record_rank_completion(R, 'completed with worker issues')
                    else:
                        _record_rank_completion(R, 'completed normally')
                    checkpoint_needed = True
            except Exception:
                break
        # Feed overall ETA estimator
        total_exp = sum(explored_counts.values())
        overall_eta_estimator.add(now - start_time, total_exp)
        # Auto-checkpoint when a rank completes
        if checkpoint_needed:
            try:
                save_checkpoint(args.checkpoint_dir, rank_results, rank_status,
                                explored_counts, leaf_counts, ranks, args.max_states)
            except Exception:
                pass

    # Per-worker tracking dicts
    _worker_explored = {}
    _worker_queued = {}
    _ram_killed = False
    _emergency_shutdown = False
    _emergency_message = None

    def _emergency_shutdown_now(message):
        nonlocal _ram_killed, _emergency_shutdown, _emergency_message, quit_reason
        if _emergency_shutdown:
            return
        _ram_killed = True
        _emergency_shutdown = True
        _emergency_message = message
        quit_reason = message
        console.print()
        console.print("[bold white on red] EMERGENCY RAM SHUTDOWN [/bold white on red]")
        console.print(f"[bold bright_red]{message}[/bold bright_red]")
        console.print("[bold bright_red]Stopping all workers and saving partial results.[/bold bright_red]")
        for p in all_procs.values():
            if p.is_alive():
                p.kill()
        for R in ranks:
            if rank_status[R] == 'running':
                leaves = rank_leaves_accum.get(R, [])
                if leaves:
                    leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))
                    rank_results[R] = leaves
                rank_status[R] = 'done'
                _record_rank_completion(R, 'emergency RAM shutdown')
        try:
            save_checkpoint(args.checkpoint_dir, rank_results, rank_status,
                            explored_counts, leaf_counts, ranks, args.max_states)
        except Exception:
            pass

    def _check_worker_health():
        nonlocal quit_reason
        for wid, proc in all_procs.items():
            if wid in worker_done:
                continue
            if proc.is_alive():
                continue
            exitcode = proc.exitcode
            R = worker_rank_map.get(wid)
            worker_done.add(wid)
            rank_pending[R] -= 1
            note = f"worker {wid} exited before sending a result"
            if exitcode is not None:
                note += f" (exit code {exitcode})"
            worker_failure_notes[R].append(note)
            if rank_pending[R] <= 0 and rank_status.get(R) == 'running':
                leaves = rank_leaves_accum.get(R, [])
                leaves.sort(key=lambda r: (-len(r.satisfied), r.depth))
                rank_results[R] = leaves
                rank_status[R] = 'done'
                _record_rank_completion(R, 'all workers exited early')
            if quit_reason is None:
                quit_reason = 'worker exited unexpectedly'

    def _check_ram():
        """Global non-silent RAM emergency shutdown."""
        nonlocal _ram_killed
        if _ram_killed:
            return
        worker_used_gb = 0.0
        for proc in all_procs.values():
            if not proc.is_alive():
                continue
            try:
                worker_used_gb += psutil.Process(proc.pid).memory_info().rss / (1024**3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)

        if available_gb <= args.ram_headroom_gb:
            _emergency_shutdown_now(
                f"System available RAM dropped to {available_gb:.1f}GB with reserve floor {args.ram_headroom_gb:.1f}GB"
            )
            return

        if worker_used_gb > args.max_ram_gb:
            _emergency_shutdown_now(
                f"ADE worker RAM exceeded budget ({worker_used_gb:.1f}GB > {args.max_ram_gb:.1f}GB)"
            )

    def _all_done():
        return all(s == 'done' for s in rank_status.values())

    def _stop_requested():
        return _emergency_shutdown or _all_done()

    def _build():
        return build_display(
            ranks, rank_results, rank_status, explored_counts,
            queue_counts, start_time, leaf_counts, cpu_stats,
            effective_budget, rank_start_times, shard_counts,
            eta_estimators, overall_eta_estimator, rank_axiom_frontier)

    # ── No-TUI mode ──
    if args.no_tui:
        reported = set()
        loop_count = 0
        while not _stop_requested():
            _drain_and_collect()
            _check_worker_health()
            _launch_more_ranks_if_possible()
            cpu_stats = _sample_cpu(all_procs, worker_rank_map)
            _check_ram()
            loop_count += 1

            for R in ranks:
                if rank_status[R] == 'done' and R not in reported:
                    reported.add(R)
                    leaves = rank_results[R]
                    elapsed = time.time() - start_time
                    console.print(f"[bold {rank_style(R)}]━━━ BFS R={R} ━━━[/bold {rank_style(R)}]")
                    console.print(f"  {len(leaves):,} leaves, "
                                  f"{explored_counts[R]:,} explored, {elapsed:.1f}s")
                    if leaves:
                        best = leaves[0]
                        axiom_str = ' '.join(f"{AXIOM_GLYPHS.get(a,'·')}{a}"
                                             for a in sorted(best.satisfied))
                        console.print(f"  [bold green]Best: {axiom_str}[/bold green]")
                        for src, desc, tgt in best.path:
                            console.print(f"    [dim]{src}[/dim] [yellow]→[/yellow] "
                                          f"[green]{tgt}[/green] [dim]{desc}[/dim]")
                    console.print()
            time.sleep(0.1)

    # ── Live TUI mode ──
    else:
        console.clear()
        with Live(_build(), console=console, refresh_per_second=4,
                  screen=True, vertical_overflow="visible") as live:
            while not _stop_requested():
                _drain_and_collect()
                _check_worker_health()
                _launch_more_ranks_if_possible()
                cpu_stats = _sample_cpu(all_procs, worker_rank_map)
                _check_ram()
                live.update(_build())
                time.sleep(0.12)
            _drain_and_collect()
            _check_worker_health()
            _launch_more_ranks_if_possible()
            cpu_stats = _sample_cpu(all_procs, worker_rank_map)
            live.update(_build())

    # Clean up workers
    for p in all_procs.values():
        p.join(timeout=5)

    # ── Final summary ──
    total_time = time.time() - start_time
    console.print()
    console.print(f"[bold]Completed in {timedelta(seconds=int(total_time))}[/bold]")
    if quit_reason:
        console.print(f"[bold bright_red]Quit reason:[/bold bright_red] {quit_reason}")
    else:
        console.print("[bold green]Quit reason:[/bold green] all ranks completed normally")
    console.print()

    # Summary table
    summary = Table(box=box.ROUNDED, title="[bold cyan]Summary[/bold cyan]",
                    title_style="bold", show_lines=True)
    summary.add_column("Rank", style="bold", justify="center")
    summary.add_column("Leaves", justify="right")
    summary.add_column("Explored", justify="right")
    summary.add_column("Best Axioms", style="green")
    summary.add_column("Best Path Depth", justify="center")

    for R in ranks:
        leaves = rank_results.get(R, [])
        if leaves:
            best = leaves[0]
            axiom_str = ', '.join(sorted(best.satisfied))
            depth = str(best.depth)
        else:
            axiom_str = "—"
            depth = "—"
        summary.add_row(
            Text(str(R), style=f"bold {rank_style(R)}"),
            f"{len(leaves):,}",
            f"{explored_counts.get(R, 0):,}",
            axiom_str,
            depth,
        )
    console.print(summary)

    if rank_completion_reason or worker_failure_notes:
        console.print()
        console.print("[bold cyan]Exit Diagnostics[/bold cyan]")
        for R in ranks:
            reason = rank_completion_reason.get(R, 'no completion record')
            style = 'bright_red' if worker_failure_notes.get(R) else 'green'
            console.print(f"  [bold {style}]R={R}[/bold {style}]: {reason}")
            for note in worker_failure_notes.get(R, [])[:5]:
                console.print(f"    [dim]- {note}[/dim]")
            extra = len(worker_failure_notes.get(R, [])) - 5
            if extra > 0:
                console.print(f"    [dim]... +{extra} more[/dim]")

    # Unique algebra types
    console.print()
    for R in ranks:
        leaves = rank_results.get(R, [])
        if not leaves:
            continue
        unique = set()
        for leaf in leaves:
            unique.add(frozenset(leaf.satisfied))
        console.print(f"[bold {rank_style(R)}]R={R}:[/bold {rank_style(R)}] "
                       f"{len(unique)} unique axiom configurations")
        for i, u in enumerate(sorted(unique, key=lambda x: (-len(x), sorted(x)))):
            if i >= 8:
                console.print(f"    [dim]… +{len(unique) - 8} more[/dim]")
                break
            axioms = ' '.join(f"{AXIOM_GLYPHS.get(a, '·')}{a}" for a in sorted(u))
            console.print(f"    {axioms}")

    # ── Save results ──
    out_name = args.output or f"bfs_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file = Path("outputs") / out_name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_s': total_time,
        'emergency_shutdown': _emergency_shutdown,
        'quit_reason': quit_reason or 'all ranks completed normally',
        'rank_completion_reason': {str(R): rank_completion_reason.get(R, 'completed normally') for R in ranks},
        'worker_failure_notes': {str(R): worker_failure_notes.get(R, []) for R in ranks if worker_failure_notes.get(R)},
        'ranks': ranks,
        'max_states': args.max_states,
        'n_operators': len(OPERATORS),
        'results': {},
    }
    for R in ranks:
        leaves = rank_results.get(R, [])
        unique_sets = set()
        deduped = []
        for l in leaves:
            key = frozenset(l.satisfied)
            if key not in unique_sets:
                unique_sets.add(key)
                deduped.append(l)
        save_data['results'][str(R)] = {
            'total_leaves': len(leaves),
            'unique_configurations': len(unique_sets),
            'explored': explored_counts.get(R, 0),
            'best_axioms': sorted(deduped[0].satisfied) if deduped else [],
            'best_depth': deduped[0].depth if deduped else 0,
            'configurations': [
                {
                    'satisfied': sorted(l.satisfied),
                    'depth': l.depth,
                    'path': l.path,
                    'constraints': l.constraints or {},
                }
                for l in deduped[:50]
            ],
        }

    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {out_file}[/dim]")

    if _emergency_shutdown:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
