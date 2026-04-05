"""
analyze_cosatisfaction.py — Joint Axiom Co-Satisfaction Analysis

"An algebra is not a thing found. It is a thing summoned by choosing which
 laws to obey." — CANON ADE

Given checkpoint data from the axiom engine, this script asks:
  For each rank R, which SUBSETS of axioms are simultaneously satisfiable?
  Where do joint pass rates diverge from marginal independence?
  What combinatorial signatures appear below R=23 that vanish above?

The output is the algebraic landscape — the map of possible universes.

Usage:
  python scripts/analyze_cosatisfaction.py [RUN_NAME]
  python scripts/analyze_cosatisfaction.py  # auto-picks latest run
"""
import sys, json, os, glob
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════
# AXIOM DEFINITIONS — Boolean gates that define algebraic identity
# ═══════════════════════════════════════════════════════════════

# Each axiom is a predicate on the result dict.
# Some are direct boolean keys, others are thresholded continuous values.
AXIOM_PREDICATES = {
    # Core structural (A1-A7)
    'A1:Conservation':   lambda r: r.get('A1_pass', 0) == 1,
    'A1:η+R=27':         lambda r: r.get('A1_conservation_ok', 0) == 1,
    'A2:Reconstruct':    lambda r: r.get('A2_pass', 0) == 1,
    'A2:Fitness<1':      lambda r: (r.get('A2_fitness', 999)) < 1.0,
    'A2:Fitness<0.1':    lambda r: (r.get('A2_fitness', 999)) < 0.1,
    'A3:Σ-Achievable':   lambda r: r.get('A3_achievable', 0) == 1,
    'A4:GenericPos':     lambda r: r.get('A4_generic_position', 0) == 1,
    'A4:MinRank':        lambda r: r.get('A4_minimal_rank', 0) == 1,
    'A5:Gate2':          lambda r: r.get('A5_gate2', 0) == 1,
    'A5:Gate3':          lambda r: r.get('A5_gate3', 0) == 1,
    'A5:BothGates':      lambda r: r.get('A5_gates_compatible', 0) == 1,
    'A6:ParitySep>0':    lambda r: r.get('A6_parity_separation', -1) > 0,
    'A7:Quantized':      lambda r: r.get('A7_quantized', 0) == 1,

    # Classical (these SHOULD fail for novel algebra)
    'CL:Associative':    lambda r: r.get('CL_assoc_pass', 0) == 1,
    'CL:Jacobi':         lambda r: r.get('CL_jacobi_pass', 0) == 1,
    'CL:Semisimple':     lambda r: r.get('CL_semisimple', 0) == 1,
    'CL:Nilpotent':      lambda r: r.get('CL_nilpotent', 0) == 1,

    # Novel diagnostics (continuous → thresholded)
    'NV:LowCoherence':   lambda r: r.get('NV_coherence_max', 1) < 0.5,
    'NV:SpecSymm<10':    lambda r: r.get('NV_spec_symmetry', 999) < 10,
    'NV:FullCouplingCA':  lambda r: r.get('NV_coupling_rank_CA', 0) == 9,
    'NV:ReconErr<1':     lambda r: r.get('NV_recon_err', 999) < 1.0,
}

# Short names for display
AXIOM_SHORT = {k: k for k in AXIOM_PREDICATES}

# Core axiom subset (the algebra-defining ones)
CORE_AXIOMS = [
    'A1:Conservation', 'A2:Fitness<1', 'A3:Σ-Achievable',
    'A4:GenericPos', 'A4:MinRank', 'A5:Gate2', 'A5:Gate3',
    'A6:ParitySep>0', 'A7:Quantized',
]


def load_run(run_name=None):
    """Load all results from a checkpoint run."""
    ckpt_dir = PROJECT_ROOT / 'CANON_ADE' / 'axiom_engine' / 'checkpoints'
    if run_name is None:
        runs = sorted(ckpt_dir.glob('run_*'))
        if not runs:
            console.print("[red]No checkpoint runs found[/red]")
            sys.exit(1)
        run_dir = runs[-1]
        run_name = run_dir.name
    else:
        run_dir = ckpt_dir / run_name

    console.print(f"[cyan]Loading run: {run_name}[/cyan]")
    results = []
    for chunk_file in sorted(run_dir.glob('results_*.json')):
        try:
            with open(chunk_file) as f:
                results.extend(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            console.print(f"  [yellow]Skipping corrupt chunk {chunk_file.name}: {e}[/yellow]")

    # Filter to CPU-evaluated results only (they have all 34 metrics)
    full_results = [r for r in results if r.get('status') == 'ok' and r.get('mode') != 'gpu_batch']
    console.print(f"  Loaded {len(results)} total, {len(full_results)} full CPU evaluations")
    return full_results, run_name


def compute_axiom_vectors(results):
    """For each result, compute boolean vector of which axioms pass."""
    axiom_names = list(AXIOM_PREDICATES.keys())
    vectors = []
    for r in results:
        vec = tuple(1 if AXIOM_PREDICATES[a](r) else 0 for a in axiom_names)
        vectors.append(vec)
    return axiom_names, vectors


def marginal_rates(results, axiom_names, vectors):
    """Marginal pass rate per axiom per rank."""
    by_rank = defaultdict(list)
    for r, v in zip(results, vectors):
        by_rank[r['R']].append(v)

    ranks = sorted(by_rank.keys())
    table = Table(
        title="[bold]Marginal Axiom Pass Rates[/bold]",
        box=box.SIMPLE_HEAVY, show_edge=False,
    )
    table.add_column("Axiom", style="cyan", width=20)
    for rank in ranks:
        table.add_column(f"R={rank}", justify="center", width=9)

    for i, name in enumerate(axiom_names):
        row = [name]
        for rank in ranks:
            vecs = by_rank[rank]
            rate = sum(v[i] for v in vecs) / len(vecs) * 100
            style = "bold green" if rate >= 50 else ("yellow" if rate > 0 else "red")
            row.append(Text(f"{rate:.1f}%", style=style))
        table.add_row(*row)

    console.print(table)
    return by_rank, ranks


def cosatisfaction_pairs(by_rank, ranks, axiom_names):
    """Pairwise co-satisfaction: P(A∧B) vs P(A)·P(B).
    Positive lift = axioms attract. Negative = they repel."""
    core_idx = [axiom_names.index(a) for a in CORE_AXIOMS if a in axiom_names]
    core_names = [axiom_names[i] for i in core_idx]

    for rank in ranks:
        vecs = by_rank[rank]
        n = len(vecs)
        if n < 10:
            continue

        table = Table(
            title=f"[bold]Pairwise Co-Satisfaction Lift — R={rank}[/bold] (n={n})",
            box=box.SIMPLE, show_edge=False, padding=(0, 1),
        )
        table.add_column("A \\ B", style="cyan", width=16)
        for name in core_names:
            table.add_column(name.split(':')[1][:8], justify="center", width=9)

        for i, ni in zip(core_idx, core_names):
            row = [ni]
            pi = sum(v[i] for v in vecs) / n
            for j, nj in zip(core_idx, core_names):
                pj = sum(v[j] for v in vecs) / n
                pij = sum(v[i] and v[j] for v in vecs) / n
                expected = pi * pj
                if expected > 0.001:
                    lift = pij / expected
                    style = "bold green" if lift > 1.5 else ("green" if lift > 1.1 else ("yellow" if lift > 0.9 else "red"))
                    row.append(Text(f"{lift:.2f}", style=style))
                elif pij > 0:
                    row.append(Text("∞", style="bold magenta"))
                else:
                    row.append(Text("—", style="dim"))
            table.add_row(*row)

        console.print(table)


def joint_signatures(by_rank, ranks, axiom_names):
    """Find unique axiom combination signatures and their frequency per rank.
    This is the algebraic fingerprint — each unique signature is a candidate algebra."""
    core_idx = [axiom_names.index(a) for a in CORE_AXIOMS if a in axiom_names]
    core_names = [axiom_names[i] for i in core_idx]

    console.print(Panel("[bold]Joint Axiom Signatures — Candidate Algebras[/bold]\n"
                        "Each unique combination of passing/failing core axioms defines\n"
                        "an algebraic signature. We seek signatures that exist below R=23.",
                        border_style="cyan"))

    # Collect signatures per rank
    all_sigs = defaultdict(lambda: defaultdict(int))  # rank -> sig_tuple -> count
    for rank in ranks:
        for v in by_rank[rank]:
            sig = tuple(v[i] for i in core_idx)
            all_sigs[rank][sig] += 1

    # Find signatures that appear at ANY sub-23 rank
    sub23_sigs = set()
    for rank in ranks:
        if rank < 23:
            for sig in all_sigs[rank]:
                if sum(sig) >= 2:  # at least 2 axioms pass simultaneously
                    sub23_sigs.add(sig)

    if not sub23_sigs:
        console.print("[yellow]No multi-axiom signatures found below R=23[/yellow]")
        # Show top single-axiom signatures instead
        sub23_sigs = set()
        for rank in ranks:
            if rank < 23:
                for sig in all_sigs[rank]:
                    if sum(sig) >= 1:
                        sub23_sigs.add(sig)

    # Sort by total axioms satisfied (descending), then by frequency
    sig_scores = []
    for sig in sub23_sigs:
        total_count = sum(all_sigs[r].get(sig, 0) for r in ranks)
        n_axioms = sum(sig)
        # Does it appear at R=27 control signal?
        at_27 = all_sigs.get(27, {}).get(sig, 0)
        sig_scores.append((n_axioms, total_count, at_27, sig))
    sig_scores.sort(reverse=True)

    # Display top signatures
    table = Table(
        title="[bold]Top Algebraic Signatures (≥2 axioms, present below R=23)[/bold]",
        box=box.ROUNDED, show_edge=True,
    )
    table.add_column("#Ax", justify="center", width=4)
    for name in core_names:
        short = name.split(':')[1][:6]
        table.add_column(short, justify="center", width=7)
    for rank in ranks:
        table.add_column(f"R={rank}", justify="center", width=7)
    table.add_column("Interpretation", width=30)

    for n_ax, total, at27, sig in sig_scores[:30]:
        row = [str(n_ax)]
        # Axiom pattern
        for bit in sig:
            row.append(Text("✓", style="bold green") if bit else Text("✗", style="red"))
        # Counts per rank
        for rank in ranks:
            count = all_sigs[rank].get(sig, 0)
            n_total = sum(all_sigs[rank].values())
            pct = count / n_total * 100 if n_total > 0 else 0
            style = "bold green" if pct > 50 else ("yellow" if pct > 0 else "dim")
            row.append(Text(f"{count}", style=style))
        # Interpretation
        passing = [core_names[i].split(':')[1] for i, b in enumerate(sig) if b]
        failing = [core_names[i].split(':')[1] for i, b in enumerate(sig) if not b]
        interp = f"+{','.join(passing[:4])}"
        if len(failing) <= 3:
            interp += f" −{','.join(failing[:3])}"
        row.append(interp)
        table.add_row(*row)

    console.print(table)
    return all_sigs, core_names


def rank_boundary_analysis(by_rank, ranks, axiom_names):
    """The critical question: what changes at the R=23 boundary?
    Compare axiom co-satisfaction structure above vs below."""
    core_idx = [axiom_names.index(a) for a in CORE_AXIOMS if a in axiom_names]

    console.print(Panel("[bold]Rank Boundary Analysis — What changes at R=23?[/bold]\n"
                        "For each axiom pair, compute correlation below vs above R=23.\n"
                        "Sign flips indicate phase transitions in the algebraic landscape.",
                        border_style="magenta"))

    below = []
    above = []
    for rank in ranks:
        for v in by_rank[rank]:
            core_v = [v[i] for i in core_idx]
            if rank < 23:
                below.append(core_v)
            else:
                above.append(core_v)

    if len(below) < 10 or len(above) < 10:
        console.print("[yellow]Insufficient data for boundary analysis[/yellow]")
        return

    below = np.array(below, dtype=float)
    above = np.array(above, dtype=float)

    core_names = [axiom_names[i] for i in core_idx]
    n = len(core_names)

    # Correlation matrices
    def corr_matrix(data):
        if data.shape[0] < 2:
            return np.zeros((n, n))
        means = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-10] = 1  # avoid div by zero
        centered = (data - means) / std
        return (centered.T @ centered) / data.shape[0]

    corr_below = corr_matrix(below)
    corr_above = corr_matrix(above)
    diff = corr_below - corr_above

    table = Table(
        title="[bold]Correlation Shift: Below R=23 minus Above R=23[/bold]",
        box=box.SIMPLE, show_edge=False,
    )
    table.add_column("", style="cyan", width=16)
    for name in core_names:
        table.add_column(name.split(':')[1][:8], justify="center", width=9)

    for i in range(n):
        row = [core_names[i]]
        for j in range(n):
            d = diff[i, j]
            if abs(d) < 0.01:
                row.append(Text("·", style="dim"))
            else:
                style = "bold green" if d > 0.1 else ("green" if d > 0 else ("red" if d > -0.1 else "bold red"))
                row.append(Text(f"{d:+.2f}", style=style))
        table.add_row(*row)

    console.print(table)

    # Highlight strongest shifts
    shifts = []
    for i in range(n):
        for j in range(i+1, n):
            shifts.append((abs(diff[i,j]), diff[i,j], core_names[i], core_names[j]))
    shifts.sort(reverse=True)

    console.print("\n[bold]Strongest correlation shifts at R=23 boundary:[/bold]")
    for mag, val, a, b in shifts[:10]:
        direction = "[green]ATTRACTS below 23[/green]" if val > 0 else "[red]REPELS below 23[/red]"
        console.print(f"  {a} × {b}: {val:+.3f}  {direction}")


def max_joint_depth(by_rank, ranks, axiom_names):
    """For each rank: what is the maximum number of core axioms
    simultaneously satisfied by any single seed?"""
    core_idx = [axiom_names.index(a) for a in CORE_AXIOMS if a in axiom_names]
    core_names = [axiom_names[i] for i in core_idx]

    console.print(Panel("[bold]Maximum Joint Depth — Most axioms satisfied simultaneously[/bold]",
                        border_style="yellow"))

    table = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    table.add_column("Rank", style="cyan", width=8)
    table.add_column("Max Depth", justify="center", width=10)
    table.add_column("Avg Depth", justify="center", width=10)
    table.add_column("n", justify="center", width=8)
    table.add_column("Best Signature", width=50)

    for rank in ranks:
        vecs = by_rank[rank]
        depths = [sum(v[i] for i in core_idx) for v in vecs]
        max_d = max(depths) if depths else 0
        avg_d = np.mean(depths) if depths else 0

        # Find the best seed's signature
        best_idx = np.argmax(depths)
        best_v = vecs[best_idx]
        best_sig = [core_names[j].split(':')[1] for j, ci in enumerate(core_idx) if best_v[ci]]

        style = "bold green" if max_d >= 5 else ("yellow" if max_d >= 3 else "white")
        table.add_row(
            f"R={rank}",
            Text(str(max_d), style=style),
            f"{avg_d:.1f}",
            str(len(vecs)),
            ", ".join(best_sig) if best_sig else "—",
        )

    console.print(table)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    results, run_name = load_run(run_name)

    if not results:
        console.print("[red]No full CPU evaluations found in this run.[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]ADE3×3 — JOINT AXIOM CO-SATISFACTION ANALYSIS[/bold cyan]\n"
        "\"Every combination of constraints is a universe.\n"
        " Most are barren. Some contain exactly one miracle.\"\n"
        f"\nRun: {run_name}  |  {len(results)} full evaluations",
        border_style="cyan",
    ))

    axiom_names, vectors = compute_axiom_vectors(results)

    # 1. Marginal rates
    by_rank, ranks = marginal_rates(results, axiom_names, vectors)

    # 2. Maximum joint depth per rank
    max_joint_depth(by_rank, ranks, axiom_names)

    # 3. Joint signatures — the candidate algebras
    all_sigs, core_names = joint_signatures(by_rank, ranks, axiom_names)

    # 4. Pairwise co-satisfaction lift
    cosatisfaction_pairs(by_rank, ranks, axiom_names)

    # 5. Rank boundary analysis
    rank_boundary_analysis(by_rank, ranks, axiom_names)

    console.print("\n[bold green]Analysis complete.[/bold green]")


if __name__ == '__main__':
    main()
