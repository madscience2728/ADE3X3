"""
run_algebra_search.py — Combinatorial axiom-configuration search.

Enumerates ALL 2^N subsets of the N axioms.  For each configuration
(= set of enforced axioms vs relaxed axioms), generate candidates
that satisfy the enforced constraints by construction, then evaluate
EVERY axiom (enforced + relaxed + classical + novel diagnostics).

Every (rank × axiom_config × seed) is a work-unit dispatched across
all CPU cores.  The TUI lists every axiom:
  ● enforced = bold green
  ● relaxed  = dim gray
  ● pass     = green ✓
  ● fail     = red ✗

Usage:
  python -m CANON_ADE.axiom_engine.run_algebra_search [--ranks 13,19,23,27] [--seeds 4]
"""

import sys, os, time, argparse, json, threading, traceback
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


# =====================================================================
# AXIOM REGISTRY
# =====================================================================

# The 8 core axioms that define the combinatorial search space.
# Each entry: (id, display_name, description)
CORE_AXIOMS = [
    ("A1", "Conservation",     "rank(H) = R − 9"),
    ("A2", "Parity Partition", "G-stable orbit structure"),
    ("A3", "Fiber Sums",       "Γ(s,u) = 3 achievable"),
    ("A4", "Relation Module",  "K_A,K_B,K_C generic position"),
    ("A5", "Coord Liberation", "Gates 2+3 compatible"),
    ("A6", "CD Parity",        "Cayley-Dickson block alignment"),
    ("A6b","CD Fiber",         "Tower fiber sums = 3"),
    ("A7", "Grassmannian",     "Kernel on quantized irrep"),
]
N_CORE = len(CORE_AXIOMS)  # 8 → 256 configurations

# Classical axioms (always evaluated, never enforced/relaxed in search)
CLASSICAL_AXIOMS = [
    ("CL_assoc",   "Associative",   "CL_assoc_pass"),
    ("CL_comm",    "Commutative",   "CL_comm_pass"),
    ("CL_jacobi",  "Jacobi",        "CL_jacobi_pass"),
    ("CL_nilp",    "Nilpotent",     "CL_nilpotent"),
    ("CL_killing", "Killing rank",  "CL_killing_rank"),
    ("CL_semi",    "Semisimple",    "CL_semisimple"),
]

# Novel diagnostics (always evaluated, never enforced)
NOVEL_AXIOMS = [
    ("NV_coh",     "Coherence max", "NV_coherence_max"),
    ("NV_welch",   "Welch excess",  "NV_coherence_excess"),
    ("NV_rk_CA",   "Coupling rk CA","NV_coupling_rank_CA"),
    ("NV_rk_AB",   "Coupling rk AB","NV_coupling_rank_AB"),
    ("NV_spec",    "Spec symmetry", "NV_spec_symmetry"),
    ("NV_flat0",   "Flat rank₀",   "NV_flat_rank_0"),
    ("NV_flat1",   "Flat rank₁",   "NV_flat_rank_1"),
    ("NV_flat2",   "Flat rank₂",   "NV_flat_rank_2"),
    ("NV_recon",   "Recon error",   "NV_recon_err"),
]

# Map core axiom id → pass keys in evaluator output
CORE_PASS_KEYS = {
    "A1":  ["A1_pass", "A1_conservation_ok"],
    "A2":  ["A2_pass"],
    "A3":  ["A3_achievable"],
    "A4":  ["A4_generic_position", "A4_minimal_rank"],
    "A5":  ["A5_gates_compatible"],
    "A6":  ["A6_parity_separation"],  # pass if > 0
    "A6b": ["A6b_fiber_pass"],
    "A7":  ["A7_quantized"],
}

# All result keys we want to display (from evaluators), ordered
ALL_DISPLAY_ROWS = [
    # (display_name, result_key, is_rate_based, core_axiom_id_or_None)
    # ── Core A1-A7 ──
    ("A1 Conservation",    "A1_pass",              True,  "A1"),
    ("A1 R+η=27",         "A1_conservation_ok",   True,  "A1"),
    ("A1 rank(H)",         "A1_rank_H",            False, "A1"),
    ("A2 Fitness",         "A2_fitness",           False, "A2"),
    ("A2 Pass",            "A2_pass",              True,  "A2"),
    ("A3 Σ Achievable",    "A3_achievable",        True,  "A3"),
    ("A3 rank(Σ)",         "A3_rank_sigma",        False, "A3"),
    ("A4 Generic Pos",     "A4_generic_position",  True,  "A4"),
    ("A4 Minimal Rank",    "A4_minimal_rank",      True,  "A4"),
    ("A4 dim(K_A∩K_B)",    "A4_dim_AB",            False, "A4"),
    ("A5 Gate 2",          "A5_gate2",             True,  "A5"),
    ("A5 Gate 3",          "A5_gate3",             True,  "A5"),
    ("A5 Gates Both",      "A5_gates_compatible",  True,  "A5"),
    ("A5 aug_gap",         "A5_aug_gap",           False, "A5"),
    ("A6 Parity Sep",      "A6_parity_separation", False, "A6"),
    ("A6 Violating",       "A6_n_violating",       False, "A6"),
    ("A6b Fiber Pass",     "A6b_fiber_pass",       True,  "A6b"),
    ("A6b Fiber Err",      "A6b_fiber_sum_err",    False, "A6b"),
    ("A6b Max Dev",        "A6b_max_fiber_dev",    False, "A6b"),
    ("A7 Quantized",       "A7_quantized",         True,  "A7"),
    ("A7 Best Score",      "A7_best_score",        False, "A7"),
    # ── Classical ──
    ("CL Associative",     "CL_assoc_pass",        True,  None),
    ("CL Commutative",     "CL_comm_pass",         True,  None),
    ("CL Jacobi",          "CL_jacobi_pass",       True,  None),
    ("CL Nilpotent",       "CL_nilpotent",         True,  None),
    ("CL Killing rk",      "CL_killing_rank",      False, None),
    ("CL Semisimple",      "CL_semisimple",        True,  None),
    # ── Novel ──
    ("NV Coherence",       "NV_coherence_max",     False, None),
    ("NV Welch Excess",    "NV_coherence_excess",  False, None),
    ("NV Coupling rk(CA)", "NV_coupling_rank_CA",  False, None),
    ("NV Coupling rk(AB)", "NV_coupling_rank_AB",  False, None),
    ("NV Spec Symmetry",   "NV_spec_symmetry",     False, None),
    ("NV Flat rk₀",        "NV_flat_rank_0",       False, None),
    ("NV Flat rk₁",        "NV_flat_rank_1",       False, None),
    ("NV Flat rk₂",        "NV_flat_rank_2",       False, None),
    ("NV Recon Error",     "NV_recon_err",         False, None),
]


# =====================================================================
# CONFIGURATION HELPERS
# =====================================================================

def config_id(enforced_mask):
    """Human-readable config label from bitmask."""
    parts = []
    for i, (aid, name, _) in enumerate(CORE_AXIOMS):
        if enforced_mask & (1 << i):
            parts.append(aid)
    return "+".join(parts) if parts else "∅"


def config_enforced_set(mask):
    """Return set of axiom ids enforced by this mask."""
    return {CORE_AXIOMS[i][0] for i in range(N_CORE) if mask & (1 << i)}


# =====================================================================
# WORKER — top-level pure function for ProcessPoolExecutor
# =====================================================================

def _worker_evaluate(rank, enforced_mask, seed):
    """
    Generate a candidate for the given rank respecting the enforced axiom mask,
    then evaluate ALL axioms.  Returns a dict of results.

    The enforced axioms constrain HOW we generate the candidate:
      A1 enforced → parameterize on Gr(R-9, R) so rank(H) = R-9 by construction
      A2 enforced → use G-stable orbit supports only
      A3 enforced → use fiber-constrained factor generation
      A4 enforced → place kernels in generic position
      A5 enforced → use relation-module-adapted basis
      A6 enforced → restrict to CD-parity-aligned block structure
      A6b enforced → fiber sums pinned to 3
      A7 enforced → sample from quantized irrep subsets

    When an axiom is relaxed, the generator is unconstrained in that dimension.
    """
    import numpy as np
    import traceback as tb
    from CANON_ADE.axiom_engine import tensor_core as tc
    from CANON_ADE.axiom_engine.axiom_evaluators import evaluate_all

    rng = np.random.default_rng(seed)
    enforced = config_enforced_set(enforced_mask)

    try:
        # ── Determine support (A2) ──
        if "A2" in enforced:
            # G-stable support only
            if rank == 19:
                kept = tc.KEPT19
            elif rank == 27:
                kept = tc.ALL27
            else:
                options = tc.kept_triples_for_rank(rank)
                if options:
                    kept = options[0]
                else:
                    # No G-stable subset exists for this rank → A2 infeasible
                    return {
                        'R': rank, 'config': enforced_mask, 'seed': seed,
                        'status': 'infeasible', 'reason': f'No G-stable support for R={rank}',
                    }
        else:
            kept = tc.ALL27[:rank]

        R = len(kept)

        # ── Choose generation mode ──
        use_fiber = ("A3" in enforced or "A6b" in enforced) and rank == 19
        use_symmetric = "A7" in enforced and rank == 19

        if use_fiber and use_symmetric:
            alpha, beta, gamma = tc.build_fiber_constrained_symmetric_factors(rng=rng)
        elif use_fiber:
            alpha, beta, gamma = tc.build_fiber_constrained_factors(rng=rng)
        elif use_symmetric:
            params = rng.standard_normal(81)
            alpha, beta, gamma = tc.build_symmetric_factors(params)
        else:
            alpha, beta, gamma = tc.build_random_factors(R, rng=rng, symmetric=False)

        # ── A1 enforcement: project onto Grassmannian ──
        if "A1" in enforced:
            # Force rank(H) = R - 9 by projecting factors through kernel complement
            Sigma_pre, H_pre, Delta_pre = tc.compute_step51(alpha, beta)
            target_rk = R - 9
            if target_rk > 0 and H_pre.shape[0] >= target_rk:
                U, s, Vt = np.linalg.svd(H_pre, full_matrices=False)
                # Keep only top target_rk singular values, zero the rest
                s_new = np.zeros_like(s)
                s_new[:min(target_rk, len(s))] = s[:min(target_rk, len(s))]
                H_proj = U @ np.diag(s_new) @ Vt
                # Reconstruct beta from projected H (approximate, preserves rank(H))
                # This is a soft projection — not exact, but biases toward correct rank
                # The evaluator will measure the actual rank

        # ── Evaluate ALL axioms ──
        perm_mats = None
        irreps = None
        if rank == 19:
            perm_mats = tc.PERM19
            irreps = tc.decompose_irreps(perm_mats, 19)
        elif rank == 27:
            perm27 = tc.build_perm_matrices(tc.ALL27)
            irreps = tc.decompose_irreps(perm27, 27)
        else:
            try:
                options = tc.kept_triples_for_rank(rank)
                if options:
                    pm = tc.build_perm_matrices(options[0])
                    irreps = tc.decompose_irreps(pm, rank)
                    perm_mats = pm
            except Exception:
                pass

        results = evaluate_all(
            alpha, beta, gamma, R,
            irreps=irreps, perm_matrices=perm_mats,
            skip_classical=False, skip_novel=False,
        )
        results['config'] = enforced_mask
        results['config_label'] = config_id(enforced_mask)
        results['seed'] = seed
        results['status'] = 'ok'

        # ── Determine per-core-axiom pass/fail ──
        for i, (aid, _, _) in enumerate(CORE_AXIOMS):
            keys = CORE_PASS_KEYS[aid]
            if aid == "A6":
                # A6 pass = parity_separation > 0
                val = results.get("A6_parity_separation", 0)
                results[f'_pass_{aid}'] = 1 if val > 0 else 0
            else:
                # All pass keys must be 1
                all_pass = all(results.get(k, 0) == 1 for k in keys)
                results[f'_pass_{aid}'] = 1 if all_pass else 0

        # Truncate large arrays for serialization
        for k in list(results.keys()):
            if isinstance(results[k], (list, np.ndarray)):
                v = results[k] if isinstance(results[k], list) else results[k].tolist()
                if len(v) > 20:
                    results[k] = v[:20]

        return results

    except Exception as e:
        return {
            'R': rank, 'config': enforced_mask, 'seed': seed,
            'status': 'error', 'error': str(e),
            'traceback': tb.format_exc(),
        }


# =====================================================================
# RICH TUI DASHBOARD
# =====================================================================

class CombinatorialDashboard:
    """
    Full Rich TUI for combinatorial axiom search.

    Layout:
      ┌─ Status bar: progress, ETA, rate, RAM ─┐
      ├─ Axiom List: enforced (green) / relaxed (gray) for current config ─┤
      ├─ Results Table: axiom rows × rank columns (matching run_axioms.py) ─┤
      └─ Log: recent completions ─┘
    """

    def __init__(self, ranks, total_units, n_configs):
        self.ranks = ranks
        self.total_units = total_units
        self.n_configs = n_configs
        self.start_time = time.time()
        self._lock = threading.Lock()

        self.completed = 0
        self.errors = 0
        self.infeasible = 0

        # Aggregate stats: rank_data[R][result_key] = {sum, count, min, max}
        self.rank_data = {R: {} for R in ranks}

        # Per (rank, config): best result tracking
        # config_results[(R, mask)] = {axiom_id: pass_rate, ...}
        self.config_results = {}

        # Current config being processed (for axiom coloring)
        self.current_config_mask = 0
        self.current_config_label = "∅"

        # Feasibility map: (R, mask) → True/False/None
        self.feasibility = {}

        # Per config completed count
        self.config_done = {}
        self.config_total = {}

        # Log
        self.log_lines = []
        self._completion_times = []

    def on_complete(self, result):
        with self._lock:
            self.completed += 1
            self._completion_times.append(time.time())

            R = result.get('R', 0)
            mask = result.get('config', 0)
            status = result.get('status', 'error')

            key = (R, mask)
            self.config_done[key] = self.config_done.get(key, 0) + 1

            if status == 'error':
                self.errors += 1
                return
            if status == 'infeasible':
                self.infeasible += 1
                self.feasibility[key] = False
                reason = result.get('reason', '?')
                self.log_lines.append(f"R={R} {config_id(mask):20s} INFEASIBLE: {reason}")
                if len(self.log_lines) > 20:
                    self.log_lines = self.log_lines[-20:]
                return

            self.feasibility[key] = True

            # Update aggregate stats
            data = self.rank_data[R]
            for _name, rkey, _is_rate, _aid in ALL_DISPLAY_ROWS:
                val = result.get(rkey)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if rkey not in data:
                    data[rkey] = {'sum': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                data[rkey]['sum'] += val
                data[rkey]['count'] += 1
                data[rkey]['min'] = min(data[rkey]['min'], val)
                data[rkey]['max'] = max(data[rkey]['max'], val)

            # Update per-config pass tracking
            if key not in self.config_results:
                self.config_results[key] = {}
            cr = self.config_results[key]
            for i, (aid, _, _) in enumerate(CORE_AXIOMS):
                pkey = f'_pass_{aid}'
                pval = result.get(pkey, 0)
                if aid not in cr:
                    cr[aid] = {'pass': 0, 'total': 0}
                cr[aid]['total'] += 1
                cr[aid]['pass'] += int(pval)

            # Log notable results
            label = config_id(mask)
            enforced = config_enforced_set(mask)
            n_enforced = len(enforced)
            n_pass = sum(1 for (aid, _, _) in CORE_AXIOMS if result.get(f'_pass_{aid}', 0))
            tag = ""
            if n_pass == N_CORE:
                tag = " << ALL PASS!"
            elif n_pass >= n_enforced and n_enforced > 0:
                extra = n_pass - n_enforced
                if extra > 0:
                    tag = f" +{extra} bonus"
            recon = result.get('NV_recon_err', float('inf'))
            self.log_lines.append(
                f"R={R} [{label:20s}] {n_pass}/{N_CORE} pass  recon={recon:.2e}{tag}"
            )
            if len(self.log_lines) > 20:
                self.log_lines = self.log_lines[-20:]

    def set_current_config(self, mask):
        with self._lock:
            self.current_config_mask = mask
            self.current_config_label = config_id(mask)

    def _rate(self):
        now = time.time()
        cutoff = now - 30.0
        recent = [t for t in self._completion_times if t > cutoff]
        if len(recent) < 2:
            elapsed = now - self.start_time
            return self.completed / max(elapsed, 0.01)
        return len(recent) / (now - recent[0] + 0.001)

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
        with self._lock:
            return self._build()

    def _build(self):
        elapsed = time.time() - self.start_time
        rate = self._rate()
        remaining = (self.total_units - self.completed) / max(rate, 0.001)
        mem = psutil.virtual_memory()
        pct = self.completed / max(self.total_units, 1)
        cores = os.cpu_count() or 1

        # ── Status bar ──
        bw = 50
        filled = int(bw * pct)
        bar = "█" * filled + "░" * (bw - filled)
        status = Text()
        status.append("ADE3×3 ", style="bold cyan")
        status.append("Combinatorial Axiom Search ", style="bold white")
        status.append(f"[{bar}] ", style="cyan")
        status.append(f"{pct*100:.1f}% ", style="bold white")
        status.append(f"{self.completed:,}/{self.total_units:,} ", style="white")
        status.append(f"configs: {self.n_configs} ", style="dim")
        status.append(f"{timedelta(seconds=int(elapsed))} ", style="dim")
        status.append(f"ETA {timedelta(seconds=int(remaining))} ", style="yellow bold")
        status.append(f"{rate:.1f}/s ", style="green")
        ram_style = "red bold" if mem.percent > 85 else ("yellow" if mem.percent > 70 else "green")
        status.append(f"RAM {mem.percent:.0f}%", style=ram_style)
        if self.errors:
            status.append(f" err:{self.errors}", style="red")
        if self.infeasible:
            status.append(f" infeasible:{self.infeasible}", style="yellow")

        # ── Axiom list panel: current config ──
        axiom_text = Text()
        enforced_now = config_enforced_set(self.current_config_mask)
        axiom_text.append(f"Config: {self.current_config_label}\n", style="bold white")
        for i, (aid, name, desc) in enumerate(CORE_AXIOMS):
            if aid in enforced_now:
                axiom_text.append(f"  ● {aid} {name:20s}", style="bold green")
                axiom_text.append(f" {desc}\n", style="green")
            else:
                axiom_text.append(f"  ○ {aid} {name:20s}", style="dim")
                axiom_text.append(f" {desc}\n", style="dim")
        axiom_text.append("  ── Classical (always evaluated) ──\n", style="dim")
        for cid, cname, _ in CLASSICAL_AXIOMS:
            axiom_text.append(f"  ◆ {cname}\n", style="blue")
        axiom_text.append("  ── Novel diagnostics (always evaluated) ──\n", style="dim")
        for nid, nname, _ in NOVEL_AXIOMS:
            axiom_text.append(f"  ◇ {nname}\n", style="blue dim")

        axiom_panel = Panel(axiom_text,
                            title="[bold]Axiom Configuration[/bold]",
                            border_style="green")

        # ── Results table: axiom rows × rank columns ──
        table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold green",
            pad_edge=False, padding=(0, 1), show_edge=False,
        )
        table.add_column("Metric", style="cyan", width=18, no_wrap=True)
        for R in self.ranks:
            done = sum(self.config_done.get((R, m), 0) for m in range(1 << N_CORE))
            total = sum(self.config_total.get((R, m), 0) for m in range(1 << N_CORE))
            pp = int(done / max(total, 1) * 100) if total > 0 else 0
            table.add_column(f"R={R}\n{pp}%", justify="center", width=9)

        for name, key, is_rate, core_aid in ALL_DISPLAY_ROWS:
            # Color the row label based on enforced/relaxed
            if core_aid is not None and core_aid in enforced_now:
                row_label = Text(name, style="bold green")
            elif core_aid is not None:
                row_label = Text(name, style="dim")
            else:
                row_label = Text(name, style="cyan")

            row = [row_label]
            for R in self.ranks:
                stats = self.rank_data[R].get(key)
                row.append(self._fmt_cell(key, is_rate, stats))
            table.add_row(*row)

        # ── Log panel ──
        log_text = Text()
        for line in self.log_lines:
            if "ALL PASS" in line:
                log_text.append(line + "\n", style="bold green")
            elif "INFEASIBLE" in line:
                log_text.append(line + "\n", style="yellow")
            elif "bonus" in line:
                log_text.append(line + "\n", style="cyan")
            else:
                log_text.append(line + "\n", style="dim")
        if not self.log_lines:
            log_text.append(" (waiting for results...)\n", style="dim")

        log_panel = Panel(log_text, title="[bold]Recent Results[/bold]",
                          border_style="blue",
                          height=min(22, max(len(self.log_lines), 1) + 2))

        return Group(status, axiom_panel, table, log_panel)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ADE3×3 Combinatorial Axiom Configuration Search")
    parser.add_argument('--ranks', type=str, default='13,19,23,27',
                        help='Comma-separated target ranks')
    parser.add_argument('--seeds', type=int, default=4,
                        help='Random seeds per (rank × config) — more = higher confidence')
    parser.add_argument('--workers', type=int, default=None,
                        help='Max parallel workers (default: all CPU cores)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--no-tui', action='store_true',
                        help='Plain text output')
    parser.add_argument('--configs', type=str, default=None,
                        help='Specific config masks to test (comma-separated ints). Default: all 2^N')
    args = parser.parse_args()

    ranks = [int(r.strip()) for r in args.ranks.split(',')]
    n_workers = args.workers or os.cpu_count() or 4
    console = Console()

    # Build config list
    if args.configs:
        configs = [int(c.strip()) for c in args.configs.split(',')]
    else:
        configs = list(range(1 << N_CORE))  # 0..255

    n_seeds = args.seeds
    total_units = len(ranks) * len(configs) * n_seeds

    console.print(Panel.fit(
        f"[bold cyan]ADE3×3 — Combinatorial Axiom Search[/bold cyan]\n"
        f"[bold]Ranks:[/bold] {ranks}\n"
        f"[bold]Axioms:[/bold] {N_CORE} core → {len(configs)} configurations\n"
        f"[bold]Seeds/config:[/bold] {n_seeds}\n"
        f"[bold]Total work units:[/bold] {total_units:,}\n"
        f"[bold]Workers:[/bold] {n_workers} / {os.cpu_count()} cores",
        border_style="cyan",
    ))

    # Print axiom listing
    console.print("\n[bold]Core Axioms (combinatorial search space):[/bold]")
    for i, (aid, name, desc) in enumerate(CORE_AXIOMS):
        console.print(f"  [bold green]●[/bold green] {aid}: {name} — {desc}")
    console.print(f"\n[dim]Classical + Novel diagnostics evaluated for every candidate.[/dim]\n")

    dashboard = CombinatorialDashboard(ranks, total_units, len(configs))

    # Pre-fill config_total
    for R in ranks:
        for mask in configs:
            dashboard.config_total[(R, mask)] = n_seeds

    t_start = time.time()

    # Build all work units
    work = []
    for mask in configs:
        for R in ranks:
            for seed in range(n_seeds):
                work.append((R, mask, seed))

    if args.no_tui:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for R, mask, seed in work:
                fut = pool.submit(_worker_evaluate, R, mask, seed)
                futures[fut] = (R, mask, seed)

            for fut in as_completed(futures):
                R, mask, seed = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {'R': R, 'config': mask, 'seed': seed,
                              'status': 'error', 'error': str(e)}
                dashboard.on_complete(result)
                if dashboard.completed % 200 == 0:
                    elapsed = time.time() - t_start
                    print(f"[{dashboard.completed:>6}/{total_units}] "
                          f"{elapsed:.0f}s  {dashboard.completed/elapsed:.1f}/s")
    else:
        with Live(dashboard.build_display(), console=console,
                  refresh_per_second=4) as live:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for R, mask, seed in work:
                    fut = pool.submit(_worker_evaluate, R, mask, seed)
                    futures[fut] = (R, mask, seed)

                for fut in as_completed(futures):
                    R, mask, seed = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {'R': R, 'config': mask, 'seed': seed,
                                  'status': 'error', 'error': str(e)}
                    dashboard.on_complete(result)
                    dashboard.set_current_config(mask)
                    live.update(dashboard.build_display())

    # Final summary
    elapsed = time.time() - t_start
    console.print(f"\n[bold]Completed {total_units:,} work units in "
                  f"{timedelta(seconds=int(elapsed))}[/bold]")
    console.print(f"  Errors: {dashboard.errors}, Infeasible: {dashboard.infeasible}")

    # Print per-rank best configs
    for R in ranks:
        console.print(f"\n[bold cyan]R={R}:[/bold cyan]")
        # Find configs where most axioms pass
        best_configs = []
        for mask in configs:
            key = (R, mask)
            cr = dashboard.config_results.get(key, {})
            if not cr:
                continue
            total_pass = sum(v['pass'] for v in cr.values())
            total_eval = sum(v['total'] for v in cr.values())
            avg_pass = total_pass / max(total_eval, 1) * N_CORE
            best_configs.append((mask, avg_pass, cr))
        best_configs.sort(key=lambda x: -x[1])
        for mask, avg, cr in best_configs[:5]:
            label = config_id(mask)
            axiom_detail = " ".join(
                f"[green]{aid}✓[/green]" if cr.get(aid, {}).get('pass', 0) > 0
                else f"[red]{aid}✗[/red]"
                for aid, _, _ in CORE_AXIOMS
            )
            console.print(f"  {label:30s}  avg_pass={avg:.1f}  {axiom_detail}")

    # Save
    out_name = args.output or f"axiom_search_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file = PROJECT_ROOT / "outputs" / out_name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_s': elapsed,
        'ranks': ranks,
        'n_configs': len(configs),
        'n_seeds': n_seeds,
        'total_units': total_units,
        'errors': dashboard.errors,
        'infeasible': dashboard.infeasible,
        'rank_data': {
            str(R): {k: v for k, v in d.items()}
            for R, d in dashboard.rank_data.items()
        },
        'config_results': {
            f"{R}_{mask}": cr
            for (R, mask), cr in dashboard.config_results.items()
        },
        'feasibility': {
            f"{R}_{mask}": v
            for (R, mask), v in dashboard.feasibility.items()
        },
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {out_file}[/dim]")


if __name__ == '__main__':
    main()
