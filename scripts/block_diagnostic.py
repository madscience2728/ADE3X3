#!/usr/bin/env python
"""Block energy & interference diagnostic.

Connects the Pythagorean energy identity to the fiber/block structure
of the 3×3 matrix multiplication tensor.

Theory:
  ||T||² = 27  (9 fibers × 3 energy each)
  Per fiber: ||R_c||² + ||T_hat_c||² ≈ 3 + cross-terms
  Leakage = energy deposited in dead entries = wasted interference
  Phase coupling = shared leakage between fiber pairs from same terms

Usage:
    python scripts/block_diagnostic.py
    python scripts/block_diagnostic.py --input shotgun_best.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR
from db_optimizer.tensor import (
    pythagorean_diagnostic, dead_live_energy, block_energy_diagnostic,
    interference_flow, fiber_mode_decomposition,
)


def load_factors(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["alpha"]), np.array(d["beta"]), np.array(d["gamma"])


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    # Find best candidate
    if args.input:
        src = args.input
    else:
        candidates = []
        for p in [Path("shotgun_best.json"), Path("chain_best.json")]:
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                candidates.append((d["fitness"], p))
        if not candidates:
            print("No candidate files found.")
            return
        candidates.sort()
        src = candidates[0][1]

    alpha, beta, gamma = load_factors(src)
    R = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    fit = float(np.max(np.abs(R)))
    print(f"Loaded: {src}  fitness={fit:.10f}")

    # ── 1. Global Pythagorean ──
    print_header("1. GLOBAL PYTHAGOREAN IDENTITY")
    pyth = pythagorean_diagnostic(alpha, beta, gamma)
    dl = dead_live_energy(alpha, beta, gamma)
    print(f"  ||R||²     = {pyth['R_fro2']:10.6f}")
    print(f"  ||T_hat||² = {pyth['T_hat_fro2']:10.6f}")
    print(f"  Sum        = {pyth['sum']:10.6f}  (theory: {pyth['theory']})")
    print(f"  Captured   = {pyth['T_hat_fro2']/pyth['theory']*100:.2f}% of target energy")
    print(f"  Live  residual = {dl['live_energy']:.6f}  (max-abs={dl['live_maxabs']:.6f})")
    print(f"  Dead  residual = {dl['dead_energy']:.6f}  (max-abs={dl['dead_maxabs']:.6f})")
    print(f"  Dead/Total = {dl['dead_energy']/pyth['R_fro2']*100:.1f}%  "
          f"← {'DEAD DOMINATED' if dl['dead_energy'] > dl['live_energy'] else 'live dominated'}")

    # ── 2. Per-Fiber Energy ──
    print_header("2. PER-FIBER ENERGY (Pythagorean per block)")
    bed = block_energy_diagnostic(alpha, beta, gamma)
    fibers = bed["fibers"]

    print(f"\n  {'Fiber':<7} {'Tgt':>5} {'Signal':>8} {'Resid':>8} "
          f"{'Dead↗':>7} {'max|R|':>8} {'live↗':>7} {'dead↗':>7} {'Σγ':>7}")
    print(f"  {'─'*72}")

    worst_fiber = None
    worst_resid = 0
    for key in sorted(fibers.keys()):
        f = fibers[key]
        print(f"  {key:<7} {f['target_energy']:5.1f} {f['signal_energy']:8.4f} "
              f"{f['residual_energy']:8.4f} {f['dead_leakage_energy']:7.4f} "
              f"{f['maxabs']:8.5f} {f['live_maxabs']:7.5f} {f['dead_maxabs']:7.5f} "
              f"{f['sigma_gamma_sum']:7.3f}")
        if f['residual_energy'] > worst_resid:
            worst_resid = f['residual_energy']
            worst_fiber = key

    print(f"\n  Worst fiber: {worst_fiber} (residual energy = {worst_resid:.4f})")
    print(f"  Target per fiber: 3.000 (live) + 0.000 (dead) = 3.000 total")

    # Energy imbalance
    energies = [fibers[k]["signal_energy"] for k in sorted(fibers.keys())]
    print(f"\n  Signal energy spread: min={min(energies):.4f}  max={max(energies):.4f}  "
          f"std={np.std(energies):.4f}")
    print(f"  Perfect would be: all ≈ 3.000, std = 0.000")

    # Dead leakage per fiber
    dead_leaks = [fibers[k]["dead_leakage_energy"] for k in sorted(fibers.keys())]
    print(f"  Dead leakage:      min={min(dead_leaks):.4f}  max={max(dead_leaks):.4f}  "
          f"total={sum(dead_leaks):.4f}")

    # ── 3. Leakage Flow Matrix ──
    print_header("3. CROSS-FIBER LEAKAGE MATRIX")
    labels = bed["leakage_matrix_labels"]
    L = bed["leakage_matrix"]
    print(f"\n  L[c1,c2] = energy in dead(c1) that overlaps live(c2)")
    print(f"  {'':>8}", end="")
    for lab in labels:
        print(f"  {lab:>7}", end="")
    print()
    for i, lab in enumerate(labels):
        print(f"  {lab:>7}", end="")
        for j in range(len(labels)):
            val = L[i, j]
            if val < 1e-6:
                print(f"  {'·':>7}", end="")
            else:
                print(f"  {val:7.4f}", end="")
        print()

    # ── 4. Interference Flow ──
    print_header("4. PER-TERM INTERFERENCE FLOW")
    iflow = interference_flow(alpha, beta, gamma)

    print(f"\n  {'Term':>4} {'Total_E':>8} {'Signal':>8} {'Leak':>8} "
          f"{'Effic':>6} {'Top Fiber':>10} {'Top Leak Fiber':>15}")
    print(f"  {'─'*68}")

    for k in range(RANK):
        tot = iflow['total_per_term'][k]
        sig = float(iflow['signal'][k].sum())
        lk = float(iflow['leak'][k].sum())
        eff = iflow['efficiency'][k]
        top_sig = int(np.argmax(iflow['signal'][k]))
        top_leak = int(np.argmax(iflow['leak'][k]))
        marker = " ← SPRAY" if eff < 0.5 else ""
        print(f"  {k:4d} {tot:8.4f} {sig:8.4f} {lk:8.4f} "
              f"{eff:6.1%} {iflow['labels'][top_sig]:>10} "
              f"{iflow['labels'][top_leak]:>15}{marker}")

    n_spray = len(iflow['spray_terms'])
    n_focused = len(iflow['focused_terms'])
    print(f"\n  Focused terms: {n_focused}/{RANK}  |  Spray terms: {n_spray}/{RANK}")
    print(f"  Spray terms: {iflow['spray_terms']}")

    # ── 5. Phase Coupling ──
    print_header("5. PHASE COUPLING MATRIX")
    C = iflow['coupling_matrix']
    print(f"\n  C[c1,c2] = Σ_k leak[k,c1]·leak[k,c2]  (shared interference source)")
    print(f"  {'':>8}", end="")
    for lab in iflow['labels']:
        print(f"  {lab:>7}", end="")
    print()
    for i, lab in enumerate(iflow['labels']):
        print(f"  {lab:>7}", end="")
        for j in range(len(iflow['labels'])):
            val = C[i, j]
            if i == j:
                print(f"  {'─':>7}", end="")
            elif val < 1e-4:
                print(f"  {'·':>7}", end="")
            else:
                print(f"  {val:7.4f}", end="")
        print()

    # Find strongest coupling pairs
    pairs = []
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append((C[i,j], iflow['labels'][i], iflow['labels'][j]))
    pairs.sort(reverse=True)
    print(f"\n  Top-5 phase-coupled fiber pairs:")
    for val, l1, l2 in pairs[:5]:
        print(f"    {l1} ↔ {l2}: {val:.4f}")

    # ── 6. Structural interpretation ──
    print_header("6. STRUCTURAL INTERPRETATION")

    # Which fibers have more dead leakage than live residual?
    print(f"\n  Fiber         dead_resid  live_resid  dead>live?  Interpretation")
    print(f"  {'─'*65}")
    for key in sorted(fibers.keys()):
        f = fibers[key]
        dom = f['dead_residual_energy'] > f['live_residual_energy']
        interp = "dead-dominated → need interference cleanup" if dom else "live-dominated → need more signal"
        print(f"  {key:<12} {f['dead_residual_energy']:10.5f} {f['live_residual_energy']:10.5f}  "
              f"{'YES':>10}  {interp}" if dom else
              f"  {key:<12} {f['dead_residual_energy']:10.5f} {f['live_residual_energy']:10.5f}  "
              f"{'no':>10}  {interp}")

    # Energy budget
    total_dead_leak = sum(fibers[k]['dead_leakage_energy'] for k in fibers)
    total_resid_e = pyth['R_fro2']
    print(f"\n  Total dead leakage in T_hat: {total_dead_leak:.4f}")
    print(f"  Total ||R||²:                {total_resid_e:.4f}")
    print(f"  If we could redirect all dead leakage → live signal:")
    print(f"    Potential gain = {total_dead_leak:.4f} ({total_dead_leak/total_resid_e*100:.1f}% of residual)")

    # Fiber-mode decomposition
    fm = fiber_mode_decomposition(alpha, beta, gamma)
    print(f"\n  Fiber-mode ranks:")
    print(f"    Sigma rank:    {fm['sigma_rank']} / {RANK}")
    print(f"    Nuisance rank: {fm['nuisance_rank']} / {RANK}")
    print(f"    Gamma rank:    {fm['gamma_rank']} / {RANK}")
    print(f"    Gamma nullity: {fm['gamma_nullity']} (budget for nuisance annihilation)")
    print(f"    ΓΣ residual:   {fm['gs_residual']:.6f} (target: 0 → ΓΣ = 3I)")


if __name__ == "__main__":
    main()
