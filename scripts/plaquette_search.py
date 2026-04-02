#!/usr/bin/env python
"""Systematic plaquette search: find optimal coupling patterns between
the 9 output-fiber interfaces of the 3×3 matrix multiplication tensor.

Theory (running canon): The conjugate phase squeeze rotates nuisance
interference between f+/f- channels while preserving signal. Applied
to the right combination of interfaces with the right signs, it creates
a landscape where the least-squares γ refit achieves lower residual.

Searches all subsets of size 1–9, all sign patterns, ε grid.
Reports best coupling patterns per size and overall.

Usage:
    python scripts/plaquette_search.py
    python scripts/plaquette_search.py --input shotgun_best.json --refine
    python scripts/plaquette_search.py --max-size 6 --refine
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db_optimizer.config import RANK, DIM, TARGET_TENSOR

# ── Real orthonormal basis (p0, p1, p2) ──────────────────────────
_p0 = np.array([1, 1, 1]) / np.sqrt(3)
_p1 = np.array([1, -1, 0]) / np.sqrt(2)
_p2 = np.array([1, 1, -2]) / np.sqrt(6)
P_BASIS = np.vstack([_p0, _p1, _p2])  # (3,3): rows = basis vectors
P_INV = P_BASIS.T  # orthogonal → inverse = transpose

# 9 output interfaces: C[i,j] = Σ_k A[i,k]B[k,j]
INTERFACES = [(i, j) for i in range(3) for j in range(3)]


def load_factors(path: Path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["alpha"]), np.array(d["beta"]), np.array(d["gamma"])


def fitness(a, b, g):
    R = np.einsum('ra,rb,rc->abc', a, b, g, optimize=True) - TARGET_TENSOR
    return float(np.max(np.abs(R)))


def refit_gamma(alpha, beta):
    """Optimal γ given α, β via per-column least squares."""
    M = np.einsum('ra,rb->abr', alpha, beta).reshape(-1, RANK)  # (81, R)
    gamma = np.zeros((RANK, DIM))
    for c in range(DIM):
        gamma[:, c], _, _, _ = np.linalg.lstsq(M, TARGET_TENSOR[:, :, c].ravel(), rcond=None)
    return gamma


def refit_alpha(beta, gamma):
    """Optimal α given β, γ via per-column least squares."""
    M = np.einsum('rb,rc->bcr', beta, gamma).reshape(-1, RANK)
    alpha = np.zeros((RANK, DIM))
    for a in range(DIM):
        alpha[:, a], _, _, _ = np.linalg.lstsq(M, TARGET_TENSOR[a, :, :].ravel(), rcond=None)
    return alpha


def refit_beta(alpha, gamma):
    """Optimal β given α, γ via per-column least squares."""
    M = np.einsum('ra,rc->acr', alpha, gamma).reshape(-1, RANK)
    beta = np.zeros((RANK, DIM))
    for b in range(DIM):
        beta[:, b], _, _, _ = np.linalg.lstsq(M, TARGET_TENSOR[:, b, :].ravel(), rcond=None)
    return beta


def apply_squeeze(alpha, beta, iface_indices, signs, eps):
    """Conjugate phase squeeze on (α, β) at given interfaces.

    For interface (row_i, col_j) with sign s:
      α row-i:  p1 *= (1 + s·ε),  p2 *= (1 - s·ε)
      β col-j:  p1 *= (1 - s·ε),  p2 *= (1 + s·ε)
    Accumulates multiplicatively when interfaces share rows/cols.
    Returns squeezed (α, β) + refitted γ.
    """
    a_p1 = np.ones(3)
    a_p2 = np.ones(3)
    b_p1 = np.ones(3)
    b_p2 = np.ones(3)

    for idx, s in zip(iface_indices, signs):
        ri, cj = INTERFACES[idx]
        a_p1[ri] *= (1 + s * eps)
        a_p2[ri] *= (1 - s * eps)
        b_p1[cj] *= (1 - s * eps)
        b_p2[cj] *= (1 + s * eps)

    a = alpha.copy()
    for i in range(3):
        if a_p1[i] == 1.0 and a_p2[i] == 1.0:
            continue
        coords = a[:, 3*i:3*i+3] @ P_INV          # (R,3) in p0/p1/p2
        coords[:, 1] *= a_p1[i]
        coords[:, 2] *= a_p2[i]
        a[:, 3*i:3*i+3] = coords @ P_BASIS

    b = beta.copy()
    for j in range(3):
        if b_p1[j] == 1.0 and b_p2[j] == 1.0:
            continue
        coords = b[:, j::3] @ P_INV
        coords[:, 1] *= b_p1[j]
        coords[:, 2] *= b_p2[j]
        b[:, j::3] = coords @ P_BASIS

    g = refit_gamma(a, b)
    return a, b, g


def fiber_diagnostics(alpha, beta, gamma):
    """Per-fiber residuals and basis decomposition."""
    R_tensor = np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True) - TARGET_TENSOR
    print("\n── Fiber Diagnostics ──")
    print(f"{'Fiber':<8} {'max|R|':>8} {'‖R‖_F':>8}  α-p0/p1/p2       β-p0/p1/p2")
    print("─" * 72)
    for i in range(3):
        for j in range(3):
            c_idx = 3*i + j
            fR = R_tensor[:, :, c_idx]
            mx = float(np.max(np.abs(fR)))
            fro = float(np.linalg.norm(fR))
            a_rms = np.sqrt(np.mean((alpha[:, 3*i:3*i+3] @ P_INV)**2, axis=0))
            b_rms = np.sqrt(np.mean((beta[:, j::3] @ P_INV)**2, axis=0))
            print(f"C[{i},{j}]   {mx:8.5f} {fro:8.5f}  "
                  f"{a_rms[0]:.3f}/{a_rms[1]:.3f}/{a_rms[2]:.3f}   "
                  f"{b_rms[0]:.3f}/{b_rms[1]:.3f}/{b_rms[2]:.3f}")


def format_pattern(iface_indices, signs):
    return " ".join(f"{'+'if s>0 else '-'}C[{INTERFACES[i][0]},{INTERFACES[i][1]}]"
                    for i, s in zip(iface_indices, signs))


def search_all(alpha, beta, gamma, max_size=9, eps_grid=None):
    """Exhaustive search over interface subsets × signs × ε."""
    if eps_grid is None:
        eps_grid = [0.003, 0.01, 0.03, 0.1, 0.3]

    baseline = fitness(alpha, beta, gamma)
    g_refit = refit_gamma(alpha, beta)
    refit_base = fitness(alpha, beta, g_refit)
    print(f"\nBaseline max-abs:      {baseline:.10f}")
    print(f"γ-refit-only max-abs:  {refit_base:.10f}  (Δ={baseline - refit_base:+.10f})")

    n = len(INTERFACES)
    best_per_size = {}
    overall_best = None
    total = 0
    t0 = time.time()

    for size in range(1, min(max_size + 1, n + 1)):
        size_best = None
        size_top = []   # keep top-10

        for subset in itertools.combinations(range(n), size):
            # Fix first sign to +1 (global flip ≡ negate ε)
            sign_tails = list(itertools.product([1, -1], repeat=size - 1))
            for tail in sign_tails:
                signs = (1,) + tail
                for eps in eps_grid:
                    a_s, b_s, g_s = apply_squeeze(alpha, beta, subset, signs, eps)
                    fit = fitness(a_s, b_s, g_s)
                    total += 1

                    if size_best is None or fit < size_best[0]:
                        size_best = (fit, subset, signs, eps)

                    if len(size_top) < 10 or fit < size_top[-1][0]:
                        size_top.append((fit, subset, signs, eps))
                        size_top.sort(key=lambda x: x[0])
                        if len(size_top) > 10:
                            size_top.pop()

        best_per_size[size] = {"best": size_best, "top10": size_top}
        dt = time.time() - t0

        fit_b, sub_b, sgn_b, eps_b = size_best
        imp = refit_base - fit_b  # improvement vs refit-only baseline
        tag = " *** IMPROVED ***" if imp > 1e-10 else ""
        pat = format_pattern(sub_b, sgn_b)
        print(f"\n  Size {size}: {fit_b:.10f}  Δ(vs refit)={imp:+.10f}  ε={eps_b}{tag}")
        print(f"    {pat}   ({total} evals, {dt:.1f}s)")

        if overall_best is None or fit_b < overall_best[0]:
            overall_best = size_best

    return best_per_size, overall_best, baseline, refit_base


def fine_tune_eps(alpha, beta, pattern, n_pts=80):
    """Dense 1D ε scan around the coarse best."""
    _, subset, signs, coarse_eps = pattern
    lo = max(1e-4, coarse_eps * 0.05)
    hi = coarse_eps * 5
    eps_fine = np.linspace(lo, hi, n_pts)

    best_fit, best_eps, best_factors = float('inf'), coarse_eps, None
    for eps in eps_fine:
        a_s, b_s, g_s = apply_squeeze(alpha, beta, subset, signs, float(eps))
        fit = fitness(a_s, b_s, g_s)
        if fit < best_fit:
            best_fit = fit
            best_eps = float(eps)
            best_factors = (a_s, b_s, g_s)
    return best_factors, best_fit, best_eps


def als_polish(a, b, g, iters=10):
    """A few ALS iterations to relax all three factors."""
    best_a, best_b, best_g = a.copy(), b.copy(), g.copy()
    best_fit = fitness(a, b, g)
    for _ in range(iters):
        a = refit_alpha(b, g)
        b = refit_beta(a, g)
        g = refit_gamma(a, b)
        fit = fitness(a, b, g)
        if fit < best_fit:
            best_a, best_b, best_g, best_fit = a.copy(), b.copy(), g.copy(), fit
    return best_a, best_b, best_g, best_fit


def save_json(a, b, g, fit, pattern_info, path):
    data = {
        "fitness": fit,
        "rank": RANK, "dim": DIM,
        "pattern": pattern_info,
        "alpha": a.tolist(), "beta": b.tolist(), "gamma": g.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Systematic plaquette search")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--max-size", type=int, default=9)
    parser.add_argument("--refine", action="store_true",
                        help="Run full CPU refine on best result")
    parser.add_argument("--out", type=Path, default=Path("plaquette_best.json"))
    args = parser.parse_args()

    # Load best available candidate
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
    print(f"Loaded from {src}:  fitness={fitness(alpha, beta, gamma):.10f}")

    fiber_diagnostics(alpha, beta, gamma)

    # ── Exhaustive search ──
    print(f"\n{'='*70}")
    print(f"Exhaustive plaquette search: sizes 1–{args.max_size}")
    print(f"{'='*70}")

    results, overall_best, base, refit_base = search_all(
        alpha, beta, gamma, max_size=args.max_size)

    # ── Summary ──
    fit_ob, sub_ob, sgn_ob, eps_ob = overall_best
    print(f"\n{'='*70}")
    print(f"OVERALL BEST:  {fit_ob:.10f}  (Δ vs baseline={base - fit_ob:+.10f}, "
          f"Δ vs refit={refit_base - fit_ob:+.10f})")
    print(f"  {format_pattern(sub_ob, sgn_ob)}  ε={eps_ob}")

    # ── Top-10 per size ──
    print(f"\n{'='*70}")
    print(f"TOP-5 PER SIZE")
    print(f"{'='*70}")
    for size in sorted(results.keys()):
        top = results[size]["top10"][:5]
        print(f"\n  Size {size}:")
        for rank_i, (fit, sub, sgn, eps) in enumerate(top):
            imp = refit_base - fit
            print(f"    #{rank_i+1} {fit:.10f} Δ={imp:+.10f} ε={eps:6.3f}  "
                  f"{format_pattern(sub, sgn)}")

    # ── Fine-tune ε ──
    if base - fit_ob > 1e-12 or refit_base - fit_ob > 1e-12:
        print(f"\nFine-tuning ε ...")
        (a_ft, b_ft, g_ft), fit_ft, eps_ft = fine_tune_eps(alpha, beta, overall_best)
        print(f"  Fine-tuned: {fit_ft:.10f}  ε={eps_ft:.6f}")

        # ALS polish: let all 3 factors relax
        print(f"  ALS polish (10 iters) ...")
        a_als, b_als, g_als, fit_als = als_polish(a_ft, b_ft, g_ft, iters=10)
        print(f"  After ALS:  {fit_als:.10f}")

        best_a, best_b, best_g = (a_als, b_als, g_als) if fit_als < fit_ft else (a_ft, b_ft, g_ft)
        best_fit = min(fit_ft, fit_als)

        pattern_info = {
            "interfaces": [list(INTERFACES[i]) for i in sub_ob],
            "signs": list(sgn_ob),
            "eps": eps_ft,
        }
        save_json(best_a, best_b, best_g, best_fit, pattern_info, args.out)
        print(f"  Saved to {args.out}")

        # Optional full refine
        if args.refine:
            from db_optimizer.cpu_refine import cpu_full_refine
            print(f"\nFull CPU refine on plaquette-squeezed candidate ...")
            t0 = time.time()
            a_r, b_r, g_r, fit_r = cpu_full_refine(
                best_a, best_b, best_g, sweeps=20, patience=5)
            dt = time.time() - t0
            print(f"  Refined: {fit_r:.10f}  ({dt:.1f}s)")
            if fit_r < best_fit:
                save_json(a_r, b_r, g_r, fit_r, pattern_info, args.out)
                print(f"  Updated {args.out}")
    else:
        print("\nNo improvement found from any squeeze pattern.")


if __name__ == "__main__":
    main()
