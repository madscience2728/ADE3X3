#!/usr/bin/env python3
"""Attack 3 — Test Δ ⊂ span(H) on rank-19 near-miss candidates and known exact decompositions.

H = [Eta1 | Eta2]  (R × 18)
Delta                (R × 54)

If rank([H | Delta]) == rank(H), then Delta ⊂ span(H) (column-wise, transposed view).

More precisely, we check row-space containment: each row of Delta should lie in 
the row-space of H. Equivalently, Delta^T should lie in col-span of H^T.

We compute:
  1. rank(H) vs rank([H | Delta])
  2. Projection residual: ||Delta - H @ lstsq(H, Delta)||_max
  3. Per-term residuals
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_optimizer.tensor import fiber_mode_decomposition


def load_candidate(path):
    with open(path) as f:
        d = json.load(f)
    alpha = np.array(d["alpha"])
    beta = np.array(d["beta"])
    gamma = np.array(d["gamma"])
    fitness = d.get("fitness", None)
    return alpha, beta, gamma, fitness


def load_alphatensor():
    """Load the AlphaTensor rank-23 decomposition."""
    # Try attack_common
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "outputs", "ade3x3_attack"))
    try:
        from attack_common import load_public_terms  # type: ignore[import-not-found]
        terms = load_public_terms()
        R = len(terms)
        alpha = np.array([t.a for t in terms])
        beta = np.array([t.b for t in terms])
        gamma = np.array([t.c for t in terms])
        return alpha, beta, gamma, 0.0, "AlphaTensor-R23"
    except Exception as e:
        print(f"  [skip AlphaTensor: {e}]")
        return None


def analyze(name, alpha, beta, gamma, fitness=None):
    R = alpha.shape[0]
    fm = fiber_mode_decomposition(alpha, beta, gamma)
    
    Sigma = fm["Sigma"]   # (R, 9)
    Eta1 = fm["Eta1"]     # (R, 9)
    Eta2 = fm["Eta2"]     # (R, 9)
    Delta = fm["Delta"]   # (R, 54)
    Gamma = fm["Gamma"]   # (9, R)
    
    H = np.hstack([Eta1, Eta2])  # (R, 18)
    
    # Ranks
    rank_H = np.linalg.matrix_rank(H, tol=1e-8)
    rank_HD = np.linalg.matrix_rank(np.hstack([H, Delta]), tol=1e-8)
    rank_Delta = np.linalg.matrix_rank(Delta, tol=1e-8)
    
    # Projection residual: project each column of Delta onto col-span of H
    # Delta is (R, 54), H is (R, 18). We want Delta ≈ H @ C for some C.
    # This is column-space containment of Delta's columns in H's columns.
    # Or equivalently, row-space: each row of Delta in row-span of H.
    # Use lstsq on transposed: Delta.T ≈ H.T @ ... no.
    # Actually: Delta = H @ C means each column of Delta is in col(H).
    # lstsq: C = lstsq(H, Delta)[0], residual = Delta - H @ C
    C, res, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
    proj_residual = Delta - H @ C
    max_residual = np.max(np.abs(proj_residual))
    frob_residual = np.linalg.norm(proj_residual, 'fro')
    frob_delta = np.linalg.norm(Delta, 'fro')
    
    # Per-term residual (row-wise)
    row_residuals = np.linalg.norm(proj_residual, axis=1)
    
    # ker(Gamma) dimension
    gamma_rank = fm["gamma_rank"]
    ker_gamma_dim = R - gamma_rank
    
    # Nuisance rank
    nuisance = np.hstack([H, Delta])
    nuisance_rank = fm["nuisance_rank"]
    
    # Conservation check
    eta_nullity = R - fm["sigma_rank"]
    
    print(f"\n{'='*60}")
    print(f"  {name}  (R={R}, fitness={fitness})")
    print(f"{'='*60}")
    print(f"  rank(Sigma)    = {fm['sigma_rank']}")
    print(f"  rank(H)        = {rank_H}")
    print(f"  rank(Delta)    = {rank_Delta}")
    print(f"  rank([H|Delta])= {rank_HD}   {'✓ Δ⊂span(H)' if rank_HD == rank_H else f'✗ GAP={rank_HD - rank_H}'}")
    print(f"  nuisance_rank  = {nuisance_rank}")
    print(f"  gamma_rank     = {gamma_rank},  ker(Gamma) dim = {ker_gamma_dim}")
    print(f"  η_nullity      = {eta_nullity}")
    print(f"  R + η_null     = {R + eta_nullity}  (should be 27)")
    print(f"  Γ·Σ residual   = {fm['gs_residual']:.2e}")
    print(f"  ||proj resid||_max  = {max_residual:.6e}")
    print(f"  ||proj resid||_fro  = {frob_residual:.6e}  (||Δ||_fro = {frob_delta:.2f})")
    print(f"  relative resid      = {frob_residual/max(frob_delta,1e-30):.6e}")
    
    # Show worst 5 terms
    worst_idx = np.argsort(row_residuals)[::-1][:5]
    print(f"  Worst per-term residuals:")
    for i in worst_idx:
        print(f"    term {i:2d}: {row_residuals[i]:.6e}")
    
    return rank_HD == rank_H


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    
    # Test rank-19 candidates
    candidates = [
        ("slp_turbo_best", "slp_turbo_best.json"),
        ("slp_best@0.074", "slp_best_at_0.074.json"),
        ("slp_best", "slp_best.json"),
        ("chain_best", "chain_best.json"),
        ("plaquette_best", "plaquette_best.json"),
        ("shotgun_best", "shotgun_best.json"),
    ]
    
    results = {}
    
    for name, fname in candidates:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        try:
            alpha, beta, gamma, fitness = load_candidate(path)
            ok = analyze(name, alpha, beta, gamma, fitness)
            results[name] = ok
        except Exception as e:
            print(f"\n  {name}: ERROR — {e}")
    
    # Try AlphaTensor
    at = load_alphatensor()
    if at is not None:
        alpha, beta, gamma, fitness, name = at
        ok = analyze(name, alpha, beta, gamma, fitness)
        results[name] = ok
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {name:25s}: {'Δ⊂span(H) ✓' if ok else 'Δ⊄span(H) ✗'}")
    
    # ── Attack 2: Genericity / dimension counting ──
    print(f"\n{'='*60}")
    print("  ATTACK 2: Genericity dimension counting")
    print(f"{'='*60}")
    for R in [19, 20, 21, 22, 23, 27]:
        # H is (R, 18), so col-rank(H) ≤ min(R, 18)
        # Delta is (R, 54), col-rank(Delta) ≤ min(R, 54)
        # For Δ⊂colspan(H): need rank(H) ≥ rank([H|Δ])
        # In generic position, rank(H) = min(R, 18)
        # rank([H|Δ]) = min(R, 72) since [H|Δ] is (R, 72)
        # So generically Δ⊂span(H) iff min(R,18) = min(R,72), i.e. R ≤ 18
        # BUT this is column containment. Actually...
        # H has R rows, 18 cols. Delta has R rows, 54 cols.
        # We need: each column of Delta ∈ col-span of H (as vectors in R^R).
        # col-span(H) has dimension min(R, 18). 
        # If R > 18, col-span(H) is at most 18-dim subspace of R^R.
        # Delta columns live in R^R, so generically they DON'T lie in an 18-dim subspace.
        # UNLESS there's algebraic structure forcing it.
        #
        # Wait — re-examine. The constraint is that all nuisance = [H|Delta] rows
        # must lie in ker(Gamma). ker(Gamma) has dim R-9.
        # So nuisance rows lie in (R-9)-dim subspace.
        # H has 18 column-directions, projected into this (R-9)-dim space.
        # For column containment of Delta in span(H), we need:
        # The 18 H-columns span enough of R^R to cover the 54 Delta-columns.
        # Since all are constrained to ker(Gamma) (R-9 dim), 
        # H columns span min(18, R-9) dims, Delta needs to fit in that.
        # rank(Delta) in ker(Gamma) ≤ R-9-rank(H).  Actually:
        # For Δ⊂colspan(H): rank(H) must equal rank([H|Δ]).
        # All columns of [H|Δ] are in ker(Gamma) (R-9 dim), so rank([H|Δ]) ≤ R-9.
        # H has 18 cols → can span min(18, R-9). 
        # If R-9 ≤ 18 (R ≤ 27), H can potentially span all of ker(Gamma).
        # For R=19: ker(Gamma) is 10-dim, H has 18 cols → generically spans all 10 dims.
        # Delta columns also in this 10-dim space → Δ⊂span(H) generically!
        ker_dim = R - 9
        h_cols = 18
        h_effective_rank = min(h_cols, ker_dim)
        delta_cols = 54
        contained = "YES (generically)" if h_effective_rank >= ker_dim else "MAYBE"
        print(f"  R={R:2d}: ker(Γ) dim={ker_dim:2d}, H cols=18 → eff rank={h_effective_rank:2d}, "
              f"Δ⊂span(H)? {contained}")


if __name__ == "__main__":
    main()
