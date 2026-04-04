"""
alphatensor_delete.py
=====================
Exhaustive search over all C(23,4) = 8,855 four-term deletions of the
AlphaTensor R=23 decomposition of 3x3 matrix multiplication.

For each 19-term subset, compute:
  - Gate 1: rank(H) == 10
  - Gate 2: rank([H|Delta]) == rank(H)  (delta_leak == 0)
  - Gate 3: rank([Sigma|H|Delta]) == 19  (augmented_gap == 0)
  - Reconstruction error (lstsq)

Sort results and print top 20 in a Rich table.
If any deletion achieves Gate 1 + Gate 2 + Gate 3: SOLUTION FOUND.

Usage:
    python CANON_OPTIMIZER/core/alphatensor_delete.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# AlphaTensor public rank-23 factorization
# Source: https://github.com/google-deepmind/alphatensor
# ---------------------------------------------------------------------------

def _alphatensor_uvw():
    u = np.array([
        [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 1, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.int64)
    v = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [-1, -1, 0, 0, -1, 0, -1, -1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1, -1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [-1, -1, 0, 0, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 1, 0, 1, 0, -1, 0, 0],
        [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.int64)
    w = np.array([
        [0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
        [-1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, -1, 1, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0, -1, 0, 0, -1, 0, 1],
        [-1, 1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0],
    ], dtype=np.int64)
    return u, v, w


def _build_terms():
    """Extract the 23 terms as (alpha, beta, gamma) numpy arrays each (3,3)."""
    u, v, w = _alphatensor_uvw()
    # Verify reconstruction, finding correct gamma orientation
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for r in range(3):
        for s in range(3):
            for col in range(3):
                T[r*3+s, s*3+col, r*3+col] = 1

    terms = []
    for k in range(23):
        alpha = u[:, k].reshape(3, 3)
        beta  = v[:, k].reshape(3, 3)
        gamma = w[:, k].reshape(3, 3)
        terms.append((alpha, beta, gamma))

    # Check reconstruction
    recon = np.zeros((9,9,9), dtype=np.int64)
    for alpha, beta, gamma in terms:
        recon += np.einsum('a,b,c->abc',
                           alpha.ravel(), beta.ravel(), gamma.ravel())
    if np.max(np.abs(recon - T)) != 0:
        # Try gamma.T
        terms2 = [(a, b, g.T) for a, b, g in terms]
        recon2 = np.zeros((9,9,9), dtype=np.int64)
        for alpha, beta, gamma in terms2:
            recon2 += np.einsum('a,b,c->abc',
                                alpha.ravel(), beta.ravel(), gamma.ravel())
        if np.max(np.abs(recon2 - T)) != 0:
            raise RuntimeError("AlphaTensor reconstruction failed — coefficient mismatch")
        terms = terms2

    return terms


# ---------------------------------------------------------------------------
# Gate computations from raw (alpha, beta, gamma) matrices
# ---------------------------------------------------------------------------

_ST_PAIRS = [(s, t) for s in range(3) for t in range(3) if s != t]  # 6 pairs


def _compute_rows(terms_subset):
    """
    Given a list of (alpha, beta, gamma) tuples, compute:
      sigma:   (R, 9)  — (alpha @ beta).ravel()
      H:       (R, 18) — [Eta1 | Eta2]
      delta:   (R, 54) — alpha[r,s]*beta[t,u] for s!=t
    """
    R = len(terms_subset)
    sigma = np.zeros((R, 9), dtype=np.float64)
    H     = np.zeros((R, 18), dtype=np.float64)
    delta = np.zeros((R, 54), dtype=np.float64)

    for k, (alpha, beta, gamma) in enumerate(terms_subset):
        a = alpha.astype(np.float64)
        b = beta.astype(np.float64)

        # Sigma
        sigma[k] = (a @ b).ravel()

        # Eta1[r,u] = a[r,0]*b[0,u] - a[r,1]*b[1,u]
        eta1 = np.outer(a[:, 0], b[0, :]) - np.outer(a[:, 1], b[1, :])
        # Eta2[r,u] = a[r,1]*b[1,u] - a[r,2]*b[2,u]
        eta2 = np.outer(a[:, 1], b[1, :]) - np.outer(a[:, 2], b[2, :])
        H[k, :9]  = eta1.ravel()
        H[k, 9:]  = eta2.ravel()

        # Delta: for each (r,u), for each (s,t) with s!=t: a[r,s]*b[t,u]
        col = 0
        for r in range(3):
            for uu in range(3):
                for s, t in _ST_PAIRS:
                    delta[k, col] = a[r, s] * b[t, uu]
                    col += 1

    return sigma, H, delta


def _numerical_rank(M, tol=1e-9):
    if M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    thresh = tol * (max(M.shape) * sv[0]) if sv[0] > 0 else tol
    return int(np.sum(sv > thresh))


def _score_subset(terms_subset):
    """
    Returns dict of diagnostics for a 19-term subset.
    """
    R = len(terms_subset)
    target_rank_H = R - 9  # = 10 for R=19

    sigma, H, delta = _compute_rows(terms_subset)
    N  = np.hstack([H, delta])        # (R, 72)
    SN = np.hstack([sigma, N])        # (R, 81)

    rank_H  = _numerical_rank(H)
    rank_N  = _numerical_rank(N)
    rank_SN = _numerical_rank(SN)

    gate1_gap     = abs(rank_H - target_rank_H)
    delta_leak    = rank_N - rank_H
    augmented_gap = R - rank_SN
    sigma_innov   = rank_SN - rank_N

    # Continuous delta residual: project Delta columns onto col(H) complement
    delta_resid = 0.0
    if rank_H > 0:
        U_H, _, _ = np.linalg.svd(H, full_matrices=True)
        P = U_H[:, :rank_H]
        delta_proj = P @ (P.T @ delta)
        delta_resid = float(np.linalg.norm(delta - delta_proj, 'fro'))

    # Reconstruction error via lstsq: solve Gamma @ SN = [3*I_9 | 0]
    rhs = np.zeros((R, 9 + 72), dtype=np.float64)
    rhs[:, :9] = 3.0 * np.eye(R)[:R, :9]  # wrong shape
    # Correct: Gamma is (9, R), solve SN.T @ Gamma.T = [3*I|0].T
    # target is (9, 81): first 9 cols = 3*I_9, rest = 0
    target_mat = np.zeros((9, 81), dtype=np.float64)
    target_mat[:, :9] = 3.0 * np.eye(9)
    # lstsq: SN.T (81, R) @ X (R, 9) = target_mat.T (81, 9)
    Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target_mat.T, rcond=None)
    recon_err = float(np.max(np.abs(target_mat.T - SN.T @ Gamma_T)))

    return {
        'rank_H':        rank_H,
        'rank_N':        rank_N,
        'rank_SN':       rank_SN,
        'gate1_gap':     gate1_gap,
        'delta_leak':    delta_leak,
        'delta_resid':   delta_resid,
        'augmented_gap': augmented_gap,
        'sigma_innov':   sigma_innov,
        'recon_err':     recon_err,
    }


def _verify_exact(terms_subset):
    """Independently verify reconstruction from Gamma (exact integer check)."""
    R = len(terms_subset)
    sigma, H, delta = _compute_rows(terms_subset)
    N  = np.hstack([H, delta])
    SN = np.hstack([sigma, N])

    target_mat = np.zeros((9, 81), dtype=np.float64)
    target_mat[:, :9] = 3.0 * np.eye(9)

    Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target_mat.T, rcond=None)
    Gamma = Gamma_T.T  # (9, R)

    # Reconstruct tensor from Gamma
    gammas = np.array([g.ravel() for _, _, g in terms_subset])  # (R, 9)
    # T_hat[a,b,c] = sum_k Gamma[c,k] * alpha_k[a] * beta_k[b]
    # Actually: T = sum_k gamma_k x alpha_k x beta_k where
    # T_hat = sum_k (Gamma @ SN)[k] ... let's do it directly
    # The correct equation: Gamma @ sigma = 3*I means C = (1/3)*Gamma @ A*B
    # Direct check: build T_hat = sum_k gamma_k(x)alpha_k(x)beta_k
    T_true = np.zeros((9, 9, 9), dtype=np.int64)
    for r in range(3):
        for s in range(3):
            for col in range(3):
                T_true[r*3+s, s*3+col, r*3+col] = 1

    T_hat = np.zeros((9, 9, 9), dtype=np.float64)
    for alpha, beta, gamma in terms_subset:
        T_hat += np.einsum('a,b,c->abc',
                           alpha.ravel().astype(float),
                           beta.ravel().astype(float),
                           gamma.ravel().astype(float))

    max_err = float(np.max(np.abs(T_hat - T_true)))
    return max_err


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run():
    console = Console()
    console.print("[bold cyan]AlphaTensor 4-term Deletion Search[/bold cyan]")
    console.print("Loading AlphaTensor R=23 terms...")

    terms = _build_terms()
    assert len(terms) == 23, f"Expected 23 terms, got {len(terms)}"

    # Verify full-rank reconstruction
    T_check = np.zeros((9,9,9), dtype=np.int64)
    for a, b, g in terms:
        T_check += np.einsum('a,b,c->abc',
                             a.ravel().astype(np.int64),
                             b.ravel().astype(np.int64),
                             g.ravel().astype(np.int64))
    T_true = np.zeros((9,9,9), dtype=np.int64)
    for r in range(3):
        for s in range(3):
            for col in range(3):
                T_true[r*3+s, s*3+col, r*3+col] = 1
    assert np.max(np.abs(T_check - T_true)) == 0, "AlphaTensor reconstruction FAILED"
    console.print("[green]AlphaTensor reconstruction verified (exact).[/green]")

    # Full-basis diagnostics
    sigma23, H23, delta23 = _compute_rows(terms)
    N23 = np.hstack([H23, delta23])
    SN23 = np.hstack([sigma23, N23])
    console.print(f"Full R=23: rank(H)={_numerical_rank(H23)}, rank(N)={_numerical_rank(N23)}, "
                  f"rank(SN)={_numerical_rank(SN23)}")

    console.print(f"\nEnumerating C(23,4) = 8,855 four-term deletions...")

    all_results = []
    solutions   = []
    n_combos    = 0

    for del_indices in itertools.combinations(range(23), 4):
        keep = [i for i in range(23) if i not in del_indices]
        subset = [terms[i] for i in keep]
        diag = _score_subset(subset)
        diag['deleted'] = del_indices
        all_results.append(diag)
        n_combos += 1

        # Check for solution
        if diag['gate1_gap'] == 0 and diag['delta_leak'] == 0 and diag['augmented_gap'] == 0:
            err = _verify_exact(subset)
            diag['verify_err'] = err
            solutions.append(diag)
            console.print(
                f"\n[bold bright_green]*** SOLUTION FOUND! ***[/bold bright_green]\n"
                f"  Deleted terms (0-indexed): {del_indices}\n"
                f"  rank(H)={diag['rank_H']}, delta_leak={diag['delta_leak']}, "
                f"augmented_gap={diag['augmented_gap']}\n"
                f"  Reconstruction error: {err:.2e}"
            )
            # Save solution
            sol_path = ROOT / "CANON_OPTIMIZER" / "core" / "ALPHATENSOR_SOLUTION.json"
            sol_data = {
                'deleted_indices': list(del_indices),
                'kept_indices':    keep,
                'rank_H':          diag['rank_H'],
                'delta_leak':      diag['delta_leak'],
                'augmented_gap':   diag['augmented_gap'],
                'recon_err':       diag['recon_err'],
                'verify_err':      err,
            }
            sol_path.write_text(json.dumps(sol_data, indent=2))
            console.print(f"  Saved to [bold]{sol_path}[/bold]")

    if n_combos % 500 == 0:
        pass  # rich live would be nice but simple is fine for 8855 iterations

    console.print(f"\nDone. Checked {n_combos:,} deletions.")

    if solutions:
        console.print(f"\n[bold bright_green]{len(solutions)} SOLUTION(S) FOUND![/bold bright_green]")
    else:
        console.print("\n[yellow]No deletion achieves Gate1+Gate2+Gate3. Best near-misses below.[/yellow]")

    # Sort: (gate1_gap, delta_leak, delta_resid, augmented_gap, recon_err)
    all_results.sort(key=lambda d: (
        d['gate1_gap'], d['delta_leak'], d['delta_resid'], d['augmented_gap'], d['recon_err']))

    # Print top 20
    table = Table(title="Top 20 Deletions (sorted by gate hierarchy)", box=box.SIMPLE_HEAD)
    table.add_column("Deleted", style="dim")
    table.add_column("G1 gap", justify="right")
    table.add_column("leak", justify="right")
    table.add_column("d_resid", justify="right")
    table.add_column("aug_gap", justify="right")
    table.add_column("sig_innov", justify="right")
    table.add_column("rank_H", justify="right")
    table.add_column("rank_SN", justify="right")
    table.add_column("recon_err", justify="right")

    for row in all_results[:20]:
        leak_str = str(row['delta_leak'])
        if row['delta_leak'] == 0:
            leak_str = f"[green]{leak_str}[/green]"
        aug_str = str(row['augmented_gap'])
        if row['augmented_gap'] == 0:
            aug_str = f"[green]{aug_str}[/green]"
        table.add_row(
            str(row['deleted']),
            str(row['gate1_gap']),
            leak_str,
            f"{row['delta_resid']:.3f}",
            aug_str,
            str(row['sigma_innov']),
            str(row['rank_H']),
            str(row['rank_SN']),
            f"{row['recon_err']:.3e}",
        )

    console.print(table)

    # Summary stats
    console.print("\n[bold]Summary statistics:[/bold]")
    leaks  = [d['delta_leak']    for d in all_results]
    g1gaps = [d['gate1_gap']     for d in all_results]
    agaps  = [d['augmented_gap'] for d in all_results]
    console.print(f"  Gate1 gap=0: {sum(g==0 for g in g1gaps):,} / {len(g1gaps):,}")
    console.print(f"  Delta leak=0: {sum(l==0 for l in leaks):,} / {len(leaks):,}")
    console.print(f"  Aug gap=0:   {sum(a==0 for a in agaps):,} / {len(agaps):,}")
    console.print(f"  Min delta_leak: {min(leaks)}")
    console.print(f"  Min augmented_gap: {min(agaps)}")

    # Show distribution of delta_leak values
    from collections import Counter
    leak_dist = Counter(leaks)
    console.print(f"\n  delta_leak distribution: {dict(sorted(leak_dist.items()))}")
    g1_dist = Counter(g1gaps)
    console.print(f"  gate1_gap distribution:  {dict(sorted(g1_dist.items()))}")

    # Save all results
    out_path = ROOT / "CANON_OPTIMIZER" / "core" / "alphatensor_delete_results.json"
    save_data = sorted(
        [{'deleted': list(d['deleted']), **{k: v for k,v in d.items() if k != 'deleted'}}
         for d in all_results],
        key=lambda d: (d['gate1_gap'], d['delta_leak'], d['delta_resid'], d['augmented_gap'])
    )
    out_path.write_text(json.dumps(save_data[:100], indent=2))
    console.print(f"\nTop 100 results saved to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    run()
