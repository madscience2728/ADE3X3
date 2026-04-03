#!/usr/bin/env python3
"""
Symbolic exact proof of Δ ⊂ span(H) for ⟨3,3,3⟩ at AlphaTensor R=23.

Lifts the n=2 Strassen proof to n=3 using AlphaTensor's exact integer factors.
"""

import sys, os
import numpy as np
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    load_public_rank23_terms,
)
# from db_optimizer.tensor import fiber_mode_decomposition  # not needed


def exact_int_factors():
    """Extract exact integer alpha, beta, gamma matrices from AlphaTensor."""
    terms, orientation, max_abs = load_public_rank23_terms()
    R = len(terms)
    n = 3
    alpha = np.zeros((R, n * n), dtype=int)
    beta = np.zeros((R, n * n), dtype=int)
    gamma = np.zeros((R, n * n), dtype=int)
    for k, t in enumerate(terms):
        alpha[k] = t.alpha.flatten().astype(int)
        beta[k] = t.beta.flatten().astype(int)
        gamma[k] = t.gamma.flatten().astype(int)
    # Verify exact integer
    for k, t in enumerate(terms):
        assert np.allclose(alpha[k], t.alpha.flatten()), f"alpha[{k}] not integer"
        assert np.allclose(beta[k], t.beta.flatten()), f"beta[{k}] not integer"
        assert np.allclose(gamma[k], t.gamma.flatten()), f"gamma[{k}] not integer"
    return alpha, beta, gamma, R, n


def build_exact_fiber_modes(alpha, beta, gamma, R, n):
    """Build Sigma, Eta1, Eta2, Delta, Gamma as exact Fraction matrices."""
    n2 = n * n
    
    # Gamma: (n2, R)
    Gamma = [[Fraction(int(gamma[k, c])) for k in range(R)] for c in range(n2)]
    
    # Fiber products: p_k[s, r, u] = alpha_k[n*r+s] * beta_k[n*s+u]
    # Sigma_k[r,u] = sum_s p[s,r,u]
    # Eta1_k[r,u] = p[0,r,u] - p[1,r,u]
    # Eta2_k[r,u] = p[1,r,u] - p[2,r,u]  (for n=3)
    
    Sigma = []  # list of columns, each R-long
    Eta1 = []
    Eta2 = []
    
    for r in range(n):
        for u in range(n):
            sig_col = []
            eta1_col = []
            eta2_col = []
            for k in range(R):
                p = [Fraction(int(alpha[k, n * r + s]) * int(beta[k, n * s + u])) for s in range(n)]
                sig_col.append(sum(p))
                eta1_col.append(p[0] - p[1])
                eta2_col.append(p[1] - p[2])
            Sigma.append(sig_col)
            Eta1.append(eta1_col)
            Eta2.append(eta2_col)
    
    # Delta: cross-fiber products alpha_k[n*r+s] * beta_k[n*t+u] for s != t
    Delta = []
    for r in range(n):
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                for u in range(n):
                    d_col = []
                    for k in range(R):
                        d_col.append(Fraction(int(alpha[k, n * r + s]) * int(beta[k, n * t + u])))
                    Delta.append(d_col)
    
    # H = [Eta1 | Eta2]: R × 18
    H_cols = Eta1 + Eta2  # 18 columns, each R-long
    
    return Gamma, Sigma, H_cols, Delta


def fraction_matrix_rank(M_rows):
    """Exact rank of a matrix given as list of rows of Fractions via row echelon."""
    if not M_rows or not M_rows[0]:
        return 0
    m = len(M_rows)
    n = len(M_rows[0])
    # Copy
    A = [list(row) for row in M_rows]
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(rank, m):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        inv = Fraction(1, 1) / A[rank][col]
        A[rank] = [x * inv for x in A[rank]]
        for row in range(m):
            if row == rank:
                continue
            if A[row][col] != 0:
                factor = A[row][col]
                A[row] = [A[row][j] - factor * A[rank][j] for j in range(n)]
        rank += 1
    return rank


def fraction_nullspace(M_rows):
    """Exact nullspace basis of matrix (rows) over Q. Returns list of column vectors."""
    if not M_rows or not M_rows[0]:
        return []
    m = len(M_rows)
    n = len(M_rows[0])
    # Augmented row echelon
    A = [list(row) for row in M_rows]
    pivot_cols = []
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        inv = Fraction(1) / A[rank][col]
        A[rank] = [x * inv for x in A[rank]]
        for row in range(m):
            if row == rank and A[row][col] != 0:
                continue
            if row != rank and A[row][col] != 0:
                factor = A[row][col]
                A[row] = [A[row][j] - factor * A[rank][j] for j in range(n)]
        pivot_cols.append(col)
        rank += 1
    
    # Free columns
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        vec = [Fraction(0)] * n
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            vec[pc] = -A[i][fc]
        basis.append(vec)
    return basis


def fraction_solve(H_rows, delta_col):
    """Solve H @ c = delta for c over Q. H_rows is list of rows, delta_col is list.
    Returns c (list) or None if no solution."""
    m = len(H_rows)
    n = len(H_rows[0])
    # Augmented matrix [H | delta]
    A = [list(H_rows[i]) + [delta_col[i]] for i in range(m)]
    pivot_cols = []
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        inv = Fraction(1) / A[rank][col]
        A[rank] = [x * inv for x in A[rank]]
        for row in range(m):
            if row != rank and A[row][col] != 0:
                factor = A[row][col]
                A[row] = [A[row][j] - factor * A[rank][j] for j in range(n + 1)]
        pivot_cols.append(col)
        rank += 1
    
    # Check consistency: rows below rank should have zero RHS
    for row in range(rank, m):
        if A[row][n] != 0:
            return None
    
    # Extract solution (set free variables to 0)
    c = [Fraction(0)] * n
    for i, pc in enumerate(pivot_cols):
        c[pc] = A[i][n]
    return c


def main():
    print("=" * 70)
    print("  SYMBOLIC EXACT PROOF: Δ ⊂ span(H) for ⟨3,3,3⟩ R=23")
    print("=" * 70)
    
    print("\n[1] Loading exact integer factors...")
    alpha, beta, gamma, R, n = exact_int_factors()
    print(f"    R={R}, n={n}")
    print(f"    alpha range: [{alpha.min()}, {alpha.max()}]")
    print(f"    beta range:  [{beta.min()}, {beta.max()}]")
    print(f"    gamma range: [{gamma.min()}, {gamma.max()}]")
    
    print("\n[2] Building exact Fraction fiber modes...")
    Gamma, Sigma, H_cols, Delta = build_exact_fiber_modes(alpha, beta, gamma, R, n)
    print(f"    Gamma: {len(Gamma)} × {R}")
    print(f"    H: {R} × {len(H_cols)}")
    print(f"    Delta: {R} × {len(Delta)}")
    
    print("\n[3] Verifying Gamma @ H = 0 and Gamma @ Delta = 0...")
    # Gamma is n2 rows × R cols. H_cols[m] is R-long. Check Gamma @ h = 0.
    for m, hcol in enumerate(H_cols):
        for c in range(len(Gamma)):
            val = sum(Gamma[c][k] * hcol[k] for k in range(R))
            assert val == 0, f"Gamma @ H[:,{m}] row {c} = {val} ≠ 0"
    for j, dcol in enumerate(Delta):
        for c in range(len(Gamma)):
            val = sum(Gamma[c][k] * dcol[k] for k in range(R))
            assert val == 0, f"Gamma @ Delta[:,{j}] row {c} = {val} ≠ 0"
    print("    ✓ Gamma @ H = 0 (exact)")
    print("    ✓ Gamma @ Delta = 0 (exact)")
    
    print("\n[4] Computing ker(Gamma)...")
    # Gamma as rows for nullspace computation
    null_basis = fraction_nullspace(Gamma)
    ker_dim = len(null_basis)
    print(f"    dim(ker(Gamma)) = {ker_dim}  (expected R - n² = {R - n*n})")
    assert ker_dim == R - n * n
    
    print("\n[5] Projecting H and Delta into ker(Gamma) coordinates...")
    # null_basis[i] is a R-long vector. Form projection matrix.
    # Project each H column: express in nullspace coordinates.
    # H_proj[i, m] = null_basis[i] . H_cols[m]
    H_proj_rows = []
    for i in range(ker_dim):
        row = []
        for m in range(len(H_cols)):
            row.append(sum(null_basis[i][k] * H_cols[m][k] for k in range(R)))
        H_proj_rows.append(row)
    
    Delta_proj_rows = []
    for i in range(ker_dim):
        row = []
        for j in range(len(Delta)):
            row.append(sum(null_basis[i][k] * Delta[j][k] for k in range(R)))
        Delta_proj_rows.append(row)
    
    print(f"    H_proj: {ker_dim} × {len(H_cols)}")
    print(f"    Delta_proj: {ker_dim} × {len(Delta)}")
    
    print("\n[6] Computing exact ranks...")
    rank_H = fraction_matrix_rank(H_proj_rows)
    print(f"    rank(H_proj) = {rank_H}  (need = {ker_dim} for full span)")
    
    # Combined [H | Delta]
    combined_rows = [H_proj_rows[i] + Delta_proj_rows[i] for i in range(ker_dim)]
    rank_combined = fraction_matrix_rank(combined_rows)
    print(f"    rank([H|Delta]_proj) = {rank_combined}")
    
    if rank_H == ker_dim:
        print(f"\n    ★ rank(H_proj) = dim(ker(Gamma)) = {ker_dim}")
        print(f"    ★ H spans ALL of ker(Gamma)")
        print(f"    ★ Therefore Δ ⊂ span(H)  ■  (QED)")
    elif rank_H == rank_combined:
        print(f"\n    ★ rank(H_proj) = rank([H|Delta]_proj) = {rank_H}")
        print(f"    ★ Delta adds no new directions beyond H")
        print(f"    ★ Therefore Δ ⊂ span(H)  ■  (QED)")
    else:
        print(f"\n    ✗ rank(H_proj) = {rank_H} < rank([H|Delta]_proj) = {rank_combined}")
        print(f"    ✗ Delta extends beyond span(H) — containment FAILS")
        return
    
    print("\n[7] Solving for explicit coefficients: Delta = H @ C ...")
    n_solved = 0
    n_failed = 0
    max_denom = 1
    # Work in ker(Gamma) projected space
    # Transpose: solve H_proj^T @ c = Delta_proj^T column by column
    H_T = [[H_proj_rows[i][m] for i in range(ker_dim)] for m in range(len(H_cols))]
    
    for j in range(len(Delta)):
        dcol = [Delta_proj_rows[i][j] for i in range(ker_dim)]
        # Check if dcol is zero
        if all(v == 0 for v in dcol):
            n_solved += 1
            continue
        c = fraction_solve(H_proj_rows, dcol)
        if c is not None:
            # Verify
            for i in range(ker_dim):
                reconstructed = sum(c[m] * H_proj_rows[i][m] for m in range(len(H_cols)))
                assert reconstructed == dcol[i], f"Mismatch at Delta col {j}, ker coord {i}"
            n_solved += 1
            for ci in c:
                if ci.denominator > max_denom:
                    max_denom = ci.denominator
        else:
            n_failed += 1
            print(f"    ✗ Delta column {j}: NO SOLUTION")
    
    print(f"    Solved: {n_solved}/{len(Delta)}")
    print(f"    Failed: {n_failed}/{len(Delta)}")
    print(f"    Max denominator in coefficients: {max_denom}")
    
    if n_failed == 0:
        print(f"\n{'=' * 70}")
        print(f"  THEOREM (exact, symbolic, verified):")
        print(f"  For the AlphaTensor R=23 decomposition of ⟨3,3,3⟩,")
        print(f"  Δ ⊂ span(H) with all coefficients in Q.")
        print(f"  rank(H_proj) = {rank_H} = dim(ker(Γ)) = {ker_dim}")
        print(f"  All {len(Delta)} Delta columns solved exactly.  ■")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
