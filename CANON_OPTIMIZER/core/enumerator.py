"""
Core tree-search engine — 3-phase architecture, per-term interleaved.

For each term k (depth-first, one term per tree level):
  Phase A: Enumerate active entries (alpha_k[r,:], beta_k[:,u]) — 676 pairs.
  Phase B: Enumerate non-active alpha rows (729 combos), then SOLVE for
           non-active beta via linear algebra over the coefficient field.
    - Basis terms (first target_rank_H): H row must add exactly 1 new dimension.
    - Dependent terms (remaining): H row must lie in existing span V.
  Phase C: After all R terms placed, check Gate 2 + Gate 3.

Key insight: with alpha fully fixed, H[k,:] is LINEAR in beta_k.
The constraint "H row in/out of subspace V" becomes a small linear system
in the 6 non-active beta entries. Solve, don't enumerate.
"""

import numpy as np
from itertools import product as iter_product
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass

from .tensor import fast_H_row, fast_sigma_row, fast_delta_row
from .gates import check_gate3, _numerical_rank
from .fiber import enumerate_fiber_partitions
from .config import SearchConfig


@dataclass
class SearchStats:
    candidates_checked: int = 0
    gate1_passes: int = 0
    gate2_passes: int = 0
    gate3_passes: int = 0
    solutions_found: int = 0
    pruned_rank_high: int = 0
    pruned_rank_low: int = 0
    pruned_no_solution: int = 0
    partitions_searched: int = 0
    partitions_total: int = 0


def _enumerate_vectors(coeff_field: List[int], n: int = 3,
                       allow_zero: bool = False) -> np.ndarray:
    """Enumerate all n-vectors over coeff_field."""
    vecs = np.array(list(iter_product(coeff_field, repeat=n)), dtype=np.int64)
    if not allow_zero:
        nonzero_mask = np.any(vecs != 0, axis=1)
        vecs = vecs[nonzero_mask]
    return vecs


def _build_fiber_assignment(partition: Tuple[int, ...]) -> List[Tuple[int, int, int]]:
    """Given partition (n1,...,n9), return [(term_idx, fiber_r, fiber_u), ...]."""
    assignments = []
    term_idx = 0
    for fiber_idx, n_terms in enumerate(partition):
        r = fiber_idx // 3
        u = fiber_idx % 3
        for _ in range(n_terms):
            assignments.append((term_idx, r, u))
            term_idx += 1
    return assignments


# ── H-row linearity machinery ──

def _compute_h_row_decomposed(alpha_k: np.ndarray, beta_active_col: np.ndarray,
                               r: int, u: int, other_cols: List[int],
                               n: int = 3):
    """Decompose H[k,:] = h_const + h_linear @ beta_nonactive.

    h_const: (2*n^2,) — contributions from active column beta[:,u] (known).
    h_linear: (2*n^2, n*len(other_cols)) — linear map from non-active beta.

    beta_nonactive packing: [beta[0,oc0], beta[1,oc0], beta[2,oc0],
                              beta[0,oc1], beta[1,oc1], beta[2,oc1]]
    """
    nn = n * n
    n_unknowns = n * len(other_cols)

    h_const = np.zeros(2 * nn, dtype=np.float64)
    h_linear = np.zeros((2 * nn, n_unknowns), dtype=np.float64)

    oc_positions = {}
    for i, oc in enumerate(other_cols):
        for s in range(n):
            oc_positions[(s, oc)] = i * n + s

    for rr in range(n):
        a0 = float(alpha_k[rr, 0])
        a1 = float(alpha_k[rr, 1])
        a2 = float(alpha_k[rr, 2])

        for uu in range(n):
            idx_e1 = rr * n + uu
            idx_e2 = nn + rr * n + uu

            if uu == u:
                b0 = float(beta_active_col[0])
                b1 = float(beta_active_col[1])
                b2 = float(beta_active_col[2])
                h_const[idx_e1] = a0 * b0 - a1 * b1
                h_const[idx_e2] = a1 * b1 - a2 * b2
            else:
                b0_idx = oc_positions[(0, uu)]
                b1_idx = oc_positions[(1, uu)]
                b2_idx = oc_positions[(2, uu)]

                h_linear[idx_e1, b0_idx] = a0
                h_linear[idx_e1, b1_idx] = -a1
                h_linear[idx_e2, b1_idx] = a1
                h_linear[idx_e2, b2_idx] = -a2

    return h_const, h_linear


def _snap_to_field(x: np.ndarray, coeff_field: List[int]) -> Optional[np.ndarray]:
    """Snap each entry of x to nearest value in coeff_field. Return None if any
    entry is too far from the field."""
    field_arr = np.array(coeff_field, dtype=np.float64)
    result = np.empty(len(x), dtype=np.int64)
    for i in range(len(x)):
        dists = np.abs(field_arr - x[i])
        best = np.argmin(dists)
        if dists[best] > 0.4:
            return None
        result[i] = coeff_field[best]
    return result


def _solve_beta_nonactive(h_const: np.ndarray, h_linear: np.ndarray,
                           complement: np.ndarray,
                           coeff_field: List[int], n: int = 3,
                           n_other_cols: int = 2) -> List[np.ndarray]:
    """Solve complement @ (h_const + h_linear @ x) = 0 for x over coeff_field.

    complement = (I - P_V): projects onto orthogonal complement of current span V.
    System: M @ x = -b where M = complement @ h_linear, b = complement @ h_const.

    For basis terms: complement = I (all 18 dims, requiring h_row ≠ 0 and new dimension).
    For dependent terms: complement projects away from established span.
    """
    n_unknowns = n * n_other_cols
    M = complement @ h_linear
    b = complement @ h_const

    # Clean near-zeros
    M[np.abs(M) < 1e-14] = 0
    b[np.abs(b) < 1e-14] = 0

    # Remove inactive rows (all-zero row with zero b)
    row_norms = np.sqrt(np.sum(M ** 2, axis=1) + b ** 2)
    active = row_norms > 1e-12
    M_act = M[active]
    b_act = b[active]

    if M_act.shape[0] == 0:
        # No constraints — all field vectors valid
        return [np.array(v, dtype=np.int64)
                for v in iter_product(coeff_field, repeat=n_unknowns)]

    target = -b_act
    rank_M = np.linalg.matrix_rank(M_act, tol=1e-10)

    if rank_M >= n_unknowns:
        # Fully determined
        x_real, _, _, _ = np.linalg.lstsq(M_act, target, rcond=None)
        if np.max(np.abs(M_act @ x_real - target)) > 1e-8:
            return []
        sol = _snap_to_field(x_real, coeff_field)
        if sol is not None and np.max(np.abs(M_act @ sol.astype(np.float64) - target)) < 1e-8:
            return [sol]
        return []

    # Underdetermined: particular + null space
    x_part = np.linalg.pinv(M_act) @ target
    if np.max(np.abs(M_act @ x_part - target)) > 1e-8:
        return []

    _, S_vals, Vt = np.linalg.svd(M_act, full_matrices=True)
    null_dim = n_unknowns - rank_M
    null_basis = Vt[rank_M:].T  # (n_unknowns, null_dim)

    if null_dim > 5:
        return []  # Too many free params

    solutions = []
    max_field = max(abs(c) for c in coeff_field)
    search_range = list(range(-3 * max_field, 3 * max_field + 1))

    for coeffs in iter_product(search_range, repeat=null_dim):
        c = np.array(coeffs, dtype=np.float64)
        x_cand = x_part + null_basis @ c
        sol = _snap_to_field(x_cand, coeff_field)
        if sol is not None:
            if np.max(np.abs(M_act @ sol.astype(np.float64) - target)) < 1e-8:
                is_dup = any(np.array_equal(sol, s) for s in solutions)
                if not is_dup:
                    solutions.append(sol)

    return solutions


def _solve_beta_for_new_dimension(h_const: np.ndarray, h_linear: np.ndarray,
                                   H_prev: np.ndarray, prev_rank: int,
                                   coeff_field: List[int], tol: float,
                                   n: int = 3, n_other_cols: int = 2
                                   ) -> List[np.ndarray]:
    """For a basis term: find all beta_nonactive in coeff_field such that
    h_row = h_const + h_linear @ beta_na adds exactly 1 new dimension to H.

    Strategy: h_row must NOT lie entirely in span(H_prev). We enumerate all
    field solutions for the non-active beta (small: 3^6 = 729 max) and filter.

    Since 729 is small enough, we just brute-force enumerate all non-active beta
    vectors and check the rank increment.
    """
    n_unknowns = n * n_other_cols
    solutions = []

    for v in iter_product(coeff_field, repeat=n_unknowns):
        beta_na = np.array(v, dtype=np.int64)
        h_row = h_const + h_linear @ beta_na.astype(np.float64)

        # Check that appending this row increases rank by exactly 1
        if prev_rank == 0:
            # First term: just need nonzero H row
            if np.max(np.abs(h_row)) < 1e-12:
                continue
        else:
            H_test = np.vstack([H_prev, h_row.reshape(1, -1)])
            new_rank, _ = _numerical_rank(H_test, tol)
            if new_rank != prev_rank + 1:
                continue

        solutions.append(beta_na)

    return solutions


# ── Main search ──

def search_partition(config: SearchConfig,
                     partition: Tuple[int, ...],
                     fiber_assign: List[Tuple[int, int, int]],
                     callback: Optional[Callable] = None,
                     progress_callback: Optional[Callable] = None,
                     stop_flag: Optional[Callable] = None,
                     max_solutions: int = 1) -> Tuple[list, SearchStats]:
    """
    Per-term interleaved search for a single fiber partition.

    For each term k (depth-first):
      Phase A: enumerate active entries (alpha_k[r,:], beta_k[:,u])
      Phase B: enumerate non-active alpha, solve for non-active beta
      Prune: rank(H) constraints
    After all R terms: Phase C (Gate 2 + Gate 3).
    """
    R = config.R
    n = config.n
    nn = n * n
    target_rank_H = config.target_rank_H
    coeff_field = config.coeff_field
    tol = config.rank_tolerance

    stats = SearchStats()
    solutions = []

    nonzero_vecs = _enumerate_vectors(coeff_field, n, allow_zero=False)
    all_vecs = _enumerate_vectors(coeff_field, n, allow_zero=True)
    n_nz = len(nonzero_vecs)
    n_all = len(all_vecs)

    # Working arrays
    H_rows = np.zeros((R, 2 * nn), dtype=np.float64)
    S_rows = np.zeros((R, nn), dtype=np.int64)
    D_rows = np.zeros((R, nn * n * (n - 1)), dtype=np.int64)
    alpha_stack = np.zeros((R, n, n), dtype=np.int64)
    beta_stack = np.zeros((R, n, n), dtype=np.int64)

    # Track rank of H as we build it
    rank_so_far = [0]  # mutable via closure

    # Pre-compute the complement projector cache
    # complement_cache[depth] = (I - P_V) for V = span(H_rows[:depth])
    # Recomputed when needed

    _ops_counter = [0]  # for progress reporting

    def _should_stop():
        return (stop_flag and stop_flag()) or len(solutions) >= max_solutions

    def _check_gates_c():
        """Phase C: Gate 1 (verify) + Gate 2 + Gate 3."""
        stats.candidates_checked += 1

        H_full = H_rows[:R]
        rank_H, _ = _numerical_rank(H_full, tol)
        if rank_H != target_rank_H:
            return

        stats.gate1_passes += 1
        if callback:
            callback('gate1', alpha_stack.copy(), beta_stack.copy(), stats)

        # Gate 2: rank([H | Delta]) == rank(H)
        D_full = D_rows[:R].astype(np.float64)
        combined = np.hstack([H_full, D_full])
        rank_N, _ = _numerical_rank(combined, tol)
        if rank_N != rank_H:
            return
        stats.gate2_passes += 1

        # Gate 3: Gamma solvability
        S_full = S_rows[:R].astype(np.float64)
        g3_pass, g3_diag = check_gate3(S_full, combined, R, n, tol)
        if not g3_pass:
            return
        stats.gate3_passes += 1
        stats.solutions_found += 1

        solutions.append({
            'alpha': alpha_stack.copy(),
            'beta': beta_stack.copy(),
            'gamma': g3_diag.get('gamma'),
            'residual': g3_diag.get('residual'),
        })

    def _place_term(depth: int):
        """Place term at given depth: Phase A (active) + Phase B (non-active)."""
        if _should_stop():
            return

        term_idx, r, u = fiber_assign[depth]
        other_rows = [i for i in range(n) if i != r]
        other_cols = [j for j in range(n) if j != u]
        is_basis_term = (depth < target_rank_H)

        prev_rank = rank_so_far[0]

        # Upper-bound check: can we still reach target_rank_H?
        # We need target_rank_H - prev_rank more independent rows from R - depth remaining terms.
        remaining_terms = R - depth
        needed_rank_gain = target_rank_H - prev_rank
        if needed_rank_gain > remaining_terms:
            stats.pruned_rank_low += 1
            return
        # If we already have enough rank for H, remaining terms are all dependent
        # (they must not increase rank). Actually, remaining basis terms = max(0, target_rank_H - depth),
        # but we manage this via is_basis_term.

        # Pre-compute H_prev for rank checks in Phase B
        H_prev = H_rows[:depth] if depth > 0 else np.zeros((0, 2 * nn), dtype=np.float64)

        # Phase A: enumerate active entries
        for ai in range(n_nz):
            if _should_stop():
                return
            alpha_active_row = nonzero_vecs[ai]
            alpha_stack[term_idx, r] = alpha_active_row

            for bi in range(n_nz):
                if _should_stop():
                    return
                beta_active_col = nonzero_vecs[bi]
                beta_stack[term_idx, :, u] = beta_active_col

                # Phase B: enumerate non-active alpha rows, solve for non-active beta
                for ar0_idx in range(n_all):
                    if _should_stop():
                        return
                    alpha_stack[term_idx, other_rows[0]] = all_vecs[ar0_idx]

                    for ar1_idx in range(n_all):
                        if _should_stop():
                            return
                        alpha_stack[term_idx, other_rows[1]] = all_vecs[ar1_idx]

                        ak = alpha_stack[term_idx]
                        h_const, h_linear = _compute_h_row_decomposed(
                            ak, beta_active_col, r, u, other_cols, n)

                        if is_basis_term:
                            # Must add exactly 1 new dimension to H
                            beta_solutions = _solve_beta_for_new_dimension(
                                h_const, h_linear, H_prev, prev_rank,
                                coeff_field, tol, n, len(other_cols))
                        else:
                            # Dependent: H row must lie in span(H_prev)
                            if prev_rank == 0:
                                # No span yet — only zero H row works (degenerate)
                                # Actually this shouldn't happen: if depth >= target_rank_H > 0,
                                # we must have prev_rank > 0 already.
                                stats.pruned_no_solution += 1
                                continue

                            # Build complement projector
                            U, S_vals, Vt = np.linalg.svd(H_prev, full_matrices=False)
                            rb = int(np.sum(S_vals > tol * S_vals[0]))
                            V_basis = Vt[:rb]
                            P_V = V_basis.T @ V_basis
                            complement = np.eye(2 * nn) - P_V

                            beta_solutions = _solve_beta_nonactive(
                                h_const, h_linear, complement, coeff_field,
                                n, len(other_cols))

                        _ops_counter[0] += 1
                        if progress_callback and _ops_counter[0] % 50000 == 0:
                            progress_callback(
                                f"depth={depth}/{R}, rank={prev_rank}/{target_rank_H}, "
                                f"ops={_ops_counter[0]:,}, sols={stats.solutions_found}")

                        for beta_na in beta_solutions:
                            if _should_stop():
                                return

                            # Unpack non-active beta
                            beta_stack[term_idx, :, other_cols[0]] = beta_na[:n]
                            beta_stack[term_idx, :, other_cols[1]] = beta_na[n:]

                            bk = beta_stack[term_idx]
                            h_row = h_const + h_linear @ beta_na.astype(np.float64)
                            H_rows[term_idx] = h_row
                            S_rows[term_idx] = fast_sigma_row(ak, bk)
                            D_rows[term_idx] = fast_delta_row(ak, bk)

                            # Update rank tracking
                            if is_basis_term:
                                rank_so_far[0] = prev_rank + 1
                            # else: rank stays same (dependent term)

                            if depth == R - 1:
                                _check_gates_c()
                            else:
                                _place_term(depth + 1)

                            # Restore rank
                            rank_so_far[0] = prev_rank

    _place_term(0)
    return solutions, stats


def search(config: SearchConfig,
           callback: Optional[Callable] = None,
           progress_callback: Optional[Callable] = None,
           stop_flag: Optional[Callable] = None,
           max_solutions: int = 1) -> Tuple[list, SearchStats]:
    """Main search: iterate over all fiber partitions."""
    R = config.R
    partitions = enumerate_fiber_partitions(R, 9)

    total_stats = SearchStats()
    total_stats.partitions_total = len(partitions)
    all_solutions = []

    if progress_callback:
        progress_callback(f"R={R}: {len(partitions)} fiber partitions to search")

    for part_idx, partition in enumerate(partitions):
        if (stop_flag and stop_flag()) or len(all_solutions) >= max_solutions:
            break

        fiber_assign = _build_fiber_assignment(partition)

        if progress_callback:
            progress_callback(f"Partition {part_idx+1}/{len(partitions)}: {partition}")

        sols, stats = search_partition(
            config, partition, fiber_assign,
            callback=callback, progress_callback=progress_callback,
            stop_flag=stop_flag, max_solutions=max_solutions - len(all_solutions))

        all_solutions.extend(sols)
        total_stats.candidates_checked += stats.candidates_checked
        total_stats.gate1_passes += stats.gate1_passes
        total_stats.gate2_passes += stats.gate2_passes
        total_stats.gate3_passes += stats.gate3_passes
        total_stats.solutions_found += stats.solutions_found
        total_stats.pruned_rank_high += stats.pruned_rank_high
        total_stats.pruned_rank_low += stats.pruned_rank_low
        total_stats.pruned_no_solution += stats.pruned_no_solution
        total_stats.partitions_searched += 1

    return all_solutions, total_stats
