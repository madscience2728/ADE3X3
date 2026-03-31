"""
ade3x3_step84_canonical_constraints.py

Canonical constraint infrastructure for the step84 metaheuristic search.
Implements three constraint layers:

1. FIBER STRUCTURE: Live/dead awareness for the 3x3 matrix multiplication tensor.
   Every X atom X[r,s|t,u] is live iff s==t. Of the 81 X atoms, 27 are live
   and 54 are dead. For a rank-1 term (alpha, beta, gamma), the product
   alpha[a] * beta[b] * gamma[c] contributes to tensor entry T[a,b,c] where
   a = 3*r+s, b = 3*t+u, c = 3*r'+u'. A decomposition must reproduce T exactly,
   meaning the live entries sum correctly and all dead entries cancel to zero.

2. S3 x S3 x S3 SYMMETRY: The 216-element group acts on factor indices as:
     pi_rA on rows: A[r,s] -> A[pi_rA(r),pi_shared(s)]
     pi_shared on middle: B[t,u] -> B[pi_shared(t),pi_cB(u)]
     pi_cB on columns: C[r,u] -> C[pi_rA(r),pi_cB(u)]
   This maps equivalent decompositions to a canonical representative.

3. FIBER-MODE VALIDATION: Computes sigma, eta, delta matrices for structural
   diagnostics (conservation law, delta-containment) as soft fitness penalties.
"""

from __future__ import annotations

import numpy as np
from itertools import permutations

# ---------------------------------------------------------------------------
# 1. FIBER STRUCTURE — precomputed live/dead masks and index tables
# ---------------------------------------------------------------------------

def _build_live_mask() -> np.ndarray:
    """Build 9x9 boolean mask: live_ab[a,b] = True iff T[a,b,:] has a live entry."""
    mask = np.zeros((9, 9), dtype=bool)
    for r in range(3):
        for s in range(3):
            a = 3 * r + s
            b = 3 * s  # t must equal s for live; b = 3*s + u for any u
            for u in range(3):
                mask[a, 3 * s + u] = True
    return mask


def _build_live_ab_pairs() -> np.ndarray:
    """Return (N, 2) array of (a_idx, b_idx) pairs that contribute to live entries."""
    pairs = []
    for r in range(3):
        for s in range(3):
            a = 3 * r + s
            for u in range(3):
                b = 3 * s + u
                pairs.append((a, b))
    return np.array(pairs, dtype=np.intp)


def _build_fiber_index_table() -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map (a_idx, b_idx) -> list of (r, u) fiber memberships for live entries.

    For live atom X[r,s|s,u]: a=3r+s, b=3s+u -> fiber (r,u) = output C[r,u].
    """
    table: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(3):
        for s in range(3):
            a = 3 * r + s
            for u in range(3):
                b = 3 * s + u
                table.setdefault((a, b), []).append((r, u))
    return table


# Precomputed at import time
LIVE_AB_MASK = _build_live_mask()
LIVE_AB_PAIRS = _build_live_ab_pairs()
FIBER_INDEX_TABLE = _build_fiber_index_table()

# For each a_idx (0..8), which b_idx values form live pairs?
LIVE_B_FOR_A: dict[int, list[int]] = {}
for _a, _b in LIVE_AB_PAIRS:
    LIVE_B_FOR_A.setdefault(int(_a), []).append(int(_b))

# For each b_idx (0..8), which a_idx values form live pairs?
LIVE_A_FOR_B: dict[int, list[int]] = {}
for _a, _b in LIVE_AB_PAIRS:
    LIVE_A_FOR_B.setdefault(int(_b), []).append(int(_a))

# Dead entry mask: complement of live
DEAD_AB_MASK = ~LIVE_AB_MASK


def dead_leakage_score(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Compute total squared mass on dead tensor entries for one rank-1 term.

    alpha: (9,) factor vector
    beta: (9,) factor vector
    Returns sum of (alpha[a] * beta[b])^2 for all dead (a,b) pairs.
    """
    outer = np.outer(alpha, beta)  # (9, 9)
    return float(np.sum(outer[DEAD_AB_MASK] ** 2))


def dead_leakage_batch(alpha_matrix: np.ndarray, beta_matrix: np.ndarray) -> np.ndarray:
    """Compute dead leakage for a batch of R terms.

    alpha_matrix: (R, 9) factor matrix
    beta_matrix: (R, 9) factor matrix
    Returns (R,) array of dead leakage per term.
    """
    # outer product per term: (R, 9, 9) — use einsum for efficiency
    outers = np.einsum('ri,rj->rij', alpha_matrix, beta_matrix)
    return np.sum(outers[:, DEAD_AB_MASK] ** 2, axis=1)


def total_dead_leakage(alpha_matrix: np.ndarray, beta_matrix: np.ndarray) -> float:
    """Total dead leakage across all terms."""
    return float(np.sum(dead_leakage_batch(alpha_matrix, beta_matrix)))


# Fiber-biased support selection ------------------------------------------

def live_compatible_beta_positions(alpha_support: tuple[int, ...]) -> list[int]:
    """Return beta index positions that form live (a,b) pairs with given alpha support.

    For alpha position a = 3r+s, live beta positions are b = 3s+u for u in {0,1,2}.
    """
    s_values = {a % 3 for a in alpha_support}
    return sorted({3 * s + u for s in s_values for u in range(3)})


def live_compatible_gamma_positions(
    alpha_support: tuple[int, ...], beta_support: tuple[int, ...],
) -> list[int]:
    """Return gamma positions that complete live triples with given alpha and beta support.

    For live triple (a,b,c): a=3r+s, b=3s+u, c=3r+u.
    Gamma position c is live-compatible when there exist a in alpha_support
    and b in beta_support with a%3 == b//3.
    """
    pairs: set[tuple[int, int]] = set()
    for a in alpha_support:
        s = a % 3
        r = a // 3
        for b in beta_support:
            if b // 3 == s:
                u = b % 3
                pairs.add((r, u))
    return sorted({3 * r + u for r, u in pairs})


def fiber_biased_support(
    rng, count: int, preferred: list[int], bias: float = 0.7,
) -> tuple[int, ...]:
    """Choose `count` support positions from {0..8}, biased toward `preferred` set.

    Each position is drawn from `preferred` with probability `bias` (if candidates
    remain), else drawn uniformly from the complement. Falls back to uniform
    selection when preferred is empty.
    """
    all_positions = list(range(9))
    if not preferred or count >= 9:
        chosen = rng.choice(np.arange(9), size=min(count, 9), replace=False).tolist()
        return tuple(sorted(int(i) for i in chosen))
    chosen: set[int] = set()
    preferred_set = set(preferred)
    for _ in range(count):
        remaining_pref = [p for p in preferred if p not in chosen]
        remaining_other = [p for p in all_positions if p not in chosen and p not in preferred_set]
        remaining_all = [p for p in all_positions if p not in chosen]
        if not remaining_all:
            break
        if remaining_pref and rng.random() < bias:
            chosen.add(int(rng.choice(np.array(remaining_pref, dtype=np.int64))))
        elif remaining_other:
            chosen.add(int(rng.choice(np.array(remaining_other, dtype=np.int64))))
        elif remaining_pref:
            chosen.add(int(rng.choice(np.array(remaining_pref, dtype=np.int64))))
        else:
            chosen.add(int(rng.choice(np.array(remaining_all, dtype=np.int64))))
    return tuple(sorted(chosen))


# ---------------------------------------------------------------------------
# 2. S3 x S3 x S3 SYMMETRY — orbit canonicalization
# ---------------------------------------------------------------------------

# All 6 permutations of {0,1,2}
_S3_ELEMENTS = list(permutations(range(3)))


def _apply_permutation_to_factor(factor_9: np.ndarray, perm_row: tuple[int, ...], perm_col: tuple[int, ...]) -> np.ndarray:
    """Apply (perm_row, perm_col) to a 9-element factor vector.

    factor_9[3*r + s] -> new_factor[3*perm_row[r] + perm_col[s]]
    """
    result = np.zeros(9, dtype=factor_9.dtype)
    for r in range(3):
        for s in range(3):
            result[3 * perm_row[r] + perm_col[s]] = factor_9[3 * r + s]
    return result


def _apply_group_element_to_term(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    pi_rA: tuple[int, ...], pi_shared: tuple[int, ...], pi_cB: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply (pi_rA, pi_shared, pi_cB) to a single rank-1 term.

    A[r,s] -> A[pi_rA(r), pi_shared(s)]:  alpha uses (pi_rA, pi_shared)
    B[t,u] -> B[pi_shared(t), pi_cB(u)]:  beta uses (pi_shared, pi_cB)
    C[r,u] -> C[pi_rA(r), pi_cB(u)]:      gamma uses (pi_rA, pi_cB)
    """
    new_alpha = _apply_permutation_to_factor(alpha, pi_rA, pi_shared)
    new_beta = _apply_permutation_to_factor(beta, pi_shared, pi_cB)
    new_gamma = _apply_permutation_to_factor(gamma, pi_rA, pi_cB)
    return new_alpha, new_beta, new_gamma


def canonicalize_term_under_s3(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Find the canonical group representative for a single rank-1 term.

    Returns (canon_alpha, canon_beta, canon_gamma, pi_rA, pi_shared, pi_cB)
    where the canonical form is the lexicographically smallest under the group action.
    """
    best_key = None
    best = None
    best_perms = None
    for pi_rA in _S3_ELEMENTS:
        for pi_shared in _S3_ELEMENTS:
            for pi_cB in _S3_ELEMENTS:
                new_a, new_b, new_g = _apply_group_element_to_term(
                    alpha, beta, gamma, pi_rA, pi_shared, pi_cB
                )
                # Key: support pattern, then values
                key = (
                    tuple(int(i) for i in np.nonzero(np.abs(new_a) > 1e-12)[0]),
                    tuple(int(i) for i in np.nonzero(np.abs(new_b) > 1e-12)[0]),
                    tuple(int(i) for i in np.nonzero(np.abs(new_g) > 1e-12)[0]),
                    tuple(float(v) for v in new_a),
                    tuple(float(v) for v in new_b),
                    tuple(float(v) for v in new_g),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (new_a.copy(), new_b.copy(), new_g.copy())
                    best_perms = (pi_rA, pi_shared, pi_cB)
    return (*best, *best_perms)


def canonicalize_support_under_s3(
    alpha_support: tuple[int, ...], beta_support: tuple[int, ...], gamma_support: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Canonicalize a support pattern under S3 x S3 x S3 (support only, no values).

    Returns the lexicographically smallest (alpha_support, beta_support, gamma_support).
    """
    best = None
    for pi_rA in _S3_ELEMENTS:
        for pi_shared in _S3_ELEMENTS:
            for pi_cB in _S3_ELEMENTS:
                new_a_support = tuple(sorted(3 * pi_rA[i // 3] + pi_shared[i % 3] for i in alpha_support))
                new_b_support = tuple(sorted(3 * pi_shared[i // 3] + pi_cB[i % 3] for i in beta_support))
                new_g_support = tuple(sorted(3 * pi_rA[i // 3] + pi_cB[i % 3] for i in gamma_support))
                candidate = (new_a_support, new_b_support, new_g_support)
                if best is None or candidate < best:
                    best = candidate
    return best


def canonical_support_hash(terms_supports: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]) -> str:
    """Compute a canonical hash for a full individual's support pattern.

    Applies the SAME group element to ALL terms simultaneously (correct orbit
    comparison), then sorts transformed terms and picks the lex-smallest result.
    """
    best = None
    for pi_rA in _S3_ELEMENTS:
        for pi_shared in _S3_ELEMENTS:
            for pi_cB in _S3_ELEMENTS:
                transformed = []
                for alpha_sup, beta_sup, gamma_sup in terms_supports:
                    new_a = tuple(sorted(3 * pi_rA[i // 3] + pi_shared[i % 3] for i in alpha_sup))
                    new_b = tuple(sorted(3 * pi_shared[i // 3] + pi_cB[i % 3] for i in beta_sup))
                    new_g = tuple(sorted(3 * pi_rA[i // 3] + pi_cB[i % 3] for i in gamma_sup))
                    transformed.append((new_a, new_b, new_g))
                transformed.sort()
                if best is None or transformed < best:
                    best = transformed
    return ";".join(f"{a}|{b}|{g}" for a, b, g in best)


# ---------------------------------------------------------------------------
# 3. FIBER-MODE VALIDATION — structural diagnostics
# ---------------------------------------------------------------------------

def fiber_mode_decomposition(
    alpha_matrix: np.ndarray, beta_matrix: np.ndarray, gamma_matrix: np.ndarray,
) -> dict:
    """Compute the fiber-mode matrices for an R-term decomposition.

    alpha_matrix: (R, 9) — alpha factors
    beta_matrix: (R, 9) — beta factors
    gamma_matrix: (R, 9) — gamma factors (the C-factor)

    Returns dict with:
        'Gamma': (9, R) output weight matrix
        'Sigma': (R, 9) fiber-sum matrix
        'H1': (R, 9) first anisotropy matrix
        'H2': (R, 9) second anisotropy matrix
        'H': (R, 18) combined anisotropy
        'rank_H': int
        'rank_Gamma': int
        'eta_nullity': int
        'conservation_sum': int (R + eta_nullity, should be 27)
        'delta_containment': bool
        'dead_leakage': float
    """
    R = alpha_matrix.shape[0]

    # Gamma: (9, R) — output weights, gamma_k indexed by c = 3*r + u
    Gamma = gamma_matrix.T  # (9, R)

    # Sigma: (R, 9) — fiber sums sigma_k[r,u] = sum_s alpha_k[3r+s] * beta_k[3s+u]
    Sigma = np.zeros((R, 9), dtype=np.float64)
    for r in range(3):
        for u in range(3):
            c_idx = 3 * r + u
            for s in range(3):
                Sigma[:, c_idx] += alpha_matrix[:, 3 * r + s] * beta_matrix[:, 3 * s + u]

    # H1: eta1_k[r,u] = alpha_k[3r+0]*beta_k[0+u] - alpha_k[3r+1]*beta_k[3+u]
    H1 = np.zeros((R, 9), dtype=np.float64)
    for r in range(3):
        for u in range(3):
            c_idx = 3 * r + u
            H1[:, c_idx] = alpha_matrix[:, 3 * r + 0] * beta_matrix[:, 0 + u] - alpha_matrix[:, 3 * r + 1] * beta_matrix[:, 3 + u]

    # H2: eta2_k[r,u] = alpha_k[3r+1]*beta_k[3+u] - alpha_k[3r+2]*beta_k[6+u]
    H2 = np.zeros((R, 9), dtype=np.float64)
    for r in range(3):
        for u in range(3):
            c_idx = 3 * r + u
            H2[:, c_idx] = alpha_matrix[:, 3 * r + 1] * beta_matrix[:, 3 + u] - alpha_matrix[:, 3 * r + 2] * beta_matrix[:, 6 + u]

    H = np.hstack([H1, H2])  # (R, 18)

    # Delta: (R, 54) — dead-X coordinates
    Delta = np.zeros((R, 54), dtype=np.float64)
    col = 0
    for r in range(3):
        for s in range(3):
            for t in range(3):
                if s == t:
                    continue
                for u in range(3):
                    Delta[:, col] = alpha_matrix[:, 3 * r + s] * beta_matrix[:, 3 * t + u]
                    col += 1

    rank_Gamma = int(np.linalg.matrix_rank(Gamma, tol=1e-10))
    rank_H = int(np.linalg.matrix_rank(H, tol=1e-10))
    eta_nullity = 18 - rank_H
    conservation_sum = R + eta_nullity

    # Delta-containment: col(Delta) subset span(H)?
    # Check by projecting Delta onto ker(H^T) — if residual is zero, contained.
    if rank_H > 0:
        U_H, s_H, Vt_H = np.linalg.svd(H.T, full_matrices=True)
        # Null space of H^T is U_H[:, rank_H:]
        null_H = U_H[:, rank_H:]  # (R, R - rank_H)
        if null_H.shape[1] > 0:
            projection_residual = null_H.T @ Delta  # should be zero if contained
            delta_containment = bool(np.max(np.abs(projection_residual)) < 1e-8)
        else:
            delta_containment = True  # H has full row rank, trivially contained
    else:
        delta_containment = False

    dead_leak = total_dead_leakage(alpha_matrix, beta_matrix)

    return {
        'Gamma': Gamma,
        'Sigma': Sigma,
        'H1': H1,
        'H2': H2,
        'H': H,
        'Delta': Delta,
        'rank_Gamma': rank_Gamma,
        'rank_H': rank_H,
        'eta_nullity': eta_nullity,
        'conservation_sum': conservation_sum,
        'delta_containment': delta_containment,
        'dead_leakage': dead_leak,
    }


def structural_penalty(
    alpha_matrix: np.ndarray, beta_matrix: np.ndarray, gamma_matrix: np.ndarray,
    weight_dead_leakage: float = 0.01,
    weight_conservation: float = 0.001,
) -> float:
    """Compute a structural penalty for use as a soft fitness modifier.

    Higher penalty = further from canonical structure. Returns a non-negative float.
    This is intentionally lightweight — the full fiber_mode_decomposition is
    only called for promising candidates, not every evaluation.
    """
    # Fast dead-leakage penalty (no SVD needed)
    dead_leak = total_dead_leakage(alpha_matrix, beta_matrix)
    return weight_dead_leakage * dead_leak
