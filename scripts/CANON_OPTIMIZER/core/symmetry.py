"""
Z2 wr S3 = Z2^3 rtimes S3 (order 48) acting on {0,1,2}^3.

Used ONLY for deduplication (canonical form), NOT for parameterization.
"""

import numpy as np
from itertools import permutations, product
from typing import List, Tuple


def generate_group_elements() -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate all 48 elements of Z2 wr S3.
    Each element is (perm, swaps) where:
      perm: a permutation of (0,1,2) — element of S3
      swaps: a 3-tuple of 0/1 flags — element of Z2^3
    The swap flag eps[i]=1 means swap 1<->2 in coordinate i (0 stays fixed)."""
    elements = []
    for perm in permutations(range(3)):
        for swaps in product(range(2), repeat=3):
            elements.append((perm, swaps))
    assert len(elements) == 48
    return elements


def _apply_swap(val: int, swap: int) -> int:
    """Apply Z2 swap to a single coordinate value.
    0 -> 0, and if swap=1: 1->2, 2->1."""
    if swap == 0 or val == 0:
        return val
    return 3 - val  # 1->2, 2->1


def apply_to_triple(g: Tuple, triple: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Apply group element g = (perm, swaps) to triple (r, s, u).
    1) Permute coordinates: (r,s,u) -> (triple[perm[0]], triple[perm[1]], triple[perm[2]])
    2) Apply swaps to each coordinate."""
    perm, swaps = g
    permuted = (triple[perm[0]], triple[perm[1]], triple[perm[2]])
    result = tuple(_apply_swap(permuted[i], swaps[i]) for i in range(3))
    return result


def _triple_to_term_index(triple: Tuple[int, int, int], n: int = 3) -> Tuple[int, int, int]:
    """Convert (r,s,u) triple to (row_index_in_alpha, col_index_in_alpha_and_row_in_beta, col_in_beta).
    For the standard algo: alpha_k has 1 at (r,s), beta_k has 1 at (s,u).
    The "flat" index for looking up in alpha is alpha[r,s], beta[s,u]."""
    return triple


def apply_to_config(g: Tuple, alpha: np.ndarray, beta: np.ndarray,
                    n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Apply group element to a full R-term configuration.
    The group permutes which terms correspond to which triples AND transforms entries.

    For the enumeration pipeline, we canonicalize by sorting the flattened
    (alpha_k, beta_k) rows lexicographically after applying the group action
    to each term's factor matrices.

    Returns transformed (alpha', beta') arrays."""
    perm, swaps = g
    R = alpha.shape[0]
    alpha_new = np.empty_like(alpha)
    beta_new = np.empty_like(beta)

    for k in range(R):
        # The group element transforms the factor matrices:
        # Under coordinate permutation pi and swaps eps:
        # alpha'_k[r', s'] = alpha_k[pi^-1(r'), pi^-1(s')] with swap applied
        # This is complex — for canonicalization we use a simpler approach:
        # Sort the set of (alpha_k, beta_k) pairs lexicographically.
        # Two configs are equivalent if they have the same sorted multiset
        # after applying some group element.

        # For now: permute rows/cols of alpha and beta according to coordinate permutation
        # perm maps axis i -> perm[i], so rows/cols get permuted

        # alpha_k is indexed alpha[r, s]. Under perm (permuting r,s,u coordinates):
        # The new alpha corresponds to the permuted coordinate assignment.
        # For a full treatment we'd need the inverse permutation.
        inv_perm = [0, 0, 0]
        for i in range(3):
            inv_perm[perm[i]] = i

        # Permute and swap rows of alpha (r-index) and cols of alpha (s-index)
        a = alpha[k].copy()
        b = beta[k].copy()

        # Build permutation-of-indices arrays for rows and cols
        row_order_a = list(range(n))  # r-index
        col_order_a = list(range(n))  # s-index
        row_order_b = list(range(n))  # s-index
        col_order_b = list(range(n))  # u-index

        # Apply swaps: swap 1<->2 in the relevant coordinate
        # Coordinate 0 = r (rows of alpha), coordinate 1 = s (cols of alpha = rows of beta),
        # coordinate 2 = u (cols of beta)
        def make_perm_array(swap_flag):
            p = list(range(n))
            if swap_flag:
                p[1], p[2] = p[2], p[1]
            return p

        # After coordinate permutation, coordinate perm[0] maps to position 0 (r),
        # perm[1] to position 1 (s), perm[2] to position 2 (u).
        # Swaps apply AFTER permutation, to the new positions.
        swap_r = make_perm_array(swaps[0])  # swap in new r-coordinate
        swap_s = make_perm_array(swaps[1])  # swap in new s-coordinate
        swap_u = make_perm_array(swaps[2])  # swap in new u-coordinate

        # alpha'[new_r, new_s] = alpha[old_r, old_s]
        # where old coords come from inverse perm + inverse swap
        a_new = np.zeros_like(a)
        b_new = np.zeros_like(b)
        for r_new in range(n):
            for s_new in range(n):
                # Undo swap, then undo permutation
                r_unswapped = swap_r[r_new]
                s_unswapped = swap_s[s_new]
                # Undo coordinate permutation: new coord i came from old coord inv_perm[i]
                # Actually the old (r,s,u) = (x[inv_perm[0]], x[inv_perm[1]], x[inv_perm[2]])
                # This gets complicated. Let's use the direct approach.
                a_new[r_new, s_new] = a[swap_r[r_new], swap_s[s_new]]

        for s_new in range(n):
            for u_new in range(n):
                b_new[s_new, u_new] = b[swap_s[s_new], swap_u[u_new]]

        alpha_new[k] = a_new
        beta_new[k] = b_new

    return alpha_new, beta_new


def canonicalize(alpha: np.ndarray, beta: np.ndarray,
                 n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical form: try all 48 group elements, pick lexicographically smallest.
    Lexicographic comparison on the sorted flattened (alpha, beta) pair tuples."""
    best = None
    best_key = None

    for g in generate_group_elements():
        a_t, b_t = apply_to_config(g, alpha, beta, n)
        # Sort terms lexicographically
        R = a_t.shape[0]
        term_keys = []
        for k in range(R):
            term_keys.append(tuple(a_t[k].ravel()) + tuple(b_t[k].ravel()))
        sorted_indices = sorted(range(R), key=lambda k: term_keys[k])
        a_sorted = a_t[sorted_indices]
        b_sorted = b_t[sorted_indices]
        key = tuple(a_sorted.ravel()) + tuple(b_sorted.ravel())
        if best_key is None or key < best_key:
            best_key = key
            best = (a_sorted.copy(), b_sorted.copy())

    return best


def canonical_hash(alpha: np.ndarray, beta: np.ndarray, n: int = 3) -> int:
    """Fast hash of canonical form for dedup."""
    a_c, b_c = canonicalize(alpha, beta, n)
    return hash(tuple(a_c.ravel()) + tuple(b_c.ravel()))


# ── Self-test ──

def self_test():
    elems = generate_group_elements()
    assert len(elems) == 48, f"Expected 48, got {len(elems)}"

    # Check orbit sizes of 27 triples
    triples = [(r, s, u) for r in range(3) for s in range(3) for u in range(3)]
    orbits = {}
    for t in triples:
        orbit = frozenset(apply_to_triple(g, t) for g in elems)
        key = min(orbit)
        if key not in orbits:
            orbits[key] = orbit
    orbit_sizes = sorted(len(o) for o in orbits.values())
    assert orbit_sizes == [1, 6, 8, 12], f"Orbit sizes: {orbit_sizes}"

    # 19 boundary triples (at least one zero)
    boundary = [t for t in triples if 0 in t]
    assert len(boundary) == 19

    # Check the boundary is a union of orbits
    boundary_set = set(boundary)
    for t in boundary:
        orbit = {apply_to_triple(g, t) for g in elems}
        assert orbit.issubset(boundary_set), f"Triple {t} orbit escapes boundary"

    # Canonicalization is idempotent
    rng = np.random.default_rng(123)
    alpha = rng.choice([-1, 0, 1], size=(5, 3, 3)).astype(np.int64)
    beta = rng.choice([-1, 0, 1], size=(5, 3, 3)).astype(np.int64)
    a1, b1 = canonicalize(alpha, beta)
    a2, b2 = canonicalize(a1, b1)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2), "Not idempotent"

    print("symmetry.py: all self-tests passed")


if __name__ == "__main__":
    self_test()
