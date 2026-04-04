"""
Enumerate fiber partitions and valid per-fiber (alpha, beta) configurations.

Each of the R terms primarily "serves" one of 9 output fibers.
A fiber is indexed by (r, u) in {0,1,2}^2, giving 9 fibers.
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Generator
from functools import lru_cache


def _partitions_into_k_positive_parts(n: int, k: int) -> List[Tuple[int, ...]]:
    """Enumerate all compositions of n into k positive parts (order matters).
    Returns sorted tuples for canonical comparison."""
    if k == 1:
        return [(n,)]
    results = []
    for first in range(1, n - k + 2):  # leave at least 1 for each remaining part
        for rest in _partitions_into_k_positive_parts(n - first, k - 1):
            results.append((first,) + rest)
    return results


def enumerate_fiber_partitions(R: int = 19, n_fibers: int = 9) -> List[Tuple[int, ...]]:
    """Enumerate all SORTED partitions of R terms into n_fibers fibers.
    Each fiber gets >= 1 term. Returns canonical (sorted descending) tuples,
    deduplicated — since fibers are interchangeable under permutation,
    we only keep one representative per multiset."""
    raw = _partitions_into_k_positive_parts(R, n_fibers)
    # Canonicalize: sort each partition descending, deduplicate
    canonical = set()
    for p in raw:
        canonical.add(tuple(sorted(p, reverse=True)))
    return sorted(canonical, reverse=True)


def fiber_sigma_values(alpha_row: np.ndarray, beta_col: np.ndarray) -> int:
    """Sigma contribution: dot(alpha_row, beta_col). All length-3 int vectors."""
    return int(alpha_row[0] * beta_col[0] + alpha_row[1] * beta_col[1]
               + alpha_row[2] * beta_col[2])


def enumerate_nonzero_vectors(coeff_field: List[int], n: int = 3) -> List[Tuple[int, ...]]:
    """All n-vectors over coeff_field that are not all-zero."""
    vecs = []
    for v in product(coeff_field, repeat=n):
        if any(x != 0 for x in v):
            vecs.append(v)
    return vecs


def enumerate_fiber_term_pairs(coeff_field: List[int],
                                n: int = 3) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """For a single fiber (r,u), enumerate all (alpha_row_r, beta_col_u) pairs.
    Returns list of (alpha_row, beta_col, sigma_value).
    Excludes all-zero rows/cols (degenerate)."""
    vecs = enumerate_nonzero_vectors(coeff_field, n)
    pairs = []
    for a in vecs:
        for b in vecs:
            a_arr = np.array(a, dtype=np.int64)
            b_arr = np.array(b, dtype=np.int64)
            sigma = fiber_sigma_values(a_arr, b_arr)
            pairs.append((a_arr, b_arr, sigma))
    return pairs


def valid_sigma_tuples_for_fiber(n_terms: int, coeff_field: List[int],
                                  n: int = 3) -> List[List[Tuple]]:
    """For a fiber with n_terms terms, enumerate all n_terms-tuples of
    (alpha_row, beta_col, sigma) that could potentially satisfy
    the diagonal coupling: sum_k gamma_k * sigma_k = 3.

    We don't yet know gamma, but sigma values must allow integer gamma
    (or rational) satisfying the coupling. For now, just enumerate all
    n_terms-tuples of non-degenerate pairs — the coupling check happens
    at the full assembly level.

    Returns list of n_terms-length lists of (alpha_row, beta_col, sigma)."""
    pairs = enumerate_fiber_term_pairs(coeff_field, n)
    if n_terms == 1:
        return [[p] for p in pairs]
    elif n_terms == 2:
        return [[p1, p2] for p1 in pairs for p2 in pairs]
    else:
        # For n_terms >= 3, use itertools.product
        return [list(combo) for combo in product(pairs, repeat=n_terms)]


def count_fiber_partitions(R: int = 19, n_fibers: int = 9) -> int:
    """Count the number of canonical fiber partitions."""
    return len(enumerate_fiber_partitions(R, n_fibers))


# ── Self-test ──

def self_test():
    # Count partitions of 19 into 9 positive parts
    parts = enumerate_fiber_partitions(19, 9)
    print(f"Fiber partitions of 19 into 9 positive parts: {len(parts)}")
    for p in parts[:10]:
        assert sum(p) == 19
        assert len(p) == 9
        assert all(x >= 1 for x in p)
        print(f"  {p}")

    # Count term pairs over {-1,0,1}
    pairs = enumerate_fiber_term_pairs([-1, 0, 1])
    print(f"Non-degenerate (alpha_row, beta_col) pairs over {{-1,0,1}}: {len(pairs)}")

    # Sigma values distribution
    from collections import Counter
    sigma_dist = Counter(p[2] for p in pairs)
    print(f"Sigma value distribution: {dict(sorted(sigma_dist.items()))}")

    print("fiber.py: all self-tests passed")


if __name__ == "__main__":
    self_test()
