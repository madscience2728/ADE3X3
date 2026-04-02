"""Candidate generation: mutation, crossover, shadow re-injection."""

import numpy as np

from .config import RANK, DIM, ALGEBRAIC_LOOKUP


def _nearest_algebraic(val: float, k: int = 8) -> np.ndarray:
    """Return k nearest algebraic magnitudes to |val|."""
    absv = abs(val)
    dists = np.abs(ALGEBRAIC_LOOKUP - absv)
    idx = np.argpartition(dists, min(k, len(dists) - 1))[:k]
    return ALGEBRAIC_LOOKUP[idx[np.argsort(dists[idx])]]


def mutate_coefficient(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    rng: np.random.Generator, n_moves: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap one or more coefficients to a nearby algebraic value."""
    alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    factors = [alpha, beta, gamma]
    for _ in range(n_moves):
        fi = rng.integers(3)
        ri = rng.integers(RANK)
        di = rng.integers(DIM)
        old_val = factors[fi][ri, di]
        sign = 1.0 if old_val >= 0 else -1.0
        neighbors = _nearest_algebraic(old_val, k=8)
        # Pick a random neighbor (not the closest, for diversity)
        choice = neighbors[rng.integers(len(neighbors))]
        factors[fi][ri, di] = sign * choice
    return alpha, beta, gamma


def mutate_gaussian(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    rng: np.random.Generator, sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add Gaussian noise to a random subset of coefficients."""
    alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    factors = [alpha, beta, gamma]
    # Perturb 1-10 random coefficients (wider range for more exploration)
    n_perturb = rng.integers(1, 11)
    for _ in range(n_perturb):
        fi = rng.integers(3)
        ri = rng.integers(RANK)
        di = rng.integers(DIM)
        factors[fi][ri, di] += rng.normal(0, sigma)
    return alpha, beta, gamma


def mutate_support_flip(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip one support entry: zero → random algebraic, nonzero → zero."""
    alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    factors = [alpha, beta, gamma]
    fi = rng.integers(3)
    ri = rng.integers(RANK)
    di = rng.integers(DIM)
    if abs(factors[fi][ri, di]) < 1e-12:
        # Zero → random algebraic value with random sign
        val = ALGEBRAIC_LOOKUP[rng.integers(1, len(ALGEBRAIC_LOOKUP))]  # skip 0
        sign = rng.choice([-1.0, 1.0])
        factors[fi][ri, di] = sign * val
    else:
        factors[fi][ri, di] = 0.0
    return alpha, beta, gamma


# Ternary alphabet — the only known exact 3×3 decomposition uses {-1, 0, 1}
_TERNARY = np.array([-1.0, 0.0, 1.0])


def mutate_ternary(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    rng: np.random.Generator, n_moves: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap one or more random coefficients to a ternary {-1, 0, 1} value."""
    alpha, beta, gamma = alpha.copy(), beta.copy(), gamma.copy()
    factors = [alpha, beta, gamma]
    for _ in range(n_moves):
        fi = rng.integers(3)
        ri = rng.integers(RANK)
        di = rng.integers(DIM)
        factors[fi][ri, di] = _TERNARY[rng.integers(3)]
    return alpha, beta, gamma


def _support_vector(parent, threshold=1e-8):
    """Binary support vector: 1 where |coeff| > threshold, 0 elsewhere."""
    return np.concatenate([
        (np.abs(parent[0]).ravel() > threshold).astype(np.float32),
        (np.abs(parent[1]).ravel() > threshold).astype(np.float32),
        (np.abs(parent[2]).ravel() > threshold).astype(np.float32),
    ])


def _pick_dissimilar_pair(parents, rng):
    """Pick two parents with the most dissimilar support patterns.

    Disassortative mating: crossing structurally different parents produces
    offspring that explore the intersection of different solution basins.
    """
    n = len(parents)
    if n == 2:
        return 0, 1

    # Precompute support vectors
    svecs = [_support_vector(p) for p in parents]

    # Pick parent A randomly, then find most dissimilar B
    a = int(rng.integers(n))
    sa = svecs[a]
    best_b, best_dist = -1, -1.0
    for b in range(n):
        if b == a:
            continue
        # Hamming distance (number of differing support entries)
        dist = float(np.sum(sa != svecs[b]))
        if dist > best_dist:
            best_dist = dist
            best_b = b
    return a, best_b


def crossover(
    parent_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    parent_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform term-swap crossover: each rank-1 term comes from parent A or B."""
    a_alpha, a_beta, a_gamma = parent_a
    b_alpha, b_beta, b_gamma = parent_b
    mask = rng.random(RANK) < 0.5
    alpha = np.where(mask[:, None], a_alpha, b_alpha).copy()
    beta = np.where(mask[:, None], a_beta, b_beta).copy()
    gamma = np.where(mask[:, None], a_gamma, b_gamma).copy()
    return alpha, beta, gamma


def generate_batch(
    parents: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n: int,
    rng: np.random.Generator,
    shadow_pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    p_coeff: float = 0.60,
    p_gaussian: float = 0.25,
    p_crossover: float = 0.10,
    p_shadow: float = 0.05,
    sigma: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Generate n children with origin labels.

    Parameters
    ----------
    sigma : Gaussian mutation scale.  Island configs pass smaller values for
        exploit islands and larger for explore islands.

    Returns list of (alpha, beta, gamma, origin_str).
    """
    children = []
    for _ in range(n):
        r = rng.random()
        if r < p_coeff:
            parent = parents[rng.integers(len(parents))]
            child = mutate_coefficient(*parent, rng)
            children.append((*child, "mutate_coeff"))
        elif r < p_coeff + p_gaussian:
            parent = parents[rng.integers(len(parents))]
            child = mutate_gaussian(*parent, rng, sigma=sigma)
            children.append((*child, "mutate_gaussian"))
        elif r < p_coeff + p_gaussian + p_crossover:
            if len(parents) >= 2:
                ia, ib = _pick_dissimilar_pair(parents, rng)
                pa = (parents[ia][0], parents[ia][1], parents[ia][2])
                pb = (parents[ib][0], parents[ib][1], parents[ib][2])
                child = crossover(pa, pb, rng)
                children.append((*child, "crossover"))
            else:
                parent = parents[rng.integers(len(parents))]
                child = mutate_gaussian(*parent, rng)
                children.append((*child, "mutate_gaussian"))
        elif shadow_pool and r < p_coeff + p_gaussian + p_crossover + p_shadow:
            shadow_factors = shadow_pool[rng.integers(len(shadow_pool))]
            # Light mutation on shadow candidate
            child = mutate_gaussian(*shadow_factors, rng, sigma=0.005)
            children.append((*child, "shadow_reinject"))
        else:
            parent = parents[rng.integers(len(parents))]
            # Split: 20% ternary snap, 80% support flip
            if rng.random() < 0.2:
                child = mutate_ternary(*parent, rng, n_moves=rng.integers(1, 4))
                children.append((*child, "mutate_ternary"))
            else:
                child = mutate_support_flip(*parent, rng)
                children.append((*child, "mutate_support"))
    return children
