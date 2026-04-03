"""Basin fingerprinting: identify and track structurally distinct basins.

A basin fingerprint captures the *qualitative structure* of a solution —
which coefficients are positive, negative, or zero, and the rough magnitude
pattern. Two solutions in the same basin will have nearly identical
fingerprints even if their exact coefficients differ slightly.

The key insight: support_hash (from blob.py) is too fine-grained — it
changes with tiny coefficient perturbations. Basin fingerprints are
deliberately coarse, grouping solutions that share the same structural
skeleton.

Fingerprint levels (coarse → fine):
  1. sign_pattern: signs of all 3×R×9 coefficients ({+, -, 0})
  2. magnitude_pattern: quantized to {0, small, medium, large}
  3. term_topology: which (i,j) pairs have nonzero outer-product overlap

The basin registry tracks:
  - Known basin fingerprints
  - Best fitness per basin
  - Basin discovery time
  - Population count per basin
  - Novelty score (inverse of population)
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import RANK, DIM


# ── Fingerprint computation ──────────────────────────────────────────

def _sign_vector(arr: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Map array to ternary sign vector: -1, 0, +1."""
    signs = np.zeros_like(arr, dtype=np.int8)
    signs[arr > threshold] = 1
    signs[arr < -threshold] = -1
    return signs


def _magnitude_class(arr: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Quantize |arr| into 4 classes: 0=zero, 1=small(<0.1), 2=medium(<1.0), 3=large."""
    absv = np.abs(arr)
    classes = np.zeros_like(arr, dtype=np.int8)
    classes[absv > threshold] = 1
    classes[absv > 0.1] = 2
    classes[absv > 1.0] = 3
    return classes


def compute_sign_fingerprint(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    threshold: float = 1e-6,
) -> str:
    """Coarsest fingerprint: ternary sign pattern of all coefficients.

    Returns a 16-char hex hash. Solutions in the same sign basin will
    have identical fingerprints.
    """
    # Sort terms by their sign pattern to make fingerprint permutation-invariant
    # across term reordering
    signs = []
    for k in range(RANK):
        s = np.concatenate([
            _sign_vector(alpha[k:k+1], threshold).ravel(),
            _sign_vector(beta[k:k+1], threshold).ravel(),
            _sign_vector(gamma[k:k+1], threshold).ravel(),
        ])
        signs.append(tuple(s.tolist()))
    # Canonical ordering: sort the term signatures
    signs.sort()
    raw = str(signs).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def compute_magnitude_fingerprint(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    threshold: float = 1e-6,
) -> str:
    """Medium fingerprint: sign + magnitude class per coefficient."""
    patterns = []
    for k in range(RANK):
        s = np.concatenate([
            _sign_vector(alpha[k:k+1], threshold).ravel(),
            _sign_vector(beta[k:k+1], threshold).ravel(),
            _sign_vector(gamma[k:k+1], threshold).ravel(),
        ])
        m = np.concatenate([
            _magnitude_class(alpha[k:k+1], threshold).ravel(),
            _magnitude_class(beta[k:k+1], threshold).ravel(),
            _magnitude_class(gamma[k:k+1], threshold).ravel(),
        ])
        # Combine sign and magnitude into a single descriptor per term
        combined = tuple((int(si), int(mi)) for si, mi in zip(s, m))
        patterns.append(combined)
    patterns.sort()
    raw = str(patterns).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def compute_topology_fingerprint(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    threshold: float = 1e-6,
) -> str:
    """Fine fingerprint: per-term support overlaps between factor pairs.

    Captures which (a,b), (a,c), (b,c) index pairs have joint nonzero support,
    revealing the combinatorial skeleton of the decomposition.
    """
    descriptors = []
    for k in range(RANK):
        a_sup = frozenset(int(j) for j in range(DIM) if abs(alpha[k, j]) > threshold)
        b_sup = frozenset(int(j) for j in range(DIM) if abs(beta[k, j]) > threshold)
        g_sup = frozenset(int(j) for j in range(DIM) if abs(gamma[k, j]) > threshold)
        desc = (tuple(sorted(a_sup)), tuple(sorted(b_sup)), tuple(sorted(g_sup)))
        descriptors.append(desc)
    descriptors.sort()
    raw = str(descriptors).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class BasinFingerprint:
    """Multi-resolution fingerprint for a candidate solution."""
    sign: str        # coarsest: ternary sign pattern
    magnitude: str   # medium: sign + magnitude class
    topology: str    # finest: support overlaps

    @property
    def coarse_key(self) -> str:
        """Use sign fingerprint for basin grouping."""
        return self.sign

    @property
    def fine_key(self) -> str:
        """Use topology for sub-basin distinction."""
        return f"{self.sign}:{self.topology}"


def compute_basin_fingerprint(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
    threshold: float = 1e-6,
) -> BasinFingerprint:
    """Compute multi-resolution basin fingerprint."""
    return BasinFingerprint(
        sign=compute_sign_fingerprint(alpha, beta, gamma, threshold),
        magnitude=compute_magnitude_fingerprint(alpha, beta, gamma, threshold),
        topology=compute_topology_fingerprint(alpha, beta, gamma, threshold),
    )


# ── Basin Registry ───────────────────────────────────────────────────

@dataclass
class BasinRecord:
    """Track a discovered basin."""
    fingerprint: str              # coarse key
    best_fitness: float           # best fitness seen in this basin
    discovery_time: float = 0.0   # wall-clock time of first sighting
    generation_discovered: int = 0
    population: int = 0           # current candidates in this basin
    total_seen: int = 0           # lifetime candidates seen
    last_improved_gen: int = 0    # generation of last fitness improvement
    representative_blob: Optional[bytes] = None  # best candidate blob


class BasinRegistry:
    """Thread-safe registry of discovered basins.

    Tracks basin novelty: basins seen fewer times get higher novelty scores,
    incentivizing the search to explore uncharted territory.
    """

    def __init__(self):
        self._basins: dict[str, BasinRecord] = {}
        self._candidate_basin: dict[int, str] = {}  # candidate_id → basin key
        import threading
        self._lock = threading.Lock()

    @property
    def n_basins(self) -> int:
        with self._lock:
            return len(self._basins)

    def register(
        self,
        candidate_id: int,
        fingerprint: BasinFingerprint,
        fitness: float,
        generation: int = 0,
        wall_time: float = 0.0,
        blob: Optional[bytes] = None,
    ) -> tuple[bool, float]:
        """Register a candidate's basin. Returns (is_novel_basin, novelty_score).

        is_novel_basin: True if this is a never-before-seen basin fingerprint.
        novelty_score: Higher for rarer basins (1/sqrt(population)).
        """
        key = fingerprint.coarse_key
        with self._lock:
            is_novel = key not in self._basins
            if is_novel:
                self._basins[key] = BasinRecord(
                    fingerprint=key,
                    best_fitness=fitness,
                    discovery_time=wall_time,
                    generation_discovered=generation,
                    population=1,
                    total_seen=1,
                    last_improved_gen=generation,
                    representative_blob=blob,
                )
            else:
                rec = self._basins[key]
                rec.population += 1
                rec.total_seen += 1
                if fitness < rec.best_fitness:
                    rec.best_fitness = fitness
                    rec.last_improved_gen = generation
                    if blob is not None:
                        rec.representative_blob = blob

            self._candidate_basin[candidate_id] = key

            # Novelty score: inverse sqrt of population (rarer = higher)
            pop = self._basins[key].population
            novelty = 1.0 / max(1.0, pop ** 0.5)
            return is_novel, novelty

    def unregister(self, candidate_id: int) -> None:
        """Remove a candidate from its basin count (e.g., on pruning)."""
        with self._lock:
            key = self._candidate_basin.pop(candidate_id, None)
            if key and key in self._basins:
                self._basins[key].population = max(0, self._basins[key].population - 1)

    def get_novelty_score(self, fingerprint: BasinFingerprint) -> float:
        """Get novelty score for a fingerprint without registering."""
        key = fingerprint.coarse_key
        with self._lock:
            if key not in self._basins:
                return 1.0  # completely novel
            pop = self._basins[key].population
            return 1.0 / max(1.0, pop ** 0.5)

    def get_basin_stats(self) -> list[dict]:
        """Return summary stats for all known basins, sorted by best fitness."""
        with self._lock:
            stats = []
            for key, rec in self._basins.items():
                stats.append({
                    "fingerprint": key,
                    "best_fitness": rec.best_fitness,
                    "population": rec.population,
                    "total_seen": rec.total_seen,
                    "generation_discovered": rec.generation_discovered,
                    "last_improved_gen": rec.last_improved_gen,
                    "stale_gens": 0,  # filled by caller
                })
            stats.sort(key=lambda s: s["best_fitness"])
            return stats

    def get_underexplored_basins(self, min_fitness: float = 0.5, max_pop: int = 5) -> list[BasinRecord]:
        """Find promising but underexplored basins — good targets for exploitation."""
        with self._lock:
            return [
                rec for rec in self._basins.values()
                if rec.best_fitness < min_fitness and rec.population <= max_pop
            ]

    def get_stale_basins(self, current_gen: int, stale_threshold: int = 50) -> list[BasinRecord]:
        """Find basins that haven't improved in many generations."""
        with self._lock:
            return [
                rec for rec in self._basins.values()
                if (current_gen - rec.last_improved_gen) > stale_threshold
            ]

    def get_top_basins(self, k: int = 10) -> list[BasinRecord]:
        """Return k basins with best fitness."""
        with self._lock:
            return sorted(self._basins.values(), key=lambda r: r.best_fitness)[:k]

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        with self._lock:
            return {
                "n_basins": len(self._basins),
                "top_10": [
                    {"fp": r.fingerprint, "fit": r.best_fitness, "pop": r.population,
                     "total": r.total_seen, "disc_gen": r.generation_discovered}
                    for r in sorted(self._basins.values(), key=lambda r: r.best_fitness)[:10]
                ],
            }
