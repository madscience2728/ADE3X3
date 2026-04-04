from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import numpy as np

from term_db import TermDB


@dataclass
class BasisResult:
    rank: int
    indices: list[int]
    V_basis: np.ndarray
    V_perp: np.ndarray
    sigma_basis: np.ndarray
    dependent_estimate: int


@dataclass
class BasisBuildStats:
    explored_nodes: int = 0
    pruned_dependent: int = 0
    pruned_sigma: int = 0
    pruned_budget: int = 0
    pruned_norm: int = 0
    built_bases: int = 0
    seeds_processed: int = 0
    seeds_total: int = 0
    depth1_total: int = 0      # candidates at depth 1 for current seed
    depth1_done: int = 0       # how many depth-1 candidates processed


def _row_rank(rows: np.ndarray, tol: float) -> int:
    if rows.size == 0:
        return 0
    return int(np.linalg.matrix_rank(rows.astype(np.float64), tol=tol))


def _integer_nullspace(V_basis: np.ndarray) -> np.ndarray:
    return TermDB._integer_nullspace(np.asarray(V_basis, dtype=np.int64))


def _representative_record_index(db, unique_idx: int) -> int:
    if hasattr(db, "unique_record_index"):
        return int(db.unique_record_index[unique_idx])
    return int(np.flatnonzero(db.H_dedup_map == unique_idx)[0])


def quick_dependent_count(db, V_partial: np.ndarray, sample_size: int, rng: np.random.Generator) -> int:
    V_perp = _integer_nullspace(V_partial)
    if V_perp.shape[0] == 0:
        return int(db.N_PAIRS)

    sample_size = min(sample_size, int(db.N_PAIRS))
    sample_idx = rng.choice(db.N_PAIRS, size=sample_size, replace=False)
    H_sample = np.asarray(db.H[sample_idx], dtype=np.int16)
    proj = H_sample @ np.asarray(V_perp, dtype=np.int16).T
    hit_rate = float(np.mean(np.all(proj == 0, axis=1)))
    return int(hit_rate * db.N_PAIRS)


def _batch_find_independent(V_partial_f64: np.ndarray, candidates: np.ndarray,
                            current_rank: int, batch_size: int = 4096) -> np.ndarray:
    """
    Vectorized independence filter: find candidate rows that increase rank of V_partial.
    Uses QR-based projection residual — if ||candidate - Q Q^T candidate|| > tol, it's independent.
    Returns indices into `candidates` array that are independent of V_partial.
    """
    if V_partial_f64.shape[0] == 0:
        # Everything with nonzero norm is independent
        norms = np.linalg.norm(candidates.astype(np.float64), axis=1)
        return np.flatnonzero(norms > 0.5)

    Q, _ = np.linalg.qr(V_partial_f64.T, mode='reduced')  # (18, d)
    result = []
    n = candidates.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = candidates[start:end].astype(np.float64)  # (B, 18)
        projected = (batch @ Q) @ Q.T  # (B, 18)
        residuals = np.linalg.norm(batch - projected, axis=1)
        hits = np.flatnonzero(residuals > 1e-8)
        result.append(hits + start)
    if result:
        return np.concatenate(result)
    return np.empty(0, dtype=np.int64)


class BasisBuilder:
    def __init__(self, db, config, rank: int, rng: Optional[np.random.Generator] = None):
        self.db = db
        self.config = config
        self.rank = rank
        self.target_rank = config.target_rank_H(rank)
        self.rng = rng or np.random.default_rng(0)
        self.stats = BasisBuildStats()
        # Pre-compute norms once for all unique H-rows
        self._unique_H_i64 = np.asarray(db.unique_H, dtype=np.int64)
        self._norms = np.linalg.norm(self._unique_H_i64.astype(np.float64), axis=1)

    def _seed_unique_indices(self) -> np.ndarray:
        mask = self._norms >= float(self.config.basis_norm_min)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return np.arange(min(self.config.basis_seed_count, self._unique_H_i64.shape[0]), dtype=np.int64)

        scores = self._norms[candidates]
        order = np.argsort(scores)[::-1]
        return np.asarray(candidates[order[: min(self.config.basis_seed_count, candidates.size)]], dtype=np.int64)

    def build_bases(self) -> Generator[BasisResult, None, None]:
        """Greedy random sampling: pick independent rows until rank is reached.

        Generates up to max_bases_per_rank bases, each in O(target_rank * pool_size) time.
        Falls back to DFS if config.basis_mode == 'dfs'.
        """
        mode = getattr(self.config, 'basis_mode', 'greedy')
        if mode == 'dfs':
            yield from self._build_bases_dfs()
            return

        target_d = self.target_rank
        unique_H = self._unique_H_i64
        norms = self._norms
        n_unique = unique_H.shape[0]

        # Pool: all rows with sufficient norm
        pool_mask = norms >= float(self.config.basis_norm_min)
        pool_indices = np.flatnonzero(pool_mask)
        if pool_indices.size < target_d:
            return

        self.stats.seeds_total = self.config.max_bases_per_rank
        self.stats.seeds_processed = 0
        built = 0
        seen_keys: set[tuple[int, ...]] = set()
        max_attempts = self.config.max_bases_per_rank * 10  # avoid infinite loop
        attempts = 0

        while built < self.config.max_bases_per_rank and attempts < max_attempts:
            attempts += 1
            self.stats.seeds_processed = attempts
            self.stats.depth1_total = target_d
            self.stats.depth1_done = 0

            # Greedy: randomly pick rows, keep if independent
            order = self.rng.permutation(pool_indices.size)
            chosen: list[int] = []
            rows: list[np.ndarray] = []
            V_f64 = np.empty((0, unique_H.shape[1]), dtype=np.float64)

            for j in range(order.size):
                if len(chosen) >= target_d:
                    break
                idx = int(pool_indices[order[j]])
                row = unique_H[idx].astype(np.float64)

                # Quick independence check via QR
                if V_f64.shape[0] > 0:
                    Q, _ = np.linalg.qr(V_f64.T, mode='reduced')
                    residual = np.linalg.norm(row - (row @ Q) @ Q.T)
                    if residual < 1e-8:
                        continue
                elif np.linalg.norm(row) < 0.5:
                    continue

                chosen.append(idx)
                rows.append(unique_H[idx])
                V_f64 = np.vstack([V_f64, row.reshape(1, -1)])
                self.stats.depth1_done = len(chosen)

            if len(chosen) < target_d:
                continue

            # Dedup by sorted index tuple
            key = tuple(sorted(chosen))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            V_basis = np.vstack([r.reshape(1, -1) for r in rows]).astype(np.int64)
            dep_est = quick_dependent_count(
                self.db, V_basis, self.config.dependent_count_sample_size, self.rng,
            )
            if dep_est < self.config.min_dependent_count:
                self.stats.pruned_dependent += 1
                continue

            sigma_rows = [self._sigma_for_unique_index(idx) for idx in chosen]
            self.stats.built_bases += 1
            built += 1
            yield BasisResult(
                rank=self.rank,
                indices=chosen,
                V_basis=V_basis,
                V_perp=_integer_nullspace(V_basis),
                sigma_basis=np.vstack(sigma_rows).astype(np.int64),
                dependent_estimate=dep_est,
            )

    def _build_bases_dfs(self) -> Generator[BasisResult, None, None]:
        """Original DFS-based basis builder (slow but exhaustive)."""
        seeds = self._seed_unique_indices()
        self.stats.seeds_total = len(seeds)
        self.stats.seeds_processed = 0
        built = 0

        for seed in seeds:
            self.stats.seeds_processed += 1
            self.stats.depth1_total = 0
            self.stats.depth1_done = 0
            if built >= self.config.max_bases_per_rank:
                break

            row = self._unique_H_i64[int(seed)]
            if not np.any(row):
                continue

            indices = [int(seed)]
            sigma_rows = [self._sigma_for_unique_index(int(seed))]
            partial = row.reshape(1, -1)

            for result in self._dfs(indices, partial, sigma_rows, int(seed) + 1):
                yield result
                built += 1
                if built >= self.config.max_bases_per_rank:
                    break

    def _dfs(
        self,
        chosen_indices: list[int],
        V_partial: np.ndarray,
        sigma_rows: list[np.ndarray],
        start_idx: int,
    ) -> Generator[BasisResult, None, None]:
        self.stats.explored_nodes += 1
        depth = len(chosen_indices)
        unique_H = self._unique_H_i64

        if depth == self.target_rank:
            dependent_estimate = quick_dependent_count(
                self.db,
                V_partial,
                self.config.dependent_count_sample_size,
                self.rng,
            )
            if dependent_estimate < self.config.min_dependent_count:
                self.stats.pruned_dependent += 1
                return

            self.stats.built_bases += 1
            yield BasisResult(
                rank=self.rank,
                indices=list(chosen_indices),
                V_basis=np.asarray(V_partial, dtype=np.int64),
                V_perp=_integer_nullspace(V_partial),
                sigma_basis=np.vstack(sigma_rows).astype(np.int64),
                dependent_estimate=dependent_estimate,
            )
            return

        remaining_needed = self.target_rank - depth
        n_total = unique_H.shape[0]
        available = n_total - start_idx
        if available < remaining_needed:
            self.stats.pruned_budget += 1
            return

        sigma_rank = _row_rank(np.vstack(sigma_rows), self.config.rank_tol)
        if depth >= self.config.sigma_rank_prune_min_depth and sigma_rank == 0:
            self.stats.pruned_sigma += 1
            return

        # ── Vectorized candidate filtering ──
        # 1. Norm filter: skip rows with norm < threshold
        cand_range = np.arange(start_idx, n_total, dtype=np.int64)
        norm_mask = self._norms[start_idx:n_total] >= self.config.basis_norm_min
        normed_pruned = int(np.sum(~norm_mask))
        self.stats.pruned_norm += normed_pruned
        cand_range = cand_range[norm_mask]

        if cand_range.size < remaining_needed:
            self.stats.pruned_budget += 1
            return

        # 2. Batch independence check via QR projection
        V_f64 = V_partial.astype(np.float64)
        indep_local = _batch_find_independent(V_f64, unique_H[cand_range], depth)
        cand_range = cand_range[indep_local]

        if cand_range.size < remaining_needed:
            self.stats.pruned_budget += 1
            return

        # 3. Iterate only over the survivors (typically a tiny fraction)
        if depth == 1:
            # Cap depth-1 to avoid scanning all 36M unique rows
            cap = getattr(self.config, 'max_depth1_candidates', 10000)
            if cand_range.size > cap:
                cand_range = cand_range[:cap]
            self.stats.depth1_total = int(cand_range.size)
            self.stats.depth1_done = 0
        for idx in cand_range:
            idx = int(idx)
            if depth == 1:
                self.stats.depth1_done += 1
            candidate = unique_H[idx]

            next_partial = np.vstack([V_partial, candidate.reshape(1, -1)])
            dependent_estimate = quick_dependent_count(
                self.db,
                next_partial,
                self.config.dependent_count_sample_size,
                self.rng,
            )
            if dependent_estimate < self.config.min_dependent_count:
                self.stats.pruned_dependent += 1
                continue

            yield from self._dfs(
                chosen_indices + [idx],
                next_partial,
                sigma_rows + [self._sigma_for_unique_index(idx)],
                idx + 1,
            )

    def _sigma_for_unique_index(self, unique_idx: int) -> np.ndarray:
        record_idx = _representative_record_index(self.db, unique_idx)
        return np.asarray(self.db.sigma[record_idx], dtype=np.int64)


def build_bases(db, config, rank: int, rng: Optional[np.random.Generator] = None) -> Iterable[BasisResult]:
    return BasisBuilder(db=db, config=config, rank=rank, rng=rng).build_bases()