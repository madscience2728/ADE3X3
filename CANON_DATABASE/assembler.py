from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from gate_check import full_gate_check

_RNG = np.random.default_rng(42)

# Tolerance for "in row-span" (gate-2 neutrality) check
_NEUTRAL_TOL = 1e-7
# Tolerance for rank-increasing (SN innovation) check
_INNOV_TOL = 1e-8


@dataclass
class AssemblyStats:
    bases_tested: int = 0
    avg_hits: float = 0.0
    gate2_passes: int = 0
    gate3_passes: int = 0
    configs_tested: int = 0
    # Depth profiles: list of (n_neutral_at_depth_0, ..., n_neutral_at_depth_8) per trial
    depth_profiles: list[list[int]] = field(default_factory=list)

    def record_depth_profile(self, profile: list[int]) -> None:
        self.depth_profiles.append(profile)

    def depth_summary(self) -> dict:
        """Aggregate depth profiles: median pool size at each depth."""
        if not self.depth_profiles:
            return {}
        max_depth = max(len(p) for p in self.depth_profiles)
        summary = {}
        for d in range(max_depth):
            vals = [p[d] for p in self.depth_profiles if len(p) > d]
            if vals:
                summary[d] = {
                    "n_trials": len(vals),
                    "min": min(vals),
                    "median": int(np.median(vals)),
                    "max": max(vals),
                    "zero_count": sum(1 for v in vals if v == 0),
                }
        return summary


def compute_nd_rows_batch(db, indices: np.ndarray) -> np.ndarray:
    """Return (N, 72) float64 nuisance rows [H|Delta] for each index."""
    H = np.asarray(db.H[indices], dtype=np.float64)
    delta = np.asarray(db.compute_delta_batch(indices), dtype=np.float64)
    return np.hstack([H, delta])


def compute_sn_rows_batch(db, indices: np.ndarray) -> np.ndarray:
    """Return (N, 81) float64 [Sigma|H|Delta] rows for each index."""
    sigma = np.asarray(db.sigma[indices], dtype=np.float64)
    nd = compute_nd_rows_batch(db, indices)
    return np.hstack([sigma, nd])


def _row_space_projector(A: np.ndarray, tol: float = _NEUTRAL_TOL):
    """Return (Q, rank) where rows of Q are an ONB for the row space of A."""
    if A.size == 0:
        return np.zeros((0, A.shape[1]), dtype=np.float64), 0
    _, sv, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)
    cutoff = tol * sv[0] if sv[0] > 0 else tol
    r = int(np.sum(sv > cutoff))
    return Vt[:r], r


def _residual_norms(rows: np.ndarray, QQt: np.ndarray) -> np.ndarray:
    """‖row - QQt @ row‖ for each row; QQt is the (n_cols, n_cols) projection matrix."""
    proj = rows @ QQt
    return np.linalg.norm(rows - proj, axis=1)


class DependentAssembler:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.stats = AssemblyStats()

    def assemble(self, rank: int, basis_indices: list[int], hit_indices: np.ndarray) -> list[dict]:
        self.stats.bases_tested += 1
        self.stats.avg_hits = (
            ((self.stats.bases_tested - 1) * self.stats.avg_hits) + float(len(hit_indices))
        ) / self.stats.bases_tested

        target_dependents = rank - len(basis_indices)
        if target_dependents <= 0:
            passed, diag = full_gate_check(self.db, basis_indices, self.config)
            self.stats.configs_tested += 1
            if passed:
                self.stats.gate2_passes += 1
                self.stats.gate3_passes += 1
                return [diag]
            return []

        basis_arr = np.asarray(basis_indices, dtype=np.int64)
        basis_set = set(basis_indices)

        # ── Candidate pool ────────────────────────────────────────────────────
        mask = np.ones(len(hit_indices), dtype=bool)
        for i, idx in enumerate(hit_indices):
            if int(idx) in basis_set:
                mask[i] = False
        candidates = hit_indices[mask]
        if candidates.size == 0:
            return []
        if candidates.size > self.config.max_assembly_candidates:
            candidates = candidates[:self.config.max_assembly_candidates]

        # ── Precompute basis matrices ─────────────────────────────────────────
        ND_basis = compute_nd_rows_batch(self.db, basis_arr)   # (n_basis, 72)
        SN_basis = compute_sn_rows_batch(self.db, basis_arr)   # (n_basis, 81)

        # ── Precompute candidate rows (one batch read of the DB) ─────────────
        cand_ND = compute_nd_rows_batch(self.db, candidates)   # (pool, 72)
        cand_SN = compute_sn_rows_batch(self.db, candidates)   # (pool, 81)

        # ── Gate-2 neutral prefilter ──────────────────────────────────────────
        # Neutral: [h|d] ∈ row-span(ND_basis) → adding this term does NOT
        # increase rank([H|Delta]).  Only neutral terms can maintain delta_leak.
        Q_N, rk_N = _row_space_projector(ND_basis, tol=_NEUTRAL_TOL)
        QQt_N = Q_N.T @ Q_N  # (72, 72)

        resid_nd = _residual_norms(cand_ND, QQt_N)
        neutral_mask = resid_nd < _NEUTRAL_TOL * 100

        n_total     = candidates.size
        n_neutral   = int(neutral_mask.sum())

        if n_neutral == 0:
            # No neutral hits — log depth-0 profile and bail
            self.stats.record_depth_profile([0])
            return []

        pool_idx = candidates[neutral_mask]      # (n_neutral,)
        pool_ND  = cand_ND[neutral_mask]         # (n_neutral, 72)
        pool_SN  = cand_SN[neutral_mask]         # (n_neutral, 81)

        n_trials = getattr(self.config, 'n_assembly_trials', 2000)
        top_k    = getattr(self.config, 'trial_top_k', 64)
        rng      = _RNG
        results: list[dict] = []

        for _ in range(n_trials):
            if len(results) >= self.config.max_solutions_per_rank:
                break
            trial_indices, depth_profile = self._gate2_trial(
                basis_indices, ND_basis, SN_basis,
                pool_idx, pool_ND, pool_SN,
                target_dependents, top_k, rng,
                n_neutral_initial=n_neutral,
            )
            self.stats.record_depth_profile(depth_profile)
            if trial_indices is None:
                continue
            self.stats.configs_tested += 1
            passed, diag = full_gate_check(self.db, trial_indices, self.config)
            if passed:
                self.stats.gate2_passes += 1
                self.stats.gate3_passes += 1
                results.append(diag)

        return results

    def _gate2_trial(
        self,
        basis_indices: list[int],
        ND_basis: np.ndarray,      # (n_basis, 72)
        SN_basis: np.ndarray,      # (n_basis, 81)
        pool_idx: np.ndarray,      # (n_neutral,)  initial neutral pool
        pool_ND:  np.ndarray,      # (n_neutral, 72)
        pool_SN:  np.ndarray,      # (n_neutral, 81)
        target_dependents: int,
        top_k: int,
        rng: np.random.Generator,
        n_neutral_initial: int,
    ) -> tuple[list[int] | None, list[int]]:
        """
        Gate-2-aware greedy trial.

        At each depth:
          1. Pool = neutral hits w.r.t. current N = [H|Delta].
          2. Among neutral hits, find those that are SN-rank-increasing (innovation > 0).
          3. Sample one from the top-K innovative.
          4. Update N and SN; re-project pool to maintain neutrality invariant.

        Returns (chosen_indices | None, depth_profile).
        depth_profile[d] = size of neutral pool BEFORE picking at depth d.
        """
        chosen = list(basis_indices)
        ND_cur = ND_basis.copy()   # (n_selected, 72) — grows with each pick
        SN_cur = SN_basis.copy()   # (n_selected, 81) — grows with each pick

        # Working copies of pool (may shrink after each pick)
        cur_pool_idx = pool_idx.copy()
        cur_pool_ND  = pool_ND.copy()
        cur_pool_SN  = pool_SN.copy()

        # Row-space projectors
        Q_N,   _  = _row_space_projector(ND_cur, tol=_NEUTRAL_TOL)
        QQt_N     = Q_N.T @ Q_N if Q_N.size > 0 else np.zeros((72, 72))

        Q_SN,  _  = _row_space_projector(SN_cur, tol=_INNOV_TOL)
        QQt_SN    = Q_SN.T @ Q_SN if Q_SN.size > 0 else np.zeros((81, 81))

        depth_profile: list[int] = []

        for depth in range(target_dependents):
            pool_size = cur_pool_idx.shape[0]
            depth_profile.append(pool_size)

            if pool_size == 0:
                return None, depth_profile

            # Among neutral pool, find SN-rank-increasing candidates
            resid_sn = _residual_norms(cur_pool_SN, QQt_SN)
            viable   = np.flatnonzero(resid_sn > _INNOV_TOL)

            if viable.size == 0:
                return None, depth_profile

            # Sample from top-K by SN innovation
            k = min(top_k, viable.size)
            top_viable = viable[np.argsort(resid_sn[viable])[-k:]]
            pick_local = int(rng.choice(top_viable))

            # Record the pick
            chosen.append(int(cur_pool_idx[pick_local]))
            picked_ND = cur_pool_ND[pick_local:pick_local + 1]  # (1, 72)
            picked_SN = cur_pool_SN[pick_local:pick_local + 1]  # (1, 81)

            ND_cur = np.vstack([ND_cur, picked_ND])
            SN_cur = np.vstack([SN_cur, picked_SN])

            # Remove chosen from pool
            keep = np.ones(cur_pool_idx.shape[0], dtype=bool)
            keep[pick_local] = False
            cur_pool_idx = cur_pool_idx[keep]
            cur_pool_ND  = cur_pool_ND[keep]
            cur_pool_SN  = cur_pool_SN[keep]

            if cur_pool_idx.shape[0] == 0:
                if depth + 1 < target_dependents:
                    depth_profile.append(0)
                break

            # Re-project pool against UPDATED N to maintain neutral invariant.
            # (Theoretically: adding a neutral row doesn't expand row-span of N,
            # so QQt_N is unchanged.  We recompute anyway to catch numerical drift.)
            Q_N,  _ = _row_space_projector(ND_cur, tol=_NEUTRAL_TOL)
            QQt_N   = Q_N.T @ Q_N if Q_N.size > 0 else np.zeros((72, 72))

            still_neutral = _residual_norms(cur_pool_ND, QQt_N) < _NEUTRAL_TOL * 100
            cur_pool_idx = cur_pool_idx[still_neutral]
            cur_pool_ND  = cur_pool_ND[still_neutral]
            cur_pool_SN  = cur_pool_SN[still_neutral]

            # Update SN projector
            Q_SN,  _ = _row_space_projector(SN_cur, tol=_INNOV_TOL)
            QQt_SN   = Q_SN.T @ Q_SN if Q_SN.size > 0 else np.zeros((81, 81))

        if len(chosen) - len(basis_indices) < target_dependents:
            return None, depth_profile

        return chosen, depth_profile


def assemble_dependents(db, rank: int, basis_indices: list[int], hit_indices: np.ndarray, config) -> list[dict]:
    return DependentAssembler(db=db, config=config).assemble(rank=rank, basis_indices=basis_indices, hit_indices=hit_indices)
