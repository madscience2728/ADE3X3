from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from gate_check import full_gate_check


@dataclass
class AssemblyStats:
    bases_tested: int = 0
    avg_hits: float = 0.0
    gate2_passes: int = 0
    gate3_passes: int = 0
    configs_tested: int = 0


def classify_hits(db, hit_indices: np.ndarray) -> dict[int, list[int]]:
    sigmas = np.asarray(db.sigma[hit_indices], dtype=np.int64)
    fiber_map: dict[int, list[int]] = defaultdict(list)
    for local_i, idx in enumerate(hit_indices):
        for fiber_id in np.flatnonzero(sigmas[local_i]).tolist():
            fiber_map[int(fiber_id)].append(int(idx))
    return fiber_map


def compute_sn_rows_batch(db, candidate_indices: np.ndarray) -> np.ndarray:
    sigma = np.asarray(db.sigma[candidate_indices], dtype=np.int64)
    H = np.asarray(db.H[candidate_indices], dtype=np.int64)
    delta = np.asarray(db.compute_delta_batch(candidate_indices), dtype=np.int64)
    nuisance = np.hstack([H, delta])
    return np.hstack([sigma, nuisance])


def _innovation_scores(SN_matrix: np.ndarray, candidate_rows: np.ndarray) -> np.ndarray:
    if SN_matrix.size == 0:
        return np.linalg.norm(candidate_rows.astype(np.float64), axis=1)

    Q, _ = np.linalg.qr(SN_matrix.T.astype(np.float64), mode="reduced")
    projected = (candidate_rows.astype(np.float64) @ Q) @ Q.T
    residuals = candidate_rows.astype(np.float64) - projected
    return np.linalg.norm(residuals, axis=1)


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

        basis_set = set(basis_indices)
        candidates = np.array([idx for idx in hit_indices if idx not in basis_set], dtype=np.int64)
        if candidates.size == 0:
            return []
        if candidates.size > self.config.max_assembly_candidates:
            candidates = candidates[: self.config.max_assembly_candidates]

        SN_basis = compute_sn_rows_batch(self.db, np.asarray(basis_indices, dtype=np.int64))
        candidate_rows = compute_sn_rows_batch(self.db, candidates)
        innovations = _innovation_scores(SN_basis, candidate_rows)
        viable = np.flatnonzero(innovations > self.config.innovation_threshold)
        if viable.size == 0:
            return []

        ordered = viable[np.argsort(innovations[viable])[::-1]]
        ordered_candidates = candidates[ordered[: self.config.max_dependents_per_branch]]
        ordered_rows = candidate_rows[ordered[: self.config.max_dependents_per_branch]].astype(np.float64)

        results: list[dict] = []
        self._search(
            rank=rank,
            chosen=list(basis_indices),
            SN_matrix=SN_basis.astype(np.float64),
            candidate_indices=ordered_candidates,
            candidate_rows=ordered_rows,
            depth=0,
            target_dependents=target_dependents,
            results=results,
        )
        return results

    def _search(
        self,
        rank: int,
        chosen: list[int],
        SN_matrix: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_rows: np.ndarray,
        depth: int,
        target_dependents: int,
        results: list[dict],
    ) -> None:
        if len(results) >= self.config.max_solutions_per_rank:
            return

        if depth == target_dependents:
            self.stats.configs_tested += 1
            passed, diag = full_gate_check(self.db, chosen, self.config)
            if passed:
                self.stats.gate2_passes += 1
                self.stats.gate3_passes += 1
                results.append(diag)
            return

        # Batch rank-increasing check: one QR of the current SN_matrix,
        # then matmul all candidates against Q to get residual norms.
        # Candidates with residual > tol are rank-increasing.
        SN_f64 = SN_matrix.astype(np.float64) if SN_matrix.dtype != np.float64 else SN_matrix
        current_rank = int(np.linalg.matrix_rank(SN_f64, tol=self.config.rank_tol))
        remaining = target_dependents - depth
        if current_rank + remaining < rank:
            return

        if candidate_rows.shape[0] == 0:
            return

        # One QR decomposition, then batch projection residuals
        cand_f64 = candidate_rows.astype(np.float64) if candidate_rows.dtype != np.float64 else candidate_rows
        if SN_f64.shape[0] > 0:
            Q, _ = np.linalg.qr(SN_f64.T, mode='reduced')  # (cols, current_rank)
            projected = (cand_f64 @ Q) @ Q.T                # (N, cols) — one batch matmul
            residuals = np.linalg.norm(cand_f64 - projected, axis=1)
        else:
            residuals = np.linalg.norm(cand_f64, axis=1)

        # Rank-increasing candidates are those with residual > tolerance
        viable_mask = residuals > 1e-8
        viable_indices = np.flatnonzero(viable_mask)

        for vi in viable_indices:
            i = int(vi)
            new_matrix = np.vstack([SN_f64, cand_f64[i].reshape(1, -1)])
            self._search(
                rank=rank,
                chosen=chosen + [int(candidate_indices[i])],
                SN_matrix=new_matrix,
                candidate_indices=candidate_indices[i + 1 :],
                candidate_rows=candidate_rows[i + 1 :],
                depth=depth + 1,
                target_dependents=target_dependents,
                results=results,
            )
            if len(results) >= self.config.max_solutions_per_rank:
                return


def assemble_dependents(db, rank: int, basis_indices: list[int], hit_indices: np.ndarray, config) -> list[dict]:
    return DependentAssembler(db=db, config=config).assemble(rank=rank, basis_indices=basis_indices, hit_indices=hit_indices)