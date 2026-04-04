from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SearchConfig:
    target_ranks: list[int] = field(default_factory=lambda: [13, 19, 20, 21, 22])
    coeff_field: list[int] = field(default_factory=lambda: [-1, 0, 1])

    basis_seed_count: int = 512
    max_bases_per_rank: int = 256
    min_dependent_count: int = 100
    dependent_count_sample_size: int = 10000
    basis_norm_min: int = 1
    sigma_rank_prune_min_depth: int = 3

    max_assembly_candidates: int = 200000
    max_dependents_per_branch: int = 256   # kept for legacy DFS (unused)
    max_depth1_candidates: int = 10000
    max_solutions_per_rank: int = 1
    innovation_threshold: float = 1e-8

    # Random greedy trial search (replaces DFS)
    n_assembly_trials: int = 2000         # random trials per packet
    trial_top_k: int = 64                 # at each depth, sample from top-K viable by innovation

    gate3_residual_tol: float = 1e-10
    rank_tol: float = 1e-10

    n_workers: int = 24
    checkpoint_interval: int = 60
    load_db_to_ram: bool = True
    ram_floor_gb: float = 8.0  # stop submitting new work if available RAM drops below this

    db_path: str = "./CANON_DATABASE/data"
    checkpoint_file: str = "./CANON_DATABASE/search_checkpoint.json"
    solution_file: str = "./CANON_DATABASE/SOLUTION.json"
    survivors_file: str = "./CANON_DATABASE/gate2_survivors.jsonl"

    dashboard_refresh_rate: float = 0.5

    def target_rank_H(self, rank: int) -> int:
        return rank - 9

    def validate(self) -> None:
        if self.max_bases_per_rank <= 0:
            raise ValueError("max_bases_per_rank must be positive")
        if self.max_dependents_per_branch <= 0:
            raise ValueError("max_dependents_per_branch must be positive")
        if self.dependent_count_sample_size <= 0:
            raise ValueError("dependent_count_sample_size must be positive")

    def resolve_paths(self, workspace_root: Path) -> "SearchConfig":
        def _resolve(path_str: str) -> str:
            path = Path(path_str)
            if path.is_absolute():
                return str(path)
            return str((workspace_root / path).resolve())

        self.db_path = _resolve(self.db_path)
        self.checkpoint_file = _resolve(self.checkpoint_file)
        self.solution_file = _resolve(self.solution_file)
        self.survivors_file = _resolve(self.survivors_file)
        return self