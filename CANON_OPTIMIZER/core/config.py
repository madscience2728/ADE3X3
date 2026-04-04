"""
Central configuration. All magic numbers in one place.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchConfig:
    # Problem parameters
    n: int = 3                          # matrix size (3x3)
    R: int = 19                         # target rank
    conservation_target: int = 27       # = n^3

    # Derived targets (recomputed per R)
    @property
    def target_rank_H(self) -> int:
        return self.R - self.n * self.n  # R - 9

    @property
    def target_nuisance_rank(self) -> int:
        return self.target_rank_H

    @property
    def target_augmented_rank(self) -> int:
        return self.R

    @property
    def eta_nullity(self) -> int:
        return 18 - self.target_rank_H

    # Coefficient field
    coeff_field: List[int] = field(default_factory=lambda: [-1, 0, 1])

    # Hardware
    n_workers: int = 24
    use_gpu: bool = False
    gpu_batch_size: int = 4096

    # Checkpointing
    checkpoint_interval_seconds: int = 60
    checkpoint_file: str = "./CANON OPTIMIZER/core/checkpoint.json"
    solution_file: str = "./CANON OPTIMIZER/core/SOLUTION.json"

    # Pruning thresholds
    rank_tolerance: float = 1e-10
    gate3_residual_tolerance: float = 1e-12

    # Dashboard
    dashboard_refresh_rate: float = 0.5

    # Logging
    log_file: str = "./CANON OPTIMIZER/core/search.log"
    log_gate1_survivors: bool = True
    gate1_survivor_file: str = "./CANON OPTIMIZER/core/gate1_survivors.jsonl"

    # Default ranks to search
    default_ranks: List[int] = field(default_factory=lambda: [13, 19, 20, 21, 22])
