"""
swap_config.py — Configuration for the R=19 swap optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SwapConfig:
    # Search target
    R: int = 19
    target_rank_H: int = 10

    # Workers
    n_workers: int = 24

    # Move probabilities (must sum to 1)
    single_swap_prob: float = 0.70
    double_swap_prob: float = 0.20
    triple_swap_prob: float = 0.10

    # Biased sampling: fraction drawn from hit pool vs full DB
    hit_pool_bias: float = 0.80

    # Escalation: non-improving swaps before we restart
    stall_threshold: int = 2000

    # Stopping conditions
    time_limit_seconds: int = 0        # 0 = no limit
    target_delta_leak: int = 0         # stop when best reaches this
    max_restarts_per_worker: int = 500

    # Checkpointing / logging
    checkpoint_interval: int = 60      # seconds
    checkpoint_file: str = str(_ROOT / "CANON_DATABASE" / "swap_checkpoint.json")
    solution_file:   str = str(_ROOT / "CANON_DATABASE" / "SOLUTION.json")
    log_improvements: bool = True
    log_file: str = str(_ROOT / "CANON_DATABASE" / "swap_improvements.jsonl")

    # DB
    db_path: str = str(_ROOT / "CANON_DATABASE" / "data")

    # Numerical tolerances
    rank_tol: float = 1e-9
