"""Constants and CLI argument parsing."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR

R = RANK        # 19
D = DIM         # 9
N_ENTRIES = D ** 3   # 729
N_PARAMS = 3 * R * D  # 513
T_NP = TARGET_TENSOR

# ── Fiber-mode tier classification ────────────────────────────────────
# Live entries: T[a,b,c]=1 iff a//3==c//3 and a%3==b//3 and b%3==c%3
# (output_row matches, summation index matches, output_col matches)
import numpy as _np

_T_flat = T_NP.ravel()
LIVE_MASK = (_T_flat != 0)                      # 27 True entries
DEAD_MASK = ~LIVE_MASK                           # 702 True entries
LIVE_IDX = _np.where(LIVE_MASK)[0]              # (27,)
DEAD_IDX = _np.where(DEAD_MASK)[0]              # (702,)

# Tier weights for residual weighting (applied multiplicatively)
# Live entries get 3x, Dead entries get 1x (dead are numerous, so lower per-entry)
TIER_WEIGHTS = _np.ones(N_ENTRIES, dtype=_np.float64)
TIER_WEIGHTS[LIVE_IDX] = 3.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SLP Turbo: concurrent GPU+CPU minimax")
    p.add_argument("--input", default="slp_turbo_best.json")
    p.add_argument("--out", default="slp_turbo_best.json")

    # CPU SLP settings
    p.add_argument("--workers", type=int, default=22,
                   help="Number of SLP workers (CPU LP processes)")
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--trust-init", type=float, default=0.003)
    p.add_argument("--trust-max", type=float, default=0.5)
    p.add_argument("--trust-min", type=float, default=1e-8)
    p.add_argument("--eta-accept", type=float, default=0.01)
    p.add_argument("--active-k", type=int, default=500)

    # GPU smooth descent settings
    p.add_argument("--gpu-batch", type=int, default=16_384,
                   help="GPU exploration batch size")
    p.add_argument("--gpu-steps", type=int, default=200,
                   help="Adam steps per GPU burst")
    p.add_argument("--gpu-lr", type=float, default=5e-5)
    p.add_argument("--gpu-beta", type=float, default=200.0,
                   help="Log-sum-exp sharpness")

    # General
    p.add_argument("--perturb-scale", type=float, default=0.001)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--live-weight", type=float, default=3.0,
                   help="Multiplier for 27 live residuals in LP/GPU objectives")
    p.add_argument("--dead-weight", type=float, default=1.0,
                   help="Multiplier for 702 dead residuals")
    p.add_argument("--sparsity-lambda", type=float, default=0.0,
                   help="L1 sparsity penalty on factor entries in GPU descent (0=off)")

    return p.parse_args(argv)
