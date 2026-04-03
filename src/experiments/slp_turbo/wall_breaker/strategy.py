"""Strategy assignment for wall-breaker workers.

Each CPU LP worker gets a strategy that determines how its LP is
modified each cycle. Strategies rotate so the pool explores diversely.
"""

from enum import IntEnum


class Strategy(IntEnum):
    NORMAL = 0       # Standard weighted LP (existing behavior)
    MASK_WORST = 1   # Mask the single worst binding entry
    MASK_RANDOM = 2  # Mask a random binding entry
    ROTATED = 3      # LP in a rotated coordinate frame


# Number of distinct strategies
N_STRATEGIES = len(Strategy)


def assign_strategies(n_workers, iteration):
    """Assign a strategy to each worker.

    Split: ~50% NORMAL, ~17% MASK_WORST, ~17% MASK_RANDOM, ~17% ROTATED.
    Rotate the assignment each iteration so every worker tries everything.
    """
    strategies = []
    for i in range(n_workers):
        shifted = (i + iteration) % N_STRATEGIES
        # Give 2 slots to NORMAL for every 1 of the others
        # Pattern: N, N, MW, MR, R, N, N, MW, MR, R, ...
        cycle_pos = (i + iteration) % (N_STRATEGIES + 1)
        if cycle_pos <= 1:
            strategies.append(Strategy.NORMAL)
        else:
            strategies.append(Strategy(cycle_pos - 1))
    return strategies
