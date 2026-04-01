"""Central configuration for db_optimizer."""

from pathlib import Path

import numpy as np

# ── Tensor geometry ──────────────────────────────────────────────────
RANK = 19
DIM = 9
N_ENTRIES = DIM ** 3  # 729
N_PARAMS = RANK * DIM * 3  # 513
BLOB_BYTES = N_PARAMS * 8  # 4104

# ── Paths ────────────────────────────────────────────────────────────
DB_PATH = Path("K:/ade3x3_optimizer/candidates.db")
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Evaluation tiers ─────────────────────────────────────────────────
TIER1_SURVIVOR_THRESHOLD = 0.20
TIER2_REFINE_THRESHOLD = 0.15
HIT_THRESHOLD = 1e-10

# ── GPU batching ─────────────────────────────────────────────────────
MAX_GPU_BATCH_SIZE = 100_000  # RTX 3060 12GB can handle this
GPU_MINIMAX_SWEEPS = 1
GPU_MINIMAX_FINE_RANGE = 0.003
GPU_MINIMAX_N_TRIALS = 32

# ── Population ───────────────────────────────────────────────────────
N_ISLANDS = 8
GENERATION_BATCH_SIZE = 8192
MAX_PENDING = 200_000
TOURNAMENT_SIZE = 4
ELITE_K = 20  # per island
MIGRATION_PROB = 0.08

# ── Mutation probabilities (global defaults) ─────────────────────
P_MUTATE_COEFF = 0.55
P_MUTATE_GAUSSIAN = 0.25
P_CROSSOVER = 0.12
P_SHADOW_REINJECT = 0.08

# ── Power-law islands (explore / exploit structure) ──────────────
ISLAND_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]

ISLAND_ROLES: list[str] = [
    "elite_exploit",   # 0 — deterministic best, deep polish
    "strong_exploit",  # 1 — tight tournament, low mutation
    "exploit",         # 2 — standard tournament
    "balanced",        # 3 — default EA behavior
    "balanced",        # 4 — default EA behavior
    "explore",         # 5 — loose selection, heavy mutation
    "explore",         # 6 — loose selection, heavy mutation
    "wide_explore",    # 7 — random diverse, maximal exploration
]

ISLAND_CONFIG: dict[str, dict] = {
    "elite_exploit":  {"selection": "deterministic", "sigma": 0.002, "cpu_sweeps": 20,
                       "p_coeff": 0.80, "p_gaussian": 0.15, "p_crossover": 0.03, "p_shadow": 0.02},
    "strong_exploit": {"selection": "tournament_tight", "sigma": 0.005, "cpu_sweeps": 15,
                       "p_coeff": 0.70, "p_gaussian": 0.20, "p_crossover": 0.05, "p_shadow": 0.05},
    "exploit":        {"selection": "tournament", "sigma": 0.008, "cpu_sweeps": 10,
                       "p_coeff": 0.60, "p_gaussian": 0.25, "p_crossover": 0.10, "p_shadow": 0.05},
    "balanced":       {"selection": "tournament", "sigma": 0.01, "cpu_sweeps": 7,
                       "p_coeff": 0.55, "p_gaussian": 0.25, "p_crossover": 0.12, "p_shadow": 0.08},
    "explore":        {"selection": "random", "sigma": 0.02, "cpu_sweeps": 4,
                       "p_coeff": 0.40, "p_gaussian": 0.30, "p_crossover": 0.15, "p_shadow": 0.10},
    "wide_explore":   {"selection": "random_diverse", "sigma": 0.05, "cpu_sweeps": 3,
                       "p_coeff": 0.30, "p_gaussian": 0.25, "p_crossover": 0.20, "p_shadow": 0.15},
}

# ── CPU refinement ───────────────────────────────────────────────
CPU_MINIMAX_SWEEPS = 5
CPU_MINIMAX_FINE_RANGE = 0.003
CPU_MIN_SWEEPS = 3
CPU_MAX_SWEEPS = 20

# ── GPU minimax (Tier 2) ─────────────────────────────────────────
GPU_MINIMAX_BATCH_SIZE = 20_000


def build_target_tensor() -> np.ndarray:
    """Build the 9×9×9 matrix multiplication tensor for n=3."""
    n = 3
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                a = n * r + s
                b = n * s + u
                c = n * r + u
                tensor[a, b, c] = 1.0
    return tensor


# Singleton: built once, reused everywhere
TARGET_TENSOR: np.ndarray = build_target_tensor()


def build_algebraic_lookup() -> np.ndarray:
    """Build sorted array of ~1473 algebraic magnitudes for coefficient mutations."""
    table: dict[str, float] = {}

    for x in range(0, 21):
        for y in range(1, 21):
            v = x / y
            if v > 3.5:
                continue
            table[f"{x}/{y}"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = np.sqrt(x / y)
            if v > 3.5:
                continue
            table[f"sqrt({x}/{y})"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** (1 / 3)
            if v > 3.5:
                continue
            table[f"cbrt({x}/{y})"] = v

    for x in range(1, 21):
        for y in range(1, 21):
            v = (x / y) ** 0.25
            if v > 3.5:
                continue
            table[f"4rt({x}/{y})"] = v

    for x in range(1, 11):
        for y in range(1, 11):
            v = (x / y) ** (1 / 6)
            if v > 3.5:
                continue
            table[f"6rt({x}/{y})"] = v

    # 27-family: tensor norm is 27
    for y in range(1, 28):
        v = np.sqrt(27.0 / y)
        if v < 3.5:
            table[f"sqrt(27/{y})"] = v
        v = (27.0 / y) ** (1 / 3)
        if v < 3.5:
            table[f"cbrt(27/{y})"] = v
        v = (27.0 / y) ** 0.25
        if v < 3.5:
            table[f"4rt(27/{y})"] = v
        v = (27.0 / y) ** (1 / 6)
        if v < 3.5:
            table[f"6rt(27/{y})"] = v

    for k in range(-6, 7):
        for n in [2, 3, 4, 6]:
            v = 3.0 ** (k / n)
            if 0.01 < v < 3.5:
                table[f"3^({k}/{n})"] = v

    for m in [1, 2, 3, 4, 6, 9, 12, 18, 27]:
        base = np.sqrt(27.0 / m)
        for p in range(1, 4):
            for q in range(1, 4):
                v = base * p / q
                if 0.01 < v < 3.5:
                    table[f"{p}/{q}*sqrt(27/{m})"] = v

    # Products: sqrt * cbrt, rational * sqrt, rational * cbrt
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                for d in range(1, 6):
                    v = np.sqrt(a / b) * ((c / d) ** (1 / 3))
                    if 0.01 < v < 3.0:
                        table[f"sqrt({a}/{b})*cbrt({c}/{d})"] = v

    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = np.sqrt(a / b) * (c / d)
                    if 0.01 < v < 3.0:
                        table[f"{c}/{d}*sqrt({a}/{b})"] = v

    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 4):
                for d in range(1, 4):
                    v = ((a / b) ** (1 / 3)) * (c / d)
                    if 0.01 < v < 3.0:
                        table[f"{c}/{d}*cbrt({a}/{b})"] = v

    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in [2, 3, 4, 6]:
                v = abs((2**a) * (3**b)) ** (1 / c)
                if 0.01 < v < 3.0:
                    table[f"(2^{a}*3^{b})^(1/{c})"] = v

    for p in range(1, 10):
        for q in range(1, 10):
            v = 3.0 * p / q
            if 0.01 < v < 3.5:
                table[f"3*{p}/{q}"] = v
            v = np.sqrt(3.0) * p / q
            if 0.01 < v < 3.5:
                table[f"sqrt3*{p}/{q}"] = v
            v = 3.0 ** (2 / 3) * p / q
            if 0.01 < v < 3.5:
                table[f"3^(2/3)*{p}/{q}"] = v

    table["0"] = 0.0

    deduped: dict[float, float] = {}
    for _name, val in sorted(table.items(), key=lambda x: (len(x[0]), x[0])):
        rounded = round(val, 8)
        if rounded not in deduped:
            deduped[rounded] = val

    return np.array(sorted(deduped.values()))


ALGEBRAIC_LOOKUP: np.ndarray = build_algebraic_lookup()
