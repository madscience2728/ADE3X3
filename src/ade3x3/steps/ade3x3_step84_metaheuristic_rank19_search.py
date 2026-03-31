"""
ade3x3_step84_metaheuristic_rank19_search.py

Step 84: Metaheuristic rank-19 search.

This step abandons continuation in favor of a sparse-support memetic search.
Each individual is a 19-term sparse CP decomposition candidate for the 3x3
matrix multiplication tensor. The evolutionary layer mutates the support pattern
and coefficient guesses, while a local Lamarckian layer runs support-fixed
least-squares refinement.

Design points:
1. Sparse supports stay inside the empirically tractable Step 83b regime
   (2..6 nonzeros per factor entry vector).
2. The outer search is a 4-island (mu+lambda) evolutionary algorithm.
3. The inner local search uses one or more alternating linear least-squares
   sweeps for all candidates plus an occasional gauge-fixed nonlinear polish for
   promising individuals.
4. The run checkpoints periodically and can resume from a prior checkpoint.

Primary exports:
- outputs/exports/step84_evolution_log.csv
- outputs/exports/step84_best_individual.json
- outputs/exports/step84_final_population.json
- outputs/exports/step84_summary.json
- outputs/exports/step84_checkpoint_genXXXX.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    matrix_multiplication_tensor,
)
from src.ade3x3.steps.ade3x3_step81_homotopy_bridge_pilot import (
    build_jacobian,
    build_support_model,
    evaluate_coordinates,
    full_coordinate_list,
    full_tensor_from_factors,
    reduce_model_to_independent_variables,
    scalar_to_str,
    unpack_free_vector,
    witness_free_vector,
    write_json,
)
from src.ade3x3.steps.ade3x3_step83b_support_expansion_sparse_meta_analysis import (
    build_track2_terms_from_row,
)
from src.ade3x3.steps.ade3x3_step84_canonical_constraints import (
    canonical_support_hash,
    dead_leakage_batch,
    fiber_biased_support,
    live_compatible_beta_positions,
    live_compatible_gamma_positions,
    structural_penalty,
    LIVE_B_FOR_A,
)


BASE_EXPORTS = Path("outputs/exports")
EXPORTS = Path((os.environ.get("STEP84_EXPORTS_DIR") or str(BASE_EXPORTS)).strip())
LOG_PATH = EXPORTS / "step84_evolution_log.csv"
BEST_PATH = EXPORTS / "step84_best_individual.json"
FINAL_POP_PATH = EXPORTS / "step84_final_population.json"
SUMMARY_PATH = EXPORTS / "step84_summary.json"
TRACK2_SCREENING_PATH = BASE_EXPORTS / "step83b_track2_screening.csv"


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw.strip())


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or not raw.strip() else int(raw.strip())


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw.strip())


def env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [int(part.strip()) for part in re.split(r"[;,\s]+", raw) if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def build_power_island_populations(default: list[int]) -> list[int]:
    explicit_list = os.environ.get("STEP84_ISLAND_POPULATIONS")
    if explicit_list is not None and explicit_list.strip():
        return env_int_list("STEP84_ISLAND_POPULATIONS", default)

    min_exp = env_optional_int("STEP84_ISLAND_MIN_EXP")
    max_exp = env_optional_int("STEP84_ISLAND_MAX_EXP")
    copies = env_optional_int("STEP84_ISLAND_COPIES")
    if min_exp is None and max_exp is None and copies is None:
        return list(default)

    min_exp = 2 if min_exp is None else min_exp
    max_exp = 10 if max_exp is None else max_exp
    copies = 8 if copies is None else copies
    if min_exp > max_exp:
        raise ValueError("STEP84_ISLAND_MIN_EXP must be <= STEP84_ISLAND_MAX_EXP")
    if copies <= 0:
        raise ValueError("STEP84_ISLAND_COPIES must be positive")

    populations: list[int] = []
    for exponent in range(max_exp, min_exp - 1, -1):
        populations.extend([2**exponent] * copies)
    return populations


RANK_VALUE = 19
ISLAND_POPULATIONS = build_power_island_populations([32, 24, 16, 12, 8, 8, 6, 6])
NUM_ISLANDS = len(ISLAND_POPULATIONS)
TOTAL_POP = sum(ISLAND_POPULATIONS)
OFFSPRING_MULTIPLIER = max(1, env_int("STEP84_OFFSPRING_MULTIPLIER", 2))
OFFSPRING_PER_ISLANDS = [population_size * OFFSPRING_MULTIPLIER for population_size in ISLAND_POPULATIONS]
NUM_GENERATIONS = env_int("STEP84_GENERATIONS", 500)
MIGRATION_INTERVAL = env_int("STEP84_MIGRATION_INTERVAL", 25)
MIGRATION_SIZE = env_int("STEP84_MIGRATION_SIZE", 1)
WORKERS = env_int("STEP84_WORKERS", min(24, max(1, os.cpu_count() or 24)))
EXECUTOR_KIND = (os.environ.get("STEP84_EXECUTOR_KIND") or ("thread" if platform.system() == "Windows" else "process")).strip().lower()
SEED = env_int("STEP84_SEED", 8401001)
WARM_SEED_COUNT = min(env_int("STEP84_WARM_SEEDS", 12), TOTAL_POP)
TOURNAMENT_SIZE = env_int("STEP84_TOURNAMENT_SIZE", 3)
TIMEOUT_SECONDS = env_int("STEP84_TIMEOUT_SECONDS", 3600)
SIGNATURE333_SURVIVOR_FRACTION = env_float("STEP84_SIGNATURE333_SURVIVOR_FRACTION", 0.15)
SHADOW_SURVIVOR_FRACTION = env_float("STEP84_SHADOW_SURVIVOR_FRACTION", 0.30)
SHADOW_PARENT_RATE = env_float("STEP84_SHADOW_PARENT_RATE", 0.40)
SHADOW_VARIABLE_BUCKET = env_int("STEP84_SHADOW_VARIABLE_BUCKET", 16)
SHADOW_METRICS_INTERVAL = max(1, env_int("STEP84_SHADOW_METRICS_INTERVAL", 10))

CROSSOVER_RATE = env_float("STEP84_CROSSOVER_RATE", 0.7)
FACTOR_CROSSOVER_RATE = env_float("STEP84_FACTOR_CROSSOVER_RATE", 0.2)
MUTATION_SUPPORT_RATE = env_float("STEP84_MUTATION_SUPPORT_RATE", 0.3)
MUTATION_COEFF_RATE = env_float("STEP84_MUTATION_COEFF_RATE", 0.5)
MUTATION_REPLACE_RATE = env_float("STEP84_MUTATION_REPLACE_RATE", 0.05)
SUPPORT_ADD_PROB = env_float("STEP84_SUPPORT_ADD_PROB", 0.5)
SUPPORT_DROP_PROB = env_float("STEP84_SUPPORT_DROP_PROB", 0.5)

SUPPORT_MIN = env_int("STEP84_SUPPORT_MIN", 2)
SUPPORT_MAX = env_int("STEP84_SUPPORT_MAX", 3)
INIT_SUPPORT_MIN = env_int("STEP84_INIT_SUPPORT_MIN", 2)
INIT_SUPPORT_MAX = env_int("STEP84_INIT_SUPPORT_MAX", 3)
ALS_SWEEPS = env_int("STEP84_ALS_SWEEPS", 1)
NEWTON_THRESHOLD = env_float("STEP84_NEWTON_THRESHOLD", 0.25)
POLISH_MAX_NFEV = env_int("STEP84_POLISH_MAX_NFEV", 20)
POLISH_VARIABLE_CAP = env_int("STEP84_POLISH_VARIABLE_CAP", 260)
CHECKPOINT_INTERVAL = env_int("STEP84_CHECKPOINT_INTERVAL", 50)
CHECKPOINT_KEEP_LAST = env_int("STEP84_CHECKPOINT_KEEP_LAST", 3)
HIT_THRESHOLD = env_float("STEP84_HIT_THRESHOLD", 1e-8)
ACTIVE_VALUE_FLOOR = env_float("STEP84_ACTIVE_VALUE_FLOOR", 1e-4)
BEST_EXPORT_INCLUDE_DENSE = bool(env_int("STEP84_BEST_EXPORT_INCLUDE_DENSE", 0))
COEFF_SIGMA_MIN = env_float("STEP84_COEFF_SIGMA_MIN", 0.01)
COEFF_SIGMA_MAX = env_float("STEP84_COEFF_SIGMA_MAX", 0.35)
COEFF_INIT_LOW = env_float("STEP84_COEFF_INIT_LOW", 1.0)
COEFF_INIT_HIGH = env_float("STEP84_COEFF_INIT_HIGH", 2.0)
SUPPORT_ADD_LOW = env_float("STEP84_SUPPORT_ADD_LOW", 0.01)
SUPPORT_ADD_HIGH = env_float("STEP84_SUPPORT_ADD_HIGH", 0.10)
RESUME_CHECKPOINT = (os.environ.get("STEP84_RESUME_CHECKPOINT") or "").strip() or None

TARGET_TENSOR = matrix_multiplication_tensor(3).astype(np.float64)
TARGET_FLAT = TARGET_TENSOR.reshape(-1)
FULL_COORDINATES = full_coordinate_list()
LOG_FIELDNAMES = [
    "generation",
    "island",
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "best_support_signature",
    "best_variable_count",
    "shadow_pool_size",
    "niche_count",
    "signature_333_count",
    "favored_regime_count",
    "favored_regime_best_fitness",
    "signature_histogram_json",
    "newton_polished_count",
    "wall_seconds_cumulative",
]

PROFILE_PATH = EXPORTS / "step84_profile.json"
_PROFILE_WRITE_INTERVAL = max(1, env_int("STEP84_PROFILE_INTERVAL", 1))


class _ProfileAccumulator:
    """Lightweight per-phase timing accumulator for profiling."""

    __slots__ = (
        "eval_count", "phase_sums", "phase_maxes",
        "window_count", "window_sums",
        "mainloop_sums", "gen_timings",
        "_gen_start", "_gen_evals",
    )

    def __init__(self) -> None:
        self.eval_count: int = 0
        self.phase_sums: dict[str, float] = {}
        self.phase_maxes: dict[str, float] = {}
        self.window_count: int = 0
        self.window_sums: dict[str, float] = {}
        self.mainloop_sums: dict[str, float] = {}
        self.gen_timings: list[list] = []
        self._gen_start: float = time.perf_counter()
        self._gen_evals: int = 0

    def record_eval(self, timing: dict | None) -> None:
        if not timing:
            return
        self.eval_count += 1
        self.window_count += 1
        self._gen_evals += 1
        for key, value in timing.items():
            fv = float(value)
            self.phase_sums[key] = self.phase_sums.get(key, 0.0) + fv
            cur_max = self.phase_maxes.get(key, 0.0)
            if fv > cur_max:
                self.phase_maxes[key] = fv
            self.window_sums[key] = self.window_sums.get(key, 0.0) + fv

    def record_generation(self, gen: int) -> None:
        now = time.perf_counter()
        wall = now - self._gen_start
        evals = self._gen_evals
        self.gen_timings.append([gen, round(wall, 4), evals, round(evals / max(wall, 1e-9), 2)])
        if len(self.gen_timings) > 500:
            self.gen_timings = self.gen_timings[-400:]
        self._gen_start = now
        self._gen_evals = 0
        self.window_sums.clear()
        self.window_count = 0

    def record_mainloop(self, phase: str, ms: float) -> None:
        self.mainloop_sums[phase] = self.mainloop_sums.get(phase, 0.0) + ms

    def to_dict(self, generation: int) -> dict:
        mean_ms: dict[str, float] = {}
        max_ms: dict[str, float] = {}
        if self.eval_count > 0:
            for key in self.phase_sums:
                mean_ms[key] = round(self.phase_sums[key] / self.eval_count, 3)
                max_ms[key] = round(self.phase_maxes.get(key, 0.0), 3)
        window_mean: dict[str, float] = {}
        if self.window_count > 0:
            for key in self.window_sums:
                window_mean[key] = round(self.window_sums[key] / self.window_count, 3)
        return {
            "updated_at": time.time(),
            "generation": generation,
            "eval_count": self.eval_count,
            "eval_phases_mean_ms": mean_ms,
            "eval_phases_max_ms": max_ms,
            "recent_window_eval_count": self.window_count,
            "recent_window_mean_ms": window_mean,
            "mainloop_cumulative_ms": {k: round(v, 1) for k, v in self.mainloop_sums.items()},
            "generation_timings": self.gen_timings[-300:],
        }


def ensure_exports_dir() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)


def executor_factory(max_workers: int):
    if EXECUTOR_KIND == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    if EXECUTOR_KIND == "process":
        return ProcessPoolExecutor(max_workers=max_workers, initializer=set_single_thread_blas_env)
    raise ValueError("STEP84_EXECUTOR_KIND must be 'thread' or 'process'")


def set_single_thread_blas_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def random_signed_values(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=count)
    magnitudes = rng.uniform(low, high, size=count)
    return signs * magnitudes


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        cast = float(value)
        return cast if math.isfinite(cast) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def fitness_key(individual: "Individual") -> tuple[float, float, int, str]:
    return (
        float(individual.fitness),
        float(individual.fro_residual),
        support_complexity(individual),
        individual.support_hash,
    )


def is_better(candidate: "Individual", incumbent: "Individual | None") -> bool:
    if incumbent is None:
        return True
    left = fitness_key(candidate)
    right = fitness_key(incumbent)
    return left < right


def is_meaningful_numeric_improvement(candidate: "Individual", incumbent: "Individual | None") -> bool:
    if incumbent is None:
        return True
    if float(candidate.fitness) < float(incumbent.fitness) - 1e-15:
        return True
    if abs(float(candidate.fitness) - float(incumbent.fitness)) <= 1e-15 and float(candidate.fro_residual) < float(incumbent.fro_residual) - 1e-12:
        return True
    return False


def short_origin_label(label: str) -> str:
    if len(label) <= 48:
        return label
    return label[:45] + "..."


@dataclass
class SparseTermGene:
    alpha_support: tuple[int, ...]
    beta_support: tuple[int, ...]
    gamma_support: tuple[int, ...]
    alpha_values: np.ndarray
    beta_values: np.ndarray
    gamma_values: np.ndarray

    def copy(self) -> "SparseTermGene":
        return SparseTermGene(
            tuple(self.alpha_support),
            tuple(self.beta_support),
            tuple(self.gamma_support),
            np.asarray(self.alpha_values, dtype=np.float64).copy(),
            np.asarray(self.beta_values, dtype=np.float64).copy(),
            np.asarray(self.gamma_values, dtype=np.float64).copy(),
        )

    def signature(self) -> tuple[int, int, int]:
        return (len(self.alpha_support), len(self.beta_support), len(self.gamma_support))

    def dense_factors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = np.zeros(9, dtype=np.float64)
        beta = np.zeros(9, dtype=np.float64)
        gamma = np.zeros(9, dtype=np.float64)
        alpha[list(self.alpha_support)] = self.alpha_values
        beta[list(self.beta_support)] = self.beta_values
        gamma[list(self.gamma_support)] = self.gamma_values
        return alpha, beta, gamma

    def update_from_dense(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> None:
        self.alpha_values = np.array([float(alpha[index]) for index in self.alpha_support], dtype=np.float64)
        self.beta_values = np.array([float(beta[index]) for index in self.beta_support], dtype=np.float64)
        self.gamma_values = np.array([float(gamma[index]) for index in self.gamma_support], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "alpha_support": [int(index) for index in self.alpha_support],
            "beta_support": [int(index) for index in self.beta_support],
            "gamma_support": [int(index) for index in self.gamma_support],
            "alpha_values": [float(value) for value in self.alpha_values],
            "beta_values": [float(value) for value in self.beta_values],
            "gamma_values": [float(value) for value in self.gamma_values],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SparseTermGene":
        return cls(
            tuple(int(index) for index in payload["alpha_support"]),
            tuple(int(index) for index in payload["beta_support"]),
            tuple(int(index) for index in payload["gamma_support"]),
            np.asarray(payload["alpha_values"], dtype=np.float64),
            np.asarray(payload["beta_values"], dtype=np.float64),
            np.asarray(payload["gamma_values"], dtype=np.float64),
        )


@dataclass
class Individual:
    terms: list[SparseTermGene]
    origin: str
    fitness: float = math.inf
    fro_residual: float = math.inf
    support_signature: str = ""
    support_hash: str = ""
    newton_polished: bool = False

    def copy(self) -> "Individual":
        return Individual(
            [term.copy() for term in self.terms],
            str(self.origin),
            float(self.fitness),
            float(self.fro_residual),
            str(self.support_signature),
            str(self.support_hash),
            bool(self.newton_polished),
        )

    def reset_metrics(self) -> None:
        self.fitness = math.inf
        self.fro_residual = math.inf
        self.newton_polished = False
        self.support_signature = ""
        self.support_hash = ""

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "fitness": finite_or_none(self.fitness),
            "fro_residual": finite_or_none(self.fro_residual),
            "support_signature": self.support_signature,
            "support_hash": self.support_hash,
            "newton_polished": bool(self.newton_polished),
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Individual":
        fitness = payload.get("fitness")
        fro_residual = payload.get("fro_residual")
        return cls(
            [SparseTermGene.from_dict(term_payload) for term_payload in payload["terms"]],
            str(payload.get("origin", "resume")),
            math.inf if fitness is None else float(fitness),
            math.inf if fro_residual is None else float(fro_residual),
            str(payload.get("support_signature", "")),
            str(payload.get("support_hash", "")),
            bool(payload.get("newton_polished", False)),
        )


def support_complexity(individual: Individual) -> int:
    return sum(len(term.alpha_support) + len(term.beta_support) + len(term.gamma_support) for term in individual.terms)


def term_rank1_norm(term: SparseTermGene) -> float:
    return float(np.linalg.norm(term.alpha_values) * np.linalg.norm(term.beta_values) * np.linalg.norm(term.gamma_values))


def apply_floor_to_values(values: np.ndarray, rng: np.random.Generator) -> None:
    for index, value in enumerate(values):
        if math.isfinite(float(value)) and abs(float(value)) >= ACTIVE_VALUE_FLOOR:
            continue
        sign = 1.0 if float(value) >= 0.0 else -1.0
        if not math.isfinite(float(value)) or abs(float(value)) == 0.0:
            sign = -1.0 if rng.random() < 0.5 else 1.0
        values[index] = sign * ACTIVE_VALUE_FLOOR


def rebalance_term(term: SparseTermGene, rng: np.random.Generator) -> None:
    apply_floor_to_values(term.alpha_values, rng)
    apply_floor_to_values(term.beta_values, rng)
    apply_floor_to_values(term.gamma_values, rng)
    alpha_norm = max(float(np.linalg.norm(term.alpha_values)), ACTIVE_VALUE_FLOOR)
    beta_norm = max(float(np.linalg.norm(term.beta_values)), ACTIVE_VALUE_FLOOR)
    gamma_norm = max(float(np.linalg.norm(term.gamma_values)), ACTIVE_VALUE_FLOOR)
    geometric_mean = (alpha_norm * beta_norm * gamma_norm) ** (1.0 / 3.0)
    term.alpha_values *= geometric_mean / alpha_norm
    term.beta_values *= geometric_mean / beta_norm
    term.gamma_values *= geometric_mean / gamma_norm
    apply_floor_to_values(term.alpha_values, rng)
    apply_floor_to_values(term.beta_values, rng)
    apply_floor_to_values(term.gamma_values, rng)


def canonicalize_term_order(individual: Individual) -> Individual:
    individual.terms = sorted(
        individual.terms,
        key=lambda term: (
            -term_rank1_norm(term),
            term.signature(),
            term.alpha_support,
            term.beta_support,
            term.gamma_support,
        ),
    )
    return individual


def finalize_individual_structure(individual: Individual, rng: np.random.Generator) -> Individual:
    for term in individual.terms:
        rebalance_term(term, rng)
    canonicalize_term_order(individual)
    individual.support_signature = support_signature_string(individual)
    individual.support_hash = support_pattern_hash(individual)
    return individual


def support_signature_string(individual: Individual) -> str:
    alpha_counts = [len(term.alpha_support) for term in individual.terms]
    beta_counts = [len(term.beta_support) for term in individual.terms]
    gamma_counts = [len(term.gamma_support) for term in individual.terms]
    return f"({int(np.median(alpha_counts))},{int(np.median(beta_counts))},{int(np.median(gamma_counts))})"


def support_histogram(individual: Individual) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for term in individual.terms:
        key = f"({len(term.alpha_support)},{len(term.beta_support)},{len(term.gamma_support)})"
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: item[0]))


def parse_support_signature(signature: str) -> tuple[int, int, int] | None:
    cleaned = signature.strip().strip("()")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def is_step83b_favored_signature(signature: str) -> bool:
    parsed = parse_support_signature(signature)
    if parsed is None:
        return False
    return max(parsed) <= 4 and sum(parsed) <= 11


def is_exact_333_signature(signature: str) -> bool:
    return signature == "(3,3,3)"


def variable_bucket(variable_count_estimate: int) -> str:
    if variable_count_estimate < 0:
        return "na"
    lower = (int(variable_count_estimate) // SHADOW_VARIABLE_BUCKET) * SHADOW_VARIABLE_BUCKET
    upper = lower + SHADOW_VARIABLE_BUCKET - 1
    return f"{lower:03d}-{upper:03d}"


def variable_count_estimate(individual: Individual) -> int:
    # Fast serial proxy for shadow bucketing. This tracks support volume rather than
    # rebuilding the exact reduced support model in the main thread.
    return support_complexity(individual)


def selection_shadow_key(individual: Individual) -> tuple[str, str]:
    return (
        individual.support_signature or support_signature_string(individual),
        variable_bucket(variable_count_estimate(individual)),
    )


def dedupe_by_support_hash(individuals: list[Individual], presorted: bool = False) -> list[Individual]:
    seen: set[str] = set()
    unique: list[Individual] = []
    source = individuals if presorted else sorted(individuals, key=fitness_key)
    for individual in source:
        support_hash = individual.support_hash or support_pattern_hash(individual)
        if support_hash in seen:
            continue
        seen.add(support_hash)
        unique.append(individual)
    return unique


def build_shadow_pool(population: list[Individual]) -> list[Individual]:
    grouped: dict[tuple[str, str], list[Individual]] = defaultdict(list)
    for individual in population:
        grouped[selection_shadow_key(individual)].append(individual)
    representatives = [min(group, key=fitness_key).copy() for group in grouped.values()]
    representatives.sort(key=fitness_key)
    return representatives


def population_shadow_metrics(population: list[Individual]) -> dict:
    signature_histogram: dict[str, int] = defaultdict(int)
    grouped: dict[tuple[str, str], list[Individual]] = defaultdict(list)
    signature_333_count = 0
    favored_regime_count = 0
    favored_regime_best_fitness = math.inf

    for individual in population:
        signature = individual.support_signature or support_signature_string(individual)
        signature_histogram[signature] += 1
        grouped[selection_shadow_key(individual)].append(individual)
        if signature == "(3,3,3)":
            signature_333_count += 1
        if is_step83b_favored_signature(signature):
            favored_regime_count += 1
            favored_regime_best_fitness = min(favored_regime_best_fitness, float(individual.fitness))

    shadow_pool = [min(group, key=fitness_key) for group in grouped.values()]
    sorted_histogram = dict(sorted(signature_histogram.items(), key=lambda item: item[0]))
    return {
        "shadow_pool_size": len(shadow_pool),
        "niche_count": len(grouped),
        "signature_333_count": signature_333_count,
        "favored_regime_count": favored_regime_count,
        "favored_regime_best_fitness": None if not math.isfinite(favored_regime_best_fitness) else float(favored_regime_best_fitness),
        "signature_histogram": sorted_histogram,
    }


def support_pattern_hash(individual: Individual) -> str:
    parts = []
    for term in individual.terms:
        parts.append(
            "|".join(
                [
                    ",".join(str(index) for index in term.alpha_support),
                    ",".join(str(index) for index in term.beta_support),
                    ",".join(str(index) for index in term.gamma_support),
                ]
            )
        )
    return ";".join(parts)


def sample_support_count(rng: np.random.Generator, low: int = INIT_SUPPORT_MIN, high: int = INIT_SUPPORT_MAX) -> int:
    return int(rng.integers(low, high + 1))


def random_support_positions(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in rng.choice(np.arange(9), size=count, replace=False).tolist()))


def build_random_term_gene(rng: np.random.Generator) -> SparseTermGene:
    alpha_support = random_support_positions(rng, sample_support_count(rng))
    beta_count = sample_support_count(rng)
    preferred_beta = live_compatible_beta_positions(alpha_support)
    beta_support = fiber_biased_support(rng, beta_count, preferred_beta)
    gamma_count = sample_support_count(rng)
    preferred_gamma = live_compatible_gamma_positions(alpha_support, beta_support)
    gamma_support = fiber_biased_support(rng, gamma_count, preferred_gamma)
    return SparseTermGene(
        alpha_support,
        beta_support,
        gamma_support,
        random_signed_values(rng, len(alpha_support), COEFF_INIT_LOW, COEFF_INIT_HIGH),
        random_signed_values(rng, len(beta_support), COEFF_INIT_LOW, COEFF_INIT_HIGH),
        random_signed_values(rng, len(gamma_support), COEFF_INIT_LOW, COEFF_INIT_HIGH),
    )


def build_random_individual(rng: np.random.Generator, origin: str) -> Individual:
    individual = Individual([build_random_term_gene(rng) for _ in range(RANK_VALUE)], origin)
    return finalize_individual_structure(individual, rng)


def coerce_dense_factor(
    dense: np.ndarray,
    rng: np.random.Generator,
    min_nnz: int = SUPPORT_MIN,
    max_nnz: int = SUPPORT_MAX,
) -> tuple[tuple[int, ...], np.ndarray]:
    nonzero = [int(index) for index in np.flatnonzero(np.abs(dense) > 0)]
    if len(nonzero) > max_nnz:
        ranked = sorted(nonzero, key=lambda index: (-abs(float(dense[index])), index))
        nonzero = ranked[:max_nnz]
    support = set(nonzero)
    while len(support) < min_nnz:
        candidate = int(rng.integers(0, 9))
        if candidate in support:
            continue
        support.add(candidate)
        dense[candidate] = random_signed_values(rng, 1, SUPPORT_ADD_LOW, SUPPORT_ADD_HIGH)[0]
    support_tuple = tuple(sorted(support))
    values = np.array([float(dense[index]) for index in support_tuple], dtype=np.float64)
    apply_floor_to_values(values, rng)
    return support_tuple, values


def individual_from_terms(terms: list[Term], origin: str, rng: np.random.Generator) -> Individual:
    genes: list[SparseTermGene] = []
    for term in terms:
        alpha = np.asarray(term.alpha, dtype=np.float64).reshape(9).copy()
        beta = np.asarray(term.beta, dtype=np.float64).reshape(9).copy()
        gamma = np.asarray(term.gamma, dtype=np.float64).reshape(9).copy()
        alpha_support, alpha_values = coerce_dense_factor(alpha, rng)
        beta_support, beta_values = coerce_dense_factor(beta, rng)
        gamma_support, gamma_values = coerce_dense_factor(gamma, rng)
        genes.append(SparseTermGene(alpha_support, beta_support, gamma_support, alpha_values, beta_values, gamma_values))
    individual = Individual(genes, origin)
    return finalize_individual_structure(individual, rng)


def load_warm_seed_individuals() -> list[Individual]:
    rows = read_csv_rows(TRACK2_SCREENING_PATH)
    if not rows:
        return []
    viable_rows = [row for row in rows if row.get("status") == "screened_ok"]
    viable_rows.sort(
        key=lambda row: (
            float(row["square_jacobian_condition_number"]),
            float(row["start_full_tensor_max_abs_residual"]),
            row["case_id"],
        )
    )
    rng = np.random.default_rng(SEED + 901)
    seeds: list[Individual] = []
    for row in viable_rows[:WARM_SEED_COUNT]:
        terms = build_track2_terms_from_row(row)
        seeds.append(individual_from_terms(terms, f"warm_seed:{row['case_id']}", rng))
    return seeds


def initialize_populations() -> tuple[list[list[Individual]], list[np.random.Generator], int]:
    island_rngs = [np.random.default_rng(SEED + 1000 * (index + 1)) for index in range(NUM_ISLANDS)]
    warm_seeds = load_warm_seed_individuals()
    master_rng = np.random.default_rng(SEED)
    all_individuals = [seed.copy() for seed in warm_seeds]
    while len(all_individuals) < TOTAL_POP:
        all_individuals.append(build_random_individual(master_rng, f"random:{len(all_individuals):02d}"))
    populations = [[] for _ in range(NUM_ISLANDS)]
    cursor = 0
    for island_index, population_size in enumerate(ISLAND_POPULATIONS):
        next_cursor = cursor + population_size
        populations[island_index].extend(individual.copy() for individual in all_individuals[cursor:next_cursor])
        cursor = next_cursor
    return populations, island_rngs, len(warm_seeds)


def dense_factor_arrays(individual: Individual) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.zeros((RANK_VALUE, 9), dtype=np.float64)
    beta = np.zeros((RANK_VALUE, 9), dtype=np.float64)
    gamma = np.zeros((RANK_VALUE, 9), dtype=np.float64)
    for term_index, term in enumerate(individual.terms):
        alpha[term_index, list(term.alpha_support)] = term.alpha_values
        beta[term_index, list(term.beta_support)] = term.beta_values
        gamma[term_index, list(term.gamma_support)] = term.gamma_values
    return alpha, beta, gamma


def update_individual_from_dense(individual: Individual, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> None:
    for term_index, term in enumerate(individual.terms):
        term.update_from_dense(alpha[term_index], beta[term_index], gamma[term_index])


def rebalance_dense_terms(
    individual: Individual,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for term_index, term in enumerate(individual.terms):
        alpha_support = list(term.alpha_support)
        beta_support = list(term.beta_support)
        gamma_support = list(term.gamma_support)
        for support_list, factor in ((alpha_support, alpha), (beta_support, beta), (gamma_support, gamma)):
            for support_index in support_list:
                value = float(factor[term_index, support_index])
                if math.isfinite(value) and abs(value) >= ACTIVE_VALUE_FLOOR:
                    continue
                sign = -1.0 if rng.random() < 0.5 else 1.0
                if math.isfinite(value) and value < 0.0:
                    sign = -1.0
                factor[term_index, support_index] = sign * ACTIVE_VALUE_FLOOR
        alpha_norm = max(float(np.linalg.norm(alpha[term_index, alpha_support])), ACTIVE_VALUE_FLOOR)
        beta_norm = max(float(np.linalg.norm(beta[term_index, beta_support])), ACTIVE_VALUE_FLOOR)
        gamma_norm = max(float(np.linalg.norm(gamma[term_index, gamma_support])), ACTIVE_VALUE_FLOOR)
        geometric_mean = (alpha_norm * beta_norm * gamma_norm) ** (1.0 / 3.0)
        alpha[term_index, alpha_support] *= geometric_mean / alpha_norm
        beta[term_index, beta_support] *= geometric_mean / beta_norm
        gamma[term_index, gamma_support] *= geometric_mean / gamma_norm


def solve_linear_factor(
    individual: Individual,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    factor_name: str,
) -> None:
    variable_count = sum(len(getattr(term, f"{factor_name}_support")) for term in individual.terms)
    if variable_count == 0:
        return
    design = np.zeros((TARGET_FLAT.size, variable_count), dtype=np.float64)
    locations: list[tuple[int, int]] = []
    column_index = 0
    for term_index, term in enumerate(individual.terms):
        if factor_name == "alpha":
            outer = np.outer(beta[term_index], gamma[term_index])
            for flat_index in term.alpha_support:
                design[:, column_index].reshape(9, 9, 9)[flat_index, :, :] = outer
                locations.append((term_index, flat_index))
                column_index += 1
        elif factor_name == "beta":
            outer = np.outer(alpha[term_index], gamma[term_index])
            for flat_index in term.beta_support:
                design[:, column_index].reshape(9, 9, 9)[:, flat_index, :] = outer
                locations.append((term_index, flat_index))
                column_index += 1
        else:
            outer = np.outer(alpha[term_index], beta[term_index])
            for flat_index in term.gamma_support:
                design[:, column_index].reshape(9, 9, 9)[:, :, flat_index] = outer
                locations.append((term_index, flat_index))
                column_index += 1
    solution, *_ = np.linalg.lstsq(design, TARGET_FLAT, rcond=None)
    for value, (term_index, flat_index) in zip(solution, locations, strict=False):
        if factor_name == "alpha":
            alpha[term_index, flat_index] = float(value)
        elif factor_name == "beta":
            beta[term_index, flat_index] = float(value)
        else:
            gamma[term_index, flat_index] = float(value)


def alternating_least_squares(individual: Individual, eval_seed: int) -> Individual:
    if ALS_SWEEPS <= 0:
        return individual
    rng = np.random.default_rng(eval_seed)
    alpha, beta, gamma = dense_factor_arrays(individual)
    for _ in range(ALS_SWEEPS):
        solve_linear_factor(individual, alpha, beta, gamma, "alpha")
        solve_linear_factor(individual, alpha, beta, gamma, "beta")
        solve_linear_factor(individual, alpha, beta, gamma, "gamma")
        rebalance_dense_terms(individual, alpha, beta, gamma, rng)
    update_individual_from_dense(individual, alpha, beta, gamma)
    return finalize_individual_structure(individual, rng)


def individual_to_terms(individual: Individual, source_label: str) -> list[Term]:
    terms: list[Term] = []
    for term_index, term in enumerate(individual.terms, start=1):
        alpha, beta, gamma = term.dense_factors()
        terms.append(
            Term(
                f"t{term_index:02d}",
                source_label,
                alpha.reshape(3, 3),
                beta.reshape(3, 3),
                gamma.reshape(3, 3),
            )
        )
    return terms


def residual_metrics(individual: Individual) -> tuple[float, float]:
    alpha, beta, gamma = dense_factor_arrays(individual)
    candidate_tensor = full_tensor_from_factors(alpha, beta, gamma)
    residual = candidate_tensor - TARGET_TENSOR
    return float(np.max(np.abs(residual))), float(np.linalg.norm(residual))


def support_fixed_polish(
    individual: Individual,
    eval_seed: int,
) -> tuple[Individual, float, float, bool, int | None]:
    terms = individual_to_terms(individual, "step84_polish")
    try:
        model = build_support_model(terms)
        witness_vector = witness_free_vector(model)
        model, _independent_rank, _dropped_count = reduce_model_to_independent_variables(model, witness_vector)
        witness_vector = witness_free_vector(model)
        variable_count = len(model.variable_specs)
        if variable_count > POLISH_VARIABLE_CAP:
            fitness, fro_residual = residual_metrics(individual)
            return individual, fitness, fro_residual, False, variable_count

        def residual_fn(free_vector: np.ndarray) -> np.ndarray:
            alpha, beta, gamma = unpack_free_vector(model, free_vector)
            return evaluate_coordinates(alpha, beta, gamma, FULL_COORDINATES) - TARGET_FLAT

        def jacobian_fn(free_vector: np.ndarray) -> np.ndarray:
            return build_jacobian(model, free_vector, FULL_COORDINATES)

        result = least_squares(
            residual_fn,
            witness_vector,
            jac=jacobian_fn,
            method="trf",
            loss="linear",
            x_scale="jac",
            max_nfev=POLISH_MAX_NFEV,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        polished_alpha, polished_beta, polished_gamma = unpack_free_vector(model, result.x)
        polished = individual.copy()
        update_individual_from_dense(polished, polished_alpha, polished_beta, polished_gamma)
        polished = finalize_individual_structure(polished, np.random.default_rng(eval_seed + 17))
        fitness, fro_residual = residual_metrics(polished)
        if not math.isfinite(fitness) or not math.isfinite(fro_residual):
            fitness, fro_residual = residual_metrics(individual)
            return individual, fitness, fro_residual, False, variable_count
        return polished, fitness, fro_residual, True, variable_count
    except Exception:
        fitness, fro_residual = residual_metrics(individual)
        return individual, fitness, fro_residual, False, None


def evaluate_individual_worker(payload: dict) -> dict:
    set_single_thread_blas_env()
    _pt = {}
    _t0 = time.perf_counter()
    eval_seed = int(payload["eval_seed"])
    individual = Individual.from_dict(payload["individual"])
    rng = np.random.default_rng(eval_seed)
    _t1 = time.perf_counter()
    _pt["deser_ms"] = (_t1 - _t0) * 1000
    try:
        individual = finalize_individual_structure(individual, rng)
        _t2 = time.perf_counter()
        _pt["finalize_ms"] = (_t2 - _t1) * 1000
        individual = alternating_least_squares(individual, eval_seed)
        _t3 = time.perf_counter()
        _pt["als_ms"] = (_t3 - _t2) * 1000
        fitness, fro_residual = residual_metrics(individual)
        _t4 = time.perf_counter()
        _pt["residual_ms"] = (_t4 - _t3) * 1000
        polished = False
        variable_count = None
        if math.isfinite(fitness) and fitness < NEWTON_THRESHOLD:
            polished_candidate, polished_fitness, polished_fro, polished, variable_count = support_fixed_polish(individual, eval_seed)
            if math.isfinite(polished_fitness) and (polished_fitness < fitness or (abs(polished_fitness - fitness) <= 1e-15 and polished_fro < fro_residual)):
                individual = polished_candidate
                fitness = polished_fitness
                fro_residual = polished_fro
        _t5 = time.perf_counter()
        _pt["polish_ms"] = (_t5 - _t4) * 1000
        # Soft dead-leakage penalty: nudge search toward fiber-respecting solutions
        if math.isfinite(fitness):
            alpha_m, beta_m, _gamma_m = dense_factor_arrays(individual)
            fitness += structural_penalty(alpha_m, beta_m, _gamma_m)
        _t6 = time.perf_counter()
        _pt["penalty_ms"] = (_t6 - _t5) * 1000
        individual.fitness = float(fitness)
        individual.fro_residual = float(fro_residual)
        individual.newton_polished = bool(polished)
        individual.support_signature = support_signature_string(individual)
        individual.support_hash = canonical_support_hash([
            (t.alpha_support, t.beta_support, t.gamma_support) for t in individual.terms
        ])
        _t7 = time.perf_counter()
        _pt["hash_ms"] = (_t7 - _t6) * 1000
        _pt["total_ms"] = (_t7 - _t0) * 1000
        return {
            "status": "ok",
            "individual": individual.to_dict(),
            "fitness": float(individual.fitness),
            "fro_residual": float(individual.fro_residual),
            "newton_polished": bool(individual.newton_polished),
            "support_signature": individual.support_signature,
            "support_hash": individual.support_hash,
            "variable_count": variable_count,
            "timing": _pt,
        }
    except Exception as exc:
        fallback = Individual.from_dict(payload["individual"])
        fallback.fitness = math.inf
        fallback.fro_residual = math.inf
        fallback.newton_polished = False
        fallback.support_signature = support_signature_string(fallback)
        fallback.support_hash = canonical_support_hash([
            (t.alpha_support, t.beta_support, t.gamma_support) for t in fallback.terms
        ])
        return {
            "status": "error",
            "error": str(exc),
            "individual": fallback.to_dict(),
            "fitness": math.inf,
            "fro_residual": math.inf,
            "newton_polished": False,
            "support_signature": fallback.support_signature,
            "support_hash": fallback.support_hash,
            "variable_count": None,
        }


def evaluate_batches(
    executor: ProcessPoolExecutor,
    populations: list[list[Individual]],
    generation: int,
) -> tuple[list[list[Individual]], dict[int, int], int, bool]:
    payloads = []
    locations: list[tuple[int, int]] = []
    for island_index, population in enumerate(populations):
        for individual_index, individual in enumerate(population):
            payloads.append(
                {
                    "eval_seed": SEED + generation * 100000 + island_index * 1000 + individual_index,
                    "individual": individual.to_dict(),
                }
            )
            locations.append((island_index, individual_index))

    evaluated = [[None for _ in island] for island in populations]
    polished_counts = {index: 0 for index in range(len(populations))}
    exact_hit_found = False
    results = list(executor.map(evaluate_individual_worker, payloads))
    for (island_index, individual_index), result in zip(locations, results, strict=False):
        individual = Individual.from_dict(result["individual"])
        individual.fitness = float(result.get("fitness", math.inf))
        individual.fro_residual = float(result.get("fro_residual", math.inf))
        individual.newton_polished = bool(result.get("newton_polished", False))
        individual.support_signature = str(result.get("support_signature", support_signature_string(individual)))
        individual.support_hash = str(result.get("support_hash", support_pattern_hash(individual)))
        evaluated[island_index][individual_index] = individual
        if individual.newton_polished:
            polished_counts[island_index] += 1
        if math.isfinite(individual.fitness) and individual.fitness < HIT_THRESHOLD:
            exact_hit_found = True
    return evaluated, polished_counts, len(payloads), exact_hit_found


def evaluate_initial_populations_async(
    executor: ProcessPoolExecutor,
    populations: list[list[Individual]],
) -> tuple[list[list[Individual]], dict[int, int], int, bool, list[dict]]:
    pending = {}
    evaluated = [[None for _ in island] for island in populations]
    polished_counts = {index: 0 for index in range(len(populations))}
    exact_hit_found = False
    evaluated_count = 0
    timings: list[dict] = []

    for island_index, population in enumerate(populations):
        for individual_index, individual in enumerate(population):
            payload = {
                "eval_seed": SEED + island_index * 1000 + individual_index,
                "individual": individual.to_dict(),
            }
            future = executor.submit(evaluate_individual_worker, payload)
            pending[future] = (island_index, individual_index)

    while pending:
        done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            island_index, individual_index = pending.pop(future)
            result = future.result()
            individual = Individual.from_dict(result["individual"])
            individual.fitness = float(result.get("fitness", math.inf))
            individual.fro_residual = float(result.get("fro_residual", math.inf))
            individual.newton_polished = bool(result.get("newton_polished", False))
            individual.support_signature = str(result.get("support_signature", support_signature_string(individual)))
            individual.support_hash = str(result.get("support_hash", support_pattern_hash(individual)))
            evaluated[island_index][individual_index] = individual
            evaluated_count += 1
            timing = result.get("timing")
            if timing:
                timings.append(timing)
            if individual.newton_polished:
                polished_counts[island_index] += 1
            if math.isfinite(individual.fitness) and individual.fitness < HIT_THRESHOLD:
                exact_hit_found = True

    return evaluated, polished_counts, evaluated_count, exact_hit_found, timings


def build_single_offspring(
    population: list[Individual],
    rng: np.random.Generator,
    best_fitness: float,
    shadow_pool: list[Individual],
) -> Individual:
    if rng.random() < CROSSOVER_RATE:
        parent_a, parent_b = tournament_pick(population, rng, count=2, shadow_pool=shadow_pool)
        child = crossover(parent_a, parent_b, rng)
    else:
        child = tournament_pick(population, rng, count=1, shadow_pool=shadow_pool)[0]
    return mutate(child, rng, best_fitness)


_INTEGRATION_BATCH_SIZE = 10
_SHADOW_REBUILD_BATCH = 10


def integrate_offspring_batch(
    populations: list[list[Individual]],
    island_index: int,
    offspring_batch: list[Individual],
    variable_count_cache: dict[str, int],
) -> None:
    """Integrate a batch of offspring into an island in one select_survivors call."""
    if not offspring_batch:
        return
    populations[island_index] = select_survivors(
        populations[island_index],
        offspring_batch,
        variable_count_cache,
        ISLAND_POPULATIONS[island_index],
    )


def integrate_evaluated_offspring(
    populations: list[list[Individual]],
    island_index: int,
    offspring: Individual,
    variable_count_cache: dict[str, int],
) -> None:
    populations[island_index] = select_survivors(
        populations[island_index],
        [offspring],
        variable_count_cache,
        ISLAND_POPULATIONS[island_index],
    )


def steady_state_generation(offspring_evaluations_completed: int) -> int:
    if TOTAL_POP <= 0:
        return 0
    return offspring_evaluations_completed // TOTAL_POP


def adaptive_sigma(best_fitness: float) -> float:
    scale = 0.01 + 0.05 * math.sqrt(max(best_fitness, 1e-12))
    return max(COEFF_SIGMA_MIN, min(COEFF_SIGMA_MAX, scale))


def tournament_pick(
    population: list[Individual],
    rng: np.random.Generator,
    count: int = 1,
    shadow_pool: list[Individual] | None = None,
) -> list[Individual]:
    source = population
    if shadow_pool and rng.random() < SHADOW_PARENT_RATE:
        source = shadow_pool
    size = min(TOURNAMENT_SIZE, len(source))
    picks: list[Individual] = []
    for _ in range(count):
        contender_indices = rng.choice(np.arange(len(source)), size=size, replace=False)
        contenders = [source[int(index)] for index in contender_indices]
        picks.append(min(contenders, key=fitness_key).copy())
    return picks


def term_level_crossover(parent_a: Individual, parent_b: Individual, rng: np.random.Generator) -> Individual:
    cut = int(rng.integers(1, RANK_VALUE))
    child_terms = [term.copy() for term in parent_a.terms[:cut]] + [term.copy() for term in parent_b.terms[cut:]]
    child = Individual(child_terms, short_origin_label(f"term_xover:{parent_a.support_signature}|{parent_b.support_signature}"))
    return finalize_individual_structure(child, rng)


def factor_level_crossover(parent_a: Individual, parent_b: Individual, rng: np.random.Generator) -> Individual:
    better_parent = parent_a if fitness_key(parent_a) <= fitness_key(parent_b) else parent_b
    child = better_parent.copy()
    term_index = int(rng.integers(0, RANK_VALUE))
    term_a = parent_a.terms[term_index]
    term_b = parent_b.terms[term_index]
    term_g = better_parent.terms[term_index]
    child.terms[term_index] = SparseTermGene(
        tuple(term_a.alpha_support),
        tuple(term_b.beta_support),
        tuple(term_g.gamma_support),
        np.asarray(term_a.alpha_values, dtype=np.float64).copy(),
        np.asarray(term_b.beta_values, dtype=np.float64).copy(),
        np.asarray(term_g.gamma_values, dtype=np.float64).copy(),
    )
    child.origin = short_origin_label(f"factor_xover:{parent_a.support_signature}|{parent_b.support_signature}")
    return finalize_individual_structure(child, rng)


def crossover(parent_a: Individual, parent_b: Individual, rng: np.random.Generator) -> Individual:
    if rng.random() < FACTOR_CROSSOVER_RATE:
        return factor_level_crossover(parent_a, parent_b, rng)
    return term_level_crossover(parent_a, parent_b, rng)


def mutate_support(individual: Individual, rng: np.random.Generator) -> None:
    term = individual.terms[int(rng.integers(0, RANK_VALUE))]
    factor_name = str(rng.choice(np.array(["alpha", "beta", "gamma"])))
    support = list(getattr(term, f"{factor_name}_support"))
    values = np.asarray(getattr(term, f"{factor_name}_values"), dtype=np.float64).copy()
    do_add = rng.random() < SUPPORT_ADD_PROB / max(SUPPORT_ADD_PROB + SUPPORT_DROP_PROB, 1e-12)
    if len(support) <= SUPPORT_MIN:
        do_add = True
    elif len(support) >= SUPPORT_MAX:
        do_add = False

    if do_add:
        missing = [index for index in range(9) if index not in support]
        if missing:
            # Fiber-biased: prefer positions forming live (a,b) pairs
            if factor_name == "beta":
                preferred = set(live_compatible_beta_positions(term.alpha_support))
                preferred_missing = [i for i in missing if i in preferred]
            elif factor_name == "alpha":
                # Reverse: prefer alpha positions compatible with existing beta
                s_values = {b // 3 for b in term.beta_support}
                preferred_missing = [i for i in missing if i % 3 in s_values]
            elif factor_name == "gamma":
                preferred = set(live_compatible_gamma_positions(term.alpha_support, term.beta_support))
                preferred_missing = [i for i in missing if i in preferred]
            else:
                preferred_missing = []
            if preferred_missing and rng.random() < 0.7:
                new_index = int(rng.choice(np.array(preferred_missing, dtype=np.int64)))
            else:
                new_index = int(rng.choice(np.array(missing, dtype=np.int64)))
            support.append(new_index)
            values = np.append(values, random_signed_values(rng, 1, SUPPORT_ADD_LOW, SUPPORT_ADD_HIGH))
    else:
        if len(support) > SUPPORT_MIN:
            drop_offset = int(rng.integers(0, len(support)))
            support.pop(drop_offset)
            values = np.delete(values, drop_offset)

    order = np.argsort(np.array(support, dtype=np.int64))
    setattr(term, f"{factor_name}_support", tuple(int(support[index]) for index in order))
    values = np.asarray(values[order], dtype=np.float64)
    apply_floor_to_values(values, rng)
    setattr(term, f"{factor_name}_values", values)


def mutate_coefficients(individual: Individual, rng: np.random.Generator, best_fitness: float) -> None:
    sigma = adaptive_sigma(best_fitness)
    term = individual.terms[int(rng.integers(0, RANK_VALUE))]
    factor_name = str(rng.choice(np.array(["alpha", "beta", "gamma"])))
    values = np.asarray(getattr(term, f"{factor_name}_values"), dtype=np.float64).copy()
    if values.size == 0:
        return
    value_index = int(rng.integers(0, values.size))
    values[value_index] += sigma * rng.standard_normal()
    apply_floor_to_values(values, rng)
    setattr(term, f"{factor_name}_values", values)


def replace_random_term(individual: Individual, rng: np.random.Generator) -> None:
    replace_index = int(rng.integers(0, RANK_VALUE))
    individual.terms[replace_index] = build_random_term_gene(rng)


def mutate(individual: Individual, rng: np.random.Generator, best_fitness: float) -> Individual:
    applied = False
    if rng.random() < MUTATION_SUPPORT_RATE:
        mutate_support(individual, rng)
        applied = True
    if rng.random() < MUTATION_COEFF_RATE:
        mutate_coefficients(individual, rng, best_fitness)
        applied = True
    if rng.random() < MUTATION_REPLACE_RATE:
        replace_random_term(individual, rng)
        applied = True
    if not applied:
        mutate_coefficients(individual, rng, best_fitness)
    individual.reset_metrics()
    return finalize_individual_structure(individual, rng)


def build_offspring(
    population: list[Individual],
    rng: np.random.Generator,
    best_fitness: float,
    shadow_pool: list[Individual],
    offspring_target: int,
) -> list[Individual]:
    offspring: list[Individual] = []
    while len(offspring) < offspring_target:
        if rng.random() < CROSSOVER_RATE:
            parent_a, parent_b = tournament_pick(population, rng, count=2, shadow_pool=shadow_pool)
            child = crossover(parent_a, parent_b, rng)
        else:
            child = tournament_pick(population, rng, count=1, shadow_pool=shadow_pool)[0]
        offspring.append(mutate(child, rng, best_fitness))
    return offspring


def select_survivors(
    population: list[Individual],
    offspring: list[Individual],
    variable_count_cache: dict[str, int],
    survivor_target: int,
) -> list[Individual]:
    # Sort once; timsort is O(N) when population is already sorted + appended offspring
    all_individuals = sorted(population + offspring, key=fitness_key)
    combined = dedupe_by_support_hash(all_individuals, presorted=True)
    if len(combined) <= survivor_target:
        return combined[:survivor_target]

    # combined is already sorted by fitness_key from dedupe_by_support_hash
    exact_333_candidates = [
        individual for individual in combined if is_exact_333_signature(individual.support_signature or support_signature_string(individual))
    ]
    # already sorted since combined is sorted
    exact_333_slots = min(
        len(exact_333_candidates),
        max(0, survivor_target - 1),
        int(math.ceil(survivor_target * SIGNATURE333_SURVIVOR_FRACTION)),
    )
    exact_333_survivors = [individual.copy() for individual in exact_333_candidates[:exact_333_slots]]

    remaining_target = survivor_target - len(exact_333_survivors)
    shadow_slots = 0 if remaining_target <= 1 else max(1, min(remaining_target - 1, int(math.ceil(remaining_target * SHADOW_SURVIVOR_FRACTION))))
    elite_slots = remaining_target - shadow_slots

    selected_hashes = {individual.support_hash for individual in exact_333_survivors}
    # combined is already sorted — no need to re-sort
    elites: list[Individual] = []
    for individual in combined:
        if len(elites) >= elite_slots:
            break
        if individual.support_hash not in selected_hashes:
            elites.append(individual)
    selected_hashes.update(individual.support_hash for individual in elites)

    grouped: dict[tuple[str, str], list[Individual]] = defaultdict(list)
    for individual in combined:
        grouped[selection_shadow_key(individual)].append(individual)

    shadow_candidates: list[Individual] = []
    for group in grouped.values():
        # groups preserve insertion order from sorted combined
        champion = group[0]
        if champion.support_hash not in selected_hashes:
            shadow_candidates.append(champion)
    shadow_candidates.sort(key=fitness_key)

    survivors = exact_333_survivors + elites + shadow_candidates[:shadow_slots]
    if len(survivors) < survivor_target:
        selected_hashes = {individual.support_hash for individual in survivors}
        for individual in combined:
            if len(survivors) >= survivor_target:
                break
            if individual.support_hash not in selected_hashes:
                survivors.append(individual)
                selected_hashes.add(individual.support_hash)
    survivors.sort(key=fitness_key)
    return survivors[:survivor_target]


def migrate(populations: list[list[Individual]], island_rngs: list[np.random.Generator]) -> None:
    if NUM_ISLANDS <= 1 or MIGRATION_SIZE <= 0:
        return
    migrants: list[tuple[int, int, Individual]] = []
    for island_index, population in enumerate(populations):
        sorted_population = sorted(population, key=fitness_key)
        for migrant in sorted_population[:MIGRATION_SIZE]:
            direction = -1 if island_rngs[island_index].random() < 0.5 else 1
            destination = (island_index + direction) % NUM_ISLANDS
            migrants.append((island_index, destination, migrant.copy()))
    for _source, destination, migrant in migrants:
        destination_population = populations[destination]
        destination_population.sort(key=fitness_key)
        destination_population[-1] = migrant
        destination_population.sort(key=fitness_key)


def compute_variable_count(individual: Individual, cache: dict[str, int]) -> int | None:
    support_hash = individual.support_hash or support_pattern_hash(individual)
    if support_hash in cache:
        return cache[support_hash]
    try:
        terms = individual_to_terms(individual, "step84_variable_count")
        model = build_support_model(terms)
        witness_vector = witness_free_vector(model)
        model, _independent_rank, _dropped_count = reduce_model_to_independent_variables(model, witness_vector)
        cache[support_hash] = len(model.variable_specs)
        return cache[support_hash]
    except Exception:
        return None


def append_log_rows(rows: list[dict]) -> None:
    if not rows:
        return
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_best_individual(
    individual: Individual,
    generation: int,
    variable_count_cache: dict[str, int],
    potential_exact_hit: bool,
) -> None:
    variable_count = compute_variable_count(individual, variable_count_cache)
    alpha, beta, gamma = dense_factor_arrays(individual)
    payload = {
        "generation_found": generation,
        "origin": short_origin_label(individual.origin),
        "fitness": float(individual.fitness),
        "fro_residual": float(individual.fro_residual),
        "support_signature": individual.support_signature or support_signature_string(individual),
        "support_histogram": support_histogram(individual),
        "variable_count": variable_count,
        "newton_polished": bool(individual.newton_polished),
        "potential_exact_hit": bool(potential_exact_hit),
        "verification": {
            "coordinate_count": 729,
            "max_abs_residual": float(individual.fitness),
            "fro_residual": float(individual.fro_residual),
            "threshold": HIT_THRESHOLD,
        },
        "terms": [],
    }
    for term_index, term in enumerate(individual.terms, start=1):
        payload["terms"].append(
            {
                "term_index": term_index,
                "support_signature": f"({len(term.alpha_support)},{len(term.beta_support)},{len(term.gamma_support)})",
                "alpha_support": [int(index) for index in term.alpha_support],
                "beta_support": [int(index) for index in term.beta_support],
                "gamma_support": [int(index) for index in term.gamma_support],
                "alpha_values": [float(value) for value in term.alpha_values],
                "beta_values": [float(value) for value in term.beta_values],
                "gamma_values": [float(value) for value in term.gamma_values],
            }
        )
        if BEST_EXPORT_INCLUDE_DENSE:
            payload["terms"][-1]["alpha_dense"] = [float(value) for value in alpha[term_index - 1]]
            payload["terms"][-1]["beta_dense"] = [float(value) for value in beta[term_index - 1]]
            payload["terms"][-1]["gamma_dense"] = [float(value) for value in gamma[term_index - 1]]
            payload["terms"][-1]["alpha_matrix"] = alpha[term_index - 1].reshape(3, 3).tolist()
            payload["terms"][-1]["beta_matrix"] = beta[term_index - 1].reshape(3, 3).tolist()
            payload["terms"][-1]["gamma_matrix"] = gamma[term_index - 1].reshape(3, 3).tolist()
    write_json(BEST_PATH, json_safe(payload))


def serialize_individual(individual: Individual, include_dense: bool = False) -> dict:
    alpha, beta, gamma = dense_factor_arrays(individual) if include_dense else (None, None, None)
    payload = {
        "origin": short_origin_label(individual.origin),
        "fitness": finite_or_none(individual.fitness),
        "fro_residual": finite_or_none(individual.fro_residual),
        "support_signature": individual.support_signature or support_signature_string(individual),
        "support_hash": individual.support_hash or support_pattern_hash(individual),
        "support_histogram": support_histogram(individual),
        "newton_polished": bool(individual.newton_polished),
        "terms": [],
    }
    for term_index, term in enumerate(individual.terms, start=1):
        term_payload = {
            "term_index": term_index,
            "alpha_support": [int(index) for index in term.alpha_support],
            "beta_support": [int(index) for index in term.beta_support],
            "gamma_support": [int(index) for index in term.gamma_support],
            "alpha_values": [float(value) for value in term.alpha_values],
            "beta_values": [float(value) for value in term.beta_values],
            "gamma_values": [float(value) for value in term.gamma_values],
        }
        if include_dense and alpha is not None and beta is not None and gamma is not None:
            term_payload["alpha_dense"] = [float(value) for value in alpha[term_index - 1]]
            term_payload["beta_dense"] = [float(value) for value in beta[term_index - 1]]
            term_payload["gamma_dense"] = [float(value) for value in gamma[term_index - 1]]
        payload["terms"].append(term_payload)
    return payload


def write_final_population(
    populations: list[list[Individual]],
    generation: int,
    total_evaluations: int,
    total_wall_seconds: float,
) -> None:
    payload = {
        "generation": generation,
        "total_evaluations": total_evaluations,
        "total_wall_seconds": total_wall_seconds,
        "islands": [
            {
                "island": island_index,
                "population": [serialize_individual(individual, include_dense=False) for individual in population],
            }
            for island_index, population in enumerate(populations)
        ],
    }
    write_json(FINAL_POP_PATH, json_safe(payload))


def checkpoint_payload(
    populations: list[list[Individual]],
    generation: int,
    total_evaluations: int,
    elapsed_wall_seconds: float,
    global_best: Individual | None,
    best_generation: int,
    improvement_history: list[list[float]],
    island_rngs: list[np.random.Generator],
    warm_seed_count: int,
    steady_state_completed: int,
) -> dict:
    return {
        "generation": generation,
        "total_evaluations": total_evaluations,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "best_generation": best_generation,
        "improvement_history": improvement_history,
        "warm_seed_count": warm_seed_count,
        "steady_state_completed": steady_state_completed,
        "global_best": None if global_best is None else serialize_individual(global_best, include_dense=False),
        "islands": [
            {
                "island": island_index,
                "population": [individual.to_dict() for individual in population],
                "rng_state": island_rngs[island_index].bit_generator.state,
            }
            for island_index, population in enumerate(populations)
        ],
    }


def write_checkpoint(
    populations: list[list[Individual]],
    generation: int,
    total_evaluations: int,
    elapsed_wall_seconds: float,
    global_best: Individual | None,
    best_generation: int,
    improvement_history: list[list[float]],
    island_rngs: list[np.random.Generator],
    warm_seed_count: int,
    steady_state_completed: int,
) -> None:
    checkpoint_path = EXPORTS / f"step84_checkpoint_gen{generation:04d}.json"
    payload = checkpoint_payload(
        populations,
        generation,
        total_evaluations,
        elapsed_wall_seconds,
        global_best,
        best_generation,
        improvement_history,
        island_rngs,
        warm_seed_count,
        steady_state_completed,
    )
    write_json(checkpoint_path, json_safe(payload))
    _purge_old_checkpoints()


def _purge_old_checkpoints() -> None:
    if CHECKPOINT_KEEP_LAST <= 0:
        return
    candidates = sorted(EXPORTS.glob("step84_checkpoint_gen*.json"))
    if len(candidates) <= CHECKPOINT_KEEP_LAST:
        return
    for stale in candidates[: len(candidates) - CHECKPOINT_KEEP_LAST]:
        try:
            stale.unlink()
        except OSError:
            pass


def load_checkpoint(path: Path) -> tuple[list[list[Individual]], list[np.random.Generator], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json_safe(json.load(handle))
    populations: list[list[Individual]] = []
    island_rngs: list[np.random.Generator] = []
    for island_payload in payload["islands"]:
        populations.append([Individual.from_dict(item) for item in island_payload["population"]])
        rng = np.random.default_rng()
        rng.bit_generator.state = island_payload["rng_state"]
        island_rngs.append(rng)
    state = {
        "generation": int(payload["generation"]),
        "total_evaluations": int(payload.get("total_evaluations", 0)),
        "elapsed_wall_seconds": float(payload.get("elapsed_wall_seconds", 0.0)),
        "best_generation": int(payload.get("best_generation", -1)),
        "improvement_history": [[int(item[0]), float(item[1])] for item in payload.get("improvement_history", [])],
        "warm_seed_count": int(payload.get("warm_seed_count", 0)),
        "steady_state_completed": int(payload.get("steady_state_completed", 0)),
        "global_best": None if payload.get("global_best") is None else Individual.from_dict(payload["global_best"]),
    }
    return populations, island_rngs, state


def maybe_resume() -> tuple[list[list[Individual]], list[np.random.Generator], dict] | None:
    if RESUME_CHECKPOINT is None:
        return None
    checkpoint_path = Path(RESUME_CHECKPOINT)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Requested STEP84_RESUME_CHECKPOINT does not exist: {checkpoint_path}")
    return load_checkpoint(checkpoint_path)


def log_generation(
    populations: list[list[Individual]],
    generation: int,
    polished_counts: dict[int, int],
    elapsed_wall_seconds: float,
    variable_count_cache: dict[str, int],
) -> None:
    rows: list[dict] = []
    for island_index, population in enumerate(populations):
        sorted_population = sorted(population, key=fitness_key)
        best = sorted_population[0]
        best_variable_count = None
        shadow_metrics = None
        if generation % SHADOW_METRICS_INTERVAL == 0:
            best_variable_count = compute_variable_count(best, variable_count_cache)
            shadow_metrics = population_shadow_metrics(population)
        mean_fitness = float(np.mean([individual.fitness for individual in population]))
        worst_fitness = float(max(individual.fitness for individual in population))
        rows.append(
            {
                "generation": generation,
                "island": island_index,
                "best_fitness": scalar_to_str(best.fitness),
                "mean_fitness": scalar_to_str(mean_fitness),
                "worst_fitness": scalar_to_str(worst_fitness),
                "best_support_signature": best.support_signature,
                "best_variable_count": "" if best_variable_count is None else int(best_variable_count),
                "shadow_pool_size": "" if shadow_metrics is None else int(shadow_metrics["shadow_pool_size"]),
                "niche_count": "" if shadow_metrics is None else int(shadow_metrics["niche_count"]),
                "signature_333_count": "" if shadow_metrics is None else int(shadow_metrics["signature_333_count"]),
                "favored_regime_count": "" if shadow_metrics is None else int(shadow_metrics["favored_regime_count"]),
                "favored_regime_best_fitness": "" if shadow_metrics is None or shadow_metrics["favored_regime_best_fitness"] is None else scalar_to_str(shadow_metrics["favored_regime_best_fitness"]),
                "signature_histogram_json": "" if shadow_metrics is None else json.dumps(shadow_metrics["signature_histogram"], sort_keys=True),
                "newton_polished_count": int(polished_counts.get(island_index, 0)),
                "wall_seconds_cumulative": scalar_to_str(elapsed_wall_seconds),
            }
        )
    append_log_rows(rows)


def generation_best(populations: list[list[Individual]]) -> Individual:
    # Islands are kept sorted by fitness_key; only need to compare leaders
    return min((island[0] for island in populations if island), key=fitness_key)


def summary_payload(
    completed_generation: int,
    total_evaluations: int,
    total_wall_seconds: float,
    populations: list[list[Individual]],
    global_best: Individual | None,
    best_generation: int,
    improvement_history: list[list[float]],
    stop_reason: str,
    warm_seed_count: int,
    variable_count_cache: dict[str, int],
) -> dict:
    best_variable_count = None if global_best is None else compute_variable_count(global_best, variable_count_cache)
    island_shadow_metrics = [population_shadow_metrics(population) for population in populations]
    global_signature_histogram: dict[str, int] = defaultdict(int)
    total_shadow_pool_size = 0
    total_niche_count = 0
    total_signature_333_count = 0
    total_favored_regime_count = 0
    global_favored_regime_best_fitness = math.inf
    for metrics in island_shadow_metrics:
        total_shadow_pool_size += int(metrics["shadow_pool_size"])
        total_niche_count += int(metrics["niche_count"])
        total_signature_333_count += int(metrics["signature_333_count"])
        total_favored_regime_count += int(metrics["favored_regime_count"])
        favored_best = metrics["favored_regime_best_fitness"]
        if favored_best is not None:
            global_favored_regime_best_fitness = min(global_favored_regime_best_fitness, float(favored_best))
        for signature, count in metrics["signature_histogram"].items():
            global_signature_histogram[signature] += int(count)
    return {
        "total_generations": completed_generation,
        "total_evaluations": total_evaluations,
        "total_wall_seconds": total_wall_seconds,
        "best_fitness_ever": None if global_best is None else finite_or_none(global_best.fitness),
        "best_fitness_generation": best_generation,
        "best_support_signature": None if global_best is None else global_best.support_signature,
        "best_variable_count": best_variable_count,
        "fitness_improvement_history": [[int(item[0]), float(item[1])] for item in improvement_history],
        "any_exact_hit": bool(global_best is not None and math.isfinite(global_best.fitness) and global_best.fitness < HIT_THRESHOLD),
        "islands": NUM_ISLANDS,
        "population_per_island": None,
        "island_populations": ISLAND_POPULATIONS,
        "offspring_per_island": OFFSPRING_PER_ISLANDS,
        "stop_reason": stop_reason,
        "warm_seed_count": warm_seed_count,
        "best_origin": None if global_best is None else global_best.origin,
        "best_fro_residual": None if global_best is None else float(global_best.fro_residual),
        "final_shadow_metrics": {
            "global_shadow_pool_size": total_shadow_pool_size,
            "global_niche_count": total_niche_count,
            "global_signature_333_count": total_signature_333_count,
            "global_favored_regime_count": total_favored_regime_count,
            "global_favored_regime_best_fitness": None if not math.isfinite(global_favored_regime_best_fitness) else float(global_favored_regime_best_fitness),
            "global_signature_histogram": dict(sorted(global_signature_histogram.items(), key=lambda item: item[0])),
            "islands": island_shadow_metrics,
        },
    }


def main() -> None:
    ensure_exports_dir()
    set_single_thread_blas_env()
    run_start = time.time()
    variable_count_cache: dict[str, int] = {}
    profiler = _ProfileAccumulator()

    resume_state = maybe_resume()
    if resume_state is None:
        populations, island_rngs, warm_seed_count = initialize_populations()
        current_generation = 0
        total_evaluations = 0
        steady_state_completed = 0
        elapsed_base = 0.0
        global_best = None
        best_generation = -1
        improvement_history: list[list[float]] = []
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        print(
            f"Step 84 starting fresh with islands={NUM_ISLANDS}, island_populations={ISLAND_POPULATIONS}, offspring_per_island={OFFSPRING_PER_ISLANDS}, workers={WORKERS}, executor={EXECUTOR_KIND}, warm_seeds={warm_seed_count}",
            flush=True,
        )
        print("Step 84 evaluating initial populations asynchronously", flush=True)
    else:
        populations, island_rngs, state = resume_state
        warm_seed_count = int(state["warm_seed_count"])
        current_generation = int(state["generation"])
        total_evaluations = int(state["total_evaluations"])
        steady_state_completed = int(state.get("steady_state_completed", current_generation * TOTAL_POP))
        elapsed_base = float(state["elapsed_wall_seconds"])
        global_best = state["global_best"]
        best_generation = int(state["best_generation"])
        improvement_history = list(state["improvement_history"])
        print(
            f"Step 84 resuming from generation {current_generation} with workers={WORKERS}, executor={EXECUTOR_KIND}, offspring_per_island={OFFSPRING_PER_ISLANDS}, and previous best={None if global_best is None else global_best.fitness}",
            flush=True,
        )

    stop_reason = "generation_limit"

    with executor_factory(WORKERS) as executor:
        if current_generation == 0 and (global_best is None or not math.isfinite(global_best.fitness)):
            populations, polished_counts, evaluated_count, exact_hit, init_timings = evaluate_initial_populations_async(executor, populations)
            total_evaluations += evaluated_count
            for _it in init_timings:
                profiler.record_eval(_it)
            profiler.record_generation(0)
            write_json(PROFILE_PATH, json_safe(profiler.to_dict(0)))
            current_best = generation_best(populations)
            if is_better(current_best, global_best):
                global_best = current_best.copy()
                best_generation = 0
                improvement_history.append([0, float(global_best.fitness)])
                write_best_individual(global_best, 0, variable_count_cache, potential_exact_hit=global_best.fitness < HIT_THRESHOLD)
            elapsed_now = elapsed_base + (time.time() - run_start)
            log_generation(populations, 0, polished_counts, elapsed_now, variable_count_cache)
            print(f"Step 84 generation 0 best residual: {current_best.fitness:.6e}", flush=True)
            if exact_hit:
                stop_reason = "exact_hit"

        shadow_pools = [build_shadow_pool(population) for population in populations]
        pending = {}
        pending_counts = [0 for _ in range(NUM_ISLANDS)]
        polished_counts = {index: 0 for index in range(NUM_ISLANDS)}
        last_logged_generation = current_generation
        last_checkpoint_generation = current_generation

        # Batched integration: accumulate offspring per island, flush periodically
        offspring_buffers: list[list[Individual]] = [[] for _ in range(NUM_ISLANDS)]
        shadow_rebuild_counters: list[int] = [0] * NUM_ISLANDS

        def flush_island(island_index: int) -> None:
            """Flush buffered offspring for one island via a single select_survivors call."""
            buf = offspring_buffers[island_index]
            if not buf:
                return
            integrate_offspring_batch(populations, island_index, buf, variable_count_cache)
            offspring_buffers[island_index] = []

        def flush_all_islands() -> None:
            for idx in range(NUM_ISLANDS):
                flush_island(idx)
                shadow_pools[idx] = build_shadow_pool(populations[idx])
                shadow_rebuild_counters[idx] = 0

        def submit_one(island_index: int) -> None:
            global_best_fitness = math.inf if global_best is None else float(global_best.fitness)
            child = build_single_offspring(
                populations[island_index],
                island_rngs[island_index],
                global_best_fitness,
                shadow_pools[island_index],
            )
            payload = {
                "eval_seed": SEED + 100000 + total_evaluations + steady_state_completed + pending_counts[island_index] + island_index * 1000,
                "individual": child.to_dict(),
            }
            future = executor.submit(evaluate_individual_worker, payload)
            pending[future] = island_index
            pending_counts[island_index] += 1

        for island_index, target in enumerate(OFFSPRING_PER_ISLANDS):
            initial_pending = min(max(1, target), WORKERS)
            for _ in range(initial_pending):
                submit_one(island_index)

        while stop_reason == "generation_limit" and current_generation < NUM_GENERATIONS:
            elapsed_now = elapsed_base + (time.time() - run_start)
            if elapsed_now >= TIMEOUT_SECONDS:
                stop_reason = "time_limit"
                break
            if not pending:
                break

            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            exact_hit = False
            for future in done:
                island_index = pending.pop(future)
                pending_counts[island_index] = max(0, pending_counts[island_index] - 1)
                result = future.result()
                individual = Individual.from_dict(result["individual"])
                individual.fitness = float(result.get("fitness", math.inf))
                individual.fro_residual = float(result.get("fro_residual", math.inf))
                individual.newton_polished = bool(result.get("newton_polished", False))
                individual.support_signature = str(result.get("support_signature", support_signature_string(individual)))
                individual.support_hash = str(result.get("support_hash", support_pattern_hash(individual)))

                profiler.record_eval(result.get("timing"))

                # Buffer offspring; flush when batch is full
                offspring_buffers[island_index].append(individual)
                if len(offspring_buffers[island_index]) >= _INTEGRATION_BATCH_SIZE:
                    _mf0 = time.perf_counter()
                    flush_island(island_index)
                    _mf1 = time.perf_counter()
                    profiler.record_mainloop("flush_ms", (_mf1 - _mf0) * 1000)
                    shadow_rebuild_counters[island_index] += 1
                    if shadow_rebuild_counters[island_index] >= _SHADOW_REBUILD_BATCH:
                        shadow_pools[island_index] = build_shadow_pool(populations[island_index])
                        profiler.record_mainloop("shadow_rebuild_ms", (time.perf_counter() - _mf1) * 1000)
                        shadow_rebuild_counters[island_index] = 0

                total_evaluations += 1
                steady_state_completed += 1
                if individual.newton_polished:
                    polished_counts[island_index] = polished_counts.get(island_index, 0) + 1
                if math.isfinite(individual.fitness) and individual.fitness < HIT_THRESHOLD:
                    exact_hit = True

                if pending_counts[island_index] < max(1, min(OFFSPRING_PER_ISLANDS[island_index], max(1, WORKERS // max(1, NUM_ISLANDS)) + 1)):
                    submit_one(island_index)

            next_generation = steady_state_generation(steady_state_completed)
            # Flush all buffers on generation transition so generation_best sees all evaluated offspring
            if next_generation > current_generation:
                _mfa0 = time.perf_counter()
                flush_all_islands()
                profiler.record_mainloop("flush_all_ms", (time.perf_counter() - _mfa0) * 1000)
            current_best = generation_best(populations)
            if is_better(current_best, global_best):
                should_print_improvement = is_meaningful_numeric_improvement(current_best, global_best)
                global_best = current_best.copy()
                best_generation = next_generation
                improvement_history.append([next_generation, float(global_best.fitness)])
                write_best_individual(global_best, next_generation, variable_count_cache, potential_exact_hit=global_best.fitness < HIT_THRESHOLD)
                if should_print_improvement:
                    print(
                        f"Step 84 new best at generation {next_generation}: residual={global_best.fitness:.6e}, signature={global_best.support_signature}",
                        flush=True,
                    )

            if next_generation > current_generation:
                profiler.record_generation(current_generation)
                current_generation = next_generation
                elapsed_now = elapsed_base + (time.time() - run_start)
                if current_generation > last_logged_generation:
                    _ml0 = time.perf_counter()
                    log_generation(populations, current_generation, polished_counts, elapsed_now, variable_count_cache)
                    profiler.record_mainloop("log_ms", (time.perf_counter() - _ml0) * 1000)
                    last_logged_generation = current_generation
                    polished_counts = {index: 0 for index in range(NUM_ISLANDS)}
                if current_generation % 10 == 0 or current_generation == NUM_GENERATIONS:
                    print(
                        f"Step 84 generation {current_generation}: best={current_best.fitness:.6e}, wall_seconds={elapsed_now:.1f}",
                        flush=True,
                    )
                if MIGRATION_INTERVAL > 0 and current_generation % MIGRATION_INTERVAL == 0:
                    _mm0 = time.perf_counter()
                    migrate(populations, island_rngs)
                    shadow_pools = [build_shadow_pool(population) for population in populations]
                    shadow_rebuild_counters = [0] * NUM_ISLANDS
                    profiler.record_mainloop("migrate_ms", (time.perf_counter() - _mm0) * 1000)
                if current_generation % CHECKPOINT_INTERVAL == 0 and current_generation != last_checkpoint_generation:
                    _mc0 = time.perf_counter()
                    write_checkpoint(
                        populations,
                        current_generation,
                        total_evaluations,
                        elapsed_now,
                        global_best,
                        best_generation,
                        improvement_history,
                        island_rngs,
                        warm_seed_count,
                        steady_state_completed,
                    )
                    profiler.record_mainloop("checkpoint_ms", (time.perf_counter() - _mc0) * 1000)
                    last_checkpoint_generation = current_generation
                if current_generation % _PROFILE_WRITE_INTERVAL == 0:
                    write_json(PROFILE_PATH, json_safe(profiler.to_dict(current_generation)))

            if exact_hit or (global_best is not None and global_best.fitness < HIT_THRESHOLD):
                stop_reason = "exact_hit"
                break

    total_wall_seconds = elapsed_base + (time.time() - run_start)
    profiler.record_generation(current_generation)
    write_json(PROFILE_PATH, json_safe(profiler.to_dict(current_generation)))
    if global_best is not None:
        write_best_individual(global_best, best_generation, variable_count_cache, potential_exact_hit=global_best.fitness < HIT_THRESHOLD)
    write_final_population(populations, current_generation, total_evaluations, total_wall_seconds)
    if current_generation % CHECKPOINT_INTERVAL != 0:
        write_checkpoint(
            populations,
            current_generation,
            total_evaluations,
            total_wall_seconds,
            global_best,
            best_generation,
            improvement_history,
            island_rngs,
            warm_seed_count,
            steady_state_completed,
        )

    summary = summary_payload(
        current_generation,
        total_evaluations,
        total_wall_seconds,
        populations,
        global_best,
        best_generation,
        improvement_history,
        stop_reason,
        warm_seed_count,
        variable_count_cache,
    )
    write_json(SUMMARY_PATH, json_safe(summary))

    print("=== Step 84: Metaheuristic rank-19 search ===", flush=True)
    print(f"Completed generations: {current_generation}", flush=True)
    print(f"Total evaluations: {total_evaluations}", flush=True)
    print(f"Stop reason: {stop_reason}", flush=True)
    print(
        f"Best residual: {summary['best_fitness_ever'] if summary['best_fitness_ever'] is not None else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()