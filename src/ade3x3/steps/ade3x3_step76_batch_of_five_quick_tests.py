"""
ade3x3_step76_batch_of_five_quick_tests.py

Step 76: Batch of Five Quick Tests.

This step groups five independent follow-up tasks:

Task A. Pareto frontier and addition-count accounting for known 3x3 schemes.
Task B. Cayley-Hamilton audit in the bilinear model.
Task C. Squaring-as-primitive counts and a bounded GF(3) rank search.
Task D. Randomized search over small division-augmented 2x2 circuits.
Task E. GF(9) construction plus a conservative exact GF(9) rank proxy search.

The heavy searches are intentionally bounded. Exact outputs are labeled
EXACT_DERIVED when proved from algebra or exact finite-field elimination, and
MEASURED_FROM_CODE when they come from bounded randomized search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
    matrix_multiplication_tensor,
)
from src.ade3x3.steps.ade3x3_step65_polyomino_subtensor_ranks_tiling_analysis import (
    reduced_output_tensor,
)


EXPORTS = Path("outputs/exports")
EXACT_TOL = 1e-12
SQUARE_SUPPORT_TOL = 1e-8


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


SQUARE_RESTARTS = env_int("STEP76_SQUARE_RESTARTS", 300)
SQUARE_MAXITER = env_int("STEP76_SQUARE_MAXITER", 500)
GF3_RANDOM_RESTARTS = env_int("STEP76_GF3_RANDOM_RESTARTS", 4000)
GF9_SWEEP_RESTARTS = env_int("STEP76_GF9_SWEEP_RESTARTS", 500)
GF9_FOCUS_RESTARTS = env_int("STEP76_GF9_FOCUS_RESTARTS", 5000)
D_SWEEP_RESTARTS = env_int("STEP76_D_SWEEP_RESTARTS", 500)
D_FOCUS_RESTARTS = env_int("STEP76_D_FOCUS_RESTARTS", 5000)
D_TRAIN_POINTS = env_int("STEP76_D_TRAIN_POINTS", 160)
D_VALID_POINTS = env_int("STEP76_D_VALID_POINTS", 160)
SEARCH_WORKERS = env_int("STEP76_SEARCH_WORKERS", max(1, os.cpu_count() or 1))
SEARCH_CHUNK_SIZE = env_int("STEP76_SEARCH_CHUNK_SIZE", 25)
GF9_SWEEP_RANKS = list(range(9, 23))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"  Wrote text -> {path}")


def union_fieldnames(rows: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix_to_json(matrix: np.ndarray, digits: int = 6) -> str:
    rounded = np.round(matrix.astype(float), digits)
    return json.dumps(rounded.tolist(), separators=(",", ":"))


def tensor_flat_c(target: np.ndarray) -> np.ndarray:
    return target.reshape(81, 9)


def cp_pack(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.concatenate([alpha.reshape(-1), beta.reshape(-1), gamma.reshape(-1)])


def cp_unpack(vector: np.ndarray, rank: int, output_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split1 = rank * 9
    split2 = 2 * rank * 9
    alpha = vector[:split1].reshape(rank, 9)
    beta = vector[split1:split2].reshape(rank, 9)
    gamma = vector[split2:].reshape(rank, output_dim)
    return alpha, beta, gamma


def cp_objective(vector: np.ndarray, target: np.ndarray, rank: int) -> tuple[float, np.ndarray]:
    output_dim = target.shape[2]
    alpha, beta, gamma = cp_unpack(vector, rank, output_dim)
    approx = np.einsum("ra,rb,rc->abc", alpha, beta, gamma, optimize=True)
    residual = approx - target
    loss = 0.5 * float(np.sum(residual * residual))
    grad_alpha = np.einsum("abc,rb,rc->ra", residual, beta, gamma, optimize=True)
    grad_beta = np.einsum("abc,ra,rc->rb", residual, alpha, gamma, optimize=True)
    grad_gamma = np.einsum("abc,ra,rb->rc", residual, alpha, beta, optimize=True)
    return loss, cp_pack(grad_alpha, grad_beta, grad_gamma)


def random_cp_initialization(rank: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = 0.25 * rng.standard_normal((rank, 9))
    beta = 0.25 * rng.standard_normal((rank, 9))
    gamma = 0.25 * rng.standard_normal((rank, output_dim))
    return cp_pack(alpha, beta, gamma)


def refine_gamma(alpha: np.ndarray, beta: np.ndarray, target: np.ndarray) -> np.ndarray:
    basis = np.einsum("ra,rb->rab", alpha, beta, optimize=True).reshape(alpha.shape[0], -1).T
    gamma = np.zeros((alpha.shape[0], target.shape[2]), dtype=np.float64)
    flat_target = target.reshape(-1, target.shape[2])
    for output_idx in range(target.shape[2]):
        solution, *_ = np.linalg.lstsq(basis, flat_target[:, output_idx], rcond=None)
        gamma[:, output_idx] = solution
    return gamma


def extract_square_tetromino_witness() -> tuple[list[dict], dict]:
    print("Task A helper: extracting a concrete rank-11 square-tetromino witness...")
    outputs = ((0, 0), (0, 1), (1, 0), (1, 1))
    target = reduced_output_tensor(outputs)
    best_loss = float("inf")
    best_abs = float("inf")
    best_payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for restart in range(SQUARE_RESTARTS):
        seed = 7611000 + 997 * restart
        x0 = random_cp_initialization(11, 4, seed)
        result = minimize(
            lambda x: cp_objective(x, target, 11),
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": SQUARE_MAXITER, "ftol": 1e-18, "gtol": 1e-12, "maxls": 50},
        )
        alpha, beta, gamma = cp_unpack(result.x, 11, 4)
        gamma = refine_gamma(alpha, beta, target)
        approx = np.einsum("ra,rb,rc->abc", alpha, beta, gamma, optimize=True)
        residual = approx - target
        loss = float(np.sum(residual * residual))
        max_abs = float(np.max(np.abs(residual)))
        if loss < best_loss:
            best_loss = loss
            best_abs = max_abs
            best_payload = (alpha.copy(), beta.copy(), gamma.copy())
        if restart == 0 or (restart + 1) % 50 == 0 or loss < EXACT_TOL:
            print(
                f"  square_tetromino restart {restart + 1}/{SQUARE_RESTARTS}: "
                f"best_loss={best_loss:.3e}, best_max_abs={best_abs:.3e}"
            )
        if loss < EXACT_TOL:
            break

    assert best_payload is not None
    alpha, beta, gamma = best_payload
    gamma_counts = Counter()
    rows: list[dict] = []
    scatter_adds = 0
    form_adds = 0
    for term_idx in range(11):
        alpha_nz = int(np.count_nonzero(np.abs(alpha[term_idx]) > SQUARE_SUPPORT_TOL))
        beta_nz = int(np.count_nonzero(np.abs(beta[term_idx]) > SQUARE_SUPPORT_TOL))
        gamma_nz = int(np.count_nonzero(np.abs(gamma[term_idx]) > SQUARE_SUPPORT_TOL))
        form_adds += max(alpha_nz - 1, 0) + max(beta_nz - 1, 0)
        scatter_adds += gamma_nz
        for output_idx in range(4):
            if abs(float(gamma[term_idx, output_idx])) > SQUARE_SUPPORT_TOL:
                gamma_counts[output_idx] += 1
        rows.append(
            {
                "term_id": f"sq{term_idx + 1:02d}",
                "alpha": matrix_to_json(alpha[term_idx].reshape(3, 3)),
                "beta": matrix_to_json(beta[term_idx].reshape(3, 3)),
                "gamma": matrix_to_json(gamma[term_idx].reshape(2, 2)),
                "alpha_nonzero": alpha_nz,
                "beta_nonzero": beta_nz,
                "gamma_nonzero": gamma_nz,
                "provenance": "MEASURED_FROM_CODE",
            }
        )

    scheduled_output_adds = sum(max(gamma_counts[idx] - 1, 0) for idx in range(4))
    summary = {
        "square_tetromino_rank": 11,
        "square_tetromino_best_loss": best_loss,
        "square_tetromino_best_max_abs_residual": best_abs,
        "square_tetromino_form_additions": form_adds,
        "square_tetromino_gamma_scatter_additions": scatter_adds,
        "square_tetromino_gamma_scheduled_additions": scheduled_output_adds,
        "square_tetromino_total_scatter_additions": form_adds + scatter_adds,
        "square_tetromino_total_scheduled_additions": form_adds + scheduled_output_adds,
    }
    return rows, summary


def alpha_tensor_addition_counts(terms: list[Term]) -> dict[str, int]:
    alpha_adds = 0
    beta_adds = 0
    gamma_scatter_adds = 0
    gamma_output_counts = Counter()
    for term in terms:
        alpha_nz = int(np.count_nonzero(term.alpha))
        beta_nz = int(np.count_nonzero(term.beta))
        alpha_adds += max(alpha_nz - 1, 0)
        beta_adds += max(beta_nz - 1, 0)
        gamma_nz = int(np.count_nonzero(term.gamma))
        gamma_scatter_adds += gamma_nz
        for row_idx in range(3):
            for col_idx in range(3):
                if int(term.gamma[row_idx, col_idx]) != 0:
                    gamma_output_counts[(row_idx, col_idx)] += 1
    gamma_scheduled_adds = sum(max(count - 1, 0) for count in gamma_output_counts.values())
    return {
        "alpha_adds": alpha_adds,
        "beta_adds": beta_adds,
        "gamma_scatter_adds": gamma_scatter_adds,
        "gamma_scheduled_adds": gamma_scheduled_adds,
        "total_requested_adds": alpha_adds + beta_adds + gamma_scatter_adds,
        "total_scheduled_adds": alpha_adds + beta_adds + gamma_scheduled_adds,
    }


def best_26_tiling_rows(square_summary: dict[str, float]) -> list[dict]:
    tilings = [
        ("domino_col + 3 monomino + square_tetromino", 6 + 3 + 3 + 3 + 11, 4 + 2 + 2 + 2, "step67 tiling_id 11"),
        ("2 monomino + tromino_col + square_tetromino", 3 + 3 + 9 + 11, 2 + 2 + 6, "step67 tiling_id 18"),
        ("domino_col + domino_row + monomino + square_tetromino", 6 + 6 + 3 + 11, 4 + 4 + 2, "step67 tiling_id 23"),
        ("L_tetromino + monomino + square_tetromino", 12 + 3 + 11, 8 + 2, "step67 tiling_id 31/32"),
        ("domino_row + tromino_col + square_tetromino", 6 + 9 + 11, 4 + 6, "step67 tiling_id 37"),
    ]
    rows: list[dict] = []
    for name, mults, base_adds, source in tilings:
        assert mults == 26
        rows.append(
            {
                "configuration": name,
                "M": mults,
                "A_scatter_proxy": int(base_adds + square_summary["square_tetromino_total_scatter_additions"]),
                "A_scheduled_proxy": int(base_adds + square_summary["square_tetromino_total_scheduled_additions"]),
                "source": source,
                "status": "tiling_configuration_not_certified_global_algorithm",
                "provenance": "EXACT_DERIVED + MEASURED_FROM_CODE",
            }
        )
    rows.sort(key=lambda row: (row["A_scheduled_proxy"], row["configuration"]))
    return rows


def cost_table_rows(alpha_counts: dict[str, int], tiling_rows: list[dict]) -> list[dict]:
    p_values = [1, 2, 5, 10, 20, 50, 100]
    best_tiling = tiling_rows[0]
    rows: list[dict] = []
    for p_value in p_values:
        rows.extend(
            [
                {
                    "scheme": "standard_27_term",
                    "p": p_value,
                    "M": 27,
                    "A": 18,
                    "cost": p_value * 27 + 18,
                    "count_model": "scheduled_output_accumulation",
                    "provenance": "EXACT_DERIVED",
                },
                {
                    "scheme": "alphatensor_23_term",
                    "p": p_value,
                    "M": 23,
                    "A": alpha_counts["total_requested_adds"],
                    "cost": p_value * 23 + alpha_counts["total_requested_adds"],
                    "count_model": "requested_term_scatter",
                    "provenance": "EXACT_DERIVED",
                },
                {
                    "scheme": "alphatensor_23_term_scheduled_reference",
                    "p": p_value,
                    "M": 23,
                    "A": alpha_counts["total_scheduled_adds"],
                    "cost": p_value * 23 + alpha_counts["total_scheduled_adds"],
                    "count_model": "scheduled_output_accumulation",
                    "provenance": "EXACT_DERIVED",
                },
                {
                    "scheme": "best_26_tiling_proxy",
                    "p": p_value,
                    "M": 26,
                    "A": best_tiling["A_scheduled_proxy"],
                    "cost": p_value * 26 + best_tiling["A_scheduled_proxy"],
                    "count_model": "scheduled_proxy_from_piece_witnesses",
                    "provenance": "EXACT_DERIVED + MEASURED_FROM_CODE",
                },
            ]
        )
    return rows


def crossover_point(m1: int, a1: int, m2: int, a2: int) -> str:
    if m1 == m2:
        return "never" if a1 >= a2 else "always"
    threshold = (a2 - a1) / (m1 - m2)
    if m1 > m2:
        integer_threshold = math.floor(threshold) + 1
        return str(integer_threshold)
    integer_threshold = math.floor((-threshold)) + 1
    return f"p <= {integer_threshold}"


def task_a() -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, str], list[dict]]:
    print("=== Task A: Pareto Frontier ===")
    terms, _, _ = load_public_rank23_terms()
    alpha_counts = alpha_tensor_addition_counts(terms)
    square_rows, square_summary = extract_square_tetromino_witness()
    tiling_rows = best_26_tiling_rows(square_summary)
    best_tiling = tiling_rows[0]
    costs = cost_table_rows(alpha_counts, tiling_rows)

    requested_crossover = crossover_point(27, 18, 23, alpha_counts["total_requested_adds"])
    scheduled_crossover = crossover_point(27, 18, 23, alpha_counts["total_scheduled_adds"])

    a4_rows = []
    for p_value in [1, 2, 5, 10, 20, 50, 100]:
        alpha_cost = 23 * p_value + alpha_counts["total_requested_adds"]
        a4_rows.append(
            {
                "p": p_value,
                "max_A_for_24M_to_beat_requested_alphatensor": alpha_cost - 24 * p_value - 1,
                "max_A_for_25M_to_beat_requested_alphatensor": alpha_cost - 25 * p_value - 1,
                "max_A_for_26M_to_beat_requested_alphatensor": alpha_cost - 26 * p_value - 1,
                "known_certified_24M_or_25M_algorithm_in_repo": "False",
                "known_26M_configuration_in_repo": "True",
                "provenance": "EXACT_DERIVED",
            }
        )

    summary_rows = [
        {
            "summary_name": "taskA_standard_M",
            "summary_value": "27",
            "provenance": "EXACT_DERIVED",
            "note": "Standard 3x3 algorithm scalar multiplications.",
        },
        {
            "summary_name": "taskA_standard_A",
            "summary_value": "18",
            "provenance": "EXACT_DERIVED",
            "note": "Standard 3x3 output-accumulation additions.",
        },
        {
            "summary_name": "taskA_alphatensor_M",
            "summary_value": "23",
            "provenance": "EXACT_DERIVED",
            "note": "Public AlphaTensor 3x3 term count.",
        },
        {
            "summary_name": "taskA_alphatensor_A_requested",
            "summary_value": str(alpha_counts["total_requested_adds"]),
            "provenance": "EXACT_DERIVED",
            "note": "User-requested AlphaTensor count using per-term gamma scatter.",
        },
        {
            "summary_name": "taskA_alphatensor_A_scheduled",
            "summary_value": str(alpha_counts["total_scheduled_adds"]),
            "provenance": "EXACT_DERIVED",
            "note": "Output-accumulation-normalized AlphaTensor addition count.",
        },
        {
            "summary_name": "taskA_best_tiling_M",
            "summary_value": "26",
            "provenance": "EXACT_DERIVED",
            "note": "Best Step 67 flat tiling multiplication upper bound.",
        },
        {
            "summary_name": "taskA_best_tiling_A_scheduled_proxy",
            "summary_value": str(best_tiling["A_scheduled_proxy"]),
            "provenance": "EXACT_DERIVED + MEASURED_FROM_CODE",
            "note": "Scheduled proxy additions from concrete piece witnesses; tiling not certified as a global algorithm.",
        },
        {
            "summary_name": "taskA_alphatensor_beats_standard_requested_after_p",
            "summary_value": requested_crossover,
            "provenance": "EXACT_DERIVED",
            "note": "Crossover using the user-requested AlphaTensor scatter count.",
        },
        {
            "summary_name": "taskA_alphatensor_beats_standard_scheduled_after_p",
            "summary_value": scheduled_crossover,
            "provenance": "EXACT_DERIVED",
            "note": "Crossover using output-accumulation-normalized AlphaTensor counting.",
        },
        {
            "summary_name": "taskA_any_certified_24_or_25_mult_algorithm_known_here",
            "summary_value": "False",
            "provenance": "EXACT_DERIVED",
            "note": "No certified 24- or 25-multiplication 3x3 algorithm is present in the repo outputs.",
        },
    ]

    notes = {
        "requested_crossover": requested_crossover,
        "scheduled_crossover": scheduled_crossover,
        "best_tiling_configuration": best_tiling["configuration"],
        "best_tiling_A_scheduled_proxy": str(best_tiling["A_scheduled_proxy"]),
        "alpha_requested_A": str(alpha_counts["total_requested_adds"]),
        "alpha_scheduled_A": str(alpha_counts["total_scheduled_adds"]),
    }

    print(
        f"Task A complete: AlphaTensor A(requested)={alpha_counts['total_requested_adds']}, "
        f"A(scheduled)={alpha_counts['total_scheduled_adds']}, best 26-mult tiling proxy A={best_tiling['A_scheduled_proxy']}."
    )
    return summary_rows, costs, tiling_rows, a4_rows, notes, square_rows


def task_b() -> tuple[list[dict], list[dict], dict[str, str]]:
    print("=== Task B: Cayley-Hamilton Constraint ===")
    formula_rows = [
        {
            "formula_name": "tr(AB)",
            "formula": "sum_{i=0}^2 sum_{s=0}^2 A[i,s] B[s,i]",
            "total_degree_in_(A,B)": "2",
            "provenance": "EXACT_DERIVED",
        },
        {
            "formula_name": "tr((AB)^2)",
            "formula": "sum_{i=0}^2 sum_{j=0}^2 sum_{s=0}^2 sum_{t=0}^2 A[i,s] B[s,j] A[j,t] B[t,i]",
            "total_degree_in_(A,B)": "4",
            "provenance": "EXACT_DERIVED",
        },
        {
            "formula_name": "det(AB)",
            "formula": "det(A) det(B)",
            "total_degree_in_(A,B)": "6",
            "provenance": "EXACT_DERIVED",
        },
        {
            "formula_name": "cayley_hamilton",
            "formula": "C^3 - tr(C) C^2 + ((tr(C)^2 - tr(C^2))/2) C - det(C) I = 0 with C=AB",
            "total_degree_in_(A,B)": "6",
            "provenance": "EXACT_DERIVED",
        },
    ]

    claim_rows = [
        {
            "claim": "Cayley-Hamilton reduces the number of independently specifiable 3x3 output entries from 9 to 8.",
            "verdict": "False",
            "reason": "Matrix multiplication is surjective onto M_3: for every C choose A=I and B=C. Since every 3x3 matrix occurs as AB, no nontrivial algebraic relation can eliminate one output coordinate globally.",
            "provenance": "EXACT_DERIVED",
        },
        {
            "claim": "Cayley-Hamilton can be used directly in the pure bilinear model to replace one bilinear output coordinate by the other eight.",
            "verdict": "False",
            "reason": "The identity is nonlinear in C and degree 6 in (A,B). The pure bilinear model allows only linear recombination of bilinear quantities, not quadratic or cubic elimination among outputs.",
            "provenance": "EXACT_DERIVED",
        },
        {
            "claim": "Allowing division changes that conclusion for direct computation of the ninth entry from the other eight.",
            "verdict": "False_as_a_general_reduction",
            "reason": "Division can rearrange the identity, but solving for a missing entry still requires nonlinear operations on the already-known matrix C. It does not decrease the number of bilinear quantities needed to obtain C in the first place.",
            "provenance": "EXACT_DERIVED",
        },
    ]

    summary_rows = [
        {
            "summary_name": "taskB_cayley_hamilton_reduces_bilinear_independent_outputs",
            "summary_value": "False",
            "provenance": "EXACT_DERIVED",
            "note": "CH is a universal polynomial identity on all 3x3 matrices, not a dimension-reducing relation for the image of (A,B) -> AB.",
        },
        {
            "summary_name": "taskB_surjectivity_counterexample",
            "summary_value": "for_every_C_choose_A=I_and_B=C",
            "provenance": "EXACT_DERIVED",
            "note": "Shows every 3x3 matrix occurs as AB.",
        },
        {
            "summary_name": "taskB_bilinear_model_verdict",
            "summary_value": "Cayley_Hamilton_does_not_lower_the_required_number_of_bilinear_quantities",
            "provenance": "EXACT_DERIVED",
            "note": "The identity is too nonlinear to be exploited inside a pure bilinear decomposition.",
        },
    ]

    notes = {
        "verdict": "no",
        "proof_outline": "surjectivity_of_(A,B)->AB_plus_nonlinearity_of_CH",
    }
    print("Task B complete: Cayley-Hamilton does not reduce bilinear output complexity; the '8 independent + 1 determined' premise is false.")
    return summary_rows, formula_rows + claim_rows, notes


@dataclass(frozen=True)
class Field:
    name: str
    size: int
    add: np.ndarray
    mul: np.ndarray
    neg: np.ndarray
    inv: np.ndarray
    labels: list[str]
    frob: np.ndarray


def build_gf3() -> Field:
    add = np.zeros((3, 3), dtype=np.int64)
    mul = np.zeros((3, 3), dtype=np.int64)
    neg = np.zeros(3, dtype=np.int64)
    inv = np.zeros(3, dtype=np.int64)
    labels = ["0", "1", "2"]
    frob = np.zeros(3, dtype=np.int64)
    for a in range(3):
        neg[a] = (-a) % 3
        frob[a] = pow(a, 3, 3)
        for b in range(3):
            add[a, b] = (a + b) % 3
            mul[a, b] = (a * b) % 3
    inv[1] = 1
    inv[2] = 2
    return Field("GF(3)", 3, add, mul, neg, inv, labels, frob)


def gf9_pair_to_code(a: int, b: int) -> int:
    return a + 3 * b


def gf9_code_to_pair(code: int) -> tuple[int, int]:
    return code % 3, code // 3


def build_gf9() -> Field:
    add = np.zeros((9, 9), dtype=np.int64)
    mul = np.zeros((9, 9), dtype=np.int64)
    neg = np.zeros(9, dtype=np.int64)
    inv = np.zeros(9, dtype=np.int64)
    frob = np.zeros(9, dtype=np.int64)
    labels: list[str] = []
    for code in range(9):
        a, b = gf9_code_to_pair(code)
        if b == 0:
            labels.append(str(a))
        elif a == 0:
            labels.append({1: "w", 2: "2w"}[b])
        else:
            left = str(a)
            right = {1: "w", 2: "2w"}[b]
            labels.append(f"{left}+{right}")
    for x in range(9):
        ax, bx = gf9_code_to_pair(x)
        neg[x] = gf9_pair_to_code((-ax) % 3, (-bx) % 3)
        for y in range(9):
            ay, by = gf9_code_to_pair(y)
            add[x, y] = gf9_pair_to_code((ax + ay) % 3, (bx + by) % 3)
            const = (ax * ay + bx * by) % 3
            omega = (ax * by + bx * ay + bx * by) % 3
            mul[x, y] = gf9_pair_to_code(const, omega)
    for x in range(1, 9):
        for y in range(1, 9):
            if mul[x, y] == 1:
                inv[x] = y
                break
    for x in range(9):
        value = 1
        for _ in range(3):
            value = mul[value, x]
        frob[x] = value
    return Field("GF(9)", 9, add, mul, neg, inv, labels, frob)


GF3 = build_gf3()
GF9 = build_gf9()


def field_mat_solve(field: Field, matrix: np.ndarray, rhs: np.ndarray) -> tuple[bool, np.ndarray]:
    a = matrix.astype(np.int64).copy()
    b = rhs.astype(np.int64).copy()
    rows, cols = a.shape
    _, rhs_cols = b.shape
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    row = 0

    for col in range(cols):
        pivot = None
        for cand in range(row, rows):
            if int(a[cand, col]) != 0:
                pivot = cand
                break
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        inv = int(field.inv[int(a[row, col])])
        a[row, :] = field.mul[a[row, :], inv]
        b[row, :] = field.mul[b[row, :], inv]
        for elim in range(rows):
            if elim == row or int(a[elim, col]) == 0:
                continue
            factor = int(a[elim, col])
            a[elim, :] = field.add[a[elim, :], field.mul[field.neg[factor], a[row, :]]]
            b[elim, :] = field.add[b[elim, :], field.mul[field.neg[factor], b[row, :]]]
        pivot_rows.append(row)
        pivot_cols.append(col)
        row += 1
        if row == rows:
            break

    for check_row in range(rows):
        if np.all(a[check_row, :] == 0) and np.any(b[check_row, :] != 0):
            return False, np.zeros((cols, rhs_cols), dtype=np.int64)

    solution = np.zeros((cols, rhs_cols), dtype=np.int64)
    for prow, pcol in zip(pivot_rows, pivot_cols, strict=True):
        solution[pcol, :] = b[prow, :]
    return True, solution


def profile_vector_field(field: Field, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.zeros(81, dtype=np.int64)
    idx = 0
    for a_coeff in alpha:
        for b_coeff in beta:
            out[idx] = field.mul[int(a_coeff), int(b_coeff)]
            idx += 1
    return out


def target_flat_field(field: Field) -> np.ndarray:
    tensor = matrix_multiplication_tensor(3)
    return tensor.reshape(81, 9).astype(np.int64) % field.size


def alpha_tensor_profiles_field(field: Field) -> list[np.ndarray]:
    terms, _, _ = load_public_rank23_terms()
    profiles = []
    for term in terms:
        alpha = term.alpha.reshape(-1) % field.size
        beta = term.beta.reshape(-1) % field.size
        profiles.append(profile_vector_field(field, alpha.astype(np.int64), beta.astype(np.int64)))
    return profiles


def solvable_output_columns(field: Field, matrix: np.ndarray, target: np.ndarray) -> int:
    count = 0
    for col_idx in range(target.shape[1]):
        okay, _ = field_mat_solve(field, matrix, target[:, col_idx:col_idx + 1])
        count += int(okay)
    return count


def random_nonzero_field_vector(field: Field, length: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        vector = rng.integers(0, field.size, size=length, endpoint=False, dtype=np.int64)
        if np.any(vector != 0):
            return vector


def bounded_field_rank_search(field: Field, rank: int, restarts: int, seed_base: int) -> tuple[list[dict], dict]:
    target = target_flat_field(field)
    alpha_profiles = alpha_tensor_profiles_field(field)
    subset_rows: list[dict] = []

    if rank == 22 and len(alpha_profiles) == 23:
        for removed_idx in range(23):
            chosen = [profile for idx, profile in enumerate(alpha_profiles) if idx != removed_idx]
            matrix = np.column_stack(chosen)
            covered = solvable_output_columns(field, matrix, target)
            subset_rows.append(
                {
                    "search_family": f"{field.name}_alphatensor_subset22",
                    "removed_term": removed_idx + 1,
                    "covered_output_columns": covered,
                    "exact": str(covered == 9),
                    "provenance": "EXACT_DERIVED",
                }
            )

    rows: list[dict] = []
    best_covered = 0
    found = False
    best_restart = -1
    rng = np.random.default_rng(seed_base)
    for restart in range(restarts):
        columns = []
        for _ in range(rank):
            alpha = random_nonzero_field_vector(field, 9, rng)
            beta = random_nonzero_field_vector(field, 9, rng)
            columns.append(profile_vector_field(field, alpha, beta))
        matrix = np.column_stack(columns)
        covered = solvable_output_columns(field, matrix, target)
        if covered > best_covered:
            best_covered = covered
            best_restart = restart
        exact = covered == 9
        if restart == 0 or (restart + 1) % max(restarts // 10, 1) == 0 or exact:
            print(
                f"  {field.name} rank-{rank} restart {restart + 1}/{restarts}: "
                f"best_covered={best_covered}/9"
            )
        rows.append(
            {
                "field": field.name,
                "rank_tested": rank,
                "restart": restart + 1,
                "covered_output_columns": covered,
                "exact": str(exact),
                "provenance": "MEASURED_FROM_CODE",
            }
        )
        if exact:
            found = True
            best_restart = restart
            break

    summary = {
        "field": field.name,
        "rank_tested": rank,
        "random_restarts": restarts,
        "best_covered_output_columns": best_covered,
        "found_exact": found,
        "best_restart_index": best_restart + 1 if best_restart >= 0 else 0,
    }
    return subset_rows + rows, summary


def task_c() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    print("=== Task C: Squaring as Primitive ===")
    terms, _, _ = load_public_rank23_terms()
    alpha_forms = [tuple(term.alpha.reshape(-1).tolist()) for term in terms]
    beta_forms = [tuple(term.beta.reshape(-1).tolist()) for term in terms]
    plus_forms = {(tuple(term.alpha.reshape(-1).tolist()), tuple(term.beta.reshape(-1).tolist())) for term in terms}
    minus_forms = {(tuple(term.alpha.reshape(-1).tolist()), tuple((-term.beta).reshape(-1).tolist())) for term in terms}
    distinct_alpha = len(set(alpha_forms))
    distinct_beta = len(set(beta_forms))
    distinct_pm = len(plus_forms | minus_forms)

    count_rows = [
        {
            "quantity": "distinct_alpha_forms",
            "value": distinct_alpha,
            "provenance": "EXACT_DERIVED",
        },
        {
            "quantity": "distinct_beta_forms",
            "value": distinct_beta,
            "provenance": "EXACT_DERIVED",
        },
        {
            "quantity": "distinct_alpha_plus_or_minus_beta_forms",
            "value": distinct_pm,
            "provenance": "EXACT_DERIVED",
        },
        {
            "quantity": "naive_two_squarings_total",
            "value": 46,
            "provenance": "EXACT_DERIVED",
        },
        {
            "quantity": "sharing_saves_squarings",
            "value": str(distinct_pm < 46),
            "provenance": "EXACT_DERIVED",
        },
    ]

    gf3_rows, gf3_summary = bounded_field_rank_search(GF3, rank=22, restarts=GF3_RANDOM_RESTARTS, seed_base=7633001)

    summary_rows = [
        {
            "summary_name": "taskC_distinct_alpha_forms",
            "summary_value": str(distinct_alpha),
            "provenance": "EXACT_DERIVED",
            "note": "Distinct AlphaTensor alpha linear forms.",
        },
        {
            "summary_name": "taskC_distinct_beta_forms",
            "summary_value": str(distinct_beta),
            "provenance": "EXACT_DERIVED",
            "note": "Distinct AlphaTensor beta linear forms.",
        },
        {
            "summary_name": "taskC_distinct_alpha_pm_beta_forms",
            "summary_value": str(distinct_pm),
            "provenance": "EXACT_DERIVED",
            "note": "Distinct (alpha·A ± beta·B) forms across the 23 terms.",
        },
        {
            "summary_name": "taskC_distinct_forms_save_over_46",
            "summary_value": str(distinct_pm < 46),
            "provenance": "EXACT_DERIVED",
            "note": "Whether shared plus/minus forms lower the naive 46-squaring count.",
        },
        {
            "summary_name": "taskC_gf3_rank22_random_exact_hit",
            "summary_value": str(gf3_summary["found_exact"]),
            "provenance": "MEASURED_FROM_CODE",
            "note": "Bounded exact GF(3) random span search at rank 22.",
        },
        {
            "summary_name": "taskC_gf3_rank_bounds_after_quick_test",
            "summary_value": "9 <= rank_GF3 <= 23",
            "provenance": "EXACT_DERIVED + MEASURED_FROM_CODE",
            "note": "Flattenings give 9 and mod-3 AlphaTensor coefficients give 23; the bounded rank-22 search found no exact witness unless flagged above.",
        },
        {
            "summary_name": "taskC_user_claim_x2_equals_x_over_GF3",
            "summary_value": "False",
            "provenance": "EXACT_DERIVED",
            "note": "Over GF(3), 2^2 = 1, so squaring is not the Frobenius map; x^3 = x is the Frobenius identity.",
        },
    ]

    notes = {
        "distinct_alpha": str(distinct_alpha),
        "distinct_beta": str(distinct_beta),
        "distinct_pm": str(distinct_pm),
        "gf3_rank22_exact": str(gf3_summary["found_exact"]),
    }
    print(
        f"Task C complete: distinct alpha={distinct_alpha}, distinct beta={distinct_beta}, distinct alpha±beta={distinct_pm}; "
        f"GF(3) rank-22 quick search exact hit={gf3_summary['found_exact']}."
    )
    return summary_rows, count_rows, gf3_rows, notes


def generate_sample_points(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return rng.choice(values, size=(count, 8), replace=True)


def target_2x2(points: np.ndarray) -> np.ndarray:
    a00, a01, a10, a11, b00, b01, b10, b11 = points.T
    return np.column_stack(
        [
            a00 * b00 + a01 * b10,
            a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10,
            a10 * b01 + a11 * b11,
        ]
    )


def random_linear_form(rng: np.random.Generator, max_nonzero: int = 4) -> np.ndarray:
    coeffs = np.zeros(8, dtype=np.float64)
    nnz = int(rng.integers(1, max_nonzero + 1))
    positions = rng.choice(8, size=nnz, replace=False)
    coeffs[positions] = rng.choice(np.array([-1.0, 1.0]), size=nnz, replace=True)
    return coeffs


def evaluate_linear_form(coeffs: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ coeffs


def sample_random_circuit(op_count: int, div_count: int, rng: np.random.Generator) -> list[dict]:
    op_types = ["div"] * div_count + ["mul"] * (op_count - div_count)
    rng.shuffle(op_types)
    ops: list[dict] = []
    for op_type in op_types:
        left = random_linear_form(rng)
        right = random_linear_form(rng)
        ops.append(
            {
                "type": op_type,
                "left": left.tolist(),
                "right": right.tolist(),
            }
        )
    return ops


def evaluate_circuit_basis(ops: list[dict], points: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    valid_mask = np.ones(points.shape[0], dtype=bool)
    cached_values: list[tuple[str, np.ndarray, np.ndarray]] = []
    for op in ops:
        left = np.asarray(op["left"], dtype=np.float64)
        right = np.asarray(op["right"], dtype=np.float64)
        left_values = evaluate_linear_form(left, points)
        right_values = evaluate_linear_form(right, points)
        if op["type"] == "div":
            valid_mask &= np.abs(right_values) > 1e-9
        cached_values.append((op["type"], left_values, right_values))

    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < max(24, points.shape[0] // 3):
        return False, valid_mask, np.zeros((0, 0), dtype=np.float64)

    features = np.zeros((valid_count, len(ops)), dtype=np.float64)
    for op_idx, (op_type, left_values, right_values) in enumerate(cached_values):
        if op_type == "mul":
            features[:, op_idx] = (left_values * right_values)[valid_mask]
        else:
            features[:, op_idx] = (left_values[valid_mask] / right_values[valid_mask])
    return True, valid_mask, features


def finite_or_none(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def split_chunks(items: list[int], chunk_size: int) -> list[list[int]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def field_from_name(field_name: str) -> Field:
    if field_name == GF3.name:
        return GF3
    if field_name == GF9.name:
        return GF9
    raise ValueError(f"Unknown field name: {field_name}")


def field_rank_chunk_worker(payload: dict) -> dict:
    field_name = str(payload["field_name"])
    rank = int(payload["rank"])
    seeds = [int(seed) for seed in payload["seeds"]]
    field = field_from_name(field_name)
    target = tensor_flat_c(matrix_multiplication_tensor(3))
    best_covered = -1
    positive_restart_count = 0
    exact_hit_count = 0
    best_restart_seed = 0
    best_restart_index = 0
    for restart_index, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        columns = []
        for _ in range(rank):
            alpha = random_nonzero_field_vector(field, 9, rng)
            beta = random_nonzero_field_vector(field, 9, rng)
            columns.append(profile_vector_field(field, alpha, beta))
        matrix = np.column_stack(columns)
        covered = solvable_output_columns(field, matrix, target)
        if covered > 0:
            positive_restart_count += 1
        if covered == 9:
            exact_hit_count += 1
        if covered > best_covered:
            best_covered = covered
            best_restart_seed = seed
            best_restart_index = restart_index
    return {
        "field": field.name,
        "rank_tested": rank,
        "restart_count": len(seeds),
        "best_covered_output_columns": best_covered,
        "positive_restart_count": positive_restart_count,
        "exact_hit_count": exact_hit_count,
        "best_restart_seed": best_restart_seed,
        "best_restart_index_in_chunk": best_restart_index,
        "provenance": "MEASURED_FROM_CODE",
    }


def division_case_chunk_worker(payload: dict) -> dict:
    op_count = int(payload["op_count"])
    div_count = int(payload["div_count"])
    seeds = [int(seed) for seed in payload["seeds"]]
    train_points = generate_sample_points(D_TRAIN_POINTS, 760100 + 10 * op_count + div_count)
    valid_points = generate_sample_points(D_VALID_POINTS, 760900 + 10 * op_count + div_count)
    train_target = target_2x2(train_points)
    valid_target = target_2x2(valid_points)
    valid_structures = 0
    best_train_mse = float("inf")
    best_valid_mse = float("inf")
    exact_hit_count = 0
    best_restart_seed = 0
    best_restart_index = 0
    for restart_index, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        ops = sample_random_circuit(op_count, div_count, rng)
        okay_train, train_mask, train_features = evaluate_circuit_basis(ops, train_points)
        if not okay_train:
            continue
        okay_valid, valid_mask, valid_features = evaluate_circuit_basis(ops, valid_points)
        if not okay_valid:
            continue
        valid_structures += 1
        weights, *_ = np.linalg.lstsq(train_features, train_target[train_mask], rcond=None)
        train_residual = train_features @ weights - train_target[train_mask]
        valid_residual = valid_features @ weights - valid_target[valid_mask]
        train_loss = float(np.mean(train_residual * train_residual))
        valid_loss = float(np.mean(valid_residual * valid_residual))
        exact = train_loss < 1e-10 and valid_loss < 1e-10
        if exact:
            exact_hit_count += 1
        if valid_loss < best_valid_mse or (valid_loss == best_valid_mse and train_loss < best_train_mse):
            best_train_mse = train_loss
            best_valid_mse = valid_loss
            best_restart_seed = seed
            best_restart_index = restart_index
    return {
        "search_case": f"{op_count}_ops_{div_count}_div",
        "op_count": op_count,
        "div_count": div_count,
        "restart_count": len(seeds),
        "valid_structures": valid_structures,
        "best_train_mse": finite_or_none(best_train_mse),
        "best_valid_mse": finite_or_none(best_valid_mse),
        "exact_hit_count": exact_hit_count,
        "best_restart_seed": best_restart_seed,
        "best_restart_index_in_chunk": best_restart_index,
        "provenance": "MEASURED_FROM_CODE",
    }


def staged_field_rank_sweep(
    field: Field,
    ranks: list[int],
    sweep_restarts: int,
    focus_restarts: int,
    seed_base: int,
    workers: int,
) -> tuple[list[dict], dict]:
    total_restarts = len(ranks) * sweep_restarts
    print(
        f"  Sweep: {field.name} ranks {ranks[0]}..{ranks[-1]}, {sweep_restarts} restarts each, "
        f"workers={workers}, chunk_size={SEARCH_CHUNK_SIZE}."
    )
    best_covered_by_rank = {rank: -1 for rank in ranks}
    positive_counts = Counter({rank: 0 for rank in ranks})
    exact_counts = Counter({rank: 0 for rank in ranks})
    best_restart_index: dict[int, int] = {}
    best_restart_seed: dict[int, int] = {}
    payloads: list[dict] = []
    for rank in ranks:
        seeds = [seed_base + 100000 * rank + restart_idx for restart_idx in range(sweep_restarts)]
        running_restart_index = 0
        for chunk in split_chunks(seeds, SEARCH_CHUNK_SIZE):
            payloads.append(
                {
                    "field_name": field.name,
                    "rank": rank,
                    "restart_offset": running_restart_index,
                    "seeds": chunk,
                }
            )
            running_restart_index += len(chunk)
    completed_chunks = 0
    total_chunks = len(payloads)
    progress_step = max(total_chunks // 10, 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(field_rank_chunk_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            rank = int(result["rank_tested"])
            positive_counts[rank] += int(result["positive_restart_count"])
            exact_counts[rank] += int(result["exact_hit_count"])
            covered = int(result["best_covered_output_columns"])
            global_restart_index = int(payload["restart_offset"]) + int(result["best_restart_index_in_chunk"])
            if covered > best_covered_by_rank[rank]:
                best_covered_by_rank[rank] = covered
                best_restart_index[rank] = global_restart_index
                best_restart_seed[rank] = int(result["best_restart_seed"])
            completed_chunks += 1
            if completed_chunks == 1 or completed_chunks % progress_step == 0 or covered == 9:
                print(
                    f"    Sweep chunk progress {completed_chunks}/{total_chunks}: rank {rank} best_covered="
                    f"{best_covered_by_rank[rank]}/9"
                , flush=True)

    rows: list[dict] = []
    for rank in ranks:
        rows.append(
            {
                "field": field.name,
                "phase": "sweep",
                "rank_tested": rank,
                "restarts": sweep_restarts,
                "best_covered_output_columns": best_covered_by_rank[rank],
                "positive_restart_count": positive_counts[rank],
                "exact_hit_count": exact_counts[rank],
                "best_restart_index": best_restart_index.get(rank, 0),
                "best_restart_seed": best_restart_seed.get(rank, 0),
                "workers": workers,
                "provenance": "MEASURED_FROM_CODE",
            }
        )

    first_positive_rank = next((rank for rank in ranks if best_covered_by_rank[rank] > 0), None)
    focus_row = None
    if first_positive_rank is not None:
        print(
            f"  Focus: {field.name} rank {first_positive_rank}, {focus_restarts} restarts, workers={workers}, chunk_size={SEARCH_CHUNK_SIZE}."
        )
        best_covered = -1
        positive_restart_count = 0
        exact_hit_count = 0
        best_focus_index = 0
        best_focus_seed = 0
        focus_seeds = [seed_base + 9000000 + 100000 * first_positive_rank + restart_idx for restart_idx in range(focus_restarts)]
        payloads = []
        running_restart_index = 0
        for chunk in split_chunks(focus_seeds, SEARCH_CHUNK_SIZE):
            payloads.append(
                {
                    "field_name": field.name,
                    "rank": first_positive_rank,
                    "restart_offset": running_restart_index,
                    "seeds": chunk,
                }
            )
            running_restart_index += len(chunk)
        progress_step = max(len(payloads) // 10, 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(field_rank_chunk_worker, payload): payload for payload in payloads}
            completed_chunks = 0
            for future in as_completed(futures):
                payload = futures[future]
                result = future.result()
                covered = int(result["best_covered_output_columns"])
                positive_restart_count += int(result["positive_restart_count"])
                exact_hit_count += int(result["exact_hit_count"])
                global_restart_index = int(payload["restart_offset"]) + int(result["best_restart_index_in_chunk"])
                if covered > best_covered:
                    best_covered = covered
                    best_focus_index = global_restart_index
                    best_focus_seed = int(result["best_restart_seed"])
                completed_chunks += 1
                if completed_chunks == 1 or completed_chunks % progress_step == 0 or covered == 9:
                    print(
                        f"    Focus chunk progress {completed_chunks}/{len(payloads)}: rank {first_positive_rank} "
                        f"best_covered={best_covered}/9"
                    , flush=True)
        focus_row = {
            "field": field.name,
            "phase": "focus",
            "rank_tested": first_positive_rank,
            "restarts": focus_restarts,
            "best_covered_output_columns": best_covered,
            "positive_restart_count": positive_restart_count,
            "exact_hit_count": exact_hit_count,
            "best_restart_index": best_focus_index,
            "best_restart_seed": best_focus_seed,
            "workers": workers,
            "provenance": "MEASURED_FROM_CODE",
        }
        rows.append(focus_row)

    summary = {
        "field": field.name,
        "sweep_best_covered_by_rank": {str(rank): best_covered_by_rank[rank] for rank in ranks},
        "first_positive_rank": first_positive_rank,
        "focus_triggered": first_positive_rank is not None,
        "focus_best_covered": None if focus_row is None else focus_row["best_covered_output_columns"],
        "any_exact_hit": any(exact_counts[rank] > 0 for rank in ranks) or (focus_row is not None and int(focus_row["exact_hit_count"]) > 0),
    }
    return rows, summary


def staged_division_search(
    search_specs: list[tuple[int, int]],
    sweep_restarts: int,
    focus_restarts: int,
    workers: int,
) -> tuple[list[dict], dict]:
    total_restarts = len(search_specs) * sweep_restarts
    print(f"  Sweep: Task D cases={len(search_specs)}, {sweep_restarts} restarts each, workers={workers}, chunk_size={SEARCH_CHUNK_SIZE}.")
    stats = {}
    for op_count, div_count in search_specs:
        label = f"{op_count}_ops_{div_count}_div"
        stats[label] = {
            "search_case": label,
            "op_count": op_count,
            "div_count": div_count,
            "valid_structures": 0,
            "best_train_mse": float("inf"),
            "best_valid_mse": float("inf"),
            "exact_hit_count": 0,
            "best_restart_index": 0,
            "best_restart_seed": 0,
        }

    payloads: list[dict] = []
    for op_count, div_count in search_specs:
        seeds = [7700000 + 100000 * op_count + 1000 * div_count + restart_idx for restart_idx in range(sweep_restarts)]
        running_restart_index = 0
        for chunk in split_chunks(seeds, SEARCH_CHUNK_SIZE):
            payloads.append(
                {
                    "op_count": op_count,
                    "div_count": div_count,
                    "restart_offset": running_restart_index,
                    "seeds": chunk,
                }
            )
            running_restart_index += len(chunk)
    completed_chunks = 0
    total_chunks = len(payloads)
    progress_step = max(total_chunks // 10, 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(division_case_chunk_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            label = result["search_case"]
            stat = stats[label]
            stat["valid_structures"] += int(result["valid_structures"])
            stat["exact_hit_count"] += int(result["exact_hit_count"])
            best_train_mse = result["best_train_mse"]
            best_valid_mse = result["best_valid_mse"]
            global_restart_index = int(payload["restart_offset"]) + int(result["best_restart_index_in_chunk"])
            if best_valid_mse is not None and (
                best_valid_mse < stat["best_valid_mse"] or (
                    best_valid_mse == stat["best_valid_mse"] and float(best_train_mse) < stat["best_train_mse"]
                )
            ):
                stat["best_train_mse"] = float(best_train_mse)
                stat["best_valid_mse"] = float(best_valid_mse)
                stat["best_restart_index"] = global_restart_index
                stat["best_restart_seed"] = int(result["best_restart_seed"])
            completed_chunks += 1
            if completed_chunks == 1 or completed_chunks % progress_step == 0:
                best_valid = finite_or_none(stat["best_valid_mse"])
                best_valid_text = "n/a" if best_valid is None else f"{best_valid:.3e}"
                print(
                    f"    Sweep chunk progress {completed_chunks}/{total_chunks}: {label} valid_structures={stat['valid_structures']}, "
                    f"best_valid={best_valid_text}"
                , flush=True)

    rows: list[dict] = []
    for op_count, div_count in search_specs:
        label = f"{op_count}_ops_{div_count}_div"
        stat = stats[label]
        rows.append(
            {
                "search_case": label,
                "phase": "sweep",
                "op_count": op_count,
                "div_count": div_count,
                "restarts": sweep_restarts,
                "valid_structures": stat["valid_structures"],
                "best_train_mse": finite_or_none(stat["best_train_mse"]),
                "best_valid_mse": finite_or_none(stat["best_valid_mse"]),
                "exact_hit_count": stat["exact_hit_count"],
                "best_restart_index": stat["best_restart_index"],
                "best_restart_seed": stat["best_restart_seed"],
                "workers": workers,
                "provenance": "MEASURED_FROM_CODE",
            }
        )

    first_exact_case = next(
        (f"{op_count}_ops_{div_count}_div" for op_count, div_count in search_specs if stats[f"{op_count}_ops_{div_count}_div"]["exact_hit_count"] > 0),
        None,
    )
    focus_row = None
    if first_exact_case is not None:
        focus_op_count = int(first_exact_case.split("_", 1)[0])
        focus_div_count = int(first_exact_case.split("_ops_")[1].split("_div")[0])
        print(
            f"  Focus: Task D case {first_exact_case}, {focus_restarts} restarts, workers={workers}, chunk_size={SEARCH_CHUNK_SIZE}."
        )
        stat = {
            "search_case": first_exact_case,
            "op_count": focus_op_count,
            "div_count": focus_div_count,
            "valid_structures": 0,
            "best_train_mse": float("inf"),
            "best_valid_mse": float("inf"),
            "exact_hit_count": 0,
            "best_restart_index": 0,
            "best_restart_seed": 0,
        }
        focus_seeds = [8800000 + 100000 * focus_op_count + 1000 * focus_div_count + restart_idx for restart_idx in range(focus_restarts)]
        payloads = []
        running_restart_index = 0
        for chunk in split_chunks(focus_seeds, SEARCH_CHUNK_SIZE):
            payloads.append(
                {
                    "op_count": focus_op_count,
                    "div_count": focus_div_count,
                    "restart_offset": running_restart_index,
                    "seeds": chunk,
                }
            )
            running_restart_index += len(chunk)
        progress_step = max(len(payloads) // 10, 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(division_case_chunk_worker, payload): payload for payload in payloads}
            completed_chunks = 0
            for future in as_completed(futures):
                payload = futures[future]
                result = future.result()
                stat["valid_structures"] += int(result["valid_structures"])
                stat["exact_hit_count"] += int(result["exact_hit_count"])
                best_train_mse = result["best_train_mse"]
                best_valid_mse = result["best_valid_mse"]
                global_restart_index = int(payload["restart_offset"]) + int(result["best_restart_index_in_chunk"])
                if best_valid_mse is not None and (
                    best_valid_mse < stat["best_valid_mse"] or (
                        best_valid_mse == stat["best_valid_mse"] and float(best_train_mse) < stat["best_train_mse"]
                    )
                ):
                    stat["best_train_mse"] = float(best_train_mse)
                    stat["best_valid_mse"] = float(best_valid_mse)
                    stat["best_restart_index"] = global_restart_index
                    stat["best_restart_seed"] = int(result["best_restart_seed"])
                completed_chunks += 1
                if completed_chunks == 1 or completed_chunks % progress_step == 0:
                    best_valid = finite_or_none(stat["best_valid_mse"])
                    best_valid_text = "n/a" if best_valid is None else f"{best_valid:.3e}"
                    print(
                        f"    Focus chunk progress {completed_chunks}/{len(payloads)}: {first_exact_case} valid_structures={stat['valid_structures']}, "
                        f"best_valid={best_valid_text}"
                    , flush=True)
        focus_row = {
            "search_case": first_exact_case,
            "phase": "focus",
            "op_count": focus_op_count,
            "div_count": focus_div_count,
            "restarts": focus_restarts,
            "valid_structures": stat["valid_structures"],
            "best_train_mse": finite_or_none(stat["best_train_mse"]),
            "best_valid_mse": finite_or_none(stat["best_valid_mse"]),
            "exact_hit_count": stat["exact_hit_count"],
            "best_restart_index": stat["best_restart_index"],
            "best_restart_seed": stat["best_restart_seed"],
            "workers": workers,
            "provenance": "MEASURED_FROM_CODE",
        }
        rows.append(focus_row)

    sweep_best_case = None
    for row in rows:
        if row["phase"] != "sweep" or row["best_valid_mse"] is None:
            continue
        if sweep_best_case is None or row["best_valid_mse"] < sweep_best_case["best_valid_mse"]:
            sweep_best_case = row

    summary = {
        "first_exact_case": first_exact_case,
        "focus_triggered": first_exact_case is not None,
        "sweep_best_case": None if sweep_best_case is None else sweep_best_case["search_case"],
        "sweep_best_valid_mse": None if sweep_best_case is None else sweep_best_case["best_valid_mse"],
        "any_exact_hit": any(row["exact_hit_count"] > 0 for row in rows),
    }
    return rows, summary


def task_d() -> tuple[list[dict], list[dict], dict[str, str]]:
    print("=== Task D: Division-Augmented 2x2 ===")
    # Least squares is used instead of L-BFGS because output weights enter linearly.
    search_specs = [(5, 0), (6, 0), (6, 1), (7, 1)]
    search_rows, search_summary = staged_division_search(
        search_specs=search_specs,
        sweep_restarts=D_SWEEP_RESTARTS,
        focus_restarts=D_FOCUS_RESTARTS,
        workers=SEARCH_WORKERS,
    )
    summary_rows: list[dict] = []
    for row in search_rows:
        summary_rows.append(
            {
                "summary_name": f"taskD_{row['phase']}_{row['search_case']}",
                "summary_value": json.dumps(
                    {
                        "best_train_mse": row["best_train_mse"],
                        "best_valid_mse": row["best_valid_mse"],
                        "exact_hit_count": row["exact_hit_count"],
                        "valid_structures": row["valid_structures"],
                        "restarts": row["restarts"],
                    },
                    separators=(",", ":"),
                ),
                "provenance": "MEASURED_FROM_CODE",
                "note": f"Staged {row['phase']} search for {row['search_case']}.",
            }
        )
    summary_rows.extend(
        [
            {
                "summary_name": "taskD_any_exact_hit",
                "summary_value": str(search_summary["any_exact_hit"]),
                "provenance": "MEASURED_FROM_CODE",
                "note": "Whether any staged Task D search produced an exact candidate.",
            },
            {
                "summary_name": "taskD_focus_triggered",
                "summary_value": str(search_summary["focus_triggered"]),
                "provenance": "MEASURED_FROM_CODE",
                "note": "Whether the focus phase ran after the sweep.",
            },
            {
                "summary_name": "taskD_first_exact_case",
                "summary_value": str(search_summary["first_exact_case"]),
                "provenance": "MEASURED_FROM_CODE",
                "note": "First sweep case with any exact candidate, if one occurred.",
            },
            {
                "summary_name": "taskD_sweep_best_case",
                "summary_value": str(search_summary["sweep_best_case"]),
                "provenance": "MEASURED_FROM_CODE",
                "note": "Sweep case with the lowest validation MSE among valid structures.",
            },
            {
                "summary_name": "taskD_sweep_best_valid_mse",
                "summary_value": str(search_summary["sweep_best_valid_mse"]),
                "provenance": "MEASURED_FROM_CODE",
                "note": "Best validation MSE achieved in the sweep phase.",
            },
        ]
    )
    notes = {
        "best_case": str(search_summary["sweep_best_case"]),
        "best_valid_mse": "n/a"
        if search_summary["sweep_best_valid_mse"] is None
        else f"{search_summary['sweep_best_valid_mse']:.6e}",
        "exact_hit": str(search_summary["any_exact_hit"]),
        "focus_triggered": str(search_summary["focus_triggered"]),
    }
    print(
        f"Task D complete: best sweep case {notes['best_case']} reached valid MSE {notes['best_valid_mse']}; "
        f"exact hit={notes['exact_hit']}; focus_triggered={notes['focus_triggered']}."
    )
    return summary_rows, search_rows, notes


def task_e() -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    print("=== Task E: Frobenius over GF(9) ===")
    element_rows = []
    for code, label in enumerate(GF9.labels):
        conj = int(GF9.frob[code])
        element_rows.append(
            {
                "code": code,
                "label": label,
                "frobenius_cube_code": conj,
                "frobenius_cube_label": GF9.labels[conj],
                "frobenius_order_two": str(int(GF9.frob[conj]) == code),
                "provenance": "EXACT_DERIVED",
            }
        )

    gf9_rows, gf9_summary = staged_field_rank_sweep(
        field=GF9,
        ranks=GF9_SWEEP_RANKS,
        sweep_restarts=GF9_SWEEP_RESTARTS,
        focus_restarts=GF9_FOCUS_RESTARTS,
        seed_base=7699001,
        workers=SEARCH_WORKERS,
    )
    summary_rows = [
        {
            "summary_name": "taskE_GF9_sweep_best_covered_by_rank",
            "summary_value": json.dumps(gf9_summary["sweep_best_covered_by_rank"], separators=(",", ":")),
            "provenance": "MEASURED_FROM_CODE",
            "note": "Best covered output count at each sweep rank in the ordinary GF(9) bilinear model.",
        },
        {
            "summary_name": "taskE_GF9_first_positive_rank",
            "summary_value": str(gf9_summary["first_positive_rank"]),
            "provenance": "MEASURED_FROM_CODE",
            "note": "First sweep rank where best_covered_output_columns > 0.",
        },
        {
            "summary_name": "taskE_GF9_focus_triggered",
            "summary_value": str(gf9_summary["focus_triggered"]),
            "provenance": "MEASURED_FROM_CODE",
            "note": "Whether the 5000-restart focus phase ran.",
        },
        {
            "summary_name": "taskE_GF9_focus_best_covered",
            "summary_value": str(gf9_summary["focus_best_covered"]),
            "provenance": "MEASURED_FROM_CODE",
            "note": "Best covered output count in the focus phase, if triggered.",
        },
        {
            "summary_name": "taskE_GF9_any_exact_hit",
            "summary_value": str(gf9_summary["any_exact_hit"]),
            "provenance": "MEASURED_FROM_CODE",
            "note": "Whether any staged GF(9) search hit all 9 output columns exactly.",
        },
        {
            "summary_name": "taskE_GF9_rank_bounds_after_quick_test",
            "summary_value": "9 <= rank_GF9 <= 23",
            "provenance": "EXACT_DERIVED + MEASURED_FROM_CODE",
            "note": "Flattenings still give 9 and the GF(3)-valued AlphaTensor coefficients embed into GF(9), giving a 23-term upper bound.",
        },
        {
            "summary_name": "taskE_frobenius_free_changes_tensor_rank_model",
            "summary_value": "True",
            "provenance": "EXACT_DERIVED",
            "note": "Free Frobenius makes the model semilinear over GF(3), not an ordinary GF(9)-bilinear tensor-rank problem.",
        },
        {
            "summary_name": "taskE_conservative_proxy_used",
            "summary_value": "ordinary_GF9_tensor_rank_search_only",
            "provenance": "EXACT_DERIVED",
            "note": "The quick test computes the exact ordinary GF(9) span condition and reports Frobenius-free semilinear rank as unresolved, because that is not standard tensor rank.",
        },
    ]
    notes = {
        "gf9_first_positive_rank": str(gf9_summary["first_positive_rank"]),
        "gf9_any_exact": str(gf9_summary["any_exact_hit"]),
        "gf9_focus_triggered": str(gf9_summary["focus_triggered"]),
        "model_note": "free_Frobenius_is_semilinear_not_ordinary_tensor_rank",
    }
    print(
        f"Task E complete: first positive sweep rank={gf9_summary['first_positive_rank']}; "
        f"any exact hit={gf9_summary['any_exact_hit']}; focus_triggered={gf9_summary['focus_triggered']}."
    )
    return summary_rows, element_rows, gf9_rows, notes


def write_markdown(
    path: Path,
    task_a_notes: dict[str, str] | None,
    task_b_notes: dict[str, str] | None,
    task_c_notes: dict[str, str] | None,
    task_d_notes: dict[str, str] | None,
    task_e_notes: dict[str, str] | None,
) -> None:
    lines: list[str] = []
    w = lines.append
    w("# Step 76: Batch of Five Quick Tests")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("[EXACT_DERIVED] + [MEASURED_FROM_CODE]")
    w("")
    if task_a_notes is not None:
        w("## Task A: Pareto Frontier")
        w("")
        w(f"- AlphaTensor requested addition count: {task_a_notes['alpha_requested_A']}")
        w(f"- AlphaTensor scheduled addition reference: {task_a_notes['alpha_scheduled_A']}")
        w(f"- Best 26-multiplication tiling proxy: {task_a_notes['best_tiling_configuration']}")
        w(f"- Best 26-multiplication tiling scheduled proxy additions: {task_a_notes['best_tiling_A_scheduled_proxy']}")
        w(f"- Requested-count crossover where AlphaTensor beats standard: p >= {task_a_notes['requested_crossover']}")
        w(f"- Scheduled-count crossover where AlphaTensor beats standard: p >= {task_a_notes['scheduled_crossover']}")
        w("")
        w("The 26-multiplication tiling remains only a tiling upper-bound configuration, not a certified global 3x3 algorithm. Its addition count is therefore reported as a concrete proxy from piece witnesses rather than as a canonical algorithmic invariant.")
        w("")
    if task_b_notes is not None:
        w("## Task B: Cayley-Hamilton")
        w("")
        w("Verdict: no. Cayley-Hamilton does not reduce the number of bilinear quantities needed for 3x3 matrix multiplication.")
        w("The key proof is surjectivity: every 3x3 matrix C occurs as AB by taking A = I and B = C. So Cayley-Hamilton cannot impose a nontrivial output relation that lowers the dimension of the image.")
        w("")
    if task_c_notes is not None:
        w("## Task C: Squaring as Primitive")
        w("")
        w(f"- Distinct alpha forms: {task_c_notes['distinct_alpha']}")
        w(f"- Distinct beta forms: {task_c_notes['distinct_beta']}")
        w(f"- Distinct alpha±beta forms: {task_c_notes['distinct_pm']}")
        w(f"- GF(3) rank-22 exact hit in bounded search: {task_c_notes['gf3_rank22_exact']}")
        w("")
        w("The user-supplied GF(3) premise x^2 = x is false. Over GF(3), Frobenius is x^3 = x; squaring is a different nonlinear primitive.")
        w("")
    if task_d_notes is not None:
        w("## Task D: Division-Augmented 2x2")
        w("")
        w(f"- Best sweep case: {task_d_notes['best_case']}")
        w(f"- Best sweep validation MSE: {task_d_notes['best_valid_mse']}")
        w(f"- Any exact hit found: {task_d_notes['exact_hit']}")
        w(f"- Focus phase triggered: {task_d_notes['focus_triggered']}")
        w("")
        w("The search now runs as a staged sweep over the exported operation/division cases, with a focus phase reserved for the first case that produces any exact hit.")
        w("Output weights are still optimized by least squares rather than L-BFGS because the weighting layer is linear once the primitive operations are fixed.")
        w("")
    if task_e_notes is not None:
        w("## Task E: GF(9) and Frobenius")
        w("")
        w(f"- First positive GF(9) sweep rank: {task_e_notes['gf9_first_positive_rank']}")
        w(f"- Any exact GF(9) hit in the ordinary bilinear model: {task_e_notes['gf9_any_exact']}")
        w(f"- Focus phase triggered: {task_e_notes['gf9_focus_triggered']}")
        w("- Free Frobenius was treated conservatively as outside ordinary tensor rank.")
        w("")
        w("Free Frobenius yields a semilinear model over GF(3), not an ordinary GF(9)-bilinear tensor decomposition. The exact quick test therefore reports a staged ordinary-GF(9) sweep/focus proxy and leaves the semilinear Frobenius-free rank unresolved.")
        w("")
    write_text(path, "\n".join(lines))


def run(selected_tasks: set[str]) -> None:
    all_summary_rows: list[dict] = []
    task_a_notes = None
    task_b_notes = None
    task_c_notes = None
    task_d_notes = None
    task_e_notes = None

    if "A" in selected_tasks:
        summary_rows, cost_rows, tiling_rows, a4_rows, task_a_notes, square_rows = task_a()
        all_summary_rows.extend(summary_rows)
        write_csv(
            EXPORTS / "step76_taskA_costs.csv",
            cost_rows,
            ["scheme", "p", "M", "A", "cost", "count_model", "provenance"],
        )
        write_csv(
            EXPORTS / "step76_taskA_tiling_proxy_costs.csv",
            tiling_rows,
            ["configuration", "M", "A_scatter_proxy", "A_scheduled_proxy", "source", "status", "provenance"],
        )
        write_csv(
            EXPORTS / "step76_taskA_hypothetical_24_25_thresholds.csv",
            a4_rows,
            [
                "p",
                "max_A_for_24M_to_beat_requested_alphatensor",
                "max_A_for_25M_to_beat_requested_alphatensor",
                "max_A_for_26M_to_beat_requested_alphatensor",
                "known_certified_24M_or_25M_algorithm_in_repo",
                "known_26M_configuration_in_repo",
                "provenance",
            ],
        )
        write_csv(
            EXPORTS / "step76_taskA_square_tetromino_witness.csv",
            square_rows,
            ["term_id", "alpha", "beta", "gamma", "alpha_nonzero", "beta_nonzero", "gamma_nonzero", "provenance"],
        )

    if "B" in selected_tasks:
        summary_rows, detail_rows, task_b_notes = task_b()
        all_summary_rows.extend(summary_rows)
        formula_rows = [row for row in detail_rows if "formula_name" in row]
        claim_rows = [row for row in detail_rows if "claim" in row]
        write_csv(
            EXPORTS / "step76_taskB_cayley_hamilton_formulas.csv",
            formula_rows,
            ["formula_name", "formula", "total_degree_in_(A,B)", "provenance"],
        )
        write_csv(
            EXPORTS / "step76_taskB_cayley_hamilton_claims.csv",
            claim_rows,
            ["claim", "verdict", "reason", "provenance"],
        )

    if "C" in selected_tasks:
        summary_rows, count_rows, gf3_rows, task_c_notes = task_c()
        all_summary_rows.extend(summary_rows)
        write_csv(
            EXPORTS / "step76_taskC_squaring_counts.csv",
            count_rows,
            ["quantity", "value", "provenance"],
        )
        write_csv(
            EXPORTS / "step76_taskC_gf3_search.csv",
            gf3_rows,
            union_fieldnames(gf3_rows) if gf3_rows else ["field", "rank_tested", "restart", "covered_output_columns", "exact", "provenance"],
        )

    if "D" in selected_tasks:
        summary_rows, search_rows, task_d_notes = task_d()
        all_summary_rows.extend(summary_rows)
        write_csv(
            EXPORTS / "step76_taskD_division_augmented_search.csv",
            search_rows,
            union_fieldnames(search_rows)
            if search_rows
            else ["search_case", "phase", "op_count", "div_count", "restarts", "valid_structures", "best_train_mse", "best_valid_mse", "exact_hit_count", "best_restart_index", "best_restart_seed", "workers", "provenance"],
        )

    if "E" in selected_tasks:
        summary_rows, element_rows, gf9_rows, task_e_notes = task_e()
        all_summary_rows.extend(summary_rows)
        write_csv(
            EXPORTS / "step76_taskE_gf9_elements.csv",
            element_rows,
            ["code", "label", "frobenius_cube_code", "frobenius_cube_label", "frobenius_order_two", "provenance"],
        )
        write_csv(
            EXPORTS / "step76_taskE_gf9_search.csv",
            gf9_rows,
            union_fieldnames(gf9_rows)
            if gf9_rows
            else ["field", "phase", "rank_tested", "restarts", "best_covered_output_columns", "positive_restart_count", "exact_hit_count", "best_restart_index", "best_restart_seed", "workers", "provenance"],
        )

    write_csv(
        EXPORTS / "step76_summary.csv",
        all_summary_rows,
        ["summary_name", "summary_value", "provenance", "note"],
    )
    write_markdown(
        EXPORTS / "step76_batch_of_five_quick_tests.md",
        task_a_notes,
        task_b_notes,
        task_c_notes,
        task_d_notes,
        task_e_notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 76 quick-test batch runner.")
    parser.add_argument(
        "--tasks",
        default="A,B,C,D,E",
        help="Comma-separated task list from {A,B,C,D,E}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=SEARCH_WORKERS,
        help="Process workers for the staged Task D/E searches.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=SEARCH_CHUNK_SIZE,
        help="Restarts grouped into each submitted parallel job for Task D/E searches.",
    )
    return parser.parse_args()


def main() -> None:
    global SEARCH_WORKERS, SEARCH_CHUNK_SIZE
    multiprocessing.freeze_support()
    args = parse_args()
    SEARCH_WORKERS = max(1, int(args.workers))
    SEARCH_CHUNK_SIZE = max(1, int(args.chunk_size))
    selected_tasks = {chunk.strip().upper() for chunk in args.tasks.split(",") if chunk.strip()}
    invalid = sorted(selected_tasks - {"A", "B", "C", "D", "E"})
    if invalid:
        raise ValueError(f"Unknown task ids: {invalid}")
    print("=== Step 76: Batch of Five Quick Tests ===")
    print(f"Running tasks: {', '.join(sorted(selected_tasks))}")
    print(f"Parallel search config: workers={SEARCH_WORKERS}, chunk_size={SEARCH_CHUNK_SIZE}")
    run(selected_tasks)
    print("Step 76 completed.")


if __name__ == "__main__":
    main()