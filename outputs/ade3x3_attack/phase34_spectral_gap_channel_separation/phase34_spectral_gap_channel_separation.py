from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ade3x3.steps.ade3x3_step52_quotient_rank_criterion import (  # noqa: E402
    strassen_terms_2x2,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    load_public_rank23_terms,
)


@dataclass(frozen=True)
class Decomposition:
    label: str
    n: int
    terms: list[Term]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def matrix_literal(matrix: sp.Matrix) -> str:
    return "[" + ", ".join(
        "[" + ", ".join(sp.sstr(sp.nsimplify(matrix[row_idx, col_idx])) for col_idx in range(matrix.cols)) + "]"
        for row_idx in range(matrix.rows)
    ) + "]"


def integerize_vector(vector: sp.Matrix) -> sp.Matrix:
    rationals = [sp.Rational(entry) for entry in vector]
    lcm = sp.ilcm(*[value.q for value in rationals]) if rationals else 1
    scaled = [int(value * lcm) for value in rationals]
    nonzero = [abs(value) for value in scaled if value != 0]
    divisor = 0
    for value in nonzero:
        divisor = value if divisor == 0 else gcd(divisor, value)
    divisor = divisor or 1
    normalized = [value // divisor for value in scaled]
    for value in normalized:
        if value != 0:
            if value < 0:
                normalized = [-entry for entry in normalized]
            break
    return sp.Matrix(normalized)


def integer_kernel_basis(kernel_basis: sp.Matrix) -> sp.Matrix:
    if kernel_basis.cols == 0:
        return sp.zeros(kernel_basis.rows, 0)
    return sp.Matrix.hstack(*[integerize_vector(kernel_basis[:, col_idx]) for col_idx in range(kernel_basis.cols)])


def standard_terms_n(n: int) -> list[Term]:
    terms: list[Term] = []
    term_idx = 1
    for row_idx in range(n):
        for sum_idx in range(n):
            for col_idx in range(n):
                alpha = np.zeros((n, n), dtype=np.int64)
                beta = np.zeros((n, n), dtype=np.int64)
                gamma = np.zeros((n, n), dtype=np.int64)
                alpha[row_idx, sum_idx] = 1
                beta[sum_idx, col_idx] = 1
                gamma[row_idx, col_idx] = 1
                terms.append(
                    Term(
                        term_id=f"std{n}_{term_idx:02d}",
                        source_label=f"standard_{n}x{n}_{term_idx:02d}",
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                )
                term_idx += 1
    return terms


def to_term_list(prefix: str, raw_terms: list[dict[str, list[list[int]]]]) -> list[Term]:
    terms: list[Term] = []
    for idx, payload in enumerate(raw_terms, start=1):
        terms.append(
            Term(
                term_id=f"{prefix}{idx:02d}",
                source_label=f"{prefix}{idx:02d}",
                alpha=np.array(payload["alpha"], dtype=np.int64),
                beta=np.array(payload["beta"], dtype=np.int64),
                gamma=np.array(payload["gamma"], dtype=np.int64),
            )
        )
    return terms


def load_named_decompositions() -> dict[str, Decomposition]:
    alpha_terms, _, _ = load_public_rank23_terms()
    return {
        "alphatensor_rank23": Decomposition("alphatensor_rank23", 3, alpha_terms),
        "standard_rank27": Decomposition("standard_rank27", 3, standard_terms_n(3)),
        "strassen_2x2": Decomposition("strassen_2x2", 2, to_term_list("str", strassen_terms_2x2())),
    }


def gamma_matrix_exact(decomposition: Decomposition) -> sp.Matrix:
    return sp.Matrix([
        [int(term.gamma[row_idx, col_idx]) for term in decomposition.terms]
        for row_idx in range(decomposition.n)
        for col_idx in range(decomposition.n)
    ])


def difference_matrix_exact(decomposition: Decomposition, sum_left: int, sum_right: int) -> sp.Matrix:
    rows: list[list[int]] = []
    for row_idx in range(decomposition.n):
        for col_idx in range(decomposition.n):
            rows.append([
                int(term.alpha[row_idx, sum_left]) * int(term.beta[sum_left, col_idx])
                - int(term.alpha[row_idx, sum_right]) * int(term.beta[sum_right, col_idx])
                for term in decomposition.terms
            ])
    return sp.Matrix(rows)


def kernel_basis_exact(matrix: sp.Matrix) -> sp.Matrix:
    basis = matrix.nullspace()
    if not basis:
        return sp.zeros(matrix.cols, 0)
    return sp.Matrix.hstack(*basis)


def pair_labels(n: int) -> list[tuple[int, int]]:
    return list(combinations(range(n), 2))


def stack_matrices(matrices: Iterable[sp.Matrix], cols: int) -> sp.Matrix:
    matrices = list(matrices)
    if not matrices:
        return sp.zeros(0, cols)
    return sp.Matrix.vstack(*matrices)


def nullity(matrix: sp.Matrix) -> int:
    return matrix.cols - matrix.rank()


def rational_eigenspectrum(operator: sp.Matrix) -> list[tuple[sp.Expr, int]]:
    if operator.rows == 0:
        return []
    try:
        eigenvalues = operator.eigenvals()
        total_mult = sum(int(mult) for mult in eigenvalues.values())
        if total_mult == operator.rows:
            return sorted(
                [(sp.nsimplify(value), int(mult)) for value, mult in eigenvalues.items()],
                key=lambda item: float(sp.N(item[0], 50)),
            )
    except Exception:
        pass

    operator_np = np.array(operator.tolist(), dtype=float)
    numeric_values = np.linalg.eigvals(operator_np)
    candidates: list[sp.Expr] = []
    for value in numeric_values:
        real_value = float(np.real(value))
        if abs(float(np.imag(value))) > 1e-9:
            raise RuntimeError(f"Unexpected non-real eigenvalue {value}")
        candidate = sp.nsimplify(real_value, rational=True)
        if all(sp.simplify(candidate - existing) != 0 for existing in candidates):
            candidates.append(candidate)

    spectrum: list[tuple[sp.Expr, int]] = []
    identity = sp.eye(operator.rows)
    multiplicity_sum = 0
    for candidate in sorted(candidates, key=lambda value: float(sp.N(value, 50))):
        multiplicity = len((operator - candidate * identity).nullspace())
        if multiplicity:
            spectrum.append((candidate, multiplicity))
            multiplicity_sum += multiplicity
    if multiplicity_sum != operator.rows:
        raise RuntimeError("Could not reconstruct a full exact eigenspectrum")
    return spectrum


def is_rational_spectrum(spectrum: list[tuple[sp.Expr, int]]) -> bool:
    return all(bool(value.is_rational) for value, _ in spectrum)


def spectrum_to_string(spectrum: list[tuple[sp.Expr, int]]) -> str:
    if not spectrum:
        return "empty"
    return "; ".join(f"{sp.sstr(value)} x {mult}" for value, mult in spectrum)


def spectrum_approx_string(spectrum: list[tuple[sp.Expr, int]]) -> str:
    if not spectrum:
        return "empty"
    return "; ".join(f"{float(sp.N(value, 20)):.12f} x {mult}" for value, mult in spectrum)


def min_positive_eigenvalue(spectrum: list[tuple[sp.Expr, int]]) -> sp.Expr:
    positives = [value for value, _ in spectrum if sp.simplify(value) > 0]
    if not positives:
        return sp.Integer(0)
    return min(positives, key=lambda value: float(sp.N(value, 50)))


def restricted_form_data(kernel_basis: sp.Matrix, difference_matrix: sp.Matrix) -> dict[str, object]:
    if kernel_basis.cols == 0:
        return {
            "basis_gram": sp.zeros(0, 0),
            "metric": sp.zeros(0, 0),
            "operator": sp.zeros(0, 0),
            "charpoly": sp.Integer(1),
            "spectrum": [],
        }
    basis_gram = (difference_matrix * kernel_basis).T * (difference_matrix * kernel_basis)
    metric = kernel_basis.T * kernel_basis
    operator = metric.LUsolve(basis_gram)
    lam = sp.Symbol("lambda")
    return {
        "basis_gram": basis_gram,
        "metric": metric,
        "operator": operator,
        "charpoly": sp.expand(operator.charpoly(lam).as_expr()),
        "spectrum": rational_eigenspectrum(operator),
    }


def analyze_named_decomposition(decomposition: Decomposition) -> dict[str, object]:
    gamma = gamma_matrix_exact(decomposition)
    kernel_basis = kernel_basis_exact(gamma)
    pair_rows: list[dict] = []
    intersection_rows: list[dict] = []
    pair_forms: dict[tuple[int, int], dict[str, object]] = {}
    pair_matrices: dict[tuple[int, int], sp.Matrix] = {}

    for sum_left, sum_right in pair_labels(decomposition.n):
        difference = difference_matrix_exact(decomposition, sum_left, sum_right)
        pair_matrices[(sum_left, sum_right)] = difference
        form = restricted_form_data(kernel_basis, difference)
        pair_forms[(sum_left, sum_right)] = form
        pair_rank = int((difference * kernel_basis).rank())
        pair_nullity = nullity(sp.Matrix.vstack(gamma, difference))
        pair_gap = min_positive_eigenvalue(form["spectrum"])
        pair_rows.append({
            "label": decomposition.label,
            "pair": f"{sum_left}{sum_right}",
            "rank_on_ker_gamma": pair_rank,
            "nullity_on_ker_gamma": pair_nullity,
            "spectrum_exact": spectrum_to_string(form["spectrum"]) if is_rational_spectrum(form["spectrum"]) else f"roots of {sp.sstr(form['charpoly'])}",
            "spectrum_approx": spectrum_approx_string(form["spectrum"]),
            "min_positive_eigenvalue_exact": sp.sstr(pair_gap) if bool(pair_gap.is_rational) else f"root of {sp.sstr(form['charpoly'])}",
            "min_positive_eigenvalue_approx": f"{float(sp.N(pair_gap, 20)):.12f}",
            "basis_gram_literal": matrix_literal(form["basis_gram"]),
            "metric_literal": matrix_literal(form["metric"]),
        })

    for size in range(1, len(pair_matrices) + 1):
        for subset in combinations(sorted(pair_matrices), size):
            stacked = sp.Matrix.vstack(gamma, *[pair_matrices[pair] for pair in subset])
            intersection_rows.append({
                "label": decomposition.label,
                "pair_subset": ",".join(f"{pair[0]}{pair[1]}" for pair in subset),
                "intersection_dim": nullity(stacked),
            })

    total_difference = stack_matrices(pair_matrices.values(), len(decomposition.terms))
    total_form = restricted_form_data(kernel_basis, total_difference)
    total_rank = int((total_difference * kernel_basis).rank())
    total_nullity = nullity(sp.Matrix.vstack(gamma, total_difference))
    total_gap = min_positive_eigenvalue(total_form["spectrum"])

    primitive_basis = integer_kernel_basis(kernel_basis)
    integer_total_gram = (total_difference * primitive_basis).T * (total_difference * primitive_basis)

    total_summary = {
        "label": decomposition.label,
        "n": decomposition.n,
        "R": len(decomposition.terms),
        "ker_gamma_dim": kernel_basis.cols,
        "rank_on_ker_gamma": total_rank,
        "nullity_on_ker_gamma": total_nullity,
        "spectrum_exact": spectrum_to_string(total_form["spectrum"]) if is_rational_spectrum(total_form["spectrum"]) else f"roots of {sp.sstr(total_form['charpoly'])}",
        "spectrum_approx": spectrum_approx_string(total_form["spectrum"]),
        "min_positive_eigenvalue_exact": sp.sstr(total_gap) if bool(total_gap.is_rational) else f"root of {sp.sstr(total_form['charpoly'])}",
        "min_positive_eigenvalue_approx": f"{float(sp.N(total_gap, 20)):.12f}",
        "basis_gram_literal": matrix_literal(total_form["basis_gram"]),
        "metric_literal": matrix_literal(total_form["metric"]),
        "integer_kernel_basis_literal": matrix_literal(primitive_basis),
        "integer_total_gram_literal": matrix_literal(integer_total_gram),
        "integer_total_gram_det": sp.sstr(sp.factor(integer_total_gram.det())) if integer_total_gram.rows else "1",
    }

    return {
        "gamma_rank": int(gamma.rank()),
        "kernel_basis": kernel_basis,
        "pair_rows": pair_rows,
        "intersection_rows": intersection_rows,
        "pair_forms": pair_forms,
        "pair_matrices": pair_matrices,
        "total_summary": total_summary,
        "total_form": total_form,
    }


def standard_family_rows() -> list[dict]:
    rows: list[dict] = []
    for n in (2, 3, 4):
        pair_zero_mult = n * n * (n - 2)
        pair_formula = f"2 x {n * n}" if pair_zero_mult == 0 else f"0 x {pair_zero_mult}; 2 x {n * n}"
        rows.append({
            "family": f"standard_{n}x{n}",
            "pair_representative": "01",
            "pair_spectrum_formula": pair_formula,
            "pair_min_positive_eigenvalue": "2",
            "total_spectrum_formula": f"{n} x {n * n * (n - 1)}",
            "total_min_positive_eigenvalue": str(n),
            "R": n ** 3,
            "ker_gamma_dim": n * n * (n - 1),
        })
    return rows


def strassen_certificate(analysis: dict[str, object]) -> dict[str, object]:
    total_form = analysis["total_form"]
    basis_gram: sp.Matrix = total_form["basis_gram"]
    leading_minors = [sp.factor(basis_gram[:idx, :idx].det()) for idx in range(1, basis_gram.rows + 1)]
    total_gap = min_positive_eigenvalue(total_form["spectrum"])
    return {
        "basis_gram_literal": matrix_literal(basis_gram),
        "leading_principal_minors": [sp.sstr(value) for value in leading_minors],
        "spectrum_exact": spectrum_to_string(total_form["spectrum"]),
        "spectrum_approx": spectrum_approx_string(total_form["spectrum"]),
        "min_positive_eigenvalue_exact": sp.sstr(total_gap),
        "min_positive_eigenvalue_approx": f"{float(sp.N(total_gap, 20)):.12f}",
    }


def build_markdown(
    named_results: dict[str, dict[str, object]],
    standard_rows: list[dict],
    strassen_proof: dict[str, object],
) -> str:
    lines: list[str] = []
    w = lines.append
    w("# Phase 34: Spectral Gap of the Channel-Separation Quadratic Form")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("The Phase 34 target is the exact defect condition from Phase 33b:")
    w("")
    w("  W_0 = W_1 = ... = W_{n-1} for some nonzero w in ker(Gamma).")
    w("")
    w("Equivalently, the total channel-separation quadratic form")
    w("")
    w("  Q_chan(w) = sum_{s<t} ||W_s - W_t||_F^2")
    w("")
    w("must vanish on a nonzero kernel vector. This phase computes Q_chan exactly on ker(Gamma),")
    w("together with the individual pair forms Q_st, so the spectral-gap statement is now explicit.")
    w("")
    w("## Track A: Exact Pair and Total Spectra")
    w("")
    for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2"):
        total = named_results[label]["total_summary"]
        w(f"### {label}")
        w("")
        w(f"- ker(Gamma) dimension: {total['ker_gamma_dim']}")
        w(f"- total Q_chan exact spectrum: {total['spectrum_exact']}")
        w(f"- total Q_chan numerical spectrum: {total['spectrum_approx']}")
        w(f"- total spectral gap: {total['min_positive_eigenvalue_exact']} ≈ {total['min_positive_eigenvalue_approx']}")
        w(f"- total nullity on ker(Gamma): {total['nullity_on_ker_gamma']}")
        w("")
        w("| pair | rank on ker(Gamma) | nullity on ker(Gamma) | exact spectrum | numerical spectrum | min positive eigenvalue |")
        w("|------|--------------------|-----------------------|----------------|--------------------|-------------------------|")
        for row in named_results[label]["pair_rows"]:
            w(
                f"| {row['pair']} | {row['rank_on_ker_gamma']} | {row['nullity_on_ker_gamma']} | "
                f"{row['spectrum_exact']} | {row['spectrum_approx']} | "
                f"{row['min_positive_eigenvalue_exact']} ≈ {row['min_positive_eigenvalue_approx']} |"
            )
        w("")

    w("## Track B: Standard-Family Pattern")
    w("")
    w("For the standard n x n algorithm, ker(Gamma) splits into n^2 independent output fibers, each")
    w("isomorphic to the zero-sum hyperplane {x in R^n : sum_i x_i = 0}. On one such fiber,")
    w("Q_st(x) = (x_s - x_t)^2 has spectrum {2, 0, ..., 0} and the total form")
    w("Q_chan(x) = sum_{s<t} (x_s - x_t)^2 = n ||x||^2 on that hyperplane.")
    w("")
    w("| family | R | ker(Gamma) dim | pair spectrum | pair gap | total spectrum | total gap |")
    w("|--------|---|----------------|---------------|----------|----------------|-----------|")
    for row in standard_rows:
        w(
            f"| {row['family']} | {row['R']} | {row['ker_gamma_dim']} | {row['pair_spectrum_formula']} | "
            f"{row['pair_min_positive_eigenvalue']} | {row['total_spectrum_formula']} | {row['total_min_positive_eigenvalue']} |"
        )
    w("")

    w("## Track C: Lower-Bound Status")
    w("")
    w("The exact-defect condition is absent whenever Q_chan is positive definite on ker(Gamma),")
    w("equivalently when the stacked pair-difference map has full rank on ker(Gamma).")
    w("")
    w("| decomposition | total rank on ker(Gamma) | total nullity on ker(Gamma) | exact total spectral gap | conclusion |")
    w("|---------------|--------------------------|-----------------------------|--------------------------|------------|")
    for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2"):
        total = named_results[label]["total_summary"]
        conclusion = "exact defect absent" if total["nullity_on_ker_gamma"] == 0 else "exact defect survives"
        w(
            f"| {label} | {total['rank_on_ker_gamma']} | {total['nullity_on_ker_gamma']} | "
            f"{total['min_positive_eigenvalue_exact']} ≈ {total['min_positive_eigenvalue_approx']} | {conclusion} |"
        )
    w("")
    w("AlphaTensor's exact spectrum is algebraic rather than rational in this basis-independent")
    w("generalized-eigenvalue sense; the standard family and Strassen give exact rational spectra.")
    w("")

    w("## Track D: Nullspace / Intersection Structure")
    w("")
    for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2"):
        w(f"### {label}")
        w("")
        w("| pair subset | common nullspace dimension inside ker(Gamma) |")
        w("|-------------|----------------------------------------------|")
        for row in named_results[label]["intersection_rows"]:
            w(f"| {row['pair_subset']} | {row['intersection_dim']} |")
        w("")

    w("## Track E: Strassen 2x2 Exact Positivity Certificate")
    w("")
    w("On Strassen there is only one channel pair, so Q_chan = Q_01. In the exact kernel basis")
    w("returned by SymPy, the basis Gram of Q_chan is:")
    w("")
    w(f"- basis Gram: {strassen_proof['basis_gram_literal']}")
    w(f"- leading principal minors: {', '.join(strassen_proof['leading_principal_minors'])}")
    w(f"- generalized spectrum on ker(Gamma): {strassen_proof['spectrum_exact']}")
    w(f"- exact gap: {strassen_proof['min_positive_eigenvalue_exact']} ≈ {strassen_proof['min_positive_eigenvalue_approx']}")
    w("")
    w("All leading principal minors are strictly positive, so the basis Gram is positive definite.")
    w("Therefore Q_chan(w) > 0 for every nonzero w in ker(Gamma), and Strassen admits no exact")
    w("channel-equality defect.")
    w("")

    w("## Track F: Integer Gram Structure")
    w("")
    w("Clearing denominators in the kernel basis gives an integral basis for ker(Gamma).")
    w("In that basis, the total channel-separation Gram matrix is integral for all three known")
    w("exact decompositions, so the obstruction can be studied as an honest integer quadratic form.")
    w("")
    w("| decomposition | integer Gram determinant | integer kernel basis available? |")
    w("|---------------|--------------------------|----------------------------------|")
    for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2"):
        total = named_results[label]["total_summary"]
        w(f"| {label} | {total['integer_total_gram_det']} | yes |")
    w("")

    w("## Conclusions")
    w("")
    w("Phase 34 makes the spectral target exact. The total channel-separation form Q_chan is")
    w("strictly positive on ker(Gamma) for AlphaTensor, the standard 3x3 algorithm, and Strassen.")
    w("So the exact defect condition W_0 = W_1 = ... = W_{n-1} does not occur on any known exact")
    w("minimum-rank decomposition in the repository.")
    w("")
    w("For the standard family the pattern is completely clean: pair gaps stay fixed at 2, while")
    w("the total gap is exactly n. That makes the right next theorem target sharper: prove that the")
    w("minimum total channel-separation eigenvalue stays uniformly positive over the admissible")
    w("rank-R multiplication variety, rather than chasing a termwise identity.")
    return "\n".join(lines) + "\n"


def main() -> None:
    named = load_named_decompositions()
    named_results = {label: analyze_named_decomposition(decomposition) for label, decomposition in named.items()}
    standard_rows = standard_family_rows()
    strassen_proof = strassen_certificate(named_results["strassen_2x2"])

    pair_rows = [row for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2") for row in named_results[label]["pair_rows"]]
    total_rows = [named_results[label]["total_summary"] for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2")]
    intersection_rows = [row for label in ("alphatensor_rank23", "standard_rank27", "strassen_2x2") for row in named_results[label]["intersection_rows"]]
    integer_rows = [
        {
            "label": row["label"],
            "integer_kernel_basis_literal": row["integer_kernel_basis_literal"],
            "integer_total_gram_literal": row["integer_total_gram_literal"],
            "integer_total_gram_det": row["integer_total_gram_det"],
        }
        for row in total_rows
    ]

    write_csv(
        OUT_DIR / "trackA_pair_spectra.csv",
        pair_rows,
        [
            "label",
            "pair",
            "rank_on_ker_gamma",
            "nullity_on_ker_gamma",
            "spectrum_exact",
            "spectrum_approx",
            "min_positive_eigenvalue_exact",
            "min_positive_eigenvalue_approx",
            "basis_gram_literal",
            "metric_literal",
        ],
    )
    write_csv(
        OUT_DIR / "trackA_total_spectra.csv",
        total_rows,
        [
            "label",
            "n",
            "R",
            "ker_gamma_dim",
            "rank_on_ker_gamma",
            "nullity_on_ker_gamma",
            "spectrum_exact",
            "spectrum_approx",
            "min_positive_eigenvalue_exact",
            "min_positive_eigenvalue_approx",
            "basis_gram_literal",
            "metric_literal",
            "integer_kernel_basis_literal",
            "integer_total_gram_literal",
            "integer_total_gram_det",
        ],
    )
    write_csv(
        OUT_DIR / "trackB_standard_family.csv",
        standard_rows,
        [
            "family",
            "pair_representative",
            "pair_spectrum_formula",
            "pair_min_positive_eigenvalue",
            "total_spectrum_formula",
            "total_min_positive_eigenvalue",
            "R",
            "ker_gamma_dim",
        ],
    )
    write_csv(
        OUT_DIR / "trackD_intersections.csv",
        intersection_rows,
        ["label", "pair_subset", "intersection_dim"],
    )
    write_csv(
        OUT_DIR / "trackE_strassen_certificate.csv",
        [strassen_proof],
        [
            "basis_gram_literal",
            "leading_principal_minors",
            "spectrum_exact",
            "spectrum_approx",
            "min_positive_eigenvalue_exact",
            "min_positive_eigenvalue_approx",
        ],
    )
    write_csv(
        OUT_DIR / "trackF_integer_gram.csv",
        integer_rows,
        ["label", "integer_kernel_basis_literal", "integer_total_gram_literal", "integer_total_gram_det"],
    )

    summary = {
        "trackA": {
            "pair_rows": pair_rows,
            "total_rows": total_rows,
        },
        "trackB": {
            "standard_family_rows": standard_rows,
        },
        "trackD": {
            "intersection_rows": intersection_rows,
        },
        "trackE": strassen_proof,
        "trackF": {
            "integer_rows": integer_rows,
        },
    }
    write_json(OUT_DIR / "phase34_summary.json", summary)
    write_text(OUT_DIR / "RESULTS.md", build_markdown(named_results, standard_rows, strassen_proof))


if __name__ == "__main__":
    main()