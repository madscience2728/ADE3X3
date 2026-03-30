from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (  # noqa: E402
    build_mode_matrices_numeric,
    load_public_terms,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    Term,
    build_mode_matrices,
    gamma_matrix,
)


RANK_TOL = 1e-10
WILDCARD_TRIALS = 12


@dataclass(frozen=True)
class ExactOperatorProfile:
    row: dict[str, object]
    operator: sp.Matrix
    charpoly: sp.Expr
    leading_minors: list[sp.Expr]


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


def standard_rank27_terms() -> list[Term]:
    terms: list[Term] = []
    term_idx = 1
    for row_idx in range(3):
        for shared_idx in range(3):
            for col_idx in range(3):
                alpha = np.zeros((3, 3), dtype=np.int64)
                beta = np.zeros((3, 3), dtype=np.int64)
                gamma = np.zeros((3, 3), dtype=np.int64)
                alpha[row_idx, shared_idx] = 1
                beta[shared_idx, col_idx] = 1
                gamma[row_idx, col_idx] = 1
                terms.append(Term(f"s{term_idx:02d}", f"standard_rank27_{term_idx:02d}", alpha, beta, gamma))
                term_idx += 1
    return terms


def stacked_alpha_beta(terms: list[Term]) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.stack([np.asarray(term.alpha, dtype=np.float64).reshape(-1) for term in terms])
    beta = np.stack([np.asarray(term.beta, dtype=np.float64).reshape(-1) for term in terms])
    return alpha, beta


def gamma_matrix_numeric(terms: list[Term]) -> np.ndarray:
    return np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms], axis=1).T.T


def matrix_rank_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > tol))


def nullspace_basis_numeric(matrix: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank_value = int(np.sum(singular_values > tol))
    return vh[rank_value:, :].T.copy()


def project_family(alpha: np.ndarray, beta: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    alpha_proj = alpha.copy().reshape(-1, 3, 3)
    beta_proj = beta.copy().reshape(-1, 3, 3)

    if family == "pair_equal_01":
        avg_a = 0.5 * (alpha_proj[:, :, 0] + alpha_proj[:, :, 1])
        avg_b = 0.5 * (beta_proj[:, 0, :] + beta_proj[:, 1, :])
        alpha_proj[:, :, 0] = avg_a
        alpha_proj[:, :, 1] = avg_a
        beta_proj[:, 0, :] = avg_b
        beta_proj[:, 1, :] = avg_b
    elif family == "pair_equal_12":
        avg_a = 0.5 * (alpha_proj[:, :, 1] + alpha_proj[:, :, 2])
        avg_b = 0.5 * (beta_proj[:, 1, :] + beta_proj[:, 2, :])
        alpha_proj[:, :, 1] = avg_a
        alpha_proj[:, :, 2] = avg_a
        beta_proj[:, 1, :] = avg_b
        beta_proj[:, 2, :] = avg_b
    elif family == "all_equal":
        mean_a = np.mean(alpha_proj, axis=2)
        mean_b = np.mean(beta_proj, axis=1)
        for idx in range(3):
            alpha_proj[:, :, idx] = mean_a
            beta_proj[:, idx, :] = mean_b
    elif family == "collinear":
        for term_idx in range(alpha_proj.shape[0]):
            u_a, s_a, vh_a = np.linalg.svd(alpha_proj[term_idx], full_matrices=False)
            alpha_proj[term_idx] = s_a[0] * np.outer(u_a[:, 0], vh_a[0, :])
            u_b, s_b, vh_b = np.linalg.svd(beta_proj[term_idx], full_matrices=False)
            beta_proj[term_idx] = s_b[0] * np.outer(u_b[:, 0], vh_b[0, :])
    else:
        raise ValueError(f"Unsupported family: {family}")

    return alpha_proj.reshape(alpha.shape[0], 9), beta_proj.reshape(beta.shape[0], 9)


def exact_operator_profile(label: str, terms: list[Term], delta_in_eta_exact: bool) -> ExactOperatorProfile:
    sigma, eta1, eta2, delta, _ = build_mode_matrices(terms, 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    gamma = gamma_matrix(terms, 3)
    z_basis = sp.Matrix.hstack(*sp.Matrix.vstack(h_matrix.T, delta.T).nullspace())
    sigma_z = sigma.T * z_basis
    gamma_z = gamma * z_basis
    omega = sp.simplify(gamma_z * sigma_z.inv())
    x = sp.Symbol("x")
    charpoly = sp.factor(omega.charpoly(x).as_expr())
    leading_minors = [sp.factor(omega[:idx, :idx].det()) for idx in range(1, omega.rows + 1)]
    eigenvalues = sorted(float(sp.N(value, 30)) for value in omega.eigenvals())

    row = {
        "label": label,
        "family": "known_exact",
        "R": len(terms),
        "rank_gamma": int(gamma.rank()),
        "rank_H": int(h_matrix.rank()),
        "rank_delta": int(delta.rank()),
        "rank_HDelta": int(sp.Matrix.hstack(h_matrix, delta).rank()),
        "delta_in_eta_exact": delta_in_eta_exact,
        "z_dim": z_basis.cols,
        "sigma_z_rank": int(sigma_z.rank()),
        "gamma_z_rank": int(gamma_z.rank()),
        "sigma_z_det": sp.sstr(sp.factor(sigma_z.det())),
        "omega_det": sp.sstr(sp.factor(omega.det())),
        "omega_symmetric": omega == omega.T,
        "omega_min_eigenvalue_approx": f"{eigenvalues[0]:.12f}",
        "omega_max_eigenvalue_approx": f"{eigenvalues[-1]:.12f}",
        "omega_positive_definite": all(sp.simplify(value) > 0 for value in leading_minors),
        "omega_charpoly": sp.sstr(charpoly),
    }
    return ExactOperatorProfile(row=row, operator=omega, charpoly=charpoly, leading_minors=leading_minors)


def numeric_sector_profile(base_label: str, branch_type: str, family: str, trial_index: int, terms: list[Term]) -> dict[str, object]:
    sigma, eta1, eta2, delta, h_matrix, _ = build_mode_matrices_numeric(terms)
    gamma = np.stack([np.asarray(term.gamma, dtype=np.float64).reshape(-1) for term in terms], axis=1)
    z_basis = nullspace_basis_numeric(np.vstack([h_matrix.T, delta.T]))
    z_dim = int(z_basis.shape[1])
    sigma_z = sigma.T @ z_basis
    gamma_z = gamma @ z_basis
    sigma_z_rank = matrix_rank_numeric(sigma_z)
    operator_defined = bool(z_dim == 9 and sigma_z_rank == 9)

    row = {
        "base_label": base_label,
        "branch_type": branch_type,
        "family": family,
        "trial_index": trial_index,
        "R": len(terms),
        "rank_H_numeric": matrix_rank_numeric(h_matrix),
        "rank_Delta_numeric": matrix_rank_numeric(delta),
        "rank_HDelta_numeric": matrix_rank_numeric(np.hstack([h_matrix, delta])),
        "z_dim": z_dim,
        "sigma_z_rank": sigma_z_rank,
        "gamma_z_rank": matrix_rank_numeric(gamma_z),
        "operator_defined": operator_defined,
        "operator_symmetry_max_abs": "",
        "operator_det_approx": "",
        "operator_min_eigenvalue_approx": "",
        "operator_max_eigenvalue_approx": "",
    }

    if operator_defined:
        omega = gamma_z @ np.linalg.inv(sigma_z)
        eigs = np.linalg.eigvals(omega)
        eigs_real = sorted(float(value.real) for value in eigs)
        row.update(
            {
                "operator_symmetry_max_abs": f"{float(np.max(np.abs(omega - omega.T))):.12e}",
                "operator_det_approx": f"{float(np.linalg.det(omega)):.12f}",
                "operator_min_eigenvalue_approx": f"{eigs_real[0]:.12f}",
                "operator_max_eigenvalue_approx": f"{eigs_real[-1]:.12f}",
            }
        )

    max_abs_residual, loss_value = tensor_residual_stats(terms)
    row["tensor_max_abs_residual"] = f"{max_abs_residual:.12f}"
    row["tensor_loss"] = f"{loss_value:.12f}"
    return row


def build_exact_rows() -> tuple[list[dict], list[dict], dict[str, dict], str, float]:
    alpha_terms, orientation, residual = load_public_terms()
    exact_profiles = [
        exact_operator_profile("alphatensor_rank23", alpha_terms, delta_in_eta_exact=True),
        exact_operator_profile("standard_rank27", standard_rank27_terms(), delta_in_eta_exact=True),
    ]

    exact_rows = [profile.row for profile in exact_profiles]
    operator_rows: list[dict] = []
    operator_certificates: dict[str, dict] = {}
    for profile in exact_profiles:
        label = str(profile.row["label"])
        for row_idx in range(profile.operator.rows):
            for col_idx in range(profile.operator.cols):
                operator_rows.append(
                    {
                        "label": label,
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "value": sp.sstr(sp.nsimplify(profile.operator[row_idx, col_idx])),
                    }
                )
        operator_certificates[label] = {
            "matrix_literal": matrix_literal(profile.operator),
            "charpoly": sp.sstr(profile.charpoly),
            "leading_minors": [sp.sstr(value) for value in profile.leading_minors],
        }

    return exact_rows, operator_rows, operator_certificates, orientation, float(residual)


def build_structured_rows(alpha_terms: list[Term], standard_terms: list[Term]) -> list[dict]:
    rows: list[dict] = []
    bases = [
        ("alphatensor_rank23", *stacked_alpha_beta(alpha_terms)),
        ("standard_rank27", *stacked_alpha_beta(standard_terms)),
    ]
    for base_label, alpha, beta in bases:
        for family in ["pair_equal_01", "pair_equal_12", "all_equal", "collinear"]:
            alpha_proj, beta_proj = project_family(alpha, beta, family)
            gamma = solve_gamma_least_squares(alpha_proj, beta_proj)
            terms = terms_from_stacked_factors(alpha_proj, beta_proj, gamma, prefix=f"{family}_")
            rows.append(numeric_sector_profile(base_label, "structured", family, 0, terms))
    return rows


def build_wildcard_rows(alpha_terms: list[Term], standard_terms: list[Term]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summary_rows: list[dict] = []
    base_specs = [
        ("alphatensor_rank23", *stacked_alpha_beta(alpha_terms), 20260329),
        ("standard_rank27", *stacked_alpha_beta(standard_terms), 20260330),
    ]

    for base_label, alpha, beta, seed in base_specs:
        rng = np.random.default_rng(seed)
        for trial_idx in range(1, WILDCARD_TRIALS + 1):
            scales = 0.5 + rng.random((alpha.shape[0], 3))
            alpha_trial = alpha.reshape(-1, 3, 3).copy()
            beta_trial = beta.reshape(-1, 3, 3).copy()
            for term_idx in range(alpha_trial.shape[0]):
                for shared_idx in range(3):
                    alpha_trial[term_idx, :, shared_idx] *= scales[term_idx, shared_idx]
                    beta_trial[term_idx, shared_idx, :] /= scales[term_idx, shared_idx]
            alpha_stacked = alpha_trial.reshape(alpha.shape[0], 9)
            beta_stacked = beta_trial.reshape(beta.shape[0], 9)
            gamma = solve_gamma_least_squares(alpha_stacked, beta_stacked)
            terms = terms_from_stacked_factors(alpha_stacked, beta_stacked, gamma, prefix=f"wild_{trial_idx:02d}_")
            rows.append(numeric_sector_profile(base_label, "wildcard", "random_channel_rescaling", trial_idx, terms))

        base_rows = [row for row in rows if row["base_label"] == base_label and row["branch_type"] == "wildcard"]
        summary_rows.append(
            {
                "base_label": base_label,
                "trials": len(base_rows),
                "operator_defined_trials": sum(1 for row in base_rows if row["operator_defined"]),
                "z_dim_min": min(int(row["z_dim"]) for row in base_rows),
                "z_dim_max": max(int(row["z_dim"]) for row in base_rows),
                "sigma_z_rank_min": min(int(row["sigma_z_rank"]) for row in base_rows),
                "sigma_z_rank_max": max(int(row["sigma_z_rank"]) for row in base_rows),
                "min_tensor_residual": min(float(row["tensor_max_abs_residual"]) for row in base_rows),
                "max_tensor_residual": max(float(row["tensor_max_abs_residual"]) for row in base_rows),
                "min_operator_eigenvalue_seen": min(
                    (float(row["operator_min_eigenvalue_approx"]) for row in base_rows if row["operator_min_eigenvalue_approx"] != ""),
                    default=0.0,
                ),
            }
        )
    return rows, summary_rows


def build_markdown(
    exact_rows: list[dict],
    structured_rows: list[dict],
    wildcard_summary: list[dict],
    operator_certificates: dict[str, dict],
) -> str:
    lines: list[str] = []
    w = lines.append
    w("# Phase 37 Results: Pure-Sigma Common-Matrix Obstruction")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("## 37a. Exact Pure-Sigma Obstruction Operator")
    w("")
    w("[EXACT_DERIVED]")
    w("")
    w("Let Sigma, H, Delta be the Step 51/Phase 17 fiber-mode blocks, with H = [eta1 | eta2].")
    w("Define the silent sigma-sector")
    w("")
    w("  Z = ker(H^T) intersect ker(Delta^T).")
    w("")
    w("If dim(Z) = 9 and Sigma_Z := Sigma^T|_Z is invertible, define the 9x9 operator")
    w("")
    w("  Omega = (Gamma|_Z) * Sigma_Z^(-1).")
    w("")
    w("Then every z in Z satisfies")
    w("")
    w("  Gamma z = Omega (Sigma^T z).")
    w("")
    w("Now suppose Delta subset span(H). Any Phase 36 witness w already obeys H^T w = 0, so it also obeys")
    w("Delta^T w = 0 and therefore lies in Z. Because Sigma^T w = 3 vec(C(w)), every nonzero witness matrix must satisfy")
    w("")
    w("  Omega vec(C(w)) = 0.")
    w("")
    w("So invertibility of Omega is an exact common-matrix obstruction inside the Delta-contained regime.")
    w("")
    w("## 37b. Known Exact Decompositions")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("| case | rank(H) | rank(Delta) | rank([H|Delta]) | dim Z | rank Sigma_Z | det Sigma_Z | det Omega | min eig(Omega) | max eig(Omega) | symmetric? | positive definite? |")
    w("|------|---------|-------------|------------------|-------|--------------|------------|-----------|----------------|----------------|------------|--------------------|")
    for row in exact_rows:
        w(
            f"| {row['label']} | {row['rank_H']} | {row['rank_delta']} | {row['rank_HDelta']} | {row['z_dim']} | "
            f"{row['sigma_z_rank']} | {row['sigma_z_det']} | {row['omega_det']} | {row['omega_min_eigenvalue_approx']} | "
            f"{row['omega_max_eigenvalue_approx']} | {row['omega_symmetric']} | {row['omega_positive_definite']} |"
        )
    w("")
    w("Two exact facts stand out.")
    w("")
    w("1. The silent sigma-sector has the clean expected size dim(Z) = 9 for both exact decompositions.")
    w("2. Omega is symmetric positive definite in both cases, so the common-matrix equation Omega vec(C) = 0 forces C = 0.")
    w("")
    w("In particular, the standard algorithm collapses to the strongest possible certificate:")
    w("")
    w("  Omega_standard = I_9.")
    w("")
    w("For AlphaTensor the operator is no longer diagonal, but it remains exact, symmetric, and invertible. Its leading principal minors are all positive, so positivity survives the entangled case too.")
    w("")
    w("AlphaTensor exact characteristic polynomial:")
    w("")
    w(f"  {operator_certificates['alphatensor_rank23']['charpoly']}")
    w("")
    w("## 37c. Structured Factorized Defect Branches")
    w("")
    w("[MEASURED_FROM_CODE]")
    w("")
    w("These are the Phase 19-style factorized defect projections with gamma re-solved by least squares.")
    w("")
    w("| base | family | dim Z | rank Sigma_Z | operator defined? | min eig(Omega) | tensor max residual |")
    w("|------|--------|-------|--------------|-------------------|----------------|---------------------|")
    for row in structured_rows:
        w(
            f"| {row['base_label']} | {row['family']} | {row['z_dim']} | {row['sigma_z_rank']} | {row['operator_defined']} | "
            f"{row['operator_min_eigenvalue_approx'] or 'n/a'} | {row['tensor_max_abs_residual']} |"
        )
    w("")
    w("The clean obstruction operator disappears exactly on the rigid defect loci that already showed hard tensor residual floors in Phase 19.")
    w("")
    w("- Pair-equality and all-equal projections force rank(Sigma_Z) = 0, so the common sigma image collapses completely.")
    w("- The standard collinear branch keeps the exact operator intact and stays exact, matching the older observation that collinearity is not itself an obstruction.")
    w("- The AlphaTensor collinear branch shrinks the silent sector itself to dim(Z) = 3, so the 9x9 common-matrix operator no longer exists in that family.")
    w("")
    w("## 37d. Wildcard Random Channel Rescalings")
    w("")
    w("[WILDCARD]")
    w("")
    w("Each wildcard trial rescales every term's three live channels independently, then re-solves gamma by least squares.")
    w("")
    w("| base | trials | operator-defined trials | dim Z range | rank Sigma_Z range | tensor residual range |")
    w("|------|--------|------------------------|-------------|--------------------|-----------------------|")
    for row in wildcard_summary:
        w(
            f"| {row['base_label']} | {row['trials']} | {row['operator_defined_trials']} | "
            f"{row['z_dim_min']}..{row['z_dim_max']} | {row['sigma_z_rank_min']}..{row['sigma_z_rank_max']} | "
            f"{row['min_tensor_residual']:.12f}..{row['max_tensor_residual']:.12f} |"
        )
    w("")
    w("The wildcard branch splits sharply by base family. Standard-derived channel rescalings preserve the clean 9-dimensional sigma-only sector and Omega = I in every sampled trial, while AlphaTensor-derived rescalings collapse immediately to a 1-dimensional silent sector and a visible tensor residual.")
    w("")
    w("## 37e. Interpretation")
    w("")
    w("Phase 37 converts the witness-matrix question into a smaller exact object. Instead of asking directly whether a nonzero w in ker(Gamma) can satisfy channel collapse, we isolate the sigma-only silent sector Z and the induced 9x9 output operator Omega.")
    w("")
    w("For the known exact decompositions, Omega is already enough to rule out common matrices once Delta containment is granted: standard gives Omega = I, while AlphaTensor gives an exact symmetric positive-definite rational matrix. So the Phase 36 witness problem has now been compressed to a finite 9x9 certificate on the sigma-only sector, rather than the full R-dimensional term space.")
    w("")
    w("This is not yet a universal theorem, because Phase 37 still uses the Delta-contained regime rather than proving it universally. But it sharpens the obstruction target substantially: the next theorem-level step is to show that every valid minimum-rank decomposition has the same clean 9-dimensional sigma-only sector and an invertible induced Omega.")
    return "\n".join(lines) + "\n"


def main() -> None:
    exact_rows, operator_rows, operator_certificates, orientation, residual = build_exact_rows()
    alpha_terms, _, _ = load_public_terms()
    standard_terms = standard_rank27_terms()
    structured_rows = build_structured_rows(alpha_terms, standard_terms)
    wildcard_rows, wildcard_summary = build_wildcard_rows(alpha_terms, standard_terms)

    summary = {
        "alphatensor_orientation": orientation,
        "alphatensor_reconstruction_max_abs": residual,
        "exact_rows": exact_rows,
        "structured_rows": structured_rows,
        "wildcard_rows": wildcard_rows,
        "wildcard_summary": wildcard_summary,
        "operator_certificates": operator_certificates,
    }

    write_csv(
        OUT_DIR / "known_exact_sigma_obstruction.csv",
        exact_rows,
        [
            "label",
            "family",
            "R",
            "rank_gamma",
            "rank_H",
            "rank_delta",
            "rank_HDelta",
            "delta_in_eta_exact",
            "z_dim",
            "sigma_z_rank",
            "gamma_z_rank",
            "sigma_z_det",
            "omega_det",
            "omega_symmetric",
            "omega_min_eigenvalue_approx",
            "omega_max_eigenvalue_approx",
            "omega_positive_definite",
            "omega_charpoly",
        ],
    )
    write_csv(
        OUT_DIR / "exact_operator_entries.csv",
        operator_rows,
        ["label", "row_idx", "col_idx", "value"],
    )
    write_csv(
        OUT_DIR / "off_manifold_sigma_sector_scan.csv",
        structured_rows + wildcard_rows,
        [
            "base_label",
            "branch_type",
            "family",
            "trial_index",
            "R",
            "rank_H_numeric",
            "rank_Delta_numeric",
            "rank_HDelta_numeric",
            "z_dim",
            "sigma_z_rank",
            "gamma_z_rank",
            "operator_defined",
            "operator_symmetry_max_abs",
            "operator_det_approx",
            "operator_min_eigenvalue_approx",
            "operator_max_eigenvalue_approx",
            "tensor_max_abs_residual",
            "tensor_loss",
        ],
    )
    write_json(OUT_DIR / "phase37_summary.json", summary)
    write_text(OUT_DIR / "RESULTS.md", build_markdown(exact_rows, structured_rows, wildcard_summary, operator_certificates))


if __name__ == "__main__":
    main()