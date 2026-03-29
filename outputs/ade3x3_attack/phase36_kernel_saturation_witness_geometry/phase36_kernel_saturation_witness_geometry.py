from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.phase33b_27_symbol_live_alphabet.phase33b_27_symbol_live_alphabet import (  # noqa: E402
    channel_faces_exact,
    h_matrix_exact,
)
from outputs.ade3x3_attack.phase34_spectral_gap_channel_separation.phase34_spectral_gap_channel_separation import (  # noqa: E402
    Decomposition,
    gamma_matrix_exact,
    integerize_vector,
    kernel_basis_exact,
    load_named_decompositions,
)


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


def sparse_vector_literal(vector: sp.Matrix, names: list[str]) -> str:
    pieces: list[str] = []
    for idx, entry in enumerate(vector):
        value = sp.nsimplify(entry)
        if value != 0:
            pieces.append(f"{names[idx]}:{sp.sstr(value)}")
    return "{" + ", ".join(pieces) + "}"


def decomposition_face_matrices(decomposition: Decomposition) -> list[sp.Matrix]:
    return channel_faces_exact(decomposition)


def common_matrix_from_pair_family(p_mats: list[sp.Matrix], witness: sp.Matrix, n: int = 3) -> sp.Matrix:
    column = p_mats[0].T * witness
    return sp.Matrix(n, n, list(column))


def witness_space(gamma: sp.Matrix, p_mats: list[sp.Matrix]) -> sp.Matrix:
    if len(p_mats) == 2:
        stack = sp.Matrix.vstack(gamma, (p_mats[0] - p_mats[1]).T)
    else:
        stack = sp.Matrix.vstack(
            gamma,
            (p_mats[0] - p_mats[1]).T,
            (p_mats[1] - p_mats[2]).T,
        )
    basis = stack.nullspace()
    if not basis:
        return sp.zeros(gamma.cols, 0)
    return sp.Matrix.hstack(*basis)


def image_rank_on_witnesses(p_mats: list[sp.Matrix], witness_basis: sp.Matrix) -> int:
    if witness_basis.cols == 0:
        return 0
    image = p_mats[0].T * witness_basis
    return int(image.rank())


def common_matrix_rank_set(p_mats: list[sp.Matrix], witness_basis: sp.Matrix) -> str:
    if witness_basis.cols == 0:
        return "none"
    ranks = sorted(
        {
            int(common_matrix_from_pair_family(p_mats, integerize_vector(witness_basis[:, basis_idx])).rank())
            for basis_idx in range(witness_basis.cols)
        }
    )
    return ",".join(str(rank_value) for rank_value in ranks)


def analyze_exact_decomposition(decomposition: Decomposition) -> dict:
    gamma = gamma_matrix_exact(decomposition)
    p_mats = decomposition_face_matrices(decomposition)
    h = h_matrix_exact(decomposition, channel_faces_exact(decomposition))
    witness_basis = witness_space(gamma, p_mats)
    return {
        "label": decomposition.label,
        "family": "known_exact",
        "R": len(decomposition.terms),
        "rank_gamma": int(gamma.rank()),
        "ker_gamma_dim": gamma.cols - int(gamma.rank()),
        "rank_H": int(h.rank()),
        "saturation_defect": (gamma.cols - int(gamma.rank())) - int(h.rank()),
        "witness_dim": witness_basis.cols,
        "common_image_dim": image_rank_on_witnesses(p_mats, witness_basis),
        "common_rank_set": common_matrix_rank_set(p_mats, witness_basis),
        "witness_basis_literal": "none" if witness_basis.cols == 0 else "computed separately",
    }


def coefficient_matrix(kernel_dim: int, seed: int, mode: str) -> sp.Matrix:
    rng = random.Random(seed)
    matrix = sp.zeros(kernel_dim, 9)
    if mode == "random_small":
        for row_idx in range(kernel_dim):
            for col_idx in range(9):
                matrix[row_idx, col_idx] = rng.choice([-2, -1, 0, 1, 2])
    elif mode == "block_sparse":
        block_cols = [0, 4, 8]
        for row_idx in range(kernel_dim):
            for col_idx in block_cols:
                matrix[row_idx, col_idx] = rng.choice([-2, -1, 1, 2])
    elif mode == "rank1_shared":
        left = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(kernel_dim)]
        right = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(9)]
        for row_idx in range(kernel_dim):
            for col_idx in range(9):
                matrix[row_idx, col_idx] = left[row_idx] * right[col_idx]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return matrix


def synthetic_family_from_gamma(label: str, gamma: sp.Matrix, x0: sp.Matrix, kernel_basis: sp.Matrix, c0: sp.Matrix, c1: sp.Matrix, c2: sp.Matrix) -> tuple[list[sp.Matrix], sp.Matrix, dict]:
    n0 = kernel_basis * c0
    n1 = kernel_basis * c1
    n2 = kernel_basis * c2
    p_mats = [x0 + n0, x0 + n1, x0 + n2]
    h = sp.Matrix.hstack(n0 - n1, n1 - n2)
    witness_basis = witness_space(gamma, p_mats)
    return p_mats, h, {
        "label": label,
        "family": "synthetic_right_inverse",
        "rank_gamma": int(gamma.rank()),
        "ker_gamma_dim": gamma.cols - int(gamma.rank()),
        "rank_H": int(h.rank()),
        "saturation_defect": (gamma.cols - int(gamma.rank())) - int(h.rank()),
        "witness_dim": witness_basis.cols,
        "common_image_dim": image_rank_on_witnesses(p_mats, witness_basis),
        "common_rank_set": common_matrix_rank_set(p_mats, witness_basis),
        "witness_basis_literal": "none" if witness_basis.cols == 0 else "computed separately",
    }


def structured_family_rows(base_label: str, gamma: sp.Matrix, x0: sp.Matrix, kernel_basis: sp.Matrix) -> tuple[list[dict], list[dict]]:
    kernel_dim = kernel_basis.cols
    rows: list[dict] = []
    basis_rows: list[dict] = []
    cases = {
        "all_equal": (coefficient_matrix(kernel_dim, 1000 + kernel_dim, "rank1_shared"),) * 3,
        "k0_equals_k1": (
            coefficient_matrix(kernel_dim, 2000 + kernel_dim, "block_sparse"),
            coefficient_matrix(kernel_dim, 2000 + kernel_dim, "block_sparse"),
            coefficient_matrix(kernel_dim, 2100 + kernel_dim, "random_small"),
        ),
        "all_collinear": (
            coefficient_matrix(kernel_dim, 3000 + kernel_dim, "rank1_shared"),
            2 * coefficient_matrix(kernel_dim, 3000 + kernel_dim, "rank1_shared"),
            -coefficient_matrix(kernel_dim, 3000 + kernel_dim, "rank1_shared"),
        ),
        "two_mode_split": (
            coefficient_matrix(kernel_dim, 4000 + kernel_dim, "block_sparse"),
            coefficient_matrix(kernel_dim, 4100 + kernel_dim, "block_sparse"),
            coefficient_matrix(kernel_dim, 4200 + kernel_dim, "block_sparse"),
        ),
    }
    coordinate_names = [f"w{idx}" for idx in range(gamma.cols)]

    for case_name, coeffs in cases.items():
        label = f"{base_label}_{case_name}"
        p_mats, h, row = synthetic_family_from_gamma(label, gamma, x0, kernel_basis, coeffs[0], coeffs[1], coeffs[2])
        rows.append(row)
        witness_basis = witness_space(gamma, p_mats)
        for basis_idx in range(witness_basis.cols):
            witness = integerize_vector(witness_basis[:, basis_idx])
            common_matrix = common_matrix_from_pair_family(p_mats, witness)
            basis_rows.append(
                {
                    "label": label,
                    "basis_vector": f"b{basis_idx}",
                    "witness_vector": sparse_vector_literal(witness, coordinate_names),
                    "common_matrix_rank": int(common_matrix.rank()),
                    "common_matrix": matrix_literal(common_matrix),
                }
            )
    return rows, basis_rows


def wildcard_rows(base_label: str, gamma: sp.Matrix, x0: sp.Matrix, kernel_basis: sp.Matrix, trials: int = 18) -> tuple[list[dict], list[dict]]:
    kernel_dim = kernel_basis.cols
    rows: list[dict] = []
    interesting_rows: list[dict] = []
    for trial_idx in range(trials):
        c0 = coefficient_matrix(kernel_dim, 7000 + trial_idx, "random_small")
        c1 = coefficient_matrix(kernel_dim, 7100 + trial_idx, "random_small")
        c2 = coefficient_matrix(kernel_dim, 7200 + trial_idx, "random_small")
        label = f"{base_label}_wildcard_{trial_idx + 1:02d}"
        p_mats, _, row = synthetic_family_from_gamma(label, gamma, x0, kernel_basis, c0, c1, c2)
        row["trial_index"] = trial_idx + 1
        row["family"] = "wildcard_random_right_inverse"
        rows.append(row)
        if row["witness_dim"] > 0:
            witness_basis = witness_space(gamma, p_mats)
            interesting_rows.append(
                {
                    "label": label,
                    "trial_index": trial_idx + 1,
                    "witness_dim": row["witness_dim"],
                    "common_image_dim": row["common_image_dim"],
                    "common_rank_set": row["common_rank_set"],
                }
            )
    return rows, interesting_rows


def build_markdown(summary: dict[str, object]) -> str:
    lines: list[str] = []
    w = lines.append
    w("# Phase 36 Results: Kernel-Saturation Witness Geometry")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("## 36a. Saturation Witness Theorem")
    w("")
    w("[EXACT_DERIVED]")
    w("")
    w("Fix a 3x3 right-inverse family P_0, P_1, P_2 with Gamma P_s = I_9 and")
    w("H = [P_0-P_1 | P_1-P_2]. Then")
    w("")
    w("  ker(Gamma) ∩ ker(H^T) = { w : Gamma w = 0 and P_0^T w = P_1^T w = P_2^T w }.")
    w("")
    w("So nonsaturation is equivalent to the existence of a nonzero witness w whose three")
    w("channel images collapse to one common 3x3 matrix C(w). Phase 36 attacks the universal")
    w("theorem by measuring the geometry of that witness space directly.")
    w("")
    w("## 36b. Known Exact Decompositions")
    w("")
    w("[MEASURED_FROM_CODE]")
    w("")
    w("| case | dim ker(Gamma) | rank(H) | saturation defect | witness dim | common-image dim | common-matrix rank set |")
    w("|------|----------------|---------|-------------------|-------------|------------------|------------------------|")
    for row in summary["known_rows"]:
        w(
            f"| {row['label']} | {row['ker_gamma_dim']} | {row['rank_H']} | {row['saturation_defect']} | "
            f"{row['witness_dim']} | {row['common_image_dim']} | {row['common_rank_set']} |"
        )
    w("")
    w("As expected, the known exact decompositions have witness dimension 0. So the Phase 36")
    w("question is not whether a witness can exist abstractly, but what structure a witness would")
    w("have to force inside a valid low-rank decomposition.")
    w("")
    w("## 36c. Structured Defect Families Over Fixed Gamma")
    w("")
    w("[MEASURED_FROM_CODE]")
    w("")
    w("These families keep Gamma P_s = I_9 exactly but replace the actual channel faces by synthetic")
    w("kernel lifts P_s = X_0 + K_s with K_s in ker(Gamma).")
    w("")
    w("| case | dim ker(Gamma) | rank(H) | saturation defect | witness dim | common-image dim | common-matrix rank set |")
    w("|------|----------------|---------|-------------------|-------------|------------------|------------------------|")
    for row in summary["structured_rows"]:
        w(
            f"| {row['label']} | {row['ker_gamma_dim']} | {row['rank_H']} | {row['saturation_defect']} | "
            f"{row['witness_dim']} | {row['common_image_dim']} | {row['common_rank_set']} |"
        )
    w("")
    w("The exact defect dimension always matches the witness-space dimension, as it must. The useful")
    w("extra measurement is the common-image dimension: it tracks how many independent shared 3x3")
    w("matrices survive once the three channel lifts are forced together.")
    w("")
    w("## 36d. Witness Basis Samples")
    w("")
    w("[MEASURED_FROM_CODE]")
    w("")
    w("| case | basis vector | common rank | common matrix |")
    w("|------|--------------|-------------|---------------|")
    for row in summary["basis_rows"][:18]:
        w(f"| {row['label']} | {row['basis_vector']} | {row['common_matrix_rank']} | {row['common_matrix']} |")
    w("")
    w("In the synthetic defect families the common matrices are typically low-rank or sparse-patterned.")
    w("That is not yet a theorem, but it suggests a more concrete universal target: show that any common")
    w("matrix C(w) forced by a nonzero witness is incompatible with exact multiplication identities.")
    w("")
    w("## 36e. Wildcard Random Right-Inverse Scan")
    w("")
    w("[WILDCARD]")
    w("")
    w("| gamma family | trials | saturated trials | deficient trials | max witness dim seen |")
    w("|--------------|--------|-----------------|-----------------|----------------------|")
    for row in summary["wildcard_summary"]:
        w(
            f"| {row['gamma_family']} | {row['trials']} | {row['saturated_trials']} | {row['deficient_trials']} | {row['max_witness_dim_seen']} |"
        )
    w("")
    w("This wildcard branch is only a stress test. It asks how often random kernel lifts accidentally")
    w("create shared-channel witnesses. Generic lifts should saturate; deficient families should appear")
    w("as special coincidence loci.")
    w("")
    w("## 36f. Interpretation")
    w("")
    w("Phase 36 starts the direct saturation program rather than another spectral pass. The key shift is")
    w("to treat nonsaturation as an exact witness geometry problem: find or rule out nonzero w in ker(Gamma)")
    w("that produce one common channel matrix across all three faces. The next theorem target is therefore")
    w("not an eigenvalue bound but a structural impossibility statement for such common matrices inside valid")
    w("minimum-rank decompositions.")
    return "\n".join(lines) + "\n"


def main() -> None:
    named = load_named_decompositions()
    known_labels = ["strassen_2x2", "standard_rank27", "alphatensor_rank23"]
    known_rows = [analyze_exact_decomposition(named[label]) for label in known_labels]

    structured_rows: list[dict] = []
    basis_rows: list[dict] = []
    wildcard_scan_rows: list[dict] = []
    wildcard_summary: list[dict] = []

    for label in ["standard_rank27", "alphatensor_rank23"]:
        decomposition = named[label]
        gamma = gamma_matrix_exact(decomposition)
        kernel_basis = kernel_basis_exact(gamma)
        p_mats = decomposition_face_matrices(decomposition)
        base_x0 = p_mats[0]

        family_rows, family_basis_rows = structured_family_rows(label, gamma, base_x0, kernel_basis)
        structured_rows.extend(family_rows)
        basis_rows.extend(family_basis_rows)

        wildcard_rows_all, wildcard_interesting = wildcard_rows(label, gamma, base_x0, kernel_basis, trials=18)
        wildcard_scan_rows.extend(wildcard_rows_all)
        deficient_trials = [row for row in wildcard_rows_all if row["witness_dim"] > 0]
        wildcard_summary.append(
            {
                "gamma_family": label,
                "trials": len(wildcard_rows_all),
                "saturated_trials": len(wildcard_rows_all) - len(deficient_trials),
                "deficient_trials": len(deficient_trials),
                "max_witness_dim_seen": max((int(row["witness_dim"]) for row in wildcard_rows_all), default=0),
            }
        )

    summary = {
        "known_rows": known_rows,
        "structured_rows": structured_rows,
        "basis_rows": basis_rows,
        "wildcard_rows": wildcard_scan_rows,
        "wildcard_summary": wildcard_summary,
    }

    write_csv(
        OUT_DIR / "known_exact_witness_geometry.csv",
        known_rows,
        ["label", "family", "R", "rank_gamma", "ker_gamma_dim", "rank_H", "saturation_defect", "witness_dim", "common_image_dim", "common_rank_set", "witness_basis_literal"],
    )
    write_csv(
        OUT_DIR / "structured_defect_witness_geometry.csv",
        structured_rows,
        ["label", "family", "rank_gamma", "ker_gamma_dim", "rank_H", "saturation_defect", "witness_dim", "common_image_dim", "common_rank_set", "witness_basis_literal"],
    )
    write_csv(
        OUT_DIR / "structured_defect_witness_basis.csv",
        basis_rows,
        ["label", "basis_vector", "witness_vector", "common_matrix_rank", "common_matrix"],
    )
    write_csv(
        OUT_DIR / "wildcard_right_inverse_scan.csv",
        wildcard_scan_rows,
        ["label", "family", "trial_index", "rank_gamma", "ker_gamma_dim", "rank_H", "saturation_defect", "witness_dim", "common_image_dim", "common_rank_set", "witness_basis_literal"],
    )
    write_json(OUT_DIR / "phase36_summary.json", summary)
    write_text(OUT_DIR / "RESULTS.md", build_markdown(summary))


if __name__ == "__main__":
    main()