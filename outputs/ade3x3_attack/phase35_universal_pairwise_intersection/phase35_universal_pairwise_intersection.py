from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
    difference_matrix_exact,
    gamma_matrix_exact,
    integerize_vector,
    kernel_basis_exact,
    load_named_decompositions,
    min_positive_eigenvalue,
    pair_labels,
    restricted_form_data,
    spectrum_approx_string,
    spectrum_to_string,
    standard_terms_n,
    Term,
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


def sparse_vector_literal(vector: sp.Matrix, term_ids: list[str]) -> str:
    pieces: list[str] = []
    for idx, entry in enumerate(vector):
        value = sp.nsimplify(entry)
        if value != 0:
            pieces.append(f"{term_ids[idx]}:{sp.sstr(value)}")
    return "{" + ", ".join(pieces) + "}"


def common_channel_matrix(decomposition: Decomposition, weight: sp.Matrix, channel: int) -> sp.Matrix:
    total = sp.zeros(decomposition.n, decomposition.n)
    for term_idx, term in enumerate(decomposition.terms):
        coefficient = sp.nsimplify(weight[term_idx, 0])
        if coefficient == 0:
            continue
        alpha_col = sp.Matrix([[int(term.alpha[row_idx, channel])] for row_idx in range(decomposition.n)])
        beta_row = sp.Matrix([[int(term.beta[channel, col_idx]) for col_idx in range(decomposition.n)]])
        total += coefficient * (alpha_col * beta_row)
    return sp.Matrix(total)


def pair_nullspace_basis(decomposition: Decomposition, pair: tuple[int, int]) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    gamma = gamma_matrix_exact(decomposition)
    kernel_basis = kernel_basis_exact(gamma)
    difference = difference_matrix_exact(decomposition, pair[0], pair[1])
    if kernel_basis.cols == 0:
        return kernel_basis, difference, sp.zeros(gamma.cols, 0)
    restricted = difference * kernel_basis
    coeff_basis = restricted.nullspace()
    if not coeff_basis:
        return kernel_basis, difference, sp.zeros(gamma.cols, 0)
    weight_basis = sp.Matrix.hstack(*[kernel_basis * vector for vector in coeff_basis])
    return kernel_basis, difference, weight_basis


def pair_kernel_analysis(decomposition: Decomposition) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    basis_rows: list[dict] = []
    term_ids = [term.term_id for term in decomposition.terms]
    pairs = pair_labels(decomposition.n)
    for pair in pairs:
        kernel_basis, difference, weight_basis = pair_nullspace_basis(decomposition, pair)
        complementary_pairs = [other for other in pairs if other != pair and len(set(pair) | set(other)) == decomposition.n]
        distinct_common_ranks: set[int] = set()
        all_nonzero_complements = True
        for basis_idx in range(weight_basis.cols):
            primitive = integerize_vector(weight_basis[:, basis_idx])
            shared = common_channel_matrix(decomposition, primitive, pair[0])
            distinct_common_ranks.add(int(shared.rank()))
            complement_images: list[str] = []
            complement_zero_flags: list[bool] = []
            for other in complementary_pairs:
                image = difference_matrix_exact(decomposition, other[0], other[1]) * primitive
                is_zero = all(entry == 0 for entry in image)
                complement_zero_flags.append(is_zero)
                all_nonzero_complements &= not is_zero
                complement_images.append(f"{other[0]}{other[1]}:{matrix_literal(image.reshape(image.rows, 1))}")
            basis_rows.append({
                "label": decomposition.label,
                "pair": f"{pair[0]}{pair[1]}",
                "basis_vector": f"b{basis_idx}",
                "weight_vector": sparse_vector_literal(primitive, term_ids),
                "shared_channel_rank": int(shared.rank()),
                "shared_channel_matrix": matrix_literal(shared),
                "all_complementary_images_nonzero": all(not flag for flag in complement_zero_flags) if complement_zero_flags else True,
                "complementary_images": " | ".join(complement_images) if complement_images else "n/a",
            })
        summary_rows.append({
            "label": decomposition.label,
            "pair": f"{pair[0]}{pair[1]}",
            "ker_gamma_dim": kernel_basis.cols,
            "pair_nullity": weight_basis.cols,
            "all_basis_vectors_hit_complementary_pairs": all_nonzero_complements if weight_basis.cols else True,
            "distinct_common_matrix_ranks": ",".join(str(rank) for rank in sorted(distinct_common_ranks)) if distinct_common_ranks else "none",
        })
    return summary_rows, basis_rows


def track_b_analysis(decomposition: Decomposition) -> tuple[dict, list[dict]]:
    gamma = gamma_matrix_exact(decomposition)
    faces = channel_faces_exact(decomposition)
    h = h_matrix_exact(decomposition, faces)
    differences = {pair: difference_matrix_exact(decomposition, pair[0], pair[1]) for pair in pair_labels(decomposition.n)}
    kernel_dim = gamma.cols - gamma.rank()
    rank_h = int(h.rank())
    eta_nullity = h.cols - rank_h

    if decomposition.n == 2:
        independent_pairs = [(0, 1)]
    else:
        independent_pairs = [(0, 1), (1, 2)]

    independent_stack = sp.Matrix.vstack(*[differences[pair] for pair in independent_pairs])
    rowspace_match = sp.Matrix.vstack(independent_stack, h.T).rank() == independent_stack.rank() == h.T.rank()

    combo_rows: list[dict] = []
    for pair_a_idx in range(len(pair_labels(decomposition.n))):
        for pair_b_idx in range(pair_a_idx + 1, len(pair_labels(decomposition.n))):
            pair_a = pair_labels(decomposition.n)[pair_a_idx]
            pair_b = pair_labels(decomposition.n)[pair_b_idx]
            if len(set(pair_a) | set(pair_b)) != decomposition.n:
                continue
            stack = sp.Matrix.vstack(gamma, differences[pair_a], differences[pair_b])
            combo_rows.append({
                "label": decomposition.label,
                "pair_a": f"{pair_a[0]}{pair_a[1]}",
                "pair_b": f"{pair_b[0]}{pair_b[1]}",
                "intersection_dim": gamma.cols - stack.rank(),
                "formula_dim_ker_minus_rank_H": kernel_dim - rank_h,
            })

    theorem_row = {
        "label": decomposition.label,
        "R": len(decomposition.terms),
        "n": decomposition.n,
        "ker_gamma_dim": kernel_dim,
        "rank_H": rank_h,
        "H_cols": h.cols,
        "eta_nullity": eta_nullity,
        "rowspace_Ht_equals_independent_pair_stack": rowspace_match,
        "pairwise_intersection_dim": combo_rows[0]["intersection_dim"] if combo_rows else kernel_dim - rank_h,
        "formula_dim_ker_minus_rank_H": kernel_dim - rank_h,
        "formula_matches": all(row["intersection_dim"] == kernel_dim - rank_h for row in combo_rows) if combo_rows else True,
        "pairwise_intersection_equals_eta_nullity": (combo_rows[0]["intersection_dim"] if combo_rows else kernel_dim - rank_h) == eta_nullity,
    }
    return theorem_row, combo_rows


def track_c_analysis(decomposition: Decomposition) -> tuple[list[dict], dict]:
    gamma = gamma_matrix_exact(decomposition)
    kernel_basis = kernel_basis_exact(gamma)
    pair_rows: list[dict] = []
    difference_literals: dict[str, str] = {}
    for pair in pair_labels(decomposition.n):
        difference = difference_matrix_exact(decomposition, pair[0], pair[1])
        form = restricted_form_data(kernel_basis, difference)
        gap = min_positive_eigenvalue(form["spectrum"])
        pair_rows.append({
            "label": decomposition.label,
            "pair": f"{pair[0]}{pair[1]}",
            "full_rank": int(difference.rank()),
            "projected_rank": int((difference * kernel_basis).rank()),
            "pair_nullity_on_ker_gamma": gamma.cols - sp.Matrix.vstack(gamma, difference).rank(),
            "spectrum_exact": spectrum_to_string(form["spectrum"]),
            "spectrum_approx": spectrum_approx_string(form["spectrum"]),
            "min_positive_eigenvalue": f"{float(sp.N(gap, 20)):.12f}",
        })
        difference_literals[f"{pair[0]}{pair[1]}"] = matrix_literal(difference)
    return pair_rows, difference_literals


def track_d_analysis(named: dict[str, Decomposition], track_c_rows: list[dict]) -> dict[str, object]:
    standard_rows = [row for row in track_c_rows if row["label"] == "standard_rank27"]
    alpha_rows = [row for row in track_c_rows if row["label"] == "alphatensor_rank23"]
    strassen_rows = [row for row in track_c_rows if row["label"] == "strassen_2x2"]

    entanglement_rows = []
    for label, rows, n in [
        ("standard_rank27", standard_rows, 3),
        ("alphatensor_rank23", alpha_rows, 3),
        ("strassen_2x2", strassen_rows, 2),
    ]:
        pair_nullities = [int(row["pair_nullity_on_ker_gamma"]) for row in rows]
        max_fiber_aligned_total = (len(rows) * (n ** 2))
        entanglement_rows.append({
            "label": label,
            "pair_nullities": ",".join(str(value) for value in pair_nullities),
            "sum_pair_nullities": sum(pair_nullities),
            "max_fiber_aligned_total": max_fiber_aligned_total,
            "entanglement_deficit": max_fiber_aligned_total - sum(pair_nullities),
            "nullity_range": max(pair_nullities) - min(pair_nullities) if pair_nullities else 0,
        })

    standard3 = named["standard_rank27"]
    gamma = gamma_matrix_exact(standard3)
    kernel_basis = kernel_basis_exact(gamma)
    difference = difference_matrix_exact(standard3, 0, 1)
    form = restricted_form_data(kernel_basis, difference)
    qchan_form = restricted_form_data(kernel_basis, sp.Matrix.vstack(
        difference_matrix_exact(standard3, 0, 1),
        difference_matrix_exact(standard3, 0, 2),
        difference_matrix_exact(standard3, 1, 2),
    ))

    proof_template = {
        "standard_3x3_pair_spectrum": spectrum_to_string(form["spectrum"]),
        "standard_3x3_total_spectrum": spectrum_to_string(qchan_form["spectrum"]),
        "fiber_model_statement": "ker(Gamma) splits into 9 output fibers, each a 2-dimensional zero-sum hyperplane in R^3; on each fiber Q_st has spectrum {0,2} and Q_chan = 3 Id.",
        "entanglement_rows": entanglement_rows,
    }
    return proof_template


def random_term(rng: np.random.Generator, n: int, term_id: str) -> Term:
    alpha = rng.integers(-1, 2, size=(n, n), endpoint=False)
    beta = rng.integers(-1, 2, size=(n, n), endpoint=False)
    gamma = rng.integers(-1, 2, size=(n, n), endpoint=False)
    while not np.any(alpha):
        alpha = rng.integers(-1, 2, size=(n, n), endpoint=False)
    while not np.any(beta):
        beta = rng.integers(-1, 2, size=(n, n), endpoint=False)
    while not np.any(gamma):
        gamma = rng.integers(-1, 2, size=(n, n), endpoint=False)
    return Term(term_id=term_id, source_label=term_id, alpha=alpha.astype(np.int64), beta=beta.astype(np.int64), gamma=gamma.astype(np.int64))


def random_decomposition_with_full_gamma(n: int, rank: int, seed: int) -> Decomposition:
    rng = np.random.default_rng(seed)
    while True:
        terms = [random_term(rng, n, f"rnd_{rank}_{seed}_{idx}") for idx in range(rank)]
        decomposition = Decomposition(label=f"random_{n}x{n}_R{rank}_seed{seed}", n=n, terms=terms)
        if gamma_matrix_exact(decomposition).rank() == n * n:
            return decomposition


def track_e_analysis(trials_per_rank: int = 6) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    trial_rows: list[dict] = []
    for rank in (20, 21, 22):
        gaps: list[float] = []
        nullities: list[int] = []
        for trial_idx in range(trials_per_rank):
            seed = 3500 + 97 * rank + trial_idx
            decomposition = random_decomposition_with_full_gamma(3, rank, seed)
            gamma = gamma_matrix_exact(decomposition)
            kernel_basis = kernel_basis_exact(gamma)
            total_difference = sp.Matrix.vstack(
                difference_matrix_exact(decomposition, 0, 1),
                difference_matrix_exact(decomposition, 0, 2),
                difference_matrix_exact(decomposition, 1, 2),
            )
            total_form = restricted_form_data(kernel_basis, total_difference)
            gap_expr = min_positive_eigenvalue(total_form["spectrum"])
            gap_value = float(sp.N(gap_expr, 20)) if gap_expr != 0 else 0.0
            total_nullity = gamma.cols - sp.Matrix.vstack(gamma, total_difference).rank()
            gaps.append(gap_value)
            nullities.append(total_nullity)
            trial_rows.append({
                "label": decomposition.label,
                "R": rank,
                "seed": seed,
                "ker_gamma_dim": kernel_basis.cols,
                "total_gap": gap_value,
                "total_nullity": total_nullity,
            })
        summary_rows.append({
            "case": f"random_3x3_R{rank}",
            "trials": trials_per_rank,
            "min_gap": min(gaps),
            "median_gap": statistics.median(gaps),
            "max_gap": max(gaps),
            "min_total_nullity": min(nullities),
            "max_total_nullity": max(nullities),
        })
    return summary_rows, trial_rows


def track_f_analysis(decomposition: Decomposition) -> list[dict]:
    gamma = gamma_matrix_exact(decomposition)
    kernel_basis = kernel_basis_exact(gamma)
    pair_map = {pair: difference_matrix_exact(decomposition, pair[0], pair[1]) * kernel_basis for pair in pair_labels(decomposition.n)}
    rows: list[dict] = []
    pairs = list(pair_map)
    for pair_a_idx in range(len(pairs)):
        for pair_b_idx in range(pair_a_idx + 1, len(pairs)):
            pair_a = pairs[pair_a_idx]
            pair_b = pairs[pair_b_idx]
            stacked = sp.Matrix.vstack(pair_map[pair_a], pair_map[pair_b])
            rank_a = int(pair_map[pair_a].rank())
            rank_b = int(pair_map[pair_b].rank())
            joint_rank = int(stacked.rank())
            rows.append({
                "label": decomposition.label,
                "pair_a": f"{pair_a[0]}{pair_a[1]}",
                "pair_b": f"{pair_b[0]}{pair_b[1]}",
                "rank_a": rank_a,
                "rank_b": rank_b,
                "joint_rank": joint_rank,
                "rowspace_overlap": rank_a + rank_b - joint_rank,
            })
    return rows


def build_markdown(summary: dict[str, object]) -> str:
    lines: list[str] = []
    w = lines.append
    w("# Phase 35 Results: Universal Pairwise Intersection")
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w("")
    w("## 35a. Track A: W_s = W_t Variety Structure")
    w("")
    w("[MEASURED_FROM_CODE]")
    w("")
    w("For each pair (s,t), the single-pair nullspace inside ker(Gamma) is the exact solution space")
    w("of (P_s - P_t) w = 0 with Gamma w = 0. Phase 35 computes primitive rational bases for these")
    w("spaces and pushes each basis vector through the complementary pair maps.")
    w("")
    w("| decomposition | pair | pair nullity | all basis vectors hit every complementary pair? | shared W_s rank set |")
    w("|---------------|------|--------------|-----------------------------------------------|-------------------|")
    for row in summary["trackA"]["summary_rows"]:
        w(
            f"| {row['label']} | {row['pair']} | {row['pair_nullity']} | "
            f"{row['all_basis_vectors_hit_complementary_pairs']} | {row['distinct_common_matrix_ranks']} |"
        )
    w("")
    w("The shared property on the known decompositions is negative but useful: every computed basis")
    w("vector of a nontrivial single-pair nullspace has nonzero image under every complementary pair")
    w("map that completes the full three-channel constraint. So no measured single-pair basis direction")
    w("extends to a two-pair defect.")
    w("")

    w("## 35b. Track B: Three-Channel Constraint and Equivalence")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("For n = 3, the canon definition H = [K_0-K_1 | K_1-K_2] together with K_s = P_s - X_0 gives")
    w("")
    w("  H = [P_0-P_1 | P_1-P_2]    and therefore    H^T = [D_01 ; D_12].")
    w("")
    w("Since D_02 = D_01 + D_12, the common kernel of any two pair maps covering all channels is exactly")
    w("ker(H^T). Intersecting with ker(Gamma), the exact theorem is")
    w("")
    w("  dim(ker(Gamma) ∩ ker D_st ∩ ker D_s't') = dim(ker(Gamma)) - rank(H)")
    w("")
    w("for any two pairs whose union is {0,1,2}. This is the left-nullity of H inside ker(Gamma).")
    w("")
    w("| decomposition | dim ker(Gamma) | rank(H) | pairwise intersection dim | dim ker(Gamma)-rank(H) | eta_nullity | equals eta_nullity? |")
    w("|---------------|----------------|---------|---------------------------|-----------------------|-------------|---------------------|")
    for row in summary["trackB"]["theorem_rows"]:
        w(
            f"| {row['label']} | {row['ker_gamma_dim']} | {row['rank_H']} | {row['pairwise_intersection_dim']} | "
            f"{row['formula_dim_ker_minus_rank_H']} | {row['eta_nullity']} | {row['pairwise_intersection_equals_eta_nullity']} |"
        )
    w("")
    w("This corrects the tempting but false stronger claim from the prompt: pairwise intersection is")
    w("not eta_nullity in general. AlphaTensor is the counterexample already in hand: pairwise")
    w("intersection dimension 0, but eta_nullity = 4.")
    w("")

    w("## 35c. Track C: Khatri-Rao Rank Analysis")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("Each pair map D_st = (P_s - P_t)^T has entries alpha_k[r,s] beta_k[s,u] - alpha_k[r,t] beta_k[t,u],")
    w("so each output row is a difference of two rank-1 outer-product coefficients. The exact ranks are:")
    w("")
    w("| decomposition | pair | full rank of D_st | rank on ker(Gamma) | pair nullity on ker(Gamma) | gap spectrum |")
    w("|---------------|------|-------------------|--------------------|---------------------------|-------------|")
    for row in summary["trackC"]["rows"]:
        w(
            f"| {row['label']} | {row['pair']} | {row['full_rank']} | {row['projected_rank']} | "
            f"{row['pair_nullity_on_ker_gamma']} | {row['spectrum_exact']} |"
        )
    w("")
    w("For Strassen, D_01 already has rank 3 on ker(Gamma), which equals dim ker(Gamma), so there is no")
    w("single-pair nullity at all. That is why the Strassen case collapses immediately to positivity.")
    w("")

    w("## 35d. Track D: Standard Proof Template and Entanglement")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("The exact standard-algorithm template remains clean: ker(Gamma) splits into 9 output fibers, each")
    w("a 2-dimensional zero-sum plane. On each fiber, every pair form has spectrum {0,2}, and Q_chan acts")
    w("as 3 Id. What changes off the standard algorithm is not the theorem statement, but the loss of fiber")
    w("separation.")
    w("")
    w(f"Standard 3x3 pair spectrum: {summary['trackD']['proof_template']['standard_3x3_pair_spectrum']}")
    w(f"Standard 3x3 total spectrum: {summary['trackD']['proof_template']['standard_3x3_total_spectrum']}")
    w("")
    w("| decomposition | pair nullities | sum pair nullities | fiber-aligned maximum | entanglement deficit | nullity range |")
    w("|---------------|----------------|--------------------|----------------------|---------------------|---------------|")
    for row in summary["trackD"]["proof_template"]["entanglement_rows"]:
        w(
            f"| {row['label']} | {row['pair_nullities']} | {row['sum_pair_nullities']} | "
            f"{row['max_fiber_aligned_total']} | {row['entanglement_deficit']} | {row['nullity_range']} |"
        )
    w("")
    w("Measured only: stronger channel entanglement correlates with smaller single-pair nullities")
    w("(standard 27, AlphaTensor 19, Strassen 0 in the summed metric above), but the intersection theorem")
    w("survives in both the maximally fiber-aligned and entangled cases.")
    w("")

    w("## 35e. Track E: Tensor Rank Obstruction (WILDCARD)")
    w("")
    w("[WILDCARD]")
    w("")
    w("These are not multiplication decompositions. They are random ternary rank-1 collections with gamma")
    w("full rank 9 so the restricted gap can be sampled away from the variety.")
    w("")
    w("| random case | trials | min gap | median gap | max gap | min total nullity | max total nullity |")
    w("|-------------|--------|---------|------------|---------|-------------------|-------------------|")
    for row in summary["trackE"]["summary_rows"]:
        w(
            f"| {row['case']} | {row['trials']} | {row['min_gap']:.12f} | {row['median_gap']:.12f} | "
            f"{row['max_gap']:.12f} | {row['min_total_nullity']} | {row['max_total_nullity']} |"
        )
    w("")
    w("This wildcard scan should not be read as a lower-bound theorem. It only says the raw spectral gap")
    w("outside the multiplication variety does not exhibit an obvious sharp threshold in these random samples.")
    w("")

    w("## 35f. Track F: η_nullity Structural Constraints")
    w("")
    w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
    w("")
    w("The row-space overlap heuristic from the prompt is only partly right. For two projected pair maps")
    w("A = D_st|ker(Gamma) and B = D_s't'|ker(Gamma), the quantity rank(A)+rank(B)-rank([A;B]) measures")
    w("row-space overlap, not kernel intersection. Exact data:")
    w("")
    w("| decomposition | pair A | pair B | rank A | rank B | joint rank | row-space overlap |")
    w("|---------------|--------|--------|--------|--------|------------|-------------------|")
    for row in summary["trackF"]["rows"]:
        w(
            f"| {row['label']} | {row['pair_a']} | {row['pair_b']} | {row['rank_a']} | {row['rank_b']} | "
            f"{row['joint_rank']} | {row['rowspace_overlap']} |"
        )
    w("")
    w("So zero pairwise kernel intersection does not force zero row-space overlap. AlphaTensor already has")
    w("nonzero overlaps for some pair combinations, even though every two-pair kernel intersection is 0.")
    w("")

    w("## 35g. Interpretation")
    w("")
    w("Phase 35 does produce an exact theorem, but it is sharper than the prompt’s provisional chain and")
    w("slightly different. The universal pairwise-intersection statement for n = 3 is exactly equivalent to")
    w("H saturating ker(Gamma):")
    w("")
    w("  ker(Gamma) ∩ ker D_st ∩ ker D_s't' = {0}  for all covering pair choices")
    w("  if and only if rank(H) = dim ker(Gamma) = R - 9.")
    w("")
    w("That is Theorem A in intersection form. It is not the same as eta_nullity = 0, and AlphaTensor proves")
    w("the distinction. So the real remaining universal target is still to show that H fills ker(Gamma)")
    w("for every valid minimum-rank decomposition.")
    w("")
    w("What Phase 35 adds is structural control around that target. Single-pair nullspaces can be large, but")
    w("on all known exact decompositions every basis direction in a single-pair defect is kicked out by the")
    w("complementary pair constraints. The standard proof shows the fully fiber-aligned mechanism; AlphaTensor")
    w("shows the entangled mechanism can still end in the same zero-intersection theorem even when the pair")
    w("nullities become uneven and the row spaces overlap.")
    return "\n".join(lines) + "\n"


def main() -> None:
    named = load_named_decompositions()

    track_a_summary: list[dict] = []
    track_a_basis: list[dict] = []
    track_b_theorem_rows: list[dict] = []
    track_b_combo_rows: list[dict] = []
    track_c_rows: list[dict] = []
    track_c_literals: dict[str, dict[str, str]] = {}
    track_f_rows: list[dict] = []

    for label in ["strassen_2x2", "standard_rank27", "alphatensor_rank23"]:
        decomposition = named[label]
        a_summary, a_basis = pair_kernel_analysis(decomposition)
        track_a_summary.extend(a_summary)
        track_a_basis.extend(a_basis)

        theorem_row, combo_rows = track_b_analysis(decomposition)
        track_b_theorem_rows.append(theorem_row)
        track_b_combo_rows.extend(combo_rows)

        c_rows, c_literals = track_c_analysis(decomposition)
        track_c_rows.extend(c_rows)
        track_c_literals[label] = c_literals

        track_f_rows.extend(track_f_analysis(decomposition))

    track_d = track_d_analysis(named, track_c_rows)
    track_e_summary, track_e_trials = track_e_analysis()

    summary = {
        "trackA": {
            "summary_rows": track_a_summary,
            "basis_rows": track_a_basis,
        },
        "trackB": {
            "theorem_rows": track_b_theorem_rows,
            "combo_rows": track_b_combo_rows,
            "theorem_statement": "For n=3, H^T = [D_01; D_12], so the common kernel of any two pair maps covering all channels equals ker(H^T). Intersected with ker(Gamma), its dimension is dim(ker(Gamma)) - rank(H).",
        },
        "trackC": {
            "rows": track_c_rows,
            "difference_literals": track_c_literals,
        },
        "trackD": {
            "proof_template": track_d,
        },
        "trackE": {
            "summary_rows": track_e_summary,
            "trial_rows": track_e_trials,
        },
        "trackF": {
            "rows": track_f_rows,
        },
    }

    write_csv(
        OUT_DIR / "trackA_pair_variety_summary.csv",
        track_a_summary,
        [
            "label",
            "pair",
            "ker_gamma_dim",
            "pair_nullity",
            "all_basis_vectors_hit_complementary_pairs",
            "distinct_common_matrix_ranks",
        ],
    )
    write_csv(
        OUT_DIR / "trackA_pair_variety_basis.csv",
        track_a_basis,
        [
            "label",
            "pair",
            "basis_vector",
            "weight_vector",
            "shared_channel_rank",
            "shared_channel_matrix",
            "all_complementary_images_nonzero",
            "complementary_images",
        ],
    )
    write_csv(
        OUT_DIR / "trackB_equivalence.csv",
        track_b_theorem_rows,
        [
            "label",
            "R",
            "n",
            "ker_gamma_dim",
            "rank_H",
            "H_cols",
            "eta_nullity",
            "rowspace_Ht_equals_independent_pair_stack",
            "pairwise_intersection_dim",
            "formula_dim_ker_minus_rank_H",
            "formula_matches",
            "pairwise_intersection_equals_eta_nullity",
        ],
    )
    write_csv(
        OUT_DIR / "trackB_pairwise_combo_rows.csv",
        track_b_combo_rows,
        ["label", "pair_a", "pair_b", "intersection_dim", "formula_dim_ker_minus_rank_H"],
    )
    write_csv(
        OUT_DIR / "trackC_pair_ranks.csv",
        track_c_rows,
        [
            "label",
            "pair",
            "full_rank",
            "projected_rank",
            "pair_nullity_on_ker_gamma",
            "spectrum_exact",
            "spectrum_approx",
            "min_positive_eigenvalue",
        ],
    )
    write_csv(
        OUT_DIR / "trackD_entanglement.csv",
        track_d["entanglement_rows"],
        [
            "label",
            "pair_nullities",
            "sum_pair_nullities",
            "max_fiber_aligned_total",
            "entanglement_deficit",
            "nullity_range",
        ],
    )
    write_csv(
        OUT_DIR / "trackE_wildcard_random_gap_summary.csv",
        track_e_summary,
        ["case", "trials", "min_gap", "median_gap", "max_gap", "min_total_nullity", "max_total_nullity"],
    )
    write_csv(
        OUT_DIR / "trackE_wildcard_random_gap_trials.csv",
        track_e_trials,
        ["label", "R", "seed", "ker_gamma_dim", "total_gap", "total_nullity"],
    )
    write_csv(
        OUT_DIR / "trackF_pair_rank_overlaps.csv",
        track_f_rows,
        ["label", "pair_a", "pair_b", "rank_a", "rank_b", "joint_rank", "rowspace_overlap"],
    )

    write_json(OUT_DIR / "phase35_summary.json", summary)
    write_text(OUT_DIR / "RESULTS.md", build_markdown(summary))


if __name__ == "__main__":
    main()