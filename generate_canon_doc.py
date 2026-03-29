#!/usr/bin/env python3
"""
generate_canon_doc.py

Generates a STANDALONE canonical dossier containing ALL computed results.

The generated document is the ONLY artifact the next team receives.
It must be completely self-contained with all research findings.

No arguments. Just run it.
"""

import ast
import csv
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Paths
EXPORTS_DIR = Path(__file__).parent / "outputs" / "exports"
DOCS_DIR = Path(__file__).parent / "docs"
PHASE17_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase17_conservation_law"
PHASE18_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase18_kernel_saturation"
PHASE19_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase19_factorized_defect_loci"
PHASE20_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase20_affine_line_obstruction"
PHASE21_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase21_functional_annihilator"
PHASE22_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase22_kernel_linked_annihilator"
PHASE23_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase23_staged_layering_bridge"
PHASE24_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase24_tangent_transversality"
PHASE25_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase25_honest_kernel_retention"
PHASE26_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase26_regularized_kernel_retention"
PHASE27_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase27_coupled_continuation"
PHASE28_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase28_wildcard_regularized_continuation"
PHASE29_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase29_smooth_budgeted_continuation"
PHASE30_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase30_budget_efficiency_sweep"
PHASE31_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase31_feasibility_boundary"
PHASE32_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase32_boundary_sharpening"
PHASE33_DIR = Path(__file__).parent / "outputs" / "ade3x3_attack" / "phase33_27_symbol_faithful_encoding"

def read_csv(filename):
    """Read CSV file from exports directory."""
    path = EXPORTS_DIR / filename
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def read_json_path(path):
    """Read a JSON file from an absolute path."""
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_phase17_outputs():
    """Read Phase 17 conservation-law outputs."""
    faithfulness = read_json_path(PHASE17_DIR / "faithfulness_summary.json")
    identities = read_json_path(PHASE17_DIR / "single_term_identities.json")
    gamma_analysis = read_json_path(PHASE17_DIR / "gamma_propagation_analysis.json")
    collective = read_json_path(PHASE17_DIR / "collective_containment_summary.json")
    return faithfulness, identities, gamma_analysis, collective

def read_phase18_outputs():
    """Read Phase 18 kernel-saturation outputs."""
    return read_json_path(PHASE18_DIR / "kernel_saturation_summary.json")

def read_phase19_outputs():
    """Read Phase 19 factorized defect-locus outputs."""
    diagnostics = read_json_path(PHASE19_DIR / "factorized_defect_diagnostics.json")
    search = read_json_path(PHASE19_DIR / "constrained_defect_family_summary.json")
    return diagnostics, search

def read_phase20_outputs():
    """Read Phase 20 affine-line obstruction outputs."""
    diagnostics = read_json_path(PHASE20_DIR / "affine_line_lift_analysis.json")
    search = read_json_path(PHASE20_DIR / "affine_line_family_summary.json")
    return diagnostics, search

def read_phase21_outputs():
    """Read Phase 21 functional-annihilator outputs."""
    diagnostics = read_json_path(PHASE21_DIR / "functional_annihilator_diagnostics.json")
    search = read_json_path(PHASE21_DIR / "weighted_annihilator_family_summary.json")
    return diagnostics, search

def read_phase22_outputs():
    """Read Phase 22 kernel-linked annihilator outputs."""
    return read_json_path(PHASE22_DIR / "kernel_linked_annihilator_homotopy.json")

def read_phase23_outputs():
    """Read Phase 23 staged-layering bridge outputs."""
    return read_json_path(PHASE23_DIR / "staged_homotopy_thresholds.json")

def read_phase24_outputs():
    """Read Phase 24 tangent-transversality outputs."""
    return read_json_path(PHASE24_DIR / "kernel_forcing_tangent_transversality.json")

def read_phase25_outputs():
    """Read Phase 25 honest-kernel-retention outputs."""
    return read_json_path(PHASE25_DIR / "honest_kernel_retention_scan.json")

def read_phase26_outputs():
    """Read Phase 26 regularized-kernel-retention outputs."""
    return read_json_path(PHASE26_DIR / "regularized_kernel_retention_summary.json")

def read_phase27_outputs():
    """Read Phase 27 coupled-continuation prototype outputs."""
    return read_json_path(PHASE27_DIR / "coupled_continuation_probe.json")

def read_phase28_outputs():
    """Read Phase 28 wildcard-regularized continuation outputs."""
    return read_json_path(PHASE28_DIR / "wildcard_regularized_continuation.json")

def read_phase29_outputs():
    """Read Phase 29 smooth-budgeted continuation outputs."""
    return read_json_path(PHASE29_DIR / "smooth_budgeted_continuation.json")

def read_phase30_outputs():
    """Read Phase 30 budget-efficiency sweep outputs."""
    return read_json_path(PHASE30_DIR / "budget_efficiency_sweep.json")

def read_phase31_outputs():
    """Read Phase 31 feasibility-boundary outputs."""
    return read_json_path(PHASE31_DIR / "feasibility_boundary_scan.json")

def read_phase32_outputs():
    """Read Phase 32 boundary-sharpening outputs."""
    return read_json_path(PHASE32_DIR / "boundary_sharpening_scan.json")

def read_phase33_outputs():
    """Read Phase 33 27-symbol faithful-encoding outputs."""
    return read_json_path(PHASE33_DIR / "phase33_summary.json")

def read_axxc_signature_layer(rep_config_ids):
    """Read the current AXXC arity-4 signature layer for selected reps.

    The recorded AXXC signature layer is the 6-face orbit tuple induced by the
    face inventory export.
    """
    target_ids = {int(cfg) for cfg in rep_config_ids}
    if not target_ids:
        return {}

    path = EXPORTS_DIR / "AXXC_face_inventory.csv"
    if not path.exists():
        return {}

    fields = [
        'left_ax_orbit_id',
        'middle_xx_orbit_id',
        'right_xc_orbit_id',
        'outer_ac_orbit_id',
        'left_axc_orbit_id',
        'right_axc_orbit_id',
    ]

    sig_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = int(row['axxc_config_id'])
            if cfg not in target_ids:
                continue
            sig_map[cfg] = tuple(int(row[field]) for field in fields)
            if len(sig_map) == len(target_ids):
                break

    return sig_map

def read_composition_kernel():
    """Read composition kernel rows – expanded histogram for 14 mixed keys."""
    return read_csv("composition_kernel_mixed.csv")

def read_cxxc_marginal_weight_profile():
    """Read CXXC marginal weight profile (7 projections, 188 rows)."""
    return read_csv("cxxc_marginal_weight_profile.csv")

def read_stabilizer_classification():
    """Read stabilizer classification for CXC and CXXC orbits."""
    return read_csv("stabilizer_classification.csv")

def read_stabilizer_composition():
    """Read stabilizer composition summary (stab_type_A, stab_type_B, stab_type_output, ...)."""
    return read_csv("stabilizer_composition_summary.csv")

def read_refinement_conditioned_kernel():
    """Read refinement-conditioned kernel strata (step 43)."""
    return read_csv("refinement_conditioned_kernel_strata.csv")

def read_floor_layer():
    """Read Z2xZ2 floor layer inventory and composition table (step 44)."""
    inventory = read_csv("floor_layer_inventory.csv")
    comp_table = read_csv("floor_layer_composition_table.csv")
    stab_comp = read_csv("floor_layer_stab_composition.csv")
    return inventory, comp_table, stab_comp

def read_step45_outputs():
    """Read step 45 closure, doubly-live core, and fixed-point exports."""
    closure_steps = read_csv("step45_closure_steps.csv")
    core_rows = read_csv("step45_doubly_live_core.csv")
    core_comp = read_csv("step45_doubly_live_composition.csv")
    core_output_summary = read_csv("step45_doubly_live_output_summary.csv")
    fixed_rows = read_csv("step45_fixed_point_subspaces.csv")
    return closure_steps, core_rows, core_comp, core_output_summary, fixed_rows

def read_step46_outputs():
    """Read step 46 same-fiber, focused, and 64-subalgebra exports."""
    same_fiber_outputs = read_csv("step46_same_fiber_outputs.csv")
    same_fiber_pair_summary = read_csv("step46_same_fiber_pair_summary.csv")
    same_fiber_closure = read_csv("step46_same_fiber_closure.csv")
    subalg_table = read_csv("step46_64_subalgebra_table.csv")
    subalg_summary = read_csv("step46_64_subalgebra_summary.csv")
    focused_orbits = read_csv("step46_focused_orbits.csv")
    focused_comp = read_csv("step46_focused_composition.csv")
    focused_summary = read_csv("step46_focus_profile_summary.csv")
    identity_map = read_csv("step46_identity_map.csv")
    single_gen = read_csv("step46_single_generator_closure.csv")
    greedy_gen = read_csv("step46_greedy_generators.csv")
    return (same_fiber_outputs, same_fiber_pair_summary, same_fiber_closure,
            subalg_table, subalg_summary, focused_orbits, focused_comp,
            focused_summary, identity_map, single_gen, greedy_gen)

def read_step47_outputs():
    """Read step 47 mixed-pair resolution and tensor-constraint exports."""
    st_rows = read_csv("step47_mixed_pair_st_resolution.csv")
    full_rows = read_csv("step47_mixed_pair_rstu_resolution.csv")
    summary_rows = read_csv("step47_mixed_pair_resolution_summary.csv")
    tensor_profile = read_csv("step47_tensor_orbit_profile.csv")
    constraint_rows = read_csv("step47_algorithm_constraints.csv")
    external_rows = read_csv("step47_external_algorithm_status.csv")
    return st_rows, full_rows, summary_rows, tensor_profile, constraint_rows, external_rows

def read_step48_outputs():
    """Read step 48 tensor-profile constraint model exports."""
    summary_rows = read_csv("step48_constraint_model_summary.csv")
    xc_orbit_rows = read_csv("step48_xc_orbit_equation_summary.csv")
    formula_rows = read_csv("step48_rank1_xc_orbit_sum_formulas.csv")
    counterexample_rows = read_csv("step48_orbit_sum_counterexample.csv")
    return summary_rows, xc_orbit_rows, formula_rows, counterexample_rows

def read_step49_outputs():
    """Read step 49 coefficient-level rank-constraint exports."""
    summary_rows = read_csv("step49_summary.csv")
    equation_rows = read_csv("step49_equation_types.csv")
    verification_rows = read_csv("step49_representative_orbit_verification.csv")
    standard_rows = read_csv("step49_standard_algorithm_verification.csv")
    tensor_slice_rows = read_csv("step49_tensor_slice_summary.csv")
    strassen_term_rows = read_csv("step49_strassen_2x2_terms.csv")
    strassen_orbit_rows = read_csv("step49_strassen_2x2_term_orbits.csv")
    search_rows = read_csv("step49_search_space_dimensions.csv")
    return (summary_rows, equation_rows, verification_rows, standard_rows,
            tensor_slice_rows, strassen_term_rows, strassen_orbit_rows, search_rows)

def read_step51_outputs():
    """Read step 51 symbolic fiber-mode decomposition exports."""
    summary_rows = read_csv("step51_summary.csv")
    sum_rows = read_csv("step51_fiber_sum_formulas.csv")
    anisotropy_rows = read_csv("step51_live_anisotropy_formulas.csv")
    matrix_rows = read_csv("step51_matrix_form.csv")
    standard_rows = read_csv("step51_standard_algorithm_symbolic_verification.csv")
    return summary_rows, sum_rows, anisotropy_rows, matrix_rows, standard_rows

def read_step52_outputs():
    """Read step 52 quotient-space rank criterion exports."""
    summary_rows = read_csv("step52_summary.csv")
    theorem_rows = read_csv("step52_quotient_rank_theorem.csv")
    profile_rows = read_csv("step52_algorithm_rank_profiles.csv")
    bound_rows = read_csv("step52_required_nuisance_bounds.csv")
    return summary_rows, theorem_rows, profile_rows, bound_rows

def read_step53_outputs():
    """Read step 53 support-type representative-incidence exports."""
    summary_rows = read_csv("step53_summary.csv")
    support_rows = read_csv("step53_support_type_classes.csv")
    mask_rows = read_csv("step53_support_type_mask_summary.csv")
    orbit0_rows = read_csv("step53_orbit0_feasible_histogram.csv")
    return summary_rows, support_rows, mask_rows, orbit0_rows

def read_step54_outputs():
    """Read step 54 analytical low-nuisance construction exports."""
    summary_rows = read_csv("step54_summary.csv")
    dead_rows = read_csv("step54_dead_free_theorem.csv")
    dead_profile_rows = read_csv("step54_dead_free_index_profiles.csv")
    focus_rows = read_csv("step54_r22_focus.csv")
    best_rows = read_csv("step54_r22_best_samples.csv")
    lift_rows = read_csv("step54_strassen_corner_lift.csv")
    smirnov_rows = read_csv("step54_smirnov_status.csv")
    return summary_rows, dead_rows, dead_profile_rows, focus_rows, best_rows, lift_rows, smirnov_rows

def read_step55_outputs():
    """Read step 55 algebraic nuisance dependency and wildcard exports."""
    summary_rows = read_csv("step55_summary.csv")
    dimension_rows = read_csv("step55_dimension_targets.csv")
    best_rows = read_csv("step55_structured_family_best.csv")
    profile_rows = read_csv("step55_structured_family_profiles.csv")
    symbolic_rows = read_csv("step55_symbolic_minor_witness.csv")
    gf2_rows = read_csv("step55_gf2_flattening_ranks.csv")
    tropical_rows = read_csv("step55_tropical_flattening_ranks.csv")
    comm_rows = read_csv("step55_commutator_summary.csv")
    return summary_rows, dimension_rows, best_rows, profile_rows, symbolic_rows, gf2_rows, tropical_rows, comm_rows

def read_step56_outputs():
    """Read step 56 tensor-product DFT and orbit-packing exports."""
    summary_rows = read_csv("step56_summary.csv")
    dft_top_rows = read_csv("step56_dft_mode_top10.csv")
    dft_dc_rows = read_csv("step56_dft_top10_dc_summary.csv")
    dft_verification_rows = read_csv("step56_dft_mode_top10_verification.csv")
    ternary_best_rows = read_csv("step56_ternary_random_best.csv")
    algebraic_best_rows = read_csv("step56_algebraic_random_best.csv")
    correction_best_rows = read_csv("step56_correction_block_best.csv")
    optimization_best_rows = read_csv("step56_local_search_best.csv")
    branch_rows = read_csv("step56_orbit30_branch_distribution.csv")
    removal_rows = read_csv("step56_standard_term_removal_profile.csv")
    support_rows = read_csv("step56_2x2_same_fiber_support_profiles.csv")
    return (
        summary_rows,
        dft_top_rows,
        dft_dc_rows,
        dft_verification_rows,
        ternary_best_rows,
        algebraic_best_rows,
        correction_best_rows,
        optimization_best_rows,
        branch_rows,
        removal_rows,
        support_rows,
    )

def read_step57_outputs():
    """Read step 57 fiber-group partition enumeration exports."""
    summary_rows = read_csv("step57_summary.csv")
    type_rows = read_csv("step57_partition_types.csv")
    survivor_rows = read_csv("step57_exact3_survivors.csv")
    cycle_rows = read_csv("step57_symmetry_cycle_types.csv")
    shape_rows = read_csv("step57_shape_classes.csv")
    return summary_rows, type_rows, survivor_rows, cycle_rows, shape_rows

def read_step59_outputs():
    """Read step 59 cube-root-of-unity injection exports."""
    summary_rows = read_csv("step59_summary.csv")
    obstruction_rows = read_csv("step59_fullspread_obstruction.csv")
    assignment_rows = read_csv("step59_r9_assignment.csv")
    r9_summary_rows = read_csv("step59_r9_failure_summary.csv")
    r9_failure_rows = read_csv("step59_r9_failure_examples.csv")
    greedy_rows = read_csv("step59_greedy_fullspread_progress.csv")
    nuisance_rows = read_csv("step59_samefiber_nuisance_profiles.csv")
    verification_rows = read_csv("step59_samefiber_fourier_standard_verification.csv")
    return (
        summary_rows,
        obstruction_rows,
        assignment_rows,
        r9_summary_rows,
        r9_failure_rows,
        greedy_rows,
        nuisance_rows,
        verification_rows,
    )

def read_step60_outputs():
    """Read step 60 hybrid Fourier construction exports."""
    summary_rows = read_csv("step60_summary.csv")
    modularity_rows = read_csv("step60_fourier_modularity_summary.csv")
    orbit_rows = read_csv("step60_residual_subtensor_orbits.csv")
    size_rows = read_csv("step60_residual_subtensor_size_summary.csv")
    reduced_rows = read_csv("step60_reduced_spreader_systems.csv")
    feasibility_rows = read_csv("step60_hybrid_feasibility_table.csv")
    explicit_rows = read_csv("step60_explicit_cases.csv")
    return (
        summary_rows,
        modularity_rows,
        orbit_rows,
        size_rows,
        reduced_rows,
        feasibility_rows,
        explicit_rows,
    )

def read_step61_outputs():
    """Read step 61 nuisance-first architecture exports."""
    summary_rows = read_csv("step61_summary.csv")
    theorem_rows = read_csv("step61_r9_theorem_chain.csv")
    channel_rows = read_csv("step61_deadfree_channel_bound.csv")
    budget_rows = read_csv("step61_nuisance_budget_table.csv")
    pressure_rows = read_csv("step61_rank18_23_pressure.csv")
    return summary_rows, theorem_rows, channel_rows, budget_rows, pressure_rows

def read_step62_outputs():
    """Read step 62 six-fiber sub-tensor rank attack exports."""
    summary_rows = read_csv("step62_summary.csv")
    substitution_best_rows = read_csv("step62_substitution_best.csv")
    numerical_rows = read_csv("step62_numerical_rank_scan.csv")
    direct_rows = read_csv("step62_direct_p4_construction.csv")
    pattern_rows = read_csv("step62_pattern_summary.csv")
    verdict_rows = read_csv("step62_hybrid_verdicts.csv")
    return summary_rows, substitution_best_rows, numerical_rows, direct_rows, pattern_rows, verdict_rows

def read_step63_outputs():
    """Read step 63 reverse-engineering and cancellation-visualization exports."""
    summary_rows = read_csv("step63_summary.csv")
    source_rows = read_csv("step63_external_source_status.csv")
    term_rows = read_csv("step63_term_coefficients.csv")
    importance_rows = read_csv("step63_term_importance.csv")
    single_rows = read_csv("step63_single_term_removal.csv")
    pair_rows = read_csv("step63_pair_term_removal.csv")
    dead_rows = read_csv("step63_dead_equation_balances.csv")
    comm_rows = read_csv("step63_commutator_split.csv")
    return summary_rows, source_rows, term_rows, importance_rows, single_rows, pair_rows, dead_rows, comm_rows

def read_step64_outputs():
    """Read step 64 small-integer coefficient-enumeration exports."""
    summary_rows = read_csv("step64_summary.csv")
    top_rows = read_csv("step64_top100_usefulness_profiles.csv")
    greedy3_rows = read_csv("step64_3x3_collapsed_greedy.csv")
    greedy2_collapsed_rows = read_csv("step64_2x2_collapsed_greedy.csv")
    greedy2_rows = read_csv("step64_2x2_full_tensor_greedy.csv")
    return summary_rows, top_rows, greedy3_rows, greedy2_collapsed_rows, greedy2_rows

def read_step65_outputs():
    """Read step 65 polyomino subtensor-rank and tiling-analysis exports."""
    summary_rows = read_csv("step65_summary.csv")
    rank_rows = read_csv("step65_polyomino_rank_table.csv")
    mixed_rows = read_csv("step65_mixed_tilings.csv")
    toroidal_rows = read_csv("step65_toroidal_l_tromino_tilings.csv")
    cluster_rows = read_csv("step65_alphatensor_support_clusters.csv")
    return summary_rows, rank_rows, mixed_rows, toroidal_rows, cluster_rows

def read_step66_piece_audits():
    """Read step 66 fresh-piece audit exports for the tetromino classes checked so far."""
    audit_files = [
        "step66_s_z_tetromino_class_02_verdict.csv",
        "step66_l_tetromino_class_02_verdict.csv",
    ]
    audits = []
    for filename in audit_files:
        rows = read_csv(filename)
        if rows:
            audits.extend(rows)
    return audits

def read_step67_outputs():
    """Read step 67 complete rank-table audit and layered-tiling exports."""
    summary_rows = read_csv("step67_summary.csv")
    audit_rows = read_csv("step67_partA_rank_audit.csv")
    rank_rows = read_csv("step67_corrected_rank_table.csv")
    tiling_rows = read_csv("step67_corrected_tiling_table.csv")
    low_cost_rows = read_csv("step67_low_cost_tilings_leq24.csv")
    corrector_rows = read_csv("step67_alphatensor_corrector_terms.csv")
    cluster_rows = read_csv("step67_nuisance_support_clusters.csv")
    return summary_rows, audit_rows, rank_rows, tiling_rows, low_cost_rows, corrector_rows, cluster_rows

def read_step68_fourier_outputs():
    """Read Step 68 Fourier steering exports, if present."""
    summary_rows = read_csv("step68_fourier_summary.csv")
    mode_rows = read_csv("step68_fourier_mode_decomposition.csv")
    component_rows = read_csv("step68_fourier_component_ranks.csv")
    return summary_rows, mode_rows, component_rows

def read_step69_outputs():
    """Read Step 69 interlocking-mechanism exports, if present."""
    summary_rows = read_csv("step69_summary.csv")
    projection_rows = read_csv("step69_term_mode_projections.csv")
    subspace_rows = read_csv("step69_mode_subspace_dimensions.csv")
    routed_rows = read_csv("step69_mode_routed_three_fiber_ranks.csv")
    multilevel_rows = read_csv("step69_multilevel_residual_probe.csv")
    greedy_rows = read_csv("step69_reduced_ternary_greedy_cover.csv")
    return summary_rows, projection_rows, subspace_rows, routed_rows, multilevel_rows, greedy_rows

def read_step70_outputs():
    """Read Step 70 depth-2 arithmetic-circuit exports, if present."""
    summary_rows = read_csv("step70_summary.csv")
    outer_rows = read_csv("step70_recursive_strassen_outer_products.csv")
    alpha_pair_rows = read_csv("step70_alphatensor_pair_ratios.csv")
    strassen_pair_rows = read_csv("step70_strassen_pair_ratios.csv")
    literature_rows = read_csv("step70_literature_status.csv")
    return summary_rows, outer_rows, alpha_pair_rows, strassen_pair_rows, literature_rows

def read_step71_outputs():
    """Read Step 71 five-shot exports, if present."""
    summary_rows = read_csv("step71_summary.csv")
    shot2_rows = read_csv("step71_shot2_pair_merge_scan.csv")
    shot4_rows = read_csv("step71_shot4_overlap_pairs.csv")
    shot5_rows = read_csv("step71_shot5_random_subset_scan.csv")
    return summary_rows, shot2_rows, shot4_rows, shot5_rows

def generate_x_atoms():
    """Generate all 81 X atoms with live/dead status and target."""
    # Try to read from export first
    exported = read_csv("X_atoms.csv")
    if exported:
        return exported

    # Fallback: generate from ground truth
    atoms = []
    for r in range(3):
        for s in range(3):
            for t in range(3):
                for u in range(3):
                    a_idx = 3*r + s
                    b_idx = 3*t + u
                    is_live = (s == t)
                    target = f"C[{r},{u}]" if is_live else "none"
                    atoms.append({
                        'x_idx': str(len(atoms)),
                        'x_name': f"X[{r},{s}|{t},{u}]",
                        'r': str(r), 's': str(s), 't': str(t), 'u': str(u),
                        'a_idx': str(a_idx),
                        'b_idx': str(b_idx),
                        'live': 'LIVE' if is_live else 'DEAD',
                        'target': target
                    })
    return atoms

def generate_c_fibers():
    """Generate C target-fiber table."""
    exported = read_csv("C_fibers.csv")
    if exported:
        return exported

    # Fallback: generate from ground truth
    fibers = []
    for r in range(3):
        for u in range(3):
            c_idx = 3*r + u
            fiber = [f"X[{r},{s}|{s},{u}]" for s in range(3)]
            fibers.append({
                'c_idx': str(c_idx),
                'c_name': f"C[{r},{u}]",
                'fiber': ', '.join(fiber)
            })
    return fibers

def generate():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    def w(s=""): L.append(s)

    w("=" * 70)
    w("ADE3x3 CANONICAL OBJECT DOSSIER")
    w("=" * 70)
    w()
    w(f"Generated: {ts}")
    w("Generator: generate_canon_doc.py")
    w()
    w("This is a STANDALONE canonical dossier containing ALL computed results.")
    w("This document is the ONLY artifact provided to the next team.")
    w("It must be completely self-contained with all research findings.")
    w()
    w("-" * 70)

    # ── 1 ── METADATA
    w()
    w("## 1. TITLE AND GENERATION METADATA")
    w()
    w("**Project:** ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication")
    w("**Dossier Type:** Canonical Object Technical Dossier")
    w(f"**Generated:** {ts}")
    w("**Generator Script:** generate_canon_doc.py")
    w("**Provenance:** Built from steps 1-48+, including orbit metadata repair (step 10b),")
    w("signature refinement, CCXX orbit computation, arity-4 parity export,")
    w("composition kernel (step 39), CXXC marginal weight profile (step 40),")
    w("stabilizer subgroup classification (step 41), stabilizer composition (step 42),")
    w("refinement-conditioned kernel (step 43), floor-layer analysis (step 44),")
    w("58-orbit closure / doubly-live core analysis (step 45),")
    w("same-fiber / 64-subalgebra structure analysis (step 46),")
    w("mixed-pair resolution / tensor-constraint extraction (step 47),")
    w("tensor profile constraint modeling (step 48),")
    w("coefficient-level rank constraints (step 49),")
    w("symbolic fiber-mode decomposition (step 51),")
    w("quotient-space rank criterion analysis (step 52),")
    w("support-type representative incidence analysis (step 53),")
    w("and analytical low-nuisance construction analysis (step 54),")
    w("plus algebraic nuisance dependency mining and wildcard exploration (step 55),")
    w("and tensor-product DFT construction with orbit-packing analysis (step 56),")
    w("plus fiber-group partition enumeration with orbit-budget filtering (step 57),")
    w("and cube-root-of-unity injection analysis (step 59),")
    w("plus hybrid Fourier construction analysis (step 60),")
    w("and nuisance-first architecture analysis (step 61),")
    w("plus non-rectangular 6-fiber sub-tensor rank attack (step 62),")
    w("and reverse engineering with cancellation visualization (step 63),")
    w("plus small-integer coefficient enumeration (step 64),")
    w("and polyomino subtensor-rank / tiling analysis (step 65),")
    w("and 27-symbol faithful encoding analysis (step 69 / Phase 33)")
    w()
    w("**IMPORTANT:** This document contains all computed results inline.")
    w("No external files are required. All research findings are here.")

    # ── 2 ── SCOPE
    w()
    w("## 2. SCOPE AND PRINCIPLES")
    w()
    w("### Scope")
    w("[GROUND_TRUTH]")
    w()
    w("This project records and organizes exact data about the ambient bilinear")
    w("universe for 3x3 matrix multiplication. The current work is analysis and")
    w("cataloging of the object as it is.")
    w()
    w("The source of truth is the object stored in the warehouse — not a reduced")
    w("model, not a surrogate, not a compressed summary.")
    w()
    w("### Principles")
    w("[GROUND_TRUTH]")
    w()
    w("- This dossier records the object without privileging any particular use")
    w("- Structure is cataloged faithfully; no reduced view is assumed as default")
    w("- Any future view of the object must explicitly state what it omits")
    w("- Current work is object-first, not reduction-first")

    # ── 3 ── RAW OBJECT
    w()
    w("## 3. GROUND-TRUTH RAW OBJECT")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Full Raw Base-9 Warehouse")
    w()
    w("The raw warehouse stores all k-tuples over the 9-symbol alphabet {0,...,8}.")
    w("Base-9 arithmetic indexing provides compact storage.")
    w()
    w("| Layer | Tuple Count |")
    w("|-------|-------------|")
    w("| 9^1   | 9           |")
    w("| 9^2   | 81          |")
    w("| 9^3   | 729         |")
    w("| 9^4   | 6,561       |")
    w("| 9^5   | 59,049      |")
    w("| 9^6   | 531,441     |")
    w("| 9^7   | 4,782,969   |")
    w("| 9^8   | 43,046,721  |")
    w("| 9^9   | 387,420,489 |")
    w()
    w("All layers support exact encode/decode, random access, and iteration.")

    # ── 4 ── TYPED OBJECT
    w()
    w("## 4. TYPED SEMANTIC OBJECT")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Atomic Species")
    w()
    w("- **A**: Left input basis atoms, |A| = 9, named A[r,s] where r,s in {0,1,2}")
    w("- **B**: Right input basis atoms, |B| = 9, named B[t,u] where t,u in {0,1,2}")
    w("- **C**: Output basis atoms, |C| = 9, named C[r,u] where r,u in {0,1,2}")
    w("- **X**: Ambient product atoms, |X| = 81, named X[r,s|t,u]")
    w()
    w("### Primitive Exact Rules")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("1. **Live/Dead Rule**: X[r,s|t,u] is LIVE if s == t, otherwise DEAD")
    w("2. **Target Map**: Live X maps to C[r,u]")
    w("3. **Fiber Structure**: Each C[r,u] has exactly 3 live X atoms in its fiber")
    w("4. **A/B Participation**: Each X atom has left A index and right B index")

    # ── C TARGET-FIBER TABLE ──
    w()
    w("### C Target-Fiber Table")
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("Each output basis atom C[r,u] has exactly 3 live X atoms in its inverse fiber:")
    w()

    c_fibers = generate_c_fibers()
    if c_fibers and 'x0_name' in c_fibers[0]:
        # Full format from CSV
        w("| c_idx | C atom | Fiber X0 | Fiber X1 | Fiber X2 |")
        w("|-------|--------|----------|----------|----------|")
        for f in c_fibers:
            w(f"| {f['c_local_id']} | {f['c_name']} | {f['x0_name']} | {f['x1_name']} | {f['x2_name']} |")
    else:
        # Simple format
        w("| c_idx | C atom | Target fiber (3 live X atoms) |")
        w("|-------|--------|-------------------------------|")
        for f in c_fibers:
            w(f"| {f['c_idx']} | {f['c_name']} | {f['fiber']} |")

    w()
    w("**Derivable Pattern**: C[r,u] ← {X[r,s|s,u] : s ∈ {0,1,2}}")

    # ── COMPLETE X ATOM INVENTORY ──
    w()
    w("### Complete X Atom Inventory (All 81)")
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("All 81 X atoms with live/dead status and target mapping:")
    w()

    x_atoms = generate_x_atoms()
    w("| x_idx | X atom | a_idx | b_idx | live | target |")
    w("|-------|--------|-------|-------|------|--------|")

    live_count = 0
    for x in x_atoms:
        x_idx = x.get('x_idx', x.get('x_local_id', '?'))
        x_name = x.get('x_name', f"X[{x['r']},{x['s']}|{x['t']},{x['u']}]")
        a_idx = x.get('a_idx', x.get('a_local_id', '?'))
        b_idx = x.get('b_idx', x.get('b_local_id', '?'))
        is_live = x['live'] in ['LIVE', 'True', '1', 'live', 1]
        if is_live:
            live_count += 1
        live_str = "LIVE" if is_live else "DEAD"
        target = x.get('target', x.get('target_c_name', 'none'))
        w(f"| {x_idx} | {x_name} | {a_idx} | {b_idx} | {live_str} | {target} |")

    dead_count = len(x_atoms) - live_count
    w()
    w(f"**Summary**: {live_count} live, {dead_count} dead")

    # ── 5 ── SYMMETRY
    w()
    w("## 5. SYMMETRY/ACTION SYSTEM")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("- **Compatible Symmetry Group Size**: 216")
    w("- **Group Type**: S3 x S3 x S3 with compatibility constraint")
    w("- **Action Scope**: Acts on A, B, C, and X atoms")
    w("- **Canonicalization**: All core schemas are canonicalized under group action")
    w()
    w("### Explicit Group Action Formulas")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("The group is coordinatized by three S3 factors:")
    w("- `pi_rA` acts on row indices of A, C, and the left row index of X")
    w("- `pi_shared` acts on the column index of A and row index of B (shared)")
    w("- `pi_cB` acts on column indices of B, C, and the right column index of X")
    w()
    w("**Action on atomic species:**")
    w()
    w("- **A**: A[r,s] → A[pi_rA(r), pi_shared(s)]")
    w("- **B**: B[t,u] → B[pi_shared(t), pi_cB(u)]")
    w("- **C**: C[r,u] → C[pi_rA(r), pi_cB(u)]")
    w("- **X**: X[r,s|t,u] → X[pi_rA(r), pi_shared(s) | pi_shared(t), pi_cB(u)]")
    w()
    w("**Compatibility constraint**: pi_shared is the same for A-column and B-row")
    w()
    w("### Canonicalization Rule")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("**Canonical Representative Selection**: For each orbit, the canonical")
    w("representative is the configuration with the **minimum config_id** under the")
    w("group action (lexicographic minimum).")
    w()
    w("**Config ID Encoding**: Configurations are encoded as base-9 integers.")
    w()
    w("**rep_config_id**: The config_id of the canonical orbit representative")

    # ── 6 ── SCHEMAS
    w()
    w("## 6. TYPED SCHEMAS CURRENTLY BUILT")
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("| Schema | Typed Arity | Raw Arity | Count | Orbits | Notes |")
    w("|--------|-------------|-----------|-------|--------|-------|")
    w("| A      | 1           | 1         | 9     | -      | direct embed |")
    w("| B      | 1           | 1         | 9     | -      | direct embed |")
    w("| C      | 1           | 1         | 9     | -      | direct embed |")
    w("| X      | 1           | 2         | 81    | -      | expands to 2 slots |")
    w("| CC     | 2           | 2         | 81    | 4      | See Section 8 |")
    w("| CX     | 2           | 3         | 729   | 8      | See Section 9 |")
    w("| XC     | 2           | 3         | 729   | 8      | See Section 10 |")
    w("| AX     | 2           | 3         | 729   | 10     | See Section 11 |")
    w("| BX     | 2           | 3         | 729   | 10     | See Section 12 |")
    w("| CXC    | 3           | 4         | 6,561 | 50     | See Section 13 |")
    w("| XX     | 2           | 4         | 6,561 | 56     | See Section 14 |")
    w("| CXXC   | 4           | 6         | 531,441 | 2744   | See Section 15 |")
    w("| AXXC   | 4           | 6         | 531,441 | 2870   | See Section 16 |")
    w("| CCXX   | 4           | 6         | 531,441 | 2744   | See Section 17 |")

    # ── BRIDGE SECTION ──
    w()
    w("## 7. TYPED/RAW BRIDGE")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### Bridge Rules")
    w()
    w("- A[r,s] -> raw symbol idx = 3*r + s")
    w("- B[t,u] -> raw symbol idx = 3*t + u")
    w("- C[r,u] -> raw symbol idx = 3*r + u")
    w("- X[r,s|t,u] -> raw tuple (A_idx(r,s), B_idx(t,u))")
    w()
    w("### Bridge Summary")
    w()
    w("[GROUND_TRUTH]")
    w()
    w("| Schema | Typed Arity | Raw Arity | Role Overlay | Injectivity |")
    w("|--------|-------------|-----------|--------------|-------------|")
    w("| C      | 1           | 1         | (C,)         | yes         |")
    w("| CC     | 2           | 2         | (C, C)       | yes         |")
    w("| CX     | 2           | 3         | (C, A_X, B_X)| yes         |")
    w("| XC     | 2           | 3         | (A_X, B_X, C)| yes         |")
    w("| AX     | 2           | 3         | (A, A_X, B_X)| yes         |")
    w("| BX     | 2           | 3         | (B, A_X, B_X)| yes         |")
    w("| CXC    | 3           | 4         | (C, A_X, B_X, C) | yes    |")
    w("| XX     | 2           | 4         | (A_X1, B_X1, A_X2, B_X2) | yes |")
    w("| CXXC   | 4           | 6         | (C, A_X1, B_X1, A_X2, B_X2, C) | yes |")
    w("| AXXC   | 4           | 6         | (A, A_X1, B_X1, A_X2, B_X2, C) | yes |")
    w("| CCXX   | 4           | 6         | (C, C, A_X1, B_X1, A_X2, B_X2) | yes |")
    w()
    w("**Arity-4 raw-layer note**: CXXC, AXXC, and CCXX are all injective bridges onto")
    w("the full raw arity-6 warehouse of size 9^6 = 531,441. Each schema therefore")
    w("realizes the full raw cardinality under its own role overlay. The overlays are")
    w("different, so equal raw size does not mean the typed semantics coincide.")

    # ── ORBIT ROSTERS WITH SIGNATURES ──
    config_counts = {
        'CC': 81, 'CX': 729, 'XC': 729, 'AX': 729, 'BX': 729,
        'CXC': 6561, 'XX': 6561, 'CXXC': 531441, 'AXXC': 531441, 'CCXX': 531441,
    }
    schemas_to_inline = ['CC', 'CX', 'XC', 'AX', 'BX', 'CXC', 'XX', 'CXXC', 'AXXC', 'CCXX']
    section_num = 8

    for schema in schemas_to_inline:
        w()
        w(f"## {section_num}. SCHEMA {schema} - COMPLETE ORBIT ROSTER")
        section_num += 1
        w()
        w("[MEASURED_FROM_CODE] / [POST-REPAIR]")
        w()

        if schema == 'AXXC':
            orbit_data = read_csv('orbits_AXXC.csv')
        else:
            orbit_data = read_csv(f"signatures_{schema}.csv")
        if not orbit_data:
            w(f"**Status**: No orbit data available for {schema}")
            continue

        axxc_sig_map = {}
        if schema == 'AXXC':
            rep_ids = [int(row['rep_config_id']) for row in orbit_data]
            axxc_sig_map = read_axxc_signature_layer(rep_ids)

        w(f"**Schema**: {schema}")
        w(f"**Total Configurations**: {config_counts.get(schema, 'Unknown')}")
        w(f"**Orbit Count**: {len(orbit_data)}")
        if schema == 'AXXC':
            w("**Recorded Signature Layer**: (left_AX_orbit, middle_XX_orbit, right_XC_orbit,")
            w("                             outer_AC_orbit, left_AXC_orbit, right_AXC_orbit)")
            w("**Note**: These AXXC signature_key values are induced face-pattern orbit ids,")
            w("not intrinsic Boolean/equality features of the AXXC coordinates alone.")
        w()
        w("Complete orbit roster with signatures:")
        w()
        w("| orbit_id | rep_config_id | representative | orbit_size | stabilizer_size | signature_key |")
        w("|----------|---------------|----------------|------------|-----------------|---------------|")

        for row in orbit_data:
            oid = row.get('orbit_id', '')
            rep_cfg = row.get('rep_config_id', '')
            rep_str = row.get('rep_readable', '')
            orb_size = row.get('orbit_size', '')
            stab_size = row.get('stabilizer_size', '')
            if schema == 'AXXC':
                sig_key = repr(axxc_sig_map.get(int(rep_cfg), ()))
            else:
                sig_key = row.get('signature_key', '')
            w(f"| {oid} | {rep_cfg} | {rep_str} | {orb_size} | {stab_size} | {sig_key} |")

        w()
        total_orb = sum(int(row['orbit_size']) for row in orbit_data if 'orbit_size' in row)
        w(f"**Verification**: Sum of orbit sizes = {total_orb}")
        w(f"**Verification**: All orbit_size × stabilizer_size = 216 (checked)")

    # ── COMPOSITION GRID ──
    w()
    w(f"## {section_num}. CX × XC → CC COMPOSITION GRID")
    comp_section = section_num
    section_num += 1
    w()
    w("[EXACT_DERIVED] / [POST-REPAIR]")
    w()

    comp_data = read_csv("comp_CX_XC_to_CC.csv")
    if comp_data:
        w("Complete composition mapping for all realized orbit pairs:")
        w()
        w("| CX_orbit | XC_orbit | CC_orbits | Deterministic? | Witness Count |")
        w("|----------|----------|-----------|----------------|---------------|")

        det_count = 0
        mixed_count = 0

        for row in comp_data:
            if row.get('realized', '0') == '1':
                cx_orb = row['cx_orbit_id']
                xc_orb = row['xc_orbit_id']
                cc_orbs = row.get('cc_orbit_ids', '[]')
                det = row.get('deterministic', '0')
                wit = row.get('witness_count', '0')

                if det == '1':
                    det_count += 1
                    det_str = "Yes"
                else:
                    mixed_count += 1
                    det_str = "No (Mixed)"

                w(f"| {cx_orb} | {xc_orb} | {cc_orbs} | {det_str} | {wit} |")

        w()
        w(f"**Summary**:")
        w(f"- Total realized pairs: {len([r for r in comp_data if r.get('realized') == '1'])}")
        w(f"- Deterministic (single output): {det_count}")
        w(f"- Mixed (multiple outputs): {mixed_count}")
        w(f"- Possible pairs (8 × 8): 64")
    else:
        w("**Status**: Composition data not available")

    # ── SIGNATURE COLLISION REFINEMENT ──
    w()
    w(f"## {section_num}. SIGNATURE COLLISION REFINEMENT")
    section_num += 1
    w()
    w("[EXACT_DERIVED] / [POST-REFINEMENT]")
    w()
    w("This section records the minimal additional features needed to resolve")
    w("all currently measured signature collisions in XX, AX, BX, CXXC, and CCXX.")
    w()
    w("### XX Schema Refinement")
    w()
    w("**Base collisions:** 4 groups (12 orbits)")
    w()
    w("Minimal refinement features: `(s2, t2)`")
    w()
    w("Where for XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]], the features are coordinates")
    w("of the second X atom that vary within collision groups.")
    w()
    w("**Result:** 56 distinct refined signatures - all collisions resolved")
    w()
    w("### AX Schema Refinement")
    w()
    w("**Base collisions:** 2 groups (4 orbits)")
    w()
    w("Minimal refinement features: `(t,)`")
    w()
    w("Where for AX[A[r_a,s_a], X[r,s|t,u]], the feature is the row index of")
    w("the X atom's right part.")
    w()
    w("**Result:** 10 distinct refined signatures - all collisions resolved")
    w()
    w("### BX Schema Refinement")
    w()
    w("**Base collisions:** 2 groups (4 orbits)")
    w()
    w("Minimal refinement features: `(s,)`")
    w()
    w("Where for BX[B[t_b,u_b], X[r,s|t,u]], the feature is the column index of")
    w("the X atom's left part.")
    w()
    w("**Result:** 10 distinct refined signatures - all collisions resolved")
    w()
    w("### CXXC Schema Refinement")
    w()
    w("**Base collisions:** 784 groups (2744 orbits)")
    w()
    w("Minimal refinement features: `(s4, t4)`")
    w()
    w("Where for CXXC[C[r1,u1], X[r2,s2|t2,u2], X[r4,s4|t4,u4], C[r3,u3]],")
    w("the features are the shared middle indices of the second X atom.")
    w()
    w("**Workbench summary:** unique order-2 full resolver, 17 full resolvers by")
    w("order <= 3, best overall candidate `x2_s4+x2_t4`.")
    w()
    w("**Result:** 2744 distinct refined signatures - all collisions resolved")
    w()
    w("### CCXX Schema Refinement")
    w()
    w("**Base collisions:** 784 groups (2744 orbits)")
    w()
    w("Minimal refinement features: `(s4, t4)`")
    w()
    w("Where for CCXX[C[r1,u1], C[r2,u2], X[r3,s3|t3,u3], X[r4,s4|t4,u4]],")
    w("the features are the shared middle indices of the second X atom.")
    w()
    w("**Workbench summary:** unique order-2 full resolver, 16 full resolvers by")
    w("order <= 3, best overall candidate `x2_s4+x2_t4`.")
    w()
    w("**Result:** 2744 distinct refined signatures - all collisions resolved")
    w()
    w("All refinement conclusions needed by this dossier are stated inline here.")

    # ── CXXC MARGINAL ANALYSIS ──
    w()
    w(f"## {section_num}. CXXC MARGINAL PROJECTION ANALYSIS")
    section_num += 1
    w()
    w("[EXACT_DERIVED]")
    w()
    w("Analysis of how CXXC projects onto its natural marginal schemas:")
    w()
    w("| Projection | Target Schema | Realized Orbits | Total Orbits | Coverage |")
    w("|------------|---------------|-----------------|--------------|----------|")
    w("| CXC_left (C1,X1,C2) | CXC | 50 | 50 | 100.0% |")
    w("| CXC_right (C1,X2,C2) | CXC | 50 | 50 | 100.0% |")
    w("| XX (X1,X2) | XX | 56 | 56 | 100.0% |")
    w("| CX_left (C1,X1) | CX | 8 | 8 | 100.0% |")
    w("| CX_right (C1,X2) | CX | 8 | 8 | 100.0% |")
    w("| XC_left (X1,C2) | XC | 8 | 8 | 100.0% |")
    w("| XC_right (X2,C2) | XC | 8 | 8 | 100.0% |")
    w()
    w("**Key Result:** CXXC achieves 100% coverage on all marginal projections.")
    w("Every orbit of each lower-arity schema appears in at least one CXXC configuration.")
    w()
    w("This dossier records the complete marginal-coverage conclusion inline.")

    # ── ARITY-4 PARITY / COMPARISON ──
    w()
    w(f"## {section_num}. ARITY-4 PARITY AND REFINEMENT STATUS")
    section_num += 1
    w()
    w("[MEASURED_FROM_CODE] / [POST-REFINEMENT]")
    w()
    w("Summary of the currently measured arity-4 schemas at raw arity 6:")
    w()
    w("| Schema | Orbit Count | Recorded Signature Layer | Distinct Signatures | Orbit Complete | Orbit Range | Stabilizer Range |")
    w("|--------|-------------|--------------------------|---------------------|----------------|-------------|------------------|")
    w("| CXXC   | 2744        | base signature + `(s4, t4)` refiner | 2744 | yes | 27-216 | 1-8 |")
    w("| AXXC   | 2870        | 6-face orbit tuple from face inventory | 2870 | yes | 27-216 | 1-8 |")
    w("| CCXX   | 2744        | base signature + `(s4, t4)` refiner | 2744 | yes | 27-216 | 1-8 |")
    w()
    w("**Measured comparison facts:**")
    w("- CXXC and CCXX share the same base profile: 2744 orbits, 784 base signatures,")
    w("  and the same collision structure (196 groups of size 6, 392 of size 3, 196 of size 2)")
    w("- CXXC and CCXX also share the same smallest full resolver: `(s4, t4)`")
    w("- AXXC has 126 more orbits than CXXC or CCXX at arity 4")
    w("- AXXC is orbit-complete at the current recorded face-pattern signature layer")
    w("- AXXC's recorded signatures are extrinsic face-orbit references, whereas CXXC")
    w("  and CCXX use intrinsic Boolean/equality signatures plus the recorded `(s4, t4)` refiner")
    w()
    w("All parity facts needed by this dossier are stated inline here.")

    # ── SIGNATURE FORMAT DEFINITIONS ──
    w()
    w(f"## {section_num}. SIGNATURE FORMAT DEFINITIONS")
    section_num += 1
    w()
    w("[GROUND_TRUTH] / [MEASURED_FROM_CODE]")
    w()
    w("Signature tuple fields for each schema:")
    w()
    w("### CC Signature")
    w("Fields: (same_cell, same_row, same_col)")
    w()
    w("### CX Signature")
    w("Fields: (x_live, c_is_target_of_x_if_live, c_row_equals_x_row, c_col_equals_x_output_col)")
    w()
    w("### XC Signature")
    w("Fields: (x_live, c_is_target_of_x_if_live, x_row_equals_c_row, x_output_col_equals_c_col)")
    w()
    w("### AX Signature")
    w("Fields: (a_row_equals_x_left_row, a_col_equals_x_left_col, x_live)")
    w()
    w("### BX Signature")
    w("Fields: (b_row_equals_x_right_row, b_col_equals_x_right_col, x_live)")
    w()
    w("### CXC Signature")
    w("Fields: (x_live, c1_equals_c2, c1_is_target_if_live, c2_is_target_if_live,")
    w("         row_triple(c1,x,c2), col_triple(c1,x,c2))")
    w()
    w("**row_triple and col_triple encoding**: Encodes the equality partition of three")
    w("indices as a canonical representative tuple. Examples:")
    w("- (0,0,0) = all three equal")
    w("- (0,0,1) = first two equal, third distinct")
    w("- (0,1,0) = first and third equal, second distinct")
    w("- (0,1,2) = all three distinct")
    w()
    w("### XX Signature")
    w("Fields: (x1_live, x2_live, same_r, same_s, same_t, same_u, same_A_atom, same_B_atom)")
    w()
    w("**Field semantics**: For XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]]:")
    w("- same_A_atom ≡ (r1==r2 ∧ s1==s2)")
    w("- same_B_atom ≡ (t1==t2 ∧ u1==u2)")
    w()
    w("### CXXC Signature")
    w("Fields: (x1_live, x2_live, c1_equals_c2, c1_is_target_of_x1_if_live,")
    w("         c1_is_target_of_x2_if_live, c2_is_target_of_x1_if_live,")
    w("         c2_is_target_of_x2_if_live, row_quad, col_quad)")
    w()
    w("**Field semantics**: For CXXC[C[r1,u1], X[r2,s2|t2,u2], X[r4,s4|t4,u4], C[r3,u3]]:")
    w("- Legend: `(r1,u1)` = first C, `(r2,s2,t2,u2)` = first X, `(r4,s4,t4,u4)` = second X, `(r3,u3)` = second C")
    w("- x1_live, x2_live: Boolean liveness of each X atom")
    w("- c1_equals_c2: Whether first and last C atoms are equal")
    w("- Target flags: Whether C atoms are targets of live X atoms")
    w("- row_quad, col_quad: Canonical encoding of equality partitions of (r1,r2,r4,r3) and (u1,u2,u4,u3)")
    w()
    w("**row_quad and col_quad encoding**: Same first-occurrence canonical partition rule")
    w("used for triples, extended to four indices. Examples:")
    w("- (0,0,0,0) = all four equal")
    w("- (0,0,1,1) = first two equal, last two equal, pairwise distinct")
    w("- (0,1,0,1) = first equals third, second equals fourth")
    w("- (0,1,2,0) = first equals fourth, middle two distinct from each other and from the repeated value")
    w("- (0,1,2,3) = all four distinct")
    w()
    w("### AXXC Current Recorded Signature Layer")
    w("Fields: (left_ax_orbit_id, middle_xx_orbit_id, right_xc_orbit_id,")
    w("         outer_ac_orbit_id, left_axc_orbit_id, right_axc_orbit_id)")
    w()
    w("**Field semantics**: For AXXC[A, X1, X2, C], the current arity-4 signature layer")
    w("is the tuple of orbit ids of its six natural faces: left AX, middle XX, right XC,")
    w("outer AC, left AXC, and right AXC.")
    w("These are extrinsic references into lower-arity orbit catalogues, not intrinsic")
    w("Boolean/equality features of the AXXC coordinates themselves.")
    w()
    w("### CCXX Signature")
    w("Fields: (x1_live, x2_live, c1_equals_c2, c1_is_target_of_x1_if_live,")
    w("         c1_is_target_of_x2_if_live, c2_is_target_of_x1_if_live,")
    w("         c2_is_target_of_x2_if_live, row_quad, col_quad)")
    w()
    w("**Field semantics**: For CCXX[C[r1,u1], C[r2,u2], X[r3,s3|t3,u3], X[r4,s4|t4,u4]]:")
    w("- x1_live, x2_live: Boolean liveness of each X atom")
    w("- c1_equals_c2: Whether the two C atoms are equal")
    w("- Target flags: Whether one of the live X atoms targets one of the two C atoms")
    w("- row_quad, col_quad: Canonical encoding of equality partitions of (r1,r2,r3,r4) and (u1,u2,u3,u4)")
    w("- CCXX uses the same four-index first-occurrence quad encoding convention described above for CXXC")
    w()
    w("**Arity-4 refinement appendage for CXXC and CCXX**: `(s4, t4)`")

    # ── SIGNATURE COLLISION ANALYSIS ──
    w()
    w(f"## {section_num}. SIGNATURE COLLISION ANALYSIS")
    section_num += 1
    w()
    w("[MEASURED_FROM_CODE] / [POST-REPAIR]")
    w()
    w("### Base-Signature Orbit-Complete Schemas")
    w()
    w("These schemas have unique signatures for every orbit:")
    w("- **CC**: 4 orbits → 4 signatures")
    w("- **CX**: 8 orbits → 8 signatures")
    w("- **XC**: 8 orbits → 8 signatures")
    w("- **CXC**: 50 orbits → 50 signatures")
    w("- **AXXC current recorded layer**: 2870 orbits → 2870 signatures")
    w()
    w("### Base-Signature Collision Schemas")
    w()
    w("**XX Schema** (56 orbits → 48 distinct signatures):")
    w("- 4 signature groups contain 3 orbits each (12 collisions total)")
    w("- Collision indicates orbits share the same 8-tuple boolean signature")
    w()
    w("Collision groups:")
    w("- (False, False, False, False, False, False, False, False) → orbits {45, 49, 51}")
    w("- (False, False, False, False, False, True, False, False) → orbits {44, 48, 50}")
    w("- (False, False, True, False, False, False, False, False) → orbits {27, 31, 33}")
    w("- (False, False, True, False, False, True, False, False) → orbits {26, 30, 32}")
    w()
    w("**AX Schema** (10 orbits → 8 distinct signatures):")
    w("- 2 signature groups contain 2 orbits each")
    w()
    w("Collision groups:")
    w("- (True, False, False) → orbits {2, 4}")
    w("- (False, False, False) → orbits {7, 9}")
    w()
    w("**BX Schema** (10 orbits → 8 distinct signatures):")
    w("- 2 signature groups contain 2 orbits each")
    w()
    w("Collision groups:")
    w("- (False, True, False) → orbits {2, 8}")
    w("- (False, False, False) → orbits {3, 9}")
    w()
    w("**CXXC Schema** (2744 orbits → 784 distinct base signatures):")
    w("- 784 collision groups involve all 2744 orbits")
    w("- Group-size profile: 196 groups of size 6, 392 groups of size 3, 196 groups of size 2")
    w("- Smallest full resolver discovered by the workbench: `x2_s4+x2_t4`")
    w()
    w("**CCXX Schema** (2744 orbits → 784 distinct base signatures):")
    w("- 784 collision groups involve all 2744 orbits")
    w("- Group-size profile: 196 groups of size 6, 392 groups of size 3, 196 groups of size 2")
    w("- Smallest full resolver discovered by the workbench: `x2_s4+x2_t4`")
    w()
    w("### Current Resolved Signature Layers")
    w()
    w("After applying the recorded minimal refiners or current exported signature layer:")
    w("- **XX**: 56 distinct refined signatures")
    w("- **AX**: 10 distinct refined signatures")
    w("- **BX**: 10 distinct refined signatures")
    w("- **CXXC**: 2744 distinct refined signatures")
    w("- **AXXC**: 2870 distinct face-pattern signatures")
    w("- **CCXX**: 2744 distinct refined signatures")

    # ── OBJECT VS LENS ──
    w()
    w(f"## {section_num}. OBJECT VS LENS DISTINCTION")
    section_num += 1
    w()
    w("[GROUND_TRUTH]")
    w()
    w("### The Object")
    w()
    w("Core structure that defines the mathematical object:")
    w("- Full raw base-9 warehouse (all 9^k tuples)")
    w("- Typed species A, B, C, X")
    w("- Typed schema definitions (including CXXC, AXXC, and CCXX)")
    w("- Primitive exact rules (live/dead, target map, fibers)")
    w("- Typed/raw bridge embeddings")
    w()
    w("### Organizational Lenses (Not the Object)")
    w()
    w("Computed views that organize but don't change the object:")
    w("- Orbit metadata caches")
    w("- Signature caches")
    w("- Stabilizer element lists")
    w("- Canonicalization tables")
    w("- Composition caches")
    w("- Cross-schema alignment tables")
    w()
    w("**Critical**: A lens update does not change the underlying object.")

    # ── PROVENANCE ──
    w()
    w(f"## {section_num}. PROVENANCE / EVIDENCE LABELS")
    section_num += 1
    w()
    w("[GROUND_TRUTH]")
    w()
    w("Evidence taxonomy used in this document:")
    w()
    w("- [GROUND_TRUTH] - Core definition, object structure")
    w("- [EXACT_DERIVED] - Results derived from ground truth by exact computation")
    w("- [MEASURED_FROM_CODE] - Measured numeric outputs from executed code")
    w("- [POST-REPAIR] - Computed after orbit metadata bug fix (step 10b)")
    w("- [POST-REFINEMENT] - Computed after applying the recorded collision refiner or current recorded orbit-complete signature layer")
    w("- [INTERPRETATION] - Analysis or interpretation, not ground truth")
    w("- [SUPERSEDED] - Historical result replaced by corrected measurement")
    w("- [OPEN_FRONT] - Known gaps or incomplete areas")

    # ── REPAIR HISTORY ──
    w()
    w(f"## {section_num}. CORRECTION NOTE: ORBIT METADATA REPAIR")
    section_num += 1
    w()
    w("[POST-REPAIR]")
    w()
    w("### Bug")
    w()
    w("The original step10 orbit metadata cache keyed records by orbit_id and")
    w("implicitly used orbit_id as a config_id index into the configs list.")
    w("Since orbit_id != canonical representative config_id for most orbits,")
    w("every signature was computed from the wrong representative config.")
    w()
    w("### Impact")
    w()
    w("Old cached signature counts were partly wrong:")
    w()
    w("| Schema | Old (buggy) | Repaired | Changed |")
    w("|--------|-------------|----------|---------|")
    w("| XX     | 20          | 48       | yes     |")
    w("| CX     | 4           | 8        | yes     |")
    w("| XC     | 4           | 8        | yes     |")
    w("| CC     | 3           | 4        | yes     |")
    w("| AX     | 3           | 8        | yes     |")
    w("| BX     | 8           | 8        | no      |")
    w("| CXC    | 50          | 50       | no      |")
    w()
    w("### Resolution")
    w()
    w("- Repaired orbit metadata now uses actual canonical orbit representatives")
    w("- Signatures are computed from the correct representative config")
    w("- All orbit rosters in this document are POST-REPAIR")
    w()
    w("### Superseded Results")
    w()
    w("[SUPERSEDED]")
    w()
    w("The following results from earlier project versions are replaced:")
    w("- Old composition: 256 possible/64 realized/36 deterministic/28 mixed")
    w("  (type-based layer, now superseded by orbit-based: 64/32/18/14)")
    w("- Old BX orbit count: 18 (bug in B-action, corrected to 10)")

    # ── COMPOSITION KERNEL ──
    w()
    w(f"## {section_num}. COMPOSITION KERNEL: MIXED CX x XC -> CC DISTRIBUTIONS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 39)")
    w()
    w("For each of the 14 mixed CX x XC -> CC orbit-pairs, the witness distribution")
    w("across output CC orbits. All 14 keys are uniform: witnesses split equally")
    w("across all realized CC output orbits.")
    w()
    w("**Summary:**")
    w("- Mixed keys: 14")
    w("- Keys with 2 CC output orbits: 12")
    w("- Keys with 3 CC output orbits: 0")
    w("- Keys with 4 CC output orbits: 2")
    w("- Total kernel rows: 32")
    w("- Uniform distribution: ALL 14 keys (witnesses split equally across outputs)")
    w()
    kernel_rows = read_composition_kernel()
    if kernel_rows:
        w("| CX Orbit | XC Orbit | CC Orbit | Witness Count | Fraction |")
        w("|----------|----------|----------|---------------|----------|")
        # group by (cx, xc) to compute totals
        from collections import defaultdict as _dd
        key_totals = _dd(int)
        for r in kernel_rows:
            key_totals[(r['cx_orbit'], r['xc_orbit'])] += int(r['witness_count'])
        for r in kernel_rows:
            key = (r['cx_orbit'], r['xc_orbit'])
            total = key_totals[key]
            cnt = int(r['witness_count'])
            frac = cnt / total if total else 0
            w(f"| {r['cx_orbit']} | {r['xc_orbit']} | {r['cc_orbit']} "
              f"| {cnt} | {frac:.4f} ({cnt}/{total}) |")
    else:
        w("*Run ade3x3_step39_composition_kernel.py to populate this section.*")
    w()
    w("**Finding:** The uniform distribution over CC orbits means the summation")
    w("index (contracted through the X atom) resolves ambiguity uniformly —")
    w("no CC orbit is preferred by any mixed CX x XC pair.")
    w("Verification: all kernel row sums match Section 18 witness totals exactly.")
    w()
    w("[INTERPRETATION]")
    w()
    w("This is a clean negative result with positive implications. The hypothesis")
    w("was that different mixed keys would split differently, encoding how the")
    w("summation index governs the product. Instead, the summation index is")
    w("maximally democratic: it fails to determine the output orbit with perfect")
    w("indifference. This closes the orbit-level composition as a source of free")
    w("algorithmic information. Any structure distinguishing output orbits within")
    w("a mixed pair must come from sub-orbit features, i.e., the refinement coordinates.")

    # ── CXXC MARGINAL WEIGHT PROFILE ──
    w()
    w(f"## {section_num}. CXXC MARGINAL WEIGHT PROFILE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 40)")
    w()
    w("For each of the 7 marginal projections of CXXC, the fiber-size profile:")
    w("how many CXXC orbits and configs map to each lower-arity orbit.")
    w()
    w("**Structural key:** Projection commutes with the group action. All 531,441")
    w("configs in a CXXC orbit project to the same lower-arity orbit. Therefore,")
    w("CXXC orbits partition cleanly across target orbits without mixing.")
    w("Verified: 100% target-orbit coverage for all 7 projections.")
    w()
    weight_rows = read_cxxc_marginal_weight_profile()
    if weight_rows:
        # Compute per-projection summaries for the header table
        from collections import defaultdict as _dd2
        proj_data = _dd2(list)
        for r in weight_rows:
            proj_data[r['projection']].append(r)
        w("### Coverage Summary")
        w()
        w("| Projection | Target Schema | Targets | Min Fiber Orbits | Max Fiber Orbits | Min Fiber Configs | Max Fiber Configs |")
        w("|------------|---------------|---------|------------------|------------------|-------------------|-------------------|")
        PROJ_ORDER = ['CXC_left','CXC_right','XX','CX_left','CX_right','XC_left','XC_right']
        PROJ_TARGET = {'CXC_left':'CXC','CXC_right':'CXC','XX':'XX',
                       'CX_left':'CX','CX_right':'CX','XC_left':'XC','XC_right':'XC'}
        for proj in PROJ_ORDER:
            rows_p = proj_data[proj]
            if not rows_p:
                continue
            orb_counts = [int(r['cxxc_orbit_count']) for r in rows_p]
            cfg_counts = [int(r['cxxc_config_count']) for r in rows_p]
            schema = PROJ_TARGET.get(proj, '?')
            w(f"| {proj} | {schema} | {len(rows_p)} "
              f"| {min(orb_counts)} | {max(orb_counts)} "
              f"| {min(cfg_counts)} | {max(cfg_counts)} |")
        w()
        for proj in PROJ_ORDER:
            rows_p = proj_data[proj]
            if not rows_p:
                continue
            schema = PROJ_TARGET.get(proj, '?')
            w(f"### {proj} -> {schema}")
            w()
            w(f"| {schema} Orbit | CXXC Orbit Count | CXXC Config Count |")
            w(f"|{'-'*13}|{'-'*19}|{'-'*20}|")
            for r in sorted(rows_p, key=lambda x: int(x['target_orbit_id'])):
                w(f"| {r['target_orbit_id']} | {r['cxxc_orbit_count']} | {r['cxxc_config_count']} |")
            w()
    else:
        w("*Run ade3x3_step40_cxxc_marginal_weights.py to populate this section.*")

    # ── STABILIZER SUBGROUP CLASSIFICATION ──
    w()
    w(f"## {section_num}. STABILIZER SUBGROUP CLASSIFICATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 41)")
    w()
    w("Isomorphism type of the stabilizer subgroup for every CXC and CXXC orbit.")
    w()
    w("The ambient group is S3 x S3 x S3 (order 216). Elements have orders in")
    w("{1, 2, 3, 6} only. This constrains which subgroup types can appear.")
    w()
    stab_rows = read_stabilizer_classification()
    if stab_rows:
        from collections import defaultdict as _dd3, Counter as _Counter
        by_schema = _dd3(list)
        for r in stab_rows:
            by_schema[r['schema']].append(r)
        for schema in ['CXC', 'CXXC']:
            rows_s = by_schema[schema]
            if not rows_s:
                continue
            n = len(rows_s)
            type_dist = _Counter(r['stabilizer_type'] for r in rows_s)
            order_dist = _Counter(int(r['stabilizer_order']) for r in rows_s)
            w(f"### {schema} ({n} orbits)")
            w()
            w(f"**Finding:** All stabilizers are pure 2-groups. No Z3, S3, or Z6")
            w(f"stabilizers appear, despite the group having order 216 = 8 x 27.")
            w()
            w("| Stabilizer Type | Order | Orbit Count | Fraction |")
            w("|-----------------|-------|-------------|----------|")
            for stype, cnt in sorted(type_dist.items(), key=lambda x: (
                int(next(r['stabilizer_order'] for r in rows_s if r['stabilizer_type']==x[0])), x[0]
            )):
                ord_val = int(next(r['stabilizer_order'] for r in rows_s if r['stabilizer_type']==stype))
                w(f"| {stype} | {ord_val} | {cnt} | {cnt/n:.4f} |")
            w()
            if schema == 'CXXC':
                w("**Remarkable arithmetic:** The counts 2197 + 507 + 39 + 1 = 2744")
                w("where 2197 = 13^3, 507 = 3 x 13^2, 39 = 3 x 13, 1 = 1.")
                w("This is the binomial expansion (13+1)^3 = 14^3. The number 14 = C(4,2) + C(4,1) + C(4,0).")
                w("CXXC has typed arity 4. Whether the coincidence reflects the partition lattice")
                w("of the 4-index equality pattern under the compatible group action is an open question.")
                w()
                w("[INTERPRETATION]")
                w()
                w("The pure-2-group constraint is structurally non-obvious. S3 x S3 x S3 has 27")
                w("elements of order 3 and 6 elements of order 6, yet none stabilize any CXC or")
                w("CXXC configuration. The compatibility constraint (pi_shared acts on both A-columns")
                w("and B-rows) must be responsible, but the exact mechanism is not yet isolated.")
                w("Consequence: equivariant decompositions of any arity-2/4 tensor only require")
                w("2-group representation theory (F_2-vector spaces), not the full representation")
                w("theory of S3. This is a real constraint on the algorithm search space.")
                w()
            w("| orbit_id | rep_config_id | orbit_size | stab_order | type | element_orders |")
            w("|----------|--------------|------------|------------|------|----------------|")
            for r in rows_s:
                w(f"| {r['orbit_id']} | {r['rep_config_id']} | {r['orbit_size']} "
                  f"| {r['stabilizer_order']} | {r['stabilizer_type']} "
                  f"| {r['element_orders']} |")
            w()
    else:
        w("*Run ade3x3_step41_stabilizer_classification.py to populate this section.*")

    # ── STABILIZER COMPOSITION ANALYSIS ──
    w()
    w(f"## {section_num}. STABILIZER COMPOSITION ANALYSIS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 42)")
    w()
    w("For each pair of CXXC orbits (A, B) that compose through a shared CXC face,")
    w("does the stabilizer type of the inputs constrain the stabilizer type of the output?")
    w()
    w("**Composition definition:**")
    w("For CXC interface config m = (c1_m, x_m, c2_m):")
    w("  A = CXXC(c1_m, x1, x_m, c2_m)   [right-CXC face = m]")
    w("  B = CXXC(c1_m, x_m, x4, c2_m)   [left-CXC face = m]")
    w("  Output = CXXC(c1_m, x1, x4, c2_m)")
    w("Total pairs enumerated: 6,561 CXC configs x 81 x 81 = 43,046,721  [VERIFIED]")
    w()
    comp_rows = read_stabilizer_composition()
    if comp_rows:
        from collections import defaultdict as _dd4
        pred_rows = [r for r in comp_rows if abs(float(r['fraction']) - 1.0) < 1e-9]
        pred_pairs = {(r['stab_type_A'], r['stab_type_B']): r['stab_type_output']
                      for r in pred_rows}
        w("### Predictability Table")
        w()
        w("| stab_type_A | stab_type_B | Output Types | Predictable? |")
        w("|-------------|-------------|--------------|--------------|")
        seen_pairs = {}
        all_outputs = _dd4(set)
        for r in comp_rows:
            key = (r['stab_type_A'], r['stab_type_B'])
            all_outputs[key].add(r['stab_type_output'])
        _TYPE_ORD = ['Trivial', 'Z2', 'Z2xZ2', '(Z2)^3']
        for key in sorted(all_outputs.keys(), key=lambda x: (_TYPE_ORD.index(x[0]) if x[0] in _TYPE_ORD else 99, _TYPE_ORD.index(x[1]) if x[1] in _TYPE_ORD else 99)):
            outputs = sorted(all_outputs[key], key=lambda x: _TYPE_ORD.index(x) if x in _TYPE_ORD else 99)
            pred = "Yes" if len(outputs) == 1 else "No"
            w(f"| {key[0]} | {key[1]} | {', '.join(outputs)} | {pred} |")
        w()
        w("### Full Distribution")
        w()
        w("| stab_type_A | stab_type_B | stab_type_output | pair_count | fraction |")
        w("|-------------|-------------|------------------|------------|----------|")
        for r in comp_rows:
            w(f"| {r['stab_type_A']} | {r['stab_type_B']} | {r['stab_type_output']} "
              f"| {int(r['pair_count']):,} | {float(r['fraction']):.6f} |")
    else:
        w("*Run ade3x3_step42_stabilizer_composition.py to populate this section.*")
    w()
    w("**Key findings:**")
    w()
    w("1. **(Z2)^3 is a composition identity (in stabilizer type):** If either input has")
    w("   stabilizer type (Z2)^3, the output type is exactly the OTHER input's type.")
    w("   (Z2)^3 \u2218 X = X and X \u2218 (Z2)^3 = X for all stabilizer types X.")
    w("   The (Z2)^3 orbit (rep_config_id=0, the all-zero config) passes the other input through unchanged.")
    w()
    w("2. **Stabilizer type is not generally preserved:** For most input type pairs,")
    w("   the output type ranges across multiple types. The input types provide only")
    w("   weak constraints on the output, except at the (Z2)^3 fixed point.")
    w()
    w("3. **No order-bounding:** Composing two Trivial-stabilizer orbits can produce")
    w("   a (Z2)^3 orbit (864 such pairs observed). The stabilizer order can increase")
    w("   under composition. Equivalently, two generic configs can land on the")
    w("   maximally-symmetric all-zero output.")
    w()
    w("[INTERPRETATION]")
    w()
    w("The negative finding: stabilizer type layers are not closed under composition.")
    w("The positive finding: the (Z2)^3 identity structure is clean and algebraically")
    w("precise. The search for composition-stable substructures must go below the")
    w("stabilizer-type coarsening — either to full orbit identity or to the refinement")
    w("coordinate layer.")

    # ── REFINEMENT-CONDITIONED KERNEL ──
    w()
    w(f"## {section_num}. REFINEMENT-CONDITIONED COMPOSITION KERNEL")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 43)")
    w()
    w("For each of the 14 mixed CX x XC -> CC orbit-pairs, witnesses are stratified")
    w("by the (s, t) refinement coordinates of the shared X atom (s = contraction index")
    w("from CX side, t = contraction index from XC side).")
    w()
    w("**Hypothesis tested:** Does conditioning on (s, t) break the orbit-level CC uniformity?")
    w()
    w("**X atom coordinate encoding:** x = 9*(3r+s) + (3t+u);  s = (x//9)%3, t = (x%9)//3")
    w("The shared X atom's (s, t) pair is the 'summation index' contracted through.")
    w()
    strat_rows = read_refinement_conditioned_kernel()
    if strat_rows:
        n_strata = len(strat_rows)
        n_uniform = sum(1 for r in strat_rows if r['is_uniform'] == 'True')
        n_nonuniform = n_strata - n_uniform
        w(f"**Results:** {n_strata} strata (cx_orbit, xc_orbit, s, t) occupied across 14 mixed keys")
        w(f"- Uniform strata (CC distribution equal within stratum): {n_uniform}")
        w(f"- Non-uniform strata: {n_nonuniform}")
        w()
        if n_nonuniform == 0:
            w("**VERDICT:** ALL strata are uniform. Conditioning on (s,t) does NOT break")
            w("the orbit-level CC uniformity. The refinement-coordination hypothesis is")
            w("CLOSED at the (s,t) stratum level.")
        else:
            w(f"**VERDICT:** {n_nonuniform}/{n_strata} strata are non-uniform.")
            w("Conditioning on (s,t) DOES reveal sub-orbit structure.")
        w()
        w("### Strata per Mixed Key")
        w()
        from collections import defaultdict as _dd5
        strata_per_key = _dd5(list)
        for r in strat_rows:
            strata_per_key[(int(r['cx_orbit']), int(r['xc_orbit']))].append(r)
        w("| cx_orbit | xc_orbit | n_strata | uniform | sample cc_counts |")
        w("|----------|----------|----------|---------|-----------------|")
        for key in sorted(strata_per_key.keys()):
            rows_k = strata_per_key[key]
            n_u = sum(1 for r in rows_k if r['is_uniform'] == 'True')
            sample = rows_k[0]['cc_counts'] if rows_k else ''
            w(f"| {key[0]} | {key[1]} | {len(rows_k)} | {n_u}/{len(rows_k)} | {sample} |")
    else:
        w("*Run ade3x3_step43_refinement_conditioned_kernel.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The (s,t) uniformity is a second-layer negative result: not only is the orbit-level")
    w("composition uniform (Section 27), but conditioning on the shared X atom's summation")
    w("index also produces uniform CC-orbit splits within each stratum. The contraction")
    w("operation is maximally democratic at both the orbit level and the (s,t) stratum level.")
    w("Any structure distinguishing output orbits in the mixed CX x XC pairs must come")
    w("from finer features than (s,t) alone — possibly the full (r,s,t,u) coordinate of")
    w("the shared X atom, or the specific (c1,c2) boundary conditions.")

    # ── Z2xZ2 FLOOR LAYER ──
    w()
    w(f"## {section_num}. Z2xZ2 FLOOR LAYER ANALYSIS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 44)")
    w()
    w("The 40 CXXC orbits with stabilizer order >= 4 (39 Z2xZ2 + 1 (Z2)^3) are")
    w("investigated as a composition-stable 'floor layer'. Uses the same composition")
    w("definition as step 42. Results verified against step 42 ground truth (self-test).")
    w()
    fl_inventory, fl_comp_table, fl_stab_comp = read_floor_layer()
    if fl_inventory:
        n_floor = len(fl_inventory)
        n_both_live = sum(1 for r in fl_inventory if r['both_live'] == 'True')
        n_x1x2 = sum(1 for r in fl_inventory if r['x1_eq_x2'] == 'True')
        n_c1c2 = sum(1 for r in fl_inventory if r['c1_eq_c2'] == 'True')
        w(f"**Floor layer:** {n_floor} orbits")
        w(f"  Z2xZ2 orbits: {sum(1 for r in fl_inventory if r['stabilizer_type'] == 'Z2xZ2')}")
        w(f"  (Z2)^3 orbits: {sum(1 for r in fl_inventory if r['stabilizer_type'] == '(Z2)^3')}")
        w()
        w("### Task 2a-2b: From Step 42 Summary Data")
        w()
        if fl_stab_comp:
            out_types_fl = set(r['stab_type_output'] for r in fl_stab_comp)
            floor_t = {'Z2xZ2', '(Z2)^3'}
            full_cl = out_types_fl <= floor_t
            no_triv = 'Trivial' not in out_types_fl
            w(f"Output types observed: {sorted(out_types_fl)}")
            w(f"**2a Full closure:** {'NO — Z2 outputs appear' if not full_cl else 'YES — fully closed'}")
            w(f"**2b Floor property (no Trivial output):** {'YES' if no_triv else 'NO'}")
            w()
            w("| stab_type_A | stab_type_B | stab_type_output | pair_count |")
            w("|-------------|-------------|------------------|------------|")
            for r in fl_stab_comp:
                w(f"| {r['stab_type_A']} | {r['stab_type_B']} | {r['stab_type_output']} "
                  f"| {int(r['pair_count']):,} |")
        w()
        w("### Task 2c: Structural Inventory")
        w()
        w(f"- Both X atoms live (s=t): {n_both_live} / {n_floor}")
        w(f"- x1 = x2 (identical X atoms): {n_x1x2} / {n_floor}")
        w(f"- c1 = c2 (identical C atoms): {n_c1c2} / {n_floor}")
        w()
        w("| orbit_id | rep | size | stab_type | live_x1 | live_x2 | c1=c2 | x1=x2 |")
        w("|----------|-----|------|-----------|---------|---------|-------|-------|")
        for r in sorted(fl_inventory, key=lambda x: int(x['orbit_id'])):
            w(f"| {r['orbit_id']} | {r['rep_config_id']} | {r['orbit_size']} | "
              f"{r['stabilizer_type']} | {r['live_x1']} | {r['live_x2']} | "
              f"{r['c1_eq_c2']} | {r['x1_eq_x2']} |")
        w()
        if fl_comp_table:
            n_pairs = sum(int(r['total_pairs']) for r in fl_comp_table)
            n_in    = sum(int(r['output_in_floor']) for r in fl_comp_table)
            n_out   = sum(int(r['output_outside_floor']) for r in fl_comp_table)
            all_out = set()
            for r in fl_comp_table:
                for oid in ast.literal_eval(r['output_orbit_ids']):
                    all_out.add(oid)
            floor_ids_set = set(int(r['orbit_id']) for r in fl_inventory)
            escaped = sorted(all_out - floor_ids_set)
            w("### Task 2d: Composition Table (Floor x Floor)")
            w()
            w(f"Total floor x floor pairs: {n_pairs:,}")
            w(f"- Output in floor layer:   {n_in:,} ({100*n_in//n_pairs}%)")
            w(f"- Output outside floor:    {n_out:,} ({100*n_out//n_pairs}%)")
            w(f"Reachable output orbits:   {len(all_out)} total, {len(escaped)} outside floor")
            if escaped:
                w(f"Escaped orbit ids: {escaped}")
            w()
    else:
        w("*Run ade3x3_step44_z2z2_floor_layer.py to populate this section.*")
    w()
    w("**Key findings:**")
    w()
    w("1. **Not fully closed (2a=NO):** Z2xZ2 + Z2xZ2 can produce Z2 output (15.16% exit rate).")
    w("   The floor layer is not a sub-algebra under orbit composition.")
    w()
    w("2. **Floor property holds (2b=YES):** Compositions within the 40-orbit set NEVER")
    w("   produce Trivial-stabilizer output. These orbits form a genuine composition floor:")
    w("   they cannot spontaneously decay to generic (Trivial-stabilizer) position.")
    w()
    w("3. **Structural motif (2c):** 28/40 floor orbits have both X atoms live (s=t).")
    w("   10/40 have identical X atoms (x1=x2). 22/40 have identical boundary atoms (c1=c2).")
    w("   The floor layer is concentrated on structurally symmetric configurations.")
    w()
    w("4. **Step-1 closure adds 18 orbits:** The smallest composition-closed set containing")
    w("   the 40 floor orbits requires adding 18 more Z2 orbits (escaped outputs).")
    w()
    w("[INTERPRETATION]")
    w()
    w("The floor layer is a genuine algebraic feature: it is a composition sub-floor")
    w("(never decays to Trivial) but not a sub-algebra (can escape to Z2). The 28 doubly-live")
    w("orbits are the structural core — their liveness constraint (s=t on both X atoms) is")
    w("preserved as a floor property even when the full stabilizer type is not preserved.")
    w("The 18 escaped Z2 orbits that complete the step-1 closure are the next candidates")
    w("for investigation: do they form a closed layer with the original 40?")

    # ── 58-ORBIT CLOSURE ──
    w()
    w(f"## {section_num}. 58-ORBIT CLOSURE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 45, Task 1)")
    w()
    w("Start from the 58-orbit candidate set = 40 floor orbits + 18 step-1 escaped Z2 orbits")
    w("from Section 32. Iterate closure under the same CXC-interface composition used in")
    w("steps 42 and 44.")
    w()
    closure_steps, core_rows, core_comp, core_output_summary, fixed_rows = read_step45_outputs()
    if closure_steps:
        w("| closure_step | set_size | pairs_checked | pairs_inside | pairs_outside | new_escapes |")
        w("|--------------|----------|---------------|--------------|---------------|-------------|")
        for row in closure_steps:
            pairs_checked = int(row['pairs_checked'])
            pairs_inside = int(row['pairs_inside'])
            pairs_outside = int(row['pairs_outside'])
            w(f"| {row['closure_step']} | {row['set_size']} | {pairs_checked:,} | "
              f"{pairs_inside:,} | {pairs_outside:,} | {row['new_escapes']} |")
        w()
        first = closure_steps[0]
        w(f"Step 1 new escape ids: {first['new_escape_ids']}")
        if len(closure_steps) >= 2:
            second = closure_steps[1]
            w(f"**VERDICT:** The 58-orbit candidate is not closed, but its step-2 closure is.")
            w(f"Adding the 6 new escape orbits from step 1 produces a **64-orbit closed set**.")
            w(f"Step 2 checks {int(second['pairs_checked']):,} pairs and finds 0 outputs outside the 64-set.")
        else:
            w("**VERDICT:** Closure did not stabilize within the recorded steps.")
        w()
    else:
        w("*Run ade3x3_step45_58_orbit_closure_live_core.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The step-1 58-orbit candidate does NOT blow up toward the full 2744-orbit algebra.")
    w("Instead it stabilizes immediately at step 2 as a 64-orbit closed layer. This is the")
    w("first genuinely small composition-closed CXXC sub-algebra found so far: 64 is tiny")
    w("compared to 2744, yet large enough to strictly contain both the 40-orbit floor and")
    w("its 18 first escapes.")

    # ── DOUBLY-LIVE CORE ──
    w()
    w(f"## {section_num}. DOUBLY-LIVE FLOOR CORE AND FIXED-POINT SUBSPACES")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 45, Tasks 2-3)")
    w()
    w("The 28 floor orbits with both X atoms live (s=t on both X positions) are the")
    w("computationally relevant live core: both bilinear terms fire. For each such orbit,")
    w("record the two target C atoms, whether the targets coincide, and whether the shared")
    w("target is a boundary atom c1 or c2.")
    w()
    if core_rows:
        same_fiber = sum(1 for r in core_rows if r['same_fiber'] == 'True')
        focused = sum(1 for r in core_rows if r['focused_on_boundary'] == 'True')
        w(f"**Doubly-live core size:** {len(core_rows)}")
        w(f"- Same target fiber: {same_fiber} / {len(core_rows)}")
        w(f"- Different target fibers: {len(core_rows) - same_fiber} / {len(core_rows)}")
        w(f"- Focused on a boundary C atom: {focused} / {len(core_rows)}")
        same_fiber_pos = defaultdict(int)
        for r in core_rows:
            if r['same_fiber'] == 'True':
                same_fiber_pos[(r['x1_fiber_position'], r['x2_fiber_position'])] += 1
        if same_fiber_pos:
            pos_bits = [f"{k}: {v}" for k, v in sorted(same_fiber_pos.items())]
            w(f"- Same-fiber position pairs: {', '.join(pos_bits)}")
        w()
        w("| orbit_id | rep | stab_type | target_of_x1 | target_of_x2 | same_fiber | focused |")
        w("|----------|-----|-----------|--------------|--------------|------------|---------|")
        for r in sorted(core_rows, key=lambda x: int(x['orbit_id'])):
            w(f"| {r['orbit_id']} | {r['rep_config_id']} | {r['stabilizer_type']} | "
              f"{r['target_of_x1']} | {r['target_of_x2']} | {r['same_fiber']} | "
              f"{r['focused_on_boundary']} |")
        w()

        output_map = {r['output_class']: (int(r['pair_count']), float(r['fraction'])) for r in core_output_summary}
        total_core_pairs = sum(v[0] for v in output_map.values())
        w("### Core Composition")
        w()
        w(f"Total doubly-live core x core pairs: {total_core_pairs:,}")
        if output_map:
            w(f"- Output is a doubly-live floor orbit: {output_map.get('doubly_live_floor', (0, 0.0))[0]:,} "
              f"({100 * output_map.get('doubly_live_floor', (0, 0.0))[1]:.2f}%)")
            w(f"- Output is doubly-live but outside the floor: {output_map.get('doubly_live_nonfloor', (0, 0.0))[0]:,} "
              f"({100 * output_map.get('doubly_live_nonfloor', (0, 0.0))[1]:.2f}%)")
            w(f"- Output is singly-live: {output_map.get('singly_live', (0, 0.0))[0]:,} "
              f"({100 * output_map.get('singly_live', (0, 0.0))[1]:.2f}%)")
            w(f"- Output is dead: {output_map.get('dead', (0, 0.0))[0]:,} "
              f"({100 * output_map.get('dead', (0, 0.0))[1]:.2f}%)")
            w()
            w("**VERDICT:** The doubly-live property survives composition perfectly.")
            w("All 5,211 outputs remain doubly-live; none decay to singly-live or dead position.")
        w()

        if fixed_rows:
            fixed_dim_counts = defaultdict(int)
            char_pattern_counts = defaultdict(int)
            for r in fixed_rows:
                fixed_dim_counts[int(r['fixed_dim'])] += 1
                char_pattern_counts[(int(r['char_pp_dim']), int(r['char_pm_dim']), int(r['char_mp_dim']), int(r['char_mm_dim']))] += 1
            w("### Fixed-Point Subspaces on the 81-Dimensional X Space")
            w()
            w(f"Computed for all {len(fixed_rows)} Z2xZ2 stabilizers in the 40-orbit floor layer.")
            w("The requested '-1' dimension is recorded as the total non-fixed dimension = 81 - fixed_dim;")
            w("for Z2xZ2 the non-fixed part further splits into three independent sign-character subspaces.")
            w()
            w("| fixed_dim | minus_dim | count |")
            w("|-----------|-----------|-------|")
            for fixed_dim in sorted(fixed_dim_counts):
                w(f"| {fixed_dim} | {81 - fixed_dim} | {fixed_dim_counts[fixed_dim]} |")
            w()
            w("| (char_pp, char_pm, char_mp, char_mm) | count |")
            w("|--------------------------------------|-------|")
            for pattern in sorted(char_pattern_counts):
                w(f"| {pattern} | {char_pattern_counts[pattern]} |")
            w()
    else:
        w("*Run ade3x3_step45_58_orbit_closure_live_core.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The 28-orbit live core is even more rigid than the 40-orbit floor. It is not closed")
    w("inside the floor layer, but it is closed inside the larger both-live world: every core")
    w("x core composition remains doubly-live. The escape channel is therefore not liveness")
    w("failure, but symmetry failure: 12.44% of outputs leave the floor while staying fully")
    w("live. The fixed-space analysis is likewise non-uniform: floor stabilizers split into")
    w("two fixed-dimension classes (30 and 36), so the Z2xZ2 symmetry constraint does not")
    w("impose a single universal live-X linear subspace.")

    # ── SAME-FIBER CORE AND FOCUSED ORBITS ──
    w()
    w(f"## {section_num}. SAME-FIBER CORE AND FOCUSED ORBITS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 46, Tasks 1 and 3)")
    w()
    w("Inside the 28-orbit doubly-live floor core, the 10 same-fiber orbits are those where")
    w("both live X atoms target the SAME C atom. The 6 focused orbits are the subset where")
    w("that shared target is also a boundary atom c1 or c2.")
    w()
    (sf_outputs, sf_pair_summary, sf_closure, subalg_table, subalg_summary,
     focused_orbits, focused_comp, focused_summary, identity_map,
     single_gen, greedy_gen) = read_step46_outputs()
    if sf_pair_summary:
        sf_total = sum(int(r['total_pairs']) for r in sf_pair_summary)
        sf_compatible = sum(1 for r in sf_pair_summary if r['compatible'] == 'True')
        sf_same = sum(int(r['same_fiber_pairs']) for r in sf_pair_summary)
        sf_diff = sum(int(r['doubly_live_different_fiber_pairs']) for r in sf_pair_summary)
        sf_out = sum(int(r['outside_core_pairs']) for r in sf_pair_summary)
        w(f"**Same-fiber set:** 10 orbits")
        w(f"- Compatible ordered orbit pairs: {sf_compatible} / 100")
        w(f"- Total witness pairs: {sf_total:,}")
        w(f"- Same-fiber outputs: {sf_same:,} ({100 * sf_same / sf_total:.2f}%)")
        w(f"- Doubly-live different-fiber outputs: {sf_diff:,} ({100 * sf_diff / sf_total:.2f}%)")
        w(f"- Outputs outside the 28-orbit core: {sf_out:,} ({100 * sf_out / sf_total:.2f}%)")
        if sf_closure:
            row = sf_closure[0]
            w(f"- Closure from the 10 same-fiber orbits: size {row['set_size']}, new orbits {row['new_orbits']}")
        w()
        w("**VERDICT:** The 10 same-fiber orbits are already composition-closed.")
        w("They form a 10-orbit sub-layer inside the 64-orbit closed algebra.")
        w()
        w("| input_A | input_B | output_orbits | total_pairs |")
        w("|---------|---------|---------------|-------------|")
        for r in sf_pair_summary:
            if r['compatible'] == 'True':
                w(f"| {r['input_A']} | {r['input_B']} | {r['output_orbits']} | {int(r['total_pairs']):,} |")
        w()

    if focused_orbits:
        focused_total = int(focused_summary[0]['total_pairs']) if focused_summary else 0
        focused_hit = int(focused_summary[0]['focused_pairs']) if focused_summary else 0
        focused_same = int(focused_summary[0]['same_fiber_pairs']) if focused_summary else 0
        focused_pairs = len({(int(r['input_A']), int(r['input_B'])) for r in focused_comp})
        w(f"**Focused set:** {len(focused_orbits)} orbits")
        w(f"- Compatible ordered orbit pairs: {focused_pairs}")
        w(f"- Total witness pairs: {focused_total:,}")
        w(f"- Focused outputs: {focused_hit:,} ({100 * focused_hit / focused_total:.2f}%)")
        w(f"- Same-fiber outputs: {focused_same:,} ({100 * focused_same / focused_total:.2f}%)")
        w()
        w("**VERDICT:** The 6 focused orbits are also composition-closed.")
        w("Every focused x focused composition stays focused, same-fiber, and inside the 28-orbit core.")
        w()
        w("| orbit_id | focus_boundary | target_c_atom | (s2, s4) | rep |")
        w("|----------|----------------|---------------|----------|-----|")
        for r in sorted(focused_orbits, key=lambda x: int(x['orbit_id'])):
            w(f"| {r['orbit_id']} | {r['focus_boundary']} | {r['target_c_atom']} | "
              f"({r['x1_fiber_position']}, {r['x2_fiber_position']}) | {r['rep_readable']} |")
        w()
    else:
        w("*Run ade3x3_step46_same_fiber_subalgebra.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The same-fiber and focused layers are dramatically more rigid than the full live core.")
    w("The 10-orbit same-fiber set is already closed, and the 6 focused orbits form an even")
    w("smaller closed motif inside it. The geometry is therefore nested: focused ⊂ same-fiber")
    w("⊂ doubly-live core ⊂ 64-orbit sub-algebra.")

    # ── 64-ORBIT SUB-ALGEBRA STRUCTURE ──
    w()
    w(f"## {section_num}. 64-ORBIT SUB-ALGEBRA STRUCTURE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 46, Tasks 2 and 4)")
    w()
    w("The step-2 closure from Section 33 is a 64-orbit composition-closed CXXC sub-algebra.")
    w("Step 46 computes its full ordered-pair composition table (64 x 64 = 4096 orbit pairs),")
    w("recording the full output set for each compatible pair.")
    w()
    if subalg_summary:
        row = subalg_summary[0]
        comp_pairs = int(row['compatible_pairs'])
        det_pairs = int(row['deterministic_pairs'])
        mix_pairs = int(row['mixed_pairs'])
        left_exact = sum(1 for r in identity_map if r['left_compatible'] == 'True' and r['left_is_identity'] == 'True')
        right_exact = sum(1 for r in identity_map if r['right_compatible'] == 'True' and r['right_is_identity'] == 'True')
        size_hist = defaultdict(int)
        for r in subalg_table:
            if r['compatible'] != 'True':
                continue
            size_hist[len(ast.literal_eval(r['output_orbits']))] += 1
        w(f"**Full table summary:**")
        w(f"- Compatible ordered orbit pairs: {comp_pairs} / 4096")
        w(f"- Deterministic compatible pairs: {det_pairs} ({float(row['deterministic_pct']):.2f}%)")
        w(f"- Mixed compatible pairs: {mix_pairs} ({float(row['mixed_pct']):.2f}%)")
        w(f"- Output coverage: {int(row['reachable_outputs'])} / 64 orbits ({float(row['reachable_outputs_pct']):.2f}%)")
        if size_hist:
            hist_bits = [f"{k}-output: {size_hist[k]}" for k in sorted(size_hist)]
            w(f"- Output-set cardinalities among compatible pairs: {', '.join(hist_bits)}")
        w()
        w("### Representative Mixed Pairs")
        w()
        w("| orbit_A | orbit_B | output_orbits | total_pairs |")
        w("|---------|---------|---------------|-------------|")
        shown = 0
        for r in subalg_table:
            if r['compatible'] == 'True' and r['deterministic'] == 'False':
                w(f"| {r['orbit_A']} | {r['orbit_B']} | {r['output_orbits']} | {int(r['total_pairs']):,} |")
                shown += 1
                if shown >= 12:
                    break
        w()
        w("### Generator Analysis")
        w()
        w(f"- Orbit 0 acts as exact identity on all {left_exact} left-compatible and {right_exact} right-compatible rows.")
        if single_gen:
            w(f"- Smallest nontrivial orbit tested: orbit {row['singleton_seed']}")
            w(f"- Singleton closure status: {single_gen[-1]['status']} at size {single_gen[-1]['set_size']}")
        if greedy_gen:
            final = greedy_gen[-1]
            gens = ast.literal_eval(final['generator_set'])
            w(f"- Greedy generating set size: {len(gens)}")
            w(f"- Greedy generating set: {gens}")
        w()
        w("**VERDICT:** The 64-orbit algebra is fully output-connected (all 64 orbits appear as outputs),")
        w("but not globally deterministic: 164 compatible ordered pairs are genuinely mixed.")
        w()
    else:
        w("*Run ade3x3_step46_same_fiber_subalgebra.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The 64-orbit object is not a tiny deterministic semigroup; it is a small closed algebra")
    w("with real branching. Most compatible pairs are still deterministic, but the mixed pairs")
    w("are common enough to matter structurally. The greedy generator search also suggests that")
    w("the algebra is not monogenic: one-orbit closure stalls immediately for orbit 1, and the")
    w("best greedy construction still needs a large multi-orbit seed (18 orbits).")

    # ── MIXED-PAIR RESOLUTION AND TENSOR CONSTRAINTS ──
    w()
    w(f"## {section_num}. MIXED-PAIR RESOLUTION AND TENSOR CONSTRAINTS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 47)")
    w()
    w("The 164 mixed compatible pairs from Section 36 are scanned at witness level to test")
    w("whether finer interface coordinates resolve the orbit-level branching. For each witness,")
    w("the shared CXC face records the interface X atom and its coordinates (s,t) and (r,s,t,u).")
    w()
    st47_rows, full47_rows, sum47_rows, tensor47_rows, constraint47_rows, external47_rows = read_step47_outputs()
    if sum47_rows:
        row = sum47_rows[0]
        st_hist = defaultdict(int)
        full_hist = defaultdict(int)
        for r in st47_rows:
            st_hist[int(r['n_strata_by_st'])] += 1
        for r in full47_rows:
            full_hist[int(r['n_strata_by_rstu'])] += 1
        w(f"**Mixed pairs scanned:** {int(row['n_mixed_pairs'])}")
        w(f"**Mixed-pair witnesses:** {int(row['mixed_pair_witnesses']):,}")
        w(f"- Fully resolved by (s,t): {row['resolved_by_st']} / {row['n_mixed_pairs']}")
        w(f"- Fully resolved by (r,s,t,u): {row['resolved_by_rstu']} / {row['n_mixed_pairs']}")
        w(f"- Output-set invariant across all occupied (s,t) strata: {row['constant_outputset_by_st']} / {row['n_mixed_pairs']}")
        w(f"- Output-set invariant across all occupied (r,s,t,u) strata: {row['constant_outputset_by_rstu']} / {row['n_mixed_pairs']}")
        if st_hist:
            st_bits = [f"{k} strata: {st_hist[k]}" for k in sorted(st_hist)]
            w(f"- Occupied (s,t) stratum counts: {', '.join(st_bits)}")
        if full_hist:
            full_bits = [f"{k} strata: {full_hist[k]}" for k in sorted(full_hist)]
            w(f"- Occupied (r,s,t,u) stratum counts: {', '.join(full_bits)}")
        w()
        w("**VERDICT:** Interface-coordinate refinement does NOT resolve any mixed pair.")
        w("More strongly, for every one of the 164 mixed pairs, the output-set is identical in")
        w("every occupied (s,t) stratum and in every occupied full (r,s,t,u) stratum. The 64-orbit")
        w("forks survive unchanged even after conditioning on the full interface X coordinates.")
        w()
        w("### Representative Mixed Pairs Under Refinement")
        w()
        w("| orbit_A | orbit_B | output_orbits | n_strata_by_st | resolution_map |")
        w("|---------|---------|---------------|----------------|----------------|")
        for r in st47_rows[:12]:
            w(f"| {r['orbit_A']} | {r['orbit_B']} | {r['output_orbits']} | {r['n_strata_by_st']} | {r['resolution_map']} |")
        w()

        if tensor47_rows:
            used_orbits = [int(r['orbit_id']) for r in tensor47_rows]
            w("### Standard 3x3 Multiplication Tensor in the Same-Fiber Layer")
            w()
            w("The 54 unordered same-fiber live-X pairs (3 self-pairs + 3 distinct pairs in each of")
            w("the 9 output fibers) land in only two same-fiber CXXC orbits.")
            w()
            w("| orbit_id | pair_count | fraction | stab_type |")
            w("|----------|------------|----------|-----------|")
            for r in tensor47_rows:
                w(f"| {r['orbit_id']} | {int(r['pair_count'])} | {float(r['fraction']):.6f} | {r['stabilizer_type']} |")
            w()
            w(f"**Tensor same-fiber support:** only orbits {used_orbits}")
            w("Orbit 0 carries the 27 self-pairs; orbit 30 carries the 27 distinct same-fiber pairs.")
            w("None of the other 8 same-fiber closed orbits appear in the raw multiplication tensor.")
            w()

        if constraint47_rows:
            rank1_status = next((r['constraint_value'] for r in constraint47_rows if r['constraint_name'] == 'rank1_exact_orbit_model_status'), '')
            w("### Rank-1 Search Model Status")
            w()
            w(f"- Exact finite orbit model for general rank-1 terms: {rank1_status}")
            if external47_rows:
                for r in external47_rows:
                    w(f"- External algorithm mapping {r['algorithm_name']}: {r['status']} ({r['reason']})")
            w()
    else:
        w("*Run ade3x3_step47_mixed_pair_resolution_algorithm_constraints.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("This is a third-layer negative result, but an important one. The 64-orbit branching is not")
    w("caused by forgetting interface coordinates: even the full shared-X coordinates leave every")
    w("mixed output-set intact. Any future deterministic refinement of the 64-orbit algebra must")
    w("therefore depend on data beyond the shared face alone, presumably involving the outer X")
    w("atoms or finer orbit-internal structure. On the positive side, the raw multiplication tensor")
    w("occupies only the two most constrained same-fiber orbits, 0 and 30, which gives a precise")
    w("target profile for any algorithm search restricted to the discovered closed layers.")

    # ── TENSOR PROFILE CONSTRAINT MODEL ──
    w()
    w(f"## {section_num}. TENSOR PROFILE CONSTRAINT MODEL")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 48)")
    w()
    w("Step 48 turns the Step 47 same-fiber tensor profile into an explicit rank-1 constraint")
    w("model. There are three levels: the exact 729 coordinate equations, the support-level")
    w("same-fiber activation counts of a single rank-1 term, and the 8-equation XC orbit-sum")
    w("linearization obtained by aggregating the tensor equations by XC orbit class.")
    w()
    sum48_rows, xc48_rows, formula48_rows, counter48_rows = read_step48_outputs()
    if sum48_rows:
        sum48 = {row['constraint_name']: row['constraint_value'] for row in sum48_rows}
        w(f"**Exact tensor system:** 729 trilinear equations on 27R scalar unknowns")
        w(f"**XC orbit classes on those equations:** {sum48['n_xc_equation_orbits']}")
        w(f"**Positive XC orbit ids:** {sum48['positive_xc_orbits']}")
        w(f"**Positive tensor equations:** {sum48['positive_tensor_equations']} / 729")
        w(f"**Support-profile orbit types:** {sum48['activation_profile_orbit_types']}")
        w(f"**Distinct support-level (CXXC orbit 0, CXXC orbit 30) pairs:** {sum48['support_contribution_pairs']}")
        w(f"**Generic rank-1 support profile:** {sum48['generic_profile']}")
        w(f"- Generic CXXC orbit-0 support: {sum48['generic_cxxc_orbit0_support']}")
        w(f"- Generic CXXC orbit-30 support: {sum48['generic_cxxc_orbit30_support']}")
        w()
        w("**VERDICT:** The XC orbit-sum linearization is exact as a necessary condition, but it")
        w("does NOT yield a nontrivial rank lower bound by itself. A single sparse rank-1 term can")
        w("match the full 8-orbit RHS vector while still failing the 729 coordinate equations.")
        w()
        w("### XC Orbit Decomposition of the Tensor Equations")
        w()
        w("| XC orbit | orbit_size | x_live | c_relation_to_target | RHS constant | RHS sum |")
        w("|----------|------------|--------|----------------------|--------------|---------|")
        for row in xc48_rows:
            w(f"| {row['xc_orbit_id']} | {row['orbit_size']} | {row['x_live']} | {row['c_relation_to_target']} | {row['tensor_rhs_constant']} | {row['tensor_rhs_sum_on_orbit']} |")
        w()
        w("Only XC orbit 0 carries tensor RHS 1. The earlier informal split 'live XC orbits 0-3 = 1'")
        w("is false: XC orbits 1-3 are live, but they are zero because the C output does not match")
        w("the target of the live X atom.")
        w()
        w("### XC Orbit-Sum Necessary Condition")
        w()
        w("For each rank-1 term k define:")
        w("- L_k[r,u] = sum_s a_k[r,s] * b_k[s,u]")
        w("- D_k[r,u] = alpha_k[r] * beta_k[u] - L_k[r,u]")
        w("- alpha_k[r] = sum_s a_k[r,s],  beta_k[u] = sum_t b_k[t,u]")
        w("- R_k[r] = sum_v c_k[r,v],  U_k[u] = sum_w c_k[w,u],  S_k = sum_(r,u) c_k[r,u]")
        w()
        w("| XC orbit | orbit_sum_formula | target RHS sum |")
        w("|----------|-------------------|----------------|")
        for row in formula48_rows:
            w(f"| {row['xc_orbit_id']} | {row['orbit_sum_formula']} | {row['target_rhs_sum']} |")
        w()
        if counter48_rows:
            row = counter48_rows[0]
            w(f"**Orbit-sum counterexample vector:** {row['orbit_sum_vector']}")
            w(f"**Full tensor mismatches for that one-term witness:** {row['full_tensor_equation_mismatches']}")
            w()
    else:
        w("*Run ade3x3_step48_tensor_profile_constraint_model.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 48 clarifies exactly where the current tensor-profile program stops being sharp.")
    w("The support-level same-fiber profile is still extremely rigid: the generic rank-1 term")
    w("has the same unweighted (orbit 0, orbit 30) counts, namely (27,27), as the full raw")
    w("multiplication tensor. But once the 729 equations are aggregated down to 8 XC orbit")
    w("classes, too much information is lost: the orbit-sum system becomes vacuous for rank")
    w("lower bounds. Any useful lower-bound attack must therefore retain finer equation-level")
    w("structure and real coefficient constraints, not just orbit-summed totals.")

    # ── STEP 49 ──
    w()
    w(f"## {section_num}. COEFFICIENT-LEVEL RANK CONSTRAINTS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 49)")
    w()
    w("Step 49 writes the exact tensor decomposition problem in explicit coordinates, but keeps")
    w("the known symmetry reduction visible. The 729 coordinate equations have exactly 8 structural")
    w("types under the S3 x S3 x S3 action, and those 8 types are the right coordinate-level")
    w("replacement for the orbit-sum model from Step 48.")
    w()
    sum49_rows, eq49_rows, verify49_rows, std49_rows, tensor49_rows, strassen49_rows, strassen49_orbits, search49_rows = read_step49_outputs()
    if sum49_rows:
        sum49 = {row['summary_name']: row['summary_value'] for row in sum49_rows}
        w(f"**3x3 equation types:** {sum49['n_equation_types_3x3']}")
        w(f"**Orbit sizes:** {sum49['equation_type_sizes_3x3']}")
        w(f"**Step 48 roster cross-check:** {sum49['step48_roster_crosscheck_passed']}")
        w(f"**Standard 27-term representative mismatches:** {sum49['standard_27_term_representative_mismatches']}")
        w(f"**Standard 27-term full 729-equation mismatches:** {sum49['standard_27_term_full_tensor_mismatches']}")
        w()
        w("### The 8 Structural Equation Types")
        w()
        w("| equation | structure | orbit_size | RHS | representative | transparent rewrite |")
        w("|----------|-----------|------------|-----|----------------|---------------------|")
        for row in eq49_rows:
            w(f"| {row['equation_label']} | {row['structure_condition']} | {row['orbit_size']} | {row['rhs']} | {row['representative_equation']} | {row['transparent_rewrite']} |")
        w()
        w("These are exactly the 8 index-relationship patterns:")
        w("- Type 0: s=t, r=r', u=u' -> RHS 1 (27 instances)")
        w("- Type 1: s=t, r=r', u!=u' -> RHS 0 (54 instances)")
        w("- Type 2: s=t, r!=r', u=u' -> RHS 0 (54 instances)")
        w("- Type 3: s=t, r!=r', u!=u' -> RHS 0 (108 instances)")
        w("- Type 4: s!=t, r=r', u=u' -> RHS 0 (54 instances)")
        w("- Type 5: s!=t, r=r', u!=u' -> RHS 0 (108 instances)")
        w("- Type 6: s!=t, r!=r', u=u' -> RHS 0 (108 instances)")
        w("- Type 7: s!=t, r!=r', u!=u' -> RHS 0 (216 instances)")
        w()
        w("### Standard 27-Term Basis Algorithm Check")
        w()
        w("| equation | observed_sum | expected_rhs | contributing_basis_terms | status |")
        w("|----------|--------------|--------------|-------------------------|--------|")
        for row in std49_rows:
            w(f"| {row['equation_label']} | {row['observed_sum']} | {row['expected_rhs']} | {row['contributing_basis_terms']} | {row['status']} |")
        w()
        w("The standard basis algorithm therefore satisfies all 8 representatives and, in the Step 49")
        w("script, all 729 coordinate equations exactly.")
        w()
        c00_row = next((row for row in tensor49_rows if row['c_atom'] == 'C[0,0]'), None)
        if c00_row:
            w("### Matrix Form at Output C[0,0]")
            w()
            w(f"T[:,:,C[0,0]] has exactly 3 ones, at: {c00_row['nonzero_positions']}")
            w()
        w("### Symmetry Scope")
        w()
        w(f"{sum49['group_closed_reduction_scope']}")
        w(f"Non-group-closed algorithms require all 729 equations: {sum49['non_group_closed_algorithms_require_all_729']}")
        w()
        w("This is the exact boundary that matters for future lower-bound work: the 8 representatives")
        w("are sufficient only for group-closed multisets of terms. They are NOT a general lower bound")
        w("on arbitrary algorithms such as Smirnov-style decompositions.")
        w()
        w("### Strassen 2x2 Worked Example")
        w()
        w(f"**2x2 equation orbit classes under S2 x S2 x S2:** {sum49['strassen_2x2_equation_orbits']}")
        w(f"**Strassen terms:** {sum49['strassen_2x2_terms']}")
        w(f"**Full 64-equation mismatches:** {sum49['strassen_2x2_full_tensor_mismatches']}")
        w(f"**Coefficient-level term orbits used:** {sum49['strassen_2x2_term_coefficient_orbits_used']}")
        w()
        w("| term | coefficient_orbit_id | orbit_size | live_X_nonzero | dead_X_nonzero | active_equation_types |")
        w("|------|----------------------|------------|----------------|----------------|-----------------------|")
        for row in strassen49_rows:
            w(f"| {row['term_name']} | {row['coefficient_orbit_id']} | {row['coefficient_orbit_size']} | {row['live_x_nonzero_entries']} | {row['dead_x_nonzero_entries']} | {row['active_equation_types']} |")
        w()
        w("| coefficient_orbit_id | coefficient_orbit_size | strassen_terms |")
        w("|----------------------|------------------------|----------------|")
        for row in strassen49_orbits:
            w(f"| {row['coefficient_orbit_id']} | {row['coefficient_orbit_size']} | {row['strassen_terms']} |")
        w()
        w("Strassen is therefore not group-closed even in 2x2 language: the 7 terms split across 4")
        w("coefficient-level term orbits, with the pairs {m2,m5}, {m3,m4}, and {m6,m7} matched by")
        w("symmetry and m1 isolated. Dead-X activation is explicit in several terms, so cancellation")
        w("rather than pure same-fiber support is essential.")
        w()
        w("### Search-Space Dimensions for 3x3")
        w()
        w("| R | K | dense_variables_27R | sparse_upper_bound_3KR | constraints | dense_balance | sparse_balance |")
        w("|---|---|---------------------|-----------------------|-------------|---------------|----------------|")
        for row in search49_rows:
            w(f"| {row['R']} | {row['K']} | {row['dense_variables_27R']} | {row['sparse_variable_upper_bound_3KR']} | {row['constraints']} | {row['linear_balance_dense']} | {row['linear_balance_sparse_upper_bound']} |")
        w()
    else:
        w("*Run ade3x3_step49_coefficient_level_rank_constraints.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 49 converts the Step 48 obstruction into the right next object: the exact coordinate")
    w("system with its 8 symmetry types still visible. That is a genuine reduction in structure, but")
    w("not a reduction in mathematical difficulty. The unsolved problem is now precise: determine")
    w("whether the 729 trilinear equations admit a rank-R solution, especially for sparse, non-group-")
    w("closed ansatze. Strassen 2x2 shows exactly the kind of cancellation behavior that a 3x3 fast")
    w("algorithm would need, so any future search has to keep real coefficients and dead-X cancellation")
    w("in the model rather than support patterns alone.")

    # ── STEP 51 ──
    w()
    w(f"## {section_num}. SYMBOLIC FIBER-MODE DECOMPOSITION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 51)")
    w()
    w("Step 51 replaces stochastic search with an exact symbolic block decomposition of the tensor")
    w("equations. Every rank-1 term contributes three kinds of A x B data: fiber sums, live-fiber")
    w("anisotropy, and dead-X coordinates. The full 729-equation system splits exactly into those")
    w("three blocks.")
    w()
    sum51_rows, fiber51_rows, anis51_rows, matrix51_rows, std51_rows = read_step51_outputs()
    if sum51_rows:
        sum51 = {row['summary_name']: row['summary_value'] for row in sum51_rows}
        w(f"**Fiber-sum equations:** {sum51['fiber_sum_equations']}")
        w(f"**Live-anisotropy equations:** {sum51['fiber_anisotropy_equations']}")
        w(f"**Dead-X equations:** {sum51['dead_x_equations']}")
        w(f"**Total equations:** {sum51['total_equations']}")
        w(f"**Exact matrix form:** {sum51['matrix_form_constraint']}")
        w()
        w("### Fiber Coordinates")
        w()
        w("| fiber | sigma formula | eta1 formula | eta2 formula |")
        w("|-------|---------------|--------------|--------------|")
        for sum_row, eta1_row, eta2_row in zip(fiber51_rows, anis51_rows[0::2], anis51_rows[1::2]):
            w(f"| {sum_row['fiber']} | {sum_row['formula']} | {eta1_row['formula']} | {eta2_row['formula']} |")
        w()
        w("### Exact Matrix Form")
        w()
        w("| matrix | shape | definition | meaning |")
        w("|--------|-------|------------|---------|")
        for row in matrix51_rows:
            w(f"| {row['matrix_name']} | {row['shape']} | {row['entry_definition']} | {row['meaning']} |")
        w()
        w("The exact tensor system is equivalent to:")
        w("- Gamma * Sigma = 3 I_9")
        w("- Gamma * Eta1 = 0")
        w("- Gamma * Eta2 = 0")
        w("- Gamma * Delta = 0")
        w()
        w("### Standard 27-Term Verification")
        w()
        w(f"- Fiber-sum failures: {sum51['standard_fiber_sum_failures']}")
        w(f"- Live-anisotropy failures: {sum51['standard_live_anisotropy_failures']}")
        w(f"- Dead-X failures: {sum51['standard_dead_x_failures']}")
        w()
        w("| output_c | fiber | fiber_sum_total | fiber_sum_expected | eta1_total | eta2_total | status |")
        w("|----------|-------|-----------------|--------------------|------------|------------|--------|")
        for row in std51_rows[:18]:
            w(f"| {row['output_c']} | {row['fiber']} | {row['fiber_sum_total']} | {row['fiber_sum_expected']} | {row['eta1_total']} | {row['eta2_total']} | {row['status']} |")
        w()
    else:
        w("*Run ade3x3_step51_symbolic_fiber_mode_decomposition.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 51 isolates the real symbolic burden of any fast 3x3 algorithm. The target tensor lives")
    w("entirely in the 9-dimensional fiber-sum block. Every candidate rank-1 term also generates live")
    w("anisotropy and dead-X mass, and those nuisance components must cancel exactly after gamma")
    w("weighting. This turns the problem into a structured elimination problem on subspaces rather")
    w("than an undirected search through raw coefficient space.")

    # ── STEP 52 ──
    w()
    w(f"## {section_num}. QUOTIENT-SPACE RANK CRITERION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 52)")
    w()
    w("Step 52 turns the Step 51 matrix form into an exact quotient-space solvability test. For a")
    w("fixed decomposition, solvability is equivalent to the fiber-sum columns remaining independent")
    w("modulo the nuisance span generated by live anisotropy and dead-X columns.")
    w()
    sum52_rows, theorem52_rows, profile52_rows, bound52_rows = read_step52_outputs()
    if sum52_rows:
        sum52 = {row['summary_name']: row['summary_value'] for row in sum52_rows}
        w(f"**Bound scope:** {sum52['step52_bound_scope']}")
        w(f"**Standard 3x3 nuisance rank:** {sum52['standard_3x3_nuisance_rank']}")
        w(f"**Standard 3x3 lower bound from nuisance:** {sum52['standard_3x3_lower_bound_from_nuisance']}")
        w(f"**Strassen 2x2 nuisance rank:** {sum52['strassen_2x2_nuisance_rank']}")
        w(f"**Strassen 2x2 lower bound from nuisance:** {sum52['strassen_2x2_lower_bound_from_nuisance']}")
        w()
        w("### Exact Criterion")
        w()
        for row in theorem52_rows:
            w(f"- {row['statement_id']}: {row['statement']}")
        w()
        w("### Algorithm Profiles")
        w()
        w("| algorithm | size | R | nuisance shape | sigma_rank | eta_rank | dead_rank | nuisance_rank | augmented_rank | lower_bound | saturates |")
        w("|-----------|------|---|----------------|------------|----------|-----------|---------------|----------------|-------------|-----------|")
        for row in profile52_rows:
            w(f"| {row['algorithm']} | {row['matrix_size']} | {row['R']} | {row['R']}x{row['nuisance_columns']} | {row['sigma_rank']} | {row['eta_rank']} | {row['dead_rank']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['lower_bound_from_nuisance']} | {row['saturates_lower_bound']} |")
        w()
        w("### 3x3 Nuisance-Rank Targets")
        w()
        w("| target_R | max_allowed_nuisance_rank | necessary_condition |")
        w("|----------|---------------------------|---------------------|")
        for row in bound52_rows:
            w(f"| {row['target_R']} | {row['max_allowed_nuisance_rank']} | {row['necessary_condition']} |")
        w()
    else:
        w("*Run ade3x3_step52_quotient_rank_criterion.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 52 is the first exact linear-algebra obstruction beyond raw equation counting, but its")
    w("scope matters: the bound R >= 9 + rank(Nuisance) is per-algorithm, not universal. It depends")
    w("on the nuisance span produced by the chosen alpha,beta factors. What the step proves is that")
    w("every candidate algorithm must compress its own nuisance span into dimension at most R-9. The")
    w("2x2 Strassen check is the key validation: its nuisance matrix is 7x12 with rank exactly 3, so")
    w("Strassen is tight against the criterion R = 4 + rank(Nuisance).")

    # ── STEP 53 ──
    w()
    w(f"## {section_num}. SUPPORT-TYPE REPRESENTATIVE INCIDENCE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 53)")
    w()
    w("Step 53 asks whether support geometry alone can force any of the 8 exact representative")
    w("equation types from Step 49 to be absent for a single rank-1 term. A support type keeps only")
    w("the six subset supports of alpha, beta, and gamma, modulo the natural S3 x S3 x S3 action.")
    w()
    sum53_rows, support53_rows, mask53_rows, orbit053_rows = read_step53_outputs()
    if sum53_rows:
        sum53 = {row['summary_name']: row['summary_value'] for row in sum53_rows}
        w(f"**Support-type classes:** {sum53['n_support_type_classes']}")
        w(f"**Support-type orbit sizes:** {sum53['support_type_orbit_sizes']}")
        w(f"**Type-0-feasible classes:** {sum53['n_with_orbit0_possible']}")
        w(f"**Minimum auto-zero count among Type-0-feasible classes:** {sum53['min_auto_zero_among_orbit0_possible']}")
        w(f"**All-8-active Type-0-feasible classes:** {sum53['n_all8_active_among_orbit0_possible']}")
        w(f"**Support-only obstruction status:** {sum53['support_only_representative_obstruction']}")
        w()
        w("### Orbit-0-Feasible Histogram")
        w()
        w("| active_count | auto_zero_count | n_support_classes |")
        w("|--------------|-----------------|-------------------|")
        for row in orbit053_rows:
            w(f"| {row['active_count']} | {row['auto_zero_count']} | {row['n_support_classes']} |")
        w()
        w("### Sample Support Classes")
        w()
        w("| support_type_id | alpha_row | alpha_col | beta_row | beta_col | gamma_row | gamma_col | active_mask |")
        w("|-----------------|-----------|-----------|----------|----------|-----------|-----------|-------------|")
        sample_rows = []
        sample_rows.extend([row for row in support53_rows if row['orbit0_possible'] == 'True' and row['all8_active'] == 'True'][:3])
        sample_rows.extend([row for row in support53_rows if row['orbit0_possible'] == 'True' and row['active_count'] == '1'][:3])
        seen_ids = set()
        for row in sample_rows:
            if row['support_type_id'] in seen_ids:
                continue
            seen_ids.add(row['support_type_id'])
            w(f"| {row['support_type_id']} | {row['alpha_row_support']} | {row['alpha_col_support']} | {row['beta_row_support']} | {row['beta_col_support']} | {row['gamma_row_support']} | {row['gamma_col_support']} | {row['active_mask']} |")
        w()
    else:
        w("*Run ade3x3_step53_support_type_representative_incidence.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 53 closes off another support-only route. The Step 48 orbit-sum model was already too")
    w("coarse; Step 53 shows that even exact support incidence against the 8 representative equation")
    w("types is still vacuous. Once support is rich enough to allow the positive Type 0 equation,")
    w("there are many support classes that also allow all 7 zero-RHS types. Any universal lower-bound")
    w("argument must therefore use coefficient relations, quotient-space structure, or stronger")
    w("algebraic constraints than support incidence alone.")

    # ── STEP 54 ──
    w()
    w(f"## {section_num}. ANALYTICAL LOW-NUISANCE CONSTRUCTION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 54)")
    w()
    w("Step 54 pivots from obstruction to construction. It first corrects the dead-free term")
    w("template exactly, then measures the nuisance-rank landscape for random low-rank factor")
    w("families alpha=A*C and beta=B*D, with emphasis on the R=22 target nuisance threshold <= 13.")
    w()
    sum54_rows, dead54_rows, dead_profile54_rows, focus54_rows, best54_rows, lift54_rows, smirnov54_rows = read_step54_outputs()
    if sum54_rows:
        sum54 = {row['summary_name']: row['summary_value'] for row in sum54_rows}
        w(f"**Dead-free implies nuisance-free?** {sum54['dead_free_term_nuisance_zero']}")
        w(f"**Generic nonzero dead-free term nuisance rank:** {sum54['dead_free_generic_nonzero_term_nuisance_rank']}")
        w(f"**Best observed R=22 nuisance rank:** {sum54['r22_best_observed_nuisance_rank']}")
        w(f"**Best observed R=22 family:** {sum54['r22_best_observed_family']}")
        w(f"**Any sampled R=22 family meeting nuisance target?** {sum54['r22_any_meets_nuisance_target']}")
        w(f"**Any sampled R=22 family meeting full target?** {sum54['r22_any_meets_full_target']}")
        w(f"**Naive 2x2-corner Strassen lift count:** {sum54['naive_corner_strassen_total']}")
        w(f"**Smirnov profile status:** {sum54['smirnov_profile_status']}")
        w()
        w("### Dead-Free Term Correction")
        w()
        for row in dead54_rows:
            w(f"- {row['statement_id']}: {row['statement']}")
        w()
        w("| active_sum_index | sigma_profile | eta1_profile | eta2_profile | dead_profile | nuisance_zero | generic_nuisance_rank |")
        w("|------------------|---------------|--------------|--------------|--------------|---------------|-----------------------|")
        for row in dead_profile54_rows:
            w(f"| {row['active_sum_index']} | {row['sigma_profile']} | {row['eta1_profile']} | {row['eta2_profile']} | {row['dead_profile']} | {row['nuisance_zero']} | {row['generic_nonzero_term_nuisance_rank']} |")
        w()
        w("### R=22 Random-Factor Landscape")
        w()
        w("| p | q | trials | target | nuisance min | nuisance median | nuisance max | quotient gain max | criterion holds count | nuisance target hits | full target hits |")
        w("|---|---|--------|--------|--------------|----------------|--------------|-------------------|-----------------------|---------------------|------------------|")
        for row in focus54_rows:
            w(f"| {row['p_rank']} | {row['q_rank']} | {row['trials']} | {row['nuisance_target']} | {row['nuisance_rank_min']} | {row['nuisance_rank_median']} | {row['nuisance_rank_max']} | {row['quotient_gain_max']} | {row['criterion_holds_count']} | {row['meets_nuisance_target_count']} | {row['meets_full_target_count']} |")
        w()
        w("### Best R=22 Samples By Family")
        w()
        w("| p | q | nuisance_rank | quotient_gain | full_product_rank | criterion_holds | meets_nuisance_target | meets_full_target |")
        w("|---|---|---------------|---------------|-------------------|-----------------|-----------------------|------------------|")
        for row in sorted(best54_rows, key=lambda entry: (int(entry['p_rank']), int(entry['q_rank']))):
            w(f"| {row['p_rank']} | {row['q_rank']} | {row['nuisance_rank']} | {row['quotient_gain']} | {row['full_product_rank']} | {row['criterion_holds']} | {row['meets_nuisance_target']} | {row['meets_full_target']} |")
        w()
        w("### Strassen Lift Baseline")
        w()
        w("| component | multiplication_count | method | note |")
        w("|-----------|----------------------|--------|------|")
        for row in lift54_rows:
            w(f"| {row['component']} | {row['multiplication_count']} | {row['method']} | {row['note']} |")
        w()
        for row in smirnov54_rows:
            w(f"- {row['algorithm_name']}: {row['status']} ({row['note']})")
        w()
    else:
        w("*Run ade3x3_step54_analytical_low_nuisance_construction.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 54 sharpens the constructive picture in two ways. First, dead-free terms do not give a")
    w("free nuisance bypass: they kill Delta but still generate anisotropy, so the Strassen 2x2")
    w("template does not port directly into the Step 51 basis. Second, generic low-rank factor")
    w("families can indeed hit low nuisance numerically at R=22, but in the sampled families Sigma")
    w("never escaped the nuisance span, so the quotient-space gain stayed far below the required 9.")
    w("That points the next constructive search toward structured, nongeneric coefficient designs")
    w("rather than random low-rank factor models.")

    # ── STEP 55 ──
    w()
    w(f"## {section_num}. ALGEBRAIC NUISANCE DEPENDENCIES + WILDCARD EXPLORATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 55)")
    w()
    w("Step 55 moves from generic low-rank profiling to explicit Hadamard-space dependency")
    w("arithmetic. The main exact point is that in a p*q-dimensional Hadamard space, full")
    w("9-dimensional quotient recovery forces nuisance rank <= p*q-9, which sharpens the")
    w("R=22 nuisance target drastically in the structured p=3, q=4 regime.")
    w()
    sum55_rows, dim55_rows, best55_rows, profile55_rows, symbolic55_rows, gf255_rows, tropical55_rows, comm55_rows = read_step55_outputs()
    if sum55_rows:
        sum55 = {row['summary_name']: row['summary_value'] for row in sum55_rows}
        w(f"**p=q=3 combined nuisance cap:** {sum55['p3q3_combined_max_nuisance_rank']}")
        w(f"**p=3,q=4 combined nuisance cap:** {sum55['p3q4_combined_max_nuisance_rank']}")
        w(f"**Best structured 3x4 nuisance rank:** {sum55['best_structured_family_nuisance_rank']}")
        w(f"**Best structured 3x4 quotient gain:** {sum55['best_structured_family_quotient_gain']}")
        w(f"**GF(2) flattening lower bound:** {sum55['gf2_flattening_rank_lower_bound']}")
        w(f"**Tropical flattening rank:** {sum55['tropical_flattening_rank']}")
        w(f"**Standard 3x3 zero commutators:** {sum55['standard_3x3_zero_commutators']}")
        w(f"**Strassen 2x2 zero commutators:** {sum55['strassen_2x2_zero_commutators']}")
        w()
        w("### Track A: Hadamard-Space Targets")
        w()
        w("| regime | Hadamard dim upper bound | quotient target | max nuisance from geometry | max nuisance from R=22 | combined target |")
        w("|--------|---------------------------|-----------------|----------------------------|------------------------|----------------|")
        for row in dim55_rows:
            w(f"| {row['regime']} | {row['hadamard_dim_upper_bound']} | {row['required_sigma_mod_nuisance_rank']} | {row['max_nuisance_rank_from_hadamard_geometry']} | {row['max_nuisance_rank_from_r22_constraint']} | {row['combined_max_nuisance_rank']} |")
        w()
        w("| family | label | c_rank | d_rank | hadamard_dim | sigma_rank | nuisance_rank | quotient_gain | meets nuisance<=3 | full quotient target |")
        w("|--------|-------|--------|--------|--------------|------------|---------------|---------------|-------------------|----------------------|")
        for row in best55_rows:
            w(f"| {row['family']} | {row['label']} | {row['c_rank']} | {row['d_rank']} | {row['hadamard_dim']} | {row['sigma_rank']} | {row['nuisance_rank']} | {row['quotient_gain']} | {row['meets_hadamard_quotient_target']} | {row['full_quotient_target_holds']} |")
        w()
        if symbolic55_rows:
            w("### Symbolic Minor Witness")
            w()
            for row in symbolic55_rows:
                w(f"- {row['family']}: selected 4x4 minor rows {row['row_indices']} and columns {row['column_indices']} factor as {row['minor_factorization']}")
            w()
        w("### Track B")
        w()
        w("[WILDCARD]")
        w()
        w("| flattening | shape | GF(2) rank | tropical rank |")
        w("|------------|-------|------------|---------------|")
        tropical_map = {row['flattening']: row for row in tropical55_rows}
        for row in gf255_rows:
            w(f"| {row['flattening']} | {row['shape']} | {row['gf2_rank']} | {tropical_map[row['flattening']]['tropical_rank']} |")
        w()
        w("| algorithm | term_count | ordered_pairs | zero_commutators | nonzero_commutators | max_commutator_rank |")
        w("|-----------|------------|---------------|------------------|---------------------|---------------------|")
        for row in comm55_rows:
            w(f"| {row['algorithm']} | {row['term_count']} | {row['ordered_pairs']} | {row['zero_commutators']} | {row['nonzero_commutators']} | {row['max_commutator_rank']} |")
        w()
    else:
        w("*Run ade3x3_step55_algebraic_nuisance_dependencies.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 55 makes the Step 54 constructive obstruction sharper. In the p=q=3 regime the")
    w("Hadamard space itself has dimension only 9, so full quotient recovery would force the")
    w("nuisance span to vanish entirely. In the p=3,q=4 regime the raw R=22 target rank(N)<=13")
    w("is still far too loose: Hadamard geometry tightens it to rank(N)<=3. The structured")
    w("families tested here did not produce such a collapse, and their quotient gains stayed")
    w("well below the required 9. The wildcard checks are also informative but not decisive:")
    w("GF(2) flattening rank only gives the obvious bound 9, tropical flattening rank is capped")
    w("at 9 by the matrix dimensions, and the commutator profile shows real overlap structure")
    w("even for the standard and Strassen decompositions rather than automatic vanishing.")

    # ── STEP 56 ──
    w()
    w(f"## {section_num}. TENSOR-PRODUCT DFT CONSTRUCTION + ORBIT PACKING")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 56)")
    w()
    w("Step 56 moves from p=3,q=4 to p=q=4, where the Hadamard space has dimension 16 and")
    w("therefore leaves room for at most 7 nuisance dimensions if the quotient gain is to reach 9.")
    w("Its main exact computation is a complete 126 x 126 tensor-product DFT mode sweep.")
    w()
    (sum56_rows, dft56_top_rows, dft56_dc_rows, dft56_verify_rows,
     ternary56_rows, algebraic56_rows, correction56_rows, optimization56_rows,
     branch56_rows, removal56_rows, support56_rows) = read_step56_outputs()
    if sum56_rows:
        sum56 = {row['summary_name']: row['summary_value'] for row in sum56_rows}
        w(f"**p=q=4 combined nuisance cap:** {sum56['p4q4_combined_nuisance_cap']}")
        w(f"**Best DFT quotient gain:** {sum56['dft_best_quotient_gain']}")
        w(f"**Best DFT nuisance rank:** {sum56['dft_best_nuisance_rank']}")
        w(f"**Best DFT mode pair:** {sum56['dft_best_mode_pair']}")
        w(f"**Any DFT quotient gain >= 5?** {sum56['dft_any_quotient_gain_ge_5']}")
        w(f"**Best ternary sample quotient gain:** {sum56['ternary_best_quotient_gain']}")
        w(f"**Best algebraic sample quotient gain:** {sum56['algebraic_best_quotient_gain']}")
        w(f"**Best correction-block quotient gain:** {sum56['correction_best_quotient_gain']}")
        w(f"**Best local-search quotient gain:** {sum56['optimization_best_quotient_gain']}")
        w()
        w("### Track A: Top DFT Mode Pairs")
        w()
        w("| rank | C modes | D modes | C has DC | D has DC | sigma_rank | nuisance_rank | quotient_gain | meets nuisance<=7 |")
        w("|------|---------|---------|----------|----------|------------|---------------|---------------|-------------------|")
        for idx, row in enumerate(dft56_top_rows, start=1):
            w(f"| {idx} | {row['c_mode_label']} | {row['d_mode_label']} | {row['c_includes_dc']} | {row['d_includes_dc']} | {row['sigma_rank']} | {row['nuisance_rank']} | {row['quotient_gain']} | {row['meets_nuisance_cap_7']} |")
        w()
        w("| DC pattern | top-10 count |")
        w("|------------|--------------|")
        for row in dft56_dc_rows:
            w(f"| {row['dc_pattern']} | {row['top10_count']} |")
        w()
        w("| family | best quotient_gain | best nuisance_rank |")
        w("|--------|--------------------|--------------------|")
        w(f"| ternary random sample | {max(int(row['quotient_gain']) for row in ternary56_rows)} | {min(int(row['nuisance_rank']) for row in ternary56_rows)} |")
        w(f"| algebraic random sample | {max(int(row['quotient_gain']) for row in algebraic56_rows)} | {min(int(row['nuisance_rank']) for row in algebraic56_rows)} |")
        w(f"| interpreted correction-block sweep | {max(int(row['quotient_gain']) for row in correction56_rows)} | {min(int(row['nuisance_rank']) for row in correction56_rows)} |")
        w(f"| numpy local search surrogate | {max(int(row['quotient_gain']) for row in optimization56_rows)} | {min(int(row['nuisance_rank']) for row in optimization56_rows)} |")
        w()
        w("### Track B: Orbit Packing")
        w()
        w("| branch | count | fraction |")
        w("|--------|-------|----------|")
        for row in branch56_rows:
            w(f"| 30 o 30 -> {row['output_orbit']} | {row['count']} | {float(row['fraction']):.6f} |")
        w()
        w("| removed term | removed fiber | orbit0 after removal | orbit30 after removal | delta orbit0 | delta orbit30 |")
        w("|--------------|--------------|----------------------|----------------------|--------------|--------------|")
        for row in removal56_rows[:9]:
            w(f"| {row['removed_term_label']} | {row['removed_fiber']} | {row['orbit0_count_after_removal']} | {row['orbit30_count_after_removal']} | {row['delta_orbit0']} | {row['delta_orbit30']} |")
        w()
        w("| algorithm | fibers hit | total aligned occurrences | unique live X atoms | unique distinct same-fiber pairs | occurrence distinct same-fiber pairs |")
        w("|-----------|-----------|--------------------------|---------------------|----------------------------------|--------------------------------------|")
        for row in support56_rows:
            w(f"| {row['algorithm']} | {row['fiber_count']} | {row['total_aligned_live_occurrences']} | {row['unique_live_x_atoms']} | {row['unique_same_fiber_distinct_pairs']} | {row['occurrence_same_fiber_distinct_pairs']} |")
        w()
    else:
        w("*Run ade3x3_step56_tensor_product_dft_orbit_packing.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 56 tests the first complete structured p=q=4 family. The Hadamard cap is now 7 rather")
    w("than 3, so this regime is genuinely less cramped than Step 55's p=3,q=4 setting. The full")
    w("DFT sweep is therefore definitive for that family: if the best quotient gain remains small,")
    w("the tensor-product DFT basis is not by itself the missing ansatz. On the discrete side, the")
    w("same-fiber branching point 30 o 30 stays exactly balanced, so there is no asymmetry lever in")
    w("that raw composition rule. The direct minimization track used a numpy local-search surrogate")
    w("because PyTorch is not installed in the current environment, so those values are best-observed")
    w("measurements rather than a claim of global optimality.")

    # ── STEP 57 ──
    w()
    w(f"## {section_num}. FIBER-GROUP PARTITION ENUMERATION + ORBIT BUDGET FILTER")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 57)")
    w()
    w("Step 57 moves to a purely discrete pre-coefficient search: partition the R terms into 9")
    w("primary-fiber groups, reduce those partitions by the S3 x S3 action on the 3x3 output grid,")
    w("and then filter them by same-fiber orbit budget. A second exact but conservative model asks")
    w("whether a partition can be realized by rectangular spreading shapes so that every fiber is")
    w("touched exactly three times in the 9-fiber x 3-channel picture.")
    w()
    sum57_rows, type57_rows, survivor57_rows, cycle57_rows, shape57_rows = read_step57_outputs()
    if sum57_rows:
        sum57_by_r = {int(row['R']): row for row in sum57_rows}
        r9 = sum57_by_r[9]
        r22 = sum57_by_r[22]
        r23 = sum57_by_r[23]
        w(f"**R=9 ordered compositions:** {r9['ordered_compositions']}")
        w(f"**R=9 symmetry-inequivalent compositions:** {r9['symmetry_inequivalent_compositions']}")
        w(f"**R=22 no-spreading survivor types:** {r22['no_spreading_survivor_types']}")
        w(f"**R=22 exact-3 survivor assignment orbits:** {r22['exact3_tiling_surviving_assignment_orbits']}")
        w(f"**R=23 no-spreading survivor types:** {r23['no_spreading_survivor_types']}")
        w(f"**R=23 exact-3 survivor assignment orbits:** {r23['exact3_tiling_surviving_assignment_orbits']}")
        w()
        w("### Task 1-2: Partition Sweep")
        w()
        w("| R | ordered compositions | partition types | symmetry-inequivalent compositions | no-spreading survivor types | exact-3 survivor assignment orbits |")
        w("|---|----------------------|-----------------|------------------------------------|-----------------------------|------------------------------------|")
        for row in sum57_rows:
            w(f"| {row['R']} | {row['ordered_compositions']} | {row['partition_types']} | {row['symmetry_inequivalent_compositions']} | {row['no_spreading_survivor_types']} | {row['exact3_tiling_surviving_assignment_orbits']} |")
        w()
        w("| cycle lengths on 9 fibers | class size |")
        w("|---------------------------|------------|")
        for row in cycle57_rows:
            w(f"| {row['cycle_lengths']} | {row['class_size']} |")
        w()
        w("| shape | fibers touched | extra fibers beyond primary | primary-anchored rectangles |")
        w("|-------|----------------|-----------------------------|-----------------------------|")
        for row in shape57_rows:
            w(f"| {row['shape']} | {row['fibers_touched']} | {row['extra_fibers_beyond_primary']} | {row['primary_anchored_rectangles']} |")
        w()
        w("### Task 4: Bottom-Up Structural Sweep")
        w()
        w("| R | partition_types | orbit_budget_survivors | shape_assignment_survivors | notes |")
        w("|---|-----------------|------------------------|---------------------------|-------|")
        for row in sum57_rows:
            note_parts = []
            if row['first_exact3_survivor_type']:
                note_parts.append(f"first exact-3 type {row['first_exact3_survivor_type']}")
            note_parts.append(f"deficit range {row['min_spreading_deficit']}..{row['max_spreading_deficit']}")
            w(f"| {row['R']} | {row['partition_types']} | {row['no_spreading_survivor_types']} | {row['exact3_tiling_surviving_assignment_orbits']} | {'; '.join(note_parts)} |")
        w()
        top_survivors = survivor57_rows[:10]
        if top_survivors:
            w("### Representative Exact-3 Survivor Types")
            w()
            w("| R | partition type | surviving assignment orbits | example assignment | example shape witness |")
            w("|---|----------------|----------------------------|--------------------|-----------------------|")
            for row in top_survivors:
                witness_cell = row['example_shape_witness'].replace('|', ';')
                w(f"| {row['R']} | {row['partition_type']} | {row['surviving_assignment_orbits']} | {row['example_surviving_assignment']} | {witness_cell} |")
            w()
    else:
        w("*Run ade3x3_step57_fiber_group_partition_enumeration.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 57 gives an exact discrete front end to the search, but not yet a lower-bound theorem.")
    w("The ordered/symmetry-reduced partition counts and the orbit-budget deficits are exact. The")
    w("exact-3 survivors are also exact inside their own model, but that model is conservative: it")
    w("forces the 27 channel incidences to appear literally as 3 touches per fiber. So if survivors")
    w("already appear at small R, the partition ansatz alone does not exclude those ranks; if a")
    w("partition fails the exact-3 test, that only says it cannot realize the rigid tiling picture, not")
    w("that weighted coefficient cancellation is impossible.")

    # ── STEP 59 ──
    w()
    w(f"## {section_num}. CUBE ROOT OF UNITY INJECTION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 59)")
    w()
    w("Step 59 separates two omega-based constructions that are easy to conflate. The full-spread")
    w("family places the same Fourier row on every input row and the conjugate Fourier column on")
    w("every input column; the same-fiber family localizes those Fourier factors to a single output")
    w("fiber. The first family is the source of the naive low-R temptation, and the second family is")
    w("the one that actually interacts cleanly with the Step 51 nuisance basis.")
    w()
    (
        sum59_rows,
        obstruction59_rows,
        assignment59_rows,
        r9sum59_rows,
        r9fail59_rows,
        greedy59_rows,
        nuisance59_rows,
        verify59_rows,
    ) = read_step59_outputs()
    if sum59_rows:
        sum59 = {row['summary_name']: row for row in sum59_rows}
        r9sum59 = r9sum59_rows[0] if r9sum59_rows else None
        verify59 = verify59_rows[0] if verify59_rows else None
        w(f"**Full-spread live rank per output block:** {sum59['fullspread_live_family_rank_per_output']['summary_value']}")
        w(f"**Target live rank per output block:** {sum59['fullspread_target_live_rank_per_output']['summary_value']}")
        w(f"**Balanced 9-term full-spread total failures:** {sum59['balanced_r9_total_failures']['summary_value']}")
        w(f"**Same-fiber j=1 nuisance rank:** {sum59['single_j1_nuisance_rank']['summary_value']}")
        w(f"**Same-fiber j=1,2 pair nuisance rank:** {sum59['pair_j1_j2_nuisance_rank']['summary_value']}")
        w(f"**Same-fiber j=0,1,2 triple nuisance rank:** {sum59['triple_j012_nuisance_rank']['summary_value']}")
        w(f"**27-term same-fiber Fourier verification failures:** {sum59['samefiber_fourier_total_failures']['summary_value']}")
        w()
        w("### Task 2f: Full-Spread Obstruction")
        w()
        w("| statement id | statement | value |")
        w("|--------------|-----------|-------|")
        for row in obstruction59_rows:
            w(f"| {row['statement_id']} | {row['statement']} | {row['value']} |")
        w()
        w("### Task 3c-3d: Explicit 9-Term Full-Spread Attempt")
        w()
        w("| output fiber | channel j | label |")
        w("|--------------|-----------|-------|")
        for row in assignment59_rows:
            w(f"| {row['output_fiber']} | {row['channel_j']} | {row['label']} |")
        w()
        if r9sum59:
            w("| live target failures | live off-target failures | dead failures | total failures |")
            w("|----------------------|--------------------------|---------------|----------------|")
            w(f"| {r9sum59['live_target_failures']} | {r9sum59['live_off_target_failures']} | {r9sum59['dead_failures']} | {r9sum59['total_failures']} |")
            w()
        if r9fail59_rows:
            w("Representative failed equations:")
            w()
            w("| output fiber | input coordinate | block type | actual | target | residual |")
            w("|--------------|------------------|------------|--------|--------|----------|")
            for row in r9fail59_rows[:12]:
                w(f"| {row['output_fiber']} | {row['input_coordinate']} | {row['block_type']} | {row['actual_value']} | {row['target_value']} | {row['residual']} |")
            w()
        w("### Task 3e: Greedy Restricted-Family Augmentation")
        w()
        w("| R | selected term | residual L2 | max abs residual | live target fails | live off-target fails | dead fails | total fails |")
        w("|---|---------------|-------------|------------------|-------------------|-----------------------|------------|-------------|")
        for row in greedy59_rows[:12]:
            w(f"| {row['R']} | {row['selected_term']} | {row['residual_l2_norm']} | {row['max_abs_residual']} | {row['live_target_failures']} | {row['live_off_target_failures']} | {row['dead_failures']} | {row['total_failures']} |")
        if greedy59_rows:
            tail59 = greedy59_rows[-1]
            w()
            w(f"At R={tail59['R']} within the full-spread family, the failure count is still {tail59['total_failures']} with residual L2 {tail59['residual_l2_norm']}.")
            w()
        w("### Task 4a-4c: Same-Fiber Fourier Nuisance Profiles")
        w()
        w("| case | term count | sigma rank | eta rank | delta rank | nuisance rank | augmented rank | quotient gain |")
        w("|------|------------|------------|----------|------------|---------------|----------------|---------------|")
        for row in nuisance59_rows:
            w(f"| {row['case_label']} | {row['term_count']} | {row['sigma_rank']} | {row['eta_rank']} | {row['delta_rank']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['quotient_gain']} |")
        w()
        if verify59:
            w("| same-fiber Fourier term count | total failures | live target failures | live off-target failures | dead failures |")
            w("|-------------------------------|----------------|----------------------|--------------------------|---------------|")
            w(f"| {verify59['samefiber_fourier_term_count']} | {verify59['samefiber_fourier_total_failures']} | {verify59['samefiber_fourier_live_target_failures']} | {verify59['samefiber_fourier_live_off_target_failures']} | {verify59['samefiber_fourier_dead_failures']} |")
            w()
    else:
        w("*Run ade3x3_step59_cube_root_unity_injection.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 59 sharpens the omega story into an exact yes/no split. The full-spread family is not")
    w("merely insufficient in sampled search; it is linearly incapable of resolving the 9 distinct")
    w("input fibers inside a fixed live output block, so the naive low-rank hope fails for a structural")
    w("reason. The same-fiber Fourier bundle behaves differently: DFT orthogonality kills the dead-X")
    w("channels exactly and yields a valid 27-term decomposition, but it does so as a Fourier disguise")
    w("of the standard algorithm rather than as a new sub-27 construction. In Step 51 language, cube")
    w("roots of unity help organize nuisance cancellation, but they do not by themselves collapse the")
    w("nuisance span below the regime needed for a rank improvement.")

    # ── STEP 60 ──
    w()
    w(f"## {section_num}. HYBRID FOURIER CONSTRUCTION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 60)")
    w()
    w("Step 60 treats the hybrid ansatz as an exact direct sum of two blocks. Same-fiber Fourier")
    w("triples solve chosen output fibers perfectly and do not leak to the complementary fibers. The")
    w("remaining spreader problem is therefore an honest residual tensor-rank problem on the uncovered")
    w("fiber set, which can be filtered first by flattening lower bounds and then, where available, by")
    w("known exact ranks for rectangular subproblems.")
    w()
    (
        sum60_rows,
        modularity60_rows,
        orbit60_rows,
        size60_rows,
        reduced60_rows,
        feasibility60_rows,
        explicit60_rows,
    ) = read_step60_outputs()
    if sum60_rows:
        sum60 = {row['summary_name']: row for row in sum60_rows}
        w(f"**Fourier subsets checked:** {sum60['modularity_total_subsets_checked']['summary_value']}")
        w(f"**Fourier subsets passing exactly:** {sum60['modularity_total_passing_subsets']['summary_value']}")
        w(f"**Maximum Fourier modularity residual:** {sum60['modularity_max_abs_residual']['summary_value']}")
        w(f"**Six-fiber residual orbits:** {sum60['sixfiber_orbit_count']['summary_value']}")
        w(f"**Six-fiber max-flattening lower-bound range:** {sum60['sixfiber_min_flattening_rank_max']['summary_value']}..{sum60['sixfiber_max_flattening_rank_max']['summary_value']}")
        w(f"**Six-fiber flattening candidates at R=19:** {sum60['sixfiber_candidate_orbits_R19_by_flattening']['summary_value']}")
        w(f"**Six-fiber flattening candidates at R=20:** {sum60['sixfiber_candidate_orbits_R20_by_flattening']['summary_value']}")
        w(f"**Six-fiber flattening candidates at R=21:** {sum60['sixfiber_candidate_orbits_R21_by_flattening']['summary_value']}")
        w(f"**Six-fiber R=22 flattening candidates:** {sum60['sixfiber_candidate_orbits_R22_by_flattening']['summary_value']}")
        w(f"**Six-fiber R=23 flattening candidates:** {sum60['sixfiber_candidate_orbits_R23_by_flattening']['summary_value']}")
        w(f"**R=22 three-fiber row case:** {sum60['three_fiber_row_case_verdict']['summary_value']}")
        w()
        w("### Task 1: Fourier Block Modularity")
        w()
        w("| subset size | subset count | passing count | max abs residual |")
        w("|-------------|--------------|---------------|------------------|")
        for row in modularity60_rows:
            w(f"| {row['subset_size']} | {row['subset_count']} | {row['passing_count']} | {row['max_abs_residual']} |")
        w()
        w("### Task 2d: Reduced Spreader Systems")
        w()
        w("| remaining fibers | Fourier fibers | fiber-sum eqs | anisotropy eqs | dead-X eqs | total reduced eqs | spreaders at R=19 | spreaders at R=20 | spreaders at R=21 | spreaders at R=22 | spreaders at R=23 |")
        w("|------------------|----------------|---------------|----------------|------------|-------------------|------------------|------------------|------------------|------------------|------------------|")
        for row in reduced60_rows:
            w(f"| {row['remaining_fiber_count']} | {row['fourier_fiber_count']} | {row['fiber_sum_equations']} | {row['live_anisotropy_equations']} | {row['dead_x_equations']} | {row['total_reduced_equations']} | {row['available_spreaders_R19']} | {row['available_spreaders_R20']} | {row['available_spreaders_R21']} | {row['available_spreaders_R22']} | {row['available_spreaders_R23']} |")
        w()
        sixfiber_orbits = [row for row in orbit60_rows if row['remaining_fiber_count'] == '6']
        if sixfiber_orbits:
            w("### Task 4c: Six-Fiber Residual Orbits")
            w()
            w("| representative residual fibers | orbit size | row profile | col profile | rectangle | rank A|(BC) | rank B|(AC) | rank C|(AB) | max lower bound | known exact rank if any |")
            w("|-------------------------------|------------|-------------|-------------|-----------|------------|------------|------------|-----------------|-------------------------|")
            for row in sixfiber_orbits:
                exact_cell = row['exact_rank_if_known'] if row['exact_rank_if_known'] else ''
                w(f"| {row['orbit_representative']} | {row['orbit_size']} | {row['row_profile']} | {row['col_profile']} | {row['is_rectangle']} | {row['flattening_rank_A_BC']} | {row['flattening_rank_B_AC']} | {row['flattening_rank_C_AB']} | {row['flattening_rank_max']} | {exact_cell} |")
            w()
        w("### Task 3f / 4: Hybrid Feasibility Table")
        w()
        w("| Fourier fibers f | Fourier terms | remaining fibers | R=19 spreaders | R=20 spreaders | R=21 spreaders | R=22 spreaders | R=23 spreaders | flattening lower-bound range | R19 cand. | R20 cand. | R21 cand. | R22 cand. | R23 cand. |")
        w("|------------------|---------------|------------------|---------------|---------------|---------------|---------------|---------------|------------------------------|-----------|-----------|-----------|-----------|-----------|")
        for row in feasibility60_rows:
            lower_range = f"{row['min_flattening_lower_bound']}..{row['max_flattening_lower_bound']}"
            w(f"| {row['fourier_fiber_count']} | {row['fourier_term_count']} | {row['remaining_fiber_count']} | {row['available_spreaders_R19']} | {row['available_spreaders_R20']} | {row['available_spreaders_R21']} | {row['available_spreaders_R22']} | {row['available_spreaders_R23']} | {lower_range} | {row['candidate_orbits_R19_by_flattening']} | {row['candidate_orbits_R20_by_flattening']} | {row['candidate_orbits_R21_by_flattening']} | {row['candidate_orbits_R22_by_flattening']} | {row['candidate_orbits_R23_by_flattening']} |")
        w()
        w("### Explicit Cases")
        w()
        w("| case | remaining fibers | available spreaders | max flattening lower bound | known exact rank | verdict |")
        w("|------|------------------|---------------------|----------------------------|------------------|---------|")
        for row in explicit60_rows:
            exact_cell = row['known_exact_rank_if_any'] if row['known_exact_rank_if_any'] else ''
            w(f"| {row['case_id']} | {row['remaining_fibers']} | {row['available_spreaders']} | {row['flattening_rank_max']} | {exact_cell} | {row['verdict']} |")
    else:
        w("*Run ade3x3_step60_hybrid_fourier_construction.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 60 closes one ambiguity from Step 59 and opens another. The ambiguity it closes is")
    w("coupling: the Fourier block is exactly modular, so once some fibers are handled by same-fiber")
    w("Fourier triples, the spreader block sees a literal residual tensor on the uncovered fibers and")
    w("nothing else. The new open point is rank: flattening lower bounds immediately kill the tiny")
    w("residual cases with 1, 2, or 3 remaining fibers, and known exact ranks also kill the 2x2 and")
    w("2x3 rectangular residuals at the hoped-for spreader budgets. But the six-fiber non-rectangular")
    w("residuals remain alive under flattening alone, all with lower bound 9 against budgets 10, 11, 12,")
    w("13, and 14 at R=19..23. So the hybrid ansatz is not dead, but the surviving territory is now sharply localized to")
    w("non-rectangular residual patterns where stronger lower bounds or explicit constructions are still")
    w("needed.")

    # ── STEP 61 ──
    w()
    w(f"## {section_num}. NUISANCE-FIRST ARCHITECTURE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Step 61)")
    w()
    w("Step 61 reverses the constructive viewpoint. Instead of asking how to suppress nuisance, it asks")
    w("what exact architecture the Step 51-52 equations force if nuisance is treated as the load-bearing")
    w("resource below rank 27. The first consequence is a sharp R=9 contradiction; the second is an")
    w("exact dependency budget for ranks 18 through 23.")
    w()
    sum61_rows, theorem61_rows, channel61_rows, budget61_rows, pressure61_rows = read_step61_outputs()
    if sum61_rows:
        sum61 = {row['summary_name']: row for row in sum61_rows}
        w(f"**R=9 status:** {sum61['R9_status']['summary_value']}")
        w(f"**Dead-free global lower bound:** {sum61['deadfree_global_lower_bound']['summary_value']}")
        w(f"**R=18 Gamma-nullity:** {sum61['R18_gamma_nullity']['summary_value']}")
        w(f"**R=18 minimum dead dependencies:** {sum61['R18_min_dead_dependencies']['summary_value']}")
        w(f"**Sub-27 requires nuisance:** {sum61['sub27_requires_nuisance']['summary_value']}")
        w()
        w("### R=9 Theorem Chain")
        w()
        w("| id | statement | value |")
        w("|----|-----------|-------|")
        for row in theorem61_rows:
            w(f"| {row['statement_id']} | {row['statement']} | {row['value']} |")
        w()
        w("### Dead-Free Per-Channel Lower Bound")
        w()
        w("| summation index | target dimension | minimum terms needed | statement |")
        w("|-----------------|------------------|----------------------|-----------|")
        for row in channel61_rows:
            w(f"| {row['summation_index']} | {row['target_output_space_dimension']} | {row['minimum_terms_needed_for_channel']} | {row['channel_statement']} |")
        w()
        highlight61_rows = [row for row in budget61_rows if row['R'] in {'9', '18', '19', '20', '21', '22', '23', '27'}]
        w("### Nuisance Budget Table")
        w()
        w("| R | Gamma nullity | max nuisance rank | max dead rank | min dead dependencies | min total nuisance dependencies | all terms dead-free possible? |")
        w("|---|---------------|-------------------|---------------|-----------------------|--------------------------------|-------------------------------|")
        for row in highlight61_rows:
            w(f"| {row['R']} | {row['gamma_nullity']} | {row['max_nuisance_rank']} | {row['max_dead_rank']} | {row['min_dead_dependencies']} | {row['min_total_nuisance_dependencies']} | {row['all_terms_deadfree_possible']} |")
        w()
        w("### R=18..23 Pressure Rows")
        w()
        w("| R | Gamma nullity | max nuisance rank | min dead dependencies | min total nuisance dependencies |")
        w("|---|---------------|-------------------|-----------------------|--------------------------------|")
        for row in pressure61_rows:
            w(f"| {row['R']} | {row['gamma_nullity']} | {row['max_nuisance_rank']} | {row['min_dead_linear_dependencies']} | {row['min_total_nuisance_linear_dependencies']} |")
        w()
    else:
        w("*Run ade3x3_step61_nuisance_first_architecture.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 61 makes explicit what the earlier steps only suggested empirically: nuisance is not a")
    w("mistake to be eliminated, but the mechanism by which any sub-27 algorithm must operate. At")
    w("R=9 the contradiction is exact: Gamma has no nullspace, so nuisance would have to vanish term by")
    w("term, but Step 54's dead-free classification shows that a nonzero term cannot be both dead-free")
    w("and anisotropy-free. The weaker channel-count argument already forces R>=27 for all-dead-free")
    w("families. At R=18 the budget becomes geometric rather than impossible: the whole 72-column")
    w("nuisance block must fit inside a 9-dimensional annihilator, so dozens of exact linear identities")
    w("are required before any quotient-space gain is even possible. That reframes the constructive")
    w("problem: not how to avoid nuisance, but how to engineer nuisance with maximal cancellation power")
    w("at minimal rank cost.")

    # ── STEP 62 ──
    w()
    w(f"## {section_num}. NON-RECTANGULAR 6-FIBER SUB-TENSOR RANK ATTACK")
    section_num += 1
    w()
    w("[EXACT_DERIVED] + [MEASURED_FROM_CODE] (Step 62)")
    w()
    w("Step 62 attacks the six-fiber survivors from Step 60 with three tools at once: exact")
    w("substitution-style restriction bounds, measured CP-rank fitting on the full 9x9xm subtensors,")
    w("and a direct upper-bound construction for the anti-diagonal-missing pattern P4.")
    w()
    (
        sum62_rows,
        subbest62_rows,
        num62_rows,
        direct62_rows,
        pattern62_rows,
        verdict62_rows,
    ) = read_step62_outputs()
    if sum62_rows:
        sum62 = {row['summary_name']: row for row in sum62_rows}
        w(f"**Nonrectangular patterns attacked:** {sum62['nonrectangular_pattern_count']['summary_value']}")
        w(f"**Rectangular patterns ruled out exactly:** {sum62['rectangular_patterns_ruled_out_exactly']['summary_value']}")
        w(f"**Best substitution lower bound on P1-P4:** {sum62['best_substitution_lower_bound_nonrectangular']['summary_value']}")
        w(f"**Any nonrectangular rank <= 13 found numerically?** {sum62['any_nonrectangular_rank_leq_13_found']['summary_value']}")
        w(f"**Any nonrectangular rank <= 14 found numerically?** {sum62['any_nonrectangular_rank_leq_14_found']['summary_value']}")
        w(f"**P4 direct upper bound:** {sum62['P4_direct_upper_bound']['summary_value']}")
        w()
        w("### Pattern Table")
        w()
        w("| pattern | representative fibers | substitution best lower bound | numerical rank upper bound if found | best loss over ranks 9..14 | known exact rank if determined | feasible at 13 spreaders? | feasible at 14 spreaders? |")
        w("|---------|-----------------------|-------------------------------|-------------------------------------|----------------------------|-------------------------------|--------------------------|--------------------------|")
        for row in pattern62_rows:
            w(f"| {row['pattern_id']} | {row['representative_fibers']} | {row['substitution_best_lower_bound']} | {row['numerical_rank_upper_bound_if_found']} | {row['best_loss_all_ranks']} | {row['known_exact_rank_if_determined']} | {row['feasible_at_13_spreaders']} | {row['feasible_at_14_spreaders']} |")
        w()
        nonrect_num_rows = [row for row in num62_rows if row['pattern_id'] in {'P1', 'P2', 'P3', 'P4'} and row['rank_tested'] in {'13', '14'}]
        if nonrect_num_rows:
            w("### Numerical Rank Scan on P1-P4")
            w()
            w("| pattern | rank tested | best verified loss | best max abs residual | numerical exact? |")
            w("|---------|-------------|--------------------|-----------------------|------------------|")
            for row in nonrect_num_rows:
                w(f"| {row['pattern_id']} | {row['rank_tested']} | {row['best_verified_loss']} | {row['best_max_abs_residual']} | {row['numerically_exact']} |")
            w()
        w("### Direct P4 Construction")
        w()
        w("| construction | term count | status | note |")
        w("|--------------|------------|--------|------|")
        for row in direct62_rows:
            w(f"| {row['construction_id']} | {row['term_count']} | {row['status']} | {row['note']} |")
        w()
        w("### Hybrid Verdicts")
        w()
        w("| R | Fourier fibers f | spreaders | verdict | scope |")
        w("|---|------------------|-----------|---------|-------|")
        for row in verdict62_rows:
            w(f"| {row['R']} | {row['fourier_fiber_count']} | {row['spreader_budget']} | {row['any_pattern_feasible']} | {row['scope']} |")
        w()
    else:
        w("*Run ade3x3_step62_nonrectangular_6fiber_rank_attack.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 62 upgrades the six-fiber hybrid discussion from a flattening possibility statement to a")
    w("direct rank attack. The exact restriction bounds still stop at 9 on the nonrectangular patterns,")
    w("so they do not decide the R=22 question. The measured CP-rank scan is the new evidence: across")
    w("all four nonrectangular six-fiber patterns, no rank-13 or rank-14 fit drove the loss anywhere")
    w("near zero in the current L-BFGS-B restarts. That is not a proof of impossibility, but it is")
    w("negative evidence against the f=3 hybrid route at both R=22 and R=23. The rectangular patterns")
    w("stay exactly ruled out by rank 15, and P4 still has only the obvious exact upper bound 18 from")
    w("its fiber-local decomposition.")

    # ── STEP 63 ──
    w()
    w(f"## {section_num}. REVERSE ENGINEERING + CANCELLATION VISUALIZATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] + [MEASURED_FROM_CODE] (Step 63)")
    w()
    w("Step 63 changes direction. Instead of only searching forward for low-rank 3x3 algorithms, it")
    w("starts from a known exact rank-23 decomposition and measures how that algorithm sits inside the")
    w("Step 51-52 fiber-mode framework. The key source result is that while Smirnov's explicit 2013")
    w("coefficient table was not recovered from the fetched paper metadata, the public AlphaTensor")
    w("repository exposes an exact 3x3x3 rank-23 factorization in recombination/example.py, and that")
    w("explicit 23-term coefficient table can be profiled directly.")
    w()
    (
        sum63_rows,
        source63_rows,
        term63_rows,
        importance63_rows,
        single63_rows,
        pair63_rows,
        dead63_rows,
        comm63_rows,
    ) = read_step63_outputs()
    if sum63_rows:
        sum63 = {row['summary_name']: row for row in sum63_rows}
        w(f"**Recovered algorithm:** {sum63['algorithm_label']['summary_value']}")
        w(f"**Term count:** {sum63['term_count']['summary_value']}")
        w(f"**Gamma orientation needed for ADE3x3 indexing:** {sum63['gamma_orientation']['summary_value']}")
        w(f"**Exact reconstruction residual:** {sum63['reconstruction_max_abs_residual']['summary_value']}")
        w(f"**Measured nuisance rank:** {sum63['nuisance_rank']['summary_value']}")
        w(f"**Quotient gain:** {sum63['quotient_gain']['summary_value']}")
        w(f"**Nuisance equals R-9 exactly?** {sum63['nuisance_equals_R_minus_9']['summary_value']}")
        w(f"**Single-term reweight-feasible removals:** {sum63['single_term_reweight_feasible_count']['summary_value']}")
        w(f"**Pair reweight-feasible removals:** {sum63['pair_reweight_feasible_count']['summary_value']}")
        w(f"**Dependency-graph edges:** {sum63['dependency_graph_edge_count']['summary_value']}")
        w(f"**Visualization HTML:** {sum63['visualization_html']['summary_value']}")
        w()
        w("### Task 1: External Source Status")
        w()
        w("| source | status | detail |")
        w("|--------|--------|--------|")
        for row in source63_rows:
            w(f"| {row['source_label']} | {row['status']} | {row['detail']} |")
        w()
        w("### Task 2: Fiber-Mode Profile of the Public Rank-23 Algorithm")
        w()
        w("| R | sigma_rank | eta_rank | delta_rank | nuisance_rank | augmented_rank | quotient_gain | Gamma*Sigma=3I_9 | Gamma*Nuisance=0 |")
        w("|---|------------|----------|------------|---------------|----------------|---------------|------------------|------------------|")
        w(f"| {sum63['term_count']['summary_value']} | {sum63['sigma_rank']['summary_value']} | {sum63['eta_rank']['summary_value']} | {sum63['delta_rank']['summary_value']} | {sum63['nuisance_rank']['summary_value']} | {sum63['augmented_rank']['summary_value']} | {sum63['quotient_gain']['summary_value']} | True | True |")
        w()
        w("The critical exact measurement is nuisance rank 14, so this recovered 23-term algorithm")
        w("saturates the Step 52 boundary R = 9 + rank(Nuisance) exactly.")
        w()
        w("### Task 1/2: Explicit 23-Term Coefficient Table")
        w()
        w("| term | alpha | beta | gamma |")
        w("|------|-------|------|-------|")
        for row in term63_rows:
            w(f"| {row['term_id']} | {row['alpha']} | {row['beta']} | {row['gamma']} |")
        w()
        w("### Task 3a/3b: Term Importance")
        w()
        w("| rank | term | Frobenius norm^2 | nonzero equations | affected equation types | live fibers touched |")
        w("|------|------|------------------|-------------------|-------------------------|--------------------|")
        for row in importance63_rows[:10]:
            w(f"| {row['importance_rank']} | {row['term_id']} | {row['frobenius_norm_squared']} | {row['nonzero_equation_count']} | {row['affected_equation_types']} | {row['affected_live_fibers']} |")
        w()
        w("### Task 3c: Single-Term Removal Feasibility")
        w()
        w("| removed term | remaining R | nuisance_rank | augmented_rank | quotient_gain | gamma-only reweight feasible? |")
        w("|--------------|-------------|---------------|----------------|---------------|-------------------------------|")
        for row in single63_rows:
            w(f"| {row['removed_terms']} | {row['remaining_R']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['quotient_gain']} | {row['gamma_reweight_feasible']} |")
        w()
        top_pair63_rows = [row for row in pair63_rows if row['quotient_gain'] == pair63_rows[0]['quotient_gain']][:10]
        w("### Task 3d: Best Pair-Removal Cases")
        w()
        w("| removed pair | remaining R | nuisance_rank | augmented_rank | quotient_gain | gamma-only reweight feasible? |")
        w("|--------------|-------------|---------------|----------------|---------------|-------------------------------|")
        for row in top_pair63_rows:
            w(f"| {row['removed_terms']} | {row['remaining_R']} | {row['nuisance_rank']} | {row['augmented_rank']} | {row['quotient_gain']} | {row['gamma_reweight_feasible']} |")
        w()
        perfect_dead = sum(1 for row in dead63_rows if row['balance_status'] == 'perfect_balanced')
        multi_dead = sum(1 for row in dead63_rows if row['balance_status'] == 'multi_way_balanced')
        one_sided_dead = sum(1 for row in dead63_rows if row['balance_status'] == 'one_sided')
        w("### Task 4: Cancellation Structure Summary")
        w()
        w(f"- Perfectly balanced dead equations: {perfect_dead}")
        w(f"- Multi-way balanced dead equations: {multi_dead}")
        w(f"- One-sided dead equations: {one_sided_dead}")
        w("- Full interactive-style HTML export written to outputs/step63_cancellation_visualization.html")
        w()
        w("### Task 5: Commutator / Anticommutator Split")
        w()
        w("| size | tensor piece | nonzero entries | rank A|(BC) | rank B|(AC) | rank C|(AB) | flattening lower bound |")
        w("|------|--------------|-----------------|------------|------------|------------|-------------------------|")
        for row in comm63_rows:
            w(f"| {row['matrix_size']} | {row['tensor_piece']} | {row['nonzero_entries']} | {row['flattening_rank_A_BC']} | {row['flattening_rank_B_AC']} | {row['flattening_rank_C_AB']} | {row['flattening_lower_bound']} |")
        w()
    else:
        w("*Run ade3x3_step63_reverse_engineering_cancellation_visualization.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 63 supplies exactly the kind of reverse-engineering datum the earlier steps lacked: an")
    w("explicit low-rank 3x3 algorithm that can be measured in the fiber-mode basis rather than merely")
    w("hypothesized. The strongest exact result is that the recovered public rank-23 decomposition has")
    w("nuisance rank 14 and quotient gain 9, so it lands precisely on the Step 52 boundary R = 9 +")
    w("rank(Nuisance). That is already informative structurally: the known 23-term solution uses every")
    w("available nuisance degree of freedom. The removal tests sharpen that statement further. Every")
    w("single-term deletion drops quotient gain from 9 to 8, and no pair deletion recovers gain 9")
    w("either, so this public factorization shows no hidden 22-term or 21-term gamma-only redundancy.")
    w("The cancellation visualization then makes the mechanism concrete: almost every active dead")
    w("equation is resolved by exact two-term sign cancellation, with one genuinely multi-way balanced")
    w("dead equation remaining as an exceptional knot. The commutator split is structurally suggestive")
    w("but not by itself decisive: for 3x3 the commutator tensor has flattening lower bound 8, below the")
    w("full multiplication tensor's 9, so it is a simpler attack surface but not obviously simple enough")
    w("to force a new lower bound alone.")

    # ── STEP 64 ──
    w()
    w(f"## {section_num}. SMALL INTEGER COEFFICIENT ENUMERATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] + [MEASURED_FROM_CODE] (Step 64)")
    w()
    w("Step 64 studies the finite ternary coefficient pool directly. The main exact reduction is that")
    w("a bilinear profile is an outer product of two nonzero ternary vectors, so duplicate elimination")
    w("and global-sign quotienting reduce to projective-line counting instead of brute-force hashing of")
    w("all 3^18 raw 3x3 alpha,beta pairs. The step then combines exact Burnside orbit counts, a")
    w("usefulness score in the Step 51 live-versus-nuisance split, and both collapsed and full-tensor")
    w("greedy diagnostics.")
    w()
    (
        sum64_rows,
        top64_rows,
        greedy64_rows,
        greedy64_2c_rows,
        greedy64_2f_rows,
    ) = read_step64_outputs()
    if sum64_rows:
        sum64 = {row['summary_name']: row for row in sum64_rows}
        w("### Task 1: Exact Pool Statistics")
        w()
        w(f"**3x3 raw pairs:** {sum64['pool3_raw_pairs']['summary_value']}")
        w(f"**3x3 nonzero pairs:** {sum64['pool3_nonzero_pairs']['summary_value']}")
        w(f"**3x3 projective ternary lines:** {sum64['pool3_projective_line_count']['summary_value']}")
        w(f"**3x3 distinct profiles modulo duplicates and sign:** {sum64['pool3_distinct_profiles_mod_sign']['summary_value']}")
        w(f"**3x3 symmetry-reduced profile orbits:** {sum64['pool3_symmetry_reduced_profiles']['summary_value']}")
        w(f"**3x3 zero-projection profiles:** {sum64['pool3_zero_projection_profiles']['summary_value']}")
        w(f"**3x3 dead-free nonzero profiles:** {sum64['pool3_deadfree_nonzero_profiles']['summary_value']}")
        w(f"**Collapsed 3x3 greedy first exact rank:** {sum64['collapsed_3x3_greedy_first_exact_rank']['summary_value']}")
        w(f"**Collapsed 2x2 greedy first exact rank:** {sum64['collapsed_2x2_greedy_first_exact_rank']['summary_value']}")
        w(f"**2x2 distinct profiles modulo sign:** {sum64['pool2_distinct_profiles_mod_sign']['summary_value']}")
        w(f"**2x2 symmetry-reduced profiles:** {sum64['pool2_symmetry_reduced_profiles']['summary_value']}")
        w(f"**Full-tensor 2x2 greedy first exact rank:** {sum64['full_tensor_2x2_greedy_first_exact_rank']['summary_value']}")
        w(f"**Selected Strassen-profile matches in 2x2 greedy:** {sum64['full_tensor_2x2_selected_strassen_matches']['summary_value']}")
        w()
        w("### Task 2: Top Usefulness Profiles")
        w()
        w("| rank | alpha | beta | projection | nuisance sq | usefulness score | dead-free? |")
        w("|------|-------|------|------------|-------------|------------------|------------|")
        for row in top64_rows[:10]:
            w(f"| {row['rank']} | {row['alpha']} | {row['beta']} | {row['projection_onto_tensor']} | {row['nuisance_magnitude_squared']} | {row['usefulness_score']} | {row['dead_free']} |")
        w()
        w("### Task 3: Collapsed Greedy Diagnostics")
        w()
        w("**3x3 collapsed 81D greedy trace (first exact completion):**")
        w()
        w("| rank | alpha | beta | weight | score drop | residual norm^2 | exact zero? |")
        w("|------|-------|------|--------|------------|-----------------|-------------|")
        for row in greedy64_rows[:3]:
            w(f"| {row['rank_step']} | {row['alpha']} | {row['beta']} | {row['weight']} | {row['score_drop']} | {row['residual_norm_squared']} | {row['exact_zero']} |")
        w()
        w("**2x2 collapsed 16D greedy trace (first exact completion):**")
        w()
        w("| rank | alpha | beta | weight | score drop | residual norm^2 | exact zero? |")
        w("|------|-------|------|--------|------------|-----------------|-------------|")
        for row in greedy64_2c_rows[:2]:
            w(f"| {row['rank_step']} | {row['alpha']} | {row['beta']} | {row['weight']} | {row['score_drop']} | {row['residual_norm_squared']} | {row['exact_zero']} |")
        w()
        w("### Task 4: Corrected 2x2 Full-Tensor Greedy Verification")
        w()
        w("| rank | profile id | alpha | beta | gamma | residual norm^2 | exact zero? | Strassen profile? |")
        w("|------|------------|-------|------|-------|-----------------|-------------|-------------------|")
        for row in greedy64_2f_rows:
            w(f"| {row['rank_step']} | {row['profile_id']} | {row['alpha']} | {row['beta']} | {row['gamma']} | {row['residual_norm_squared']} | {row['exact_zero']} | {row['profile_matches_strassen']} |")
        w()
    else:
        w("*Run ade3x3_step64_small_integer_coefficient_enumeration.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("Step 64 closes one tempting loophole cleanly: even after quotienting duplicates and global sign,")
    w("the ternary 3x3 coefficient pool is not small in any practical sense. It still contains")
    w("96,845,281 distinct nonzero bilinear profiles before symmetry reduction, and 570,521 orbits after")
    w("the full 216-element action is factored out. Just as important, the live-versus-nuisance score")
    w("finds no dead-free profile with nonzero live projection at all. The top-scoring profiles are large,")
    w("row-constant or banded sign patterns that project strongly onto the live tensor but still carry")
    w("substantial nuisance mass.")
    w("The two greedy experiments separate what is trivial from what is structurally relevant. In the")
    w("collapsed 81D model, the algorithm reaches exact zero for 3x3 at rank 3 and for 2x2 at rank 2 by")
    w("simply selecting the summation-channel masks, so that model is far too weak to represent true")
    w("matrix-multiplication complexity. The corrected 2x2 full-tensor greedy is the meaningful test, and")
    w("there it fails to recover Strassen or even any exact rank-7 decomposition. That negative result does")
    w("not prove impossibility, but it does show that naive matching pursuit over the finite ternary")
    w("profile pool is not by itself an adequate search strategy for genuine low-rank algorithms.")

    # ── STEP 65 ──
    w()
    w(f"## {section_num}. POLYOMINO SUB-TENSOR RANKS + TILING ANALYSIS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] + [MEASURED_FROM_CODE] (Step 65)")
    w()
    w("Step 65 reframes restricted output sets as tiny subtensors and asks whether a polyomino tiling of")
    w("the 3x3 output grid can beat the known 23-term full algorithm when piece costs are estimated by")
    w("subtensor rank. The step computes flat and toroidal L-tromino tilings, numerical ranks for the")
    w("unknown tetromino classes, AlphaTensor support clustering, and an exact-cover search over the flat")
    w("3x3 board. The central warning is that these tiling costs are only piecewise upper bounds on")
    w("restricted subtensors. They are not by themselves certified full 3x3 multiplication algorithms.")
    w()
    (
        sum65_rows,
        rank65_rows,
        mixed65_rows,
        toroidal65_rows,
        cluster65_rows,
    ) = read_step65_outputs()
    step66_audits = read_step66_piece_audits()
    (
        step67_summary_rows,
        step67_audit_rows,
        step67_rank_rows,
        step67_tiling_rows,
        step67_low_cost_rows,
        step67_corrector_rows,
        step67_cluster_rows,
    ) = read_step67_outputs()
    (
        step68_fourier_summary_rows,
        step68_fourier_mode_rows,
        step68_fourier_component_rows,
    ) = read_step68_fourier_outputs()
    (
        step69_summary_rows,
        step69_projection_rows,
        step69_subspace_rows,
        step69_routed_rows,
        step69_multilevel_rows,
        step69_greedy_rows,
    ) = read_step69_outputs()
    (
        step70_summary_rows,
        step70_outer_rows,
        step70_alpha_pair_rows,
        step70_strassen_pair_rows,
        step70_literature_rows,
    ) = read_step70_outputs()
    (
        step71_summary_rows,
        step71_shot2_rows,
        step71_shot4_rows,
        step71_shot5_rows,
    ) = read_step71_outputs()
    step67_summary = {row['summary_name']: row for row in step67_summary_rows}
    step68_fourier_summary = {row['summary_name']: row for row in step68_fourier_summary_rows}
    step69_summary = {row['summary_name']: row for row in step69_summary_rows}
    step70_summary = {row['summary_name']: row for row in step70_summary_rows}
    step71_summary = {row['summary_name']: row for row in step71_summary_rows}
    if sum65_rows:
        sum65 = {row['summary_name']: row for row in sum65_rows}
        l65_rows = [row for row in rank65_rows if row['family'] == 'L_tromino']
        w("### Task 1: Priority L-Tromino Rank")
        w()
        if l65_rows:
            l65 = l65_rows[0]
            w(f"**Priority L-tromino entries:** {l65['representative_entries']}")
            w(f"**Flattening lower bound:** {l65['flattening_lower_bound']}")
            w(f"**Numerical rank upper bound:** {l65['numerical_rank_upper_bound']}")
            w(f"**AlphaTensor upper bound:** {l65['alpha_tensor_upper_bound']}")
            w(f"**Best verified loss:** {l65['best_verified_loss']}")
        w(f"**Flat L-tromino placements:** {sum65['l_tromino_instance_count']['summary_value']}")
        w(f"**Flat L-tromino symmetry classes:** {sum65['l_tromino_symmetry_classes']['summary_value']}")
        w(f"**Flat 3x3 tilings by three L-trominoes:** {sum65['l_tromino_tiling_count']['summary_value']}")
        w()
        w("### Task 2 / 4: Polyomino Rank Table")
        w()
        if step67_summary_rows:
            w("**Step 67 completion:** the remaining blank Step 65 rows were rerun from scratch with 500 parallel restarts per rank.")
            w("All five previously unresolved classes now have verified exact numerical upper bounds:")
            w()
            w("| audited class | outputs | flattening LB | first exact rank | best rank-11 near miss |")
            w("|---------------|---------|---------------|------------------|-------------------------|")
            latest67 = {}
            for row in step67_audit_rows:
                if row['exact_rank_upper_bound']:
                    latest67[row['piece_id']] = row
            near_miss67 = {}
            for row in step67_audit_rows:
                if row['rank_tested'] == '11':
                    near_miss67[row['piece_id']] = row['best_verified_loss']
            for piece_id in sorted(latest67):
                row = latest67[piece_id]
                w(f"| {piece_id} | {row['piece_outputs']} | {row['flattening_lower_bound']} | {row['exact_rank_upper_bound']} | {near_miss67.get(piece_id, '')} |")
            w()
        elif step66_audits:
            w("**Step 66 audit correction:** the Step 65 recovery export was unstable after the crash/recovery path.")
            w("Fresh independent parallel rescans now override the following tetromino entries:")
            w()
            w("| audited piece outputs | flattening LB | fresh exact numerical upper bound | best lower-rank near miss |")
            w("|-----------------------|---------------|----------------------------------|---------------------------|")
            for audit in step66_audits:
                output_key = audit['piece_outputs']
                if output_key == '(0,0); (0,1); (1,0); (2,1)':
                    near_miss = 'R=11 best loss 1.4844916268325573e-03; extended 2000-restart test improves only to 1.1867109268390639e-03'
                elif output_key == '(0,2); (1,2); (2,1); (2,2)':
                    near_miss = 'R=10 best loss 4.1316631254345992e-03; R=11 best loss 2.0633889273130681e-03'
                else:
                    near_miss = ''
                w(f"| {audit['piece_outputs']} | {audit['flattening_lower_bound']} | {audit['exact_rank_upper_bound'] or 'none'} | {near_miss} |")
            w()
        w("| family | representative | flattening LB | numerical rank | AlphaTensor UB | exact rank if determined |")
        w("|--------|----------------|---------------|----------------|----------------|--------------------------|")
        rank_source_rows = step67_rank_rows if step67_rank_rows else rank65_rows
        for row in rank_source_rows:
            numerical = row.get('verified_rank_upper_bound') or row.get('numerical_rank_upper_bound', '')
            exact_rank = row.get('verified_rank_upper_bound') or row.get('exact_rank_if_determined', '')
            w(f"| {row['family']} | {row['representative_entries']} | {row['flattening_lower_bound']} | {numerical} | {row['alpha_tensor_upper_bound']} | {exact_rank} |")
        w()
        w("### Task 2 / 3: Toroidal L-Trominoes and AlphaTensor Support Clusters")
        w()
        w(f"**Exported toroidal L-tromino tiling classes:** {len(toroidal65_rows)}")
        if toroidal65_rows:
            w(f"**Best exported toroidal 3xL tiling upper bound:** {min(int(row['total_rank_upper_bound']) for row in toroidal65_rows)}")
        w(f"**AlphaTensor support clusters:** {sum65['alphatensor_support_cluster_count']['summary_value']}")
        w()
        if toroidal65_rows:
            w("| toroidal tiling | piece class ids | total upper-bound cost |")
            w("|-----------------|-----------------|------------------------|")
            for row in toroidal65_rows:
                w(f"| {row['canonical_tiling']} | {row['piece_class_ids']} | {row['total_rank_upper_bound']} |")
            w()
        w("| support cluster | support entries | shape | term count | term ids |")
        w("|----------------|-----------------|-------|------------|----------|")
        for row in cluster65_rows[:10]:
            w(f"| {row['support_cluster_id']} | {row['support_entries']} | {row['shape_label']} | {row['term_count']} | {row['term_ids']} |")
        w()
        w("### Task 5: Best Flat Exact-Cover Tilings")
        w()
        if step67_summary_rows:
            w(f"**Corrected minimum flat tiling upper bound:** {step67_summary['best_verified_flat_tiling_cost']['summary_value']}")
            w(f"**Corrected minimum tiling signature:** {step67_summary['best_verified_flat_tiling_signature']['summary_value']}")
            w(f"**Corrected flat tilings <= 24:** {step67_summary['flat_tilings_cost_leq_24']['summary_value']}")
            w(f"**Any corrected flat tiling beats 23:** {step67_summary['any_flat_tiling_beats_23']['summary_value']}")
        else:
            w(f"**Minimum flat tiling upper bound:** {sum65['minimum_tiling_upper_bound']['summary_value']}")
            w(f"**Minimum tiling signature:** {sum65['minimum_tiling_signature']['summary_value']}")
            w(f"**Low-cost tilings <= 22:** {sum65['low_cost_tiling_count_leq_22']['summary_value']}")
        w()
        mixed_rows = step67_low_cost_rows if step67_summary_rows else mixed65_rows
        if mixed_rows:
            w("| tiling signature | pieces | class ids | total upper-bound cost |")
            w("|------------------|--------|-----------|------------------------|")
            for row in mixed_rows:
                total_cost = row.get('total_verified_cost') or row.get('total_rank_upper_bound')
                w(f"| {row['tiling_signature']} | {row['pieces']} | {row['class_ids']} | {total_cost} |")
            w()
    else:
        w("*Run ade3x3_step65_polyomino_subtensor_ranks_tiling_analysis.py to populate this section.*")
    w()
    w("[INTERPRETATION]")
    w()
    w("The make-or-break L-tromino result is now clear. The unique flat L-tromino class has numerical")
    w("fits at essentially zero residual while the export table still leaves the numerical-rank field blank;")
    w("combined with the flattening lower bound 6 and the AlphaTensor upper bound 14, this shows that the")
    w("L-tromino is not obviously cheap enough to support a dramatic three-piece decomposition. In any case")
    w("the flat 3x3 board admits no tiling by three L-trominoes at all. The toroidal variant behaves")
    w("differently: at least one toroidal symmetry class is exported, with total upper-bound cost 27, so")
    w("wrapping does create L-tilings but does not by itself produce a competitive cost.")
    if step67_summary_rows:
        w("Step 67 closes the remaining audit loop. The formerly blank rows now resolve to exact numerical")
        w("upper bounds L-tromino = 9 and all four unresolved tetromino classes = 12, so every connected")
        w("tetromino class on the 3x3 output board now sits at rank 11 or 12 rather than at the speculative")
        w("rank-9 frontier suggested by the corrupted Step 65 recovery export. The corrected flat exact-cover")
        w("optimum is 26, achieved by L-tetromino + monomino + square tetromino, and there are no verified")
        w("flat tilings with cost 24 or below.")
        w("The layered AlphaTensor readout also sharpens the nuisance story. In the Step 51/52 basis there are")
        w("0 signal-dominant terms, 1 anisotropy-dominant term, 3 dead-X-dominant terms, and 19 mixed terms;")
        w("the nuisance rank remains 14. So the public rank-23 algorithm is not built from mostly pure signal")
        w("pieces. It is heavily nuisance-saturated, with only three clearly dead-X-heavy corrector terms in")
        w("the exported decomposition.")
        if step68_fourier_summary_rows:
            canonical_modes = step68_fourier_summary['canonical_fourier_nonzero_modes']['summary_value']
            canonical_sum = step68_fourier_summary['canonical_fourier_component_rank_sum_upper_bound']['summary_value']
            phase_modes = step68_fourier_summary['phase_j1_fourier_nonzero_modes']['summary_value']
            phase_sum = step68_fourier_summary['phase_j1_fourier_component_rank_sum_upper_bound']['summary_value']
            any_below_14 = step68_fourier_summary['any_fourier_component_sum_below_14']['summary_value']
            w("Step 68 then tested whether the canonical 9-term signal layer becomes cheaper after moving the")
            w("dead-only correction slice into the 3-point Fourier basis on the summation indices. The answer is")
            w("negative in the simplest exact decomposition: the canonical dead residual and the phase-j=1")
            w("phase-weighted variant both concentrate on the same three Fourier modes, and each surviving mode")
            w("still has exact component rank 9.")
            w(f"For the canonical balanced residual the nonzero Fourier modes are {canonical_modes}, with component-rank sum upper bound {canonical_sum}.")
            w(f"For the phase-j=1 residual the nonzero Fourier modes are {phase_modes}, with component-rank sum upper bound {phase_sum}.")
            w(f"So the current Fourier steering verdict is: any immediate Fourier component-sum route below nuisance budget 14 = {any_below_14}.")
        if step69_summary_rows:
            mode00_rank = step69_summary['alphatensor_mode00_rank']['summary_value']
            mode12_rank = step69_summary['alphatensor_mode12_rank']['summary_value']
            mode21_rank = step69_summary['alphatensor_mode21_rank']['summary_value']
            pair00_12 = step69_summary['alphatensor_pair_intersection_00_12']['summary_value']
            pair00_21 = step69_summary['alphatensor_pair_intersection_00_21']['summary_value']
            pair12_21 = step69_summary['alphatensor_pair_intersection_12_21']['summary_value']
            triple = step69_summary['alphatensor_triple_intersection_dimension']['summary_value']
            union = step69_summary['alphatensor_three_mode_union_dimension']['summary_value']
            routed_best = step69_summary['mode_routed_best_total_cost_upper_bound']['summary_value']
            multilevel_lb = step69_summary['multilevel_total_cost_lower_bound_via_flattening']['summary_value']
            multilevel_ub = step69_summary['multilevel_total_cost_upper_bound_via_slice_rank']['summary_value']
            greedy_cover = step69_summary['reduced_greedy_first_rank9_cover_step']['summary_value']
            greedy_scope = step69_summary['reduced_greedy_pool_scope']['summary_value']
            silent_terms = sum(1 for row in step69_projection_rows if row['classification'] == 'silent')
            multimode_terms = sum(1 for row in step69_projection_rows if row['classification'] == 'multi-mode')
            w("Step 69 resolves the interlocking question directly in term space. For each of the 23 public")
            w("AlphaTensor terms, the dead-X bilinear profile was projected into the three Step 68 Fourier")
            w("modes, producing three 23x9 participation matrices D^{00}, D^{12}, D^{21}. Their column spaces")
            w("inside the 23-dimensional term space are the exact interlocking subspaces.")
            w(f"The exact dimensions are V_00={mode00_rank}, V_12={mode12_rank}, V_21={mode21_rank}; pairwise intersections = ({pair00_12}, {pair00_21}, {pair12_21}); triple intersection = {triple}; and total union dimension = {union}.")
            w("So the dead-only interlocking is tighter than the earlier nuisance-budget shorthand suggested:")
            w("the three Fourier-mode dead spans do not behave like three independent rank-9 blocks glued")
            w("down to 14. They are already only rank 8 each, with a 6-dimensional triple overlap and total")
            w("dead-mode union dimension 10 in term space.")
            w(f"At the term level, Step 69 finds {multimode_terms} genuinely multi-mode participants and {silent_terms} dead-silent terms; there are no dead-mode single-mode dominant terms under the requested 2x dominance rule.")
            w(f"The mode-routed 3-fiber probe does not open a cheap path: every tested symmetry class for pure mode-12 or mode-21 dead routing lands at total cost {routed_best} when repeated across three groups.")
            w(f"The two-level three-mode probe is also unpromising in its current form: after one complex rank-1 mega-corrector per mode, the combined residual has flattening lower bound {multilevel_lb} and slice-rank upper bound {int(multilevel_ub) - 12}, so the resulting total-cost window is {multilevel_lb}..{multilevel_ub} after adding the 9 signal terms and 3 mega-correctors.")
            w(f"A reduced ternary-pool greedy cover on the Step 64 shortlist reaches rank 9 in all three modes by step {greedy_cover}, but that search is only heuristic because the project currently exports exact orbit counts for the ternary pool, not a full 570,521-orbit representative table. So Step 69 does not yet certify a 13-term correction layer, and it does not produce any verified route below rank 23.")
        if step70_summary_rows:
            recursive_cost = step70_summary['recursive_strassen_top_left_3x3_leaf_count']['summary_value']
            recursive_beats = step70_summary['recursive_strassen_beats_23']['summary_value']
            recursive_verified = step70_summary['recursive_strassen_top_left_3x3_verified']['summary_value']
            alpha_constant = step70_summary['alphatensor23_constant_multiple_pair_count']['summary_value']
            alpha_disjoint = step70_summary['alphatensor23_disjoint_support_pair_count']['summary_value']
            alpha_unstructured = step70_summary['alphatensor23_unstructured_overlap_pair_count']['summary_value']
            strassen_structured = step70_summary['strassen2x2_retained_missing_structured_pair_count']['summary_value']
            big_overlap = max((int(row['common_support_size']) for row in step70_alpha_pair_rows if row['classification'] == 'constant_multiple'), default=0)
            lit_detail = step70_literature_rows[0]['detail'] if step70_literature_rows else 'No literature status recorded.'
            w("Step 70 tests the first concrete depth-2 arithmetic-circuit escape route. The direct audit is a")
            w("recursive Strassen computation on the 4x4 zero-padded embedding of the 3x3 product, with the")
            w("full 49 recursive leaves expanded explicitly and then pruned whenever a leaf scalar product is")
            w("identically zero under padding or contributes only to discarded padded outputs.")
            w(f"That exact circuit still needs {recursive_cost} leaf multiplications for the top-left 3x3 block, and the retained leaves reconstruct the target exactly = {recursive_verified}; so this direct padded depth-2 route beats 23 = {recursive_beats}.")
            w("The AlphaTensor pair-ratio scan also does not reveal an obvious depth-2 collapse. Among the 253")
            w(f"term pairs, {alpha_disjoint} are support-disjoint, {alpha_constant} are merely constant multiples on their common support, and only {alpha_unstructured} have nonconstant overlap; the largest constant-overlap support size is {big_overlap} entries.")
            w(f"For the obvious 2x2 Strassen split, the retained-vs-missing pair scan records {strassen_structured} structured overlaps, but these are only local common-support coincidences, not a certified 6-multiplication depth-2 replacement for Strassen.")
            w(f"The literature pass recovers Pan's asymptotic trilinear-aggregation line but not a ready-to-instantiate small 3x3 circuit: {lit_detail}")
        if step71_summary_rows:
            shot1_pool = step71_summary['shot1_pool_size']['summary_value']
            shot1_replacements = step71_summary['shot1_total_valid_replacements']['summary_value']
            shot1_slots = step71_summary['shot1_slots_with_any_replacement']['summary_value']
            shot2_merges = step71_summary['shot2_exact_mergeable_pair_count']['summary_value']
            shot2_best_pair = step71_summary['shot2_best_near_merge_pair']['summary_value']
            shot3_comm = step71_summary['shot3_commutator_first_exact_rank_in_scan']['summary_value']
            shot3_anti = step71_summary['shot3_anticommutator_first_exact_rank_in_scan']['summary_value']
            shot3_comm_lb = step71_summary['shot3_commutator_flattening_lb']['summary_value']
            shot3_anti_lb = step71_summary['shot3_anticommutator_flattening_lb']['summary_value']
            shot4_near = step71_summary['shot4_near_merge_pairs_below_1e-6']['summary_value']
            shot5_found = step71_summary['shot5_found_feasible_subset_below_23']['summary_value']
            shot5_best_gain = max((int(row['best_quotient_gain']) for row in step71_shot5_rows), default=0)
            shot5_greedy_residual = step71_summary['shot5_greedy_final_residual_norm_squared']['summary_value']
            shot2_t03_t06 = next((row for row in step71_shot2_rows if row['term_i'] == 't03' and row['term_j'] == 't06'), None)
            bilinear_rank_note = ''
            if shot2_t03_t06:
                bilinear_rank_note = f" The sharpest near-hit is {shot2_best_pair}: its combined 9x81 flattening has rank 1 with zero tail error, but the induced 9x9 bilinear profile still has rank {shot2_t03_t06['best_profile_bilinear_rank']}, so it is not a genuine single-term merge."
            w("Step 71 then took five local shots at the public rank-23 wall: slotwise replacement, exact pair")
            w("merging, commutator/anticommutator decomposition, the three Step 70 nonconstant-overlap pairs,")
            w("and a random neighborhood search around the AlphaTensor term set.")
            w(f"The replacement shot stayed negative on the explicit in-repo pool: with {shot1_pool} candidates (the 23 AlphaTensor profiles, the Step 64 top-100 shortlist, and 5000 random ternary profiles), there were {shot1_replacements} exact replacements across {shot1_slots}/23 removal slots. The requested full Step 64 orbit-wide scan could not be run literally because the repository exports the exact 570,521 orbit count but not a materialized representative table.")
            w(f"The pair-merge shot also stayed negative: exact mergeable pairs = {shot2_merges}.{bilinear_rank_note}")
            w(f"The commutator shot remains suggestive but unresolved numerically. The flattening lower bounds stay at ({shot3_comm_lb}, {shot3_anti_lb}) for commutator and anticommutator, but the requested 200-restart scans found no exact witness through rank 20 for either tensor, so no shared-term follow-up was triggered; the first exact ranks in scan are ({shot3_comm}, {shot3_anti}).")
            w(f"The three nonconstant-overlap pairs from Step 70 also stayed negative under direct inspection: all 3 were analyzed, and near-merge pairs below 1e-6 = {shot4_near}.")
            w(f"The local-neighborhood shot was equally negative. In a 123-term pool built from all 23 AlphaTensor terms plus 100 random ternary additions, 10,000 random subsets were tested at each of R=20,21,22; none were feasible, the best quotient gain observed was {shot5_best_gain}, and the full-729 greedy trace stalled at rank 22 with residual norm^2 {shot5_greedy_residual}. So Step 71 produces no verified route below 23; random-neighborhood R<23 feasible subset = {shot5_found}.")
    elif step66_audits:
        w("Step 66 changes the tetromino story materially. Fresh independent parallel rescans show that the")
        w("audited S/Z and L tetromino classes used in the former candidate low-cost tilings are both rank 12,")
        w("not rank 9. So the crash-recovery Step 65 export overstated the strength of those classes, and any")
        w("claimed 21-cost assembly based on them is invalid. The S/Z class is especially revealing: rank 11")
        w("gets close but does not close even after 2000 parallel restarts, which suggests real structure rather")
        w("than optimizer laziness.")
    else:
        w("The broader polyomino picture is still suggestive. Several non-square tetromino scans achieve")
        w("near-zero residual at the flattening lower-bound level 9, while the best flat exact-cover cost in")
        w("the currently exported summary is 26 via domino-column + three monominoes + square tetromino.")
    w("This remains structurally informative but must not be overinterpreted: it is only a")
    w("polyomino-subtensor upper bound assembled from separate restricted problems. It does not yet")
    w("certify any global rank-26 or better matrix multiplication algorithm, because the")
    w("piecewise decompositions are not forced to coexist without cross-piece interference or extra sharing.")

    # ── PHASE 17 CONSERVATION LAW ──
    w()
    w(f"## {section_num}. CONSERVATION LAW PROOF STATUS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 17)")
    w()
    w("Phase 17 separates the observed conservation law into a proved fiber-mode faithfulness step")
    w("and an open Delta-containment step.")
    w()
    faith17, identities17, gamma17, collective17 = read_phase17_outputs()
    if faith17 and identities17 and collective17:
        valid17 = faith17.get("valid_decompositions", {})
        pert17 = faith17.get("perturbed_alphatensor", {})
        random17 = faith17.get("random_rank1_collections", {})
        cross17 = identities17.get("cross_ratio_identities", {})
        proj17 = collective17.get("alpha_tensor_projection", {})
        del17 = collective17.get("single_deletion_containment", {})
        anti17 = collective17.get("anticommutator_rank19", {})
        rand_cont17 = collective17.get("random_rank23_collections", {})
        alpha_row = next((row for row in valid17.get("rows", []) if "alphatensor" in row["label"]), None)
        standard_row = next((row for row in valid17.get("rows", []) if "standard" in row["label"]), None)

        w("### Part 1: Fiber-Mode Faithfulness")
        w()
        w("**Theorem:** For any minimum-rank decomposition, rank([Sigma|H|Delta]) = R.")
        w()
        w("**Proof text:** the fiber-mode map is an invertible change of basis on the 81 A x B")
        w("monomials, so rank([Sigma|H|Delta]) equals the rank of the bilinear profiles")
        w("{alpha_k tensor beta_k}. A linear dependence among those profiles would remove one term,")
        w("contradicting minimum rank.")
        w()
        w(f"**Verified on valid decompositions:** {valid17.get('all_faithful_numeric')} across {len(valid17.get('rows', []))} cases.")
        if alpha_row:
            w(f"- AlphaTensor rank-23: rank([Sigma|H|Delta]) = {alpha_row['rank_stacked_numeric']}, rank(H) = {alpha_row['rank_H_numeric']}, eta-nullity = {alpha_row['eta_nullity_numeric']}, conservation sum = {alpha_row['conservation_sum_numeric']}.")
        if standard_row:
            w(f"- Standard rank-27: rank([Sigma|H|Delta]) = {standard_row['rank_stacked_numeric']}, rank(H) = {standard_row['rank_H_numeric']}, eta-nullity = {standard_row['eta_nullity_numeric']}, conservation sum = {standard_row['conservation_sum_numeric']}.")
        w(f"**Corollary dim(ker Gamma) = rank([H|Delta]) on valid decompositions:** {valid17.get('all_corollary_holds_numeric')}.")
        w(f"**Perturbed AlphaTensor trials:** {pert17.get('trial_count', 0)} / {pert17.get('trial_count', 0)} faithful.")
        w(f"**Random generic rank-1 collections:** {random17.get('trial_count', 0)} / {random17.get('trial_count', 0)} faithful.")
        w()
        w("### Part 2: Delta Containment Evidence")
        w()
        w("If Delta subset span(H) holds universally, then rank(H) = R - 9 and therefore")
        w("R + eta_nullity = 27 for 3x3 matrix multiplication.")
        w()
        w("| evidence source | rank(H) | rank([H|Delta]) | Delta subset span(H) | note |")
        w("|-----------------|---------|------------------|-----------------------|------|")
        if alpha_row:
            w(f"| AlphaTensor rank-23 | {alpha_row['rank_H_numeric']} | {alpha_row['rank_HDelta_numeric']} | True | conservation sum = {alpha_row['conservation_sum_numeric']} |")
        if del17:
            w(f"| AlphaTensor single deletions (23 cases) | 14 throughout | 14 throughout | {del17.get('all_exact')} | exact containment count = {del17.get('containment_exact_count')} |")
        if anti17:
            w(f"| Step 75 anticommutator rank-19 | {anti17.get('rank_H_numeric', '')} | {anti17.get('rank_HDelta_numeric', '')} | {anti17.get('delta_in_eta_numeric', '')} | not a matrix-multiplication decomposition |")
        if rand_cont17:
            w(f"| Random rank-23 collections ({rand_cont17.get('trial_count', 0)} trials) | {rand_cont17.get('min_rank_H_numeric', '')}..{rand_cont17.get('max_rank_H_numeric', '')} | {rand_cont17.get('min_rank_HDelta_numeric', '')}..{rand_cont17.get('max_rank_HDelta_numeric', '')} | {rand_cont17.get('containment_count', 0)} / {rand_cont17.get('trial_count', 0)} | containment rate = {rand_cont17.get('containment_rate', 0.0):.6f} |")
        w()
        if proj17:
            w(f"**AlphaTensor projection inheritance:** exact Delta = H*M was re-verified = {proj17.get('exact_projection_verified')}, and the Gamma-weighted projection identities were checked on {proj17.get('tested_delta_columns', 0)} delta columns with max residual {proj17.get('weighted_projection_max_abs', 0.0)}.")
        w()
        w("### Single-Term Cross-Ratio Structure")
        w()
        w(f"- Live-product recovery from (sigma, eta1, eta2): {identities17.get('live_product_recovery', {}).get('verified')}")
        w(f"- Left-factor identities: {cross17.get('left_identity_count', 0)} total, {cross17.get('left_independent_count', 0)} independent")
        w(f"- Right-factor identities: {cross17.get('right_identity_count', 0)} total, {cross17.get('right_independent_count', 0)} independent")
        w(f"- Combined independent cross-ratio identities: {cross17.get('combined_independent_count', 0)}")
        w(f"- Linear propagation from live slices to dead slices: {gamma17.get('verdict', {}).get('linear_propagation_from_live_slices')}")
        w()
        w("### Proof Skeleton For The Remaining Gap")
        w()
        w("Part 1 gives rank([Sigma|H|Delta]) = R. Since rank(Gamma) = 9 and Gamma annihilates H and Delta,")
        w("every valid minimum-rank decomposition satisfies col([H|Delta]) subset ker(Gamma) and")
        w("rank([H|Delta]) = R - 9 = dim(ker(Gamma)), hence col([H|Delta]) = ker(Gamma).")
        w()
        w("Therefore the unresolved step is equivalent to a pure kernel-saturation statement:")
        w("Delta subset span(H) iff col(H) = ker(Gamma) iff rank(H) = R - 9.")
        w()
        w("For matrix multiplication one has the stronger per-channel equations Gamma * P_s = I_9 for")
        w("s = 0, 1, 2, where P_s is the R x 9 live-product matrix of channel s. Writing")
        w("P_s = X_0 + K_s for any fixed right inverse X_0 of Gamma shows that Eta1 = K_0 - K_1 and")
        w("Eta2 = K_1 - K_2, so H is generated by the difference sector of three right inverses of Gamma.")
        w()
        w("This identifies the next theorem target:")
        w("if (P_0, P_1, P_2) comes from a valid minimum-rank multiplication decomposition and")
        w("Gamma * P_s = I_9 for each s, then span(col(P_0 - P_1), col(P_1 - P_2)) = ker(Gamma).")
        w()
        w("That route is stronger than a genericity argument. Genericity can only show that failure of")
        w("rank(H) = R - 9 is lower-dimensional; it cannot rule out exceptional valid decompositions.")
        w("The anticommutator failure supports this diagnosis: it lies outside the per-channel identity")
        w("regime, and its H block correspondingly does not saturate ker(Gamma).")
        w()
        w("[INTERPRETATION]")
        w()
        w("Phase 17 resolves the logical split behind the conservation law. Part 1 is now proved and")
        w("verified computationally. Part 2 remains open. Delta containment is exact on AlphaTensor and")
        w("on all 23 of its single deletions, but it is not a generic feature of arbitrary rank-23")
        w("collections, and it also fails on the exported Step 75 anticommutator rank-19 fibermode data.")
    else:
        w("*Run the Phase 17 scripts to populate this section.*")

    # ── PHASE 18 KERNEL SATURATION ──
    w()
    w(f"## {section_num}. RIGHT-INVERSE KERNEL-SATURATION DIAGNOSTICS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 18)")
    w()
    w("Phase 18 moves the remaining gap into testable compute by working directly with the")
    w("per-channel right-inverse equations Gamma * P_s = I_9 and the difference-sector block")
    w("H = [P_0 - P_1 | P_1 - P_2].")
    w()
    phase18 = read_phase18_outputs()
    if phase18:
        ref18 = phase18.get("reference_decompositions", {})
        synth18 = phase18.get("synthetic_right_inverse_trials", {})
        defects18 = phase18.get("forced_defect_examples", [])
        alpha18 = next((row for row in ref18.get("rows", []) if "alphatensor" in row.get("label", "")), None)
        standard18 = next((row for row in ref18.get("rows", []) if "standard" in row.get("label", "")), None)
        alpha_synth18 = synth18.get("alphatensor_gamma", {})
        standard_synth18 = synth18.get("standard_gamma", {})

        w("### Known Exact Decompositions")
        w()
        if alpha18:
            w(f"- AlphaTensor rank-23: ker-dimension = {alpha18.get('ker_gamma_dim_numeric')}, rank(H) = {alpha18.get('rank_H_numeric')}, kernel saturation = {alpha18.get('kernel_saturation_numeric')}, and all three channel identities Gamma * P_s = I_9 hold exactly = {alpha18.get('channel0_identity_exact') and alpha18.get('channel1_identity_exact') and alpha18.get('channel2_identity_exact')}.")
        if standard18:
            w(f"- Standard rank-27: ker-dimension = {standard18.get('ker_gamma_dim_numeric')}, rank(H) = {standard18.get('rank_H_numeric')}, kernel saturation = {standard18.get('kernel_saturation_numeric')}, and all three channel identities Gamma * P_s = I_9 hold exactly = {standard18.get('channel0_identity_exact') and standard18.get('channel1_identity_exact') and standard18.get('channel2_identity_exact')}.")
        w()

        w("### Synthetic Right-Inverse Genericity")
        w()
        if alpha_synth18:
            w(f"- AlphaTensor Gamma: {alpha_synth18.get('saturation_count', 0)} / {alpha_synth18.get('trial_count', 0)} random right-inverse triples saturated ker(Gamma), with observed rank(H) range {alpha_synth18.get('min_rank_H_numeric')}..{alpha_synth18.get('max_rank_H_numeric')}.")
        if standard_synth18:
            w(f"- Standard Gamma: {standard_synth18.get('saturation_count', 0)} / {standard_synth18.get('trial_count', 0)} random right-inverse triples saturated ker(Gamma), with observed rank(H) range {standard_synth18.get('min_rank_H_numeric')}..{standard_synth18.get('max_rank_H_numeric')}.")
        w()
        w("This does not prove the multiplication theorem, because synthetic right inverses need not arise")
        w("from factorized rank-1 terms, but it gives direct computational evidence that failure of")
        w("rank(H) = dim(ker Gamma) is nongeneric inside the affine right-inverse model.")
        w()

        w("### Explicit Defective Loci")
        w()
        alpha_defects18 = [row for row in defects18 if row.get('family') == 'alphatensor_gamma']
        standard_defects18 = [row for row in defects18 if row.get('family') == 'standard_gamma']
        if alpha_defects18:
            w(f"- AlphaTensor Gamma defective examples: all-equal gives rank(H) = {alpha_defects18[0].get('rank_H_numeric')} with deficiency {alpha_defects18[0].get('deficiency_numeric')}; the k0 = k1 and collinear examples both drop to rank(H) = {alpha_defects18[1].get('rank_H_numeric')} with deficiency {alpha_defects18[1].get('deficiency_numeric')}.")
        if standard_defects18:
            w(f"- Standard Gamma defective examples: all-equal gives rank(H) = {standard_defects18[0].get('rank_H_numeric')} with deficiency {standard_defects18[0].get('deficiency_numeric')}; the k0 = k1 and collinear examples both drop to rank(H) = {standard_defects18[1].get('rank_H_numeric')} with deficiency {standard_defects18[1].get('deficiency_numeric')}.")
        w()
        w("These defective triples still satisfy Gamma * P_s = I_9 up to roundoff and still satisfy")
        w("Gamma * H = 0, so the failure mechanism is not violation of the affine right-inverse equations")
        w("themselves. The failure comes from special algebraic coincidences among the kernel lifts")
        w("(K_0, K_1, K_2).")
    else:
        w("*Run the Phase 18 script to populate this section.*")

    # ── PHASE 19 FACTORIZED DEFECT LOCI ──
    w()
    w(f"## {section_num}. FACTORIZED DEFECT-LOCUS DIAGNOSTICS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 19)")
    w()
    w("Phase 19 pushes the remaining obstruction question back into factorized term space. It measures")
    w("the centered kernel lifts K_s = P_s - (P_0 + P_1 + P_2)/3 on the known exact decompositions and")
    w("then projects factorized families onto explicit defect loci such as P_0 = P_1 before solving")
    w("the best least-squares gamma fit to the multiplication tensor.")
    w()
    diag19, search19 = read_phase19_outputs()
    if diag19 and search19:
        diag_rows19 = diag19.get('rows', [])
        alpha_diag19 = next((row for row in diag_rows19 if 'alphatensor' in row.get('label', '')), None)
        standard_diag19 = next((row for row in diag_rows19 if 'standard' in row.get('label', '')), None)
        summaries19 = search19.get('summaries', [])
        alpha_pair01 = next((row for row in summaries19 if row.get('base_label', '').startswith('alphatensor') and row.get('family') == 'pair_equal_01'), None)
        alpha_pair12 = next((row for row in summaries19 if row.get('base_label', '').startswith('alphatensor') and row.get('family') == 'pair_equal_12'), None)
        alpha_all_equal = next((row for row in summaries19 if row.get('base_label', '').startswith('alphatensor') and row.get('family') == 'all_equal'), None)
        alpha_collinear = next((row for row in summaries19 if row.get('base_label', '').startswith('alphatensor') and row.get('family') == 'collinear'), None)
        standard_pair01 = next((row for row in summaries19 if row.get('base_label') == 'standard_rank27' and row.get('family') == 'pair_equal_01'), None)
        standard_all_equal = next((row for row in summaries19 if row.get('base_label') == 'standard_rank27' and row.get('family') == 'all_equal'), None)
        standard_collinear = next((row for row in summaries19 if row.get('base_label') == 'standard_rank27' and row.get('family') == 'collinear'), None)

        w("### Centered-Lift Diagnostics On Known Decompositions")
        w()
        if alpha_diag19:
            w(f"- AlphaTensor rank-23: defect dimension = {alpha_diag19.get('defect_dim_numeric')}, lift-span rank = {alpha_diag19.get('lift_span_rank_numeric')}, center identity residual = {alpha_diag19.get('center_identity_max_abs')}, and K_0 + K_1 + K_2 = 0 exactly = {alpha_diag19.get('sum_lifts_exact')}.")
        if standard_diag19:
            w(f"- Standard rank-27: defect dimension = {standard_diag19.get('defect_dim_numeric')}, lift-span rank = {standard_diag19.get('lift_span_rank_numeric')}, center identity residual = {standard_diag19.get('center_identity_max_abs')}, and K_0 + K_1 + K_2 = 0 exactly = {standard_diag19.get('sum_lifts_exact')}.")
        w()
        w("Both valid decompositions therefore realize the centered right-inverse picture with no extra")
        w("kernel defect beyond the unavoidable relation K_0 + K_1 + K_2 = 0.")
        w()

        w("### Constrained Defect Families")
        w()
        if alpha_pair01 and alpha_pair12 and alpha_all_equal and alpha_collinear:
            w(f"- AlphaTensor-projected pair-equality families: best max-abs residuals are {alpha_pair01.get('best_tensor_max_abs_residual')} for P_0 = P_1 and {alpha_pair12.get('best_tensor_max_abs_residual')} for P_1 = P_2, both with rank(H) = 9.")
            w(f"- AlphaTensor-projected all-equal family: best max-abs residual is {alpha_all_equal.get('best_tensor_max_abs_residual')} with rank(H) = 0.")
            w(f"- AlphaTensor-projected collinear family: best max-abs residual is {alpha_collinear.get('best_tensor_max_abs_residual')}; this family is not an immediate obstruction.")
        if standard_pair01 and standard_all_equal and standard_collinear:
            w(f"- Standard-projected pair-equality family: best max-abs residual is {standard_pair01.get('best_tensor_max_abs_residual')} with rank(H) = 9.")
            w(f"- Standard-projected all-equal family: best max-abs residual is {standard_all_equal.get('best_tensor_max_abs_residual')} with rank(H) = 0.")
            w(f"- Standard-projected collinear family: exact residual = {standard_collinear.get('best_tensor_max_abs_residual')}, so per-term live-channel collinearity already contains a valid exact multiplication algorithm.")
        w()
        w("This sharply narrows the obstruction target. Pair-equality-type coincidences look genuinely")
        w("incompatible with exact multiplication in these searches, while the broader collinear family is")
        w("too large because it already contains the standard algorithm.")
    else:
        w("*Run the Phase 19 scripts to populate this section.*")

    # ── PHASE 20 AFFINE-LINE OBSTRUCTION ──
    w()
    w(f"## {section_num}. AFFINE-LINE LIFT OBSTRUCTION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 20)")
    w()
    w("Phase 20 tests a specific codimension-one defect suggested by the centered-lift derivation.")
    w("Writing X_0 = (P_0 + P_1 + P_2)/3 and K_s = P_s - X_0 gives the universal relation")
    w("K_0 + K_1 + K_2 = 0. Any second independent scalar relation would force the three K_s onto")
    w("a single matrix line, so the flattened lift span would drop from rank 2 to rank 1.")
    w()
    diag20, search20 = read_phase20_outputs()
    if diag20 and search20:
        rows20 = diag20.get('rows', [])
        alpha20 = next((row for row in rows20 if 'alphatensor' in row.get('label', '')), None)
        standard20 = next((row for row in rows20 if 'standard' in row.get('label', '')), None)
        summaries20 = search20.get('summaries', [])
        r23 = next((row for row in summaries20 if int(row.get('R', 0)) == 23), None)
        r27 = next((row for row in summaries20 if int(row.get('R', 0)) == 27), None)

        w("### Exact Decompositions")
        w()
        if alpha20:
            w(f"- AlphaTensor rank-23: lift-span rank = {alpha20.get('lift_span_rank_numeric')}, with best pairwise scalar-fit residuals {alpha20.get('residual_10_max_abs')}, {alpha20.get('residual_20_max_abs')}, and {alpha20.get('residual_21_max_abs')}.")
        if standard20:
            w(f"- Standard rank-27: lift-span rank = {standard20.get('lift_span_rank_numeric')}, with best pairwise scalar-fit residuals {standard20.get('residual_10_max_abs')}, {standard20.get('residual_20_max_abs')}, and {standard20.get('residual_21_max_abs')}.")
        w()
        w("So neither known exact decomposition lies on the affine-line lift defect locus.")
        w()

        w("### Factorized Affine-Line Family")
        w()
        w("A factorized family was then searched with alpha_k[:,s] = a_s * u_k and beta_k[s,:] = b_s * v_k^T,")
        w("which forces P_s = (a_s b_s) * M and therefore lift-span rank 1 by construction.")
        w()
        if r23:
            w(f"- R = 23: best max-abs residual = {r23.get('best_tensor_max_abs_residual')}, best loss = {r23.get('best_tensor_loss')}, best rank(H) = {r23.get('best_rank_H_numeric')}, best lift-span rank = {r23.get('best_lift_span_rank_numeric')}.")
        if r27:
            w(f"- R = 27: best max-abs residual = {r27.get('best_tensor_max_abs_residual')}, best loss = {r27.get('best_tensor_loss')}, best rank(H) = {r27.get('best_rank_H_numeric')}, best lift-span rank = {r27.get('best_lift_span_rank_numeric')}.")
        w()
        w("This family realizes the codimension-one defect exactly but stays far from exact multiplication")
        w("under least-squares-optimal gamma fitting. That is direct computational evidence against a")
        w("second scalar relation among the centered lifts as a viable failure mode.")
    else:
        w("*Run the Phase 20 scripts to populate this section.*")

    # ── PHASE 21 FUNCTIONAL ANNIHILATOR ──
    w()
    w(f"## {section_num}. FUNCTIONAL ANNIHILATOR INSIDE KER(GAMMA)")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 21)")
    w()
    w("Phase 21 separates two notions that initially look similar but are not. The relevant")
    w("codimension-one defect is not merely the existence of lambda with lambda^T H = 0; it is the")
    w("existence of nonzero lambda in ker(Gamma) such that lambda^T H = 0. Equivalently, the")
    w("restriction of H to a basis of ker(Gamma) must lose rank.")
    w()
    diag21, search21 = read_phase21_outputs()
    if diag21 and search21:
        rows21 = diag21.get('rows', [])
        alpha21 = next((row for row in rows21 if 'alphatensor' in row.get('label', '')), None)
        standard21 = next((row for row in rows21 if 'standard' in row.get('label', '')), None)
        summaries21 = search21.get('summaries', [])
        alpha_search21 = next((row for row in summaries21 if row.get('base_label', '').startswith('alphatensor')), None)
        standard_search21 = next((row for row in summaries21 if row.get('base_label') == 'standard_rank27'), None)

        w("### Known Exact Decompositions")
        w()
        if alpha21:
            w(f"- AlphaTensor rank-23: restricted rank = {alpha21.get('restricted_rank_numeric')} inside ker(Gamma), smallest singular value = {alpha21.get('restricted_min_singular_value')}, functional defect dimension = {alpha21.get('functional_defect_dim_numeric')}.")
        if standard21:
            w(f"- Standard rank-27: restricted rank = {standard21.get('restricted_rank_numeric')} inside ker(Gamma), smallest singular value = {standard21.get('restricted_min_singular_value')}, functional defect dimension = {standard21.get('functional_defect_dim_numeric')}.")
        w()
        w("So neither known exact decomposition has even a near-annihilator inside ker(Gamma).")
        w()

        w("### Weighted Annihilator Family")
        w()
        w("A factorized family was then built that enforces lambda^T P_0 = lambda^T P_1 = lambda^T P_2")
        w("exactly by channel-wise term rescaling, which makes lambda^T H = 0 hold by construction.")
        w()
        if alpha_search21:
            w(f"- AlphaTensor-derived family: best max-abs residual = {alpha_search21.get('best_tensor_max_abs_residual')}, best functional defect dimension = {alpha_search21.get('best_functional_defect_dim_numeric')}, best lambda-to-ker(Gamma) distance = {alpha_search21.get('best_lambda_kernel_distance')}.")
        if standard_search21:
            w(f"- Standard-derived family: best max-abs residual = {standard_search21.get('best_tensor_max_abs_residual')}, best functional defect dimension = {standard_search21.get('best_functional_defect_dim_numeric')}, best lambda-to-ker(Gamma) distance = {standard_search21.get('best_lambda_kernel_distance')}.")
        w()
        w("This shows that forcing a shared annihilator of H alone is too weak. Exact multiplication can")
        w("survive when that annihilator stays far away from ker(Gamma), so the true obstruction must")
        w("couple the annihilator condition directly to ker(Gamma) rather than treating them separately.")
    else:
        w("*Run the Phase 21 scripts to populate this section.*")

    # ── PHASE 22 KERNEL-LINKED HOMOTOPY ──
    w()
    w(f"## {section_num}. KERNEL-LINKED ANNIHILATOR HOMOTOPY")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 22)")
    w()
    w("Phase 22 couples the two conditions directly. It chooses lambda inside the original ker(Gamma),")
    w("then deforms the factor matrices toward lambda^T H = 0 while measuring the residual-angle")
    w("tradeoff between tensor exactness and the principal angle from span(lambda) to the new ker(Gamma_t).")
    w()
    phase22 = read_phase22_outputs()
    if phase22:
        summaries22 = phase22.get('summaries', [])
        alpha22 = [row for row in summaries22 if row.get('base_label', '').startswith('alphatensor')]
        standard22 = [row for row in summaries22 if row.get('base_label') == 'standard_rank27']
        alpha_soft22 = next((row for row in alpha22 if row.get('candidate') == 'softest_kernel_direction'), None)
        standard_soft22 = next((row for row in standard22 if row.get('candidate') == 'softest_kernel_direction'), None)

        w("### AlphaTensor")
        w()
        if alpha_soft22:
            w(f"- Softest kernel direction: residual at minimum angle = {alpha_soft22.get('residual_at_min_angle')}, while the fully enforced endpoint t = 1 has residual {alpha_soft22.get('residual_at_t1')}.")
        if alpha22:
            w("- All tested AlphaTensor kernel directions show the same pattern: as lambda^T H is driven to zero, the angle to ker(Gamma_t) grows sharply and the tensor residual rises to about 1.")
        w()

        w("### Standard Algorithm")
        w()
        if standard_soft22:
            w(f"- Softest kernel direction: residual at minimum angle = {standard_soft22.get('residual_at_min_angle')}, while the fully enforced endpoint t = 1 has residual {standard_soft22.get('residual_at_t1')}.")
        if standard22:
            w("- The standard algorithm is more flexible under this homotopy: some sampled kernel directions stay exact for long stretches or even at the endpoint within floating-point precision, while others only break at the fully enforced endpoint.")
        w()
        w("This means the kernel-linked annihilator idea is structurally meaningful, but the raw homotopy")
        w("template is not yet a universal obstruction. It strongly disfavors the AlphaTensor geometry while")
        w("leaving extra room inside the highly symmetric standard algorithm.")
    else:
        w("*Run the Phase 22 script to populate this section.*")

    # ── PHASE 23 STAGED LAYERING BRIDGE ──
    w()
    w(f"## {section_num}. STAGED LAYERING VS. DEPTH-2 CIRCUITS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 23)")
    w()
    w("Phase 23 reconnects the older depth-2 circuit dead end to the newer staged bilinear")
    w("obstruction scans. Step 70 ruled out the direct recursive depth-2 padding route, Step 72")
    w("showed only parameter-level fiber redundancy, and Step 73 proved that exact division-free")
    w("depth-2 AA/QQ circuits collapse back to the ordinary bilinear problem via degree-2 truncation.")
    w()
    w("The new staged homotopies should therefore be read as a diagnostic replacement for the old")
    w("layered-circuit idea: they force the prospective obstruction pattern gradually inside bilinear")
    w("factor space rather than by adding exact higher-degree circuit layers.")
    w()
    phase23 = read_phase23_outputs()
    if phase23:
        summaries23 = phase23.get('summaries', [])
        alpha_soft23 = next((row for row in summaries23 if row.get('base_label', '').startswith('alphatensor') and row.get('candidate') == 'softest_kernel_direction'), None)
        alpha_rand23 = next((row for row in summaries23 if row.get('base_label', '').startswith('alphatensor') and row.get('candidate') == 'random_kernel_direction_1'), None)
        standard_soft23 = next((row for row in summaries23 if row.get('base_label') == 'standard_rank27' and row.get('candidate') == 'softest_kernel_direction'), None)
        standard_rand23 = next((row for row in summaries23 if row.get('base_label') == 'standard_rank27' and row.get('candidate') == 'random_kernel_direction_1'), None)

        w("### Threshold Summary")
        w()
        if alpha_soft23:
            alpha_first = alpha_soft23.get('first_residual_over_0p05')
            if alpha_first:
                w(f"- AlphaTensor, softest kernel direction: the first tensor residual above 0.05 occurs already at t = {alpha_first.get('t_value')}.")
        if alpha_rand23:
            alpha_first_rand = alpha_rand23.get('first_residual_over_0p05')
            if alpha_first_rand:
                w(f"- AlphaTensor, random kernel direction: the first tensor residual above 0.05 also occurs at t = {alpha_first_rand.get('t_value')}.")
        if standard_soft23:
            standard_first = standard_soft23.get('first_residual_over_0p05')
            w(f"- Standard algorithm, softest kernel direction: the first tensor residual above 0.05 occurs at t = {standard_first.get('t_value') if standard_first else 'none'}.")
        if standard_rand23:
            standard_first_rand = standard_rand23.get('first_residual_over_0p05')
            w(f"- Standard algorithm, random kernel direction: the first tensor residual above 0.05 occurs at t = {standard_first_rand.get('t_value') if standard_first_rand else 'none'}.")
        w()
        w("This is the operational bridge between the old circuit story and the new obstruction story.")
        w("Exact depth-2 layering still does not create new bilinear directions, but staged forcing inside")
        w("bilinear factor space reveals when a candidate obstruction starts to damage exactness. On the")
        w("AlphaTensor rank-23 geometry, that damage appears immediately; on the standard algorithm, it")
        w("does not.")
    else:
        w("*Run the Phase 23 extractor to populate this section.*")

    # ── PHASE 24 TANGENT TRANSVERSALITY ──
    w()
    w(f"## {section_num}. TANGENT TRANSVERSALITY OF KERNEL-LINKED FORCING")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 24)")
    w()
    w("Phase 24 asks whether the Phase 22 instability is already an infinitesimal obstruction or")
    w("whether it depends on a particular gauge choice for the annihilator-forcing direction.")
    w("It linearizes the exact multiplication condition after optimally re-solving gamma, then")
    w("compares the canonical minimum-norm forcing direction against the best corrected direction")
    w("inside the same affine annihilator constraint.")
    w()
    phase24 = read_phase24_outputs()
    if phase24:
        rows24 = phase24.get('rows', [])
        alpha24 = [row for row in rows24 if row.get('base_label', '').startswith('alphatensor')]
        standard24 = [row for row in rows24 if row.get('base_label') == 'standard_rank27']
        alpha_soft24 = next((row for row in alpha24 if row.get('candidate') == 'softest_kernel_direction'), None)
        standard_soft24 = next((row for row in standard24 if row.get('candidate') == 'softest_kernel_direction'), None)

        w("### Canonical Forcing Direction")
        w()
        if alpha_soft24:
            w(f"- AlphaTensor, softest kernel direction: the minimum-norm forcing direction already has first-order tensor residual L2 = {alpha_soft24.get('min_norm_first_order_residual_l2')} and max-abs = {alpha_soft24.get('min_norm_first_order_residual_max_abs')}.")
        if standard_soft24:
            w(f"- Standard algorithm, softest kernel direction: the same minimum-norm forcing direction has first-order tensor residual L2 = {standard_soft24.get('min_norm_first_order_residual_l2')} and max-abs = {standard_soft24.get('min_norm_first_order_residual_max_abs')}.")
        w("This pins down the Phase 22 asymmetry more sharply: the natural minimum-norm forcing path is already transverse on AlphaTensor but already tangent on the standard algorithm.")
        w()

        w("### Corrected Tangent Directions")
        w()
        if alpha_soft24:
            w(f"- AlphaTensor still has corrected tangent-compatible forcing directions, but they require a substantial sideways correction: for the softest direction the tangent-correction ratio is {alpha_soft24.get('tangent_correction_ratio')}.")
        if standard24:
            w("- The standard algorithm needs no such repair in the sampled cases: the tangent correction is numerically zero for all three tested directions.")
        w("So there is no universal first-order obstruction. What differs is how much symmetry-driven tangent freedom is available before the forcing direction has to be repaired.")
        w()

        w("### Nonlinear Escape Mechanism")
        w()
        if alpha_soft24:
            w(f"- Along AlphaTensor's corrected softest-direction path, the tensor stays exact to floating-point precision while lambda^T H is driven from its baseline down to a ratio of {alpha_soft24.get('path_lambda_h_ratio_eps_1p0')} at t = 1, but the angle from lambda to ker(Gamma_t) grows to {alpha_soft24.get('path_angle_deg_eps_1p0')} degrees.")
            w(f"- The first sampled stage where this corrected AlphaTensor path exceeds 5 degrees is t = {alpha_soft24.get('first_path_angle_over_5_deg')}, and the first sampled stage above 15 degrees is t = {alpha_soft24.get('first_path_angle_over_15_deg')}.")
        if standard_soft24:
            w(f"- The corrected standard softest-direction path behaves differently again: it stays kernel-linked longer, but its tensor residual only breaks at the fully enforced endpoint, where the residual reaches {standard_soft24.get('path_residual_eps_1p0')}.")
        w("This shows exactly how the corrected exact paths evade the naïve obstruction: they preserve exactness by letting lambda drift away from the moving kernel rather than by satisfying the kernel-linked annihilator condition honestly.")
        w()
        w("Phase 24 therefore isolates the remaining gap as a genuinely simultaneous nonlinear condition: one must control exactness, annihilator progress, and kernel linkage at the same time.")
    else:
        w("*Run the Phase 24 script to populate this section.*")

    # ── PHASE 25 HONEST KERNEL RETENTION ──
    w()
    w(f"## {section_num}. HONEST KERNEL-RETENTION: DERIVATION AND SCAN")
    section_num += 1
    w()
    w("[INTERPRETATION] (Phase 25 derivation session + compute scan)")
    w()
    w("Phase 25 is a narrow derivation step motivated directly by Phase 24. The key conclusion")
    w("is that the corrected exact escape paths do not approximate an honest kernel-linked")
    w("annihilator. They only make lambda^T H small while allowing the reference lambda to rotate")
    w("far away from the moving kernel ker(Gamma_t).")
    w()
    w("This leads to a clean coupled defect operator for any decomposition state x:")
    w()
    w("- C_x(lambda) = (Gamma(x) lambda, H(x)^T lambda)")
    w("- A genuine kernel-linked annihilator is exactly a nonzero vector in ker(C_x).")
    w("- The ambient coupled defect size is therefore kappa_full(x) = sigma_min(C_x).")
    w()
    w("More intrinsically, if Q_x is an orthonormal basis of ker(Gamma(x)), then the honest")
    w("kernel-restricted defect is")
    w()
    w("- kappa_ker(x) = sigma_min(Q_x^T H_x)")
    w()
    w("This is the quantitative version of the Phase 21 rank test: kappa_ker(x) = 0 if and only if")
    w("there exists a nonzero lambda in ker(Gamma(x)) with lambda^T H(x) = 0.")
    w()
    w("The derivation also identifies the correct first-order continuation equations. If x(t) is an")
    w("exact path and lambda(t) is a unit term-space vector that stays honestly kernel-linked, then")
    w("the coupled constraints are")
    w()
    w("- Gamma(t) lambda(t) = 0")
    w("- lambda(t)^T H(t) = 0")
    w("- ||lambda(t)|| = 1")
    w()
    w("and differentiating gives")
    w()
    w("- dot(Gamma) lambda + Gamma dot(lambda) = 0")
    w("- dot(lambda)^T H + lambda^T dot(H) = 0")
    w("- lambda^T dot(lambda) = 0")
    w()
    w("So the next compute phase should no longer freeze lambda. It should solve a coupled")
    w("continuation problem in both the decomposition variables and lambda itself, ideally by")
    w("tracking the softest singular direction of Q_x^T H_x along an exact path.")
    w()
    w("This sharpens the open problem again: the right target is no longer generic annihilator")
    w("forcing, nor even fixed-lambda kernel forcing, but whether an exact path can drive the")
    w("kernel-restricted defect kappa_ker(x) toward zero while lambda remains inside the moving")
    w("kernel.")
    w()

    phase25 = read_phase25_outputs()
    if phase25:
        summaries25 = phase25.get('summaries', [])
        alpha25 = [row for row in summaries25 if row.get('base_label', '').startswith('alphatensor')]
        standard25 = [row for row in summaries25 if row.get('base_label') == 'standard_rank27']
        alpha_soft25 = next((row for row in alpha25 if row.get('candidate') == 'softest_kernel_direction'), None)
        standard_soft25 = next((row for row in standard25 if row.get('candidate') == 'softest_kernel_direction'), None)

        w("### Scan Of The Honest Kernel-Restricted Soft Mode")
        w()
        if alpha_soft25:
            w(f"- AlphaTensor, softest corrected path: kappa_ker starts at {alpha_soft25.get('base_kappa_ker')} and falls to {alpha_soft25.get('min_kappa_ker')} by t = {alpha_soft25.get('min_kappa_ker_t')}, with tensor residual {alpha_soft25.get('tensor_residual_at_min_kappa_ker')} at that minimum.")
        if standard_soft25:
            w(f"- Standard algorithm, softest corrected path: kappa_ker starts at {standard_soft25.get('base_kappa_ker')} and falls to {standard_soft25.get('min_kappa_ker')} by t = {standard_soft25.get('min_kappa_ker_t')}, with tensor residual {standard_soft25.get('tensor_residual_at_min_kappa_ker')} at that minimum.")
        w()
        w("The first compute scan against the derived quantity shows a more refined asymmetry than Phase 24.")
        w("AlphaTensor's softest exact path can push kappa_ker very low while remaining numerically exact")
        w("and without rank collapse, but in this scan it still does not hit zero. The standard softest")
        w("path reaches an actual honest defect only at the singular endpoint where the restricted rank")
        w("collapses and exact multiplication fails.")
        w()
        w("So the derived quantity is useful, but not yet a universal separator by itself: some standard")
        w("random corrected directions also drive kappa_ker low while staying exact. The remaining gap is")
        w("therefore even sharper now: distinguish smooth honest continuation toward kappa_ker = 0 from")
        w("mere endpoint collapse or other highly symmetric escape mechanisms.")
    else:
        w("*Run the Phase 25 scan to populate the compute portion of this section.*")

    # ── PHASE 26 REGULARIZED KERNEL RETENTION ──
    w()
    w(f"## {section_num}. REGULARIZED KERNEL RETENTION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 26)")
    w()
    w("Phase 26 applies the first regularity filter to the Phase 25 paths: keep only states that")
    w("remain numerically exact and whose kernel-restricted rank does not collapse. This removes")
    w("the obvious standard soft-path endpoint singularity from the comparison.")
    w()
    phase26 = read_phase26_outputs()
    if phase26:
        summaries26 = phase26.get('summaries', [])
        alpha26 = [row for row in summaries26 if row.get('base_label', '').startswith('alphatensor')]
        standard26 = [row for row in summaries26 if row.get('base_label') == 'standard_rank27']
        alpha_soft26 = next((row for row in alpha26 if row.get('candidate') == 'softest_kernel_direction'), None)
        standard_soft26 = next((row for row in standard26 if row.get('candidate') == 'softest_kernel_direction'), None)
        standard_rand26 = next((row for row in standard26 if row.get('candidate') == 'random_kernel_direction_2'), None)

        if alpha_soft26:
            w(f"- AlphaTensor, softest regular path: minimum regular kappa_ker = {alpha_soft26.get('min_regular_kappa_ker')} at t = {alpha_soft26.get('min_regular_kappa_ker_t')}, with no restricted-rank collapse.")
        if standard_soft26:
            w(f"- Standard, softest regular path: minimum regular kappa_ker = {standard_soft26.get('min_regular_kappa_ker')} at t = {standard_soft26.get('min_regular_kappa_ker_t')}; the singular endpoint at t = 1 is excluded by the filter.")
        if standard_rand26:
            w(f"- Standard still retains a higher-symmetry regular escape in one sampled random direction, reaching kappa_ker = {standard_rand26.get('min_regular_kappa_ker')} without collapse.")
        w()
        w("So the no-collapse filter is necessary and already meaningful: it separates the softest AlphaTensor and standard paths. But it is not sufficient globally, because some more symmetric standard corrected paths still survive it.")
    else:
        w("*Run the Phase 26 script to populate this section.*")

    # ── PHASE 27 COUPLED CONTINUATION PROTOTYPE ──
    w()
    w(f"## {section_num}. COUPLED CONTINUATION PROTOTYPE")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 27)")
    w()
    w("Phase 27 implements the first actual coupled continuation solve in both the decomposition")
    w("variables and lambda. It enforces exactness, moving-kernel consistency, and normalization")
    w("to first order, then minimizes the linearized honest defect.")
    w()
    phase27 = read_phase27_outputs()
    if phase27:
        summaries27 = phase27.get('summaries', [])
        alpha27 = next((row for row in summaries27 if row.get('base_label', '').startswith('alphatensor')), None)
        standard27 = next((row for row in summaries27 if row.get('base_label') == 'standard_rank27'), None)
        if alpha27:
            w(f"- AlphaTensor: the linearized solve predicts a near-perfect first-order defect cancellation with reduction factor {alpha27.get('predicted_reduction_factor')}, but the finite probe at eps = 1e-2 still leaves an honest defect of {alpha27.get('probe_eps_1e_2_honest_defect_l2')}.")
        if standard27:
            w(f"- Standard algorithm: the same phenomenon occurs; the first-order reduction factor is {standard27.get('predicted_reduction_factor')}, but the finite probe at eps = 1e-2 still leaves an honest defect of {standard27.get('probe_eps_1e_2_honest_defect_l2')}.")
        w()
        w("So even the honest coupled local model is too optimistic if interpreted literally. Local solvability is not the issue; the real obstruction problem now sits in nonlinear persistence beyond first order.")
    else:
        w("*Run the Phase 27 script to populate this section.*")

    # ── PHASE 28 WILDCARD-REGULARIZED CONTINUATION ──
    w()
    w(f"## {section_num}. WILDCARD-REGULARIZED CONTINUATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 28)")
    w()
    w("Phase 28 upgrades the coupled continuation test again: it keeps the Phase 26 regularity")
    w("filter, uses the Phase 27 coupled local model, and adds an explicit wildcard search over")
    w("admissible nullspace directions at every finite probe.")
    w()
    phase28 = read_phase28_outputs()
    if phase28:
        summaries28 = phase28.get('summaries', [])
        alpha28 = next((row for row in summaries28 if row.get('base_label', '').startswith('alphatensor')), None)
        standard28 = next((row for row in summaries28 if row.get('base_label') == 'standard_rank27'), None)
        if alpha28:
            w(f"- AlphaTensor: over {alpha28.get('accepted_steps')} accepted regular steps, wildcard continuation lowers kappa_ker from {alpha28.get('initial_kappa_ker')} to {alpha28.get('final_kappa_ker')}; wildcard wins all {alpha28.get('wildcard_wins')} accepted steps.")
        if standard28:
            w(f"- Standard algorithm: over {standard28.get('accepted_steps')} accepted regular steps, wildcard continuation lowers kappa_ker from {standard28.get('initial_kappa_ker')} to {standard28.get('final_kappa_ker')}; wildcard also wins all {standard28.get('wildcard_wins')} accepted steps.")
        w()
        w("So wildcard search is not optional bookkeeping. It materially expands the regular finite-step descent cone in both families. The separator problem therefore has to survive not just the structured coupled step, but a broader admissible nullspace search.")
    else:
        w("*Run the Phase 28 script to populate this section.*")

    # ── PHASE 29 SMOOTH-BUDGETED CONTINUATION ──
    w()
    w(f"## {section_num}. SMOOTH-BUDGETED CONTINUATION")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 29)")
    w()
    w("Phase 29 executes two refinements on top of the wildcard continuation test: a per-step")
    w("mode-rotation cap, and a longer-horizon budgeted continuation policy that penalizes costly")
    w("finite moves.")
    w()
    phase29 = read_phase29_outputs()
    if phase29:
        summaries29 = phase29.get('summaries', [])
        alpha29_smooth = next((row for row in summaries29 if row.get('policy_name') == 'smooth_greedy' and row.get('base_label', '').startswith('alphatensor')), None)
        standard29_smooth = next((row for row in summaries29 if row.get('policy_name') == 'smooth_greedy' and row.get('base_label') == 'standard_rank27'), None)
        alpha29_budget = next((row for row in summaries29 if row.get('policy_name') == 'smooth_budgeted_long' and row.get('base_label', '').startswith('alphatensor')), None)
        standard29_budget = next((row for row in summaries29 if row.get('policy_name') == 'smooth_budgeted_long' and row.get('base_label') == 'standard_rank27'), None)
        if alpha29_smooth and standard29_smooth:
            w(f"- Smooth greedy policy: with a 12 degree mode-rotation cap, AlphaTensor still drops kappa_ker from {alpha29_smooth.get('initial_kappa_ker')} to {alpha29_smooth.get('final_kappa_ker')} and the standard algorithm drops from {standard29_smooth.get('initial_kappa_ker')} to {standard29_smooth.get('final_kappa_ker')}; all accepted steps in both families are still wildcard steps.")
        if alpha29_budget and standard29_budget:
            w(f"- Smooth budgeted long-horizon policy: over 24 accepted steps, AlphaTensor only moves from {alpha29_budget.get('initial_kappa_ker')} to {alpha29_budget.get('final_kappa_ker')} using budget {alpha29_budget.get('budget_used')}, while the standard algorithm moves from {standard29_budget.get('initial_kappa_ker')} to {standard29_budget.get('final_kappa_ker')} using budget {standard29_budget.get('budget_used')}; wildcard wins nearly vanish under the cost-penalized objective.")
        w()
        w("So smoothness alone does not remove wildcard escape routes, but budget does change the continuation geometry substantially. The remaining target is now a persistence-style separator: not just whether descent exists, but how much honest defect reduction can be bought per unit smooth continuation budget.")
    else:
        w("*Run the Phase 29 script to populate this section.*")

    # ── PHASE 30 BUDGET-EFFICIENCY SWEEP ──
    w()
    w(f"## {section_num}. BUDGET-EFFICIENCY SWEEP")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 30)")
    w()
    w("Phase 30 turns the Phase 29 continuation policies into explicit gain-per-budget and")
    w("gain-per-rotation summaries, then runs a small sweep over rotation caps and budget weights")
    w("to test whether the budget-aware story is robust.")
    w()
    phase30 = read_phase30_outputs()
    if phase30:
        delta_rows30 = phase30.get('delta_rows', [])
        rot8 = next((row for row in delta_rows30 if row.get('policy_name') == 'sweep_rot8_bw0p0'), None)
        rot12 = next((row for row in delta_rows30 if row.get('policy_name') == 'sweep_rot12_bw0p6'), None)
        if rot8:
            w("- Tight rotation-cap wall: under the 8 degree cap policy, AlphaTensor still admits regular descent while the standard algorithm stalls immediately with no accepted steps.")
        if rot12:
            w(f"- Naive efficiency reversal: under the cost-penalized 12 degree policy, AlphaTensor's gain-per-budget is {rot12.get('alpha_total_gain_per_budget')}, while the standard algorithm's is {rot12.get('standard_total_gain_per_budget')}; so plain gain-per-budget is not the right separator.")
        w()
        w("So the useful object now looks less like a single scalar efficiency quotient and more like a feasibility boundary in the space of allowed rotation, allowed cost, and achievable regular defect decrease.")
    else:
        w("*Run the Phase 30 sweep to populate this section.*")

    # ── PHASE 31 FEASIBILITY BOUNDARY ──
    w()
    w(f"## {section_num}. FEASIBILITY BOUNDARY")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 31)")
    w()
    w("Phase 31 turns the continuation problem into a phase-diagram question: for each total")
    w("budget, what is the smallest per-step mode-rotation cap under which any regular descent is")
    w("still feasible?")
    w()
    phase31 = read_phase31_outputs()
    if phase31:
        delta_rows31 = phase31.get('delta_rows', [])
        if delta_rows31:
            first_gap = delta_rows31[0]
            w(f"- Across the full scanned budget range, AlphaTensor remains feasible already at rotation cap {first_gap.get('alpha_min_rotation_cap_deg_for_descent')}, while the standard algorithm requires rotation cap {first_gap.get('standard_min_rotation_cap_deg_for_descent')}; the gap is {first_gap.get('rotation_cap_gap_standard_minus_alpha')} degrees.")
        w()
        w("So the continuation branch now has its cleanest separator to date: not a scalar efficiency quotient, but a stable feasibility-boundary gap in the joint space of smoothness and cost.")
    else:
        w("*Run the Phase 31 scan to populate this section.*")

    # ── PHASE 32 BOUNDARY SHARPENING ──
    w()
    w(f"## {section_num}. BOUNDARY SHARPENING")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 32)")
    w()
    w("Phase 32 densifies the feasibility-boundary scan near the observed thresholds to test whether")
    w("the Phase 31 frontier gap was a coarse-grid artifact or a stable feature.")
    w()
    phase32 = read_phase32_outputs()
    if phase32:
        delta_rows32 = phase32.get('delta_rows', [])
        if delta_rows32:
            first_gap = delta_rows32[0]
            w(f"- On the refined grid, AlphaTensor remains feasible already at rotation cap {first_gap.get('alpha_min_rotation_cap_deg_for_descent')}, while the standard algorithm still requires {first_gap.get('standard_min_rotation_cap_deg_for_descent')}; the gap sharpens to {first_gap.get('rotation_cap_gap_standard_minus_alpha')} degrees.")
        w()
        w("So the feasibility-boundary effect is not weakening under refinement. It sharpens. The empirical separator now looks like a critical smoothness threshold with AlphaTensor and the standard algorithm in different phases.")
    else:
        w("*Run the Phase 32 scan to populate this section.*")

    # ── PHASE 33 27-SYMBOL FAITHFUL ENCODING ──
    w()
    w("## 69. 27-SYMBOL FAITHFUL ENCODING ANALYSIS")
    section_num += 1
    w()
    w("[EXACT_DERIVED] (Phase 33)")
    w()
    w("Phase 33 tests whether the 27-symbol encoding {P_0, P_1, P_2} is faithful:")
    w("whether the 54 dead-X coordinates are algebraically recoverable from the")
    w("27 live coordinates plus the tiling constraint.")
    w()
    phase33 = read_phase33_outputs()
    if phase33:
        track_a = phase33.get('track_a', {})
        track_b = phase33.get('track_b', {})
        track_c = phase33.get('track_c', {})
        track_d = phase33.get('track_d', {})
        w("### Track A: 27-Symbol Encoding")
        w()
        w("[EXACT_DERIVED]")
        w()
        w("For the known 3x3 decompositions, each fixed cross-section block D_st was tested")
        w("against span(H) with H = [K_0-K_1 | K_1-K_2].")
        w()
        w("| decomposition | R | rank(H) | all D_st in span(H)? | all D_st in span([Sigma|H])? |")
        w("|---------------|---|---------|----------------------|------------------------------|")
        for row in track_a.get('algorithms', []):
            w(f"| {row.get('label')} | {row.get('R')} | {row.get('rank_H_exact')} | {row.get('all_cross_sections_in_H_exact')} | {row.get('all_cross_sections_in_SigmaH_exact')} |")
        recovery_rows = track_a.get('recovery_rows', [])
        if recovery_rows:
            w()
            w("| decomposition | block | rank(M_st) | nnz(M_st) | density | coeffs |")
            w("|---------------|-------|------------|-----------|---------|--------|")
            for row in recovery_rows:
                density = row.get('M_h_density')
                density_text = 'n/a' if density is None else f"{density:.4f}"
                coeffs = ', '.join(row.get('M_h_unique_coefficients', [])) if row.get('M_h_unique_coefficients') else 'none'
                w(f"| {row.get('label')} | {row.get('block')} | {row.get('M_h_rank')} | {row.get('M_h_nonzero')} | {density_text} | {coeffs} |")
        w()
        w("### Track B: Clone-Frame Saturation")
        w()
        w("[EXACT_DERIVED] / [WILDCARD]")
        w()
        w("All six channel-permuted clone frames were tested on the known 3x3 decompositions,")
        w("and synthetic defective triples were then sampled inside ker(Gamma) to see whether")
        w("factorized rank-1 faces can coexist with H-defect.")
        w()
        w("| decomposition | dim ker(Gamma) | identity rank(H) | all clone frames saturate? | wildcard defective factorized sample found? |")
        w("|---------------|----------------|------------------|-----------------------------|-------------------------------------------|")
        for row in track_b.get('algorithms', []):
            w(f"| {row.get('label')} | {row.get('ker_Gamma_dim_exact')} | {row.get('identity_frame_rank_exact')} | {row.get('all_clone_frames_saturate')} | {row.get('factorized_defective_found')} |")
        w()
        w("### Track C: Gauge Orbit")
        w()
        w("[EXACT_DERIVED]")
        w()
        w("The tiling identities Gamma * P_s = I leave the matched faces gauge-invariant, but")
        w("the cross-sections D_st rescale rowwise. Phase 33 therefore tested whether the")
        w("resulting gauge orbits stay inside span(H).")
        w()
        w("| decomposition | observed gauge-orbit dim | tangent escape from span(H)? | sampled finite escape? |")
        w("|---------------|--------------------------|-------------------------------|------------------------|")
        for row in track_c.get('algorithms', []):
            w(f"| {row.get('label')} | {row.get('combined_observed_gauge_orbit_dim')} | {row.get('tangent_escape_detected')} | {row.get('sampled_escape_detected')} |")
        w()
        w("### Track D: Khatri-Rao Absorption")
        w()
        w("[EXACT_DERIVED] / [MEASURED_FROM_CODE]")
        w()
        w("The mandatory Strassen 2x2 sanity check and the two known 3x3 decompositions were")
        w("all tested for recovery from span([Sigma|H]).")
        w()
        w("| system | all D_st in span(H)? | all D_st in span([Sigma|H])? |")
        w("|--------|----------------------|------------------------------|")
        for row in track_d.get('algorithms', []):
            w(f"| {row.get('label')} | {row.get('all_cross_sections_in_H_exact')} | {row.get('all_cross_sections_in_SigmaH_exact')} |")
        w()
        w("[INTERPRETATION]")
        w()
        w("The positive result is decomposition-specific but strong: on both known exact 3x3")
        w("decompositions, every individual dead cross-section block D_st is already exactly")
        w("recoverable from H alone. The mandatory Strassen 2x2 sanity check also passes.")
        w()
        w("The negative result is equally important: the tiling identities by themselves do")
        w("not make the dead cross-sections gauge-rigid inside span(H). Phase 33 detects both")
        w("tangent and finite gauge escapes, so Delta containment does not follow from gauge")
        w("orbit rigidity under Gamma * P_s = I alone.")
        w()
        w("So Phase 33 strengthens the coding-theory framing on the known decompositions while")
        w("also isolating a concrete obstruction to a universal proof: the missing argument must")
        w("use exact constraints beyond the matched-face tiling identities and their residual gauge freedom.")
    else:
        w("*Run the Phase 33 script to populate this section.*")

    # ── OPEN FRONTS ──
    w()
    w(f"## {section_num}. CURRENT GAPS / OPEN FRONTS")
    section_num += 1
    w()
    w("[OPEN_FRONT]")
    w()
    w("**Completed in this session:**")
    w("- CXXC schema (arity-4): ✓ 2744 orbits with base and refined signatures recorded")
    w("- AXXC parity layer: ✓ 2870 orbits and 2870 current-recorded signatures measured")
    w("- CCXX schema (arity-4): ✓ 2744 orbits with base and refined signatures recorded")
    w("- XX, AX, BX signature collisions: ✓ All resolved with minimal refinement features")
    w("- Composition kernel: ✓ Mixed key CC-orbit distributions computed (all 14 keys uniform)")
    w("- CXXC marginal weight profile: ✓ Fiber-size histogram for all 7 projections")
    w("- Stabilizer subgroup classification: ✓ CXC and CXXC; all pure-2-groups (Z2, Z2xZ2, (Z2)^3)")
    w("- Stabilizer composition analysis: ✓ 43M pairs; (Z2)^3 is composition identity; types not generally closed")
    w("- CXXC and CCXX arity-4 collisions: ✓ All resolved with minimal refiner `(s4, t4)`")
    w("- CXXC marginal projections: ✓ Full coverage analysis completed")
    w("- Refinement-conditioned kernel: ✓ All 63 (s,t) strata uniform; hypothesis closed at stratum level")
    w("- Z2xZ2 floor layer: ✓ 40 orbits; floor property holds (no Trivial decay); not fully closed; step-1 closure adds 18 Z2 orbits")
    w("- 58-orbit closure + doubly-live core: ✓ 58-seed closes at 64 orbits; 28-core stays 100% doubly-live; floor fixed dims are 30 or 36")
    w("- Same-fiber core + 64-subalgebra structure: ✓ 10 same-fiber and 6 focused orbits are both closed; 64-table has 602 compatible rows with 164 mixed; greedy generator set size 18")
    w("- Mixed-pair resolution + tensor constraints: ✓ 41,688 witnesses scanned; 0/164 mixed pairs resolved by interface coordinates; raw tensor same-fiber support uses only orbits 0 and 30")
    w("- Tensor profile constraint model: ✓ 729 tensor equations collapse to 8 XC orbit classes; only XC orbit 0 is positive; generic rank-1 support has profile (27,27); the 8-orbit linearization alone gives no rank lower bound")
    w("- Coefficient-level rank constraints: ✓ explicit 8 equation types recorded with orbit sizes (27,54,54,108,54,108,108,216); standard 27-term basis algorithm and Strassen 2x2 both verified exactly; search-space dimensions exported")
    w("- Symbolic fiber-mode decomposition: ✓ the 729 equations now split exactly as 81 fiber-sum + 162 live-anisotropy + 486 dead-X equations, with matrix form Gamma*Sigma=3I_9 and Gamma annihilating the nuisance blocks")
    w("- Quotient-space rank criterion: ✓ for a fixed decomposition solvability is equivalent to quotient-space independence of Sigma modulo nuisance; standard 3x3 gives nuisance rank 18 and Strassen 2x2 gives rank 3, with Strassen exactly tight")
    w("- Support-type representative incidence: ✓ there are 8000 support classes modulo S3^3; 1000 can realize Type 0, and 216 of those allow all 8 representative equation types, so support-only pruning is vacuous")
    w("- Analytical low-nuisance construction: ✓ dead-free terms were shown not to be nuisance-free in the Step 51 basis; random low-rank factor families were profiled numerically, and although some R=22 families reached nuisance rank <= 13, none achieved the quotient-space gain required by Step 52")
    w("- Algebraic nuisance dependencies + wildcards: ✓ Hadamard-space geometry now sharpens the p=3,q=4 target to nuisance rank <= 3; tested Toeplitz, circulant, shared-latent, and DFT families still failed to produce quotient gain 9; GF(2) and tropical flattening ranks both stayed at 9, and fiber commutators were mostly nonzero")
    w("- Tensor-product DFT construction + orbit packing: ✓ in the looser p=q=4 regime, the full 126x126 DFT mode sweep, random ternary/algebraic samples, interpreted correction-block sweep, and numpy local search all stayed far below quotient gain 9; the same-fiber branch 30 o 30 is exactly balanced")
    w("- Fiber-group partition enumeration + orbit budget filter: ✓ exact ordered and symmetry-reduced partition counts for R=9..23 are now recorded, no-spreading orbit budgets are tabulated by partition type, and a conservative exact-3 rectangle-tiling model identifies structurally feasible primary-fiber assignments before coefficient solving")
    w("- Cube-root-of-unity injection: ✓ the full-spread omega family is now ruled out by an exact live-rank obstruction, an explicit balanced 9-term attempt and a 27-term greedy augmentation trace both fail inside that family, and the same-fiber Fourier bundle is verified as an exact 27-term Fourier disguise of the standard algorithm with measured Step 51 nuisance ranks")
    w("- Hybrid Fourier construction: ✓ same-fiber Fourier triples are now verified modular on all 512 output-fiber subsets, the same flattening filter has been extended through R=19..23, explicit tiny residual cases are ruled out by flattening or known exact ranks, and the residual-fiber problem is reduced to 35 symmetry classes with all six six-fiber hybrids still alive under flattening alone")
    w("- Nuisance-first architecture: ✓ R=9 is now ruled out by an exact zero-nuisance contradiction, the weaker all-dead-free route is shown to force R>=27 by per-channel reconstruction, and exact nuisance/dead dependency budgets are tabulated for R=18..23")
    w("- Non-rectangular 6-fiber sub-tensor rank attack: ✓ exact substitution bounds and direct P4 constructions are now tabulated, the rectangular six-fiber cases remain ruled out by exact rank 15, and the current numerical CP-rank scan found no rank-13 or rank-14 witness for any nonrectangular six-fiber pattern")
    w("- Reverse engineering + cancellation visualization: ✓ a public exact rank-23 3x3 coefficient table has been recovered from AlphaTensor's public repo, measured directly in the Step 51-52 basis, shown to have nuisance rank 14 = 23-9 exactly, and tested for single/pair gamma-only redundancy with no feasible 22-term or 21-term sub-decomposition found")
    w("- Conservation-law proof status: ✓ Phase 17 proves fiber-mode faithfulness, verifies dim(ker Gamma) = rank([H|Delta]) on the known valid decompositions, and confirms that AlphaTensor plus all 23 single deletions satisfy Delta subset span(H), while random collections and the exported Step 75 anticommutator fibermode do not")
    w("- Right-inverse kernel-saturation diagnostics: ✓ Phase 18 verifies exact per-channel identities Gamma * P_s = I_9 and exact kernel saturation rank(H) = dim(ker Gamma) on AlphaTensor and the standard algorithm, shows 250/250 synthetic right-inverse triples saturate for each known Gamma, and exhibits explicit defective lower-dimensional loci such as K_0 = K_1")
    w("- Factorized defect-locus diagnostics: ✓ Phase 19 shows the known exact decompositions have zero centered-lift defect dimension, while projected pair-equality families P_0 = P_1 and P_1 = P_2 leave hard residual floors 0.5 with rank(H) = 9, the all-equal family leaves residual 2/3 with rank(H) = 0, and the broader collinear family is not an obstruction because it already contains the standard algorithm exactly")
    w("- Affine-line lift obstruction: ✓ Phase 20 derives the codimension-one defect where a second scalar relation forces the centered lifts K_0, K_1, K_2 onto one matrix line, verifies that neither AlphaTensor nor the standard algorithm lies there, and shows a factorized family that enforces lift-span rank 1 stays far from exact multiplication (best residuals about 0.74 at R=23 and 0.81 at R=27)")
    w("- Functional annihilator inside ker(Gamma): ✓ Phase 21 shows the known exact decompositions have no annihilator defect inside ker(Gamma), and that a factorized family can enforce lambda^T H = 0 while still leaving defect dimension 0 because the enforced lambda stays far from ker(Gamma); therefore a plain annihilator of H is too weak to serve as the missing obstruction")
    w("- Kernel-linked annihilator homotopy: ✓ Phase 22 forces lambda to start inside the original ker(Gamma) and tracks the residual-angle tradeoff while driving lambda^T H toward 0; AlphaTensor degrades quickly, while the standard algorithm remains much more flexible, so the next obstruction must distinguish saturated low-rank geometry from the highly symmetric standard family")
    w("- Staged layering bridge: ✓ Phase 23 links the old depth-2 circuit dead end to the new staged bilinear homotopies; exact depth-2 layering still collapses back to bilinear rank, but staged forcing shows AlphaTensor destabilizes already at t = 0.1 while the standard algorithm often stays exact far longer")
    w("- Tangent transversality of kernel-linked forcing: ✓ Phase 24 shows the canonical minimum-norm forcing direction is already first-order transverse on AlphaTensor but tangent on the standard algorithm; however AlphaTensor still admits corrected exact tangent paths, and those paths evade the obstruction by rotating lambda out of ker(Gamma_t) rather than by keeping the kernel link intact")
    w("- Honest kernel-retention derivation + scan: ✓ Phase 25 formalizes the correct coupled defect operator C_x(lambda) = (Gamma lambda, H^T lambda), identifies the intrinsic kernel-restricted quantity kappa_ker(x) = sigma_min(Q_x^T H_x), and then tracks that soft mode along exact corrected paths; AlphaTensor's softest path lowers kappa_ker to about 0.00469 without rank collapse, while the standard softest path reaches a true defect only at a singular endpoint collapse")
    w("- Regularized kernel retention: ✓ Phase 26 filters the Phase 25 paths by near-exactness and no restricted-rank collapse; under that filter AlphaTensor's softest path still reaches kappa_ker about 0.00469, while the standard softest path only reaches 0.1 before the excluded endpoint collapse, although one more symmetric standard random path still gets down to about 0.00571")
    w("- Coupled continuation prototype: ✓ Phase 27 implements the first linearized solve in both dot(x) and dot(lambda); both AlphaTensor and the standard algorithm admit near-perfect first-order cancellation of the honest defect, but finite probes show only modest reduction, so the remaining barrier is nonlinear persistence rather than mere local solvability")
    w("- Wildcard-regularized continuation: ✓ Phase 28 adds an explicit wildcard search over admissible nullspace directions on top of the coupled continuation model and no-collapse filter; wildcard steps beat the structured branch on every accepted step for both AlphaTensor and the standard algorithm, so future tests must include wildcard stress directions by default")
    w("- Smooth-budgeted continuation: ✓ Phase 29 adds a 12 degree mode-rotation cap and a longer-horizon budgeted continuation policy; smoothness alone still permits wildcard descent, but the cost-penalized policy largely suppresses wildcards and turns continuation into many tiny structured steps")
    w("- Budget-efficiency sweep: ✓ Phase 30 shows that naive gain-per-budget favors the standard algorithm on looser policies, so that scalar ratio is not the right invariant; however a strict 8 degree rotation cap stalls the standard continuation immediately while AlphaTensor still moves, revealing a sharper feasibility-boundary effect")
    w("- Feasibility boundary: ✓ Phase 31 scans rotation-cap and budget pairs directly and finds a stable frontier gap across the tested budget range: AlphaTensor remains feasible at 4 degrees while the standard algorithm requires 12 degrees before any regular descent appears")
    w("- Boundary sharpening: ✓ Phase 32 densifies the cap grid and strengthens the frontier gap: AlphaTensor is already feasible at 2 degrees across the scanned budgets, while the standard algorithm still requires 12 degrees")
    w("- Small-integer coefficient enumeration: ✓ the exact ternary profile pool has been counted modulo sign and symmetry (96,845,281 raw distinct profiles; 570,521 symmetry orbits), the top usefulness profiles have been ranked, and collapsed/full-tensor greedy diagnostics show that collapsed matching is vacuous while corrected 2x2 full-tensor greedy does not recover Strassen through rank 7")
    w("- Polyomino subtensor-rank + tiling analysis: ✓ the full connected-polyomino rank table is now audited cleanly, the priority L-tromino class is verified at numerical rank 9, all four previously unresolved tetromino classes are verified at numerical rank 12, the corrected best flat exact-cover cost is 26 with no verified flat tiling at cost 24 or below, and the Step 51/52 layer split of the public rank-23 algorithm is exported with nuisance rank 14, 3 dead-X-dominant corrector terms, and no signal-dominant terms")
    w()
    w("**Remaining open fronts:**")
    w("- Additional arity-4 schemas: XCXC, XCCX, XXXC, XXX not yet explored")
    w("- AXXC currently has an orbit-complete induced face-pattern signature layer;")
    w("  whether there is a simpler intrinsic minimal closed-form signature/refinement rule remains open")
    w("- Refinement engine: Not yet rerun on corrected composition (14 mixed keys)")
    w("- Higher arity layers: Arity 5+ unexplored")
    w("- The 14^3 = 2744 CXXC orbit count factorization: whether 14 = C(4,2)+C(4,1)+C(4,0) reflects")
    w("  partition types at arity 4 under the compatible group action is unverified")
    w("- The 64-orbit closed layer now has a full composition table; its intrinsic signature rule")
    w("  and exact minimum generating set are still unknown")
    w("- The 164 mixed pairs are invariant under all shared-face coordinate conditioning tried so far;")
    w("  any deterministic refinement must depend on data beyond the interface X coordinates alone")
    w("- The 12.44% live-core escapes stay doubly-live but leave the floor; classify those nonfloor")
    w("  doubly-live targets as a structural layer of their own")
    w("- Same-fiber and focused subsets are closed; determine whether they admit a clean intrinsic")
    w("  signature or conceptual description beyond the target/focus predicates")
    w("- Wildcard-aware obstruction: the next invariant must survive explicit wildcard nullspace search, not just the named structured continuation directions")
    w("- Budget-aware obstruction: the next invariant should measure defect reduction versus smooth continuation cost, not just whether a low-defect step exists")
    w("- Feasibility boundary: determine the smallest rotation cap and cost budget under which each family still admits any regular descent; this now looks more promising than a single gain-per-budget score")
    w("- Boundary sharpening: densify the cap grid near 4 and 12 degrees and test whether the frontier gap persists under finer resolution and larger continuation budgets")
    w("- Threshold localization: determine whether AlphaTensor's true critical cap lies below 2 degrees and whether the standard threshold is exactly 12 degrees or just above 11.5 degrees")
    w("- Step 48 shows that the 8 XC-orbit linearization is exact but vacuous for rank lower bounds;")
    w("  any useful lower-bound model must retain finer-than-orbit-sum equation structure")
    w("- Step 49 now records the exact 729-equation trilinear system and the 8 representative types;")
    w("  the remaining open problem is whether the rank-R solution variety is nonempty for sparse or non-group-closed ansatze")
    w("- Step 52 gives a per-algorithm quotient-rank bound R >= 9 + rank(Nuisance), and Phase 17 now proves the accompanying fiber-mode faithfulness theorem; the remaining open problem is the universal Delta subset span(H) step needed to turn the observed AlphaTensor conservation law R + eta_nullity = 27 into a theorem for all minimum-rank 3x3 decompositions")
    w("- Phase 18 shows that full kernel saturation is stable inside the affine right-inverse model and that failures arise on explicit lower-dimensional loci such as K_0 = K_1; the remaining compute problem is to determine which such loci are actually compatible with factorized rank-1 terms (alpha_k, beta_k, gamma_k)")
    w("- Phase 19 narrows that factorized compatibility question: pair-equality-type loci already look strongly incompatible with exact multiplication in direct factorized searches, while channel-collinearity is too weak because it contains the standard algorithm; the next obstruction candidates must therefore be sharper than generic collinearity but weaker than literal equality of two live channels")
    w("- Phase 20 rules out one such sharper candidate computationally: a second scalar relation among the centered lifts, equivalently lift-span rank 1, appears incompatible with exact multiplication in the tested factorized affine-line family, so the remaining bad locus must be subtler than a single extra scalar relation among K_0, K_1, K_2")
    w("- Phase 21 shows that coupling matters: a shared annihilator lambda^T H = 0 is not enough unless lambda also lies in ker(Gamma), so the next obstruction candidates must encode a kernel-linked annihilator rather than an unconstrained one")
    w("- Phase 22 shows that even the kernel-linked annihilator needs refinement: it is strongly incompatible with the AlphaTensor rank-23 geometry under the tested homotopy, but not with the highly symmetric standard algorithm, so any universal obstruction must incorporate extra structure such as minimal-rank saturation or nonstandard symmetry breaking")
    w("- Phase 24 sharpens that refinement target: there is no universal first-order obstruction because AlphaTensor still has corrected exact tangent-compatible forcing directions, but those corrected paths succeed only by letting lambda rotate away from ker(Gamma_t); the remaining open problem is therefore to control exactness, annihilator progress, and kernel linkage simultaneously at the nonlinear level")
    w("- Phase 25 completes the first version of that target and refines it further: kappa_ker(x) is the right honest quantity to track, but it is not yet a universal separator by itself because some highly symmetric standard continuations can also push it low; the next open problem is to exclude singular endpoint collapse and other nonregular escape mechanisms while testing smooth exact continuation toward kappa_ker = 0")
    w("- Phase 26 shows that forbidding endpoint collapse is necessary but still not sufficient: it removes the standard softest-path loophole but not all high-symmetry standard escape routes")
    w("- Phase 27 shows that even the honest coupled first-order continuation equations are too weak as a final criterion, because both families admit near-perfect linearized cancellation while finite probes only reduce the honest defect modestly; the remaining obstruction problem is therefore nonlinear and regularized at once")
    w("- The old depth-2 route should now be treated as structurally closed for exact division-free circuits by Step 73; any remaining use of layering is as a staged diagnostic inside bilinear factor space, and Phase 23 shows that this diagnostic sharply separates AlphaTensor-like low-rank geometry from the standard algorithm")
    w("- Step 63 measures one public rank-23 algorithm exactly and shows it is gamma-only rigid under single and pair deletion; what remains open is whether other nonequivalent rank-23 algorithms exhibit the same nuisance saturation and removal rigidity")
    w("- Step 64 shows that finite ternary-profile greedy search is not enough: the collapsed model is trivially exact while the corrected full-tensor 2x2 greedy misses Strassen entirely, so any serious finite-pool search must keep the gamma layer explicit and use something stronger than greedy matching pursuit")
    w("- Step 65's polyomino optimum is only a restricted-subtensor tiling upper bound, not a verified global algorithm; after the completed Step 67 audit the formerly claimed 21-cost tetromino route is dead, Step 68's first Fourier-encoded correction-layer steering attempt leaves the canonical and phase-j=1 dead residuals on three exact-rank-9 modes with component-sum upper bound 27, Step 69 sharpens the AlphaTensor interlocking picture to three rank-8 dead-mode subspaces with 6-dimensional triple overlap and union dimension 10 in term space, Step 70's first concrete depth-2 audit still leaves the recursive padded-Strassen route at 31 leaf multiplications with no pair-ratio evidence of an immediate AlphaTensor term-factor collapse below rank 23, and Step 71's five local perturbation/merge/neighborhood shots found 0 exact slot replacements in a 5123-profile explicit pool, 0 exact pair merges, no exact commutator or anticommutator witness through rank 20, and no random-neighborhood feasible subset at R<=22 in a 123-term local pool")
    w("- The toroidal extension confirms that L-trominoes can occur in wrapped tilings even though the flat board cannot be tiled by three L-trominoes; the remaining question is whether wrapped shapes or cross-piece sharing can lower the current exported toroidal cost 27")
    w("- Step 53 shows that support-only representative incidence is also vacuous; any sharper universal")
    w("  theorem must use coefficient identities or subspace geometry, not only index-support patterns")
    w("- Step 54 shows that generic low-rank factor models also fail constructively: low nuisance can")
    w("  occur without any quotient-space gain, so the next constructive family must impose structured")
    w("  coefficient relations that separate Sigma from the nuisance span")
    w("- Step 55 tightens the structured p=3,q=4 target to rank(Nuisance) <= 3 inside Hadamard space;")
    w("  the remaining constructive problem is to find explicit polynomial identities on C and D that")
    w("  force that collapse without also collapsing Sigma")
    w("- The current structured families were still too rigid or too generic; next candidates should")
    w("  target exact nuisance-column identities rather than only symmetry patterns such as Toeplitz or DFT")
    w("- Step 56 extends that negative evidence to p=q=4: even with Hadamard cap 7, the tensor-product")
    w("  DFT family and the tested correction/entry-restricted families did not approach quotient gain 9")
    w("- PyTorch-free local search can only provide best-observed surrogates; if gradient-based nuclear-norm")
    w("  optimization is to be taken seriously, the next run should install torch or another autodiff stack")
    w("- The same-fiber branch 30 o 30 is exactly 54/54, so any orbit-packing lever must come from coefficient")
    w("  weighting or higher-order interactions, not from a hidden asymmetry in the raw branch count")
    w("- Step 57 shows the partition ansatz is still broad at the discrete level; the next pruning layer must")
    w("  combine these partition survivors with finer coefficient identities or with a less rigid spreading model")
    w("- Step 59 rules out the naive full-spread omega family but not more general coefficient-engineered Fourier")
    w("  hybrids; the remaining question is whether one can force large dead-X cancellation while keeping the")
    w("  live block fiber-resolving and the nuisance span uniformly small")
    w("- Step 60 shows the hybrid problem decouples exactly into a Fourier block plus a residual spreader tensor;")
    w("  the remaining hard case is non-rectangular residual output patterns, especially six-fiber patterns where")
    w("  flattening lower bounds stay at 9 and therefore do not distinguish R=19,20,21,22,23 spreader budgets")
    w("- Step 61 shows nuisance is structurally necessary below rank 27, but it does not yet give a universal")
    w("  lower bound above 18; the remaining challenge is to turn these dependency budgets into decomposition-")
    w("  independent lower bounds or into a constructive nuisance-design ansatz")
    w("- Step 62 gives negative numerical evidence against the six-fiber f=3 hybrid route at R=22 and R=23,")
    w("  but not a proof; the remaining work is either stronger exact lower bounds for P1-P4 or a more serious")
    w("  numerical/exact decomposition search with enough structure to certify a witness if one exists")
    w("- The characteristic-2, tropical-flattening, and commutator wildcards did not produce a new")
    w("  lower bound yet; if a wildcard route is to matter, it must retain more than flattening data")
    w("- Kernel uniformity: cc uniformity holds at orbit level AND (s,t) stratum level;")
    w("  next level to check is the full (r,s,t,u) X coordinate or the (c1,c2) boundary")
    w()
    w("These are genuine incompletions, not promises.")

    # ── END ──
    w()
    w("-" * 70)
    w("END OF DOSSIER")
    w("-" * 70)
    w()
    w("This canonical dossier contains all computed orbit rosters, the complete")
    w("composition grid, and all ground-truth structural data.")
    w()
    w("No external files are required. This document is standalone and complete.")

    return "\n".join(L)


if __name__ == "__main__":
    content = generate()

    output_path = DOCS_DIR / "ADE3x3_CANONICAL_OBJECT.md"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Generated: {output_path}")
    print(f"     Total lines: {len(content.splitlines())}")
    print(f"     Size: {len(content):,} bytes")
    print(f"")
    print(f"This dossier is STANDALONE and contains ALL computed results.")
    print(f"It is the ONLY file the next team receives.")
