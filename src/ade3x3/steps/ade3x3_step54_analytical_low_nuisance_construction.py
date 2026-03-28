"""
ade3x3_step54_analytical_low_nuisance_construction.py

Step 54: Analytical Low-Nuisance Construction.

This step pursues the constructive direction suggested by Step 52:

1. Analyze the exact dead-free ("pure live") template for 3x3 terms.
2. Measure the nuisance-rank landscape for random low-rank factor families
   alpha in R^{R x 9}, beta in R^{R x 9} with prescribed matrix ranks (p, q).
3. Record the status of the Smirnov 23-term profile lookup.
4. Record a simple 2x2-corner Strassen lift baseline.

The main empirical target is R=22, where Step 52 requires nuisance rank <= 13.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

EXPORTS = Path('outputs/exports')
R_VALUES = [18, 19, 20, 21, 22, 23, 27]
P_VALUES = [3, 4, 5, 6]
Q_VALUES = [3, 4, 5, 6]
DEFAULT_TRIALS = 8
FOCUS_R = 22
FOCUS_TRIALS = 24
BASE_SEED = 540540


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def numeric_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix))


def sample_factor_matrix(rng: np.random.Generator, rows: int, rank: int, cols: int) -> np.ndarray:
    while True:
        left = rng.standard_normal((rows, rank))
        right = rng.standard_normal((rank, cols))
        matrix = left @ right
        if numeric_rank(matrix) == rank:
            return matrix


def build_mode_matrices(alpha: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    rank_count = alpha.shape[0]
    a = alpha.reshape(rank_count, 3, 3)
    b = beta.reshape(rank_count, 3, 3)
    full = a[:, :, :, None, None] * b[:, None, None, :, :]

    sigma = np.stack(
        [full[:, row_idx, 0, 0, col_idx] + full[:, row_idx, 1, 1, col_idx] + full[:, row_idx, 2, 2, col_idx] for row_idx in range(3) for col_idx in range(3)],
        axis=1,
    )
    eta1 = np.stack(
        [full[:, row_idx, 0, 0, col_idx] - full[:, row_idx, 1, 1, col_idx] for row_idx in range(3) for col_idx in range(3)],
        axis=1,
    )
    eta2 = np.stack(
        [full[:, row_idx, 1, 1, col_idx] - full[:, row_idx, 2, 2, col_idx] for row_idx in range(3) for col_idx in range(3)],
        axis=1,
    )
    dead = np.stack(
        [full[:, row_idx, sum_left, sum_right, col_idx] for row_idx in range(3) for sum_left in range(3) for sum_right in range(3) if sum_left != sum_right for col_idx in range(3)],
        axis=1,
    )
    nuisance = np.concatenate([eta1, eta2, dead], axis=1)
    augmented = np.concatenate([sigma, nuisance], axis=1)
    return {
        'sigma': sigma,
        'eta1': eta1,
        'eta2': eta2,
        'dead': dead,
        'nuisance': nuisance,
        'augmented': augmented,
        'full': full.reshape(rank_count, 81),
    }


def profile_from_alpha_beta(alpha: np.ndarray, beta: np.ndarray) -> dict[str, int | bool]:
    matrices = build_mode_matrices(alpha, beta)
    sigma_rank = numeric_rank(matrices['sigma'])
    eta_rank = numeric_rank(np.concatenate([matrices['eta1'], matrices['eta2']], axis=1))
    dead_rank = numeric_rank(matrices['dead'])
    nuisance_rank = numeric_rank(matrices['nuisance'])
    augmented_rank = numeric_rank(matrices['augmented'])
    full_product_rank = numeric_rank(matrices['full'])
    quotient_gain = augmented_rank - nuisance_rank
    return {
        'sigma_rank': sigma_rank,
        'eta_rank': eta_rank,
        'dead_rank': dead_rank,
        'nuisance_rank': nuisance_rank,
        'augmented_rank': augmented_rank,
        'full_product_rank': full_product_rank,
        'quotient_gain': quotient_gain,
        'criterion_holds': quotient_gain == 9,
        'nuisance_equals_full_product_rank': nuisance_rank == full_product_rank,
        'augmented_equals_nuisance_rank': augmented_rank == nuisance_rank,
    }


def dead_free_theorem_rows() -> list[dict]:
    return [
        {
            'statement_id': 'D1',
            'statement': 'A nonzero rank-1 term is dead-free (all delta coordinates vanish) iff there exists a unique summation index s* such that alpha[:,t]=0 for all t!=s* and beta[t,:]=0 for all t!=s*. Equivalently, the alpha column support and beta row support are contained in the same singleton {s*}.',
            'status': 'exact_characterization',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'D2',
            'statement': 'For a nonzero dead-free term with active index s*, define v = a[:,s*] tensor b[s*,:] in the 9-dimensional fiber space. Then Sigma contributes v, while Eta1 and Eta2 are fixed signed embeddings of the same v; only Delta vanishes.',
            'status': 'mode_profile',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'D3',
            'statement': 'Therefore dead-free does not imply nuisance-free in the Step 51 basis. A generic nonzero dead-free term has dead rank 0 but nuisance rank 1, because its anisotropy coordinates are nonzero unless v=0.',
            'status': 'critical_correction',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'D4',
            'statement': 'The Strassen-style pure-live versus cancellation split does not transfer verbatim to Step 51. What survives is only the weaker statement that dead-free terms contribute no dead-X nuisance; they still contribute anisotropy nuisance.',
            'status': 'template_scope',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'statement_id': 'D5',
            'statement': 'A family of dead-free terms has nuisance rows inside the union of three fixed 9-dimensional embeddings E0(v)=(v,0), E1(v)=(-v,v), E2(v)=(0,-v) of the fiber space into the 18-dimensional anisotropy space.',
            'status': 'family_embedding',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def dead_free_index_profile_rows() -> list[dict]:
    return [
        {
            'active_sum_index': 0,
            'sigma_profile': 'Sigma = v',
            'eta1_profile': 'Eta1 = v',
            'eta2_profile': 'Eta2 = 0',
            'dead_profile': 'Delta = 0',
            'nuisance_zero': False,
            'generic_nonzero_term_nuisance_rank': 1,
            'provenance': 'EXACT_DERIVED',
        },
        {
            'active_sum_index': 1,
            'sigma_profile': 'Sigma = v',
            'eta1_profile': 'Eta1 = -v',
            'eta2_profile': 'Eta2 = v',
            'dead_profile': 'Delta = 0',
            'nuisance_zero': False,
            'generic_nonzero_term_nuisance_rank': 1,
            'provenance': 'EXACT_DERIVED',
        },
        {
            'active_sum_index': 2,
            'sigma_profile': 'Sigma = v',
            'eta1_profile': 'Eta1 = 0',
            'eta2_profile': 'Eta2 = -v',
            'dead_profile': 'Delta = 0',
            'nuisance_zero': False,
            'generic_nonzero_term_nuisance_rank': 1,
            'provenance': 'EXACT_DERIVED',
        },
    ]


def trial_count_for_R(rank_count: int) -> int:
    return FOCUS_TRIALS if rank_count == FOCUS_R else DEFAULT_TRIALS


def run_random_factor_experiment() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    trial_rows: list[dict] = []
    grouped: dict[tuple[int, int, int], list[dict]] = {}

    sample_index = 0
    for rank_count in R_VALUES:
        nuisance_target = rank_count - 9
        for p_rank in P_VALUES:
            for q_rank in Q_VALUES:
                rows: list[dict] = []
                for trial_idx in range(trial_count_for_R(rank_count)):
                    seed = BASE_SEED + sample_index
                    sample_index += 1
                    rng = np.random.default_rng(seed)
                    alpha = sample_factor_matrix(rng, rank_count, p_rank, 9)
                    beta = sample_factor_matrix(rng, rank_count, q_rank, 9)
                    profile = profile_from_alpha_beta(alpha, beta)
                    row = {
                        'sample_id': sample_index,
                        'seed': seed,
                        'R': rank_count,
                        'p_rank': p_rank,
                        'q_rank': q_rank,
                        'expected_product_rank_cap': min(rank_count, p_rank * q_rank),
                        'nuisance_target': nuisance_target,
                        'sigma_rank': profile['sigma_rank'],
                        'eta_rank': profile['eta_rank'],
                        'dead_rank': profile['dead_rank'],
                        'nuisance_rank': profile['nuisance_rank'],
                        'augmented_rank': profile['augmented_rank'],
                        'full_product_rank': profile['full_product_rank'],
                        'quotient_gain': profile['quotient_gain'],
                        'criterion_holds': profile['criterion_holds'],
                        'meets_nuisance_target': profile['nuisance_rank'] <= nuisance_target,
                        'meets_full_target': profile['criterion_holds'] and profile['nuisance_rank'] <= nuisance_target,
                        'nuisance_equals_full_product_rank': profile['nuisance_equals_full_product_rank'],
                        'augmented_equals_nuisance_rank': profile['augmented_equals_nuisance_rank'],
                        'provenance': 'EXACT_DERIVED',
                    }
                    rows.append(row)
                    trial_rows.append(row)
                grouped[(rank_count, p_rank, q_rank)] = rows

    summary_rows: list[dict] = []
    focus_rows: list[dict] = []
    best_rows: list[dict] = []
    for (rank_count, p_rank, q_rank), rows in sorted(grouped.items()):
        nuisance_values = sorted(int(row['nuisance_rank']) for row in rows)
        quotient_values = sorted(int(row['quotient_gain']) for row in rows)
        full_values = sorted(int(row['full_product_rank']) for row in rows)
        augmented_values = sorted(int(row['augmented_rank']) for row in rows)
        mid_idx = len(rows) // 2
        summary_row = {
            'R': rank_count,
            'p_rank': p_rank,
            'q_rank': q_rank,
            'trials': len(rows),
            'expected_product_rank_cap': min(rank_count, p_rank * q_rank),
            'nuisance_target': rank_count - 9,
            'nuisance_rank_min': nuisance_values[0],
            'nuisance_rank_median': nuisance_values[mid_idx],
            'nuisance_rank_max': nuisance_values[-1],
            'full_product_rank_min': full_values[0],
            'full_product_rank_median': full_values[mid_idx],
            'full_product_rank_max': full_values[-1],
            'augmented_rank_min': augmented_values[0],
            'augmented_rank_median': augmented_values[mid_idx],
            'augmented_rank_max': augmented_values[-1],
            'quotient_gain_min': quotient_values[0],
            'quotient_gain_median': quotient_values[mid_idx],
            'quotient_gain_max': quotient_values[-1],
            'criterion_holds_count': sum(int(bool(row['criterion_holds'])) for row in rows),
            'meets_nuisance_target_count': sum(int(bool(row['meets_nuisance_target'])) for row in rows),
            'meets_full_target_count': sum(int(bool(row['meets_full_target'])) for row in rows),
            'nuisance_equals_expected_count': sum(int(int(row['nuisance_rank']) == min(rank_count, p_rank * q_rank)) for row in rows),
            'nuisance_equals_full_product_count': sum(int(bool(row['nuisance_equals_full_product_rank'])) for row in rows),
            'augmented_equals_nuisance_count': sum(int(bool(row['augmented_equals_nuisance_rank'])) for row in rows),
            'provenance': 'EXACT_DERIVED',
        }
        summary_rows.append(summary_row)
        if rank_count == FOCUS_R:
            focus_rows.append(summary_row)
            best_row = sorted(rows, key=lambda row: (int(row['nuisance_rank']), -int(row['quotient_gain']), int(row['full_product_rank']), int(row['sample_id'])))[0]
            best_rows.append(best_row)

    return trial_rows, summary_rows, focus_rows, best_rows


def strassen_lift_rows() -> list[dict]:
    return [
        {
            'component': 'top_left_2x2_block',
            'multiplication_count': 7,
            'method': 'Strassen_on_A11_B11',
            'note': 'Apply 2x2 Strassen to the top-left 2x2 corner block.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'component': 'top_left_outer_fixup',
            'multiplication_count': 4,
            'method': 'a12_times_b21_outer_product',
            'note': 'The a12*b21^T correction contributes 4 scalar products to C11.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'component': 'top_right_block',
            'multiplication_count': 6,
            'method': 'A11_b12_plus_a12_b22',
            'note': 'A11*b12 costs 4 and a12*b22 costs 2.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'component': 'bottom_left_block',
            'multiplication_count': 6,
            'method': 'a21T_B11_plus_a22_b21T',
            'note': 'a21^T*B11 costs 4 and a22*b21^T costs 2.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'component': 'bottom_right_scalar',
            'multiplication_count': 3,
            'method': 'a21T_b12_plus_a22_b22',
            'note': 'The scalar corner uses 2 + 1 standard products.',
            'provenance': 'EXACT_DERIVED',
        },
        {
            'component': 'naive_corner_strassen_total',
            'multiplication_count': 26,
            'method': 'sum_of_above',
            'note': 'A direct 2x2-corner Strassen lift already exceeds 23 before any cross-block optimization.',
            'provenance': 'EXACT_DERIVED',
        },
    ]


def smirnov_status_rows() -> list[dict]:
    return [
        {
            'algorithm_name': 'Smirnov_23x3x3_2013',
            'status': 'not_available_in_repo_or_current_tool_set',
            'note': 'The repository contains only a not_mapped status note from Step 47 and no authoritative alpha,beta,gamma term list to profile directly.',
            'provenance': 'EXACT_DERIVED',
        }
    ]


def summary_rows(focus_rows: list[dict], best_rows: list[dict], lift_rows: list[dict], smirnov_rows: list[dict]) -> list[dict]:
    best_focus = min(best_rows, key=lambda row: (int(row['nuisance_rank']), -int(row['quotient_gain']), int(row['sample_id'])))
    best_full_hits = sum(int(row['meets_full_target_count']) for row in focus_rows)
    return [
        {
            'summary_name': 'dead_free_term_nuisance_zero',
            'summary_value': 'False',
            'provenance': 'EXACT_DERIVED',
            'note': 'Dead-free terms still contribute anisotropy nuisance in the Step 51 basis.',
        },
        {
            'summary_name': 'dead_free_generic_nonzero_term_nuisance_rank',
            'summary_value': '1',
            'provenance': 'EXACT_DERIVED',
            'note': 'A nonzero dead-free term has generic nuisance rank 1 and dead rank 0.',
        },
        {
            'summary_name': 'r22_best_observed_nuisance_rank',
            'summary_value': str(best_focus['nuisance_rank']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Minimum nuisance rank observed across all random R=22 low-rank factor samples.',
        },
        {
            'summary_name': 'r22_best_observed_family',
            'summary_value': f"p={best_focus['p_rank']}, q={best_focus['q_rank']}",
            'provenance': 'EXACT_DERIVED',
            'note': 'Family attaining the minimum observed nuisance rank at R=22.',
        },
        {
            'summary_name': 'r22_any_meets_nuisance_target',
            'summary_value': str(any(int(row['meets_nuisance_target_count']) > 0 for row in focus_rows)),
            'provenance': 'EXACT_DERIVED',
            'note': 'Whether any sampled R=22 family achieved nuisance rank <= 13.',
        },
        {
            'summary_name': 'r22_any_meets_full_target',
            'summary_value': str(best_full_hits > 0),
            'provenance': 'EXACT_DERIVED',
            'note': 'Whether any sampled R=22 family achieved nuisance rank <= 13 and quotient-space independence.',
        },
        {
            'summary_name': 'naive_corner_strassen_total',
            'summary_value': str(next(row['multiplication_count'] for row in lift_rows if row['component'] == 'naive_corner_strassen_total')),
            'provenance': 'EXACT_DERIVED',
            'note': 'Direct 2x2-corner Strassen lift baseline for 3x3.',
        },
        {
            'summary_name': 'smirnov_profile_status',
            'summary_value': smirnov_rows[0]['status'],
            'provenance': 'EXACT_DERIVED',
            'note': 'Current status of the Smirnov 23-term nuisance-profile lookup.',
        },
    ]


def write_markdown_summary(
    path: Path,
    dead_rows: list[dict],
    dead_profile_rows: list[dict],
    focus_rows: list[dict],
    best_rows: list[dict],
    lift_rows: list[dict],
    smirnov_rows: list[dict],
    summary: list[dict],
) -> None:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary}
    lines: list[str] = []
    w = lines.append

    w('# Step 54: Analytical Low-Nuisance Construction')
    w(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    w('')
    w('[EXACT_DERIVED]')
    w('')
    w('Step 54 turns the constructive question into two parts: an exact correction to the dead-free term template, and a reproducible nuisance-rank experiment on low-rank factor families.')
    w('')
    w('## Dead-Free Term Analysis')
    w('')
    for row in dead_rows:
        w(f"- {row['statement_id']}: {row['statement']}")
    w('')
    w('| active_sum_index | sigma_profile | eta1_profile | eta2_profile | dead_profile | nuisance_zero | generic_nuisance_rank |')
    w('|------------------|---------------|--------------|--------------|--------------|---------------|-----------------------|')
    for row in dead_profile_rows:
        w(f"| {row['active_sum_index']} | {row['sigma_profile']} | {row['eta1_profile']} | {row['eta2_profile']} | {row['dead_profile']} | {row['nuisance_zero']} | {row['generic_nonzero_term_nuisance_rank']} |")
    w('')
    w('## Random Low-Rank Factor Sweep')
    w('')
    w('The experiment samples alpha=A*C and beta=B*D with prescribed matrix ranks p and q, then computes Sigma, Eta1, Eta2, Delta, nuisance rank, and the quotient-space gain rank([Sigma Nuisance]) - rank(Nuisance).')
    w('')
    w('| R | p | q | trials | target nuisance <= R-9 | nuisance min | nuisance median | nuisance max | quotient gain max | criterion holds count | nuisance target hits | full target hits |')
    w('|---|---|---|--------|------------------------|--------------|----------------|--------------|-------------------|-----------------------|---------------------|------------------|')
    for row in focus_rows:
        w(f"| {row['R']} | {row['p_rank']} | {row['q_rank']} | {row['trials']} | {row['nuisance_target']} | {row['nuisance_rank_min']} | {row['nuisance_rank_median']} | {row['nuisance_rank_max']} | {row['quotient_gain_max']} | {row['criterion_holds_count']} | {row['meets_nuisance_target_count']} | {row['meets_full_target_count']} |")
    w('')
    w('### Best R=22 Samples By Family')
    w('')
    w('| p | q | nuisance_rank | quotient_gain | full_product_rank | criterion_holds | meets_nuisance_target | meets_full_target |')
    w('|---|---|---------------|---------------|-------------------|-----------------|-----------------------|------------------|')
    for row in sorted(best_rows, key=lambda entry: (int(entry['p_rank']), int(entry['q_rank']))):
        w(f"| {row['p_rank']} | {row['q_rank']} | {row['nuisance_rank']} | {row['quotient_gain']} | {row['full_product_rank']} | {row['criterion_holds']} | {row['meets_nuisance_target']} | {row['meets_full_target']} |")
    w('')
    w('## Strassen Lift Baseline')
    w('')
    w('| component | multiplication_count | method | note |')
    w('|-----------|----------------------|--------|------|')
    for row in lift_rows:
        w(f"| {row['component']} | {row['multiplication_count']} | {row['method']} | {row['note']} |")
    w('')
    w('## Smirnov Status')
    w('')
    for row in smirnov_rows:
        w(f"- {row['algorithm_name']}: {row['status']} ({row['note']})")
    w('')
    w('[INTERPRETATION]')
    w('')
    w(f"The exact correction to the Strassen template is that dead-free terms are not nuisance-free in the Step 51 basis: {summary_map['dead_free_term_nuisance_zero']}. In the random-factor experiment, low nuisance at R=22 is easy to obtain numerically in some low-product-dimension families, but quotient-space independence never appeared in the sampled families: r22_any_meets_full_target = {summary_map['r22_any_meets_full_target']}. The recurring empirical pattern is that the nuisance rank tracks the full Hadamard product-space rank, while Sigma adds no extra quotient gain. That makes generic low-rank factor models poor constructive candidates even when their nuisance rank is below the Step 52 threshold. The direct 2x2-corner Strassen lift baseline is 26 scalar multiplications, so it is not competitive without additional cross-block structure.")

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Wrote markdown -> {path}")


def main() -> None:
    print('=== Step 54: Analytical Low-Nuisance Construction ===')
    print()
    print('Building exact dead-free term analysis...')
    dead_rows = dead_free_theorem_rows()
    dead_profile_rows = dead_free_index_profile_rows()
    print('  dead-free template corrected: dead rank can vanish without nuisance vanishing')
    print()
    print('Running random low-rank factor sweep...')
    trial_rows, random_summary_rows, focus_rows, best_rows = run_random_factor_experiment()
    best_focus = min(best_rows, key=lambda row: (int(row['nuisance_rank']), -int(row['quotient_gain']), int(row['sample_id'])))
    print(f"  R=22 best observed nuisance rank={best_focus['nuisance_rank']} at (p,q)=({best_focus['p_rank']},{best_focus['q_rank']})")
    print(f"  R=22 any full hits={(any(int(row['meets_full_target_count']) > 0 for row in focus_rows))}")
    print()
    print('Recording Strassen-lift baseline and Smirnov status...')
    lift_rows = strassen_lift_rows()
    smirnov_rows = smirnov_status_rows()
    summary = summary_rows(focus_rows, best_rows, lift_rows, smirnov_rows)
    print()
    print('Writing outputs...')
    write_csv(EXPORTS / 'step54_dead_free_theorem.csv', dead_rows, ['statement_id', 'statement', 'status', 'provenance'])
    write_csv(EXPORTS / 'step54_dead_free_index_profiles.csv', dead_profile_rows, ['active_sum_index', 'sigma_profile', 'eta1_profile', 'eta2_profile', 'dead_profile', 'nuisance_zero', 'generic_nonzero_term_nuisance_rank', 'provenance'])
    write_csv(EXPORTS / 'step54_random_factor_trials.csv', trial_rows, ['sample_id', 'seed', 'R', 'p_rank', 'q_rank', 'expected_product_rank_cap', 'nuisance_target', 'sigma_rank', 'eta_rank', 'dead_rank', 'nuisance_rank', 'augmented_rank', 'full_product_rank', 'quotient_gain', 'criterion_holds', 'meets_nuisance_target', 'meets_full_target', 'nuisance_equals_full_product_rank', 'augmented_equals_nuisance_rank', 'provenance'])
    write_csv(EXPORTS / 'step54_random_factor_summary.csv', random_summary_rows, ['R', 'p_rank', 'q_rank', 'trials', 'expected_product_rank_cap', 'nuisance_target', 'nuisance_rank_min', 'nuisance_rank_median', 'nuisance_rank_max', 'full_product_rank_min', 'full_product_rank_median', 'full_product_rank_max', 'augmented_rank_min', 'augmented_rank_median', 'augmented_rank_max', 'quotient_gain_min', 'quotient_gain_median', 'quotient_gain_max', 'criterion_holds_count', 'meets_nuisance_target_count', 'meets_full_target_count', 'nuisance_equals_expected_count', 'nuisance_equals_full_product_count', 'augmented_equals_nuisance_count', 'provenance'])
    write_csv(EXPORTS / 'step54_r22_focus.csv', focus_rows, ['R', 'p_rank', 'q_rank', 'trials', 'expected_product_rank_cap', 'nuisance_target', 'nuisance_rank_min', 'nuisance_rank_median', 'nuisance_rank_max', 'full_product_rank_min', 'full_product_rank_median', 'full_product_rank_max', 'augmented_rank_min', 'augmented_rank_median', 'augmented_rank_max', 'quotient_gain_min', 'quotient_gain_median', 'quotient_gain_max', 'criterion_holds_count', 'meets_nuisance_target_count', 'meets_full_target_count', 'nuisance_equals_expected_count', 'nuisance_equals_full_product_count', 'augmented_equals_nuisance_count', 'provenance'])
    write_csv(EXPORTS / 'step54_r22_best_samples.csv', best_rows, ['sample_id', 'seed', 'R', 'p_rank', 'q_rank', 'expected_product_rank_cap', 'nuisance_target', 'sigma_rank', 'eta_rank', 'dead_rank', 'nuisance_rank', 'augmented_rank', 'full_product_rank', 'quotient_gain', 'criterion_holds', 'meets_nuisance_target', 'meets_full_target', 'nuisance_equals_full_product_rank', 'augmented_equals_nuisance_rank', 'provenance'])
    write_csv(EXPORTS / 'step54_strassen_corner_lift.csv', lift_rows, ['component', 'multiplication_count', 'method', 'note', 'provenance'])
    write_csv(EXPORTS / 'step54_smirnov_status.csv', smirnov_rows, ['algorithm_name', 'status', 'note', 'provenance'])
    write_csv(EXPORTS / 'step54_summary.csv', summary, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_markdown_summary(EXPORTS / 'step54_analytical_low_nuisance_construction.md', dead_rows, dead_profile_rows, focus_rows, best_rows, lift_rows, smirnov_rows, summary)
    print()
    print('=== SUMMARY ===')
    print(f"dead_free_term_nuisance_zero={summary[0]['summary_value']}")
    print(f"r22_best_observed_nuisance_rank={summary[2]['summary_value']}")
    print(f"r22_any_meets_full_target={summary[5]['summary_value']}")


if __name__ == '__main__':
    main()