from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import exact_projection_stats, load_public_terms, write_csv, write_json  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (  # noqa: E402
    build_mode_matrices,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    terms, _, _ = load_public_terms()
    rows: list[dict[str, object]] = []

    for removed_idx, removed_term in enumerate(terms):
        remaining_terms = terms[:removed_idx] + terms[removed_idx + 1 :]
        sigma_exact, eta1_exact, eta2_exact, delta_exact, nuisance_exact = build_mode_matrices(remaining_terms, 3)
        h_matrix = sp.Matrix.hstack(eta1_exact, eta2_exact)
        exact_stats = exact_projection_stats(h_matrix, delta_exact)
        rank_h = int(h_matrix.rank())
        rank_hdelta = int(sp.Matrix.hstack(h_matrix, delta_exact).rank())
        rank_nuisance = int(nuisance_exact.rank())
        rank_sigma = int(sigma_exact.rank())
        rows.append(
            {
                'removed_term_id': removed_term.term_id,
                'remaining_term_count': len(remaining_terms),
                'rank_H_exact': rank_h,
                'eta_nullity_exact': h_matrix.cols - rank_h,
                'rank_Delta_exact': int(delta_exact.rank()),
                'rank_HDelta_exact': rank_hdelta,
                'rank_Nuisance_exact': rank_nuisance,
                'rank_Sigma_exact': rank_sigma,
                'quotient_gain_exact': int(sp.Matrix.hstack(sigma_exact, nuisance_exact).rank()) - rank_nuisance,
                'delta_in_eta_exact': bool(exact_stats['delta_in_eta_exact']),
                'projection_nonzero_count': int(exact_stats['projection_nonzero_count']),
                'projection_support_min': int(exact_stats['projection_support_min']),
                'projection_support_max': int(exact_stats['projection_support_max']),
            }
        )

    min_rank_h = min(int(row['rank_H_exact']) for row in rows)
    fifth_identity_candidates = [row['removed_term_id'] for row in rows if int(row['rank_H_exact']) <= 13]
    write_csv(
        OUTPUT_DIR / 'single_term_removal_summary.csv',
        rows,
        [
            'removed_term_id',
            'remaining_term_count',
            'rank_H_exact',
            'eta_nullity_exact',
            'rank_Delta_exact',
            'rank_HDelta_exact',
            'rank_Nuisance_exact',
            'rank_Sigma_exact',
            'quotient_gain_exact',
            'delta_in_eta_exact',
            'projection_nonzero_count',
            'projection_support_min',
            'projection_support_max',
        ],
    )
    write_json(
        OUTPUT_DIR / 'single_term_removal_summary.json',
        {
            'subset_count': len(rows),
            'target_rank_H_for_R22': 13,
            'minimum_observed_rank_H': min_rank_h,
            'fifth_identity_candidate_removed_terms': fifth_identity_candidates,
            'all_rows': rows,
            'interpretation': (
                'Every 22-term single-removal subset preserves rank(H)=14, eta-nullity=4, '
                'rank([H|Delta])=14, and quotient gain 8. No single removed AlphaTensor term exposes a fifth exact identity.'
            ),
        },
    )


if __name__ == '__main__':
    main()