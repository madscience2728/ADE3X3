from __future__ import annotations

import random
import sys
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import load_public_terms, write_csv, write_json  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import build_mode_matrices  # noqa: E402


RANDOM_ORDER_COUNT = 20


def build_orderings(term_ids: list[str]) -> list[tuple[str, list[int]]]:
    rng = random.Random(51000)
    orderings: list[tuple[str, list[int]]] = [('original', list(range(len(term_ids))))]
    for order_idx in range(RANDOM_ORDER_COUNT):
        perm = list(range(len(term_ids)))
        rng.shuffle(perm)
        orderings.append((f'random_{order_idx:02d}', perm))
    return orderings


def profile_prefix(terms) -> dict[str, int | bool]:
    sigma, eta1, eta2, delta, nuisance = build_mode_matrices(terms, 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    augmented = sp.Matrix.hstack(sigma, nuisance)
    return {
        'rank_H': int(h_matrix.rank()),
        'rank_Delta': int(delta.rank()),
        'rank_Nuisance': int(nuisance.rank()),
        'rank_Sigma': int(sigma.rank()),
        'rank_Augmented': int(augmented.rank()),
        'quotient_gain': int(augmented.rank()) - int(nuisance.rank()),
        'delta_in_eta': bool(sp.Matrix.hstack(h_matrix, delta).rank() == h_matrix.rank()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terms, orientation, residual = load_public_terms()
    orderings = build_orderings([term.term_id for term in terms])

    trajectory_rows: list[dict] = []
    order_summary_rows: list[dict] = []
    for order_name, indices in orderings:
        ordered_terms = [terms[idx] for idx in indices]
        previous_profile = {
            'rank_H': 0,
            'rank_Delta': 0,
            'rank_Nuisance': 0,
            'quotient_gain': 0,
        }
        first_h14 = None
        first_q9 = None
        redundant_term_count = 0
        for prefix_len in range(1, len(ordered_terms) + 1):
            current_terms = ordered_terms[:prefix_len]
            profile = profile_prefix(current_terms)
            marginal_h = profile['rank_H'] - previous_profile['rank_H']
            marginal_delta = profile['rank_Delta'] - previous_profile['rank_Delta']
            marginal_nuisance = profile['rank_Nuisance'] - previous_profile['rank_Nuisance']
            marginal_q = profile['quotient_gain'] - previous_profile['quotient_gain']
            redundant_term_count += int(marginal_nuisance == 0)
            if first_h14 is None and profile['rank_H'] >= 14:
                first_h14 = prefix_len
            if first_q9 is None and profile['quotient_gain'] >= 9:
                first_q9 = prefix_len
            trajectory_rows.append(
                {
                    'order_id': order_name,
                    'step_k': prefix_len,
                    'term_id_added': ordered_terms[prefix_len - 1].term_id,
                    'rank_H': profile['rank_H'],
                    'rank_Delta': profile['rank_Delta'],
                    'rank_Nuisance': profile['rank_Nuisance'],
                    'rank_Sigma': profile['rank_Sigma'],
                    'rank_Augmented': profile['rank_Augmented'],
                    'quotient_gain': profile['quotient_gain'],
                    'delta_in_eta': profile['delta_in_eta'],
                    'marginal_rank_H': marginal_h,
                    'marginal_rank_Delta': marginal_delta,
                    'marginal_rank_Nuisance': marginal_nuisance,
                    'marginal_quotient_gain': marginal_q,
                }
            )
            previous_profile = profile

        order_summary_rows.append(
            {
                'order_id': order_name,
                'first_k_rank_H_14': first_h14,
                'first_k_quotient_gain_9': first_q9,
                'final_rank_H': previous_profile['rank_H'],
                'final_rank_Delta': previous_profile['rank_Delta'],
                'final_rank_Nuisance': previous_profile['rank_Nuisance'],
                'final_quotient_gain': previous_profile['quotient_gain'],
                'terms_with_zero_additional_nuisance': redundant_term_count,
            }
        )

    write_csv(
        OUT_DIR / 'trajectory.csv',
        trajectory_rows,
        [
            'order_id',
            'step_k',
            'term_id_added',
            'rank_H',
            'rank_Delta',
            'rank_Nuisance',
            'rank_Sigma',
            'rank_Augmented',
            'quotient_gain',
            'delta_in_eta',
            'marginal_rank_H',
            'marginal_rank_Delta',
            'marginal_rank_Nuisance',
            'marginal_quotient_gain',
        ],
    )
    write_csv(
        OUT_DIR / 'trajectory_order_summary.csv',
        order_summary_rows,
        [
            'order_id',
            'first_k_rank_H_14',
            'first_k_quotient_gain_9',
            'final_rank_H',
            'final_rank_Delta',
            'final_rank_Nuisance',
            'final_quotient_gain',
            'terms_with_zero_additional_nuisance',
        ],
    )
    write_json(
        OUT_DIR / 'trajectory_summary.json',
        {
            'source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
            'gamma_orientation': orientation,
            'exact_reconstruction_residual': int(residual),
            'order_count': len(order_summary_rows),
            'original_order': next(row for row in order_summary_rows if row['order_id'] == 'original'),
            'random_orders': [row for row in order_summary_rows if row['order_id'] != 'original'],
        },
    )

    print('Phase 5b trajectory complete')
    print(f'  orderings profiled = {len(order_summary_rows)}')


if __name__ == '__main__':
    main()