from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import load_public_terms, write_csv, write_json  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import build_mode_matrices  # noqa: E402


def single_term_ranks(term) -> tuple[int, int]:
    _, eta1, eta2, delta, nuisance = build_mode_matrices([term], 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    return int(h_matrix.rank()), int(nuisance.rank())


def pair_ranks(left_term, right_term) -> tuple[int, int]:
    _, eta1, eta2, delta, nuisance = build_mode_matrices([left_term, right_term], 3)
    h_matrix = sp.Matrix.hstack(eta1, eta2)
    return int(h_matrix.rank()), int(nuisance.rank())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terms, orientation, residual = load_public_terms()
    single_cache = {term.term_id: single_term_ranks(term) for term in terms}
    rows: list[dict] = []
    for left_term in terms:
        left_h_rank, left_n_rank = single_cache[left_term.term_id]
        for right_term in terms:
            if left_term.term_id == right_term.term_id:
                continue
            right_h_rank, right_n_rank = single_cache[right_term.term_id]
            pair_h_rank, pair_n_rank = pair_ranks(left_term, right_term)
            rows.append(
                {
                    'left_term': left_term.term_id,
                    'right_term': right_term.term_id,
                    'left_h_rank': left_h_rank,
                    'right_h_rank': right_h_rank,
                    'pair_h_rank': pair_h_rank,
                    'left_nuisance_rank': left_n_rank,
                    'right_nuisance_rank': right_n_rank,
                    'pair_nuisance_rank': pair_n_rank,
                    'h_compression': left_h_rank + right_h_rank - pair_h_rank,
                    'nuisance_compression': left_n_rank + right_n_rank - pair_n_rank,
                }
            )

    rows.sort(key=lambda row: (-int(row['nuisance_compression']), -int(row['h_compression']), row['left_term'], row['right_term']))
    write_csv(
        OUT_DIR / 'interaction_matrix.csv',
        rows,
        [
            'left_term',
            'right_term',
            'left_h_rank',
            'right_h_rank',
            'pair_h_rank',
            'left_nuisance_rank',
            'right_nuisance_rank',
            'pair_nuisance_rank',
            'h_compression',
            'nuisance_compression',
        ],
    )
    write_json(
        OUT_DIR / 'interaction_summary.json',
        {
            'source': 'AlphaTensor public rank-23 decomposition via Step 63 loader',
            'gamma_orientation': orientation,
            'exact_reconstruction_residual': int(residual),
            'pair_count': len(rows),
            'max_h_compression': max(int(row['h_compression']) for row in rows),
            'max_nuisance_compression': max(int(row['nuisance_compression']) for row in rows),
            'top_rows': rows[:10],
        },
    )

    print('Phase 5c interaction analysis complete')
    print(f'  pair rows = {len(rows)}')


if __name__ == '__main__':
    main()