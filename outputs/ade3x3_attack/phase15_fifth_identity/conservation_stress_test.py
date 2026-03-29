from __future__ import annotations

import json
from pathlib import Path

from strassen_verification import build_h_matrix, matrix_rank, standard_2x2_terms, strassen_terms


OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    standard_a, standard_b, _ = standard_2x2_terms()
    standard_h = build_h_matrix(standard_a, standard_b)
    standard_rank = matrix_rank(standard_h)

    strassen_a, strassen_b, _ = strassen_terms()
    strassen_h = build_h_matrix(strassen_a, strassen_b)
    strassen_rank = matrix_rank(strassen_h)

    payload = {
        'standard_2x2': {
            'R': 8,
            'rank_H': standard_rank,
            'k': int(standard_h.shape[1] - standard_rank),
            'conservation_sum': 8 + int(standard_h.shape[1] - standard_rank),
        },
        'strassen_2x2': {
            'R': 7,
            'rank_H': strassen_rank,
            'k': int(strassen_h.shape[1] - strassen_rank),
            'conservation_sum': 7 + int(strassen_h.shape[1] - strassen_rank),
        },
        'notes': [
            'The standard 2x2 algorithm gives rank(H)=4 and k=0, so R+k=8.',
            'The Strassen 2x2 algorithm gives rank(H)=3 and k=1, so R+k=8.',
            'No in-repo 4x4 decomposition table was found, so the N=4 stress test remains unrun.',
        ],
    }
    (OUT_DIR / 'conservation_stress_test_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()