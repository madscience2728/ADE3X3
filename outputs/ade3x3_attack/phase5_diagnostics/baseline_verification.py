from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
if str(ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTACK_ROOT))

from attack_common import profile_terms, serialize_decomposition, write_json  # noqa: E402
from src.ade3x3.steps.ade3x3_step51_symbolic_fiber_mode_decomposition import standard_algorithm_terms  # noqa: E402
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import Term  # noqa: E402


def convert_standard_terms() -> list[Term]:
    terms: list[Term] = []
    for term_idx, payload in enumerate(standard_algorithm_terms(), start=1):
        terms.append(
            Term(
                term_id=f'std_{term_idx:02d}',
                source_label='standard_27_term',
                alpha=np.array(payload['alpha'], dtype=np.float64),
                beta=np.array(payload['beta'], dtype=np.float64),
                gamma=np.array(payload['gamma'], dtype=np.float64),
            )
        )
    return terms


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terms = convert_standard_terms()
    profile = profile_terms(terms)
    payload = serialize_decomposition(
        decomposition_id='standard_27_term',
        source='baseline_standard_algorithm',
        source_detail='canonical_basis_terms',
        terms=terms,
        max_abs_residual=float(profile['max_abs_residual']),
        loss_value=float(profile['loss_value']),
        metadata=profile,
    )
    (OUT_DIR / 'baseline_standard_algorithm.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_json(OUT_DIR / 'baseline_summary.json', profile)

    print('Phase 5d baseline verification complete')
    print(f"  rank(H) = {profile.get('rank_H_exact', profile['rank_H_numeric'])}")


if __name__ == '__main__':
    main()