from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


OUT_DIR = Path(__file__).resolve().parent
ATTACK_ROOT = OUT_DIR.parent
REPO_ROOT = ATTACK_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import write_csv, write_json  # noqa: E402


def make_symbolic_matrix(prefix: str) -> list[list[sp.Symbol]]:
    return [[sp.Symbol(f'{prefix}{row_idx}{col_idx}') for col_idx in range(3)] for row_idx in range(3)]


def formal_p_label(channel: int, row_idx: int, col_idx: int) -> str:
    return f'p{channel}[{row_idx},{col_idx}]'


def formal_delta_label(row_idx: int, sum_left: int, sum_right: int, col_idx: int) -> str:
    return f'delta[{row_idx},{sum_left},{sum_right},{col_idx}]'


def polynomial_rank(identities: list[dict[str, object]]) -> int:
    monomial_labels = sorted({label for identity in identities for label in identity['formal_coeffs']})
    matrix_rows = []
    for identity in identities:
        coeffs = identity['formal_coeffs']
        matrix_rows.append([coeffs.get(label, 0) for label in monomial_labels])
    return int(sp.Matrix(matrix_rows).rank()) if matrix_rows else 0


def main() -> None:
    a = make_symbolic_matrix('a')
    b = make_symbolic_matrix('b')
    variables = [symbol for row in a for symbol in row] + [symbol for row in b for symbol in row]

    p_direct: dict[tuple[int, int, int], sp.Expr] = {}
    for channel in range(3):
        for row_idx in range(3):
            for col_idx in range(3):
                p_direct[(channel, row_idx, col_idx)] = sp.expand(a[row_idx][channel] * b[channel][col_idx])

    sigma: dict[tuple[int, int], sp.Expr] = {}
    eta1: dict[tuple[int, int], sp.Expr] = {}
    eta2: dict[tuple[int, int], sp.Expr] = {}
    for row_idx in range(3):
        for col_idx in range(3):
            p0 = p_direct[(0, row_idx, col_idx)]
            p1 = p_direct[(1, row_idx, col_idx)]
            p2 = p_direct[(2, row_idx, col_idx)]
            sigma[(row_idx, col_idx)] = sp.expand(p0 + p1 + p2)
            eta1[(row_idx, col_idx)] = sp.expand(p0 - p1)
            eta2[(row_idx, col_idx)] = sp.expand(p1 - p2)

    p_recovered = {
        (0, row_idx, col_idx): sp.expand((sigma[(row_idx, col_idx)] + 2 * eta1[(row_idx, col_idx)] + eta2[(row_idx, col_idx)]) / 3)
        for row_idx in range(3)
        for col_idx in range(3)
    }
    p_recovered.update(
        {
            (1, row_idx, col_idx): sp.expand((sigma[(row_idx, col_idx)] - eta1[(row_idx, col_idx)] + eta2[(row_idx, col_idx)]) / 3)
            for row_idx in range(3)
            for col_idx in range(3)
        }
    )
    p_recovered.update(
        {
            (2, row_idx, col_idx): sp.expand((sigma[(row_idx, col_idx)] - eta1[(row_idx, col_idx)] - 2 * eta2[(row_idx, col_idx)]) / 3)
            for row_idx in range(3)
            for col_idx in range(3)
        }
    )

    live_recovery_rows = []
    recovery_verified = True
    for channel in range(3):
        for row_idx in range(3):
            for col_idx in range(3):
                residual = sp.expand(p_recovered[(channel, row_idx, col_idx)] - p_direct[(channel, row_idx, col_idx)])
                verified = residual == 0
                recovery_verified = recovery_verified and verified
                live_recovery_rows.append(
                    {
                        'channel': channel,
                        'coordinate': formal_p_label(channel, row_idx, col_idx),
                        'direct_formula': sp.sstr(p_direct[(channel, row_idx, col_idx)]),
                        'recovered_formula': sp.sstr(p_recovered[(channel, row_idx, col_idx)]),
                        'verified': verified,
                    }
                )

    delta: dict[tuple[int, int, int, int], sp.Expr] = {}
    for row_idx in range(3):
        for sum_left in range(3):
            for sum_right in range(3):
                if sum_left == sum_right:
                    continue
                for col_idx in range(3):
                    delta[(row_idx, sum_left, sum_right, col_idx)] = sp.expand(a[row_idx][sum_left] * b[sum_right][col_idx])

    identities: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []

    for sum_left in range(3):
        for sum_right in range(3):
            if sum_left == sum_right:
                continue
            for row_idx in range(3):
                for row_prime in range(3):
                    if row_idx == row_prime:
                        continue
                    for col_idx in range(3):
                        for col_prime in range(3):
                            expr = sp.expand(
                                delta[(row_idx, sum_left, sum_right, col_idx)] * p_recovered[(sum_left, row_prime, col_prime)]
                                - delta[(row_prime, sum_left, sum_right, col_idx)] * p_recovered[(sum_left, row_idx, col_prime)]
                            )
                            identity = {
                                'family': 'shared_left_factor',
                                's': sum_left,
                                't': sum_right,
                                'r': row_idx,
                                'r_prime': row_prime,
                                'u': col_idx,
                                'u_prime': col_prime,
                                'formal_coeffs': {
                                    f"{formal_delta_label(row_idx, sum_left, sum_right, col_idx)}*{formal_p_label(sum_left, row_prime, col_prime)}": 1,
                                    f"{formal_delta_label(row_prime, sum_left, sum_right, col_idx)}*{formal_p_label(sum_left, row_idx, col_prime)}": -1,
                                },
                            }
                            identities.append(identity)
                            verification_rows.append(
                                {
                                    'family': identity['family'],
                                    's': sum_left,
                                    't': sum_right,
                                    'r': row_idx,
                                    'r_prime': row_prime,
                                    'u': col_idx,
                                    'u_prime': col_prime,
                                    'formal_relation': ' + '.join(
                                        [
                                            f"{coeff:+d}*{label}" if idx else f"{coeff}*{label}"
                                            for idx, (label, coeff) in enumerate(identity['formal_coeffs'].items())
                                        ]
                                    ),
                                    'symbolic_verified': expr == 0,
                                }
                            )

    left_count = len(identities)

    for sum_left in range(3):
        for sum_right in range(3):
            if sum_left == sum_right:
                continue
            for row_idx in range(3):
                for row_prime in range(3):
                    for col_idx in range(3):
                        for col_prime in range(3):
                            if col_idx == col_prime:
                                continue
                            expr = sp.expand(
                                delta[(row_idx, sum_left, sum_right, col_idx)] * p_recovered[(sum_right, row_prime, col_prime)]
                                - delta[(row_idx, sum_left, sum_right, col_prime)] * p_recovered[(sum_right, row_prime, col_idx)]
                            )
                            identity = {
                                'family': 'shared_right_factor',
                                's': sum_left,
                                't': sum_right,
                                'r': row_idx,
                                'r_prime': row_prime,
                                'u': col_idx,
                                'u_prime': col_prime,
                                'formal_coeffs': {
                                    f"{formal_delta_label(row_idx, sum_left, sum_right, col_idx)}*{formal_p_label(sum_right, row_prime, col_prime)}": 1,
                                    f"{formal_delta_label(row_idx, sum_left, sum_right, col_prime)}*{formal_p_label(sum_right, row_prime, col_idx)}": -1,
                                },
                            }
                            identities.append(identity)
                            verification_rows.append(
                                {
                                    'family': identity['family'],
                                    's': sum_left,
                                    't': sum_right,
                                    'r': row_idx,
                                    'r_prime': row_prime,
                                    'u': col_idx,
                                    'u_prime': col_prime,
                                    'formal_relation': ' + '.join(
                                        [
                                            f"{coeff:+d}*{label}" if idx else f"{coeff}*{label}"
                                            for idx, (label, coeff) in enumerate(identity['formal_coeffs'].items())
                                        ]
                                    ),
                                    'symbolic_verified': expr == 0,
                                }
                            )

    right_count = len(identities) - left_count
    left_rank = polynomial_rank(identities[:left_count])
    right_rank = polynomial_rank(identities[left_count:])
    combined_rank = polynomial_rank(identities)

    write_csv(
        OUT_DIR / 'cross_ratio_verification.csv',
        verification_rows,
        ['family', 's', 't', 'r', 'r_prime', 'u', 'u_prime', 'formal_relation', 'symbolic_verified'],
    )

    write_json(
        OUT_DIR / 'single_term_identities.json',
        {
            'live_product_recovery': {
                'verified': recovery_verified,
                'formula_p0': '(sigma + 2*eta1 + eta2) / 3',
                'formula_p1': '(sigma - eta1 + eta2) / 3',
                'formula_p2': '(sigma - eta1 - 2*eta2) / 3',
                'rows': live_recovery_rows,
            },
            'cross_ratio_identities': {
                'left_identity_count': left_count,
                'right_identity_count': right_count,
                'left_independent_count': left_rank,
                'right_independent_count': right_rank,
                'combined_independent_count': combined_rank,
            },
        },
    )

    write_json(
        OUT_DIR / 'gamma_propagation_analysis.json',
        {
            'slice_factorization': {
                'live_slice': 'P_s = a[:,s] * b[s,:]^T',
                'dead_slice': 'Delta_{s,t} = a[:,s] * b[t,:]^T for s != t',
                'shared_left_identity': 'Delta_{s,t}[r,u] * P_s[r_prime,u_prime] = Delta_{s,t}[r_prime,u] * P_s[r,u_prime]',
                'shared_right_identity': 'Delta_{s,t}[r,u] * P_t[r_prime,u_prime] = Delta_{s,t}[r,u_prime] * P_t[r_prime,u]',
            },
            'gamma_delta_slice_formula': {
                'formula': '(Gamma * Delta_{s,t})[c,(r,u)] = sum_k Gamma[c,k] * a_k[r,s] * b_k[t,u]',
                'comparison': [
                    '(Gamma * P_s)[c,(r,u)] = sum_k Gamma[c,k] * a_k[r,s] * b_k[s,u]',
                    '(Gamma * P_t)[c,(r,u)] = sum_k Gamma[c,k] * a_k[r,t] * b_k[t,u]',
                ],
            },
            'verdict': {
                'linear_propagation_from_live_slices': False,
                'reason': 'The dead slice uses cross-channel outer products a[:,s] * b[t,:]^T, so summing Gamma-weighted dead coordinates is not a linear function of the Gamma-weighted live slices P_s and P_t. The single-term algebra provides rational/cross-ratio constraints, not a direct linear implication Gamma*Delta = 0 from Gamma*Eta = 0.',
            },
        },
    )

    print('Phase 17B complete.')
    print(f'  Live-product recovery verified: {recovery_verified}')
    print(f'  Cross-ratio counts: left={left_count}, right={right_count}, combined independent={combined_rank}')


if __name__ == '__main__':
    main()