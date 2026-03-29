"""
ade3x3_step72_depth2_circuit_dimension_census.py

Step 72: Depth-2 Circuit Dimension Census.

This step answers the dimension-count question for the depth-2 circuit model
described in the Step 72 prompt. It keeps the exact-derived bookkeeping separate
from the measured Jacobian sweep:

1. Exact parameter and raw upper-bound constraint counts for every
   (R1, R2, n_AA, n_BB, n_QQ) with R1 + R2 <= 22.
2. The AA-only case study requested in Task 4.
3. A measured Jacobian-rank sweep on the requested high-priority families:
   - AA/BB-only families for R_total in {20, 21, 22}, R1 >= 9.
   - AA/QQ families for the same totals.

The Jacobian is assembled analytically rather than by finite differences.
This measures the same local constraint-map rank while keeping the sweep fast
and numerically stable enough to run across the whole requested landscape.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


EXPORTS = Path('outputs/exports')

ATOM_COUNT = 9
OUTPUT_COUNT = 9
PAIR_I = np.array([left for left in range(ATOM_COUNT) for right in range(left, ATOM_COUNT)], dtype=np.int64)
PAIR_J = np.array([right for left in range(ATOM_COUNT) for right in range(left, ATOM_COUNT)], dtype=np.int64)
PAIR_DIAG = PAIR_I == PAIR_J
PAIR_OFF = (~PAIR_DIAG).astype(np.float64)
PAIR_COUNT = int(PAIR_I.shape[0])

BILINEAR_SIZE = OUTPUT_COUNT * ATOM_COUNT * ATOM_COUNT
AAB_SIZE = OUTPUT_COUNT * PAIR_COUNT * ATOM_COUNT
ABB_SIZE = OUTPUT_COUNT * ATOM_COUNT * PAIR_COUNT
AABB_SIZE = OUTPUT_COUNT * PAIR_COUNT * PAIR_COUNT

RANK_TOL = 1e-6
SWEEP_TOTALS = (20, 21, 22)
SWEEP_MIN_R1 = 9
BASE_SEED = 720072


@dataclass(frozen=True)
class CircuitConfig:
    r1: int
    r2: int
    n_aa: int
    n_bb: int
    n_qq: int

    @property
    def total(self) -> int:
        return self.r1 + self.r2

    @property
    def tag(self) -> str:
        return f"R{self.total}_r1{self.r1}_r2{self.r2}_aa{self.n_aa}_bb{self.n_bb}_qq{self.n_qq}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Wrote {len(rows)} rows -> {path}', flush=True)


def write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f'  Wrote text -> {path}', flush=True)


def format_config(config: CircuitConfig) -> str:
    return f"R_total={config.total}, R1={config.r1}, R2={config.r2}, AA={config.n_aa}, BB={config.n_bb}, QQ={config.n_qq}"


def parameter_count(config: CircuitConfig) -> int:
    return (
        18 * config.r1
        + config.n_aa * (config.r1 + ATOM_COUNT)
        + config.n_bb * (config.r1 + ATOM_COUNT)
        + config.n_qq * (2 * config.r1)
        + OUTPUT_COUNT * (config.r1 + config.r2)
    )


def parameter_formula_simplified(config: CircuitConfig) -> int:
    return 27 * config.r1 + config.r1 * config.r2 + 18 * config.r2 + config.n_qq * (config.r1 - 9)


def raw_constraint_upper_bound(config: CircuitConfig) -> tuple[int, int, int, int, int]:
    bilinear = BILINEAR_SIZE
    aab = AAB_SIZE if config.n_aa > 0 else 0
    abb = ABB_SIZE if config.n_bb > 0 else 0
    aabb = AABB_SIZE if config.n_qq > 0 else 0
    total = bilinear + aab + abb + aabb
    return total, bilinear, aab, abb, aabb


def matrix_multiplication_target() -> np.ndarray:
    target = np.zeros((OUTPUT_COUNT, ATOM_COUNT, ATOM_COUNT), dtype=np.float64)
    for row_idx in range(3):
        for sum_idx in range(3):
            for col_idx in range(3):
                c_idx = 3 * row_idx + col_idx
                a_idx = 3 * row_idx + sum_idx
                b_idx = 3 * sum_idx + col_idx
                target[c_idx, a_idx, b_idx] = 1.0
    return target


TARGET_BILINEAR = matrix_multiplication_target()


def sym_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        left[:, None, PAIR_I] * right[None, :, PAIR_J]
        + PAIR_OFF[None, None, :] * left[:, None, PAIR_J] * right[None, :, PAIR_I]
    )


def basis_sym(vectors: np.ndarray) -> np.ndarray:
    result = np.zeros((vectors.shape[0], ATOM_COUNT, PAIR_COUNT), dtype=np.float64)
    pair_index = np.arange(PAIR_COUNT, dtype=np.int64)
    result[:, PAIR_I, pair_index] += vectors[:, PAIR_J]
    off_idx = np.flatnonzero(~PAIR_DIAG)
    result[:, PAIR_J[off_idx], off_idx] += vectors[:, PAIR_I[off_idx]]
    return result


def random_params(config: CircuitConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        'alpha': rng.standard_normal((config.r1, ATOM_COUNT)),
        'beta': rng.standard_normal((config.r1, ATOM_COUNT)),
        'gamma1': rng.standard_normal((OUTPUT_COUNT, config.r1)),
        'mix_aa': rng.standard_normal((config.n_aa, config.r1)),
        'aform': rng.standard_normal((config.n_aa, ATOM_COUNT)),
        'gamma_aa': rng.standard_normal((OUTPUT_COUNT, config.n_aa)),
        'mix_bb': rng.standard_normal((config.n_bb, config.r1)),
        'bform': rng.standard_normal((config.n_bb, ATOM_COUNT)),
        'gamma_bb': rng.standard_normal((OUTPUT_COUNT, config.n_bb)),
        'left_qq': rng.standard_normal((config.n_qq, config.r1)),
        'right_qq': rng.standard_normal((config.n_qq, config.r1)),
        'gamma_qq': rng.standard_normal((OUTPUT_COUNT, config.n_qq)),
    }


def build_intermediates(config: CircuitConfig, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    alpha = params['alpha']
    beta = params['beta']

    intermediates: dict[str, np.ndarray] = {
        'alpha_basis': basis_sym(alpha),
        'beta_basis': basis_sym(beta),
    }

    if config.n_aa > 0:
        intermediates['aform_basis'] = basis_sym(params['aform'])
        intermediates['sym_alpha_aform'] = sym_pairs(alpha, params['aform'])
        intermediates['aa_base'] = np.einsum(
            'jk,kjp,kb->jpb',
            params['mix_aa'],
            intermediates['sym_alpha_aform'],
            beta,
            optimize=True,
        )
    else:
        intermediates['aform_basis'] = np.zeros((0, ATOM_COUNT, PAIR_COUNT), dtype=np.float64)
        intermediates['sym_alpha_aform'] = np.zeros((config.r1, 0, PAIR_COUNT), dtype=np.float64)
        intermediates['aa_base'] = np.zeros((0, PAIR_COUNT, ATOM_COUNT), dtype=np.float64)

    if config.n_bb > 0:
        intermediates['bform_basis'] = basis_sym(params['bform'])
        intermediates['sym_beta_bform'] = sym_pairs(beta, params['bform'])
        intermediates['bb_base'] = np.einsum(
            'jk,ka,kjp->jap',
            params['mix_bb'],
            alpha,
            intermediates['sym_beta_bform'],
            optimize=True,
        )
    else:
        intermediates['bform_basis'] = np.zeros((0, ATOM_COUNT, PAIR_COUNT), dtype=np.float64)
        intermediates['sym_beta_bform'] = np.zeros((config.r1, 0, PAIR_COUNT), dtype=np.float64)
        intermediates['bb_base'] = np.zeros((0, ATOM_COUNT, PAIR_COUNT), dtype=np.float64)

    if config.n_qq > 0:
        intermediates['sym_alpha_alpha'] = sym_pairs(alpha, alpha)
        intermediates['sym_beta_beta'] = sym_pairs(beta, beta)
        intermediates['qq_base'] = np.einsum(
            'jk,jl,klp,klq->jpq',
            params['left_qq'],
            params['right_qq'],
            intermediates['sym_alpha_alpha'],
            intermediates['sym_beta_beta'],
            optimize=True,
        )
    else:
        intermediates['sym_alpha_alpha'] = np.zeros((config.r1, config.r1, PAIR_COUNT), dtype=np.float64)
        intermediates['sym_beta_beta'] = np.zeros((config.r1, config.r1, PAIR_COUNT), dtype=np.float64)
        intermediates['qq_base'] = np.zeros((0, PAIR_COUNT, PAIR_COUNT), dtype=np.float64)

    return intermediates


def row_slices(config: CircuitConfig) -> dict[str, slice]:
    offset = 0
    bilinear = slice(offset, offset + BILINEAR_SIZE)
    offset += BILINEAR_SIZE
    aab = slice(offset, offset + (AAB_SIZE if config.n_aa > 0 else 0))
    offset += AAB_SIZE if config.n_aa > 0 else 0
    abb = slice(offset, offset + (ABB_SIZE if config.n_bb > 0 else 0))
    offset += ABB_SIZE if config.n_bb > 0 else 0
    aabb = slice(offset, offset + (AABB_SIZE if config.n_qq > 0 else 0))
    offset += AABB_SIZE if config.n_qq > 0 else 0
    return {
        'bilinear': bilinear,
        'aab': aab,
        'abb': abb,
        'aabb': aabb,
        'total_rows': slice(0, offset),
    }


def constraint_row_count(config: CircuitConfig) -> int:
    _, bilinear, aab, abb, aabb = raw_constraint_upper_bound(config)
    return bilinear + aab + abb + aabb


def build_jacobian(config: CircuitConfig, params: dict[str, np.ndarray], intermediates: dict[str, np.ndarray]) -> np.ndarray:
    alpha = params['alpha']
    beta = params['beta']
    gamma1 = params['gamma1']
    mix_aa = params['mix_aa']
    aform = params['aform']
    gamma_aa = params['gamma_aa']
    mix_bb = params['mix_bb']
    bform = params['bform']
    gamma_bb = params['gamma_bb']
    left_qq = params['left_qq']
    right_qq = params['right_qq']
    gamma_qq = params['gamma_qq']

    alpha_basis = intermediates['alpha_basis']
    beta_basis = intermediates['beta_basis']
    aform_basis = intermediates['aform_basis']
    bform_basis = intermediates['bform_basis']
    sym_alpha_aform = intermediates['sym_alpha_aform']
    sym_beta_bform = intermediates['sym_beta_bform']
    sym_alpha_alpha = intermediates['sym_alpha_alpha']
    sym_beta_beta = intermediates['sym_beta_beta']
    aa_base = intermediates['aa_base']
    bb_base = intermediates['bb_base']
    qq_base = intermediates['qq_base']

    slices = row_slices(config)
    total_rows = slices['total_rows'].stop
    total_params = parameter_count(config)
    jacobian = np.zeros((total_rows, total_params), dtype=np.float64)
    cursor = 0

    for k_idx in range(config.r1):
        for a_idx in range(ATOM_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)

            bilinear = np.zeros((OUTPUT_COUNT, ATOM_COUNT, ATOM_COUNT), dtype=np.float64)
            bilinear[:, a_idx, :] = np.outer(gamma1[:, k_idx], beta[k_idx])
            column[slices['bilinear']] = bilinear.reshape(-1)

            if config.n_aa > 0:
                aab = np.einsum(
                    'cj,j,jp,b->cpb',
                    gamma_aa,
                    mix_aa[:, k_idx],
                    aform_basis[:, a_idx, :],
                    beta[k_idx],
                    optimize=True,
                )
                column[slices['aab']] = aab.reshape(-1)

            if config.n_bb > 0:
                abb = np.zeros((OUTPUT_COUNT, ATOM_COUNT, PAIR_COUNT), dtype=np.float64)
                abb[:, a_idx, :] = np.einsum(
                    'cj,j,jq->cq',
                    gamma_bb,
                    mix_bb[:, k_idx],
                    sym_beta_bform[k_idx],
                    optimize=True,
                )
                column[slices['abb']] = abb.reshape(-1)

            if config.n_qq > 0:
                qq_coeff = (
                    left_qq[:, [k_idx]] * right_qq
                    + right_qq[:, [k_idx]] * left_qq
                )
                qq_core = np.einsum(
                    'jl,lp,lq->jpq',
                    qq_coeff,
                    alpha_basis[:, a_idx, :],
                    sym_beta_beta[k_idx],
                    optimize=True,
                )
                aabb = np.einsum('cj,jpq->cpq', gamma_qq, qq_core, optimize=True)
                column[slices['aabb']] = aabb.reshape(-1)

            jacobian[:, cursor] = column
            cursor += 1

    for k_idx in range(config.r1):
        for b_idx in range(ATOM_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)

            bilinear = np.zeros((OUTPUT_COUNT, ATOM_COUNT, ATOM_COUNT), dtype=np.float64)
            bilinear[:, :, b_idx] = np.outer(gamma1[:, k_idx], alpha[k_idx])
            column[slices['bilinear']] = bilinear.reshape(-1)

            if config.n_aa > 0:
                aab = np.zeros((OUTPUT_COUNT, PAIR_COUNT, ATOM_COUNT), dtype=np.float64)
                aab[:, :, b_idx] = np.einsum(
                    'cj,j,jp->cp',
                    gamma_aa,
                    mix_aa[:, k_idx],
                    sym_alpha_aform[k_idx],
                    optimize=True,
                )
                column[slices['aab']] = aab.reshape(-1)

            if config.n_bb > 0:
                abb = np.einsum(
                    'cj,j,a,jq->caq',
                    gamma_bb,
                    mix_bb[:, k_idx],
                    alpha[k_idx],
                    bform_basis[:, b_idx, :],
                    optimize=True,
                )
                column[slices['abb']] = abb.reshape(-1)

            if config.n_qq > 0:
                qq_coeff = (
                    left_qq[:, [k_idx]] * right_qq
                    + right_qq[:, [k_idx]] * left_qq
                )
                qq_core = np.einsum(
                    'jl,lp,lq->jpq',
                    qq_coeff,
                    sym_alpha_alpha[k_idx],
                    beta_basis[:, b_idx, :],
                    optimize=True,
                )
                aabb = np.einsum('cj,jpq->cpq', gamma_qq, qq_core, optimize=True)
                column[slices['aabb']] = aabb.reshape(-1)

            jacobian[:, cursor] = column
            cursor += 1

    for c_idx in range(OUTPUT_COUNT):
        for k_idx in range(config.r1):
            column = np.zeros(total_rows, dtype=np.float64)
            bilinear = np.zeros((OUTPUT_COUNT, ATOM_COUNT, ATOM_COUNT), dtype=np.float64)
            bilinear[c_idx] = np.outer(alpha[k_idx], beta[k_idx])
            column[slices['bilinear']] = bilinear.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

    for j_idx in range(config.n_aa):
        for k_idx in range(config.r1):
            column = np.zeros(total_rows, dtype=np.float64)
            aab = np.einsum(
                'c,p,b->cpb',
                gamma_aa[:, j_idx],
                sym_alpha_aform[k_idx, j_idx],
                beta[k_idx],
                optimize=True,
            )
            column[slices['aab']] = aab.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for a_idx in range(ATOM_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)
            aab_core = np.einsum(
                'k,kp,kb->pb',
                mix_aa[j_idx],
                alpha_basis[:, a_idx, :],
                beta,
                optimize=True,
            )
            aab = np.einsum('c,pb->cpb', gamma_aa[:, j_idx], aab_core, optimize=True)
            column[slices['aab']] = aab.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for c_idx in range(OUTPUT_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)
            aab = np.zeros((OUTPUT_COUNT, PAIR_COUNT, ATOM_COUNT), dtype=np.float64)
            aab[c_idx] = aa_base[j_idx]
            column[slices['aab']] = aab.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

    for j_idx in range(config.n_bb):
        for k_idx in range(config.r1):
            column = np.zeros(total_rows, dtype=np.float64)
            abb = np.einsum(
                'c,a,q->caq',
                gamma_bb[:, j_idx],
                alpha[k_idx],
                sym_beta_bform[k_idx, j_idx],
                optimize=True,
            )
            column[slices['abb']] = abb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for b_idx in range(ATOM_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)
            abb_core = np.einsum(
                'k,ka,kq->aq',
                mix_bb[j_idx],
                alpha,
                beta_basis[:, b_idx, :],
                optimize=True,
            )
            abb = np.einsum('c,aq->caq', gamma_bb[:, j_idx], abb_core, optimize=True)
            column[slices['abb']] = abb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for c_idx in range(OUTPUT_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)
            abb = np.zeros((OUTPUT_COUNT, ATOM_COUNT, PAIR_COUNT), dtype=np.float64)
            abb[c_idx] = bb_base[j_idx]
            column[slices['abb']] = abb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

    for j_idx in range(config.n_qq):
        for k_idx in range(config.r1):
            column = np.zeros(total_rows, dtype=np.float64)
            qq_core = np.einsum(
                'l,lp,lq->pq',
                right_qq[j_idx],
                sym_alpha_alpha[k_idx],
                sym_beta_beta[k_idx],
                optimize=True,
            )
            aabb = np.einsum('c,pq->cpq', gamma_qq[:, j_idx], qq_core, optimize=True)
            column[slices['aabb']] = aabb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for k_idx in range(config.r1):
            column = np.zeros(total_rows, dtype=np.float64)
            qq_core = np.einsum(
                'l,lp,lq->pq',
                left_qq[j_idx],
                sym_alpha_alpha[k_idx],
                sym_beta_beta[k_idx],
                optimize=True,
            )
            aabb = np.einsum('c,pq->cpq', gamma_qq[:, j_idx], qq_core, optimize=True)
            column[slices['aabb']] = aabb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

        for c_idx in range(OUTPUT_COUNT):
            column = np.zeros(total_rows, dtype=np.float64)
            aabb = np.zeros((OUTPUT_COUNT, PAIR_COUNT, PAIR_COUNT), dtype=np.float64)
            aabb[c_idx] = qq_base[j_idx]
            column[slices['aabb']] = aabb.reshape(-1)
            jacobian[:, cursor] = column
            cursor += 1

    if cursor != total_params:
        raise RuntimeError(f'Jacobian column mismatch for {format_config(config)}: built {cursor}, expected {total_params}')

    return jacobian


def jacobian_rank(jacobian: np.ndarray, tol: float = RANK_TOL) -> int:
    gram = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(gram)
    return int(np.count_nonzero(np.sqrt(np.clip(eigenvalues, 0.0, None)) > tol))


def exact_census_rows() -> list[dict]:
    rows: list[dict] = []
    for total in range(1, 23):
        for r1 in range(1, total + 1):
            r2 = total - r1
            for n_qq in range(r2 + 1):
                remaining = r2 - n_qq
                for n_aa in range(remaining + 1):
                    n_bb = remaining - n_aa
                    config = CircuitConfig(r1=r1, r2=r2, n_aa=n_aa, n_bb=n_bb, n_qq=n_qq)
                    p_direct = parameter_count(config)
                    p_simplified = parameter_formula_simplified(config)
                    c_total, c_bilin, c_aab, c_abb, c_aabb = raw_constraint_upper_bound(config)
                    rows.append(
                        {
                            'r_total': config.total,
                            'r1': config.r1,
                            'r2': config.r2,
                            'n_aa': config.n_aa,
                            'n_bb': config.n_bb,
                            'n_qq': config.n_qq,
                            'parameter_count': p_direct,
                            'parameter_count_simplified': p_simplified,
                            'parameter_formula_match': str(p_direct == p_simplified),
                            'bilinear_constraint_upper_bound': c_bilin,
                            'aab_constraint_upper_bound': c_aab,
                            'abb_constraint_upper_bound': c_abb,
                            'aabb_constraint_upper_bound': c_aabb,
                            'constraint_upper_bound_total': c_total,
                            'raw_upper_deficit': p_direct - c_total,
                            'qq_parameter_bonus': config.n_qq * (config.r1 - 9),
                            'provenance': 'EXACT_DERIVED',
                        }
                    )
    return rows


def aa_only_rows() -> list[dict]:
    rows: list[dict] = []
    for total in range(1, 23):
        for r1 in range(1, total + 1):
            r2 = total - r1
            config = CircuitConfig(r1=r1, r2=r2, n_aa=r2, n_bb=0, n_qq=0)
            parameter_total = parameter_count(config)
            raw_constraint_total = BILINEAR_SIZE + AAB_SIZE if r2 > 0 else BILINEAR_SIZE
            rows.append(
                {
                    'r_total': total,
                    'r1': r1,
                    'r2': r2,
                    'parameter_count': parameter_total,
                    'raw_constraint_upper_bound': raw_constraint_total,
                    'raw_upper_deficit': parameter_total - raw_constraint_total,
                    'bilinear_only_deficit': parameter_total - BILINEAR_SIZE,
                    'provenance': 'EXACT_DERIVED',
                }
            )
    return rows


def measured_configs() -> tuple[list[CircuitConfig], list[CircuitConfig]]:
    aa_bb: list[CircuitConfig] = []
    aa_qq: list[CircuitConfig] = []
    for total in SWEEP_TOTALS:
        for r1 in range(SWEEP_MIN_R1, total):
            r2 = total - r1
            for n_aa in range(r2 + 1):
                aa_bb.append(CircuitConfig(r1=r1, r2=r2, n_aa=n_aa, n_bb=r2 - n_aa, n_qq=0))
            for n_qq in range(1, r2 + 1):
                aa_qq.append(CircuitConfig(r1=r1, r2=r2, n_aa=r2 - n_qq, n_bb=0, n_qq=n_qq))
    return aa_bb, aa_qq


def measured_row(config: CircuitConfig) -> dict:
    seed = (
        BASE_SEED
        + 100000 * config.total
        + 1000 * config.r1
        + 100 * config.n_aa
        + 10 * config.n_bb
        + config.n_qq
    )
    params = random_params(config, seed)
    intermediates = build_intermediates(config, params)
    jacobian = build_jacobian(config, params, intermediates)
    rank = jacobian_rank(jacobian)
    param_total = parameter_count(config)
    raw_total, c_bilin, c_aab, c_abb, c_aabb = raw_constraint_upper_bound(config)
    row_count = constraint_row_count(config)
    local_fiber = param_total - rank

    if config.n_qq == 0:
        family = 'AA_BB_only'
    elif config.n_bb == 0:
        family = 'AA_QQ_only'
    else:
        family = 'mixed'

    return {
        'config_id': config.tag,
        'family': family,
        'r_total': config.total,
        'r1': config.r1,
        'r2': config.r2,
        'n_aa': config.n_aa,
        'n_bb': config.n_bb,
        'n_qq': config.n_qq,
        'parameter_count': param_total,
        'jacobian_row_count': row_count,
        'jacobian_rank': rank,
        'local_fiber_dimension': local_fiber,
        'full_column_rank': str(rank == param_total),
        'bilinear_constraint_upper_bound': c_bilin,
        'aab_constraint_upper_bound': c_aab,
        'abb_constraint_upper_bound': c_abb,
        'aabb_constraint_upper_bound': c_aabb,
        'constraint_upper_bound_total': raw_total,
        'raw_upper_deficit': param_total - raw_total,
        'image_codimension_in_rows': row_count - rank,
        'status': 'positive_local_fiber' if local_fiber > 0 else 'full_column_rank',
        'search_status': 'not_run_dimension_count_only',
        'jacobian_method': 'analytic',
        'rank_tolerance': RANK_TOL,
        'provenance': 'MEASURED_FROM_CODE',
    }


def sweep_rows(configs: list[CircuitConfig]) -> list[dict]:
    if not configs:
        return []
    workers = max(1, min(4, os.cpu_count() or 1))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(measured_row, config): config for config in configs}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row['r_total'], row['r1'], row['n_aa'], row['n_bb'], row['n_qq']))
    return rows


def summary_rows(
    census_rows: list[dict],
    aa_rows: list[dict],
    measured_rows_all: list[dict],
    elapsed_seconds: float,
) -> list[dict]:
    raw_best = max(census_rows, key=lambda row: row['raw_upper_deficit'])
    raw_positive = sum(1 for row in census_rows if row['raw_upper_deficit'] > 0)
    local_fiber_positive = sum(1 for row in measured_rows_all if row['local_fiber_dimension'] > 0)
    full_column_rank = sum(1 for row in measured_rows_all if row['full_column_rank'] == 'True')
    best_measured = max(measured_rows_all, key=lambda row: (row['local_fiber_dimension'], -row['jacobian_rank'])) if measured_rows_all else None
    max_local_fiber = max((row['local_fiber_dimension'] for row in measured_rows_all), default=0)
    aa_bb_count = sum(1 for row in measured_rows_all if row['family'] == 'AA_BB_only')
    aa_qq_count = sum(1 for row in measured_rows_all if row['family'] == 'AA_QQ_only')
    aa_only_best = max(aa_rows, key=lambda row: row['parameter_count'])

    return [
        {
            'summary_name': 'step72_parameter_formula',
            'summary_value': 'P = 27R1 + R1R2 + 18R2 + n_QQ(R1 - 9)',
            'provenance': 'EXACT_DERIVED',
            'note': 'Simplified parameter-count formula after collecting Layer 1, Layer 2, and output parameters.',
        },
        {
            'summary_name': 'step72_raw_constraint_upper_formula',
            'summary_value': 'C_upper = 729 + 3645*1_AA + 3645*1_BB + 18225*1_QQ',
            'provenance': 'EXACT_DERIVED',
            'note': 'Upper-bound count from bilinear, A^2B, AB^2, and A^2B^2 coefficient slots.',
        },
        {
            'summary_name': 'step72_exact_census_config_count',
            'summary_value': str(len(census_rows)),
            'provenance': 'EXACT_DERIVED',
            'note': 'All exact-derived type assignments with R1 + R2 <= 22.',
        },
        {
            'summary_name': 'step72_raw_upper_positive_deficit_count',
            'summary_value': str(raw_positive),
            'provenance': 'EXACT_DERIVED',
            'note': 'Configurations with more raw parameters than the raw upper-bound constraint count.',
        },
        {
            'summary_name': 'step72_best_raw_upper_deficit',
            'summary_value': str(raw_best['raw_upper_deficit']),
            'provenance': 'EXACT_DERIVED',
            'note': f"Largest raw upper-bound deficit attained at {raw_best['config_id'] if 'config_id' in raw_best else format_config(CircuitConfig(raw_best['r1'], raw_best['r2'], raw_best['n_aa'], raw_best['n_bb'], raw_best['n_qq']))}.",
        },
        {
            'summary_name': 'step72_aa_only_max_parameter_count',
            'summary_value': str(aa_only_best['parameter_count']),
            'provenance': 'EXACT_DERIVED',
            'note': 'Maximum AA-only parameter count among R1 + R2 <= 22.',
        },
        {
            'summary_name': 'step72_measured_config_count',
            'summary_value': str(len(measured_rows_all)),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Measured Jacobian-rank sweep rows.',
        },
        {
            'summary_name': 'step72_measured_aa_bb_config_count',
            'summary_value': str(aa_bb_count),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'AA/BB-only measured families.',
        },
        {
            'summary_name': 'step72_measured_aa_qq_config_count',
            'summary_value': str(aa_qq_count),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'AA/QQ-only measured families.',
        },
        {
            'summary_name': 'step72_jacobian_positive_local_fiber_count',
            'summary_value': str(local_fiber_positive),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Measured configurations with parameter_count - rank(J) > 0.',
        },
        {
            'summary_name': 'step72_jacobian_full_column_rank_count',
            'summary_value': str(full_column_rank),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Measured configurations with rank(J) = parameter_count.',
        },
        {
            'summary_name': 'step72_jacobian_max_local_fiber_dimension',
            'summary_value': str(max_local_fiber),
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Largest measured local fiber dimension parameter_count - rank(J).',
        },
        {
            'summary_name': 'step72_jacobian_best_config',
            'summary_value': best_measured['config_id'] if best_measured else 'none',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Configuration attaining the largest measured local fiber dimension.',
        },
        {
            'summary_name': 'step72_jacobian_method',
            'summary_value': 'analytic',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'The Jacobian was assembled analytically instead of by finite differences.',
        },
        {
            'summary_name': 'step72_search_status',
            'summary_value': 'not_run_dimension_count_only',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'The Step 72 question was treated as a dimension census, not a nonlinear solution search.',
        },
        {
            'summary_name': 'step72_major_result_flag',
            'summary_value': 'False',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'No verified R_total < 23 depth-2 algorithm was searched or claimed in this step.',
        },
        {
            'summary_name': 'step72_runtime_seconds',
            'summary_value': f'{elapsed_seconds:.6f}',
            'provenance': 'MEASURED_FROM_CODE',
            'note': 'Wall-clock runtime for Step 72.',
        },
    ]


def markdown_report(
    summary: dict[str, dict],
    census_rows: list[dict],
    aa_rows: list[dict],
    measured_rows: list[dict],
) -> str:
    lines: list[str] = []
    w = lines.append

    best_raw = max(census_rows, key=lambda row: row['raw_upper_deficit'])
    top_measured = sorted(measured_rows, key=lambda row: (-row['local_fiber_dimension'], row['jacobian_rank'], row['r_total'], row['r1']))[:12]
    top_aa_only = sorted(aa_rows, key=lambda row: (row['r_total'], row['r1']))[-12:]

    w('# Step 72: Depth-2 Circuit Dimension Census')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('## Task 1 / 2: Exact Formulas')
    w('')
    w(f"- Parameter formula: {summary['step72_parameter_formula']['summary_value']}")
    w(f"- Raw upper-bound constraint formula: {summary['step72_raw_constraint_upper_formula']['summary_value']}")
    w('- QQ terms add parameters exactly when R1 > 9, break even at R1 = 9, and reduce the raw parameter count relative to AA/BB at R1 < 9.')
    w(f"- Exact census rows written: {summary['step72_exact_census_config_count']['summary_value']}")
    w(f"- Raw upper-bound positive deficits: {summary['step72_raw_upper_positive_deficit_count']['summary_value']}")
    w(f"- Best raw upper-bound deficit: {best_raw['raw_upper_deficit']} at {format_config(CircuitConfig(best_raw['r1'], best_raw['r2'], best_raw['n_aa'], best_raw['n_bb'], best_raw['n_qq']))}")
    w('')
    w('## Task 4: AA-Only Case Study')
    w('')
    w('| R_total | R1 | R2 | parameters | raw D vs 4374/729 rule | D vs 729 only |')
    w('|---------|----|----|------------|------------------------|---------------|')
    for row in top_aa_only:
        w(f"| {row['r_total']} | {row['r1']} | {row['r2']} | {row['parameter_count']} | {row['raw_upper_deficit']} | {row['bilinear_only_deficit']} |")
    w('')
    w('## Task 5: Measured Jacobian Sweep')
    w('')
    w(f"- Jacobian method: {summary['step72_jacobian_method']['summary_value']}")
    w(f"- Measured configurations: {summary['step72_measured_config_count']['summary_value']} total = {summary['step72_measured_aa_bb_config_count']['summary_value']} AA/BB + {summary['step72_measured_aa_qq_config_count']['summary_value']} AA/QQ")
    w(f"- Positive measured local fiber dimension count: {summary['step72_jacobian_positive_local_fiber_count']['summary_value']}")
    w(f"- Full-column-rank count: {summary['step72_jacobian_full_column_rank_count']['summary_value']}")
    w(f"- Maximum measured local fiber dimension: {summary['step72_jacobian_max_local_fiber_dimension']['summary_value']}")
    w('')
    w('| family | R_total | R1 | R2 | AA | BB | QQ | parameters | rows | rank(J) | local fiber | raw upper D | status |')
    w('|--------|---------|----|----|----|----|----|------------|------|---------|-------------|-------------|--------|')
    for row in top_measured:
        w(
            f"| {row['family']} | {row['r_total']} | {row['r1']} | {row['r2']} | {row['n_aa']} | {row['n_bb']} | {row['n_qq']} | "
            f"{row['parameter_count']} | {row['jacobian_row_count']} | {row['jacobian_rank']} | {row['local_fiber_dimension']} | {row['raw_upper_deficit']} | {row['status']} |"
        )
    w('')
    w('## Task 6: Landscape Status')
    w('')
    w(f"- Search status: {summary['step72_search_status']['summary_value']}")
    w('- The deliverable table is dimension-only in this step. The Jacobian landscape is exported, but no nonlinear solve was run here.')
    w('- Any positive local fiber dimension should be interpreted as a parameterization-level signal only. It does not certify an exact solution at R_total < 23.')
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The raw upper-bound count remains brutally negative across the entire exact census: even the best parameter-rich family stays far below the naive coefficient-slot upper bound once the degree-3 and degree-4 cancellation equations are counted at face value. That confirms the prompt\'s warning that raw counts alone are too pessimistic to decide the question.')
    w('The measured Jacobian sweep gives the more relevant local picture. Instead of comparing parameter count to the full coefficient-slot total, Step 72 measures how many constraint directions are actually activated at a generic point for the requested depth-2 families. This is the right local dimension object for the census, but it still answers only a dimension question, not the existence question.')
    w('The other important Step 72 correction is methodological: the prompt proposed finite differences, but the actual sweep uses the exact analytical Jacobian. That improves numerical stability and lets the requested AA/BB and AA/QQ families run as a single measured landscape without changing the rank quantity being measured.')
    return '\n'.join(lines)


def main() -> None:
    start = time.time()
    EXPORTS.mkdir(parents=True, exist_ok=True)

    census = exact_census_rows()
    aa_only = aa_only_rows()
    aa_bb_configs, aa_qq_configs = measured_configs()
    measured = sweep_rows(aa_bb_configs + aa_qq_configs)

    for row in census:
        row['config_id'] = f"R{row['r_total']}_r1{row['r1']}_r2{row['r2']}_aa{row['n_aa']}_bb{row['n_bb']}_qq{row['n_qq']}"

    promising = [row for row in measured if row['local_fiber_dimension'] > 0]
    landscape = []
    for row in measured:
        landscape.append(
            {
                'r_total': row['r_total'],
                'r1': row['r1'],
                'r2': row['r2'],
                'type': f"AA={row['n_aa']},BB={row['n_bb']},QQ={row['n_qq']}",
                'local_fiber_dimension': row['local_fiber_dimension'],
                'best_loss': '',
                'status': row['search_status'],
                'provenance': 'MEASURED_FROM_CODE',
            }
        )

    elapsed = time.time() - start
    summary = summary_rows(census, aa_only, measured, elapsed)
    summary_map = {row['summary_name']: row for row in summary}
    report = markdown_report(summary_map, census, aa_only, measured)

    write_csv(
        EXPORTS / 'step72_exact_census.csv',
        census,
        [
            'config_id',
            'r_total',
            'r1',
            'r2',
            'n_aa',
            'n_bb',
            'n_qq',
            'parameter_count',
            'parameter_count_simplified',
            'parameter_formula_match',
            'bilinear_constraint_upper_bound',
            'aab_constraint_upper_bound',
            'abb_constraint_upper_bound',
            'aabb_constraint_upper_bound',
            'constraint_upper_bound_total',
            'raw_upper_deficit',
            'qq_parameter_bonus',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step72_aa_only_case_study.csv',
        aa_only,
        [
            'r_total',
            'r1',
            'r2',
            'parameter_count',
            'raw_constraint_upper_bound',
            'raw_upper_deficit',
            'bilinear_only_deficit',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step72_measured_jacobian_sweep.csv',
        measured,
        [
            'config_id',
            'family',
            'r_total',
            'r1',
            'r2',
            'n_aa',
            'n_bb',
            'n_qq',
            'parameter_count',
            'jacobian_row_count',
            'jacobian_rank',
            'local_fiber_dimension',
            'full_column_rank',
            'bilinear_constraint_upper_bound',
            'aab_constraint_upper_bound',
            'abb_constraint_upper_bound',
            'aabb_constraint_upper_bound',
            'constraint_upper_bound_total',
            'raw_upper_deficit',
            'image_codimension_in_rows',
            'status',
            'search_status',
            'jacobian_method',
            'rank_tolerance',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step72_promising_regimes.csv',
        promising,
        [
            'config_id',
            'family',
            'r_total',
            'r1',
            'r2',
            'n_aa',
            'n_bb',
            'n_qq',
            'parameter_count',
            'jacobian_row_count',
            'jacobian_rank',
            'local_fiber_dimension',
            'full_column_rank',
            'bilinear_constraint_upper_bound',
            'aab_constraint_upper_bound',
            'abb_constraint_upper_bound',
            'aabb_constraint_upper_bound',
            'constraint_upper_bound_total',
            'raw_upper_deficit',
            'image_codimension_in_rows',
            'status',
            'search_status',
            'jacobian_method',
            'rank_tolerance',
            'provenance',
        ],
    )
    write_csv(
        EXPORTS / 'step72_deficit_landscape.csv',
        landscape,
        ['r_total', 'r1', 'r2', 'type', 'local_fiber_dimension', 'best_loss', 'status', 'provenance'],
    )
    write_csv(
        EXPORTS / 'step72_summary.csv',
        summary,
        ['summary_name', 'summary_value', 'provenance', 'note'],
    )
    write_text(EXPORTS / 'step72_depth2_circuit_dimension_census.md', report)
    print(f'Step 72 complete in {elapsed:.2f}s', flush=True)


if __name__ == '__main__':
    main()