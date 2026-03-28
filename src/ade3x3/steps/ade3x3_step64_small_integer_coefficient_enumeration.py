"""
ade3x3_step64_small_integer_coefficient_enumeration.py

Step 64: Small Integer Coefficient Enumeration.

This step studies the finite coefficient pool alpha,beta in {-1,0,1}^{n x n}.
The core exact observation is that distinct bilinear profiles are projective
outer products of nonzero ternary vectors, so the duplicate/negation quotient
can be computed exactly without materializing all raw pairs.
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations, product
from math import comb, sqrt
from pathlib import Path

import numpy as np

EXPORTS = Path('outputs/exports')
TOP_K = 100
BLOCK_SIZE = 768


@dataclass(frozen=True)
class Profile2x2:
    alpha: np.ndarray
    beta: np.ndarray
    profile: np.ndarray


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def write_text(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f"  Wrote text -> {path}")


def matrix_to_str(matrix: np.ndarray) -> str:
    return str(matrix.astype(int).tolist())


def float_matrix_to_str(matrix: np.ndarray) -> str:
    return '[' + ', '.join('[' + ', '.join(f'{float(value):.6g}' for value in row) + ']' for row in matrix) + ']'


def ternary_projective_vectors(length: int) -> np.ndarray:
    seen: set[bytes] = set()
    vectors: list[np.ndarray] = []
    for entries in product((-1, 0, 1), repeat=length):
        vector = np.array(entries, dtype=np.int8)
        nz = np.flatnonzero(vector)
        if nz.size == 0:
            continue
        if vector[nz[0]] < 0:
            vector = -vector
        key = vector.tobytes()
        if key not in seen:
            seen.add(key)
            vectors.append(vector)
    return np.stack(vectors, axis=0)


def build_actions(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    s = list(permutations(range(n)))
    return [(pi_r, pi_s, pi_u) for pi_r in s for pi_s in s for pi_u in s]


def induced_alpha_perm(n: int, action: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> tuple[int, ...]:
    pi_r, pi_s, _ = action
    perm = [0] * (n * n)
    for r in range(n):
        for s in range(n):
            old_idx = n * r + s
            new_idx = n * pi_r[r] + pi_s[s]
            perm[old_idx] = new_idx
    return tuple(perm)


def induced_beta_perm(n: int, action: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> tuple[int, ...]:
    _, pi_s, pi_u = action
    perm = [0] * (n * n)
    for t in range(n):
        for u in range(n):
            old_idx = n * t + u
            new_idx = n * pi_s[t] + pi_u[u]
            perm[old_idx] = new_idx
    return tuple(perm)


def cycle_lengths(perm: tuple[int, ...]) -> list[int]:
    seen = [False] * len(perm)
    lengths: list[int] = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        cur = start
        length = 0
        while not seen[cur]:
            seen[cur] = True
            cur = perm[cur]
            length += 1
        lengths.append(length)
    return lengths


def count_projective_fixed_classes(perm: tuple[int, ...]) -> int:
    lengths = cycle_lengths(perm)
    cycle_count = len(lengths)
    even_cycle_count = sum(1 for length in lengths if length % 2 == 0)
    fixed_plus = (3 ** cycle_count - 1) // 2
    fixed_minus = (3 ** even_cycle_count - 1) // 2
    return fixed_plus + fixed_minus


def burnside_orbit_count(n: int) -> tuple[int, list[dict]]:
    actions = build_actions(n)
    rows: list[dict] = []
    total_fixed = 0
    for action_id, action in enumerate(actions):
        alpha_perm = induced_alpha_perm(n, action)
        beta_perm = induced_beta_perm(n, action)
        alpha_fixed = count_projective_fixed_classes(alpha_perm)
        beta_fixed = count_projective_fixed_classes(beta_perm)
        fixed_profiles = alpha_fixed * beta_fixed
        total_fixed += fixed_profiles
        rows.append({
            'action_id': action_id,
            'pi_rA': action[0],
            'pi_shared': action[1],
            'pi_cB': action[2],
            'alpha_cycle_lengths': cycle_lengths(alpha_perm),
            'beta_cycle_lengths': cycle_lengths(beta_perm),
            'fixed_projective_alpha_lines': alpha_fixed,
            'fixed_projective_beta_lines': beta_fixed,
            'fixed_profile_classes': fixed_profiles,
            'provenance': 'EXACT_DERIVED',
        })
    orbit_count = total_fixed // len(actions)
    return orbit_count, rows


def profile_target_matrix(n: int) -> np.ndarray:
    target = np.zeros((n * n, n * n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                a_idx = n * r + s
                b_idx = n * s + u
                target[a_idx, b_idx] = 1.0
    return target


def full_tensor_target(n: int) -> np.ndarray:
    tensor = np.zeros((n * n, n * n, n * n), dtype=np.float64)
    for r in range(n):
        for s in range(n):
            for u in range(n):
                a_idx = n * r + s
                b_idx = n * s + u
                c_idx = n * r + u
                tensor[a_idx, b_idx, c_idx] = 1.0
    return tensor


def collapsed_score_features(vectors: np.ndarray) -> dict[str, np.ndarray]:
    n = int(round(sqrt(vectors.shape[1])))
    mats = vectors.reshape((-1, n, n)).astype(np.int16)
    return {
        'column_sums': mats.sum(axis=1).astype(np.int16),
        'row_sums': mats.sum(axis=2).astype(np.int16),
        'column_counts': (mats != 0).sum(axis=1).astype(np.int16),
        'row_counts': (mats != 0).sum(axis=2).astype(np.int16),
        'total_nnz': (mats != 0).sum(axis=(1, 2)).astype(np.int16),
        'col_dot_01': (mats[:, :, 0] * mats[:, :, 1]).sum(axis=1).astype(np.int16),
        'col_dot_12': (mats[:, :, 1] * mats[:, :, 2]).sum(axis=1).astype(np.int16),
        'row_dot_01': (mats[:, 0, :] * mats[:, 1, :]).sum(axis=1).astype(np.int16),
        'row_dot_12': (mats[:, 1, :] * mats[:, 2, :]).sum(axis=1).astype(np.int16),
        'matrices': mats.astype(np.int8),
    }


def compute_top_usefulness_profiles(alpha_vectors: np.ndarray, beta_vectors: np.ndarray) -> tuple[list[dict], dict[str, int]]:
    alpha_feat = collapsed_score_features(alpha_vectors)
    beta_feat = collapsed_score_features(beta_vectors)
    heap: list[tuple[float, int, int, int, int]] = []
    zero_projection_count = 0
    dead_free_count = 0
    processed = 0

    for start in range(0, beta_vectors.shape[0], BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, beta_vectors.shape[0])
        beta_sum_block = beta_feat['row_sums'][start:stop]
        beta_total_block = beta_feat['total_nnz'][start:stop].astype(np.int32)
        beta_row1_block = beta_feat['row_counts'][start:stop, 1].astype(np.int32)
        beta_dot01_block = beta_feat['row_dot_01'][start:stop].astype(np.int32)
        beta_dot12_block = beta_feat['row_dot_12'][start:stop].astype(np.int32)

        projection = alpha_feat['column_sums'].astype(np.int32) @ beta_sum_block.astype(np.int32).T
        nuisance_sq = (
            alpha_feat['total_nnz'].astype(np.int32)[:, None] * beta_total_block[None, :]
            + alpha_feat['column_counts'][:, 1].astype(np.int32)[:, None] * beta_row1_block[None, :]
            - 2 * alpha_feat['col_dot_01'].astype(np.int32)[:, None] * beta_dot01_block[None, :]
            - 2 * alpha_feat['col_dot_12'].astype(np.int32)[:, None] * beta_dot12_block[None, :]
        )
        nuisance_sq = nuisance_sq.astype(np.int32)

        zero_projection_count += int(np.count_nonzero(projection == 0))
        dead_free_mask = nuisance_sq == 0
        dead_free_count += int(np.count_nonzero(dead_free_mask & (projection != 0)))

        score = np.zeros_like(projection, dtype=np.float64)
        valid = nuisance_sq > 0
        score[valid] = np.abs(projection[valid]) / np.sqrt(nuisance_sq[valid].astype(np.float64))
        score[dead_free_mask & (projection != 0)] = np.inf

        processed += score.size
        flat = score.ravel()
        take = min(TOP_K * 4, flat.size)
        idx = np.argpartition(flat, -take)[-take:]
        for flat_idx in idx:
            local_alpha = int(flat_idx // score.shape[1])
            local_beta = int(flat_idx % score.shape[1])
            global_beta = start + local_beta
            entry = (
                float(score[local_alpha, local_beta]),
                int(abs(projection[local_alpha, local_beta])),
                int(nuisance_sq[local_alpha, local_beta]),
                local_alpha,
                global_beta,
            )
            if len(heap) < TOP_K:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    top_rows: list[dict] = []
    for rank_idx, (score_value, projection_abs, nuisance_sq, alpha_idx, beta_idx) in enumerate(sorted(heap, reverse=True), start=1):
        top_rows.append({
            'rank': rank_idx,
            'alpha_index': alpha_idx,
            'beta_index': beta_idx,
            'alpha': matrix_to_str(alpha_feat['matrices'][alpha_idx]),
            'beta': matrix_to_str(beta_feat['matrices'][beta_idx]),
            'projection_onto_tensor': projection_abs,
            'nuisance_magnitude_squared': nuisance_sq,
            'usefulness_score': score_value,
            'dead_free': nuisance_sq == 0,
            'provenance': 'EXACT_DERIVED',
        })
    stats = {
        'processed_profile_count': processed,
        'zero_projection_count': zero_projection_count,
        'dead_free_nonzero_count': dead_free_count,
    }
    return top_rows, stats


def best_profile_81d(residual: np.ndarray, canonical_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if np.max(np.abs(residual)) < 1e-12:
        raise RuntimeError('Residual is already zero.')
    y_norm_sq = np.sum(canonical_vectors.astype(np.float64) ** 2, axis=1)
    projections = residual @ canonical_vectors.T.astype(np.float64)
    abs_projections = np.abs(projections)
    order = np.argsort(abs_projections, axis=0)[::-1]
    sorted_abs = np.take_along_axis(abs_projections, order, axis=0)
    cumsums = np.cumsum(sorted_abs, axis=0)
    k_vals = np.arange(1, residual.shape[0] + 1, dtype=np.float64)[:, None]
    scores = (cumsums ** 2) / (k_vals * y_norm_sq[None, :])
    flat_idx = int(np.argmax(scores))
    best_k_idx, best_y_idx = np.unravel_index(flat_idx, scores.shape)
    best_k = best_k_idx + 1
    y = canonical_vectors[best_y_idx].astype(np.float64)
    top_coords = order[:best_k, best_y_idx]
    x = np.zeros(residual.shape[0], dtype=np.float64)
    p = projections[:, best_y_idx]
    x[top_coords] = np.sign(p[top_coords])
    if np.all(x == 0):
        raise RuntimeError('Degenerate 81D greedy selection.')
    nz = np.flatnonzero(x)
    if x[nz[0]] < 0:
        x = -x
    weight = float((x @ residual @ y) / ((x @ x) * (y @ y)))
    score_drop = float(scores[best_k_idx, best_y_idx])
    return x, y, weight, score_drop


def run_collapsed_greedy(n: int, max_rank: int) -> list[dict]:
    vectors = ternary_projective_vectors(n * n)
    residual = profile_target_matrix(n)
    rows: list[dict] = []
    finished = False
    for rank_idx in range(1, max_rank + 1):
        if finished:
            rows.append({
                'rank_step': rank_idx,
                'alpha': '',
                'beta': '',
                'weight': 0.0,
                'score_drop': 0.0,
                'residual_norm_squared': float(np.sum(residual * residual)),
                'exact_zero': True,
                'provenance': 'MEASURED_FROM_CODE',
            })
            continue
        x, y, weight, score_drop = best_profile_81d(residual, vectors)
        residual = residual - weight * np.outer(x, y)
        exact_zero = bool(np.max(np.abs(residual)) < 1e-10)
        rows.append({
            'rank_step': rank_idx,
            'alpha': matrix_to_str(x.reshape(n, n)),
            'beta': matrix_to_str(y.reshape(n, n)),
            'weight': weight,
            'score_drop': score_drop,
            'residual_norm_squared': float(np.sum(residual * residual)),
            'exact_zero': exact_zero,
            'provenance': 'MEASURED_FROM_CODE',
        })
        finished = exact_zero
    return rows


def canonical_profile_sign(vector: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(vector)
    if nz.size == 0:
        return vector
    if vector[nz[0]] < 0:
        return -vector
    return vector


def enumerate_2x2_profiles() -> tuple[list[Profile2x2], list[dict], int, int]:
    profiles: dict[bytes, Profile2x2] = {}
    raw_pair_count = 0
    zero_pair_count = 0
    ternary_entries = list(product((-1, 0, 1), repeat=4))
    for alpha_entries in ternary_entries:
        alpha = np.array(alpha_entries, dtype=np.int8).reshape(2, 2)
        for beta_entries in ternary_entries:
            raw_pair_count += 1
            beta = np.array(beta_entries, dtype=np.int8).reshape(2, 2)
            if not np.any(alpha) or not np.any(beta):
                zero_pair_count += 1
                continue
            profile = np.outer(alpha.reshape(-1), beta.reshape(-1)).reshape(-1)
            profile = canonical_profile_sign(profile)
            key = profile.tobytes()
            profiles.setdefault(key, Profile2x2(alpha=alpha.copy(), beta=beta.copy(), profile=profile.copy()))
    profile_list = list(profiles.values())
    rows = []
    for idx, item in enumerate(profile_list):
        rows.append({
            'profile_id': idx,
            'alpha': matrix_to_str(item.alpha),
            'beta': matrix_to_str(item.beta),
            'profile': str(item.profile.astype(int).tolist()),
            'provenance': 'EXACT_DERIVED',
        })
    return profile_list, rows, raw_pair_count, zero_pair_count


def canonicalize_profile_orbit(profile: np.ndarray, n: int, actions: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    tensor = profile.reshape((n, n, n, n))
    for action in actions:
        pi_r, pi_s, pi_u = action
        moved = np.zeros_like(tensor)
        for r in range(n):
            for s in range(n):
                for t in range(n):
                    for u in range(n):
                        moved[pi_r[r], pi_s[s], pi_s[t], pi_u[u]] = tensor[r, s, t, u]
        flat = moved.reshape(-1)
        flat = canonical_profile_sign(flat)
        key = tuple(int(x) for x in flat)
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def strassen_profile_keys() -> set[tuple[int, ...]]:
    terms = [
        {'alpha': np.array([[1, 0], [0, 1]], dtype=np.int8), 'beta': np.array([[1, 0], [0, 1]], dtype=np.int8)},
        {'alpha': np.array([[0, 0], [1, 1]], dtype=np.int8), 'beta': np.array([[1, 0], [0, 0]], dtype=np.int8)},
        {'alpha': np.array([[1, 0], [0, 0]], dtype=np.int8), 'beta': np.array([[0, 1], [0, -1]], dtype=np.int8)},
        {'alpha': np.array([[0, 0], [0, 1]], dtype=np.int8), 'beta': np.array([[-1, 0], [1, 0]], dtype=np.int8)},
        {'alpha': np.array([[1, 1], [0, 0]], dtype=np.int8), 'beta': np.array([[0, 0], [0, 1]], dtype=np.int8)},
        {'alpha': np.array([[-1, 0], [1, 0]], dtype=np.int8), 'beta': np.array([[1, 1], [0, 0]], dtype=np.int8)},
        {'alpha': np.array([[0, 1], [0, -1]], dtype=np.int8), 'beta': np.array([[0, 0], [1, 1]], dtype=np.int8)},
    ]
    keys: set[tuple[int, ...]] = set()
    for term in terms:
        profile = np.outer(term['alpha'].reshape(-1), term['beta'].reshape(-1)).reshape(-1)
        profile = canonical_profile_sign(profile)
        keys.add(tuple(int(x) for x in profile))
    return keys


def run_full_tensor_greedy_2x2(profiles: list[Profile2x2], max_rank: int) -> tuple[list[dict], list[dict]]:
    target = full_tensor_target(2).reshape(16, 4)
    residual = target.copy()
    rows: list[dict] = []
    selected_rows: list[dict] = []
    strassen_keys = strassen_profile_keys()
    for rank_idx in range(1, max_rank + 1):
        best_idx = -1
        best_gain = -1.0
        best_gamma = None
        for idx, item in enumerate(profiles):
            profile = item.profile.astype(np.float64)
            norm_sq = float(profile @ profile)
            gamma = (profile @ residual) / norm_sq
            gain = float(np.sum(gamma * gamma) * norm_sq)
            if gain > best_gain:
                best_idx = idx
                best_gain = gain
                best_gamma = gamma
        assert best_idx >= 0 and best_gamma is not None
        chosen = profiles[best_idx]
        profile = chosen.profile.astype(np.float64)
        residual = residual - np.outer(profile, best_gamma)
        profile_key = tuple(int(x) for x in chosen.profile)
        rows.append({
            'rank_step': rank_idx,
            'profile_id': best_idx,
            'alpha': matrix_to_str(chosen.alpha),
            'beta': matrix_to_str(chosen.beta),
            'gamma': float_matrix_to_str(best_gamma.reshape(2, 2)),
            'residual_norm_squared': float(np.sum(residual * residual)),
            'exact_zero': bool(np.max(np.abs(residual)) < 1e-10),
            'profile_matches_strassen': profile_key in strassen_keys,
            'provenance': 'MEASURED_FROM_CODE',
        })
        selected_rows.append({
            'rank_step': rank_idx,
            'profile_id': best_idx,
            'profile_matches_strassen': profile_key in strassen_keys,
            'provenance': 'MEASURED_FROM_CODE',
        })
    return rows, selected_rows


def markdown_summary(summary_rows: list[dict], top_rows: list[dict], greedy3_rows: list[dict], pool2_rows: list[dict], greedy2_rows: list[dict]) -> str:
    summary_map = {row['summary_name']: row['summary_value'] for row in summary_rows}
    lines: list[str] = []
    w = lines.append
    w('# Step 64: Small Integer Coefficient Enumeration')
    w(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    w('')
    w('[EXACT_DERIVED] + [MEASURED_FROM_CODE]')
    w('')
    w('Step 64 treats alpha,beta in {-1,0,1}^{n x n} as a finite exact pool. The main exact reduction is that a bilinear profile is an outer product of two nonzero ternary vectors, so duplicate elimination and sign quotienting are projective-line counts rather than a 387M-object hash table.')
    w('')
    w('## Pool Statistics')
    w('')
    w(f"- 3x3 raw pairs = {summary_map['pool3_raw_pairs']}")
    w(f"- 3x3 nonzero pairs = {summary_map['pool3_nonzero_pairs']}")
    w(f"- 3x3 distinct nonzero profiles modulo duplicates and negation = {summary_map['pool3_distinct_profiles_mod_sign']}")
    w(f"- 3x3 symmetry-reduced profile orbits = {summary_map['pool3_symmetry_reduced_profiles']}")
    w(f"- 3x3 dead-free nonzero profiles = {summary_map['pool3_deadfree_nonzero_profiles']}")
    w('')
    w('## Top Usefulness Profiles')
    w('')
    for row in top_rows[:10]:
        w(f"- rank {row['rank']}: score={row['usefulness_score']:.6f}, projection={row['projection_onto_tensor']}, nuisance_sq={row['nuisance_magnitude_squared']}")
    w('')
    w('## 3x3 Collapsed 81D Greedy')
    w('')
    exact3 = [row for row in greedy3_rows if row['exact_zero']]
    if exact3:
        w(f"- first exact zero residual at rank {exact3[0]['rank_step']}")
    else:
        w('- no exact zero residual through the tested rank range')
    w('')
    w('## 2x2 Verification')
    w('')
    exact2 = [row for row in greedy2_rows if row['exact_zero']]
    if exact2:
        w(f"- corrected full-tensor greedy first reaches exact zero at rank {exact2[0]['rank_step']}")
    else:
        w('- corrected full-tensor greedy does not reach exact zero through rank 7')
    w(f"- 2x2 distinct profiles modulo sign = {summary_map['pool2_distinct_profiles_mod_sign']}")
    w(f"- 2x2 symmetry-reduced profiles = {summary_map['pool2_symmetry_reduced_profiles']}")
    w('')
    w('[INTERPRETATION]')
    w('')
    w('The finite pool itself is exact and much larger than the initial rough expectation: the 3x3 duplicate/negation quotient already leaves 96,845,281 distinct nonzero profiles before symmetry reduction. The collapsed 81D greedy is therefore a search over a genuine large finite dictionary, but it is still too weak to certify a full matrix-multiplication algorithm because it ignores the 9-component gamma output layer. That is why the 2x2 validation in this step uses the corrected full tensor with optimal gamma vectors per selected profile.')
    return '\n'.join(lines)


def main() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)

    vecs3 = ternary_projective_vectors(9)
    vecs2 = ternary_projective_vectors(4)

    distinct3 = int(vecs3.shape[0] ** 2)
    raw_pairs3 = 3 ** 18
    nonzero_pairs3 = (3 ** 9 - 1) ** 2
    sym3, burnside3_rows = burnside_orbit_count(3)
    sym2, burnside2_rows = burnside_orbit_count(2)

    top100_rows, top_stats = compute_top_usefulness_profiles(vecs3, vecs3)
    greedy3_rows = run_collapsed_greedy(3, 27)
    greedy2_collapsed_rows = run_collapsed_greedy(2, 7)

    profiles2, pool2_rows, raw_pairs2, zero_pairs2 = enumerate_2x2_profiles()
    actions2 = build_actions(2)
    orbit2_reps = {canonicalize_profile_orbit(item.profile, 2, actions2) for item in profiles2}
    greedy2_rows, greedy2_selected_rows = run_full_tensor_greedy_2x2(profiles2, 7)

    summary_rows = [
        {'summary_name': 'pool3_raw_pairs', 'summary_value': str(raw_pairs3), 'provenance': 'EXACT_DERIVED', 'note': 'Total 3x3 alpha,beta raw pairs in {-1,0,1}^{9} x {-1,0,1}^{9}.'},
        {'summary_name': 'pool3_nonzero_pairs', 'summary_value': str(nonzero_pairs3), 'provenance': 'EXACT_DERIVED', 'note': 'Raw pairs with alpha!=0 and beta!=0.'},
        {'summary_name': 'pool3_projective_line_count', 'summary_value': str(vecs3.shape[0]), 'provenance': 'EXACT_DERIVED', 'note': 'Nonzero ternary 9-vectors modulo global sign.'},
        {'summary_name': 'pool3_distinct_profiles_mod_sign', 'summary_value': str(distinct3), 'provenance': 'EXACT_DERIVED', 'note': 'Distinct nonzero bilinear profiles after duplicate and negation quotient.'},
        {'summary_name': 'pool3_symmetry_reduced_profiles', 'summary_value': str(sym3), 'provenance': 'EXACT_DERIVED', 'note': 'Burnside orbit count under the 216-element symmetry group.'},
        {'summary_name': 'pool3_zero_projection_profiles', 'summary_value': str(top_stats['zero_projection_count']), 'provenance': 'EXACT_DERIVED', 'note': 'Profiles whose projection onto the live tensor is zero.'},
        {'summary_name': 'pool3_deadfree_nonzero_profiles', 'summary_value': str(top_stats['dead_free_nonzero_count']), 'provenance': 'EXACT_DERIVED', 'note': 'Profiles with zero nuisance magnitude and nonzero tensor projection.'},
        {'summary_name': 'collapsed_3x3_greedy_first_exact_rank', 'summary_value': str(next((row['rank_step'] for row in greedy3_rows if row['exact_zero']), 'none')), 'provenance': 'MEASURED_FROM_CODE', 'note': 'First exact zero residual rank for the collapsed 81D greedy.'},
        {'summary_name': 'collapsed_2x2_greedy_first_exact_rank', 'summary_value': str(next((row['rank_step'] for row in greedy2_collapsed_rows if row['exact_zero']), 'none')), 'provenance': 'MEASURED_FROM_CODE', 'note': 'First exact zero residual rank for the collapsed 16D greedy.'},
        {'summary_name': 'pool2_raw_pairs', 'summary_value': str(raw_pairs2), 'provenance': 'EXACT_DERIVED', 'note': 'Total 2x2 alpha,beta raw pairs.'},
        {'summary_name': 'pool2_zero_pairs', 'summary_value': str(zero_pairs2), 'provenance': 'EXACT_DERIVED', 'note': '2x2 raw pairs with alpha=0 or beta=0.'},
        {'summary_name': 'pool2_distinct_profiles_mod_sign', 'summary_value': str(len(profiles2)), 'provenance': 'EXACT_DERIVED', 'note': 'Distinct 2x2 bilinear profiles after duplicate and negation quotient.'},
        {'summary_name': 'pool2_symmetry_reduced_profiles', 'summary_value': str(len(orbit2_reps)), 'provenance': 'EXACT_DERIVED', 'note': 'Explicit 2x2 symmetry-reduced profile count.'},
        {'summary_name': 'full_tensor_2x2_greedy_first_exact_rank', 'summary_value': str(next((row['rank_step'] for row in greedy2_rows if row['exact_zero']), 'none')), 'provenance': 'MEASURED_FROM_CODE', 'note': 'First exact rank where corrected full-tensor greedy reaches zero on 2x2.'},
        {'summary_name': 'full_tensor_2x2_selected_strassen_matches', 'summary_value': str(sum(int(row['profile_matches_strassen']) for row in greedy2_selected_rows)), 'provenance': 'MEASURED_FROM_CODE', 'note': 'How many selected 2x2 greedy profiles are canonical Strassen bilinear profiles.'},
    ]

    write_csv(EXPORTS / 'step64_summary.csv', summary_rows, ['summary_name', 'summary_value', 'provenance', 'note'])
    write_csv(EXPORTS / 'step64_burnside_3x3_actions.csv', burnside3_rows, ['action_id', 'pi_rA', 'pi_shared', 'pi_cB', 'alpha_cycle_lengths', 'beta_cycle_lengths', 'fixed_projective_alpha_lines', 'fixed_projective_beta_lines', 'fixed_profile_classes', 'provenance'])
    write_csv(EXPORTS / 'step64_burnside_2x2_actions.csv', burnside2_rows, ['action_id', 'pi_rA', 'pi_shared', 'pi_cB', 'alpha_cycle_lengths', 'beta_cycle_lengths', 'fixed_projective_alpha_lines', 'fixed_projective_beta_lines', 'fixed_profile_classes', 'provenance'])
    write_csv(EXPORTS / 'step64_top100_usefulness_profiles.csv', top100_rows, ['rank', 'alpha_index', 'beta_index', 'alpha', 'beta', 'projection_onto_tensor', 'nuisance_magnitude_squared', 'usefulness_score', 'dead_free', 'provenance'])
    write_csv(EXPORTS / 'step64_3x3_collapsed_greedy.csv', greedy3_rows, ['rank_step', 'alpha', 'beta', 'weight', 'score_drop', 'residual_norm_squared', 'exact_zero', 'provenance'])
    write_csv(EXPORTS / 'step64_2x2_collapsed_greedy.csv', greedy2_collapsed_rows, ['rank_step', 'alpha', 'beta', 'weight', 'score_drop', 'residual_norm_squared', 'exact_zero', 'provenance'])
    write_csv(EXPORTS / 'step64_2x2_profile_pool.csv', pool2_rows, ['profile_id', 'alpha', 'beta', 'profile', 'provenance'])
    write_csv(EXPORTS / 'step64_2x2_full_tensor_greedy.csv', greedy2_rows, ['rank_step', 'profile_id', 'alpha', 'beta', 'gamma', 'residual_norm_squared', 'exact_zero', 'profile_matches_strassen', 'provenance'])

    md_text = markdown_summary(summary_rows, top100_rows, greedy3_rows, pool2_rows, greedy2_rows)
    write_text(EXPORTS / 'step64_small_integer_coefficient_enumeration.md', md_text)

    print(f"3x3 projective lines: {vecs3.shape[0]}")
    print(f"3x3 distinct profiles mod sign: {distinct3}")
    print(f"3x3 symmetry-reduced profile count: {sym3}")
    print(f"2x2 distinct profiles mod sign: {len(profiles2)}")
    print(f"2x2 full-tensor greedy first exact rank: {next((row['rank_step'] for row in greedy2_rows if row['exact_zero']), 'none')}")


if __name__ == '__main__':
    main()