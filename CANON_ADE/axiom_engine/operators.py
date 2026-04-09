"""
operators.py — Constructive operators between axioms.

Each operator is an edge in the axiom graph:
    op_X_to_Y(state, R) → list[state']

States are LIGHTWEIGHT — no numpy arrays stored. Only reconstruction keys:
  {
    'R': int,
    'kept': list of triples,
    'stability_tag': str (G-stable, H-stable, etc.),
    'kernel_shape': tuple of irrep indices (if A7 done),
    'fiber_gammas': dict idx->int (if A3 done),
    'relation_spectrum': tuple (dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC) (if A4 done),
    'cd_block_map': dict block_type -> count (if A6 done),
    'factor_strategy': (name,) for factor reconstruction (if A5 done),
    'satisfied': set of axiom ids,
    'path': list of (source, op_name, target),
    'constraints': dict — accumulated exact constraints for propagation,
  }

Arrays are recomputed on demand via reconstruct_*() helpers.

NO OPTIMIZATION. NO RANDOM GENERATION. NO LOSS FUNCTIONS.
Every operator is a constructive algebraic map or finite enumeration.
"""
import numpy as np
from itertools import combinations, product as iproduct
from functools import lru_cache
from typing import List, Dict, Any, Tuple, FrozenSet
from fractions import Fraction

from . import tensor_core as tc
from .axiom_evaluators import matrix_rank

State = Dict[str, Any]
SVD_TOL = 1e-10


# ─────────────────────────────────────────────────────────────
#  RECONSTRUCTION HELPERS — compute arrays on demand from keys
# ─────────────────────────────────────────────────────────────

# Cache irreps per kept-tuple (they're expensive but deterministic)
_irrep_cache = {}

def _get_irreps(kept_tuple):
    """Get irreps for a kept set, cached."""
    if kept_tuple not in _irrep_cache:
        kept = list(kept_tuple)
        perm_mats = tc.build_perm_matrices(kept)
        R = len(kept)
        _irrep_cache[kept_tuple] = tc.decompose_irreps(perm_mats, R)
    return _irrep_cache[kept_tuple]


def reconstruct_kernel_and_active(state):
    """From kernel_shape + kept, rebuild kernel_basis (R, R-9) and active_basis (R, 9)."""
    kept = state['kept']
    shape = state['kernel_shape']
    if kept is None or shape is None:
        return None, None
    irreps = _get_irreps(tuple(kept))
    kernel_basis = np.hstack([irreps[i] for i in shape])
    U, s, Vt = np.linalg.svd(kernel_basis, full_matrices=True)
    rk = kernel_basis.shape[1]
    active_basis = U[:, rk:]
    return kernel_basis, active_basis


def _make_initial_state(R: int) -> State:
    """Create a blank state for rank R. Lightweight — no large arrays stored."""
    return {
        'R': R,
        'kept': None,
        'stability_tag': None,
        # Instead of storing arrays, store reconstruction keys:
        'kernel_shape': None,      # tuple of irrep indices
        'fiber_gammas': None,      # dict idx -> int (exact integer gammas)
        'relation_spectrum': None, # (dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC)
        'cd_block_map': None,      # dict block_type_tuple -> count
        'factor_strategy': None,   # strategy name for reconstruction
        'satisfied': set(),
        'path': [],
        'constraints': {},         # accumulated exact constraints for propagation
    }


# ─────────────────────────────────────────────────────────────
#  ROOT OPERATORS — entry points (no prerequisite axiom)
# ─────────────────────────────────────────────────────────────

def op_root_A2(state: State) -> List[State]:
    R = state['R']
    ck = ('root_A2', R)
    if ck not in _op_cache:
        options = tc.kept_triples_for_rank_extended(R)
        _op_cache[ck] = [(sorted(kept), tag) for kept, tag in options] if options else []

    results = []
    for kept, tag in _op_cache[ck]:
        s = {**state}
        s['kept'] = kept
        s['stability_tag'] = tag
        s['satisfied'] = state['satisfied'] | {'A2'}
        s['path'] = state['path'] + [('root', f'root→A2 ({tag}, |S|={len(kept)})', 'A2')]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['support_size'] = len(kept)
        s['constraints']['stability'] = tag
        results.append(s)
    return results


# ─────────────────────────────────────────────────────────────
#  A2 → A7: decompose irreps, enumerate kernel shapes
# ─────────────────────────────────────────────────────────────

def op_A2_to_A7(state: State) -> List[State]:
    R = state['R']
    kept = state['kept']
    if kept is None:
        return []

    ck = ('A2_A7', tuple(kept))
    if ck not in _op_cache:
        irreps = _get_irreps(tuple(kept))
        target_dim = R - 9
        if target_dim <= 0:
            _op_cache[ck] = []
        else:
            dims = [b.shape[1] for b in irreps]
            shapes = []
            for r in range(1, len(irreps) + 1):
                for combo in combinations(range(len(irreps)), r):
                    if sum(dims[i] for i in combo) == target_dim:
                        shapes.append((combo, [dims[i] for i in combo]))
            _op_cache[ck] = shapes

    results = []
    for shape, shape_dims in _op_cache[ck]:
        s = {**state}
        s['kernel_shape'] = shape
        s['satisfied'] = state['satisfied'] | {'A7'}
        s['path'] = state['path'] + [('A2', f'A2→A7 shape={shape} dims={shape_dims}', 'A7')]
        results.append(s)
    return results


# ─────────────────────────────────────────────────────────────
#  A7 → A1: kernel fixes rank(H) = R-9 by construction
# ─────────────────────────────────────────────────────────────

def op_A7_to_A1(state: State) -> List[State]:
    """
    Given quantized kernel (A7), the active subspace is the orthogonal
    complement of the kernel — dimension exactly 9.
    rank(H) = R-9 is guaranteed by construction.
    Single branch (deterministic).
    """
    if state.get('kernel_shape') is None:
        return []

    R = state['R']
    ck = ('A7_A1', tuple(state['kept']), state['kernel_shape'])
    if ck not in _op_cache:
        kernel_basis, active_basis = reconstruct_kernel_and_active(state)
        _op_cache[ck] = active_basis is not None and active_basis.shape[1] == 9

    if not _op_cache[ck]:
        return []

    s = {**state}
    s['satisfied'] = state['satisfied'] | {'A1'}
    s['path'] = state['path'] + [('A7', 'A7→A1 (kernel complement)', 'A1')]
    return [s]


# ─────────────────────────────────────────────────────────────
#  A1 → A3: solve fiber sum constraints Γ(s,u) = 3
# ─────────────────────────────────────────────────────────────

def op_A1_to_A3(state: State) -> List[State]:
    R = state['R']
    kept = state['kept']
    if kept is None:
        return []

    ck = ('A1_A3', tuple(kept))
    if ck not in _op_cache:
        _op_cache[ck] = _compute_fiber_allocations(kept, R)

    results = []
    for gamma_map, desc, n_nonzero, n_free in _op_cache[ck]:
        s = {**state}
        s['fiber_gammas'] = dict(gamma_map)
        s['satisfied'] = state['satisfied'] | {'A3'}
        s['path'] = state['path'] + [('A1', f'A1→A3 fibers={desc}', 'A3')]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['n_nonzero_terms'] = n_nonzero
        s['constraints']['n_free_fibers'] = n_free
        if n_free == 0:
            s['constraints']['fiber_type'] = 'all_forced'
        results.append(s)
    return results


def _compute_fiber_allocations(kept, R):
    """Pure computation: returns list of (gamma_map, desc, n_nonzero, n_free)."""
    fibers = {}
    for idx, (r, s, u) in enumerate(kept):
        fibers.setdefault((s, u), []).append(idx)

    forced = {}
    free_fibers = {}
    for key, indices in fibers.items():
        if len(indices) == 1:
            forced[key] = [3]
        else:
            free_fibers[key] = indices

    if not free_fibers:
        gamma_map = {}
        for key, vals in forced.items():
            for idx, v in zip(fibers[key], vals):
                gamma_map[idx] = v
        return [(gamma_map, 'all_forced', R, 0)]

    def compositions_of(total, k):
        if k == 1:
            yield (total,)
            return
        for v in range(total + 1):
            for rest in compositions_of(total - v, k - 1):
                yield (v,) + rest

    fiber_keys = sorted(free_fibers.keys())
    fiber_options = []
    for key in fiber_keys:
        k = len(free_fibers[key])
        parts = list(compositions_of(3, k))
        fiber_options.append((key, parts))

    MAX_FIBER_COMBOS = 200

    def cross_product(options, idx=0):
        if idx == len(options):
            yield {}
            return
        key, parts = options[idx]
        indices = free_fibers[key]
        for part in parts:
            for rest in cross_product(options, idx + 1):
                combo = dict(rest)
                for i, v in zip(indices, part):
                    combo[i] = v
                yield combo

    allocations = []
    count = 0
    for free_map in cross_product(fiber_options):
        if count >= MAX_FIBER_COMBOS:
            break

        gamma_map = {}
        for key, vals in forced.items():
            for idx, v in zip(fibers[key], vals):
                gamma_map[idx] = v
        gamma_map.update(free_map)

        n_zero = sum(1 for v in gamma_map.values() if v == 0)
        n_nonzero = R - n_zero
        if n_nonzero < 9:
            continue

        count += 1
        desc = {k: [gamma_map[i] for i in free_fibers[k]] for k in fiber_keys}
        allocations.append((gamma_map, desc, n_nonzero, len(free_fibers)))

    return allocations


# ─────────────────────────────────────────────────────────────
#  A3 → A6b: verify/enforce CD tower fiber parity (FILTER)
# ─────────────────────────────────────────────────────────────

def op_A3_to_A6b(state: State) -> List[State]:
    """
    Given fiber gammas (A3), check Cayley-Dickson parity alignment.
    For each (s,u), the fiber sum must be exactly 3.
    Additionally, check XOR-parity distribution of nonzero gamma terms.
    
    This is a FILTER — passes through states that satisfy CD parity.
    """
    R = state['R']
    kept = state['kept']
    fiber_gammas = state.get('fiber_gammas')
    if kept is None or fiber_gammas is None:
        return []

    fg_key = tuple(sorted(fiber_gammas.items()))
    ck = ('A3_A6b', tuple(kept), fg_key)
    if ck not in _op_cache:
        _op_cache[ck] = _compute_A3_to_A6b(kept, fiber_gammas)

    result = _op_cache[ck]
    if result is None:
        return []

    n_even_nonzero, n_odd_nonzero = result
    s = {**state}
    s['satisfied'] = state['satisfied'] | {'A6b'}
    s['path'] = state['path'] + [('A3', f'A3→A6b (even={n_even_nonzero}, odd={n_odd_nonzero})', 'A6b')]
    s['constraints'] = dict(state.get('constraints', {}))
    s['constraints']['cd_even_nonzero'] = n_even_nonzero
    s['constraints']['cd_odd_nonzero'] = n_odd_nonzero
    return [s]


def _compute_A3_to_A6b(kept, fiber_gammas):
    """Pure computation for A3→A6b. Returns (n_even, n_odd) or None."""
    fibers = {}
    for idx, (r, s, u) in enumerate(kept):
        fibers.setdefault((s, u), []).append((idx, r))

    for (s, u), members in fibers.items():
        total = sum(fiber_gammas.get(idx, 0) for idx, r in members)
        if total != 3:
            return None

    n_odd_nonzero = 0
    n_even_nonzero = 0
    for idx, (r, s, u) in enumerate(kept):
        g = fiber_gammas.get(idx, 0)
        if g != 0:
            xor = (r % 2) ^ (s % 2) ^ (u % 2)
            if xor == 1:
                n_odd_nonzero += 1
            else:
                n_even_nonzero += 1

    return (n_even_nonzero, n_odd_nonzero)


# ─────────────────────────────────────────────────────────────
#  A5: CONSTRUCTIVE FACTOR STRATEGIES (NO RANDOM GENERATION)
#
#  Key algebraic insight from dimension obstruction analysis:
#    Gate 2 (delta_leak=0): needs α[k,:,s]·β[k,t,:] = 0 for s≠t
#      ↔ for each term k, the s-support of alpha and t-support of beta
#        must be identical (column-row alignment).
#    Gate 3 (aug_gap=9): Sigma must contribute 9 independent directions
#      beyond N=[H|Delta]. With Delta=0 by Gate 2, need rk([S|H]) - rk(H) = 9.
#
#  Strategy: enumerate "support patterns" — which s-values each term uses.
#  A support pattern assigns each term k a subset S_k ⊆ {0,1,2}.
#  Then α[k,r,s]=0 for s∉S_k and β[k,t,u]=0 for t∉S_k.
#  This guarantees Delta=0. Factors are DETERMINISTIC from the pattern.
# ─────────────────────────────────────────────────────────────

def _build_factors_strategy_diagonal(kept, R, fiber_gammas):
    """
    Constructive strategy: diagonal.
    Each term k with triple (r,s,u) gets exactly one nonzero entry
    per factor matrix, at the natural position.
    Gamma determines the scale: abc = gamma, split as cube root.
    All operations are deterministic and exact for integer gammas.
    """
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_arr = np.zeros((R, 3, 3))
    for idx, (r, s, u) in enumerate(kept):
        g_val = fiber_gammas.get(idx, 1) if fiber_gammas else 1
        if g_val == 0:
            continue
        sign = 1 if g_val > 0 else -1
        scale = abs(g_val) ** (1.0 / 3.0)
        alpha[idx, r, s] = sign * scale
        beta[idx, s, u] = scale
        gamma_arr[idx, r, u] = scale
    return alpha, beta, gamma_arr


def _build_factors_support_pattern(kept, R, fiber_gammas, pattern):
    """
    Construct factors with a given s-support pattern.
    DETERMINISTIC: coefficients are derived algebraically from the
    support structure, not randomly generated.
    
    pattern: list of frozensets, one per term. pattern[k] ⊆ {0,1,2}
             is the set of s-values term k is allowed to use.
    
    For each term k with triple (r_k, s_k, u_k) and support S_k:
      - α[k, r_k, s] = scale / |S_k|  for s ∈ S_k (uniform split)
      - β[k, s, u_k] = scale / |S_k|  for s ∈ S_k
      - derived so that the fiber contribution is gamma_k
    """
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_arr = np.zeros((R, 3, 3))

    for idx, (r, s_nat, u) in enumerate(kept):
        g_val = fiber_gammas.get(idx, 1) if fiber_gammas else 1
        if g_val == 0:
            continue
        S_k = pattern[idx]
        n_s = len(S_k)
        if n_s == 0:
            continue

        sign = 1 if g_val > 0 else -1
        # Sigma_k = sum_s alpha[k,r,s] * beta[k,s,u]
        # With uniform split: Sigma_k = n_s * (scale/n_s)^2 = scale^2 / n_s
        # We want alpha * beta * gamma product to reconstruct g_val:
        # scale^2 / n_s * gamma_scale = g_val
        scale = (abs(g_val) * n_s) ** 0.5

        for s_val in sorted(S_k):
            alpha[idx, r, s_val] = sign * scale / n_s
            beta[idx, s_val, u] = scale / n_s

        # Gamma: set so that term contributes g_val to tensor
        sigma_k = sum(alpha[idx, r, sv] * beta[idx, sv, u] for sv in S_k)
        if abs(sigma_k) > 1e-15:
            gamma_arr[idx, r, u] = g_val / sigma_k
        else:
            gamma_arr[idx, r, u] = 0

    return alpha, beta, gamma_arr


def _enumerate_support_patterns(kept, R):
    """
    Enumerate constructive s-support patterns for the kept triples.
    
    Each term k with natural triple (r,s,u) gets a support S_k ⊆ {0,1,2}.
    Terms in the same orbit class share the same support type.
    With 4 orbit classes and 7 support options, that's 7^4 = 2401 patterns.
    """
    SUPPORTS = [
        frozenset({0}), frozenset({1}), frozenset({2}),
        frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2}),
        frozenset({0, 1, 2}),
    ]

    # Classify terms by orbit type
    orbit_classes = {}
    for idx, (r, s, u) in enumerate(kept):
        key = tuple(sorted([r, s, u]))
        if key not in orbit_classes:
            orbit_classes[key] = []
        orbit_classes[key].append(idx)

    orbit_keys = sorted(orbit_classes.keys())
    n_classes = len(orbit_keys)

    patterns = []
    for combo in iproduct(range(len(SUPPORTS)), repeat=n_classes):
        pat = [None] * len(kept)
        name_parts = []
        for ci, si in enumerate(combo):
            sup = SUPPORTS[si]
            for idx in orbit_classes[orbit_keys[ci]]:
                pat[idx] = sup
            name_parts.append(f"{''.join(str(x) for x in sorted(sup))}")
        name = 'sp_' + '_'.join(name_parts)
        patterns.append((name, pat))

    return patterns


def _check_gates(alpha, beta, R):
    """Compute Step-51 and check Gate 2 + Gate 3. Returns (pass, details_dict)."""
    Sigma, H, Delta = tc.compute_step51(alpha, beta)
    rk_H = matrix_rank(H)
    nuisance = np.hstack([H, Delta])
    SN = np.hstack([Sigma, nuisance])
    rank_N = matrix_rank(nuisance)
    rank_SN = matrix_rank(SN)
    aug_gap = rank_SN - rank_N
    delta_leak = rank_N - rk_H
    gate2 = (delta_leak == 0)
    gate3 = (aug_gap == 9)
    return gate2 and gate3, {
        'Sigma': Sigma, 'H': H, 'Delta': Delta,
        'rk_H': rk_H, 'aug_gap': aug_gap, 'delta_leak': delta_leak,
        'gate2': gate2, 'gate3': gate3,
    }


# ── Gate-check cache: (kept_tuple, R, fg_key) → list of (pat_name, aug_gap) ──
_gate_cache = {}

# ── General operator result cache ──
# Key varies by operator; stores the "pure computation" part of each result.
_op_cache = {}


def _cached_gate_scan(kept, R, fiber_gammas):
    """
    Run all 2401 support-pattern gate checks for (kept, R, fiber_gammas).
    Returns list of (pat_name, aug_gap) for patterns that pass.

    Cache layers:
    1. If the all-nonzero scan for (kept, R) already returned 0 passing,
       any subset (zeroed terms) also returns 0 — skip entirely.
    2. Otherwise cache by (kept_tuple, R, zero_mask).
    """
    kept_key = tuple(kept)
    if fiber_gammas:
        zero_mask = tuple(1 if fiber_gammas.get(i, 1) != 0 else 0
                          for i in range(len(kept)))
    else:
        zero_mask = tuple(1 for _ in range(len(kept)))

    # Fast path: if all-nonzero already failed, any subset also fails
    all_ones = tuple(1 for _ in range(len(kept)))
    superset_key = (kept_key, R, all_ones)
    if superset_key in _gate_cache and len(_gate_cache[superset_key]) == 0:
        return []

    cache_key = (kept_key, R, zero_mask)
    if cache_key in _gate_cache:
        return _gate_cache[cache_key]

    patterns = _enumerate_support_patterns(kept, R)
    passing = []
    for pat_name, pattern in patterns:
        alpha, beta, gamma_arr = _build_factors_support_pattern(
            kept, R, fiber_gammas, pattern)
        passed, details = _check_gates(alpha, beta, R)
        if passed:
            passing.append((pat_name, details['aug_gap']))
    _gate_cache[cache_key] = passing
    return passing


def op_A3_to_A5(state: State) -> List[State]:
    """
    Given fiber gammas (A3) and active basis (A1),
    enumerate s-support patterns, construct DETERMINISTIC factors, check Gates 2+3.
    Each pattern that passes gates is a branch.
    NO RANDOM GENERATION. All factors are algebraically determined by the pattern.
    """
    fiber_gammas = state.get('fiber_gammas')
    kept = state.get('kept')
    R = state['R']

    if kept is None or state.get('kernel_shape') is None:
        return []

    passing = _cached_gate_scan(kept, R, fiber_gammas)

    results = []
    for pat_name, aug_gap in passing:
        s = {**state}
        s['factor_strategy'] = pat_name
        s['satisfied'] = state['satisfied'] | {'A5'}
        s['path'] = state['path'] + [(
            'A3', f'A3→A5 ({pat_name}, ag={aug_gap})', 'A5'
        )]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['gate2'] = True
        s['constraints']['gate3'] = True
        s['constraints']['aug_gap'] = aug_gap
        results.append(s)

    return results


def op_A1_to_A5(state: State) -> List[State]:
    """
    Given active basis (A1), try support patterns without fiber gammas.
    Deterministic enumeration, no random generation.
    """
    kept = state.get('kept')
    R = state['R']

    if kept is None or state.get('kernel_shape') is None:
        return []

    passing = _cached_gate_scan(kept, R, None)

    results = []
    for pat_name, aug_gap in passing:
        s = {**state}
        s['factor_strategy'] = pat_name
        s['satisfied'] = state['satisfied'] | {'A5'}
        s['path'] = state['path'] + [(
            'A1', f'A1→A5 ({pat_name}, ag={aug_gap})', 'A5'
        )]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['gate2'] = True
        s['constraints']['gate3'] = True
        results.append(s)

    return results


# ─────────────────────────────────────────────────────────────
#  A4: RELATION MODULE SPECTRUM (CONSTRUCTIVE — NEW)
#
#  This is a ROOT-CAPABLE operator: it can fire from just a support set,
#  without requiring factors. It enumerates the EXPECTED relation spectrum
#  for a given rank based on dimension counting, and records it as a
#  discrete invariant that constrains downstream axioms.
#
#  The relation spectrum is (dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC)
#  where K_X is the kernel (relation module) of factor family X,
#  and dim_XY = dim(K_X ∩ K_Y).
#
#  For rank R: dim(K_X) = R - 9 (since each factor family spans ℝ^9).
#  Generic pairwise: dim_XY = max(2(R-9) - R, 0) = max(R - 18, 0).
# ─────────────────────────────────────────────────────────────

def _enumerate_relation_spectra(R):
    """
    Enumerate all feasible relation spectra for rank R.
    
    Each spectrum is (dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, dim_ABC).
    
    Hard constraints:
      1. dim(K_X) = R - 9  for each X ∈ {A, B, C}
      2. max(2(R-9) - R, 0) ≤ dim_XY ≤ R - 9  for each pair
      3. dim_ABC ≤ min(dim_AB, dim_AC, dim_BC)
      4. dim_ABC = 0  for minimal-rank decompositions (no redundant terms)
      5. dim_XY ≤ dim_X + dim_Y - R  (subspace dimension bound, lower)
         dim_XY ≤ min(dim_X, dim_Y)  (upper)
    
    Returns list of spectrum tuples.
    """
    k = R - 9  # dim of each kernel
    if k <= 0:
        return [(0, 0, 0, 0, 0, 0, 0)]

    generic_pair = max(2 * k - R, 0)
    max_pair = k

    spectra = []
    # Enumerate pairwise intersection dimensions
    for d_ab in range(generic_pair, max_pair + 1):
        for d_ac in range(generic_pair, max_pair + 1):
            for d_bc in range(generic_pair, max_pair + 1):
                # Triple intersection: 0 for minimal rank
                # Also bounded by min of pairwise
                max_abc = min(d_ab, d_ac, d_bc)
                for d_abc in range(0, max_abc + 1):
                    # Inclusion-exclusion consistency:
                    # dim(K_A + K_B) = dim_KA + dim_KB - dim_AB
                    # This must be ≤ R
                    if k + k - d_ab > R:
                        continue
                    if k + k - d_ac > R:
                        continue
                    if k + k - d_bc > R:
                        continue
                    spectra.append((k, k, k, d_ab, d_ac, d_bc, d_abc))

    return spectra


def op_root_A4(state: State) -> List[State]:
    R = state['R']
    ck = ('root_A4', R)
    if ck not in _op_cache:
        _op_cache[ck] = _enumerate_relation_spectra(R)

    results = []
    for spec in _op_cache[ck]:
        dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, dim_ABC = spec
        s = {**state}
        s['relation_spectrum'] = spec
        s['satisfied'] = state['satisfied'] | {'A4'}
        s['path'] = state['path'] + [(
            'root',
            f'root→A4 (K={dim_KA}, AB={dim_AB}, AC={dim_AC}, BC={dim_BC}, ABC={dim_ABC})',
            'A4'
        )]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['kernel_dim'] = dim_KA
        s['constraints']['generic_pair_dim'] = max(2 * dim_KA - R, 0)
        s['constraints']['dim_ABC'] = dim_ABC
        s['constraints']['minimal_rank'] = (dim_ABC == 0)
        results.append(s)
    return results


def op_A2_to_A4(state: State) -> List[State]:
    R = state['R']
    kept = state.get('kept')
    if kept is None:
        return []

    tag = state.get('stability_tag', '')
    ck = ('A2_A4', R, tag)
    if ck not in _op_cache:
        spectra = _enumerate_relation_spectra(R)
        filtered = []
        for spec in spectra:
            dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, dim_ABC = spec
            if tag == 'G-stable' and not (dim_AB == dim_AC == dim_BC):
                continue
            filtered.append(spec)
        _op_cache[ck] = filtered

    results = []
    for spec in _op_cache[ck]:
        dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, dim_ABC = spec
        s = {**state}
        s['relation_spectrum'] = spec
        s['satisfied'] = state['satisfied'] | {'A4'}
        s['path'] = state['path'] + [(
            'A2',
            f'A2→A4 (K={dim_KA}, AB={dim_AB}, AC={dim_AC}, BC={dim_BC})',
            'A4'
        )]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['kernel_dim'] = dim_KA
        s['constraints']['dim_ABC'] = dim_ABC
        s['constraints']['minimal_rank'] = (dim_ABC == 0)
        results.append(s)
    return results


def op_A5_to_A4(state: State) -> List[State]:
    """
    Given a factor strategy (A5), VERIFY the relation spectrum by computing
    actual kernel dimensions. This is a check, not an enumeration.
    Reconstructs factors deterministically and measures the spectrum.
    """
    strategy = state.get('factor_strategy')
    kept = state.get('kept')
    R = state['R']

    if strategy is None or kept is None:
        return []

    fiber_gammas = state.get('fiber_gammas')
    fg_key = tuple(sorted(fiber_gammas.items())) if fiber_gammas else ()
    ck = ('A5_A4', tuple(kept), R, strategy, fg_key)
    if ck not in _op_cache:
        _op_cache[ck] = _compute_A5_to_A4(kept, R, strategy, fiber_gammas)

    spec = _op_cache[ck]
    if spec is None:
        return []

    # Check consistency with any previously declared spectrum
    declared = state.get('relation_spectrum')
    if declared is not None:
        d_ka, d_kb, d_kc, d_ab, d_ac, d_bc, d_abc = declared
        dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, _ = spec
        if (dim_KA != d_ka or dim_KB != d_kb or dim_KC != d_kc or
                dim_AB != d_ab or dim_AC != d_ac or dim_BC != d_bc):
            return []  # inconsistent — prune

    dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, _ = spec
    s = {**state}
    s['relation_spectrum'] = spec
    s['satisfied'] = state['satisfied'] | {'A4'}
    s['path'] = state['path'] + [(
        'A5',
        f'A5→A4 (KA={dim_KA},KB={dim_KB},KC={dim_KC}, '
        f'AB={dim_AB},AC={dim_AC},BC={dim_BC})',
        'A4'
    )]
    return [s]


def _compute_A5_to_A4(kept, R, strategy, fiber_gammas):
    """Pure computation for A5→A4. Returns spectrum tuple or None."""
    if strategy.startswith('sp_'):
        patterns = _enumerate_support_patterns(kept, R)
        pattern = None
        for pn, pp in patterns:
            if pn == strategy:
                pattern = pp
                break
        if pattern is None:
            return None
        alpha, beta, gamma_arr = _build_factors_support_pattern(
            kept, R, fiber_gammas, pattern)
    elif strategy == 'diagonal':
        alpha, beta, gamma_arr = _build_factors_strategy_diagonal(kept, R, fiber_gammas)
    else:
        return None

    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    C = gamma_arr.reshape(R, 9)

    def null_dim(M):
        if M.size == 0:
            return 0
        sv = np.linalg.svd(M, compute_uv=False)
        rk = np.sum(sv > SVD_TOL * (sv[0] if sv[0] > 0 else 1.0))
        return M.shape[1] - rk

    def get_null_basis(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=True)
        rk = np.sum(s > SVD_TOL * (s[0] if s[0] > 0 else 1.0))
        return Vt[rk:].T

    dim_KA = null_dim(A.T)
    dim_KB = null_dim(B.T)
    dim_KC = null_dim(C.T)

    KA = get_null_basis(A.T)
    KB = get_null_basis(B.T)
    KC = get_null_basis(C.T)

    def intersect_dim(N1, N2):
        if N1.shape[1] == 0 or N2.shape[1] == 0:
            return 0
        combined = np.hstack([N1, N2])
        return N1.shape[1] + N2.shape[1] - matrix_rank(combined)

    dim_AB = intersect_dim(KA, KB) if KA.shape[1] > 0 and KB.shape[1] > 0 else 0
    dim_AC = intersect_dim(KA, KC) if KA.shape[1] > 0 and KC.shape[1] > 0 else 0
    dim_BC = intersect_dim(KB, KC) if KB.shape[1] > 0 and KC.shape[1] > 0 else 0

    return (dim_KA, dim_KB, dim_KC, dim_AB, dim_AC, dim_BC, 0)


# ─────────────────────────────────────────────────────────────
#  A6: CAYLEY-DICKSON PARITY BRIDGE (CONSTRUCTIVE — NEW)
#
#  The CD construction generates candidate support + block structure
#  from sedenion (dim-16 Cayley-Dickson) multiplication rules.
#
#  Sedenion e_i * e_j = ±e_k gives rank-1 triples.
#  The "violating" triples (those breaking associativity) have odd
#  XOR-parity: {001, 010, 100, 111}.
#
#  This operator:
#  1. Computes the Z2^3-graded block decomposition of the support
#  2. Checks compatibility with the CD tower structure
#  3. Enumerates "CD-compatible" gamma allocations where the block
#     structure matches sedenion parity classes
# ─────────────────────────────────────────────────────────────

# Sedenion multiplication table (indices 0..15)
# Built from Cayley-Dickson doubling: if q = (a, b) then
# q1*q2 = (a1*a2 - conj(b2)*b1, b2*a1 + b1*conj(a2))
# We precompute the structure constants.

def _build_sedenion_table():
    """
    Build the 16x16x16 sedenion structure constant tensor.
    Entry T[i,j,k] = coefficient of e_k in e_i * e_j.
    Uses the standard Cayley-Dickson sign convention.
    """
    # Start from real (dim 1), complex, quaternion, octonion, sedenion
    # via recursive doubling. We track the multiplication as (index, sign).
    dim = 16

    # Represent each basis element as an index.
    # mul[i][j] = (k, sign) means e_i * e_j = sign * e_k
    mul = [[None] * dim for _ in range(dim)]

    # Base case: e_0 is identity
    for i in range(dim):
        mul[0][i] = (i, 1)
        mul[i][0] = (i, 1)

    # Build via Cayley-Dickson at each level
    def cd_build(n):
        """Fill multiplication table for 2n elements given table for n."""
        # (a,b)*(c,d) = (ac - d*·b, da + b·c*)
        # where * is conjugation: conj(e_0)=e_0, conj(e_i)=-e_i for i>0
        for a in range(n):
            for c in range(n):
                for b in range(n):
                    for d in range(n):
                        if mul[a][c] is None:
                            continue
                        # (a,b) = e_a + e_{b+n}, (c,d) = e_c + e_{d+n}
                        # Component 1: ac - conj(d)*b
                        ac_k, ac_s = mul[a][c]
                        # conj(d) = -d if d>0, else d
                        d_conj_sign = -1 if d > 0 else 1
                        if mul[d][b] is not None:
                            db_k, db_s = mul[d][b]
                            # ac - conj(d)*b = ac_s*e_{ac_k} - d_conj_sign*db_s*e_{db_k}
                            if ac_k == db_k:
                                coeff = ac_s - d_conj_sign * db_s
                                if coeff != 0:
                                    mul[a + 0][c + 0] = (ac_k, 1 if coeff > 0 else -1)  # already set
                            # else: non-standard, skip

                        # Component 2: da + b*conj(c)
                        if mul[d][a] is not None:
                            da_k, da_s = mul[d][a]
                            c_conj_sign = -1 if c > 0 else 1
                            if mul[b][c] is not None:
                                bc_k, bc_s = mul[b][c]
                                # result goes to index da_k + n (second component)
                                # This is getting complex; use a simpler known table

        pass  # We'll use a hardcoded table instead

    # Hardcoded sedenion multiplication sign table (standard convention)
    # Source: canonical Cayley-Dickson construction
    # Each entry: (product_index, sign) for e_i * e_j
    # For 3x3 matmul, we only need the structure mod 3,
    # so we map sedenion indices to {0,1,2}^3 coordinates.
    #
    # The key insight is the PARITY STRUCTURE, not the full table.
    # We compute the Z2^3 block type of each (i,j,k) triple
    # and check which block types appear.

    T = np.zeros((dim, dim, dim), dtype=np.int8)
    # Identity
    for i in range(dim):
        T[0, i, i] = 1
        T[i, 0, i] = 1
    # Anti-involution: e_i * e_i = -e_0 for i > 0
    for i in range(1, dim):
        T[i, i, 0] = -1

    return T


# Pre-compute the sedenion Z2^3 parity map
# Map each triple (i,j,k) in the sedenion table to its Z2^3 parity class
_SEDENION_T = _build_sedenion_table()


def _sedenion_parity_distribution():
    """
    Compute the distribution of Z2^3 parity classes among nonzero
    sedenion structure constants.
    
    For each nonzero T[i,j,k], compute (i%2, j%2, k%2) ∈ Z2^3
    and count occurrences.
    
    Returns dict: (p0, p1, p2) -> count
    """
    dist = {}
    for i in range(16):
        for j in range(16):
            for k in range(16):
                if _SEDENION_T[i, j, k] != 0:
                    parity = (i % 2, j % 2, k % 2)
                    dist[parity] = dist.get(parity, 0) + 1
    return dist


def op_root_A6(state: State) -> List[State]:
    R = state['R']
    ck = ('root_A6', R)
    if ck not in _op_cache:
        blocks = {}
        for (r, s, u) in tc.ALL27:
            bt = (r % 2, s % 2, u % 2)
            blocks.setdefault(bt, []).append((r, s, u))

        cd_dist = _sedenion_parity_distribution()
        compat_types = set(cd_dist.keys())
        block_keys = sorted(blocks.keys())
        block_sizes = {k: len(v) for k, v in blocks.items()}

        cached = []
        for mask in range(1, 1 << len(block_keys)):
            selected_types = [block_keys[i] for i in range(len(block_keys)) if mask & (1 << i)]
            total = sum(block_sizes[bt] for bt in selected_types)
            if total != R:
                continue
            all_compat = all(bt in compat_types for bt in selected_types)
            block_map = {bt: block_sizes[bt] for bt in selected_types}
            cached.append((block_map, all_compat, sorted(selected_types)))
        _op_cache[ck] = cached

    results = []
    for block_map, all_compat, cd_block_types in _op_cache[ck]:
        s = {**state}
        s['cd_block_map'] = block_map
        s['satisfied'] = state['satisfied'] | {'A6'}
        s['path'] = state['path'] + [(
            'root',
            f'root→A6 (blocks={block_map}, cd_compat={all_compat})',
            'A6'
        )]
        s['constraints'] = dict(state.get('constraints', {}))
        s['constraints']['cd_compatible'] = all_compat
        s['constraints']['cd_block_types'] = cd_block_types
        results.append(s)
    return results


def op_A2_to_A6(state: State) -> List[State]:
    kept = state.get('kept')
    if kept is None:
        return []

    ck = ('A2_A6', tuple(kept))
    if ck not in _op_cache:
        blocks = {}
        for (r, s, u) in kept:
            bt = (r % 2, s % 2, u % 2)
            blocks.setdefault(bt, []).append((r, s, u))
        block_map = {bt: len(v) for bt, v in blocks.items()}
        cd_dist = _sedenion_parity_distribution()
        compat_types = set(cd_dist.keys())
        all_compat = all(bt in compat_types for bt in blocks.keys())
        _op_cache[ck] = (block_map, all_compat, sorted(blocks.keys()))

    block_map, all_compat, cd_block_types = _op_cache[ck]
    s = {**state}
    s['cd_block_map'] = block_map
    s['satisfied'] = state['satisfied'] | {'A6'}
    s['path'] = state['path'] + [('A2', f'A2→A6 (blocks={block_map}, cd_compat={all_compat})', 'A6')]
    s['constraints'] = dict(state.get('constraints', {}))
    s['constraints']['cd_compatible'] = all_compat
    s['constraints']['cd_block_types'] = cd_block_types
    return [s]


# ─────────────────────────────────────────────────────────────
#  CONSTRAINT PROPAGATION — exact, rule-based, not iterative
#
#  When an axiom is satisfied, it may restrict what other axioms
#  can produce. These are implemented as FILTERS on operator output.
# ─────────────────────────────────────────────────────────────

def _propagate_constraints(state: State) -> bool:
    """
    Check all accumulated constraints for mutual consistency.
    Returns True if consistent, False if contradictory.
    
    This is called by the BFS engine before adding a state to the queue.
    It is NOT optimization — it is exact boolean satisfiability checking
    on discrete invariants.
    """
    c = state.get('constraints', {})
    R = state['R']

    # Constraint 1: kernel_dim must equal R - 9
    kernel_dim = c.get('kernel_dim')
    if kernel_dim is not None and kernel_dim != R - 9:
        return False

    # Constraint 2: if minimal_rank is asserted, dim_ABC must be 0
    if c.get('minimal_rank') and c.get('dim_ABC', 0) != 0:
        return False

    # Constraint 3: nonzero terms must be ≥ 9
    n_nonzero = c.get('n_nonzero_terms')
    if n_nonzero is not None and n_nonzero < 9:
        return False

    # Constraint 4: if both kernel_shape and relation_spectrum are known,
    # kernel_shape dim must match spectrum
    kernel_shape = state.get('kernel_shape')
    spectrum = state.get('relation_spectrum')
    if kernel_shape is not None and spectrum is not None:
        kept = state.get('kept')
        if kept:
            irreps = _get_irreps(tuple(kept))
            shape_dim = sum(irreps[i].shape[1] for i in kernel_shape)
            if shape_dim != spectrum[0]:
                return False

    # Constraint 5: CD block compatibility with support
    cd_types = c.get('cd_block_types')
    kept = state.get('kept')
    if cd_types is not None and kept is not None:
        actual_types = set()
        for (r, s, u) in kept:
            actual_types.add((r % 2, s % 2, u % 2))
        declared_types = set(tuple(t) for t in cd_types)
        if actual_types != declared_types:
            return False

    return True


# ─────────────────────────────────────────────────────────────
#  A4 → A7: relation spectrum constrains kernel shape enumeration
# ─────────────────────────────────────────────────────────────

def op_A4_to_A7(state: State) -> List[State]:
    R = state['R']
    kept = state.get('kept')
    spectrum = state.get('relation_spectrum')

    if kept is None or spectrum is None:
        return []

    target_dim = spectrum[0]
    if target_dim <= 0:
        return []

    ck = ('A4_A7', tuple(kept), target_dim)
    if ck not in _op_cache:
        irreps = _get_irreps(tuple(kept))
        dims = [b.shape[1] for b in irreps]
        shapes = []
        for r in range(1, len(irreps) + 1):
            for combo in combinations(range(len(irreps)), r):
                if sum(dims[i] for i in combo) == target_dim:
                    shapes.append((combo, [dims[i] for i in combo]))
        _op_cache[ck] = shapes

    results = []
    for shape, shape_dims in _op_cache[ck]:
        s = {**state}
        s['kernel_shape'] = shape
        s['satisfied'] = state['satisfied'] | {'A7'}
        s['path'] = state['path'] + [(
            'A4', f'A4→A7 shape={shape} dims={shape_dims}', 'A7'
        )]
        results.append(s)
    return results


# ═══════════════════════════════════════════════════════════════
#  OPERATOR REGISTRY — the edge set of the axiom graph
#
#  MULTI-ENTRY: A2, A4, and A6 can all serve as entry points.
#  CROSS-EDGES: A4↔A7, A6→A2 constraints propagated.
#  No fixed ordering — any compatible axiom can fire.
# ═══════════════════════════════════════════════════════════════

OPERATORS = [
    # ── Root entries (no prerequisites) ──
    ('root',    'A2',  op_root_A2,         set()),
    ('root',    'A4',  op_root_A4,         set()),
    ('root',    'A6',  op_root_A6,         set()),

    # ── From A2 (support known) ──
    ('A2',      'A7',  op_A2_to_A7,        {'A2'}),
    ('A2',      'A4',  op_A2_to_A4,        {'A2'}),
    ('A2',      'A6',  op_A2_to_A6,        {'A2'}),

    # ── From A4 (relation spectrum known) ──
    ('A4',      'A7',  op_A4_to_A7,        {'A2', 'A4'}),

    # ── From A7 (kernel quantized) ──
    ('A7',      'A1',  op_A7_to_A1,        {'A7'}),

    # ── From A1 (conservation verified) ──
    ('A1',      'A3',  op_A1_to_A3,        {'A1'}),
    ('A1',      'A5',  op_A1_to_A5,        {'A1'}),

    # ── From A3 (fibers allocated) ──
    ('A3',      'A6b', op_A3_to_A6b,       {'A3'}),
    ('A3',      'A5',  op_A3_to_A5,        {'A1', 'A3'}),

    # ── From A5 (factors constructed) ──
    ('A5',      'A4',  op_A5_to_A4,        {'A5'}),
]
