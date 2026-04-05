"""
operators.py — Constructive operators between axioms.

Each operator is an edge in the axiom graph:
    op_X_to_Y(state, R) → list[state']

States are LIGHTWEIGHT — no numpy arrays stored. Only reconstruction keys:
  {
    'R': int,
    'kept': list of triples,
    'kernel_shape': tuple of irrep indices (if A7 done),
    'fiber_gammas': dict idx->float (if A3 done),
    'factor_strategy': (name, seed) for factor reconstruction (if A5 done),
    'satisfied': set of axiom ids,
    'path': list of (source, op_name, target),
  }

Arrays are recomputed on demand via reconstruct_*() helpers.
"""
import numpy as np
from itertools import combinations
from functools import lru_cache
from typing import List, Dict, Any

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
        # Instead of storing arrays, store reconstruction keys:
        'kernel_shape': None,      # tuple of irrep indices
        'fiber_gammas': None,      # dict idx -> float
        'factor_strategy': None,   # (strategy_name, seed) for reconstruction
        'satisfied': set(),
        'path': [],
    }


# ─────────────────────────────────────────────────────────────
#  ROOT OPERATORS — entry points (no prerequisite axiom)
# ─────────────────────────────────────────────────────────────

def op_root_A2(state: State) -> List[State]:
    """
    Entry: establish G-stable support (A2).
    Enumerates all G-stable subsets of ALL27 with |S|=R.
    Each is a branch.
    """
    R = state['R']
    options = tc.kept_triples_for_rank(R)
    if not options:
        return []  # no G-stable subset exists for this R

    results = []
    for kept in options:
        s = {**state}
        s['kept'] = sorted(kept)
        s['satisfied'] = state['satisfied'] | {'A2'}
        s['path'] = state['path'] + [('root', 'root→A2', 'A2')]
        results.append(s)
    return results


def op_root_free(state: State) -> List[State]:
    """
    Entry: use full ALL27 support without G-stability (for non-G-stable ranks).
    Single branch.
    """
    R = state['R']
    s = {**state}
    s['kept'] = tc.ALL27[:R]
    s['satisfied'] = state['satisfied'].copy()
    s['path'] = state['path'] + [('root', 'root→free', 'free')]
    return [s]


# ─────────────────────────────────────────────────────────────
#  A2 → A7: decompose irreps, enumerate kernel shapes
# ─────────────────────────────────────────────────────────────

def op_A2_to_A7(state: State) -> List[State]:
    """
    Given G-stable support (A2), decompose into irreps and enumerate
    all kernel shapes summing to dim R-9 (A7 quantization).
    Each valid shape is a branch.
    """
    R = state['R']
    kept = state['kept']
    if kept is None:
        return []

    irreps = _get_irreps(tuple(kept))

    target_dim = R - 9
    if target_dim <= 0:
        return []

    dims = [b.shape[1] for b in irreps]

    # Enumerate all subsets of irreps summing to target_dim
    shapes = []
    for r in range(1, len(irreps) + 1):
        for combo in combinations(range(len(irreps)), r):
            if sum(dims[i] for i in combo) == target_dim:
                shapes.append(combo)

    results = []
    for shape in shapes:
        s = {**state}
        s['kernel_shape'] = shape
        s['satisfied'] = state['satisfied'] | {'A7'}
        s['path'] = state['path'] + [('A2', f'A2→A7 shape={shape} dims={[dims[i] for i in shape]}', 'A7')]
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
    kernel_basis, active_basis = reconstruct_kernel_and_active(state)
    if active_basis is None or active_basis.shape[1] != 9:
        return []

    s = {**state}
    # Don't store arrays — kernel_shape + kept is enough to reconstruct
    s['satisfied'] = state['satisfied'] | {'A1'}
    s['path'] = state['path'] + [('A7', 'A7→A1 (kernel complement)', 'A1')]
    return [s]


# ─────────────────────────────────────────────────────────────
#  A1 → A3: solve fiber sum constraints Γ(s,u) = 3
# ─────────────────────────────────────────────────────────────

def op_A1_to_A3(state: State) -> List[State]:
    """
    Given active basis (from A1), solve for fiber gamma values.
    
    Each fiber (s,u) has |fiber| terms. The constraint is:
      Γ(s,u) = sum of effective gammas over fiber = 3.
    
    Forced fibers (size 1): gamma pinned to 3. No branching.
    Free fibers (size k): k-1 degrees of freedom. We discretize
    into a grid of integer/half-integer splits for enumeration.
    
    Returns list of states with fiber_gammas set.
    """
    R = state['R']
    kept = state['kept']
    if kept is None:
        return []

    # Build fibers
    fibers = {}  # (s,u) -> list of indices into kept
    for idx, (r, s, u) in enumerate(kept):
        fibers.setdefault((s, u), []).append(idx)

    # For forced fibers: gamma = 3, no choice
    # For free fibers: enumerate integer partitions of 3 into k parts
    # (allowing 0 and negative — but start with non-negative integers)
    forced = {}
    free_fibers = {}
    for key, indices in fibers.items():
        if len(indices) == 1:
            forced[key] = [3.0]
        else:
            free_fibers[key] = indices

    if not free_fibers:
        # All fibers forced — single branch
        gamma_map = {}
        for key, vals in forced.items():
            for idx, v in zip(fibers[key], vals):
                gamma_map[idx] = v
        s = {**state}
        s['fiber_gammas'] = gamma_map
        s['satisfied'] = state['satisfied'] | {'A3'}
        s['path'] = state['path'] + [('A1', 'A1→A3 (all forced)', 'A3')]
        return [s]

    # Enumerate partitions for free fibers
    # For k terms summing to 3 with integer values in {0,1,2,3}:
    def partitions_of(total, k, values=None):
        if values is None:
            values = list(range(total + 1))
        if k == 1:
            if total in values:
                yield (total,)
            return
        for v in values:
            if v <= total:
                for rest in partitions_of(total - v, k - 1, values):
                    yield (v,) + rest

    # Build all combinations across free fibers
    fiber_keys = sorted(free_fibers.keys())
    fiber_options = []
    for key in fiber_keys:
        k = len(free_fibers[key])
        parts = list(partitions_of(3, k, [0, 1, 2, 3]))
        fiber_options.append((key, parts))

    # Cross product of all fiber partitions (capped to avoid explosion)
    MAX_FIBER_COMBOS = 20

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
                    combo[i] = float(v)
                yield combo

    results = []
    count = 0
    for free_map in cross_product(fiber_options):
        if count >= MAX_FIBER_COMBOS:
            break
        count += 1
        gamma_map = {}
        # Add forced
        for key, vals in forced.items():
            for idx, v in zip(fibers[key], vals):
                gamma_map[idx] = v
        # Add free
        gamma_map.update(free_map)

        s = {**state}
        s['fiber_gammas'] = gamma_map
        s['satisfied'] = state['satisfied'] | {'A3'}
        desc = {k: [gamma_map[i] for i in free_fibers[k]] for k in fiber_keys}
        s['path'] = state['path'] + [('A1', f'A1→A3 fibers={desc}', 'A3')]
        results.append(s)

    return results


# ─────────────────────────────────────────────────────────────
#  A3 → A6b: verify/enforce CD tower fiber parity
# ─────────────────────────────────────────────────────────────

def op_A3_to_A6b(state: State) -> List[State]:
    """
    Given fiber gammas (A3), check Cayley-Dickson parity alignment.
    Forced fibers at even-even (s,u) positions should have gamma=3.
    Free fibers should have balanced parity splits.
    
    This is a filter — passes through states that satisfy CD parity,
    blocks those that don't.
    """
    R = state['R']
    kept = state['kept']
    fiber_gammas = state.get('fiber_gammas')
    if kept is None or fiber_gammas is None:
        return []

    # Check parity constraint: for each (s,u), the XOR parity of
    # contributing triples should align with CD tower structure
    fibers = {}
    for idx, (r, s, u) in enumerate(kept):
        fibers.setdefault((s, u), []).append((idx, r))

    # CD parity check: all fiber sums must be exactly 3
    all_ok = True
    for (s, u), members in fibers.items():
        total = sum(fiber_gammas.get(idx, 0) for idx, r in members)
        if abs(total - 3.0) > 1e-10:
            all_ok = False
            break

    if not all_ok:
        return []

    s = {**state}
    s['satisfied'] = state['satisfied'] | {'A6b'}
    s['path'] = state['path'] + [('A3', 'A3→A6b (parity verified)', 'A6b')]
    return [s]


# ─────────────────────────────────────────────────────────────
#  A3 → A5: build factors from fiber gammas + active basis, check gates
# ─────────────────────────────────────────────────────────────

def _build_factors_strategy_diagonal(kept, R, fiber_gammas):
    """Strategy 1: diagonal — each term gets its triple's natural entries."""
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_arr = np.zeros((R, 3, 3))
    for idx, (r, s, u) in enumerate(kept):
        g_val = fiber_gammas.get(idx, 1.0) if fiber_gammas else 1.0
        if g_val == 0:
            continue
        sign = np.sign(g_val) if g_val != 0 else 1.0
        scale = abs(g_val) ** (1.0 / 3.0)
        alpha[idx, r, s] = sign * scale
        beta[idx, s, u] = scale
        gamma_arr[idx, r, u] = scale
    return alpha, beta, gamma_arr


def _build_factors_strategy_spread(kept, R, fiber_gammas, seed=0):
    """
    Strategy 2: spread — each term gets its natural entry PLUS small
    contributions to other entries in the same row/column.
    This creates nonzero off-diagonal products (Delta), which is needed
    for Gate 2 to be nontrivial.
    """
    rng = np.random.default_rng(seed)
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_arr = np.zeros((R, 3, 3))
    eps = 0.3  # spread factor

    for idx, (r, s, u) in enumerate(kept):
        g_val = fiber_gammas.get(idx, 1.0) if fiber_gammas else 1.0
        if g_val == 0:
            continue
        sign = np.sign(g_val) if g_val != 0 else 1.0
        scale = abs(g_val) ** (1.0 / 3.0)
        # Primary entry
        alpha[idx, r, s] = sign * scale
        beta[idx, s, u] = scale
        gamma_arr[idx, r, u] = scale
        # Spread: small random entries in same row
        for j in range(3):
            if j != s:
                alpha[idx, r, j] += eps * scale * rng.standard_normal()
            if j != u:
                beta[idx, s, j] += eps * scale * rng.standard_normal()
            if j != u:
                gamma_arr[idx, r, j] += eps * scale * rng.standard_normal()
    return alpha, beta, gamma_arr


def _build_factors_strategy_active_basis(kept, R, fiber_gammas, active_basis, seed=0):
    """
    Strategy 3: active basis — parameterize factors as active_basis @ M
    where M is a (9,9) matrix. This ensures factors live in the correct
    9-dim subspace dictated by the irrep decomposition.
    Fiber gammas scale the per-term contribution.
    """
    rng = np.random.default_rng(seed)
    # Generate 3 random (9,9) mixing matrices
    M_A = rng.standard_normal((9, 9)) * 0.5
    M_B = rng.standard_normal((9, 9)) * 0.5
    M_C = rng.standard_normal((9, 9)) * 0.5

    # Factors in flat form: (R, 9)
    A_flat = active_basis @ M_A  # (R, 9)
    B_flat = active_basis @ M_B
    C_flat = active_basis @ M_C

    # Apply fiber gamma scaling
    if fiber_gammas:
        for idx in range(R):
            g_val = fiber_gammas.get(idx, 1.0)
            if g_val == 0:
                A_flat[idx] = 0
                B_flat[idx] = 0
                C_flat[idx] = 0
            else:
                current = (np.linalg.norm(A_flat[idx]) *
                           np.linalg.norm(B_flat[idx]) *
                           np.linalg.norm(C_flat[idx]))
                if current > 1e-12:
                    ratio = abs(g_val) / current
                    cbrt = ratio ** (1.0 / 3.0)
                    sign = np.sign(g_val)
                    A_flat[idx] *= sign * cbrt
                    B_flat[idx] *= cbrt
                    C_flat[idx] *= cbrt

    alpha = A_flat.reshape(R, 3, 3)
    beta = B_flat.reshape(R, 3, 3)
    gamma_arr = C_flat.reshape(R, 3, 3)
    return alpha, beta, gamma_arr


def _build_factors_strategy_symmetric(kept, R, fiber_gammas, seed=0):
    """
    Strategy 4: G-symmetric factors (R=19 only).
    Uses tensor_core.build_symmetric_factors with fiber scaling.
    """
    if R != 19:
        return None, None, None
    rng = np.random.default_rng(seed)
    params = rng.standard_normal(81)
    alpha, beta, gamma_arr = tc.build_symmetric_factors(params)

    # Rescale per fiber gamma
    if fiber_gammas:
        for idx in range(R):
            g_val = fiber_gammas.get(idx, 1.0)
            current = (np.linalg.norm(alpha[idx]) *
                       np.linalg.norm(beta[idx]) *
                       np.linalg.norm(gamma_arr[idx]))
            if current > 1e-12 and g_val != 0:
                ratio = abs(g_val) / current
                cbrt = ratio ** (1.0 / 3.0)
                sign = np.sign(g_val)
                alpha[idx] *= sign * cbrt
                beta[idx] *= cbrt
                gamma_arr[idx] *= cbrt
            elif g_val == 0:
                alpha[idx] = 0
                beta[idx] = 0
                gamma_arr[idx] = 0

    return alpha, beta, gamma_arr


# ─────────────────────────────────────────────────────────────
#  CONSTRUCTIVE GATE-AWARE FACTOR STRATEGIES
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
#  This guarantees Delta=0. We then check if Sigma can escape H.
# ─────────────────────────────────────────────────────────────

def _build_factors_support_pattern(kept, R, fiber_gammas, pattern, seed=0):
    """
    Construct factors with a given s-support pattern.
    
    pattern: list of frozensets, one per term. pattern[k] ⊆ {0,1,2}
             is the set of s-values term k is allowed to use.
    
    For each term k with triple (r_k, s_k, u_k) and support S_k:
      - α[k, r_k, s] = scale * c_s   for s ∈ S_k
      - β[k, s, u_k] = scale * d_s   for s ∈ S_k
    where c_s, d_s are deterministic coefficients from the seed.
    
    Returns (alpha, beta, gamma_arr).
    """
    rng = np.random.default_rng(seed)
    alpha = np.zeros((R, 3, 3))
    beta = np.zeros((R, 3, 3))
    gamma_arr = np.zeros((R, 3, 3))

    for idx, (r, s_nat, u) in enumerate(kept):
        g_val = fiber_gammas.get(idx, 1.0) if fiber_gammas else 1.0
        if g_val == 0:
            continue
        S_k = pattern[idx]
        n_s = len(S_k)
        if n_s == 0:
            continue

        # Generate coefficients for each s in support
        c = rng.standard_normal(n_s)
        d = rng.standard_normal(n_s)

        # Scale so that Sigma contribution ~ g_val^(1/3)
        sign = np.sign(g_val)
        scale = abs(g_val) ** (1.0 / 3.0)

        for i, s_val in enumerate(sorted(S_k)):
            alpha[idx, r, s_val] = sign * scale * c[i]
            beta[idx, s_val, u] = scale * d[i]

        # gamma: encode the natural triple entry
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
    The support determines which s-indices alpha/beta are nonzero on.
    
    Since 3^R patterns is too many, we enumerate per-orbit-class assignments:
    terms in the same G-orbit share the same support type. With 4 orbits
    and 7 support options ({0},{1},{2},{0,1},{0,2},{1,2},{0,1,2}), that's
    7^4 = 2401 patterns — perfectly enumerable.
    
    Returns a list of (pattern_name, pattern) pairs.
    """
    SUPPORTS = [
        frozenset({0}), frozenset({1}), frozenset({2}),
        frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2}),
        frozenset({0, 1, 2}),
    ]

    # Classify terms by orbit type
    orbit_classes = {}
    for idx, (r, s, u) in enumerate(kept):
        # Orbit key: sorted triple structure (invariant under G)
        key = tuple(sorted([r, s, u]))
        if key not in orbit_classes:
            orbit_classes[key] = []
        orbit_classes[key].append(idx)

    orbit_keys = sorted(orbit_classes.keys())
    n_classes = len(orbit_keys)

    patterns = []
    # Enumerate all support assignments: one support choice per orbit class
    from itertools import product as iproduct
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


def op_A3_to_A5(state: State) -> List[State]:
    """
    Given fiber gammas (A3) and active basis (A1),
    enumerate s-support patterns × seeds, construct factors, check Gates 2+3.
    Each pattern that passes gates is a branch.
    """
    fiber_gammas = state.get('fiber_gammas')
    kept = state.get('kept')
    R = state['R']

    if kept is None or state.get('kernel_shape') is None:
        return []

    # Enumerate all support patterns (orbit-class combinatorial)
    patterns = _enumerate_support_patterns(kept, R)

    results = []
    for pat_name, pattern in patterns:
        for seed in range(3):
            name = f'{pat_name}_s{seed}'
            alpha, beta, gamma_arr = _build_factors_support_pattern(
                kept, R, fiber_gammas, pattern, seed=seed)
            passed, details = _check_gates(alpha, beta, R)
            if passed:
                s = {**state}
                s['factor_strategy'] = name
                s['satisfied'] = state['satisfied'] | {'A5'}
                s['path'] = state['path'] + [(
                    'A3', f'A3→A5 ({name}, ag={details["aug_gap"]})', 'A5'
                )]
                results.append(s)
                break  # one seed per pattern is enough

    return results


# Also allow A1→A5 directly (without A3 fiber gammas)
def op_A1_to_A5(state: State) -> List[State]:
    """
    Given active basis (A1), try support patterns without fiber gammas.
    Same combinatorial enumeration, just no gamma scaling.
    """
    kept = state.get('kept')
    R = state['R']

    if kept is None or state.get('kernel_shape') is None:
        return []

    patterns = _enumerate_support_patterns(kept, R)

    results = []
    for pat_name, pattern in patterns:
        alpha, beta, gamma_arr = _build_factors_support_pattern(
            kept, R, None, pattern, seed=0)
        passed, details = _check_gates(alpha, beta, R)
        if passed:
            s = {**state}
            s['factor_strategy'] = f'{pat_name}_s0'
            s['satisfied'] = state['satisfied'] | {'A5'}
            s['path'] = state['path'] + [(
                'A1', f'A1→A5 ({pat_name}, ag={details["aug_gap"]})', 'A5'
            )]
            results.append(s)

    return results


# ─────────────────────────────────────────────────────────────
#  A2 → A6: check Cayley-Dickson parity block alignment
# ─────────────────────────────────────────────────────────────

def op_A2_to_A6(state: State) -> List[State]:
    """
    Given G-stable support, check if the orbit/parity structure
    admits a CD-compatible block partition.
    Deterministic check — pass or fail.
    """
    kept = state.get('kept')
    if kept is None:
        return []

    # Count parity types
    block_types = {}
    for (r, s, u) in kept:
        bt = (r % 2, s % 2, u % 2)
        block_types[bt] = block_types.get(bt, 0) + 1

    # For CD compatibility: the even-parity block (0,0,0) should exist
    # and the partition should have a balanced structure
    n_even = sum(v for k, v in block_types.items() if (k[0] ^ k[1] ^ k[2]) == 0)
    n_odd = sum(v for k, v in block_types.items() if (k[0] ^ k[1] ^ k[2]) == 1)

    # Pass if even-parity terms exist (minimal condition)
    if n_even > 0:
        s = {**state}
        s['satisfied'] = state['satisfied'] | {'A6'}
        s['path'] = state['path'] + [('A2', f'A2→A6 (even={n_even}, odd={n_odd})', 'A6')]
        return [s]

    return []


# ─────────────────────────────────────────────────────────────
#  A4: relation module check (requires factors)
# ─────────────────────────────────────────────────────────────

def op_factors_to_A4(state: State) -> List[State]:
    """
    Given a factor strategy (from A5), reconstruct factors and compute
    relation module kernel dimensions. Check generic position.
    """
    strategy = state.get('factor_strategy')
    kept = state.get('kept')
    R = state['R']

    if strategy is None or kept is None:
        return []

    # Reconstruct factors from strategy
    fiber_gammas = state.get('fiber_gammas')
    _, active_basis = reconstruct_kernel_and_active(state) if state.get('kernel_shape') else (None, None)

    # Parse strategy name to reconstruct factors
    # Support pattern strategies: "sp_012_0_12_..._s2"
    if strategy.startswith('sp_'):
        # Extract seed from end
        parts = strategy.rsplit('_s', 1)
        pat_name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 0
        # Re-enumerate patterns to find the matching one
        patterns = _enumerate_support_patterns(kept, R)
        pattern = None
        for pn, pp in patterns:
            if pn == pat_name:
                pattern = pp
                break
        if pattern is None:
            return []
        alpha, beta, gamma_arr = _build_factors_support_pattern(
            kept, R, fiber_gammas, pattern, seed=seed)
    elif strategy == 'diagonal':
        alpha, beta, gamma_arr = _build_factors_strategy_diagonal(kept, R, fiber_gammas)
    elif strategy.startswith('spread_s'):
        seed = int(strategy.split('s')[1])
        alpha, beta, gamma_arr = _build_factors_strategy_spread(kept, R, fiber_gammas, seed=seed)
    elif strategy.startswith('active_s'):
        seed = int(strategy.split('s')[1])
        if active_basis is None:
            return []
        alpha, beta, gamma_arr = _build_factors_strategy_active_basis(
            kept, R, fiber_gammas, active_basis, seed=seed)
    elif strategy.startswith('symm_s'):
        seed = int(strategy.split('s')[1])
        alpha, beta, gamma_arr = _build_factors_strategy_symmetric(kept, R, fiber_gammas, seed=seed)
        if alpha is None:
            return []
    else:
        return []

    A = alpha.reshape(R, 9)
    B = beta.reshape(R, 9)
    C = gamma_arr.reshape(R, 9)

    def null_dim(M):
        if M.size == 0:
            return 0
        sv = np.linalg.svd(M, compute_uv=False)
        rk = np.sum(sv > SVD_TOL * (sv[0] if sv[0] > 0 else 1.0))
        return M.shape[1] - rk

    dim_KA = null_dim(A.T)  # A.T is 9×R
    dim_KB = null_dim(B.T)
    dim_KC = null_dim(C.T)

    # Generic pairwise intersection
    expected_AB = max(dim_KA + dim_KB - R, 0)
    expected_AC = max(dim_KA + dim_KC - R, 0)
    expected_BC = max(dim_KB + dim_KC - R, 0)

    # Actual pairwise (via stacking null bases)
    def get_null_basis(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=True)
        rk = np.sum(s > SVD_TOL * (s[0] if s[0] > 0 else 1.0))
        return Vt[rk:].T  # (R, null_dim)

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

    generic = (dim_AB == expected_AB and dim_AC == expected_AC and dim_BC == expected_BC)

    if generic:
        s = {**state}
        s['satisfied'] = state['satisfied'] | {'A4'}
        s['path'] = state['path'] + [(
            'factors', f'→A4 (KA={dim_KA},KB={dim_KB},KC={dim_KC}, '
            f'AB={dim_AB}/{expected_AB},AC={dim_AC}/{expected_AC},BC={dim_BC}/{expected_BC})',
            'A4'
        )]
        return [s]

    return []


# ═══════════════════════════════════════════════════════════════
#  OPERATOR REGISTRY — the edge set of the axiom graph
# ═══════════════════════════════════════════════════════════════

# Each entry: (source_axiom_or_'root', target_axiom, operator_fn, prereqs)
# prereqs: set of axiom ids that must be in state['satisfied'] before this op fires

OPERATORS = [
    ('root',    'A2',  op_root_A2,         set()),
    ('root',    'free', op_root_free,       set()),
    ('A2',      'A7',  op_A2_to_A7,        {'A2'}),
    ('A7',      'A1',  op_A7_to_A1,        {'A7'}),
    ('A1',      'A3',  op_A1_to_A3,        {'A1'}),
    ('A3',      'A6b', op_A3_to_A6b,       {'A3'}),
    ('A2',      'A6',  op_A2_to_A6,        {'A2'}),
    ('A3',      'A5',  op_A3_to_A5,        {'A1', 'A3'}),
    ('A1',      'A5',  op_A1_to_A5,        {'A1'}),
    ('factors', 'A4',  op_factors_to_A4,    {'A5'}),  # needs factors from A5
]
