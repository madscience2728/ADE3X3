import math
import numpy as np

from tensor_core import build_T_matmul, build_G_invariant_E, G_ORBITS, frobenius

# ---------------------------------------------------------------------------
# Explicit orbit structure for non-trivial basis construction
# ---------------------------------------------------------------------------

# O1: 6 terms, 3 pairs related by swap12 on the single nonzero slot
O1_PAIRS = [
    ((0, 0, 1), (0, 0, 2)),  # nonzero in position 2
    ((0, 1, 0), (0, 2, 0)),  # nonzero in position 1
    ((1, 0, 0), (2, 0, 0)),  # nonzero in position 0
]

# O2: 12 terms, 6 pairs related by simultaneous swap12 of all nonzero slots
O2_PAIRS = [
    ((0, 1, 2), (0, 2, 1)),  # nonzero in positions 1,2 — values differ
    ((0, 1, 1), (0, 2, 2)),  # nonzero in positions 1,2 — values equal
    ((1, 0, 2), (2, 0, 1)),  # nonzero in positions 0,2 — values differ
    ((1, 0, 1), (2, 0, 2)),  # nonzero in positions 0,2 — values equal
    ((1, 2, 0), (2, 1, 0)),  # nonzero in positions 0,1 — values differ
    ((1, 1, 0), (2, 2, 0)),  # nonzero in positions 0,1 — values equal
]

# O3: 8 interior terms (r,s,u all in {1,2}), ordered as (a,b,c) = (r-1,s-1,u-1)
O3_TERMS = [(r, s, u) for r in (1, 2) for s in (1, 2) for u in (1, 2)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _T_from_entries(entries):
    """Build (9,9,9) tensor from {(r,s,u): value} dict."""
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for (r, s, u), val in entries.items():
        T[3*r + s, 3*s + u, 3*r + u] = val
    return T


def _o2_sign(r, s, u):
    """
    Sign for the O2 non-trivial character.
    +1 if the nonzero coord values are in strict ascending order,
    -1 if strict descending, 0 if equal.
    'First nonzero' = value at the earlier (smaller) nonzero position.
    """
    if r == 0:     # nonzero positions 1, 2 → compare s and u
        if s < u: return 1.0
        if s > u: return -1.0
        return 0.0
    elif s == 0:   # nonzero positions 0, 2 → compare r and u
        if r < u: return 1.0
        if r > u: return -1.0
        return 0.0
    else:          # u == 0: nonzero positions 0, 1 → compare r and s
        if r < s: return 1.0
        if r > s: return -1.0
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_basis_vectors():
    """
    Build all 15 basis vectors for the G-extended perturbation space:
        4 G-invariant  (one per orbit)
        3 O1 anti-symmetric
        1 O2 anti-symmetric
        7 O3 non-trivial (Hadamard/Z₂³ characters, excluding trivial)

    Returns dict with keys: 'G_inv', 'O1_anti', 'O2_anti', 'O3_anti', 'all'.
    Each entry is a list of (9,9,9) arrays.
    """
    # --- G-invariant directions (indicator of each orbit) ---
    G_inv = []
    for key in ('O0', 'O1', 'O2', 'O3'):
        entries = {t: 1.0 for t in G_ORBITS[key]}
        G_inv.append(_T_from_entries(entries))

    # --- O1 anti-symmetric: 3 basis vectors, one per pair ---
    O1_anti = []
    for (t_plus, t_minus) in O1_PAIRS:
        entries = {t_plus: 1.0, t_minus: -1.0}
        O1_anti.append(_T_from_entries(entries))

    # --- O2 anti-symmetric: 1 basis vector ---
    entries = {}
    for (r, s, u) in G_ORBITS['O2']:
        sg = _o2_sign(r, s, u)
        if sg != 0.0:
            entries[(r, s, u)] = sg
    O2_anti = [_T_from_entries(entries)]

    # --- O3 non-trivial: 7 Hadamard characters of Z₂³ ---
    # Index O3 by (a,b,c) = (r-1, s-1, u-1) ∈ {0,1}³
    # Character χ_{ijk}(a,b,c) = (-1)^{i*a + j*b + k*c}
    # Exclude trivial character (0,0,0); enumerate i,j,k via idx 1..7
    O3_anti = []
    for idx in range(1, 8):
        i_bit = (idx >> 2) & 1
        j_bit = (idx >> 1) & 1
        k_bit = idx & 1
        entries = {}
        for (r, s, u) in O3_TERMS:
            a, b, c = r - 1, s - 1, u - 1
            val = float((-1) ** (i_bit * a + j_bit * b + k_bit * c))
            entries[(r, s, u)] = val
        O3_anti.append(_T_from_entries(entries))

    all_vecs = G_inv + O1_anti + O2_anti + O3_anti  # 4+3+1+7 = 15
    return {
        'G_inv': G_inv,
        'O1_anti': O1_anti,
        'O2_anti': O2_anti,
        'O3_anti': O3_anti,
        'all': all_vecs,
    }


def build_E_from_coords(coords):
    """
    coords: shape (15,) — coefficients in the full 15-dim basis.
    Returns E tensor shape (9,9,9).
    """
    coords = np.asarray(coords, dtype=np.float64)
    basis = build_basis_vectors()['all']
    E = np.zeros((9, 9, 9), dtype=np.float64)
    for i, v in enumerate(basis):
        E += coords[i] * v
    return E


def E_norm_from_coords(coords):
    """||E||_F for the E built from coords."""
    return float(np.linalg.norm(build_E_from_coords(coords)))


def project_to_G_invariant(E):
    """Project E onto the 4-dim G-invariant subspace."""
    G_inv = build_basis_vectors()['G_inv']
    result = np.zeros_like(E)
    for v in G_inv:
        norm2 = float(np.dot(v.ravel(), v.ravel()))
        coeff = float(np.dot(E.ravel(), v.ravel())) / norm2
        result += coeff * v
    return result


def project_to_non_invariant(E):
    """Return E minus its G-invariant component."""
    return E - project_to_G_invariant(E)


def spectral_analysis(T):
    """
    Compute singular values of all 3 mode flattenings.
    sqrt3_gap: sigma[3] of mode-0 minus √3
    (positive = above √3, negative = below √3).
    """
    sv0 = np.linalg.svd(T.reshape(9, 81), compute_uv=False)
    sv1 = np.linalg.svd(T.transpose(1, 0, 2).reshape(9, 81), compute_uv=False)
    sv2 = np.linalg.svd(T.transpose(2, 0, 1).reshape(9, 81), compute_uv=False)
    sqrt3 = math.sqrt(3.0)
    gap = float(sv0[3]) - sqrt3
    return {
        'mode0_svs': sv0,
        'mode1_svs': sv1,
        'mode2_svs': sv2,
        'sqrt3_gap': gap,
    }
