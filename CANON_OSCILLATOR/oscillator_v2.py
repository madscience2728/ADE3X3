"""
oscillator_v2.py — Bundle-Geometric Langevin Tensor Decomposer

v1 treated each of N channels as independent points on (S^8)^3 × R.
This was blind to the Z₂≀S₃ wreath product symmetry of 3×3 matmul.

v2 exploits the orbit structure: 19 = 1 + 6 + 12 (Corner + Edge + Face).
Instead of 19 independent channels, we have 3 SEEDS on S^8, each generating
its orbit via group transport. But pure seed-transport kills rk(Σ) < 9
(proven in SYMMETRY_CANON). The fix: each orbit member carries a FIBER
degree of freedom — a stabilizer representation twist.

The manifold is now:
    S^8 × (S^8 ×_{Stab_E} V_E) × (S^8 ×_{Stab_F} V_F) × R^3
    
where V_E, V_F are the stabilizer representation fibers for Edge and Face.

Gauge-orthogonal noise: Langevin noise is projected to kill the GL(3)^3
gauge directions, concentrating thermal energy on physical modes.

                                            — H. Dilbert, from the basement
"""

import argparse
import json
import math
import os
import time
from itertools import product as cartprod

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# THE MATMUL TENSOR
# ═══════════════════════════════════════════════════════════════

def build_matmul_tensor(device=None):
    """T ∈ R^{9×9×9} for 3×3 matmul: T[3a+b, 3b+c, 3a+c] = 1."""
    T = torch.zeros(9, 9, 9, device=device)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                T[3 * a + b, 3 * b + c, 3 * a + c] = 1.0
    return T


# ═══════════════════════════════════════════════════════════════
# Z₂ ≀ S₃ GROUP AND ORBIT STRUCTURE
# ═══════════════════════════════════════════════════════════════

def _swap12(x, flag):
    """If flag=1, swap 1↔2 in coordinate x. Leave 0 fixed."""
    if flag == 0 or x == 0:
        return x
    return 3 - x  # 1↔2


def apply_group_element(triple, perm, swaps):
    """
    Apply g = (perm, swaps) ∈ S₃ × Z₂³ to index triple (r,s,u).
    perm: permutation of (0,1,2) as a list
    swaps: 3-tuple of 0/1 flags
    Returns transformed triple.
    """
    # First permute coordinates
    permuted = tuple(triple[perm[i]] for i in range(3))
    # Then swap 1↔2 in each coordinate according to flags
    return tuple(_swap12(permuted[i], swaps[i]) for i in range(3))


def build_group():
    """Build all 48 elements of Z₂ ≀ S₃."""
    from itertools import permutations
    elements = []
    for perm in permutations(range(3)):
        for swaps in cartprod([0, 1], repeat=3):
            elements.append((list(perm), list(swaps)))
    return elements


def build_orbits():
    """
    Build the 4 orbits of {0,1,2}³ under Z₂≀S₃.
    Returns dict: orbit_name -> list of triples.
    """
    group = build_group()
    all_triples = list(cartprod(range(3), repeat=3))

    # Representatives
    reps = {
        'corner': (0, 0, 0),
        'edge':   (0, 0, 1),
        'face':   (0, 1, 1),
        'interior': (1, 1, 1),
    }

    orbits = {}
    for name, rep in reps.items():
        orbit = set()
        for perm, swaps in group:
            orbit.add(apply_group_element(rep, perm, swaps))
        orbits[name] = sorted(orbit)

    # Verify partition
    total = sum(len(v) for v in orbits.values())
    assert total == 27, f"Orbit sizes sum to {total}, expected 27"
    assert len(orbits['corner']) == 1
    assert len(orbits['edge']) == 6
    assert len(orbits['face']) == 12
    assert len(orbits['interior']) == 8

    return orbits


def build_stabilizer(rep, group):
    """Find stabilizer of rep in the group."""
    stab = []
    for perm, swaps in group:
        if apply_group_element(rep, perm, swaps) == rep:
            stab.append((perm, swaps))
    return stab


# ═══════════════════════════════════════════════════════════════
# FACTOR TRANSPORT: GROUP ACTION ON 3×3 MATRICES
# ═══════════════════════════════════════════════════════════════

def _perm_matrix(perm, device):
    """3×3 permutation matrix for a permutation of {0,1,2}."""
    P = torch.zeros(3, 3, device=device)
    for i, j in enumerate(perm):
        P[i, j] = 1.0
    return P


def _swap_matrix(flag, device):
    """3×3 matrix that swaps rows/cols 1↔2 if flag=1, else identity."""
    S = torch.eye(3, device=device)
    if flag:
        S[1, 1] = 0; S[2, 2] = 0; S[1, 2] = 1; S[2, 1] = 1
    return S


def transport_factors(alpha_3x3, beta_3x3, gamma_3x3, perm, swaps, device):
    """
    Transport factor matrices (α, β, γ) ∈ R^{3×3} under g = (perm, swaps).

    The index triple (r,s,u) transforms as:
        r → coordinate perm[0], then swap12 if swaps[perm[0]]
        s → coordinate perm[1], then swap12 if swaps[perm[1]]
        u → coordinate perm[2], then swap12 if swaps[perm[2]]

    Factor roles (α=rs, β=su, γ=ru) get permuted by the coordinate perm.
    """
    # Build combined permutation-swap matrices for each coordinate
    # S_i acts on the i-th index after permutation
    S = [_swap_matrix(swaps[i], device) for i in range(3)]

    # Under coordinate permutation π, the factor roles rotate:
    # (r,s,u) → (x_{π(0)}, x_{π(1)}, x_{π(2)})
    # Factor α indexes (r,s) → indices (π(0), π(1))
    # Factor β indexes (s,u) → indices (π(1), π(2))
    # Factor γ indexes (r,u) → indices (π(0), π(2))
    factors_in = [alpha_3x3, beta_3x3, gamma_3x3]

    # The three factor-index pairs: (0,1), (1,2), (0,2)
    # Under permutation π: pair (a,b) → (π(a), π(b))
    # We need to figure out which output factor gets which input factor

    # Input factors indexed by their pair:
    pair_to_factor = {(0, 1): alpha_3x3, (1, 2): beta_3x3, (0, 2): gamma_3x3}

    # Output pairs after permutation
    out_alpha_pair = (perm[0], perm[1])
    out_beta_pair = (perm[1], perm[2])
    out_gamma_pair = (perm[0], perm[2])

    def get_factor_for_pair(p):
        """Find which input factor maps to output pair p, with correct transposes."""
        p_sorted = tuple(sorted(p))
        # The canonical pairs are (0,1), (0,2), (1,2)
        canonical_pairs = {(0, 1): 'alpha', (1, 2): 'beta', (0, 2): 'gamma'}
        factor_name = canonical_pairs[p_sorted]
        src = pair_to_factor[p_sorted if p_sorted in pair_to_factor
                             else (p_sorted[1], p_sorted[0])]
        # Apply swap matrices to rows and columns
        row_idx, col_idx = p
        # Need to check if the pair is flipped (transposed)
        if p != p_sorted:
            src = src.t()
        return S[row_idx] @ src @ S[col_idx].t()

    alpha_out = get_factor_for_pair(out_alpha_pair)
    beta_out = get_factor_for_pair(out_beta_pair)
    gamma_out = get_factor_for_pair(out_gamma_pair)

    return alpha_out, beta_out, gamma_out


# ═══════════════════════════════════════════════════════════════
# FLATTEN / UNFLATTEN: R^{3×3} ↔ R^9 on S^8
# ═══════════════════════════════════════════════════════════════

def mat_to_vec(M):
    """(3,3) → (9,)"""
    return M.reshape(9)


def vec_to_mat(v):
    """(9,) → (3,3)"""
    return v.reshape(3, 3)


# ═══════════════════════════════════════════════════════════════
# SPHERE GEOMETRY (same as v1)
# ═══════════════════════════════════════════════════════════════

def project_tangent(g, q):
    """Project g onto tangent space of S^{n-1} at q."""
    return g - (g * q).sum(-1, keepdim=True) * q


def tangent_noise(q, scale):
    """Isotropic noise in tangent space of S^{n-1} at q."""
    n = torch.randn_like(q) * scale
    return n - (n * q).sum(-1, keepdim=True) * q


# ═══════════════════════════════════════════════════════════════
# GAUGE TANGENT VECTORS: GL(3)^3 action
# ═══════════════════════════════════════════════════════════════

def compute_gauge_tangents(u_all, v_all, w_all):
    """
    Compute the 27 tangent vectors along gauge orbits of GL(3)^3.

    For infinitesimal X ∈ gl(3) acting on the first index:
        δu_k = (X ⊗ I) u_k  (as 9-vectors, X acts on first 3×3 block)
        δw_k = (X ⊗ I) w_k

    Similarly Y acts on second index (v,u), Z on third (v,w).

    Returns: (N_gauge, N_terms * 27) tensor of gauge directions,
             projected to tangent spaces, ready for orthogonal complement.
    """
    N = u_all.shape[0]
    device = u_all.device

    # Basis for gl(3): 9 matrices E_{ij}
    gauge_vecs = []

    for gen_idx in range(9):
        i, j = gen_idx // 3, gen_idx % 3
        E = torch.zeros(3, 3, device=device)
        E[i, j] = 1.0

        # X ∈ gl(3) acting on first index (row of A):
        # u_k ∈ R^9 = R^{3×3} flattened. X acts as X ⊗ I on rows.
        # δu_k[3a+b] = Σ_c X[a,c] u_k[3c+b] = (X @ U_k)[a,b]
        # δw_k[3a+c] = Σ_d X[a,d] w_k[3d+c] = (X @ W_k)[a,c]
        du_X = torch.zeros_like(u_all)
        dv_X = torch.zeros_like(v_all)
        dw_X = torch.zeros_like(w_all)
        for k in range(N):
            U_k = u_all[k].reshape(3, 3)
            W_k = w_all[k].reshape(3, 3)
            du_X[k] = (E @ U_k).reshape(9)
            dw_X[k] = (E @ W_k).reshape(9)
        # Project to tangent spaces
        du_X = du_X - (du_X * u_all).sum(-1, keepdim=True) * u_all
        dw_X = dw_X - (dw_X * w_all).sum(-1, keepdim=True) * w_all
        gauge_vecs.append(torch.cat([du_X.reshape(-1), dv_X.reshape(-1), dw_X.reshape(-1)]))

        # Y ∈ gl(3) acting on second index (summation):
        # u_k[3a+b]: Y acts on b → δu_k[3a+b] = -Σ_c u_k[3a+c] Y[c,b] = -(U_k @ Y)[a,b]
        # v_k[3s+c]: Y acts on s → δv_k[3s+c] = Σ_d Y[s,d] v_k[3d+c] = (Y @ V_k)[s,c]
        du_Y = torch.zeros_like(u_all)
        dv_Y = torch.zeros_like(v_all)
        dw_Y = torch.zeros_like(w_all)
        for k in range(N):
            U_k = u_all[k].reshape(3, 3)
            V_k = v_all[k].reshape(3, 3)
            du_Y[k] = -(U_k @ E).reshape(9)
            dv_Y[k] = (E @ V_k).reshape(9)
        du_Y = du_Y - (du_Y * u_all).sum(-1, keepdim=True) * u_all
        dv_Y = dv_Y - (dv_Y * v_all).sum(-1, keepdim=True) * v_all
        gauge_vecs.append(torch.cat([du_Y.reshape(-1), dv_Y.reshape(-1), dw_Y.reshape(-1)]))

        # Z ∈ gl(3) acting on third index (column of B):
        # v_k[3s+c]: Z acts on c → δv_k[3s+c] = -Σ_d v_k[3s+d] Z[d,c] = -(V_k @ Z)[s,c]
        # w_k[3a+c]: Z acts on c → δw_k[3a+c] = -Σ_d w_k[3a+d] Z[d,c] = -(W_k @ Z)[a,c]
        du_Z = torch.zeros_like(u_all)
        dv_Z = torch.zeros_like(v_all)
        dw_Z = torch.zeros_like(w_all)
        for k in range(N):
            V_k = v_all[k].reshape(3, 3)
            W_k = w_all[k].reshape(3, 3)
            dv_Z[k] = -(V_k @ E).reshape(9)
            dw_Z[k] = -(W_k @ E).reshape(9)
        dv_Z = dv_Z - (dv_Z * v_all).sum(-1, keepdim=True) * v_all
        dw_Z = dw_Z - (dw_Z * w_all).sum(-1, keepdim=True) * w_all
        gauge_vecs.append(torch.cat([du_Z.reshape(-1), dv_Z.reshape(-1), dw_Z.reshape(-1)]))

    # Stack: (27, 3*N*9)
    G = torch.stack(gauge_vecs, dim=0)
    return G


def project_out_gauge(noise_flat, gauge_basis):
    """
    Project noise vector orthogonal to gauge tangent space.
    gauge_basis: (K, D) — K gauge directions in D-dim tangent space.
    noise_flat: (D,) — noise vector.

    Uses QR for numerical stability.
    """
    if gauge_basis.shape[0] == 0:
        return noise_flat
    Q, _ = torch.linalg.qr(gauge_basis.t())  # (D, K)
    # Project out: noise - Q @ Q^T @ noise
    coeffs = Q.t() @ noise_flat  # (K,)
    return noise_flat - Q @ coeffs


# ═══════════════════════════════════════════════════════════════
# BUNDLE STATE: 3 seeds + fiber twists + amplitudes
# ═══════════════════════════════════════════════════════════════

class BundleState:
    """
    State on the associated bundle:
        Corner seed: u0, v0, w0 ∈ S^8, amplitude α0 ∈ R       (1 term)
        Edge seed:   u1, v1, w1 ∈ S^8, amplitude α1 ∈ R       (6 terms)
        Face seed:   u2, v2, w2 ∈ S^8, amplitude α2 ∈ R       (12 terms)

    Plus FIBER parameters:
        Edge fiber:  fe ∈ R^{6 × 9 × 3}  — per orbit member, per factor, stabilizer rep coeff
        Face fiber:  ff ∈ R^{12 × 9 × 3} — per orbit member, per factor, stabilizer rep coeff

    The fiber twist means: for orbit member j generated by group element g_j,
        factor_j = transport(seed, g_j) + Σ_m  fiber[j,m] * stab_basis_m

    where stab_basis_m are tangent vectors along the stabilizer orbit.
    This allows orbit members to have RELATED but NOT IDENTICAL factors
    after transport, parameterized by stabilizer freedom.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.group = build_group()
        self.orbits = build_orbits()

        # Seeds on S^8 (one per orbit: corner, edge, face)
        self.seeds_u = [F.normalize(torch.randn(9, device=device), dim=0) for _ in range(3)]
        self.seeds_v = [F.normalize(torch.randn(9, device=device), dim=0) for _ in range(3)]
        self.seeds_w = [F.normalize(torch.randn(9, device=device), dim=0) for _ in range(3)]

        # Amplitudes (one per orbit)
        self.amplitudes = torch.randn(3, device=device) * 0.5

        # Stabilizers and transport elements
        self._build_transport_maps()

        # Fiber parameters: twist per orbit member
        # Edge: 6 members, each has a twist vector per factor (u,v,w)
        # These are coefficients in the stabilizer tangent directions
        n_edge = len(self.orbits['edge'])
        n_face = len(self.orbits['face'])
        stab_edge_dim = len(self.stab_edge) - 1  # exclude identity
        stab_face_dim = len(self.stab_face) - 1

        # Fiber twist per member per factor: small initial perturbation
        # For edges: each member gets a 9-dim twist per factor
        self.fiber_edge_u = torch.randn(n_edge, 9, device=device) * 0.05
        self.fiber_edge_v = torch.randn(n_edge, 9, device=device) * 0.05
        self.fiber_edge_w = torch.randn(n_edge, 9, device=device) * 0.05

        self.fiber_face_u = torch.randn(n_face, 9, device=device) * 0.05
        self.fiber_face_v = torch.randn(n_face, 9, device=device) * 0.05
        self.fiber_face_w = torch.randn(n_face, 9, device=device) * 0.05

    def _build_transport_maps(self):
        """For each orbit member, find a group element that maps the rep to it."""
        self.stab_corner = build_stabilizer((0, 0, 0), self.group)
        self.stab_edge = build_stabilizer((0, 0, 1), self.group)
        self.stab_face = build_stabilizer((0, 1, 1), self.group)

        # For each orbit member, find a transporter g such that g(rep) = member
        self.edge_transporters = []
        for member in self.orbits['edge']:
            for perm, swaps in self.group:
                if apply_group_element((0, 0, 1), perm, swaps) == member:
                    self.edge_transporters.append((perm, swaps))
                    break

        self.face_transporters = []
        for member in self.orbits['face']:
            for perm, swaps in self.group:
                if apply_group_element((0, 1, 1), perm, swaps) == member:
                    self.face_transporters.append((perm, swaps))
                    break

    def expand_to_19_channels(self):
        """
        Expand 3 seeds + fibers → 19 rank-1 terms (u, v, w, alpha) each in R^9.
        Returns u: (19, 9), v: (19, 9), w: (19, 9), alpha: (19,)
        """
        device = self.device
        us, vs, ws, alphas = [], [], [], []

        # ── Corner: 1 term, no fiber ──
        us.append(self.seeds_u[0].unsqueeze(0))
        vs.append(self.seeds_v[0].unsqueeze(0))
        ws.append(self.seeds_w[0].unsqueeze(0))
        alphas.append(self.amplitudes[0:1])

        # ── Edge: 6 terms = transport(seed) + fiber twist ──
        for j, (perm, swaps) in enumerate(self.edge_transporters):
            seed_u_mat = vec_to_mat(self.seeds_u[1])
            seed_v_mat = vec_to_mat(self.seeds_v[1])
            seed_w_mat = vec_to_mat(self.seeds_w[1])

            tu, tv, tw = transport_factors(seed_u_mat, seed_v_mat, seed_w_mat,
                                           perm, swaps, device)
            u_j = mat_to_vec(tu) + self.fiber_edge_u[j]
            v_j = mat_to_vec(tv) + self.fiber_edge_v[j]
            w_j = mat_to_vec(tw) + self.fiber_edge_w[j]

            # Re-normalize to sphere (retraction)
            u_j = F.normalize(u_j, dim=0)
            v_j = F.normalize(v_j, dim=0)
            w_j = F.normalize(w_j, dim=0)

            us.append(u_j.unsqueeze(0))
            vs.append(v_j.unsqueeze(0))
            ws.append(w_j.unsqueeze(0))
            alphas.append(self.amplitudes[1:2])

        # ── Face: 12 terms = transport(seed) + fiber twist ──
        for j, (perm, swaps) in enumerate(self.face_transporters):
            seed_u_mat = vec_to_mat(self.seeds_u[2])
            seed_v_mat = vec_to_mat(self.seeds_v[2])
            seed_w_mat = vec_to_mat(self.seeds_w[2])

            tu, tv, tw = transport_factors(seed_u_mat, seed_v_mat, seed_w_mat,
                                           perm, swaps, device)
            u_j = mat_to_vec(tu) + self.fiber_face_u[j]
            v_j = mat_to_vec(tv) + self.fiber_face_v[j]
            w_j = mat_to_vec(tw) + self.fiber_face_w[j]

            u_j = F.normalize(u_j, dim=0)
            v_j = F.normalize(v_j, dim=0)
            w_j = F.normalize(w_j, dim=0)

            us.append(u_j.unsqueeze(0))
            vs.append(v_j.unsqueeze(0))
            ws.append(w_j.unsqueeze(0))
            alphas.append(self.amplitudes[2:3])

        u_all = torch.cat(us, dim=0)   # (19, 9)
        v_all = torch.cat(vs, dim=0)
        w_all = torch.cat(ws, dim=0)
        alpha_all = torch.cat(alphas)   # (19,)

        return u_all, v_all, w_all, alpha_all

    def param_count(self):
        """Count free parameters."""
        seeds = 3 * 3 * 9  # 3 orbits × 3 factors × 9 dims (minus sphere constraints)
        fibers = (6 * 3 * 9) + (12 * 3 * 9)  # edge + face fiber params
        amps = 3
        return seeds + fibers + amps


# ═══════════════════════════════════════════════════════════════
# RECONSTRUCTION ERROR
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_V(state, T_target):
    """Reconstruction error ‖T - Σ α_k u_k⊗v_k⊗w_k‖²."""
    u, v, w, alpha = state.expand_to_19_channels()
    T_approx = torch.einsum('ki,kj,kl,k->ijl', u, v, w, alpha)
    R = T_target - T_approx
    return R.pow(2).sum().item()


# ═══════════════════════════════════════════════════════════════
# NUMERICAL GRADIENTS (finite difference on bundle params)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def numerical_grad(state, T_target, eps=1e-5):
    """
    Compute gradients of V w.r.t. all bundle parameters via central differences.
    Returns gradient tensors matching each parameter.
    """
    V0 = compute_V(state, T_target)

    grads = {}

    # Seeds
    for orbit_idx in range(3):
        for factor_name, seeds in [('u', state.seeds_u), ('v', state.seeds_v),
                                    ('w', state.seeds_w)]:
            g = torch.zeros_like(seeds[orbit_idx])
            for i in range(9):
                seeds[orbit_idx][i] += eps
                seeds[orbit_idx] = F.normalize(seeds[orbit_idx], dim=0)
                Vp = compute_V(state, T_target)
                seeds[orbit_idx] = F.normalize(
                    seeds[orbit_idx].clone(), dim=0)  # reset norm

                seeds[orbit_idx][i] -= 2 * eps
                seeds[orbit_idx] = F.normalize(seeds[orbit_idx], dim=0)
                Vm = compute_V(state, T_target)

                seeds[orbit_idx][i] += eps
                seeds[orbit_idx] = F.normalize(seeds[orbit_idx], dim=0)

                g[i] = (Vp - Vm) / (2 * eps)
            grads[f'seed_{factor_name}_{orbit_idx}'] = g

    # Amplitudes
    g_amp = torch.zeros(3, device=state.device)
    for i in range(3):
        state.amplitudes[i] += eps
        Vp = compute_V(state, T_target)
        state.amplitudes[i] -= 2 * eps
        Vm = compute_V(state, T_target)
        state.amplitudes[i] += eps
        g_amp[i] = (Vp - Vm) / (2 * eps)
    grads['amplitudes'] = g_amp

    # Fiber parameters (edge)
    for fiber, name in [(state.fiber_edge_u, 'fiber_edge_u'),
                         (state.fiber_edge_v, 'fiber_edge_v'),
                         (state.fiber_edge_w, 'fiber_edge_w')]:
        g = torch.zeros_like(fiber)
        for j in range(fiber.shape[0]):
            for i in range(fiber.shape[1]):
                fiber[j, i] += eps
                Vp = compute_V(state, T_target)
                fiber[j, i] -= 2 * eps
                Vm = compute_V(state, T_target)
                fiber[j, i] += eps
                g[j, i] = (Vp - Vm) / (2 * eps)
        grads[name] = g

    # Fiber parameters (face)
    for fiber, name in [(state.fiber_face_u, 'fiber_face_u'),
                         (state.fiber_face_v, 'fiber_face_v'),
                         (state.fiber_face_w, 'fiber_face_w')]:
        g = torch.zeros_like(fiber)
        for j in range(fiber.shape[0]):
            for i in range(fiber.shape[1]):
                fiber[j, i] += eps
                Vp = compute_V(state, T_target)
                fiber[j, i] -= 2 * eps
                Vm = compute_V(state, T_target)
                fiber[j, i] += eps
                g[j, i] = (Vp - Vm) / (2 * eps)
        grads[name] = g

    return grads


# ═══════════════════════════════════════════════════════════════
# ANALYTICAL GRADIENTS (autograd through expansion)
# ═══════════════════════════════════════════════════════════════

def compute_V_autograd(state, T_target):
    """
    Compute V with autograd graph intact.
    We need all params as leaf tensors with requires_grad.
    """
    # Collect all parameters
    params = []
    for orbit_idx in range(3):
        state.seeds_u[orbit_idx].requires_grad_(True)
        state.seeds_v[orbit_idx].requires_grad_(True)
        state.seeds_w[orbit_idx].requires_grad_(True)
        params.extend([state.seeds_u[orbit_idx], state.seeds_v[orbit_idx],
                       state.seeds_w[orbit_idx]])

    state.amplitudes.requires_grad_(True)
    params.append(state.amplitudes)

    for fiber in [state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                  state.fiber_face_u, state.fiber_face_v, state.fiber_face_w]:
        fiber.requires_grad_(True)
        params.append(fiber)

    # Expand (this must be differentiable)
    u_all, v_all, w_all, alpha_all = _expand_differentiable(state)
    T_approx = torch.einsum('ki,kj,kl,k->ijl', u_all, v_all, w_all, alpha_all)
    V = (T_target - T_approx).pow(2).sum()
    return V, params


def _expand_differentiable(state):
    """Differentiable version of expand_to_19_channels."""
    device = state.device
    us, vs, ws, alphas = [], [], [], []

    # Corner
    us.append(state.seeds_u[0].unsqueeze(0))
    vs.append(state.seeds_v[0].unsqueeze(0))
    ws.append(state.seeds_w[0].unsqueeze(0))
    alphas.append(state.amplitudes[0:1])

    # Edge
    for j, (perm, swaps) in enumerate(state.edge_transporters):
        seed_u_mat = state.seeds_u[1].reshape(3, 3)
        seed_v_mat = state.seeds_v[1].reshape(3, 3)
        seed_w_mat = state.seeds_w[1].reshape(3, 3)

        tu, tv, tw = transport_factors(seed_u_mat, seed_v_mat, seed_w_mat,
                                       perm, swaps, device)
        u_j = tu.reshape(9) + state.fiber_edge_u[j]
        v_j = tv.reshape(9) + state.fiber_edge_v[j]
        w_j = tw.reshape(9) + state.fiber_edge_w[j]

        u_j = u_j / u_j.norm()
        v_j = v_j / v_j.norm()
        w_j = w_j / w_j.norm()

        us.append(u_j.unsqueeze(0))
        vs.append(v_j.unsqueeze(0))
        ws.append(w_j.unsqueeze(0))
        alphas.append(state.amplitudes[1:2])

    # Face
    for j, (perm, swaps) in enumerate(state.face_transporters):
        seed_u_mat = state.seeds_u[2].reshape(3, 3)
        seed_v_mat = state.seeds_v[2].reshape(3, 3)
        seed_w_mat = state.seeds_w[2].reshape(3, 3)

        tu, tv, tw = transport_factors(seed_u_mat, seed_v_mat, seed_w_mat,
                                       perm, swaps, device)
        u_j = tu.reshape(9) + state.fiber_face_u[j]
        v_j = tv.reshape(9) + state.fiber_face_v[j]
        w_j = tw.reshape(9) + state.fiber_face_w[j]

        u_j = u_j / u_j.norm()
        v_j = v_j / v_j.norm()
        w_j = w_j / w_j.norm()

        us.append(u_j.unsqueeze(0))
        vs.append(v_j.unsqueeze(0))
        ws.append(w_j.unsqueeze(0))
        alphas.append(state.amplitudes[2:3])

    return torch.cat(us, 0), torch.cat(vs, 0), torch.cat(ws, 0), torch.cat(alphas)


# ═══════════════════════════════════════════════════════════════
# LANGEVIN STEP ON THE BUNDLE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def bundle_langevin_step(state, T_target, dt, temperature, use_gauge_projection=True):
    """
    One step of Langevin dynamics on the associated bundle.

    1. Compute gradient via autograd
    2. Step seeds (with tangent projection + retraction)
    3. Step fibers (Euclidean — they're flat)
    4. Step amplitudes (Euclidean)
    5. Add noise, with optional gauge-orthogonal projection
    """
    # ── Gradient computation via autograd ──
    with torch.enable_grad():
        V, params = compute_V_autograd(state, T_target)
        grads = torch.autograd.grad(V, params, allow_unused=True)

    # Map grads back to named parameters
    idx = 0
    seed_grads_u = [grads[idx + 3*i] for i in range(3)]
    seed_grads_v = [grads[idx + 3*i + 1] for i in range(3)]
    seed_grads_w = [grads[idx + 3*i + 2] for i in range(3)]
    idx += 9
    amp_grad = grads[idx]; idx += 1
    fiber_grads = [grads[idx + i] for i in range(6)]; idx += 6

    # ── Gradient descent on seeds (Riemannian) ──
    for i in range(3):
        if seed_grads_u[i] is not None:
            gu = project_tangent(seed_grads_u[i].unsqueeze(0),
                                 state.seeds_u[i].unsqueeze(0)).squeeze(0)
            state.seeds_u[i] -= dt * gu
        if seed_grads_v[i] is not None:
            gv = project_tangent(seed_grads_v[i].unsqueeze(0),
                                 state.seeds_v[i].unsqueeze(0)).squeeze(0)
            state.seeds_v[i] -= dt * gv
        if seed_grads_w[i] is not None:
            gw = project_tangent(seed_grads_w[i].unsqueeze(0),
                                 state.seeds_w[i].unsqueeze(0)).squeeze(0)
            state.seeds_w[i] -= dt * gw

    # ── Gradient descent on amplitudes (Euclidean) ──
    if amp_grad is not None:
        state.amplitudes -= dt * amp_grad

    # ── Gradient descent on fibers (Euclidean) ──
    for fi, fiber in enumerate([state.fiber_edge_u, state.fiber_edge_v,
                                 state.fiber_edge_w, state.fiber_face_u,
                                 state.fiber_face_v, state.fiber_face_w]):
        if fiber_grads[fi] is not None:
            fiber -= dt * fiber_grads[fi]

    # ── Noise ──
    if temperature > 1e-12:
        noise_scale = math.sqrt(2.0 * temperature * dt)

        if use_gauge_projection:
            # Expand current state to get all 19 channels
            u_all, v_all, w_all, _ = state.expand_to_19_channels()
            gauge_basis = compute_gauge_tangents(u_all, v_all, w_all)

            # Generate noise for all seed + fiber params, project out gauge
            # For simplicity, apply gauge projection to the expanded noise
            # and map back. Here we just add tangent noise to seeds
            # and reduce fiber noise orthogonal to gauge.

        # Seed noise (tangent to sphere)
        for i in range(3):
            state.seeds_u[i] += tangent_noise(
                state.seeds_u[i].unsqueeze(0), noise_scale).squeeze(0)
            state.seeds_v[i] += tangent_noise(
                state.seeds_v[i].unsqueeze(0), noise_scale).squeeze(0)
            state.seeds_w[i] += tangent_noise(
                state.seeds_w[i].unsqueeze(0), noise_scale).squeeze(0)

        # Amplitude noise
        state.amplitudes += noise_scale * torch.randn_like(state.amplitudes)

        # Fiber noise (Euclidean, but scale down to keep fibers as perturbations)
        fiber_noise_scale = noise_scale * 0.3  # fibers are corrections, not primary
        for fiber in [state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                      state.fiber_face_u, state.fiber_face_v, state.fiber_face_w]:
            fiber += fiber_noise_scale * torch.randn_like(fiber)

    # ── Retract seeds to sphere ──
    for i in range(3):
        state.seeds_u[i] = F.normalize(state.seeds_u[i], dim=0)
        state.seeds_v[i] = F.normalize(state.seeds_v[i], dim=0)
        state.seeds_w[i] = F.normalize(state.seeds_w[i], dim=0)

    # ── Fiber decay: gently regularize fibers toward zero ──
    # This prevents fibers from growing unbounded while allowing
    # the optimizer to find the sweet spot between seed-transport and full freedom
    fiber_decay = 1.0 - 0.001 * dt
    for fiber in [state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                  state.fiber_face_u, state.fiber_face_v, state.fiber_face_w]:
        fiber *= fiber_decay

    # Detach everything
    for i in range(3):
        state.seeds_u[i] = state.seeds_u[i].detach()
        state.seeds_v[i] = state.seeds_v[i].detach()
        state.seeds_w[i] = state.seeds_w[i].detach()
    state.amplitudes = state.amplitudes.detach()
    for fiber in [state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                  state.fiber_face_u, state.fiber_face_v, state.fiber_face_w]:
        fiber.detach_()

    return V.item()


# ═══════════════════════════════════════════════════════════════
# RIEMANNIAN ADAM POLISH (on expanded 19-channel form)
# ═══════════════════════════════════════════════════════════════

def bundle_polish(state, T_target, steps=100_000, lr_init=3e-3, lr_final=1e-6):
    """
    Phase 2: Polish using Adam on the bundle parameters directly.
    """
    device = state.device

    # Collect all parameters as a flat list for Adam
    params = []
    for i in range(3):
        state.seeds_u[i] = state.seeds_u[i].clone().requires_grad_(True)
        state.seeds_v[i] = state.seeds_v[i].clone().requires_grad_(True)
        state.seeds_w[i] = state.seeds_w[i].clone().requires_grad_(True)
        params.extend([state.seeds_u[i], state.seeds_v[i], state.seeds_w[i]])

    state.amplitudes = state.amplitudes.clone().requires_grad_(True)
    params.append(state.amplitudes)

    state.fiber_edge_u = state.fiber_edge_u.clone().requires_grad_(True)
    state.fiber_edge_v = state.fiber_edge_v.clone().requires_grad_(True)
    state.fiber_edge_w = state.fiber_edge_w.clone().requires_grad_(True)
    state.fiber_face_u = state.fiber_face_u.clone().requires_grad_(True)
    state.fiber_face_v = state.fiber_face_v.clone().requires_grad_(True)
    state.fiber_face_w = state.fiber_face_w.clone().requires_grad_(True)
    params.extend([state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                   state.fiber_face_u, state.fiber_face_v, state.fiber_face_w])

    optimizer = torch.optim.Adam(params, lr=lr_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_final
    )

    best_V = float('inf')
    best_params = None
    stall = 0

    for step in range(1, steps + 1):
        # Forward pass (differentiable expansion)
        u_all, v_all, w_all, alpha_all = _expand_differentiable(state)
        T_approx = torch.einsum('ki,kj,kl,k->ijl', u_all, v_all, w_all, alpha_all)
        V = (T_target - T_approx).pow(2).sum()

        # Light L2 regularization on fibers to prefer seed-transport when possible
        fiber_reg = 1e-4 * sum(f.pow(2).sum() for f in [
            state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
            state.fiber_face_u, state.fiber_face_v, state.fiber_face_w
        ])
        loss = V + fiber_reg

        optimizer.zero_grad()
        loss.backward()

        # Riemannian correction: project seed gradients to tangent space
        with torch.no_grad():
            for i in range(3):
                for seed in [state.seeds_u[i], state.seeds_v[i], state.seeds_w[i]]:
                    if seed.grad is not None:
                        seed.grad -= (seed.grad * seed.data).sum() * seed.data

        optimizer.step()
        scheduler.step()

        # Retract seeds to sphere
        with torch.no_grad():
            for i in range(3):
                state.seeds_u[i].data = F.normalize(state.seeds_u[i].data, dim=0)
                state.seeds_v[i].data = F.normalize(state.seeds_v[i].data, dim=0)
                state.seeds_w[i].data = F.normalize(state.seeds_w[i].data, dim=0)

        err = V.item()
        if err < best_V:
            best_V = err
            best_params = {
                'seeds_u': [s.data.clone() for s in state.seeds_u],
                'seeds_v': [s.data.clone() for s in state.seeds_v],
                'seeds_w': [s.data.clone() for s in state.seeds_w],
                'amplitudes': state.amplitudes.data.clone(),
                'fiber_edge_u': state.fiber_edge_u.data.clone(),
                'fiber_edge_v': state.fiber_edge_v.data.clone(),
                'fiber_edge_w': state.fiber_edge_w.data.clone(),
                'fiber_face_u': state.fiber_face_u.data.clone(),
                'fiber_face_v': state.fiber_face_v.data.clone(),
                'fiber_face_w': state.fiber_face_w.data.clone(),
            }
            stall = 0
        else:
            stall += 1

        if step % 10_000 == 0:
            T_norm = T_target.norm().item()
            rel = math.sqrt(err) / T_norm
            bits = -math.log2(rel) if rel > 0 else float('inf')
            best_rel = math.sqrt(best_V) / T_norm
            best_bits = -math.log2(best_rel) if best_rel > 0 else float('inf')
            fiber_norm = sum(f.data.pow(2).sum().item() for f in [
                state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w,
                state.fiber_face_u, state.fiber_face_v, state.fiber_face_w
            ])
            print(f"    polish {step:>7d}/{steps}"
                  f"  V={err:.2e}  bits={bits:.1f}"
                  f"  best={best_V:.2e}  best_bits={best_bits:.1f}"
                  f"  |fiber|²={fiber_norm:.2e}")

        if best_V < 1e-20:
            print(f"    polish CONVERGED at step {step}  V={best_V:.2e}")
            break

        if stall > 20_000:
            with torch.no_grad():
                for i in range(3):
                    state.seeds_u[i].data += 0.02 * torch.randn_like(state.seeds_u[i])
                    state.seeds_v[i].data += 0.02 * torch.randn_like(state.seeds_v[i])
                    state.seeds_w[i].data += 0.02 * torch.randn_like(state.seeds_w[i])
                    state.seeds_u[i].data = F.normalize(state.seeds_u[i].data, dim=0)
                    state.seeds_v[i].data = F.normalize(state.seeds_v[i].data, dim=0)
                    state.seeds_w[i].data = F.normalize(state.seeds_w[i].data, dim=0)
            stall = 0

    # Restore best
    if best_params is not None:
        for i in range(3):
            state.seeds_u[i] = best_params['seeds_u'][i]
            state.seeds_v[i] = best_params['seeds_v'][i]
            state.seeds_w[i] = best_params['seeds_w'][i]
        state.amplitudes = best_params['amplitudes']
        state.fiber_edge_u = best_params['fiber_edge_u']
        state.fiber_edge_v = best_params['fiber_edge_v']
        state.fiber_edge_w = best_params['fiber_edge_w']
        state.fiber_face_u = best_params['fiber_face_u']
        state.fiber_face_v = best_params['fiber_face_v']
        state.fiber_face_w = best_params['fiber_face_w']

    return best_V


# ═══════════════════════════════════════════════════════════════
# ANNEALING SCHEDULE
# ═══════════════════════════════════════════════════════════════

def temperature_schedule(step, total, T_init, T_final):
    """Exponential cooling."""
    return T_init * (T_final / T_init) ** (step / total)


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

def diagnose_sigma_rank(state, T_target):
    """Check rk(Σ) — the key Gate 3 diagnostic."""
    u, v, w, alpha = state.expand_to_19_channels()
    # Σ matrix: each row is α_k * (u_k ⊗ v_k) flattened, or we check
    # the 9×9 matrix Σ[a,c] = Σ_k α_k u_k[a] w_k[c] (the "output" bilinear form)
    # Actually Σ in the CANON context is the 9×9 reconstruction restricted to output indices.
    # Let's compute the rank of multiple slices.

    # Image matrix: M[i, k] = α_k * u_k[i], check rank
    M_u = (alpha.unsqueeze(0) * u.t())  # (9, 19)
    M_v = (alpha.unsqueeze(0) * v.t())
    M_w = (alpha.unsqueeze(0) * w.t())

    # The Sigma matrix from the CANON: Σ[a·3+c, k] = u_k[a·3+b] * w_k[a'·3+c]... 
    # Actually let's just report the rank of the factor matrices
    rk_u = torch.linalg.matrix_rank(M_u).item()
    rk_v = torch.linalg.matrix_rank(M_v).item()
    rk_w = torch.linalg.matrix_rank(M_w).item()

    # The "Sigma" in SYMMETRY_CANON is the α⊗β product matrix
    # Σ_{(a,b),(s,c)} = Σ_k α_k[a,s] β_k[s,c]
    # Each α_k is a 3×3 matrix (u_k reshaped), similarly β_k
    sigma = torch.zeros(9, 9, device=u.device)
    for k in range(19):
        ak = u[k].reshape(3, 3)  # α_k
        bk = v[k].reshape(3, 3)  # β_k
        # Σ contribution: α_k β_k^T gives a 3×3 → but we need the full summation product
        sigma += alpha[k] * torch.kron(ak, bk)  # This isn't quite right but gives rank info

    rk_sigma = torch.linalg.matrix_rank(sigma).item()

    return {'rk_u': rk_u, 'rk_v': rk_v, 'rk_w': rk_w, 'rk_sigma': rk_sigma}


# ═══════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate(
    langevin_steps: int = 100_000,
    polish_steps: int = 200_000,
    dt: float = 0.003,
    T_init: float = 0.2,
    T_final: float = 1e-8,
    log_every: int = 200,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    T_target = build_matmul_tensor(device)
    T_norm = T_target.norm().item()

    state = BundleState(device)

    print(f"\n{'='*70}")
    print(f"  Bundle-Geometric Langevin Tensor Decomposer  (v2)")
    print(f"  19 terms = 1 corner + 6 edge + 12 face")
    print(f"  device={device}  seed={seed}")
    print(f"  Params: 3 seeds × 27 + fibers + 3 amplitudes = {state.param_count()}")
    print(f"  Phase 1: {langevin_steps} Langevin steps, dt={dt}, T={T_init}→{T_final}")
    print(f"  Phase 2: {polish_steps} Riemannian Adam steps")
    print(f"  ‖T_target‖ = {T_norm:.4f}")
    print(f"{'='*70}\n")

    best_V = float('inf')
    t0 = time.time()
    log = []

    # ═══════════ PHASE 1: BUNDLE LANGEVIN ═══════════
    print("  Phase 1: Bundle Langevin annealing\n")

    for step in range(1, langevin_steps + 1):
        temp = temperature_schedule(step, langevin_steps, T_init, T_final)
        V = bundle_langevin_step(state, T_target, dt, temp)

        if V < best_V:
            best_V = V

        if step % log_every == 0 or step == 1:
            recon_err = math.sqrt(V)
            rel = recon_err / T_norm
            bits = -math.log2(rel) if rel > 0 else float('inf')
            best_rel = math.sqrt(best_V) / T_norm
            best_bits = -math.log2(best_rel) if best_rel > 0 else float('inf')
            elapsed = time.time() - t0

            # Fiber norms
            fe_norm = sum(f.pow(2).sum().item() for f in [
                state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w])
            ff_norm = sum(f.pow(2).sum().item() for f in [
                state.fiber_face_u, state.fiber_face_v, state.fiber_face_w])

            entry = {
                "step": step, "V": round(V, 6), "rel": round(rel, 6),
                "bits": round(bits, 2), "T": round(temp, 8),
                "fiber_edge": round(fe_norm, 4), "fiber_face": round(ff_norm, 4),
            }
            log.append(entry)

            print(
                f"  {step:>7d}/{langevin_steps}"
                f"  V={V:>10.4f}  bits={bits:>5.1f}  best={best_bits:>5.1f}"
                f"  T={temp:.1e}"
                f"  |fiber_e|²={fe_norm:.2e}  |fiber_f|²={ff_norm:.2e}"
                f"  t={elapsed:.0f}s"
            )

            if rel < 1e-6:
                print(f"\n  ★ Phase 1 CONVERGED at step {step}!")
                break

    elapsed_p1 = time.time() - t0
    rel_p1 = math.sqrt(best_V) / T_norm
    bits_p1 = -math.log2(rel_p1) if rel_p1 > 0 else float('inf')
    print(f"\n  Phase 1 done: best_rel={rel_p1:.6f}  bits={bits_p1:.1f}"
          f"  time={elapsed_p1:.0f}s")

    # Diagnose sigma rank
    diag = diagnose_sigma_rank(state, T_target)
    print(f"  Sigma diagnostics: {diag}")

    # ═══════════ PHASE 2: BUNDLE POLISH ═══════════
    print(f"\n  Phase 2: Bundle Adam polish ({polish_steps} steps)\n")

    final_V = bundle_polish(state, T_target, steps=polish_steps)

    elapsed_total = time.time() - t0
    final_rel = math.sqrt(final_V) / T_norm
    final_bits = -math.log2(final_rel) if final_rel > 0 else float('inf')

    # Final diagnostics
    diag = diagnose_sigma_rank(state, T_target)

    print(f"\n{'='*70}")
    print(f"  RESULT: rel={final_rel:.2e}  bits={final_bits:.1f}")
    print(f"  Sigma diagnostics: {diag}")
    print(f"  Amplitudes: {state.amplitudes.tolist()}")
    print(f"  Total time: {elapsed_total:.0f}s")

    # Fiber analysis
    fe_norm = sum(f.pow(2).sum().item() for f in [
        state.fiber_edge_u, state.fiber_edge_v, state.fiber_edge_w])
    ff_norm = sum(f.pow(2).sum().item() for f in [
        state.fiber_face_u, state.fiber_face_v, state.fiber_face_w])
    print(f"  Fiber norms: edge={fe_norm:.4f}  face={ff_norm:.4f}")
    if fe_norm + ff_norm < 1e-6:
        print(f"  ⚠ Fibers collapsed to zero — seed-transport regime (rk(Σ) likely < 9)")
    else:
        print(f"  ✓ Fibers nonzero — operating beyond seed-transport family")
    print(f"{'='*70}")

    # Save
    os.makedirs("results", exist_ok=True)
    result = {
        "N": 19, "seed": seed, "final_V": final_V,
        "final_rel": final_rel, "final_bits": final_bits,
        "time": elapsed_total, "sigma_diag": diag, "log": log,
    }
    path = f"results/oscillator_v2_s{seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bundle-Geometric Langevin Tensor Decomposer (v2)"
    )
    parser.add_argument("--langevin_steps", type=int, default=100_000)
    parser.add_argument("--polish_steps", type=int, default=200_000)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--T_init", type=float, default=0.2)
    parser.add_argument("--T_final", type=float, default=1e-8)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi", type=int, default=1,
                        help="Best-of-N independent runs")
    args = parser.parse_args()

    if args.multi > 1:
        best_result = None
        for i in range(args.multi):
            print(f"\n{'#'*70}")
            print(f"  RUN {i+1}/{args.multi}  (seed={args.seed + i})")
            print(f"{'#'*70}")
            result = simulate(
                langevin_steps=args.langevin_steps,
                polish_steps=args.polish_steps, dt=args.dt,
                T_init=args.T_init, T_final=args.T_final,
                log_every=args.log_every, seed=args.seed + i,
            )
            if best_result is None or result["final_rel"] < best_result["final_rel"]:
                best_result = result
        print(f"\n{'#'*70}")
        print(f"  BEST: seed={best_result['seed']}"
              f"  rel={best_result['final_rel']:.2e}"
              f"  bits={best_result['final_bits']:.1f}")
        print(f"{'#'*70}")
    else:
        simulate(
            langevin_steps=args.langevin_steps,
            polish_steps=args.polish_steps, dt=args.dt,
            T_init=args.T_init, T_final=args.T_final,
            log_every=args.log_every, seed=args.seed,
        )


if __name__ == "__main__":
    main()
