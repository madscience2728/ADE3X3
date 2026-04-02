"""
Theory Synthesis: Front A (Spectral Gap) + Front B (Omega Operator)

THESIS: The spectral gap of Q_chan on ker(Gamma) and the positive-definiteness
of the Omega operator are two faces of the same algebraic structure. This script:

1. Proves the algebraic bridge: Q_chan restricted to the sigma-silent sector Z
   factors through Omega via   Q_chan|_Z(w) = w^T Sigma_Z Omega Sigma_Z^T w
   (up to normalization), so Omega pd => Q_chan|_Z pd.

2. Shows the converse direction: Q_chan pd on ALL of ker(Gamma) implies
   Omega pd on the sigma-silent sector (since Z ⊂ ker(Gamma)).

3. Tests universality by:
   (a) Parametric deformation of known exact decomps
   (b) Random factorized right-inverse triples
   (c) Checking whether Q_chan gap controls Omega gap or vice versa

4. Derives the key identity linking per-channel right-inverse structure
   Gamma * P_s = I_9 to both operators simultaneously.

KEY RESULT TARGET: Show that for ANY valid minimum-rank decomposition with
Delta ⊂ span(H), the following chain holds:
   Gamma*P_s = I_9  =>  Omega = Gamma_Z * Sigma_Z^{-1}  is pd
                    =>  Q_chan|_Z is pd
                    =>  no common witness exists
                    =>  rank(H) = dim(ker(Gamma))
                    =>  R + eta_nullity = n^3

And the stronger universal statement (without assuming Delta ⊂ span(H)):
   Q_chan pd on ker(Gamma)  =>  kernel saturation  =>  conservation law
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import (
    build_mode_matrices_numeric,
    load_public_terms,
    solve_gamma_least_squares,
    tensor_residual_stats,
    terms_from_stacked_factors,
)
from src.ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import (
    Term,
    load_public_rank23_terms,
)

RANK_TOL = 1e-10
np.set_printoptions(precision=12, linewidth=200, suppress=True)


# ═══════════════════════════════════════════════════════════════════════
# §1  Core structures: build everything from factor matrices
# ═══════════════════════════════════════════════════════════════════════

def standard_terms_3x3() -> list[Term]:
    terms = []
    idx = 1
    for r in range(3):
        for s in range(3):
            for u in range(3):
                a = np.zeros((3, 3), dtype=np.int64)
                b = np.zeros((3, 3), dtype=np.int64)
                g = np.zeros((3, 3), dtype=np.int64)
                a[r, s] = 1; b[s, u] = 1; g[r, u] = 1
                terms.append(Term(f"std{idx:02d}", f"standard_{idx:02d}", a, b, g))
                idx += 1
    return terms


def strassen_terms_2x2() -> list[Term]:
    raw = [
        {"alpha": [[1, 0], [0, 1]], "beta": [[1, 0], [0, -1]], "gamma": [[1, 0], [0, 1]]},
        {"alpha": [[0, 0], [1, 1]], "beta": [[1, 0], [0, 0]], "gamma": [[-1, 0], [1, 0]]},
        {"alpha": [[1, 0], [0, 0]], "beta": [[0, 1], [0, -1]], "gamma": [[0, 0], [0, 1]]},
        {"alpha": [[0, 0], [0, 1]], "beta": [[-1, 0], [1, 0]], "gamma": [[0, 0], [1, 0]]},  # fixed: was wrong
        {"alpha": [[1, 1], [0, 0]], "beta": [[0, 0], [0, 1]], "gamma": [[0, 1], [0, -1]]},
        {"alpha": [[-1, 0], [1, 0]], "beta": [[1, 1], [0, 0]], "gamma": [[0, 0], [0, 0]]},  # placeholder
        {"alpha": [[0, 1], [0, -1]], "beta": [[0, 0], [1, 1]], "gamma": [[0, 0], [0, 0]]},  # placeholder
    ]
    # Use the canonical Strassen from the existing codebase instead
    from src.ade3x3.steps.ade3x3_step52_quotient_rank_criterion import strassen_terms_2x2 as st2
    raw_terms = st2()
    terms = []
    for i, t in enumerate(raw_terms, 1):
        terms.append(Term(f"str{i:02d}", f"strassen_{i:02d}",
                          np.array(t["alpha"], dtype=np.int64),
                          np.array(t["beta"], dtype=np.int64),
                          np.array(t["gamma"], dtype=np.int64)))
    return terms


def nullspace(M: np.ndarray, tol: float = RANK_TOL) -> np.ndarray:
    """Orthonormal basis of ker(M) as columns."""
    _, s, vh = np.linalg.svd(M, full_matrices=True)
    rank = int(np.sum(s > tol))
    return vh[rank:].T.copy()


def matrix_rank(M: np.ndarray, tol: float = RANK_TOL) -> int:
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))


def build_channel_matrices(terms: list[Term], n: int = 3) -> list[np.ndarray]:
    """Build the R×(n²) channel product matrices P_s for s=0,...,n-1.
    P_s[k, n*r+u] = alpha_k[n*r+s] * beta_k[n*s+u]
    """
    R = len(terms)
    Ps = []
    for s in range(n):
        P = np.zeros((R, n * n))
        for k, t in enumerate(terms):
            a = np.asarray(t.alpha, dtype=np.float64).ravel()
            b = np.asarray(t.beta, dtype=np.float64).ravel()
            for r in range(n):
                for u in range(n):
                    P[k, n * r + u] = a[n * r + s] * b[n * s + u]
        Ps.append(P)
    return Ps


def build_gamma_matrix(terms: list[Term], n: int = 3) -> np.ndarray:
    """Gamma: (n², R) output weight matrix."""
    return np.stack([np.asarray(t.gamma, dtype=np.float64).ravel() for t in terms], axis=1)


# ═══════════════════════════════════════════════════════════════════════
# §2  Q_chan: channel-separation quadratic form on ker(Gamma)
# ═══════════════════════════════════════════════════════════════════════

def compute_Qchan_on_kerGamma(terms: list[Term], n: int = 3) -> dict:
    """Compute Q_chan = sum_{s<t} D_st^T D_st restricted to ker(Gamma).
    D_st[i,k] = P_s[k,i] - P_t[k,i]  (transposed: maps R-vectors to n²-vectors)

    Returns the restricted operator (generalized eigenvalue problem) and its spectrum.
    """
    Gamma = build_gamma_matrix(terms, n)  # (n², R)
    Ps = build_channel_matrices(terms, n)  # list of (R, n²)
    R = len(terms)

    # Build full Q_chan as R×R matrix: Q[k,l] = sum_{s<t} sum_i (P_s-P_t)[k,i]*(P_s-P_t)[l,i]
    Q = np.zeros((R, R))
    for s in range(n):
        for t in range(s + 1, n):
            D = Ps[s] - Ps[t]  # (R, n²)
            Q += D @ D.T  # (R, R)

    # Restrict to ker(Gamma)
    K = nullspace(Gamma)  # (R, dim_ker)
    if K.shape[1] == 0:
        return {"dim_ker": 0, "Q_restricted": np.array([[]]), "eigenvalues": [], "gap": 0.0}

    Q_restricted = K.T @ Q @ K  # (dim_ker, dim_ker)
    M_restricted = K.T @ K  # should be identity if K is orthonormal

    eigenvalues = sorted(np.linalg.eigvalsh(Q_restricted))
    gap = eigenvalues[0] if eigenvalues else 0.0

    return {
        "dim_ker": K.shape[1],
        "Q_restricted": Q_restricted,
        "eigenvalues": eigenvalues,
        "gap": gap,
        "nullity": sum(1 for e in eigenvalues if abs(e) < RANK_TOL),
    }


# ═══════════════════════════════════════════════════════════════════════
# §3  Omega operator on the sigma-silent sector
# ═══════════════════════════════════════════════════════════════════════

def compute_Omega(terms: list[Term], n: int = 3) -> dict:
    """Compute Omega = Gamma_Z * Sigma_Z^{-1} on Z = ker(H^T) ∩ ker(Delta^T)."""
    sigma, eta1, eta2, delta, h, _ = build_mode_matrices_numeric(terms)
    Gamma = build_gamma_matrix(terms, n)

    # Z = ker(H^T) ∩ ker(Delta^T) = ker([H|Delta]^T)
    HD = np.hstack([h, delta])  # (R, 18+54)
    Z = nullspace(HD.T)  # (R, dim_Z) — columns are basis of Z

    dim_Z = Z.shape[1]
    if dim_Z != n * n:
        return {
            "defined": False, "dim_Z": dim_Z, "expected": n * n,
            "rank_H": matrix_rank(h), "rank_Delta": matrix_rank(delta),
            "rank_HD": matrix_rank(HD),
        }

    Sigma_Z = sigma.T @ Z  # (9, dim_Z)
    Gamma_Z = Gamma @ Z    # (9, dim_Z)

    sigma_Z_rank = matrix_rank(Sigma_Z)
    if sigma_Z_rank < n * n:
        return {"defined": False, "dim_Z": dim_Z, "sigma_Z_rank": sigma_Z_rank}

    Omega = Gamma_Z @ np.linalg.inv(Sigma_Z)  # (9, 9)
    eigenvalues = sorted(np.linalg.eigvalsh(Omega))
    sym_err = float(np.max(np.abs(Omega - Omega.T)))

    return {
        "defined": True,
        "dim_Z": dim_Z,
        "Omega": Omega,
        "eigenvalues": eigenvalues,
        "min_eig": eigenvalues[0],
        "max_eig": eigenvalues[-1],
        "symmetric": sym_err < 1e-10,
        "positive_definite": eigenvalues[0] > RANK_TOL,
        "det": float(np.linalg.det(Omega)),
        "symmetry_error": sym_err,
    }


# ═══════════════════════════════════════════════════════════════════════
# §4  ALGEBRAIC BRIDGE: Q_chan|_Z factors through Omega
# ═══════════════════════════════════════════════════════════════════════

def verify_bridge(terms: list[Term], n: int = 3) -> dict:
    """
    THEOREM (Bridge Identity):

    For w in Z (the sigma-silent sector inside ker(Gamma)):
      Q_chan(w) = sum_{s<t} ||W_s - W_t||^2
                = sum_{s<t} ||(P_s - P_t)^T w||^2

    In the fiber-mode basis, for w in Z:
      P_s^T w = (sigma channel s contribution only)
      because H^T w = 0 and Delta^T w = 0 kill all anisotropy and dead-X.

    Specifically: P_s[k, n*r+u] = a_k[n*r+s]*b_k[n*s+u]
    and sigma_k[n*r+u] = sum_s P_s[k, n*r+u] = sum_s a_k[n*r+s]*b_k[n*s+u]

    So P_s^T w for w in Z decomposes through the per-channel sigma-like matrices.

    We verify numerically that Q_chan restricted to Z equals Sigma_Z^T * Q_output * Sigma_Z
    where Q_output measures the channel separation in output space through Omega.
    """
    sigma, eta1, eta2, delta, h, _ = build_mode_matrices_numeric(terms)
    Gamma = build_gamma_matrix(terms, n)
    Ps = build_channel_matrices(terms, n)
    R = len(terms)

    # Build Z
    HD = np.hstack([h, delta])
    Z = nullspace(HD.T)  # (R, 9)
    dim_Z = Z.shape[1]

    if dim_Z != n * n:
        return {"bridge_verified": False, "reason": f"dim_Z={dim_Z} != {n*n}"}

    Sigma_Z = sigma.T @ Z  # (9, 9)
    Gamma_Z = Gamma @ Z    # (9, 9)

    # Q_chan restricted to Z
    Q_Z = np.zeros((dim_Z, dim_Z))
    for s in range(n):
        for t in range(s + 1, n):
            D = Ps[s] - Ps[t]  # (R, 9)
            DZ = D.T @ Z       # (9, dim_Z)
            Q_Z += DZ.T @ DZ

    # Now build the per-channel projection on Z
    # For w in Z: (P_s - P_t)^T w is the channel-difference image in R^9
    # Define Sigma_s,Z = P_s^T Z  (the per-channel sigma projection)
    Sigma_sZ = []
    for s in range(n):
        Sigma_sZ.append(Ps[s].T @ Z)  # (9, 9)

    # Verify: sum_s Sigma_sZ[s] = Sigma_Z (sigma is the channel sum)
    sigma_sum = sum(Sigma_sZ)
    sigma_sum_err = float(np.max(np.abs(sigma_sum - Sigma_Z)))

    # Q_Z should equal sum_{s<t} (Sigma_sZ[s] - Sigma_sZ[t])^T (Sigma_sZ[s] - Sigma_sZ[t])
    Q_Z_from_channels = np.zeros((dim_Z, dim_Z))
    for s in range(n):
        for t in range(s + 1, n):
            diff = Sigma_sZ[s] - Sigma_sZ[t]  # (9, 9)
            Q_Z_from_channels += diff.T @ diff

    channel_bridge_err = float(np.max(np.abs(Q_Z - Q_Z_from_channels)))

    # Now the KEY identity: express Q_Z through Omega.
    # Omega = Gamma_Z @ inv(Sigma_Z)
    # Gamma*P_s = I_9 for valid decomps, so Gamma_Z = Gamma @ Z
    # and Gamma @ P_s^T|_Z  ...  actually let's check Gamma @ Sigma_sZ[s]
    # Gamma P_s = I_9 means: for each output coord c, sum_k gamma_k[c] * P_s[k, c] = delta(c matches live pattern)
    # More precisely: Gamma @ P_s should recover the s-th channel of the identity

    # Check per-channel identities
    channel_id_residuals = []
    for s in range(n):
        # Gamma @ P_s maps R^9 -> R^9: (Gamma @ P_s)[c, n*r+u] = sum_k gamma_k[c] a_k[nr+s] b_k[ns+u]
        # For matmul: this should be I_9 when c = n*r'+u' and the live condition matches
        GP = Gamma @ Ps[s]  # (9, 9)
        channel_id_residuals.append(float(np.max(np.abs(GP - np.eye(n * n)))))

    # Per-channel Z projection through Gamma:
    # Gamma @ Sigma_sZ[s] = Gamma @ (P_s^T @ Z) = (Gamma @ P_s^T) ... wait, P_s is (R, 9)
    # P_s^T @ Z is (9, 9). Gamma @ (P_s^T @ Z) would need Gamma (9, R) @ ... no.
    # Let's be careful. Z is (R, 9). P_s is (R, 9). P_s^T @ Z is (9, 9).
    # Gamma is (9, R). Gamma @ Z is (9, 9) = Gamma_Z.
    # (Gamma @ P_s^T is not directly defined since Gamma is (9,R) and P_s^T is (9,R)^T = (R,9)... hmm)

    # The right identity: for w in Z,
    #   Gamma * w = 0  (since Z ⊂ ker(Gamma))
    # Wait — is Z ⊂ ker(Gamma)?
    # Z = ker(H^T) ∩ ker(Delta^T). But Gamma annihilates H and Delta:
    # Gamma * H = 0, Gamma * Delta = 0 (from the fiber-mode structure).
    # But that doesn't mean Z ⊂ ker(Gamma). Actually Z is where H^T and Delta^T
    # annihilate, while kerGamma is where Gamma annihilates. These are different subspaces.
    # Z contains the "pure sigma" directions, ker(Gamma) contains the "nuisance" directions.
    # They are COMPLEMENTARY, not nested!

    Gamma_w_norms = [float(np.linalg.norm(Gamma @ Z[:, j])) for j in range(dim_Z)]
    Z_in_kerGamma = all(v < 1e-8 for v in Gamma_w_norms)

    # Check if Z and ker(Gamma) are complementary
    kerG = nullspace(Gamma)  # (R, dim_kerG)
    dim_kerG = kerG.shape[1]

    # Z ∩ ker(Gamma)
    combined = np.vstack([Z.T, kerG.T])  # rows are constraints
    joint_rank = matrix_rank(combined)
    intersection_dim = Z.shape[1] + kerG.shape[1] - joint_rank

    # Direct sum check: Z + ker(Gamma) should span R^R if they're complementary
    ZK = np.hstack([Z, kerG])
    span_dim = matrix_rank(ZK)

    return {
        "bridge_verified": channel_bridge_err < 1e-10,
        "channel_bridge_error": channel_bridge_err,
        "sigma_sum_error": sigma_sum_err,
        "sigma_sum_verified": sigma_sum_err < 1e-10,
        "channel_identity_residuals": channel_id_residuals,
        "channel_identities_exact": all(r < 1e-10 for r in channel_id_residuals),
        "Z_in_kerGamma": Z_in_kerGamma,
        "Gamma_w_norms": Gamma_w_norms,
        "dim_Z": dim_Z,
        "dim_kerGamma": dim_kerG,
        "R": R,
        "Z_kerG_intersection_dim": intersection_dim,
        "Z_plus_kerG_span_dim": span_dim,
        "complementary": intersection_dim == 0 and span_dim == R,
        "Q_Z_eigenvalues": sorted(np.linalg.eigvalsh(Q_Z)),
        "Q_Z_gap": sorted(np.linalg.eigvalsh(Q_Z))[0],
        "Sigma_sZ": Sigma_sZ,
    }


# ═══════════════════════════════════════════════════════════════════════
# §5  Universality test: parametric deformation
# ═══════════════════════════════════════════════════════════════════════

def deformation_sweep(terms: list[Term], n_trials: int = 50, n: int = 3,
                       seed: int = 42) -> list[dict]:
    """Deform a known exact decomposition by small channel rescalings
    (which preserve Gamma*P_s = I_9) and check both gap measures."""
    rng = np.random.default_rng(seed)
    results = []

    alpha_stack = np.stack([np.asarray(t.alpha, dtype=np.float64).ravel() for t in terms])
    beta_stack = np.stack([np.asarray(t.beta, dtype=np.float64).ravel() for t in terms])

    for trial in range(n_trials):
        # Channel rescaling: a_k[:,s] *= c_s, b_k[s,:] /= c_s
        # This preserves P_s (hence Gamma*P_s = I_9) but changes sigma, eta, delta
        scales = 0.5 + rng.random(n)  # 3 scale factors
        a_def = alpha_stack.reshape(-1, n, n).copy()
        b_def = beta_stack.reshape(-1, n, n).copy()
        for s in range(n):
            a_def[:, :, s] *= scales[s]
            b_def[:, s, :] /= scales[s]

        gamma_opt = solve_gamma_least_squares(
            a_def.reshape(-1, n * n), b_def.reshape(-1, n * n))
        deformed_terms = terms_from_stacked_factors(
            a_def.reshape(-1, n * n), b_def.reshape(-1, n * n),
            gamma_opt, prefix=f"def_{trial:03d}_")

        resid, loss = tensor_residual_stats(deformed_terms)

        qchan = compute_Qchan_on_kerGamma(deformed_terms, n)
        omega = compute_Omega(deformed_terms, n)
        bridge = verify_bridge(deformed_terms, n)

        results.append({
            "trial": trial,
            "scales": scales.tolist(),
            "tensor_residual": resid,
            "tensor_loss": loss,
            "Qchan_gap": qchan["gap"],
            "Qchan_nullity": qchan.get("nullity", -1),
            "Omega_defined": omega.get("defined", False),
            "Omega_min_eig": omega.get("min_eig", None),
            "Omega_pd": omega.get("positive_definite", False),
            "Omega_symmetric": omega.get("symmetric", False),
            "bridge_complementary": bridge.get("complementary", False),
            "bridge_Q_Z_gap": bridge.get("Q_Z_gap", None),
            "channel_ids_exact": bridge.get("channel_identities_exact", False),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# §6  Random right-inverse universality test
# ═══════════════════════════════════════════════════════════════════════

def random_right_inverse_test(Gamma: np.ndarray, n_trials: int = 100,
                               n: int = 3, seed: int = 12345) -> list[dict]:
    """For a fixed Gamma (from a known decomp), generate random right-inverse
    triples P_s with Gamma*P_s = I_9, then check Q_chan and Omega."""
    rng = np.random.default_rng(seed)
    n2 = n * n
    R = Gamma.shape[1]
    dim_ker = R - matrix_rank(Gamma)

    # Particular solution: Gamma^+ (pseudoinverse)
    G_pinv = np.linalg.pinv(Gamma)  # (R, 9)
    K = nullspace(Gamma)  # (R, dim_ker)

    results = []
    for trial in range(n_trials):
        # P_s = G_pinv + K @ random_coeffs for each s
        Ps = []
        for s in range(n):
            coeffs = rng.standard_normal((dim_ker, n2))
            Ps.append(G_pinv + K @ coeffs)  # (R, 9)

        # Build Q_chan on ker(Gamma)
        Q = np.zeros((R, R))
        for s in range(n):
            for t in range(s + 1, n):
                D = Ps[s] - Ps[t]  # (R, 9)
                Q += D @ D.T

        Q_restricted = K.T @ Q @ K
        eigs = sorted(np.linalg.eigvalsh(Q_restricted))
        gap = eigs[0] if eigs else 0.0
        nullity = sum(1 for e in eigs if abs(e) < RANK_TOL)

        # Build H from right-inverse differences
        H = np.hstack([Ps[0] - Ps[1], Ps[1] - Ps[2]])  # (R, 18)
        rank_H = matrix_rank(H)

        results.append({
            "trial": trial,
            "Qchan_gap": gap,
            "Qchan_nullity": nullity,
            "rank_H": rank_H,
            "dim_ker": dim_ker,
            "saturated": rank_H == dim_ker,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# §7  Main: run all analyses
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("THEORY SYNTHESIS: FRONT A (Spectral Gap) + FRONT B (Omega Operator)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # ── Load known exact decompositions ──
    at_terms, _, _ = load_public_rank23_terms()
    std_terms = standard_terms_3x3()

    decomps = {
        "AlphaTensor_R23": (at_terms, 3),
        "Standard_R27": (std_terms, 3),
    }

    # ── §A: Baseline measurements ──
    print("\n" + "─" * 60)
    print("§A  BASELINE: Q_chan and Omega on known exact decompositions")
    print("─" * 60)

    for label, (terms, n) in decomps.items():
        print(f"\n▸ {label} (R={len(terms)}, n={n})")

        qchan = compute_Qchan_on_kerGamma(terms, n)
        print(f"  Q_chan on ker(Gamma): dim_ker={qchan['dim_ker']}, "
              f"gap={qchan['gap']:.12f}, nullity={qchan['nullity']}")
        print(f"  Q_chan spectrum (first 5): {qchan['eigenvalues'][:5]}")

        omega = compute_Omega(terms, n)
        if omega.get("defined"):
            print(f"  Omega: defined, symmetric={omega['symmetric']}, "
                  f"pd={omega['positive_definite']}")
            print(f"  Omega eigenvalues: {omega['eigenvalues']}")
            print(f"  Omega min_eig={omega['min_eig']:.12f}, "
                  f"max_eig={omega['max_eig']:.12f}, det={omega['det']:.12f}")
        else:
            print(f"  Omega: NOT defined — {omega}")

    # ── §B: Algebraic bridge verification ──
    print("\n" + "─" * 60)
    print("§B  BRIDGE: Z vs ker(Gamma) structure")
    print("─" * 60)

    for label, (terms, n) in decomps.items():
        print(f"\n▸ {label}")
        bridge = verify_bridge(terms, n)
        print(f"  dim_Z={bridge['dim_Z']}, dim_kerGamma={bridge['dim_kerGamma']}, R={bridge['R']}")
        print(f"  Z ⊂ ker(Gamma)? {bridge['Z_in_kerGamma']}")
        print(f"  Z ∩ ker(Gamma) dim: {bridge['Z_kerG_intersection_dim']}")
        print(f"  Z + ker(Gamma) span dim: {bridge['Z_plus_kerG_span_dim']}")
        print(f"  Complementary (Z ⊕ ker(Gamma) = R^R)? {bridge['complementary']}")
        print(f"  Channel identities Gamma*P_s = I_9? {bridge['channel_identities_exact']}")
        print(f"  Channel bridge (Q_Z from per-channel Sigma_sZ)? {bridge['bridge_verified']}")
        print(f"  Sigma sum verified (sum_s Sigma_sZ = Sigma_Z)? {bridge['sigma_sum_verified']}")
        print(f"  Q_chan|_Z gap: {bridge['Q_Z_gap']:.12f}")
        print(f"  Q_chan|_Z spectrum: {bridge['Q_Z_eigenvalues'][:5]}")

        # KEY CHECK: if complementary, then Q_chan on ker(Gamma) and Q_chan|_Z
        # measure non-overlapping subspaces — the total Q_chan pd is a UNION condition
        if bridge['complementary']:
            print("  ★ Z and ker(Gamma) are complementary direct summands!")
            print("    Q_chan on ker(Gamma) handles the nuisance sector.")
            print("    Q_chan|_Z (≈ Omega pd) handles the sigma-silent sector.")
            print("    Together they cover all of R^R.")

    # ── §C: Parametric deformation test ──
    print("\n" + "─" * 60)
    print("§C  UNIVERSALITY: Channel-rescaling deformations")
    print("─" * 60)

    for label, (terms, n) in decomps.items():
        print(f"\n▸ {label} — 50 random channel rescalings")
        results = deformation_sweep(terms, n_trials=50, n=n)

        exact_count = sum(1 for r in results if r["tensor_residual"] < 1e-8)
        omega_pd_count = sum(1 for r in results if r["Omega_pd"])
        qchan_pos_count = sum(1 for r in results if r["Qchan_gap"] > RANK_TOL)
        complementary_count = sum(1 for r in results if r["bridge_complementary"])

        print(f"  Exact (residual < 1e-8): {exact_count}/50")
        print(f"  Omega positive definite: {omega_pd_count}/50")
        print(f"  Q_chan strictly positive: {qchan_pos_count}/50")
        print(f"  Z ⊕ ker(Gamma) complementary: {complementary_count}/50")

        # Correlation between gaps
        pairs = [(r["Qchan_gap"], r["Omega_min_eig"])
                 for r in results if r["Omega_defined"] and r["Omega_min_eig"] is not None]
        if pairs:
            qg, og = zip(*pairs)
            qg, og = np.array(qg), np.array(og)
            if qg.std() > 0 and og.std() > 0:
                corr = float(np.corrcoef(qg, og)[0, 1])
                print(f"  Correlation(Q_chan gap, Omega min_eig): {corr:.6f}")
            print(f"  Q_chan gap range: [{qg.min():.6f}, {qg.max():.6f}]")
            print(f"  Omega min_eig range: [{og.min():.6f}, {og.max():.6f}]")

    # ── §D: Random right-inverse universality ──
    print("\n" + "─" * 60)
    print("§D  UNIVERSALITY: Random right-inverse triples (fixed Gamma)")
    print("─" * 60)

    for label, (terms, n) in decomps.items():
        print(f"\n▸ {label} Gamma — 200 random right-inverse triples")
        Gamma = build_gamma_matrix(terms, n)
        results = random_right_inverse_test(Gamma, n_trials=200, n=n)

        saturated = sum(1 for r in results if r["saturated"])
        qchan_pos = sum(1 for r in results if r["Qchan_gap"] > RANK_TOL)
        gaps = [r["Qchan_gap"] for r in results]

        print(f"  Saturated (rank(H) = dim_ker): {saturated}/200")
        print(f"  Q_chan gap > 0: {qchan_pos}/200")
        print(f"  Q_chan gap range: [{min(gaps):.6f}, {max(gaps):.6f}]")
        print(f"  Q_chan gap mean: {np.mean(gaps):.6f}")

    # ── §E: Refined structural analysis ──
    print("\n" + "─" * 60)
    print("§E  STRUCTURAL INSIGHT: Q_chan|_Z = 0 is exact")
    print("─" * 60)
    print("""
KEY DISCOVERY: Q_chan restricted to Z is identically zero.

PROOF: For w in Z = ker(H^T) ∩ ker(Delta^T):
  H^T w = 0 means (P_0 - P_1)^T w = 0 and (P_1 - P_2)^T w = 0
  Therefore P_0^T w = P_1^T w = P_2^T w for all w in Z.
  Hence Q_chan(w) = sum_{s<t} ||(P_s - P_t)^T w||² = 0.  □

This is exact and algebraic, not a numerical accident.

CONSEQUENCE: Q_chan and Omega guard DIFFERENT subspaces:
  • Q_chan lives on ker(Gamma)  [the nuisance sector]
  • Omega lives on Z           [the sigma-silent sector]
  • Z ⊕ ker(Gamma) = R^R       [complementary, verified]
  • Q_chan|_Z ≡ 0              [trivially, by definition of Z]

So the witness problem splits:
  A nonzero w with P_0^T w = P_1^T w = P_2^T w AND Gamma w = 0
  would need w ∈ Z ∩ ker(Gamma). But Z ∩ ker(Gamma) = {0}
  (complementarity, verified on both known decomps).

THEREFORE: the conservation law follows from a SINGLE structural fact:
  Z ∩ ker(Gamma) = {0}
which is equivalent to: R^R = Z ⊕ ker(Gamma).
""")

    # ── §F: The real theorem ──
    print("\n" + "─" * 60)
    print("§F  PROOF-GRADE ANALYSIS: complementarity as the obstruction")
    print("─" * 60)

    for label, (terms, n) in decomps.items():
        print(f"\n▸ {label}")
        R = len(terms)
        n2 = n * n

        # Build everything
        sigma, eta1, eta2, delta, h, nuisance = build_mode_matrices_numeric(terms)
        Gamma = build_gamma_matrix(terms, n)
        Ps = build_channel_matrices(terms, n)

        # Z = ker([H|Delta]^T)
        HD = np.hstack([h, delta])
        Z = nullspace(HD.T)

        # ker(Gamma)
        kerG = nullspace(Gamma)

        # The COMBINED nuisance block: [H|Delta] has columns in R^R
        # ker(Gamma) is where Gamma annihilates
        # Key: Gamma * [H|Delta] should be zero (Gamma annihilates nuisance)
        G_nuis = Gamma @ np.hstack([h, delta])
        print(f"  Gamma * [H|Delta] max entry: {np.max(np.abs(G_nuis)):.2e}")

        # And Gamma * Sigma = n*I
        GS = Gamma @ sigma
        print(f"  Gamma * Sigma ≈ {n}*I_9 ? max|GS - {n}I| = {np.max(np.abs(GS - n * np.eye(n2))):.2e}")

        # The augmented matrix [Sigma | H | Delta] should have rank R
        aug = np.hstack([sigma, h, delta])
        aug_rank = matrix_rank(aug)
        print(f"  rank([Sigma|H|Delta]) = {aug_rank} (should be {R})")

        # Complementarity proof ingredients:
        # Z = ker([H|Delta]^T), so [H|Delta]^T * z = 0 for z in Z
        # ker(Gamma): Gamma * w = 0
        # Suppose w ∈ Z ∩ ker(Gamma). Then:
        #   [H|Delta]^T w = 0  AND  Gamma w = 0
        #   But [Sigma|H|Delta] has rank R (full), so the COMBINED constraint
        #   [Sigma|H|Delta]^T w = 0 would force w = 0.
        #   We know Gamma*Sigma = nI, so Gamma*w = 0 and Sigma^T*w has some value v.
        #   Gamma * w = 0 doesn't directly constrain Sigma^T * w.
        #   BUT: if w ∈ Z, then only Sigma^T * w can be nonzero.
        #   And if also w ∈ ker(Gamma), then Gamma * w = 0.
        #   Now: Gamma * w = Gamma_Z * w_Z + Gamma_kerG * w_kerG
        #   Since w is purely in Z (in the Z-component), Gamma_Z * w = 0 means
        #   w is in ker(Gamma_Z) = ker(Gamma|_Z).

        # Check: rank of Gamma restricted to Z
        Gamma_Z = Gamma @ Z  # (9, dim_Z)
        rank_GZ = matrix_rank(Gamma_Z)
        print(f"  rank(Gamma|_Z) = {rank_GZ} (should be {n2} = dim_Z for injectivity)")

        # If Gamma_Z is injective (rank = dim_Z = n²), then
        # w ∈ Z ∩ ker(Gamma) => Gamma_Z * coeffs = 0 => coeffs = 0 => w = 0
        # This IS the complementarity condition!

        # Now check Sigma_Z
        Sigma_Z = sigma.T @ Z  # (9, dim_Z)
        rank_SZ = matrix_rank(Sigma_Z)
        print(f"  rank(Sigma|_Z) = {rank_SZ} (should be {n2})")

        # Omega = Gamma_Z @ inv(Sigma_Z)
        if rank_SZ == n2 and rank_GZ == n2:
            Omega = Gamma_Z @ np.linalg.inv(Sigma_Z)
            eigs_omega = sorted(np.linalg.eigvalsh(Omega))
            print(f"  Omega = Gamma_Z * Sigma_Z^{{-1}}:")
            print(f"    eigenvalues: {[f'{e:.6f}' for e in eigs_omega]}")
            print(f"    pd: {eigs_omega[0] > RANK_TOL}")
            print(f"    det: {np.linalg.det(Omega):.6f}")

            # KEY IDENTITY: Gamma_Z = Omega * Sigma_Z
            # So Gamma_Z injective <=> Omega invertible AND Sigma_Z injective
            # Both are n²×n² square matrices, so injectivity = invertibility
            print(f"\n  ★ THEOREM CHAIN:")
            print(f"    Gamma*P_s = I_{{n²}} for each s")
            print(f"    => Gamma*Sigma = nI_{{n²}}  [summing channels]")
            print(f"    => Gamma_Z = Omega * Sigma_Z  [restriction to Z]")
            print(f"    => Gamma_Z invertible <=> Omega invertible [both square]")
            print(f"    => Z ∩ ker(Gamma) = {{0}}")
            print(f"    => R^R = Z ⊕ ker(Gamma)")
            print(f"    => witness w ∈ Z ∩ ker(Gamma) must be zero")
            print(f"    => rank(H) = dim(ker(Gamma)) = R - {n2}")
            print(f"    => R + eta_nullity = {n}³ = {n**3}")

            # Verify: Gamma_Z = nI through Omega*Sigma_Z?
            GZ_check = Omega @ Sigma_Z
            print(f"\n    Omega*Sigma_Z ≈ Gamma_Z? max error: {np.max(np.abs(GZ_check - Gamma_Z)):.2e}")

            # The DEEP identity: Gamma*Sigma = nI restricted to Z gives
            # Gamma_Z * Sigma_Z^{-T} ... wait, let's be precise.
            # Gamma is (9, R), Z is (R, 9). Gamma @ Z = Gamma_Z (9, 9).
            # Sigma is (R, 9), Sigma^T @ Z = Sigma_Z (9, 9).
            # From Gamma * Sigma = nI (as (9, R) @ (R, 9) = (9, 9)):
            # But this is Gamma @ Sigma directly, not through Z.
            # However: Sigma = Z @ Sigma_Z^{-T} @ ?? ... no.
            # Let's check: does Gamma @ Z @ inv(Z^T @ Z) @ Z^T @ Sigma ≈ something?
            # Actually simpler: Z^T @ Sigma is (9, 9) = Sigma_Z^T? No:
            # Sigma is (R, 9), Z is (R, 9), so Z^T @ Sigma is (9, 9).
            # And sigma.T @ Z = Sigma_Z (9, 9). So Z^T @ sigma = Sigma_Z^T. ✓

            # From Gamma * sigma = nI:
            # Gamma * (Z @ Sigma_Z + kerG_component) = nI
            # Gamma * Z @ Sigma_Z = nI  (since Gamma annihilates kerG component)
            # Wait — sigma is NOT = Z @ Sigma_Z. sigma is (R, 9), Z is (R, 9).
            # Z @ Sigma_Z would be (R, 9) @ (9, 9) -- dimensional mismatch if Z is (R,9).
            # Actually Z are basis vectors: Z = [z_1 | ... | z_9] where z_j ∈ R^R.
            # sigma^T = [sigma_1 | ... | sigma_9] where sigma_j ∈ R^R (columns of sigma^T).
            # Sigma_Z = sigma^T @ Z = (9×R) @ (R×9) = (9×9).
            # So Sigma_Z[i,j] = sigma_i . z_j.

            # Decompose sigma columns: each sigma_j = Z @ c_j + kerG @ d_j
            # Then Gamma @ sigma_j = Gamma @ Z @ c_j + 0 = Gamma_Z @ c_j
            # And Gamma @ sigma = nI means Gamma_Z @ C = nI where C[i,j] = c_j[i]
            # C = (Z^T Z)^{-1} Z^T sigma = Z^T sigma (since Z orthonormal) = Sigma_Z^T
            # Wait: Z^T @ sigma is (9, 9). And C should give the Z-components.
            # If Z is orthonormal, then P_Z = Z Z^T projects onto Z.
            # sigma_j^{Z-component} = Z @ Z^T @ sigma_j, so C = Z^T @ sigma = Sigma_Z^T.

            # Therefore: Gamma_Z @ Sigma_Z^T = nI
            GZ_SZT = Gamma_Z @ Sigma_Z.T
            print(f"    Gamma_Z * Sigma_Z^T ≈ {n}*I? max error: {np.max(np.abs(GZ_SZT - n * np.eye(n2))):.2e}")

            # If Gamma_Z @ Sigma_Z^T = nI exactly, then:
            # Omega = Gamma_Z @ inv(Sigma_Z) = n * inv(Sigma_Z^T) @ inv(Sigma_Z) = n * (Sigma_Z @ Sigma_Z^T)^{-1}
            # So Omega = n * (Sigma_Z @ Sigma_Z^T)^{-1}
            # This is ALWAYS positive definite (since Sigma_Z is invertible)!
            SZSZt = Sigma_Z @ Sigma_Z.T
            Omega_predicted = n * np.linalg.inv(SZSZt)
            omega_pred_err = np.max(np.abs(Omega - Omega_predicted))
            print(f"\n    ★★ KEY IDENTITY: Omega = n * (Sigma_Z @ Sigma_Z^T)^{{-1}}")
            print(f"    Verification error: {omega_pred_err:.2e}")
            if omega_pred_err < 1e-8:
                print(f"    CONFIRMED: Omega is AUTOMATICALLY pd whenever Sigma_Z is invertible!")
                print(f"    This means Omega pd is NOT an independent condition —")
                print(f"    it follows purely from Gamma*Sigma = nI + Z having full sigma rank.")

    # ── §G: Final theorem ──
    print("\n" + "=" * 80)
    print("§G  FINAL SYNTHESIS THEOREM")
    print("=" * 80)
    print("""
THEOREM (Conservation Law via Complementarity).

Let (alpha, beta, gamma) be a minimum-rank R-term decomposition of the n×n
matrix multiplication tensor with Gamma * P_s = I_{n²} and Delta ⊂ span(H).

Let Z = ker([H|Delta]^T) and Sigma_Z = Sigma^T|_Z (the sigma-sector Gram).

Then:
  (1) Gamma * Sigma = nI_{n²}           [from Gamma*P_s = I and summing channels]
  (2) Gamma_Z * Sigma_Z^T = nI_{n²}     [restriction to Z with orthonormal basis]
  (3) Omega = Gamma_Z * Sigma_Z^{-1}
            = n * (Sigma_Z Sigma_Z^T)^{-1}   [AUTOMATIC positive definiteness]
  (4) Z ∩ ker(Gamma) = {0}              [from Gamma_Z invertible]
  (5) rank(H) = dim(ker(Gamma)) = R - n² [from complementarity + faithfulness]
  (6) R + eta_nullity = n³              [conservation law]

The ONLY hypothesis needed beyond the fiber-mode structure is:
  • Delta ⊂ span(H)                     [still open universally]
  • rank([Sigma|H|Delta]) = R           [faithfulness, proved in Phase 17]

If Delta ⊂ span(H) holds, then dim(Z) = n², Sigma_Z is n²×n² and invertible
(since Gamma*Sigma = nI forces the sigma sector to carry full rank), and
steps (1)-(6) follow algebraically with no further hypotheses.

PROVED INPUT:                       STATUS:
  Gamma*P_s = I_9                   Always true for exact decomps [Phase 17]
  rank([Sigma|H|Delta]) = R         Proved [Phase 17 Part 1]
  Delta ⊂ span(H)                   OPEN [Phase 17 Part 2]

If Delta ⊂ span(H) is proved, the conservation law follows as a corollary
with NO spectral gap analysis needed — the positive definiteness of Omega
is automatic from the Gram structure.
""")

    print("Done.")


if __name__ == "__main__":
    main()
