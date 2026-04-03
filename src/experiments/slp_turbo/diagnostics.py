"""Structural diagnostics from the canonical object.

Computes fiber-mode health metrics on a candidate x (513,):
  - live_maxabs / dead_maxabs: tier-split fitness
  - dead_energy: total dead-entry squared residual
  - dead_cancellation: ratio of combined vs individual dead energies
  - kappa_ker: sigma_min(Q_Gamma^T H) — kernel-saturation health
  - nuisance_rank: rank of the H = [Eta1|Eta2] block
"""

import numpy as np
from .config import R, D, N_ENTRIES, N_PARAMS, T_NP, LIVE_IDX, DEAD_IDX


def _reconstruct(x):
    """Return (a, b, g, T_hat_flat, residual_flat)."""
    a = x[:R*D].reshape(R, D)
    b = x[R*D:2*R*D].reshape(R, D)
    g = x[2*R*D:].reshape(R, D)
    T_hat = np.einsum('ra,rb,rc->abc', a, b, g).ravel()
    res = T_hat - T_NP.ravel()
    return a, b, g, T_hat, res


def tier_fitness(x):
    """Return (live_maxabs, dead_maxabs, overall_maxabs)."""
    _, _, _, _, res = _reconstruct(x)
    abs_res = np.abs(res)
    return float(abs_res[LIVE_IDX].max()), float(abs_res[DEAD_IDX].max()), float(abs_res.max())


def dead_energy_stats(x):
    """Return (dead_energy_combined, dead_energy_sum_individual, cancel_ratio).

    cancel_ratio < 1 means destructive interference is working.
    """
    a, b, g, _, _ = _reconstruct(x)
    # Each rank-1 term's contribution to dead entries
    dead_sq_individual = 0.0
    terms = []
    for k in range(R):
        tk = np.einsum('a,b,c->abc', a[k], b[k], g[k]).ravel()
        terms.append(tk)
        dead_sq_individual += np.sum(tk[DEAD_IDX] ** 2)

    # Combined dead residual
    combined = sum(terms)
    dead_sq_combined = np.sum(combined[DEAD_IDX] ** 2)

    ratio = dead_sq_combined / max(dead_sq_individual, 1e-30)
    return float(dead_sq_combined), float(dead_sq_individual), float(ratio)


def _build_gamma_sigma(a, b, g):
    """Build Gamma (9×R) and Sigma (R×9) for the fiber-mode decomposition.

    Gamma[u, k] = g[k, u]  (output-coordinate factor)
    Sigma[k, f] = sum_s a[k, 3*f_row + s] * b[k, 3*s + f_col]
        where f indexes the 9 output fibers (row, col) and s is the summation index.

    Actually the exact construction from the doc:
    For 3×3 matmul, the fiber-sum equation for output fiber (i,j) is:
      sum_k gamma_k[3i+j_col?] ... this requires the exact encoding.

    Simpler approach: Gamma = g^T (9×R), and Sigma is the 'live product' P_s matrices
    stacked. But for the kernel-saturation diagnostic we only need Gamma and H.

    Gamma = g.T  (9 × R)
    For each term k, the fiber-mode 81-vector is:
      sigma[k, (i,j)] = sum_s a[k, 3i+s] * b[k, 3s+j]  for output fiber (i,j)
    """
    # Gamma = g.T  (9 × R)
    Gamma = g.T  # (D, R) = (9, 19)

    # Sigma: for each term k and output fiber f=(i,j), i=f//3, j=f%3
    # sigma[k,f] = sum_{s=0}^{2} a[k, 3*i + s] * b[k, 3*s + j]
    # This is the live product for the combined summation channel
    Sigma = np.zeros((R, D))  # (19, 9)
    for f in range(D):
        i, j = f // 3, f % 3
        for s in range(3):
            Sigma[:, f] += a[:, 3*i + s] * b[:, 3*s + j]

    return Gamma, Sigma


def _build_nuisance_H(a, b, g):
    """Build H = [Eta1 | Eta2] nuisance block (R × 18).

    Eta1[k, f] = sigma_s0[k,f] - sigma_s1[k,f]
    Eta2[k, f] = sigma_s1[k,f] - sigma_s2[k,f]
    where sigma_s[k,f] = a[k, 3*i + s_inner] * b[k, 3*s + j]
    Wait — need to be careful about what 's' means in the fiber mode.

    From the doc: the fiber-sum is the sum over the 3 summation indices.
    The anisotropy is the within-fiber differences.
    For output fiber (i,j):
      P_s[k, f] = a[k, 3*i + ...] * b[k, 3*s + j]  for each summation channel s

    Actually, looking at the live entries more carefully:
    T[a,b,c]=1 when a//3==c//3, a%3==b//3, b%3==c%3
    So for output fiber (i,j) where i=a//3=c//3, j=b%3=c%3:
      summation index s = a%3 = b//3
      P_s[k, (i,j)] = alpha_k[3i+s] * beta_k[3s+j]  ... but also involves gamma

    The full fiber-mode decomposition involves gamma. Let me use the simpler
    formulation: Gamma = g.T, and the nuisance block is the kernel complement.
    """
    Gamma = g.T  # (9, R)

    # Per-channel live products P_s: for summation channel s=0,1,2
    P = np.zeros((3, R, 9))  # P[s, k, f]
    for s in range(3):
        for f in range(9):
            i, j = f // 3, f % 3
            P[s, :, f] = a[:, 3*i + s] * b[:, 3*s + j]

    # H = [P_0 - P_1 | P_1 - P_2]  (R × 18)
    Eta1 = P[0] - P[1]  # (R, 9)
    Eta2 = P[1] - P[2]  # (R, 9)
    H = np.hstack([Eta1, Eta2])  # (R, 18)
    return Gamma, H


def kappa_ker_diagnostic(x):
    """Compute kernel-saturation health: sigma_min(Q^T H).

    Returns (kappa_ker, nuisance_rank, ker_dim).
    kappa_ker > 0 means the nuisance block saturates ker(Gamma) — healthy.
    kappa_ker → 0 means drifting toward a structurally defective locus.
    """
    a = x[:R*D].reshape(R, D)
    b = x[R*D:2*R*D].reshape(R, D)
    g = x[2*R*D:].reshape(R, D)

    Gamma, H = _build_nuisance_H(a, b, g)

    # ker(Gamma): null space of Gamma (9 × R)
    # Gamma has rank 9 (for a good decomposition), so ker has dim R-9 = 10
    U, s, Vt = np.linalg.svd(Gamma, full_matrices=True)
    # Kernel of Gamma = last R-9 rows of Vt
    ker_dim = R - min(R, 9)  # should be 10
    tol = 1e-10
    actual_rank = np.sum(s > tol)
    actual_ker_dim = R - actual_rank

    if actual_ker_dim == 0:
        return 0.0, 0, 0

    Q = Vt[actual_rank:].T  # (R, ker_dim) orthonormal basis of ker(Gamma)

    # Project H into ker(Gamma)
    QH = Q.T @ H  # (ker_dim, 18)
    sv = np.linalg.svd(QH, compute_uv=False)
    nuisance_rank = int(np.sum(sv > tol))
    kappa = float(sv[-1]) if len(sv) > 0 else 0.0

    return kappa, nuisance_rank, actual_ker_dim


def full_diagnostic(x):
    """Run all diagnostics, return a dict."""
    live_fit, dead_fit, overall_fit = tier_fitness(x)
    dead_comb, dead_indiv, cancel_ratio = dead_energy_stats(x)
    kappa, nuis_rank, ker_dim = kappa_ker_diagnostic(x)

    return {
        "live_maxabs": live_fit,
        "dead_maxabs": dead_fit,
        "overall_maxabs": overall_fit,
        "dead_energy": dead_comb,
        "dead_energy_individual": dead_indiv,
        "cancel_ratio": cancel_ratio,
        "kappa_ker": kappa,
        "nuisance_rank": nuis_rank,
        "ker_dim": ker_dim,
    }
