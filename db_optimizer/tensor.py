"""CPU-side tensor operations: reconstruction, fitness, Frobenius residual,
and structural diagnostics from the canonical document."""

import numpy as np

from .config import TARGET_TENSOR, DIM, RANK

# ── Precompute live/dead masks ───────────────────────────────────────
# Live entries: T[a,b,c] != 0 (27 entries for 3×3 matmul)
# Dead entries: T[a,b,c] == 0 (702 entries)
_LIVE_MASK: np.ndarray = (TARGET_TENSOR != 0)  # (9,9,9) bool
_DEAD_MASK: np.ndarray = (TARGET_TENSOR == 0)  # (9,9,9) bool
N_LIVE = int(_LIVE_MASK.sum())   # 27
N_DEAD = int(_DEAD_MASK.sum())   # 702


def reconstruct(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Reconstruct 9×9×9 tensor from factor matrices via CP decomposition.

    alpha, beta, gamma: each (R, 9) where R is the number of rank-1 terms.
    """
    return np.einsum('ra,rb,rc->abc', alpha, beta, gamma, optimize=True)


def residual(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
             target: np.ndarray | None = None) -> np.ndarray:
    """Compute element-wise residual: reconstructed - target."""
    if target is None:
        target = TARGET_TENSOR
    return reconstruct(alpha, beta, gamma) - target


def fitness_maxabs(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                   target: np.ndarray | None = None) -> float:
    """Compute max-abs fitness (minimax objective)."""
    return float(np.max(np.abs(residual(alpha, beta, gamma, target))))


def fitness_frobenius(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                      target: np.ndarray | None = None) -> float:
    """Compute Frobenius norm of the residual."""
    return float(np.linalg.norm(residual(alpha, beta, gamma, target)))


# ── §80 Constraint 1: Dead vs Live energy split ─────────────────────

def dead_live_energy(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Compute energy split between live (27) and dead (702) tensor entries.

    Returns dict with keys: live_energy, dead_energy, live_maxabs, dead_maxabs,
    total_fro2 (= live_energy + dead_energy).
    """
    R = residual(alpha, beta, gamma)
    R2 = R ** 2
    live_e = float(R2[_LIVE_MASK].sum())
    dead_e = float(R2[_DEAD_MASK].sum())
    live_mx = float(np.max(np.abs(R[_LIVE_MASK])))
    dead_mx = float(np.max(np.abs(R[_DEAD_MASK])))
    return {
        "live_energy": live_e,
        "dead_energy": dead_e,
        "live_maxabs": live_mx,
        "dead_maxabs": dead_mx,
        "total_fro2": live_e + dead_e,
    }


# ── §80 Constraint 5: Pythagorean diagnostic ────────────────────────

def pythagorean_diagnostic(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Compute ||R||² and ||T_hat||² where T_hat = reconstructed tensor.

    At every ALS local minimum: ||R||² + ||T_hat||² = 27.
    At the 0.5 basin: ||R||² ≈ 8 = 27 - 19.
    """
    T_hat = reconstruct(alpha, beta, gamma)
    R = T_hat - TARGET_TENSOR
    R_fro2 = float(np.sum(R ** 2))
    T_hat_fro2 = float(np.sum(T_hat ** 2))
    return {
        "R_fro2": R_fro2,
        "T_hat_fro2": T_hat_fro2,
        "sum": R_fro2 + T_hat_fro2,
        "theory": 27.0,
    }


# ── Block-level Pythagorean: per-fiber energy decomposition ─────────

def block_energy_diagnostic(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Per-output-fiber Pythagorean decomposition.

    The 9×9×9 tensor has 9 output fibers C[r,u] (c = 3r+u).
    Each fiber slice T[:,:,c] has ||T[:,:,c]||² = 3 (exactly 3 ones per output column).

    Pythagorean per fiber: ||R[:,:,c]||² + ||T_hat[:,:,c]||² = ||T[:,:,c]||² = 3

    Signal energy = ||T_hat[:,:,c]||² captured by reconstruction
    Residual energy = ||R[:,:,c]||² = what's missing
    Leakage = energy in dead entries of T_hat[:,:,c] (should be zero)

    Also decomposes into live/dead per fiber and computes the p0/p1/p2
    basis projection of the signal per fiber (to see which mode is deficient).
    """
    n = 3
    T_hat = reconstruct(alpha, beta, gamma)
    R = T_hat - TARGET_TENSOR

    # p-basis vectors (unnormalized, for projection)
    p0 = np.array([1, 1, 1]) / np.sqrt(3)
    p1 = np.array([1, -1, 0]) / np.sqrt(2)
    p2 = np.array([1, 1, -2]) / np.sqrt(6)

    fibers = {}
    for r in range(n):
        for u in range(n):
            c = n * r + u
            T_slice = TARGET_TENSOR[:, :, c]    # (9,9) — target fiber
            Th_slice = T_hat[:, :, c]            # (9,9) — reconstructed fiber
            R_slice = R[:, :, c]                 # (9,9) — residual fiber

            # Energy accounting
            target_e = float(np.sum(T_slice ** 2))      # should be 3.0
            signal_e = float(np.sum(Th_slice ** 2))
            resid_e = float(np.sum(R_slice ** 2))

            # Live/dead split within this fiber
            live_mask_c = _LIVE_MASK[:, :, c]
            dead_mask_c = _DEAD_MASK[:, :, c]
            live_resid = float(np.sum(R_slice[live_mask_c] ** 2))
            dead_resid = float(np.sum(R_slice[dead_mask_c] ** 2))
            dead_signal = float(np.sum(Th_slice[dead_mask_c] ** 2))  # leakage into dead

            # Max-abs per fiber
            maxabs = float(np.max(np.abs(R_slice)))
            live_maxabs = float(np.max(np.abs(R_slice[live_mask_c]))) if live_mask_c.any() else 0.0
            dead_maxabs = float(np.max(np.abs(R_slice[dead_mask_c]))) if dead_mask_c.any() else 0.0

            # Gamma vector for this fiber: gamma[:, c]
            g_vec = gamma[:, c]  # (R,)
            g_p0 = float(np.dot(g_vec, np.repeat(p0, RANK // n + 1)[:RANK]))
            # Actually, project gamma per-term through the fiber-mode decomposition
            # More useful: per-term signal contribution
            # Sigma_k[r,u] = sum_s alpha_k[3r+s] * beta_k[3s+u]
            sigma_per_term = np.array([
                sum(alpha[k, n*r+s] * beta[k, n*s+u] for s in range(n))
                for k in range(alpha.shape[0])
            ])
            # Per-term contribution to this fiber: sigma_k * gamma_k[c]
            contrib = sigma_per_term * g_vec
            # Target per fiber: T[:,:,c] has 3 ones at positions (3r'+s, 3s+u) for s=0,1,2
            # but contracted through gamma → Σ_k sigma_k[r,u] * gamma_k[c] should = I[r,u]*3
            # Actually the target is delta(fiber_idx == c_idx) * 3 but let's just use the
            # total signal = sum of contributions
            total_contrib = float(np.sum(contrib))
            # Imbalance: how unevenly terms contribute
            contrib_std = float(np.std(contrib))

            fibers[f"C[{r},{u}]"] = {
                "target_energy": target_e,
                "signal_energy": signal_e,
                "residual_energy": resid_e,
                "pythagorean_check": resid_e + signal_e,   # should ≈ target_e + cross terms
                "live_residual_energy": live_resid,
                "dead_residual_energy": dead_resid,
                "dead_leakage_energy": dead_signal,         # energy wasted in dead entries
                "maxabs": maxabs,
                "live_maxabs": live_maxabs,
                "dead_maxabs": dead_maxabs,
                "sigma_gamma_sum": total_contrib,           # should be 3.0 for diagonal
                "contrib_std": contrib_std,                 # imbalance across terms
            }

    # Cross-fiber leakage matrix: how much each fiber's dead energy
    # could be "donated" to another fiber's live deficit
    leakage_matrix = np.zeros((n * n, n * n))
    for c1 in range(n * n):
        for c2 in range(n * n):
            if c1 == c2:
                continue
            # T_hat dead entries in fiber c1 that share (a,b) positions
            # with live entries in fiber c2 — this is the interference coupling
            dead_c1 = _DEAD_MASK[:, :, c1]
            live_c2 = _LIVE_MASK[:, :, c2]
            shared = dead_c1 & live_c2
            if shared.any():
                leak = float(np.sum(T_hat[:, :, c1][shared] ** 2))
                leakage_matrix[c1, c2] = leak

    return {
        "fibers": fibers,
        "leakage_matrix": leakage_matrix,
        "leakage_matrix_labels": [f"C[{i//n},{i%n}]" for i in range(n*n)],
    }


def interference_flow(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Compute per-term interference flow between fiber blocks.

    For each rank-1 term k, computes:
    - signal[k,c]: energy deposited in live entries of fiber c
    - leak[k,c]: energy deposited in dead entries of fiber c
    - The "interference budget": how each term distributes its energy

    A well-structured decomposition has terms that are "focused" (high signal,
    low leak). A stuck decomposition has terms that spray energy everywhere.

    Also computes the inter-fiber coupling: which pairs of fibers
    share the most interference from the same terms (phase coupling).
    """
    n = 3
    R_terms = alpha.shape[0]

    signal = np.zeros((R_terms, n * n))   # live energy per term per fiber
    leak = np.zeros((R_terms, n * n))     # dead energy per term per fiber

    for k in range(R_terms):
        # Rank-1 contribution: alpha_k ⊗ beta_k ⊗ gamma_k
        term_k = np.einsum('a,b,c->abc', alpha[k], beta[k], gamma[k])
        for c in range(n * n):
            fiber_slice = term_k[:, :, c]
            signal[k, c] = float(np.sum(fiber_slice[_LIVE_MASK[:, :, c]] ** 2))
            leak[k, c] = float(np.sum(fiber_slice[_DEAD_MASK[:, :, c]] ** 2))

    # Per-term total energy and signal efficiency
    total_per_term = signal.sum(axis=1) + leak.sum(axis=1)  # (R,)
    efficiency = signal.sum(axis=1) / np.maximum(total_per_term, 1e-30)  # (R,)

    # Per-fiber: which terms contribute most signal, most leak
    fiber_signal_total = signal.sum(axis=0)  # (9,)
    fiber_leak_total = leak.sum(axis=0)      # (9,)

    # Phase coupling matrix: for each fiber pair (c1, c2), sum of
    # |leak[k,c1] * leak[k,c2]| across terms — terms that leak to
    # both c1 and c2 are "phase coupled" (fixing one affects the other)
    coupling = np.zeros((n * n, n * n))
    for c1 in range(n * n):
        for c2 in range(c1 + 1, n * n):
            # Correlation of leak patterns
            coupling[c1, c2] = float(np.sum(leak[:, c1] * leak[:, c2]))
            coupling[c2, c1] = coupling[c1, c2]

    # Identify "spray" terms (high leak, low efficiency) vs "focused" terms
    spray_threshold = 0.5
    spray_terms = [int(k) for k in range(R_terms) if efficiency[k] < spray_threshold]
    focused_terms = [int(k) for k in range(R_terms) if efficiency[k] >= spray_threshold]

    return {
        "signal": signal,             # (R, 9)
        "leak": leak,                 # (R, 9)
        "total_per_term": total_per_term.tolist(),
        "efficiency": efficiency.tolist(),
        "fiber_signal": fiber_signal_total.tolist(),
        "fiber_leak": fiber_leak_total.tolist(),
        "coupling_matrix": coupling,  # (9, 9) phase coupling
        "spray_terms": spray_terms,
        "focused_terms": focused_terms,
        "labels": [f"C[{i//n},{i%n}]" for i in range(n*n)],
    }

def fiber_mode_decomposition(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Compute the Step 51 fiber-mode matrices Sigma, Eta1, Eta2, Delta, Gamma.

    For n=3 matrix multiplication, each rank-1 term k has:
      alpha_k reshaped as (3,3): rows indexed by (r, s)
      beta_k  reshaped as (3,3): rows indexed by (s, t) → cols (t, u)

    Sigma[k, (r,u)] = sum_s a_k[r,s] * b_k[s,u]    (fiber sum)
    Eta1[k, (r,u)]  = a_k[r,0]*b_k[0,u] - a_k[r,1]*b_k[1,u]
    Eta2[k, (r,u)]  = a_k[r,1]*b_k[1,u] - a_k[r,2]*b_k[2,u]
    Delta[k, ...]   = dead-X coordinates (a_k[r,s]*b_k[t,u] for s!=t)
    Gamma[c, k]      = gamma_k[c]  (output weight)

    Returns dict with Sigma, Eta1, Eta2, Delta, Gamma matrices and ranks.
    """
    n = 3
    R = alpha.shape[0]

    # Reshape factor matrices to (R, 3, 3)
    A = alpha.reshape(R, n, n)  # A[k, r, s]
    B = beta.reshape(R, n, n)   # B[k, s_idx, u] — but beta index is (s*n+t) mapped to (n*s+u)
    # Actually beta is indexed as b_k[n*s+u] in flat form, reshape to (R, n, n) gives B[k,s,u]
    # Wait — need to be careful. The tensor T[a,b,c] where a=n*r+s, b=n*s'+u, c=n*r'+u'
    # but for matmul: T[n*r+s, n*s+u, n*r+u] = 1
    # So alpha_k has indices in the 'a' dimension: a = n*r + s
    # beta_k has indices in the 'b' dimension: b = n*s + u  (note: s here contracts with a's s)
    # gamma_k has indices in the 'c' dimension: c = n*r' + u'

    # For fiber-mode: Sigma_k[r,u] = sum_s alpha_k[n*r+s] * beta_k[n*s+u]
    Sigma = np.zeros((R, n * n))
    Eta1 = np.zeros((R, n * n))
    Eta2 = np.zeros((R, n * n))

    for k in range(R):
        for r in range(n):
            for u in range(n):
                fiber_idx = n * r + u
                # sigma = sum_s a[n*r+s] * b[n*s+u]
                s_vals = [alpha[k, n * r + s] * beta[k, n * s + u] for s in range(n)]
                Sigma[k, fiber_idx] = sum(s_vals)
                Eta1[k, fiber_idx] = s_vals[0] - s_vals[1]
                Eta2[k, fiber_idx] = s_vals[1] - s_vals[2]

    # Delta: dead-X coordinates — a[n*r+s]*b[n*t+u] for s != t
    dead_coords = []
    for r in range(n):
        for s in range(n):
            for t in range(n):
                if s != t:
                    for u in range(n):
                        dead_coords.append((r, s, t, u))

    Delta = np.zeros((R, len(dead_coords)))
    for k in range(R):
        for j, (r, s, t, u) in enumerate(dead_coords):
            Delta[k, j] = alpha[k, n * r + s] * beta[k, n * t + u]

    # Gamma: (9, R) — output weight matrix
    Gamma = gamma.T.copy()  # gamma is (R, 9), Gamma is (9, R)

    # Compute ranks
    sigma_rank = int(np.linalg.matrix_rank(Sigma, tol=1e-10))
    nuisance = np.hstack([Eta1, Eta2, Delta])  # (R, 9+9+54) = (R, 72)
    nuisance_rank = int(np.linalg.matrix_rank(nuisance, tol=1e-10))
    gamma_rank = int(np.linalg.matrix_rank(Gamma, tol=1e-10))
    gamma_nullity = R - gamma_rank

    # Check Gamma * Sigma ≈ 3 I_9
    GS = Gamma @ Sigma  # (9, 9)
    gs_residual = float(np.max(np.abs(GS - 3.0 * np.eye(n * n))))

    return {
        "Sigma": Sigma,
        "Eta1": Eta1,
        "Eta2": Eta2,
        "Delta": Delta,
        "Gamma": Gamma,
        "sigma_rank": sigma_rank,
        "nuisance_rank": nuisance_rank,
        "gamma_rank": gamma_rank,
        "gamma_nullity": gamma_nullity,
        "gs_residual": gs_residual,
    }


# ── §61 Constraint 4: Nuisance rank check ───────────────────────────

def check_nuisance_feasible(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                            max_nuisance_rank: int = 10) -> tuple[bool, int]:
    """Check if candidate's nuisance rank is within the R=19 budget.

    At R=19, the Step 52 bound requires rank(Nuisance) ≤ 10.
    Returns (is_feasible, nuisance_rank).
    """
    fm = fiber_mode_decomposition(alpha, beta, gamma)
    return fm["nuisance_rank"] <= max_nuisance_rank, fm["nuisance_rank"]


# ── Constraint 6: Support sparsity ──────────────────────────────────

def support_sparsity(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                     tol: float = 1e-8) -> dict:
    """Compute support signature: count nonzeros per column of each factor.

    The best step84 cold-start used support (2,3,3) — 2-3 nonzeros per column.
    """
    a_nnz = np.sum(np.abs(alpha) > tol, axis=1)  # (R,)
    b_nnz = np.sum(np.abs(beta) > tol, axis=1)
    g_nnz = np.sum(np.abs(gamma) > tol, axis=1)
    total_nnz = int(np.sum(np.abs(alpha) > tol) + np.sum(np.abs(beta) > tol) +
                    np.sum(np.abs(gamma) > tol))
    total_possible = 3 * RANK * DIM
    return {
        "alpha_nnz_per_term": a_nnz.tolist(),
        "beta_nnz_per_term": b_nnz.tolist(),
        "gamma_nnz_per_term": g_nnz.tolist(),
        "mean_nnz": float(np.mean(np.concatenate([a_nnz, b_nnz, g_nnz]))),
        "total_nnz": total_nnz,
        "total_possible": total_possible,
        "sparsity": 1.0 - total_nnz / total_possible,
    }
