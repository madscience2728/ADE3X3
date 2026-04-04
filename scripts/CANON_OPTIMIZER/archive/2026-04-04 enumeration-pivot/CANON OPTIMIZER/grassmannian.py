"""
grassmannian.py — Grassmannian search with bilinear alternation.

The core insight: the search space is Gr(10, 19), not R^513.

For a FIXED K in Gr(10, 19):
  - "H columns lie in K" => K_perp.T @ H = 0  (linear in beta for fixed alpha)
  - "Delta columns lie in K" => K_perp.T @ Delta = 0  (linear in beta for fixed alpha)
  - "Gamma @ Sigma = 3I" => linear in gamma for fixed (alpha, beta)

The inner loop is ALL linear algebra. The outer search is 90-dimensional
on a compact manifold.
"""

import numpy as np
from scipy.linalg import null_space

from tensor import build_fiber_coordinates, build_decomposition, build_target_tensor
from gates import compute_jacobians_beta, compute_jacobians_alpha, solve_gamma


class GrassmannianSolver:
    """Search Gr(10, 19) with bilinear alternation in the fiber."""

    def __init__(self, R=19, ker_dim=10):
        self.R = R
        self.ker_dim = ker_dim          # dim(K) = R - 9 = 10
        self.complement_dim = R - ker_dim  # 9
        self.T = build_target_tensor()

    # ── Grassmannian primitives ────────────────────────────────────────────

    def random_K(self, rng=None):
        """Random point on Gr(ker_dim, R)."""
        rng = rng or np.random.default_rng()
        Q, _ = np.linalg.qr(rng.standard_normal((self.R, self.ker_dim)))
        return Q  # (R, ker_dim) orthonormal columns spanning K

    def K_from_H(self, H):
        """Extract K from an H matrix: K = rowspace(H) when rank(H)=10.

        The kernel K is the 10-dim subspace that H rows span.
        K_perp is the 8-dim (or 9-dim) complement where H vanishes.
        Actually: K is right-singular space of H. H is (R, 18).
        We want the 10-dim subspace of R^R such that P_K @ rows-of-terms = ...

        Wait — K lives in R^R (term space). The constraint is:
        Nuisance rows (R, 72) lie in a 10-dim subspace of R^R.
        I.e., rank(Nuisance) <= 10 and K = rowspace(Nuisance).

        For H (R, 18): rank(H) = 10 means the R rows of H span a
        10-dim subspace of R^18. But that's column space of H.T (18, R).

        Let me reconsider: the design doc says K subset R^R (R=19).
        Gamma is (9, R). ker(Gamma) = K, a 10-dim subspace of R^19.
        The constraint is: columns of H.T (and Delta.T) lie in K.

        H is (R, 18). H.T is (18, R). Each column of H.T is an R-vector.
        "Columns of H.T lie in K" means colspace(H.T) subset K.
        Equivalently: rank(H) <= dim(K) = 10, AND the left nullspace of H
        contains K_perp.

        So K_perp.T @ H = 0 means: each row i of K_perp.T (a 1xR vector)
        dotted with H (RxF) gives zeros. I.e. K_perp is in left null(H).

        For K from H: K = left column space of H, i.e. U[:, :10] from SVD.
        """
        U, s, _ = np.linalg.svd(H, full_matrices=True)
        return U[:, :self.ker_dim]  # (R, 10)

    def K_from_nuisance(self, nuisance):
        """Best rank-ker_dim subspace in term space from the nuisance block."""
        U, _, _ = np.linalg.svd(nuisance, full_matrices=True)
        return U[:, :self.ker_dim]

    def K_complement(self, K):
        """K_perp: (R, complement_dim) orthonormal basis for K^perp in R^R."""
        # K is (R, ker_dim). K_perp spans the orthogonal complement.
        # Use null_space of K.T
        Kp = null_space(K.T)  # (R, R - ker_dim)
        return Kp

    # ── Inner linear systems ──────────────────────────────────────────────

    def build_system_beta(self, K_perp, alpha):
        """Build the linear system for beta given fixed K and alpha.

        Constraints:
          K_perp.T @ H = 0   (H part)    -> complement_dim * 18 equations
          K_perp.T @ Delta = 0 (Delta)    -> complement_dim * 54 equations

        H[k, j] = A_k_H[j, :] @ beta_k   (linear in beta_k)
        Delta[k, j] = D_k[j, :] @ beta_k  (linear in beta_k)

        System: A_sys @ vec(beta) = 0  (homogeneous)

        Returns A_sys (n_eqs, R*9).
        """
        cd = K_perp.shape[1]  # complement_dim (9)
        n_H = cd * 18
        n_D = cd * 54
        n_eqs = n_H + n_D
        n_vars = self.R * 9

        A_sys = np.zeros((n_eqs, n_vars))

        for k in range(self.R):
            A_k_H, _, D_k = compute_jacobians_beta(alpha[k])
            var_start = k * 9

            # H constraint: (K_perp.T @ H)[i, j] = sum_k K_perp[k, i] * H[k, j]
            # H[k, j] = A_k_H[j, :] @ beta_k
            # So contribution of term k: K_perp[k, i] * A_k_H[j, :] @ beta_k
            for i in range(cd):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for j in range(18):
                    eq = i * 18 + j
                    A_sys[eq, var_start:var_start+9] += coeff * A_k_H[j, :]

            # Delta constraint: same structure
            for i in range(cd):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for j in range(54):
                    eq = n_H + i * 54 + j
                    A_sys[eq, var_start:var_start+9] += coeff * D_k[j, :]

        return A_sys

    def build_system_alpha(self, K_perp, beta):
        """Build the linear system for alpha given fixed K and beta.

        Same structure as build_system_beta but transposed role.
        """
        cd = K_perp.shape[1]
        n_H = cd * 18
        n_D = cd * 54
        n_eqs = n_H + n_D
        n_vars = self.R * 9

        A_sys = np.zeros((n_eqs, n_vars))

        for k in range(self.R):
            A_k_H, _, D_k = compute_jacobians_alpha(beta[k])
            var_start = k * 9

            for i in range(cd):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for j in range(18):
                    eq = i * 18 + j
                    A_sys[eq, var_start:var_start+9] += coeff * A_k_H[j, :]

            for i in range(cd):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for j in range(54):
                    eq = n_H + i * 54 + j
                    A_sys[eq, var_start:var_start+9] += coeff * D_k[j, :]

        return A_sys

    def build_sigma_system_beta(self, K_perp, alpha):
        """Build the inhomogeneous Sigma system for beta.

        Uses Gamma = K_perp.T, so the equations are:
            Gamma @ Sigma(alpha, beta) = 3 I_9.
        """
        n_eqs = self.complement_dim * 9
        n_vars = self.R * 9
        A_sys = np.zeros((n_eqs, n_vars))
        b_sys = np.zeros(n_eqs)

        for k in range(self.R):
            _, S_k, _ = compute_jacobians_beta(alpha[k])
            var_start = k * 9
            for i in range(self.complement_dim):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for f in range(9):
                    eq = i * 9 + f
                    A_sys[eq, var_start:var_start+9] += coeff * S_k[f, :]

        for i in range(self.complement_dim):
            b_sys[i * 9 + i] = 3.0

        return A_sys, b_sys

    def build_sigma_system_alpha(self, K_perp, beta):
        """Build the inhomogeneous Sigma system for alpha.

        Uses Gamma = K_perp.T, so the equations are:
            Gamma @ Sigma(alpha, beta) = 3 I_9.
        """
        n_eqs = self.complement_dim * 9
        n_vars = self.R * 9
        A_sys = np.zeros((n_eqs, n_vars))
        b_sys = np.zeros(n_eqs)

        for k in range(self.R):
            _, S_k, _ = compute_jacobians_alpha(beta[k])
            var_start = k * 9
            for i in range(self.complement_dim):
                coeff = K_perp[k, i]
                if abs(coeff) < 1e-15:
                    continue
                for f in range(9):
                    eq = i * 9 + f
                    A_sys[eq, var_start:var_start+9] += coeff * S_k[f, :]

        for i in range(self.complement_dim):
            b_sys[i * 9 + i] = 3.0

        return A_sys, b_sys

    def solve_inner_beta(self, K, alpha):
        """Solve the full structured inner system for beta by least squares.

        The system combines:
          K_perp.T @ H = 0,
          K_perp.T @ Delta = 0,
          K_perp.T @ Sigma = 3 I_9.
        """
        K_perp = self.K_complement(K)
        A_hom = self.build_system_beta(K_perp, alpha)
        b_hom = np.zeros(A_hom.shape[0])
        A_inh, b_inh = self.build_sigma_system_beta(K_perp, alpha)

        A_full = np.vstack([A_hom, A_inh])
        b_full = np.concatenate([b_hom, b_inh])

        beta_flat, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)
        total_residual = float(np.linalg.norm(A_full @ beta_flat - b_full))
        hom_residual = float(np.linalg.norm(A_hom @ beta_flat))
        inh_residual = float(np.linalg.norm(A_inh @ beta_flat - b_inh))

        return beta_flat.reshape(self.R, 3, 3), total_residual, hom_residual, inh_residual

    def solve_inner_alpha(self, K, beta):
        """Solve the full structured inner system for alpha by least squares."""
        K_perp = self.K_complement(K)
        A_hom = self.build_system_alpha(K_perp, beta)
        b_hom = np.zeros(A_hom.shape[0])
        A_inh, b_inh = self.build_sigma_system_alpha(K_perp, beta)

        A_full = np.vstack([A_hom, A_inh])
        b_full = np.concatenate([b_hom, b_inh])

        alpha_flat, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)
        total_residual = float(np.linalg.norm(A_full @ alpha_flat - b_full))
        hom_residual = float(np.linalg.norm(A_hom @ alpha_flat))
        inh_residual = float(np.linalg.norm(A_inh @ alpha_flat - b_inh))

        return alpha_flat.reshape(self.R, 3, 3), total_residual, hom_residual, inh_residual

    # ── Grassmannian gradient ─────────────────────────────────────────────

    def compute_residual_and_gradient(self, K, alpha, beta):
        """Residual and Riemannian gradient of ||K_perp.T @ Nuisance||^2_F on Gr.

        The tangent space at K is: {K_perp @ M : M in R^{complement_dim x ker_dim}}.
        """
        coords = build_fiber_coordinates(alpha, beta)
        Nuis = coords['Nuisance']  # (R, 72)

        K_perp = self.K_complement(K)
        # Residual: ||K_perp.T @ Nuis||^2_F
        R_mat = K_perp.T @ Nuis  # (complement_dim, 72)
        residual = float(np.linalg.norm(R_mat, 'fro') ** 2)

        # Riemannian gradient on Grassmannian:
        # grad = -2 * K_perp @ K_perp.T @ Nuis @ Nuis.T @ K
        # This is an (R, ker_dim) matrix in the tangent space.
        grad = -2.0 * K_perp @ (R_mat @ Nuis.T @ K)  # (R, ker_dim)

        return grad, residual

    # ── Retraction ────────────────────────────────────────────────────────

    def retract(self, K, tangent, step_size):
        """Retract from K along tangent direction on Gr(ker_dim, R).

        Uses QR retraction: orthogonalize K + step * tangent.
        """
        K_new = K + step_size * tangent
        Q, _ = np.linalg.qr(K_new)
        return Q[:, :self.ker_dim]

    # ── Diagnostics ───────────────────────────────────────────────────────

    def diagnostics(self, alpha, beta, gamma):
        """Full diagnostic dict."""
        coords = build_fiber_coordinates(alpha, beta)
        H = coords['H']; Delta = coords['Delta']
        Nuisance = coords['Nuisance']; Sigma = coords['Sigma']

        rank_H = int(np.linalg.matrix_rank(H, tol=1e-10))
        rank_N = int(np.linalg.matrix_rank(Nuisance, tol=1e-10))
        aug_rank = int(np.linalg.matrix_rank(np.hstack([Sigma, Nuisance]), tol=1e-10))

        Mls, _, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
        delta_resid = float(np.linalg.norm(Delta - H @ Mls, 'fro'))

        gf = gamma.reshape(self.R, 9)
        gam_delta = float(np.linalg.norm(gf.T @ Delta, 'fro'))

        T_hat = build_decomposition(alpha, beta, gamma)
        fitness = float(np.max(np.abs(self.T - T_hat)))

        return {
            'fitness': fitness, 'rank_H': rank_H, 'rank_nuisance': rank_N,
            'delta_residual': delta_resid, 'gamma_delta': gam_delta,
            'augmented_rank': aug_rank, 'conservation': self.R + (18 - rank_H),
        }

    # ── Main search ───────────────────────────────────────────────────────

    def search(self, max_outer=500, max_inner=20, lr=0.01,
               seed=42, verbose=True, fitness_tol=1e-12,
               warm_start=None):
        """Main Grassmannian search loop.

        Outer loop: gradient descent on Gr(10, 19).
        Inner loop: bilinear alternation with exact linear solves.
        """
        import time
        t0 = time.time()
        rng = np.random.default_rng(seed)

        # Initialize
        if warm_start is not None:
            alpha = warm_start['alpha'].copy()
            beta = warm_start['beta'].copy()
            gamma = warm_start.get('gamma', np.zeros((self.R, 3, 3)))
            coords = build_fiber_coordinates(alpha, beta)
            K = self.K_from_nuisance(coords['Nuisance'])
        else:
            alpha = rng.normal(0, 0.3, (self.R, 3, 3))
            beta = rng.normal(0, 0.3, (self.R, 3, 3))
            coords = build_fiber_coordinates(alpha, beta)
            K = self.K_from_nuisance(coords['Nuisance'])
            gamma = np.zeros((self.R, 3, 3))

        best = {'fitness': float('inf')}
        history = []

        for outer in range(max_outer):
            # ── Inner loop: bilinear alternation with K fixed ──
            for inner in range(max_inner):
                beta, beta_total_res, beta_hom_res, beta_inh_res = self.solve_inner_beta(K, alpha)
                alpha, alpha_total_res, alpha_hom_res, alpha_inh_res = self.solve_inner_alpha(K, beta)

                # Solve gamma
                coords = build_fiber_coordinates(alpha, beta)
                Gamma, gamma_res = solve_gamma(coords['Sigma'], coords['Nuisance'])
                gamma = Gamma.T.reshape(self.R, 3, 3)

            # ── Outer step: update K on Grassmannian ──
            grad, K_residual = self.compute_residual_and_gradient(K, alpha, beta)
            K = self.retract(K, grad, lr)  # grad already has sign

            # Also: re-extract K from current H (hybrid approach)
            coords = build_fiber_coordinates(alpha, beta)
            K_from_data = self.K_from_nuisance(coords['Nuisance'])
            # Blend: mostly data-driven K, slightly gradient-corrected
            K_blend = 0.7 * K_from_data + 0.3 * K
            Q, _ = np.linalg.qr(K_blend)
            K = Q[:, :self.ker_dim]

            # ── Diagnostics ──
            d = self.diagnostics(alpha, beta, gamma)
            d['K_residual'] = K_residual
            d['beta_total_residual'] = beta_total_res
            d['beta_hom_residual'] = beta_hom_res
            d['beta_inh_residual'] = beta_inh_res
            d['alpha_total_residual'] = alpha_total_res
            d['alpha_hom_residual'] = alpha_hom_res
            d['alpha_inh_residual'] = alpha_inh_res
            history.append({'outer': outer, **d})

            if verbose and outer % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"outer={outer:04d}  fit={d['fitness']:.5f}  "
                    f"rk(H)={d['rank_H']:2d}  rk(N)={d['rank_nuisance']:2d}  "
                    f"delta={d['delta_residual']:.4f}  "
                    f"K_res={K_residual:.4f}  "
                    f"b_res={beta_total_res:.4f}  a_res={alpha_total_res:.4f}  "
                    f"C={d['conservation']}  {elapsed:.1f}s"
                )

            if d['fitness'] < best['fitness']:
                best = {
                    'alpha': alpha.copy(), 'beta': beta.copy(),
                    'gamma': gamma.copy(), 'fitness': d['fitness'],
                    'diagnostics': d,
                }

            if d['fitness'] < fitness_tol and d['conservation'] == 27:
                if verbose:
                    print(f"\n*** SUCCESS at outer={outer}, fit={d['fitness']:.4e} ***")
                best['success'] = True
                best['history'] = history
                return best

            # Adaptive learning rate
            if outer > 0 and outer % 50 == 0:
                recent = [h['fitness'] for h in history[-10:]]
                if len(recent) >= 2 and max(recent) - min(recent) < 1e-8:
                    lr *= 0.5
                    if verbose:
                        print(f"  lr -> {lr:.6f} (stalled)")

        best['success'] = False
        best['history'] = history
        return best

