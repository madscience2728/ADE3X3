"""
Constrained search for rank-21 (and other ranks) with soft gate penalties.

Key finding: unconstrained optimizer finds solutions where:
  - Gate 2 (delta_leak=0): softly violated — ||P_{H⊥} Δ||_F drives Frobenius
  - Gate 3 (aug_gap=9):    hard-violated  — aug_gap=0 in all found solutions

Correlation:  rank22 ||Δ_⊥H|| = 0.0035 → frob = 0.0076
              rank21 ||Δ_⊥H|| = 0.058  → frob = 0.090

Strategy: add λ₂ * ||P_{H⊥} Δ||² and λ₃ * (9 - aug_gap_soft)² to objective.

The Gate-2 penalty is differentiable via:
  P_{H⊥} Δ = Δ - H (Hᵀ H + εI)⁻¹ Hᵀ Δ
  gradient flows through both H and Δ (both depend on α, β).

Gate-3 soft proxy: maximize σ_min of the projection of Σ onto [H|Δ]-perp,
  approximated as a log-det on the 9×9 Gram matrix.
"""
import numpy as np
import json, os, time
from scipy.optimize import minimize
from multiprocessing import Pool

# ── Target tensor ─────────────────────────────────────────────
T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0
T_FLAT = T.ravel()
DIM = 9

DEAD_PAIRS = [(s, t) for s in range(3) for t in range(3) if s != t]  # 6 pairs

CKPT_DIR = 'CANON_ADE_EPSILON/results/gated_search'
os.makedirs(CKPT_DIR, exist_ok=True)


# ── Step-51 decomposition (differentiable) ────────────────────
def step51(A3, B3, R):
    """
    Compute Σ, H, Δ from α (R,3,3) and β (R,3,3).

    Σ[k, r*3+u]  = Σ_s α[k,r,s] β[k,s,u]         — (R, 9)
    Eta1[k,r*3+u] = α[k,r,0]β[k,0,u]-α[k,r,1]β[k,1,u]
    Eta2[k,r*3+u] = α[k,r,1]β[k,1,u]-α[k,r,2]β[k,2,u]
    H = [Eta1 | Eta2]                               — (R, 18)
    Δ[k, i]  = α[k,r,s] β[k,t,u]  for s≠t         — (R, 54)
    """
    Sigma = np.einsum('krs,ksu->kru', A3, B3).reshape(R, 9)
    e0 = (A3[:, :, 0:1] * B3[:, 0:1, :]).reshape(R, 9)
    e1 = (A3[:, :, 1:2] * B3[:, 1:2, :]).reshape(R, 9)
    e2 = (A3[:, :, 2:3] * B3[:, 2:3, :]).reshape(R, 9)
    H = np.hstack([e0 - e1, e1 - e2])               # (R, 18)
    Delta = np.hstack(
        [(A3[:, :, s:s+1] * B3[:, t:t+1, :]).reshape(R, 9)
         for s, t in DEAD_PAIRS]
    )                                                 # (R, 54)
    return Sigma, H, Delta, e0, e1, e2


def gate2_penalty(H, Delta, reg=1e-6):
    """
    Soft Gate-2: ||P_{H⊥} Δ||_F²
    P_{H⊥} Δ = Δ - H (HᵀH + reg·I)⁻¹ Hᵀ Δ

    Returns (penalty_value, grad_H, grad_Delta).
    """
    HtH = H.T @ H + reg * np.eye(H.shape[1])   # (18,18)
    HtH_inv = np.linalg.solve(HtH, np.eye(18))  # (18,18)
    # Projection: P_H Δ = H (HᵀH)⁻¹ Hᵀ Δ
    coef    = HtH_inv @ (H.T @ Delta)           # (18, 54)
    P_H_D   = H @ coef                          # (R, 54) — in-H component
    D_perp  = Delta - P_H_D                     # (R, 54) — leakage

    pen = 0.5 * float(np.sum(D_perp ** 2))

    # Gradient w.r.t. Delta: ∂pen/∂Δ = P_{H⊥} Δ_perp = D_perp
    # (chain rule: d/dΔ ||Δ - H coef(Δ)||² where coef depends on Δ)
    # Full gradient: 2*(I - P_H)*(Δ - P_H Δ) ... = D_perp (P_H⊥ is symmetric)
    grad_Delta = D_perp                          # (R, 54)

    # Gradient w.r.t. H:
    # pen = 0.5 ||D - H M||² where M = (HᵀH)⁻¹ HᵀD
    # d/dH: complex (H appears in both P_H and M).
    # Using the chain rule carefully:
    #   dL/dH = -D_perp @ M.T  +  H @ (M @ M.T) - H @ (HtH_inv @ H.T @ D_perp @ M.T)
    # Simplified (first-order in perturbation of H, holding M fixed):
    #   dL/dH ≈ -D_perp @ M.T
    grad_H = -D_perp @ coef.T                   # (R, 18)

    return pen, grad_H, grad_Delta


def gate3_penalty_soft(Sigma, H, Delta, reg=1e-6, eps_sv=1e-4):
    """
    Soft Gate-3: aug_gap should equal 9.
    Proxy: project Σ onto orthogonal complement of N=[H|Δ].
    Want rank(Σ_perp) = 9 → maximize log-det of (Σ_perp.T Σ_perp + ε I).
    Penalty: -log det(Σ_perp.T Σ_perp + ε I) [minimizing this maximizes det].

    Returns (penalty_value, grad_Sigma, grad_N)
    where grad_N is the combined gradient w.r.t. N=[H|Delta].
    """
    N = np.hstack([H, Delta])                   # (R, 72)
    NtN = N.T @ N + reg * np.eye(N.shape[1])    # (72, 72)
    NtN_inv = np.linalg.solve(NtN, np.eye(NtN.shape[0]))
    # Projection of Sigma onto N-perp:
    coef_S   = NtN_inv @ (N.T @ Sigma)          # (72, 9)
    P_N_S    = N @ coef_S                        # (R, 9)
    S_perp   = Sigma - P_N_S                    # (R, 9)

    # Gram matrix of S_perp:
    G = S_perp.T @ S_perp + eps_sv * np.eye(9) # (9, 9)
    sign, logdet = np.linalg.slogdet(G)
    if sign <= 0:
        # Degenerate — return large penalty with zero gradient signal
        return 50.0, np.zeros_like(Sigma), np.zeros_like(N)

    pen = -logdet                               # minimize = maximize det

    # Gradient:
    # dL/dG = -G^{-1}  (from d(-log det G) = -G^{-T} = -G^{-1} since symmetric)
    G_inv = np.linalg.solve(G, np.eye(9))
    # dL/d(S_perp) = 2 * S_perp @ G^{-1}  (from d/dX tr(G_inv X^T X) chain)
    dL_dSperp = 2.0 * S_perp @ G_inv           # (R, 9)
    # dL/dSigma = P_{N⊥} dL_dSperp  (chain through S_perp = (I-P_N)Sigma)
    coef_g = NtN_inv @ (N.T @ dL_dSperp)
    grad_Sigma = dL_dSperp - N @ coef_g         # (R, 9)

    # dL/dN (holding Sigma fixed): P_N changes with N
    # Approximate: grad_N ≈ -dL_dSperp @ coef_S.T  (first-order)
    grad_N = -dL_dSperp @ coef_S.T             # (R, 72)

    return pen, grad_Sigma, grad_N


# ── Main objective ─────────────────────────────────────────────
def obj(x, R, lam2, lam3):
    """
    Total loss = Frobenius residual + λ₂ * Gate2_pen + λ₃ * Gate3_pen
    Returns (loss, gradient).
    """
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)

    A3 = A.reshape(R, 3, 3)
    B3 = B.reshape(R, 3, 3)

    # ── Frobenius term ──
    T_approx = np.einsum('ki,kj,kl->ijl', A, B, C)
    E = T_approx - T
    frob_loss = 0.5 * float(np.dot(E.ravel(), E.ravel()))
    dA_f = np.einsum('ijl,kj,kl->ki', E, B, C)
    dB_f = np.einsum('ijl,ki,kl->kj', E, A, C)
    dC_f = np.einsum('ijl,ki,kj->kl', E, A, B)

    total_loss = frob_loss
    dA = dA_f.copy()
    dB = dB_f.copy()
    dC = dC_f.copy()

    if lam2 > 0 or lam3 > 0:
        Sigma, H, Delta, e0, e1, e2 = step51(A3, B3, R)

        if lam2 > 0:
            g2_pen, grad_H, grad_D = gate2_penalty(H, Delta)
            total_loss += lam2 * g2_pen

            # Back-prop through H = [e0-e1 | e1-e2]:
            # grad_H is (R, 18): first 9 cols = dL/d(e0-e1), next 9 = dL/d(e1-e2)
            gh1 = lam2 * grad_H[:, :9]   # dL/d(Eta1) = dL/d(e0-e1)
            gh2 = lam2 * grad_H[:, 9:]   # dL/d(Eta2) = dL/d(e1-e2)
            # e0[k,r*3+u] = A3[k,r,0]*B3[k,0,u]  → chain to A3 and B3
            # Eta1 = e0 - e1; Eta2 = e1 - e2
            de0 = (gh1).reshape(R, 3, 3)     # dL/de0
            de1 = (-gh1 + gh2).reshape(R, 3, 3)  # dL/de1
            de2 = (-gh2).reshape(R, 3, 3)    # dL/de2
            # e_s[k,r,u] = A3[k,r,s] * B3[k,s,u]
            # dL/dA3[k,r,s] = sum_u de_s[k,r,u] * B3[k,s,u]
            # dL/dB3[k,s,u] = sum_r de_s[k,r,u] * A3[k,r,s]
            dA3_gate2 = np.zeros_like(A3)
            dB3_gate2 = np.zeros_like(B3)
            for s_idx, de_s in enumerate([de0, de1, de2]):
                dA3_gate2[:, :, s_idx] += np.sum(de_s * B3[:, s_idx:s_idx+1, :], axis=2)
                dB3_gate2[:, s_idx, :] += np.sum(de_s * A3[:, :, s_idx:s_idx+1], axis=1)

            # Back-prop through Delta:
            # Delta[dead_pair_idx * 9 + r*3+u] = A3[k,r,s] * B3[k,t,u] for (s,t)
            gD = lam2 * grad_D             # (R, 54)
            for pi, (s, t) in enumerate(DEAD_PAIRS):
                gD_block = gD[:, pi*9:(pi+1)*9].reshape(R, 3, 3)  # (R, r, u)
                # dL/dA3[k,r,s] += sum_u gD_block[k,r,u] * B3[k,t,u]
                dA3_gate2[:, :, s] += np.sum(gD_block * B3[:, t:t+1, :], axis=2)
                # dL/dB3[k,t,u] += sum_r gD_block[k,r,u] * A3[k,r,s]
                dB3_gate2[:, t, :] += np.sum(gD_block * A3[:, :, s:s+1], axis=1)

            dA += dA3_gate2.reshape(R, DIM)
            dB += dB3_gate2.reshape(R, DIM)

        if lam3 > 0:
            g3_pen, grad_S, grad_N = gate3_penalty_soft(Sigma, H, Delta)
            total_loss += lam3 * g3_pen

            # Back-prop Gate-3 through Sigma, H, Delta
            gS = lam3 * grad_S   # (R, 9)
            # Sigma[k,r*3+u] = sum_s A3[k,r,s]*B3[k,s,u]
            # dL/dA3[k,r,s] = sum_u gS[k,r*3+u] * B3[k,s,u]
            # dL/dB3[k,s,u] = sum_r gS[k,r*3+u] * A3[k,r,s]
            gS3 = gS.reshape(R, 3, 3)
            dA3_g3 = np.einsum('kru,ksu->krs', gS3, B3)
            dB3_g3 = np.einsum('kru,krs->ksu', gS3, A3)

            # grad_N = (R, 72) — split back into H-part and Delta-part
            gN_H = lam3 * grad_N[:, :18]   # (R, 18)
            gN_D = lam3 * grad_N[:, 18:]   # (R, 54)

            # Back-prop gN_H through H = [Eta1|Eta2]
            gnh1 = gN_H[:, :9]
            gnh2 = gN_H[:, 9:]
            de0_g3 = gnh1.reshape(R, 3, 3)
            de1_g3 = (-gnh1 + gnh2).reshape(R, 3, 3)
            de2_g3 = (-gnh2).reshape(R, 3, 3)
            for s_idx, de_s in enumerate([de0_g3, de1_g3, de2_g3]):
                dA3_g3[:, :, s_idx] += np.sum(de_s * B3[:, s_idx:s_idx+1, :], axis=2)
                dB3_g3[:, s_idx, :] += np.sum(de_s * A3[:, :, s_idx:s_idx+1], axis=1)

            # Back-prop gN_D through Delta
            for pi, (s, t) in enumerate(DEAD_PAIRS):
                gD3_block = gN_D[:, pi*9:(pi+1)*9].reshape(R, 3, 3)
                dA3_g3[:, :, s] += np.sum(gD3_block * B3[:, t:t+1, :], axis=2)
                dB3_g3[:, t, :] += np.sum(gD3_block * A3[:, :, s:s+1], axis=1)

            dA += dA3_g3.reshape(R, DIM)
            dB += dB3_g3.reshape(R, DIM)

    return total_loss, np.concatenate([dA.ravel(), dB.ravel(), dC.ravel()])


def frob_only(x, R):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    return float(np.linalg.norm(np.einsum('ki,kj,kl->ijl', A, B, C) - T))


def gate_metrics(x, R):
    A3 = x[:R*DIM].reshape(R, 3, 3)
    B3 = x[R*DIM:2*R*DIM].reshape(R, 3, 3)
    Sigma, H, Delta, *_ = step51(A3, B3, R)
    N  = np.hstack([H, Delta])
    SN = np.hstack([Sigma, N])
    tol = 1e-8
    rk_H  = np.linalg.matrix_rank(H,  tol=tol)
    rk_N  = np.linalg.matrix_rank(N,  tol=tol)
    rk_SN = np.linalg.matrix_rank(SN, tol=tol)
    U, s_H, _ = np.linalg.svd(H, full_matrices=True)
    rk = np.sum(s_H > tol * s_H[0]) if len(s_H) > 0 else 0
    H_null = U[:, rk:]
    D_perp_norm = float(np.linalg.norm(H_null.T @ Delta, 'fro'))
    return {
        'delta_leak': rk_N - rk_H,
        'aug_gap':    rk_SN - rk_N,
        'D_perp':     D_perp_norm,
        'rk_H':       rk_H,
    }


def one_restart(args):
    R, seed, x_warm, lam2, lam3 = args
    rng = np.random.default_rng(seed)
    n = 3 * R * DIM
    if x_warm is not None and rng.random() < 0.5:
        sigma = rng.uniform(0.005, 0.15)
        x0 = x_warm + rng.standard_normal(n) * sigma
    else:
        scale = rng.uniform(0.3, 1.2) / R**0.5
        x0 = rng.standard_normal(n) * scale
    res = minimize(
        lambda x: obj(x, R, lam2, lam3), x0, jac=True, method='L-BFGS-B',
        options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    f = frob_only(res.x, R)
    return f, res.x


def save_best(path, R, x, frob):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    with open(path, 'w') as fp:
        json.dump({'rank': R, 'frobenius': frob,
                   'alpha': A.tolist(), 'beta': B.tolist(), 'gamma': C.tolist()}, fp)


# ── Annealing schedule ─────────────────────────────────────────
# Phase 1: high λ₂ to rapidly force Gate-2 satisfaction
# Phase 2: lower λ₂, add λ₃ nudge
# Phase 3: λ₂ → 0 (pure Frobenius, but initialized from gate-satisfying solution)
SCHEDULE = [
    # (lam2,  lam3,  n_restarts, label)
    (1.0,    0.0,   40,  'force Gate-2'),
    (0.1,    0.01,  40,  'Gate-2 + Gate-3 nudge'),
    (0.01,   0.01,  40,  'relax Gate-2'),
    (0.001,  0.001, 60,  'fine tune'),
    (0.0,    0.0,   80,  'pure Frobenius from structured init'),
]


def run_rank(R, warm_path=None, n_workers=6):
    rng = np.random.default_rng(R * 7919)
    best_f = np.inf
    best_x = None

    # Try to load prior best as warm start
    if warm_path and os.path.exists(warm_path):
        with open(warm_path) as f:
            d = json.load(f)
        best_x = np.concatenate([
            np.array(d['alpha']).ravel(),
            np.array(d['beta']).ravel(),
            np.array(d['gamma']).ravel()
        ])
        best_f = frob_only(best_x, R)
        print(f'  Warm start: frob={best_f:.6f}')

    ckpt_path = os.path.join(CKPT_DIR, f'rank{R}_gated_best.json')
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            d = json.load(f)
        x_ck = np.concatenate([np.array(d['alpha']).ravel(),
                                np.array(d['beta']).ravel(),
                                np.array(d['gamma']).ravel()])
        f_ck = frob_only(x_ck, R)
        if f_ck < best_f:
            best_f = f_ck
            best_x = x_ck
            print(f'  Checkpoint loaded: frob={best_f:.6f}')

    print(f'\nRank {R} gated search')
    print(f'Float16 closure target: frob ≤ {3*R*2**-10:.5f}')
    print(f'bfloat16 closure target: frob ≤ {3*R*2**-7:.5f}')
    print()

    for lam2, lam3, n_restarts, label in SCHEDULE:
        t0 = time.time()
        seeds = rng.integers(0, 2**31, size=n_restarts)
        args = [(R, int(s), best_x, lam2, lam3) for s in seeds]

        with Pool(n_workers) as pool:
            results = pool.map(one_restart, args)

        improved = False
        for f, x in results:
            if f < best_f:
                best_f = f
                best_x = x.copy()
                improved = True

        if best_x is not None:
            save_best(ckpt_path, R, best_x, best_f)
            gm = gate_metrics(best_x, R)
            elapsed = time.time() - t0
            print(f'  [{label}] λ₂={lam2} λ₃={lam3}  '
                  f'frob={best_f:.6f}  '
                  f'D_perp={gm["D_perp"]:.4f}  '
                  f'leak={gm["delta_leak"]} gap={gm["aug_gap"]}  '
                  f'({elapsed:.0f}s) {"↑ improved" if improved else ""}')
        else:
            print(f'  [{label}] no solution yet')

    print(f'\nFinal: rank={R} frob={best_f:.8f}')
    if best_x is not None:
        gm = gate_metrics(best_x, R)
        print(f'  delta_leak={gm["delta_leak"]} aug_gap={gm["aug_gap"]} '
              f'||D_perp||={gm["D_perp"]:.6f}')
        closes = np.log2(3*R / best_f) if best_f > 0 else np.inf
        print(f'  Closes at b ≤ {closes:.2f} mantissa bits')
        for name, b in [('float16',10),('bfloat16',7),('int8',8)]:
            budget = 3*R*2**(-b)
            status = 'CLOSED' if budget >= best_f else 'OPEN'
            print(f'  {name:10s} (b={b}): budget={budget:.5f}  {status}')
    return best_f, best_x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=21)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--warm', type=str, default=None,
                        help='Path to JSON with warm-start factors')
    args = parser.parse_args()

    # Default warm start: use rank-(R+1) best if available
    warm = args.warm
    if warm is None:
        candidate = os.path.join(CKPT_DIR, f'rank{args.rank+1}_gated_best.json')
        if os.path.exists(candidate):
            warm = candidate
        else:
            # Try existing unconstrained results
            candidate2 = f'CANON_ADE_EPSILON/results/cliff_search/rank{args.rank}_best.json'
            if os.path.exists(candidate2):
                warm = candidate2
                print(f'Using unconstrained rank-{args.rank} as warm start')

    run_rank(args.rank, warm_path=warm, n_workers=args.workers)
