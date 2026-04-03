#!/usr/bin/env python3
"""Generate structured initialization JSONs for slp_turbo warm-start."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

def build_matmul_tensor(n=3):
    T = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                T[n*i+j, n*j+k, n*i+k] = 1.0
    return T

def kronecker_init(R, n=3):
    """9 Kronecker-structured terms + (R-9) random suppression terms."""
    n2 = n * n
    alpha, beta = [], []
    
    for r in range(n):
        for u in range(n):
            a = np.zeros(n2)
            b = np.zeros(n2)
            for j in range(n):
                a[n*r + j] = 1.0
                b[n*j + u] = 1.0
            alpha.append(a)
            beta.append(b)
    
    rng = np.random.default_rng(2026)
    for _ in range(R - 9):
        alpha.append(rng.standard_normal(n2) * 0.3)
        beta.append(rng.standard_normal(n2) * 0.3)
    
    alpha = np.array(alpha)
    beta = np.array(beta)
    
    # Compute gamma via least squares per output column
    T = build_matmul_tensor(n)
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    
    gamma = np.zeros((R, n2))
    for c in range(n2):
        target_col = T[:, :, c].ravel()
        gamma[:, c] = np.linalg.lstsq(A.T, target_col, rcond=None)[0]
    
    # Compute fitness
    T_recon = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma)
    fitness = float(np.max(np.abs(T_recon - T)))
    
    return alpha, beta, gamma, fitness

def diverse_random_init(R, n=3, seed=0):
    """Random init with varied structure."""
    n2 = n * n
    rng = np.random.default_rng(seed)
    
    # Mix of sparse and dense terms
    alpha, beta = [], []
    for k in range(R):
        if k < R // 3:
            # Sparse ±1
            a = np.zeros(n2)
            b = np.zeros(n2)
            na = rng.integers(2, 5)
            nb = rng.integers(2, 5)
            a[rng.choice(n2, na, replace=False)] = rng.choice([-1.0, 1.0], na)
            b[rng.choice(n2, nb, replace=False)] = rng.choice([-1.0, 1.0], nb)
        else:
            a = rng.standard_normal(n2) * 0.5
            b = rng.standard_normal(n2) * 0.5
        alpha.append(a)
        beta.append(b)
    
    alpha = np.array(alpha)
    beta = np.array(beta)
    
    T = build_matmul_tensor(n)
    A = np.zeros((R, n2 * n2))
    for k in range(R):
        A[k] = np.outer(alpha[k], beta[k]).ravel()
    
    gamma = np.zeros((R, n2))
    for c in range(n2):
        target_col = T[:, :, c].ravel()
        gamma[:, c] = np.linalg.lstsq(A.T, target_col, rcond=None)[0]
    
    T_recon = np.einsum('ki,kj,kc->ijc', alpha, beta, gamma)
    fitness = float(np.max(np.abs(T_recon - T)))
    
    return alpha, beta, gamma, fitness

# Generate seeds
for R in [19, 20]:
    # Kronecker
    a, b, g, f = kronecker_init(R)
    path = ROOT / f"structured_init_kronecker_R{R}.json"
    path.write_text(json.dumps({
        "alpha": a.tolist(), "beta": b.tolist(), "gamma": g.tolist(),
        "fitness": f, "rank": R, "dim": 9
    }, indent=2))
    print(f"Kronecker R={R}: fitness={f:.6f} -> {path.name}")
    
    # Several random seeds
    for seed in range(5):
        a, b, g, f = diverse_random_init(R, seed=seed*1000+R)
        path = ROOT / f"structured_init_random{seed}_R{R}.json"
        path.write_text(json.dumps({
            "alpha": a.tolist(), "beta": b.tolist(), "gamma": g.tolist(),
            "fitness": f, "rank": R, "dim": 9
        }, indent=2))
        print(f"Random{seed} R={R}: fitness={f:.6f} -> {path.name}")
