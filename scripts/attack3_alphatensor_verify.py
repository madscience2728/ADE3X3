#!/usr/bin/env python3
"""Verify Δ⊂span(H) on AlphaTensor exact R=23 decomposition."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from db_optimizer.tensor import fiber_mode_decomposition
from ade3x3.steps.ade3x3_step63_reverse_engineering_cancellation_visualization import load_public_rank23_terms

terms, orientation, max_abs = load_public_rank23_terms()
R = len(terms)
alpha = np.array([t.alpha.flatten() for t in terms])  # (23, 9)
beta = np.array([t.beta.flatten() for t in terms])
gamma = np.array([t.gamma.flatten() for t in terms])

fm = fiber_mode_decomposition(alpha, beta, gamma)
H = np.hstack([fm["Eta1"], fm["Eta2"]])
Delta = fm["Delta"]
Gam = fm["Gamma"]

print(f"AlphaTensor R={R}, orientation={orientation}")
print(f"  rank(Sigma)     = {fm['sigma_rank']}")
print(f"  rank(H)         = {np.linalg.matrix_rank(H, tol=1e-8)}")
print(f"  rank(Delta)     = {np.linalg.matrix_rank(Delta, tol=1e-8)}")
print(f"  rank([H|Delta]) = {np.linalg.matrix_rank(np.hstack([H, Delta]), tol=1e-8)}")
print(f"  gamma_rank      = {fm['gamma_rank']}, ker(Γ) dim = {R - fm['gamma_rank']}")
print(f"  Γ·Σ residual    = {fm['gs_residual']:.2e}")
print(f"  ||Γ@H||         = {np.linalg.norm(Gam @ H):.2e}")
print(f"  ||Γ@Δ||         = {np.linalg.norm(Gam @ Delta):.2e}")

# Containment test
C, _, _, _ = np.linalg.lstsq(H, Delta, rcond=None)
resid = np.max(np.abs(Delta - H @ C))
print(f"  proj residual   = {resid:.2e}")
print(f"  Δ⊂span(H)?     {'YES ✓' if resid < 1e-8 else 'NO ✗'}")

# Conservation
eta_null = R - fm['sigma_rank']
print(f"  R + η_null      = {R} + {eta_null} = {R + eta_null} (should be 27)")
