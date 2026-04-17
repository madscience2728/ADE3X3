import sys, time
sys.path.insert(0, '.')
import numpy as np
from scripts.symmetric_lm import SymmetricDecomposition

decomp = SymmetricDecomposition(27)
print(f"R=27: {decomp.n_free} free params, expansion matrix shape: {decomp._expansion_matrix.shape}")

x0 = np.random.randn(decomp.n_free) * 0.5

# Time a single residual evaluation
t0 = time.time()
for _ in range(1000):
    r = decomp.params_to_residual(x0)
dt = time.time() - t0
print(f"1000 residual evals: {dt:.3f}s ({dt/1000*1e6:.0f} us each)")
print(f"Residual norm: {np.linalg.norm(r):.4f}")

# Try the optimization with verbose
from scipy.optimize import least_squares
print("\nStarting LM optimization...")
t0 = time.time()
result = least_squares(
    decomp.params_to_residual, x0,
    method='lm',
    ftol=1e-15, xtol=1e-15, gtol=1e-15,
    max_nfev=50000,
    verbose=2,
)
dt = time.time() - t0
print(f"\nDone in {dt:.1f}s, nfev={result.nfev}, cost={result.cost:.6e}")
