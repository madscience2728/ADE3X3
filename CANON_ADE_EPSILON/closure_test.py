"""
Closure test: cast R=22 decomposition to bfloat16, run on 100 random matrix pairs,
compare output to float64 exact matmul. Report actual arithmetic errors.
"""
import numpy as np
import json
import struct

# ── Load best R=22 decomposition ────────────────────────────────────────────
with open('CANON_ADE_EPSILON/results/rank22_best.json') as f:
    d = json.load(f)

A_f64 = np.array(d['alpha'])   # (22, 9)
B_f64 = np.array(d['beta'])
C_f64 = np.array(d['gamma'])
R = d['rank']
print(f"Loaded R={R} decomposition, ε_Frobenius = {d['frobenius']:.8f}")

# ── bfloat16 simulation via float32 truncation ───────────────────────────────
# bfloat16 = top 16 bits of float32 (1 sign + 8 exp + 7 mantissa)
def to_bf16(x):
    """Simulate bfloat16 by zeroing low 16 bits of float32."""
    f32 = x.astype(np.float32)
    raw = f32.view(np.uint32)
    raw = (raw & 0xFFFF0000).astype(np.uint32)
    return raw.view(np.float32)

A_bf16 = to_bf16(A_f64)
B_bf16 = to_bf16(B_f64)
C_bf16 = to_bf16(C_f64)

print(f"Factor quantization error (max): "
      f"A={np.max(np.abs(A_f64-A_bf16)):.2e}  "
      f"B={np.max(np.abs(B_f64-B_bf16)):.2e}  "
      f"C={np.max(np.abs(C_f64-C_bf16)):.2e}")

# ── Run 100 random matrix pairs ──────────────────────────────────────────────
rng = np.random.default_rng(42)
N_PAIRS = 100

errors_abs = []
errors_rel = []
exact_norms = []

for trial in range(N_PAIRS):
    M1 = rng.uniform(-1, 1, (3, 3)).astype(np.float32)
    M2 = rng.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Exact matmul in float64
    C_exact = (M1.astype(np.float64) @ M2.astype(np.float64))  # (3,3)

    # Vectorize inputs: a = M1.ravel() (len 9), b = M2.ravel() (len 9)
    a = M1.ravel().astype(np.float64)
    b = M2.ravel().astype(np.float64)

    # Approximate matmul using bfloat16 factors
    # c[k] = sum_k alpha[k]·a * beta[k]·b → scalar, then c_out = sum_k c[k] * gamma[k]
    scalars = (A_bf16.astype(np.float64) @ a) * (B_bf16.astype(np.float64) @ b)  # (R,)
    c_approx_vec = C_bf16.astype(np.float64).T @ scalars  # (9,)
    C_approx = c_approx_vec.reshape(3, 3)

    err = np.max(np.abs(C_exact - C_approx))
    norm_c = np.max(np.abs(C_exact))
    errors_abs.append(err)
    errors_rel.append(err / max(norm_c, 1e-10))
    exact_norms.append(norm_c)

errors_abs = np.array(errors_abs)
errors_rel = np.array(errors_rel)

print(f"\n── Results over {N_PAIRS} random matrix pairs (entries uniform in [-1,1]) ──")
print(f"  Max absolute error:    {errors_abs.max():.6e}")
print(f"  Mean absolute error:   {errors_abs.mean():.6e}")
print(f"  Max relative error:    {errors_rel.max():.6e}")
print(f"  Mean relative error:   {errors_rel.mean():.6e}")
print(f"  Mean |C_exact| max:    {np.mean(exact_norms):.4f}")

# bfloat16 unit roundoff = 2^-7 ≈ 0.0078
BF16_EPS = 2**-7
print(f"\n  bfloat16 unit roundoff (2^-7):  {BF16_EPS:.6f}")
print(f"  Errors within 1× bf16_eps:      {np.mean(errors_abs <= BF16_EPS)*100:.1f}%")
print(f"  Errors within 2× bf16_eps:      {np.mean(errors_abs <= 2*BF16_EPS)*100:.1f}%")
print(f"  Errors within 10× bf16_eps:     {np.mean(errors_abs <= 10*BF16_EPS)*100:.1f}%")

print(f"\n  Max abs error / bf16_eps = {errors_abs.max()/BF16_EPS:.2f}×")

# Verdict
if errors_abs.max() <= BF16_EPS:
    verdict = "CLOSED — all errors within 1× bfloat16 unit roundoff"
elif errors_abs.max() <= 10 * BF16_EPS:
    verdict = f"NEAR-CLOSED — max error = {errors_abs.max()/BF16_EPS:.1f}× bf16_eps"
else:
    verdict = f"NOT CLOSED — max error = {errors_abs.max()/BF16_EPS:.1f}× bf16_eps"
print(f"\n  VERDICT: {verdict}")
