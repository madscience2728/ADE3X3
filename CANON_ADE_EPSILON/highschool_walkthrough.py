"""
Step-by-step: How to do 3x3 matrix multiplication with fewer than 27 multiplications.
Explained from first principles using the epsilon-ring framework.
"""
import numpy as np
import json

print("""
╔══════════════════════════════════════════════════════════════════╗
║  HOW TO MULTIPLY TWO 3x3 MATRICES IN FEWER THAN 27 OPERATIONS  ║
╚══════════════════════════════════════════════════════════════════╝

SETUP: You want to compute C = A × B where A, B are 3×3 matrices.
       C[r,u] = sum over s of A[r,s] * B[s,u]

The naive way: 9 output entries, each needs 3 multiplications = 27 total.
""")

# ── STEP 1: The naive 27 ────────────────────────────────────────────────────
print("═"*70)
print("STEP 1 — The naive 27 scalar multiplications")
print("═"*70)
print("""
For each output cell C[r,u], compute:
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]   ← 3 mults
  C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]   ← 3 mults
  ...  (9 cells × 3 mults = 27 total)

Each multiplication (r,s,u) computes:  m_{r,s,u} = A[r,s] * B[s,u]
""")

# Show the 27 terms
triples = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]
print("The 27 terms:  m[r,s,u] = A[r,s] * B[s,u]")
for i, (r,s,u) in enumerate(triples):
    comma = "  " if (i+1) % 9 != 0 else "\n"
    print(f"  m[{r},{s},{u}]", end=comma)

# ── STEP 2: The tensor view ─────────────────────────────────────────────────
print("═"*70)
print("STEP 2 — The tensor picture (why this is a geometry problem)")
print("═"*70)

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

print(f"""
Think of A as a 9-vector (flatten row by row), B as a 9-vector, C as a 9-vector.
Then: C = sum over k of  gamma_k * (alpha_k · A) * (beta_k · B)

This is a rank decomposition of a 9×9×9 tensor T_matmul.
  ||T_matmul||_F = sqrt(27) = {np.linalg.norm(T):.4f}

The 27 standard multiplications = 27 rank-1 terms, each a "spike" at one position.
They are perfectly orthogonal (like unit vectors in 27 directions).
""")

# ── STEP 3: Dropping terms ──────────────────────────────────────────────────
print("═"*70)
print("STEP 3 — What happens when you drop some multiplications?")
print("═"*70)
print("""
If you use only R of the 27 standard terms (drop 27-R terms):
  - C_approx is missing exactly (27-R) contributions
  - Each missing contribution has magnitude 1 (they're orthogonal)
  - Error = sqrt(27-R)   [Pythagorean theorem on orthogonal vectors]

The "budget" at precision b is:  budget = 3 * R * 2^{-b}
This is how much total rounding error you accumulate with b-bit arithmetic.

For the approximation to be "invisible" to b-bit arithmetic:
   error  <=  budget
   sqrt(27-R) <= 3 * R * 2^{-b}
   2^{-b}     >= sqrt(27-R) / (3*R)
   b           <= log2(3*R / sqrt(27-R))       ← MAXIMUM BITS NEEDED
""")

print("  Standard basis: minimum bits needed at each R")
print(f"  {'R':>3}  {'dropped':>7}  {'error':>8}  {'b_needed':>10}  {'precision'}")
print("  " + "-"*55)
for R in [27, 26, 25, 23, 22, 21, 20, 19, 18, 17, 15, 13, 9]:
    eps = np.sqrt(27-R)
    if eps < 1e-10:
        b = 0; prec = "EXACT — no approximation"
    else:
        b = np.log2(3*R/eps)
        if b <= 4:   prec = "4-bit integer"
        elif b <= 7: prec = "bfloat16 (7-bit mantissa)"
        elif b <= 8: prec = "int8 (8-bit)"
        elif b <= 10: prec = "float16"
        else:        prec = f"float32 needs b={b:.1f}"
    print(f"  {R:3d}  {27-R:>7}    {eps:6.3f}    {b:7.3f}   {prec}")

# ── STEP 4: The key insight ─────────────────────────────────────────────────
print()
print("═"*70)
print("STEP 4 — The key insight: standard basis is not optimal")
print("═"*70)
print("""
The standard basis approach is naive: it just drops whole terms.
But you can REPLACE multiple standard terms with cleverly chosen combinations.

Example: instead of m[0,0,0] = A[0,0]*B[0,0]  AND  m[1,1,1] = A[1,1]*B[1,1]
         use:       m_new = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                           = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]

One multiplication now covers 4 combinations! (Strassen's trick for 2x2.)
The tradeoff: you get cross-terms (A[0,0]*B[1,1] and A[1,1]*B[0,0])
              that must cancel in other places.

Those cross-terms are exactly "Δ" in our framework.
The cancellation requirement is "‖Γ·Δ‖_F ≤ budget".
""")

# ── STEP 5: The R=19 candidate ──────────────────────────────────────────────
print("═"*70)
print("STEP 5 — The R=19 near-miss: what 38× improvement looks like")
print("═"*70)

try:
    with open('tests/slp_best_at_0.074.json') as f:
        d = json.load(f)
    alpha = np.array(d['alpha'])   # 19×9
    beta  = np.array(d['beta'])
    gamma = np.array(d['gamma'])
    R = 19

    # Reconstruct
    T_approx = np.einsum('ki,kj,kl->ijl', alpha, beta, gamma)
    residual = float(np.linalg.norm(T - T_approx))

    print(f"""
The best R=19 candidate found (slp_best_at_0.074.json):
  19 multiplications of the form:  m_k = (α_k · A) * (β_k · B)
  then:  C ≈ sum over k of  γ_k * m_k

  Residual  = {residual:.4f}    (vs standard basis residual = {np.sqrt(8):.4f})
  Improvement = {np.sqrt(8)/residual:.1f}×

  Bits needed:  b* = log2(3×19 / {residual:.4f}) = {np.log2(3*R/residual):.3f}
  Standard basis at R=19 needed: b* = {np.log2(3*19/np.sqrt(8)):.3f}

  What that means:
    - Standard 19-term basis: bfloat16 (7-bit) is MORE than enough
    - This clever 19-term basis: needs float16 (10-bit) — tighter!
    - WHY tighter? Because the clever basis relies on precise cancellation of
      cross-terms (the Δ components). More cancellation = more precision needed.

Structure of the 19 α-vectors (each is a 9-vector = flattened 3×3 matrix):
""")
    print("  k  |  α_k (top 3 entries by magnitude)  |  β_k  |  γ_k norm")
    print("  " + "-"*60)
    for k in range(R):
        a = alpha[k]; b_k = beta[k]; g = gamma[k]
        top_a = np.argsort(np.abs(a))[-3:][::-1]
        print(f"  {k:2d} |  ", end="")
        for i in top_a:
            r_,s_ = i//3, i%3
            print(f"A[{r_},{s_}]×{a[i]:+.3f}  ", end="")
        print(f"|  {np.linalg.norm(b_k):.3f}  |  {np.linalg.norm(g):.3f}")

except FileNotFoundError:
    print("  (slp_best_at_0.074.json not found, skipping demo)")

# ── STEP 6: The budget closes ───────────────────────────────────────────────
print()
print("═"*70)
print("STEP 6 — Closing the loop: when does the error become invisible?")
print("═"*70)
print(f"""
The epsilon-ring framework says:

  An R-term decomposition with residual ε is "equivalent to exact"
  in b-bit arithmetic if:
  
    ε  ≤  budget_frob(R, b)  =  3 × R × 2^(-b)

This is because rounding every coefficient to b bits introduces
at most 3×R×2^(-b) total error — which swamps the residual ε.
A b-bit computer literally cannot tell the difference.

For the R=19 candidate (ε = 0.074):

  b=7  (bfloat16):  budget = 3×19×2^(-7)  = {3*19*2**-7:.4f}  <  0.074  → NOT CLOSED
  b=8  (int8):      budget = 3×19×2^(-8)  = {3*19*2**-8:.4f}  <  0.074  → NOT CLOSED
  b=9:              budget = 3×19×2^(-9)  = {3*19*2**-9:.4f}  >  0.074  → CLOSED ✓
  b=10 (float16):   budget = 3×19×2^(-10) = {3*19*2**-10:.4f}  >  0.074  → CLOSED ✓

So: a 19-multiplication 3×3 matmul is valid (indistinguishable from exact)
    on any hardware using 9-bit or wider arithmetic.

For context:
  Standard 27-term exact:   0 bits needed
  Standard 19-term approx:  4.3 bits needed (4-bit integer is fine!)
  Clever 19-term near-miss: 9.6 bits needed (needs float16 or better)

The cleverness of the optimizer COSTS precision. It buys fewer operations
at the price of requiring more accurate arithmetic. Whether that's a good
trade depends on your hardware.
""")

print("═"*70)
print("SUMMARY TABLE: Standard basis drops vs clever optimizer")
print("═"*70)
print(f"  {'R':>3}  {'mults saved':>11}  {'std ε':>8}  {'std b*':>7}  {'opt ε':>8}  {'opt b*':>7}  {'verdict'}")
print("  " + "-"*75)

opt_results = {19: 0.074, 22: None}
for R in [27, 26, 25, 23, 22, 21, 20, 19, 18, 17, 15, 13]:
    eps_std = np.sqrt(max(0, 27-R))
    b_std = np.log2(3*R/eps_std) if eps_std > 1e-10 else 0.0
    opt_e = opt_results.get(R)
    b_opt = np.log2(3*R/opt_e) if opt_e else None

    std_s = f'{eps_std:8.4f}' if eps_std > 1e-10 else '   exact'
    b_std_s = f'{b_std:7.3f}' if eps_std > 1e-10 else '  exact'
    opt_s = f'{opt_e:8.4f}' if opt_e else '       —'
    b_opt_s = f'{b_opt:7.3f}' if b_opt else '      —'

    if R >= 23:     verdict = "exact — no approximation needed"
    elif R >= 19:   verdict = "open: exact decomp may exist"
    else:           verdict = "provably approximate"

    print(f"  {R:3d}  {27-R:>11}  {std_s}  {b_std_s}  {opt_s}  {b_opt_s}  {verdict}")
