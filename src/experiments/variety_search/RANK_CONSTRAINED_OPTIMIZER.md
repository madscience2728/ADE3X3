# Rank-Constrained Optimizer for the Low-Nuisance Variety

## 1. Problem Statement

We seek (α, β, γ) ∈ ℝ^{19×9} × ℝ^{19×9} × ℝ^{19×9} minimizing

    ‖T − Σ_k α_k ⊗ β_k ⊗ γ_k‖_∞

subject to the variety conditions from §41 of the canon:

1. **rank(H) ≤ 10**, where H = [Eta1 | Eta2] ∈ ℝ^{19×18} is bilinear in (α, β)
2. **Δ ⊂ span(H)**: all 54 Delta columns lie in the column span of H
3. **Γ·Σ = 3I₉** and **Γ·Nuisance = 0**: the exact solvability criterion

For R = 19, the nuisance budget table gives rank(Nuisance) ≤ 10, which forces rank(H) ≤ 10 since H generates the anisotropy portion of the nuisance span.

## 2. Current State of Attempts

| Approach | Result | Why it fails |
|----------|--------|--------------|
| Alternating LP + rank penalty | LP erases rank progress each cycle | LP operates in full ℝ^19, unaware of rank structure |
| L-BFGS-B on ‖R‖²_F + λ·tail_energy | fitness 0.086, H_rank=18 always | Smooth optimizer can't cross codim-1 rank strata |
| Gradient on (α,β) + LP on γ | H_rank=18 persists | Same barrier — gradient steps stay on the wrong stratum |

**Key SVD observation:** For all candidates (warm and cold), the SVD of H shows:
- σ₀–σ₇ are O(1) — the "live" directions
- σ₈–σ₁₀ are O(0.1–0.35) — the barrier zone  
- σ₁₂–σ₁₇ ≈ 0 — already effectively dead

The formal rank is 18, but the effective rank is 11–12. Only σ₁₀ ≈ 0.35 and σ₁₁ ≈ 0.13 separate the current basins from the rank-10 variety. The problem is topological: the rank-10 locus is a codimension-~64 algebraic subvariety in the 342-dimensional (α, β) parameter space, and smooth optimizers cannot cross into it.

## 3. Why Standard Methods Fail

### 3.1 The Rank Stratum Barrier

The set {(α,β) : rank(H(α,β)) ≤ 10} is an algebraic variety of codimension
(19−10)(18−10) = 72 in ℝ^{19×18}, but the preimage under the bilinear map H(α,β) has lower codimension due to the kernel of the Jacobian. Still, it is a measure-zero set surrounded by the rank-11+ open stratum. No amount of smooth penalty can land exactly on it — the gradient vanishes along the variety while pointing away from it in normal directions.

### 3.2 The LP Incompatibility

Linear programming over γ produces the globally optimal γ for fixed (α,β), but it does so in full ℝ^{19} with no awareness of which directions in term-space correspond to nuisance. Each LP cycle can (and does) rotate the solution in directions that increase rank(H), undoing any progress from the (α,β) step.

## 4. Proposed Attack: Grassmannian Reparameterization

### 4.1 Core Idea

Instead of constraining rank(H) ≤ 10 from outside, we **parameterize directly on the rank-10 variety** so the constraint holds by construction.

Every rank-10 matrix H ∈ ℝ^{19×18} factors as H = P Q^T where P ∈ ℝ^{19×10}, Q ∈ ℝ^{18×10}. Equivalently, the row space of H is a 10-dimensional subspace V ∈ Gr(10, 18) (the Grassmannian), and each row h_k = P[k,:] @ Q^T is a linear combination in V.

The bilinear structure of H constrains which (P, Q) are reachable. Specifically, for each term k:

    H[k,:] = [Eta1_k | Eta2_k]

where Eta1_k and Eta2_k are specific bilinear functions of (α_k, β_k). The 9-dimensional vectors α_k and β_k produce an 18-dimensional row H[k,:] through a known quadratic map.

### 4.2 Fiber Structure

Fix V ∈ Gr(10, 18). Then the constraint "H[k,:] ∈ V" gives 8 linear constraints on the 18-dimensional row. Since H[k,:] is bilinear in (α_k, β_k) with 18 free parameters, we have:

- The Jacobian of H[k,:] w.r.t. (α_k, β_k) has rank ≤ 18 (generically ~14–16 due to the bilinear structure)
- "H[k,:] ∈ V" imposes 8 constraints
- Net degrees of freedom per term: ~6–10

For each term k and fixed V, define:

    F_k(V) = { (α_k, β_k) : H(α_k, β_k)·V_⊥ = 0 }

where V_⊥ ∈ ℝ^{18×8} is the orthogonal complement of V. This is a system of 8 bilinear equations in 18 unknowns — generically a 10-dimensional (or higher) manifold.

### 4.3 The Three-Block Alternating Scheme

**Block 1: Fix V and γ, optimize (α, β)**  
For each term k independently, solve the bilinear feasibility problem:
- Minimize ‖T − Σ α_k ⊗ β_k ⊗ γ_k‖_∞ 
- Subject to H(α_k, β_k). V_⊥ = 0

This decomposes into 19 independent 18-variable problems with 8 bilinear constraints each. Use projected gradient descent on the constraint manifold, or solve the KKT system directly since the constraints are quadratic.

**Block 2: Fix (α, β), solve γ via LP**  
Standard column-wise LP: for each column c of γ, minimize ‖T[:,:,c] − Σ_k α_k ⊗ β_k · γ_{kc}‖_∞. This is exact and doesn't affect H.

**Block 3: Fix (α, β, γ), update V**  
Compute H = H(α, β) and set V = span of the top-10 right singular vectors of H. This is a Procrustes-type update on the Grassmannian. If the current H already has rank ≤ 10, V doesn't change. If rank(H) = 11 due to numerical drift, this projects back.

### 4.4 Block 1 in Detail: Constrained (α_k, β_k) Optimization

For term k with fixed V_⊥ ∈ ℝ^{18×8} and fixed γ, we need:

    min_{α_k, β_k}  contribution to ‖T − approx‖_∞
    s.t.  C(α_k, β_k) := [Eta1(α_k, β_k) | Eta2(α_k, β_k)] · V_⊥ = 0   (8 equations)

The constraint C is bilinear: C(α, β) = M(α) · β_expanded + N(β) · α_expanded where M, N are linear operators derived from the Eta construction.

**Approach A: Nullspace parameterization.** At a current point (α⁰, β⁰), linearize C to get the Jacobian J_k ∈ ℝ^{8×18}. The tangent space to the constraint manifold is ker(J_k), dimension ≥ 10. Project the fitness gradient onto ker(J_k) and take a step. Retract back to the constraint manifold via Newton correction on C = 0.

**Approach B: Augmented Lagrangian.** Solve min ‖residual‖ + μ‖C(α_k,β_k)‖² + λ^T C(α_k,β_k) with increasing μ. This avoids explicit manifold operations but converges more slowly.

**Approach C: Direct elimination.** For each k, α_k has 9 components. Given α_k, the constraint C = 0 becomes linear in β_k (since η is bilinear in α, β). So for fixed α_k, the feasible β_k lies in a ≥1-dimensional affine subspace. Parameterize β_k = β⁰_k + N_k · t_k where N_k is the nullspace basis. This reduces to optimizing over (α_k ∈ ℝ^9, t_k ∈ ℝ^{d_k}).

**Approach C is the winner** — it exploits the bilinear structure directly:

1. Fix V, fix α_k for all k
2. For each k: compute the linear system C(α_k, ·) β_k = 0, get the solution space β_k(t_k)
3. Now optimize over (α ∈ ℝ^{19×9}, t ∈ ℝ^{19×d}) with γ solved by LP
4. Update V from the resulting H

### 4.5 Initialization

Start from the current best (fitness 0.073). Compute H, take its top-10 right singular subspace as V₀. This already captures 99.9986% of H's energy. Then project each (α_k, β_k) onto the nearest point satisfying H[k,:] ∈ V₀ — a small perturbation since σ₁₁ = 0.13 and σ₁₂–σ₁₇ ≈ 0.

Expected fitness increase from initialization: ≤ 0.01 (since we're only killing 0.0014% of H's energy).

## 5. Convergence Properties

### 5.1 Why This Should Work

The key insight is **separation of concerns**:
- V captures the global coupling (which 10-dim subspace all rows share)
- (α_k, β_k) captures the local fiber geometry (where each term sits within V)
- γ captures the linear reconstruction (always solved exactly)

Each block is either convex (Block 2: LP), closed-form (Block 3: SVD), or a well-posed low-dimensional nonlinear program (Block 1: 9+d variables per term with smooth objective). The coupling through V creates a natural annealing: early iterations allow large V updates, while near convergence V stabilizes and the per-term problems become nearly independent.

### 5.2 Potential Failure Modes

1. **Fiber dimension collapse:** If d_k = 0 for some k (the constraint C(α_k, ·) has no nullspace), that term is fully determined by α_k and V, leaving no room for fitness optimization. Mitigation: choose V to maximize min_k d_k.

2. **V oscillation:** If Block 3 changes V significantly each iteration, Blocks 1–2 never converge. Mitigation: use a damped V update — interpolate between old and new V on the Grassmannian via geodesic.

3. **Local minima on the variety:** Even with rank(H) = 10 by construction, there may be no R=19 decomposition with fitness < 0.07 on this variety. This would actually be an important theoretical result — it would mean the 0.073 basin is not connected to the exact solution variety.

## 6. Implementation Plan

### Phase A: Infrastructure
1. Implement `compute_V_perp(H, target_rank=10)` — returns V_⊥ ∈ ℝ^{18×8}
2. Implement `constraint_system(alpha_k, V_perp)` — returns the linear system in β_k
3. Implement `beta_nullspace(alpha_k, V_perp)` — returns (β⁰, N_k) particular solution + nullspace
4. Validate: check that for β_k = β⁰ + N_k @ t, H[k,:] ∈ V for all t

### Phase B: Core Optimizer
1. Implement the three-block alternating scheme
2. Inner loop: for fixed V, alternate between:
   - Gradient steps on (α, t) minimizing L∞ fitness (or a smooth surrogate)  
   - LP on γ
3. Outer loop: update V via truncated SVD of H

### Phase C: Hardening
1. Damped Grassmannian updates
2. Multi-start from the top-N existing candidates
3. Adaptive λ schedule if using augmented Lagrangian fallback
4. Track σ₁₁(H) convergence to verify we're on the variety

## 7. Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| rank(H) | 18 (formal) / 11-12 (effective) | 10 exact | 10 exact |
| fitness | 0.073 | < 0.073 | < 0.05 |
| σ₁₁(H) | 0.13 | < 1e-10 | 0 (exact) |
| Δ ⊂ span(H) residual | 0.040 | < 0.01 | 0 (exact) |
| R + η_nullity | 19 | 27 | 27 (conservation law) |

If we achieve rank(H) = 10 exact with fitness < 0.073, we have strong evidence that the variety contains deeper basins. If fitness improves to near zero while maintaining rank(H) = 10, we have found an exact R=19 decomposition of 3×3 matrix multiplication — which would be a major result.
