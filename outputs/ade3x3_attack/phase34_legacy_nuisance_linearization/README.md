# Phase 34: Nuisance Linearization

## Objective

Prove that the channel-separation quadratic form on `ker(Gamma)` has a positive spectral gap on the multiplication variety, thereby ruling out the defect condition `W_0 = W_1 = ... = W_{n-1}` for nonzero `w`.

## Phase 33 Input

- Track A found exact recoveries `D_{st} = H M_{st}` on all known exact 3x3 decompositions.
- Every observed `M_{st}` entry lies in `{ -1, 0, 1 }`.
- Track C showed only that tiling-preserving rescalings preserve `Gamma * P_s = I` but generally violate `Gamma * H = 0` and `Gamma * Delta = 0`.
- Therefore the full nuisance cancellation system, not the 81 tiling equations alone, is the candidate source of universal containment.

## Deliverable

Define the exact defect functional on `ker(Gamma)`:

`W_s(w) = Σ_k w_k (alpha_k[:,s] ⊗ beta_k[s,:])`

and prove that for a valid minimum-rank multiplication decomposition,

`max_s ||W_s(w) - W_{s+1}(w)|| > 0`

for every nonzero `w ∈ ker(Gamma)`.

Equivalently: establish a positive lower bound for the minimum eigenvalue of the channel-separation quadratic form restricted to `ker(Gamma)`.

## Planned Route

1. Write `W_s(w) = Σ_k w_k (alpha_k[:,s] ⊗ beta_k[s,:])` and the defect equations `W_0 = W_1 = ... = W_{n-1}` as a linear map on `w`.
2. Restrict that map to `ker(Gamma)` and identify the induced quadratic form measuring channel separation.
3. Express `||W_s - W_t||_F^2` as a rational quadratic form in `w` using the factorized coefficient structure.
4. Relate this quadratic form to the nuisance coordinates `H` and `Delta`, using `Gamma * H = 0` and `Gamma * Delta = 0`.
5. Explain the observed square-root-of-small-rational values on the known exact decompositions as spectral data of that restricted quadratic form.
6. Prove or sharply bound the smallest eigenvalue on the multiplication variety.

## Non-Goals

- Do not search for new decompositions.
- Do not treat tiling-preserving rescalings as algorithm symmetries.
- Do not use floating-point evidence as a proof substitute.
- Do not attempt to prove a universal per-term identity `e_k = 0`; containment is collective, not per-term.

## Artifacts To Reuse

- `phase33_summary.json`
- `phase33b_summary.json`
- `recovery_L_coordinate_matrices.csv`
- `defect_condition_known.csv`
- `defect_condition_basis_vectors.csv`
- `defect_condition_random_controls.csv`

## Success Criterion

Produce a symbolic derivation or exact lower bound showing that the channel-separation quadratic form has a strictly positive spectral gap on valid minimum-rank multiplication decompositions, so no nonzero `w ∈ ker(Gamma)` can satisfy `W_0 = W_1 = ... = W_{n-1}`.