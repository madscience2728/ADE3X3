# ADE3x3 Handoff Prompt: Post-Phase 36 Continuation

You are continuing the ADE3x3 investigation from a completed first-pass Phase 36 analysis.

## First Read

Read these files before making changes:

1. `docs/ADE3x3_CANONICAL_OBJECT.md`
2. `outputs/ade3x3_attack/phase35_universal_pairwise_intersection/RESULTS.md`
3. `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/RESULTS.md`
4. `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/phase36_kernel_saturation_witness_geometry.py`

The canonical object is still the single source of truth. Phase 36 has not yet been canonized.

## Current State

Phase 35 is complete, validated, and canon-integrated.

Its key exact correction is:

`dim(ker(Gamma) ∩ ker D_st ∩ ker D_s't') = dim ker(Gamma) - rank(H)`

for every covering pair choice in the 3x3 case. This is not equal to `eta_nullity` in general. AlphaTensor rank-23 is the standing counterexample to the stronger false claim.

Phase 36 was started directly from that correction and reframed nonsaturation as a witness problem:

`ker(Gamma) ∩ ker(H^T) = { w : Gamma w = 0 and P_0^T w = P_1^T w = P_2^T w }`

So nonsaturation is equivalent to existence of a nonzero witness whose three channel images collapse to one common matrix `C(w)`.

## What Phase 36 Already Established

All three known exact decompositions have witness dimension `0`:

- `strassen_2x2`
- `standard_rank27`
- `alphatensor_rank23`

Structured synthetic right-inverse defect families over fixed `Gamma` do create witness spaces, with defect exactly matching witness dimension.

Wildcard random right-inverse scans showed generic saturation:

- `standard_rank27`: `18/18` saturated trials, `0` deficient
- `alphatensor_rank23`: `18/18` saturated trials, `0` deficient

Interpretation: nonsaturation looks like a special coincidence locus, not a generic phenomenon.

## Important Implementation Facts

These conventions were verified during debugging and matter:

- `channel_faces_exact(...)` returns faces in canonical orientation `P_s ∈ R^(R x n^2)`
- `gamma_matrix_exact(...)` returns `Gamma ∈ R^(n^2 x R)`
- For `n = 3`, use `(P_0 - P_1)^T` and `(P_1 - P_2)^T` in the witness stack
- For the Strassen `n = 2` edge case, only stack one pair equation in addition to `Gamma`

Two real Phase 36 bugs were already fixed:

1. The `2x2` witness-space construction incorrectly assumed three channel faces
2. The face matrices were initially transposed incorrectly

Do not reintroduce those mistakes.

## Files Created In Phase 36

- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/phase36_kernel_saturation_witness_geometry.py`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/RESULTS.md`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/phase36_summary.json`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/known_exact_witness_geometry.csv`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/structured_defect_witness_geometry.csv`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/structured_defect_witness_basis.csv`
- `outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/wildcard_right_inverse_scan.csv`

## Explicit Decision Already Made

Do not canonize Phase 36 as-is unless the user explicitly wants a canon snapshot.

Reason: the current Phase 36 result is useful and validated, but it is still exploratory. It shows a strong pattern, not yet a theorem-level obstruction. The next meaningful step is to sharpen the witness-matrix argument before patching `generate_canon_doc.py`.

## Recommended Next Objective

Start Phase 37 as a direct attack on the common-matrix obstruction.

The right next question is not another spectral experiment. It is:

`What matrices C(w) can occur if w ∈ ker(Gamma)`

and simultaneously

`P_0^T w = P_1^T w = P_2^T w = vec(C(w))`?

Try to turn the measured Phase 36 pattern into an exact obstruction of the form:

- any nonzero witness forces `C(w)` into a constrained family, and
- that family is incompatible with exact multiplication identities in a valid minimum-rank decomposition.

## Good Phase 37 Directions

Pick one or more of these and push to theorem strength if possible:

1. Derive exact linear identities satisfied by `C(w)` from `Gamma w = 0`
2. Compare witness matrices across the known exact decompositions and synthetic defect families to isolate invariant forbidden features
3. Express `C(w)` in terms of the decomposition tensors and look for rank, support, row/column-sum, or bilinear annihilation constraints
4. Test whether a nonzero witness would force a forbidden common output fiber pattern in the standard decomposition template
5. Build an exact symbolic obstruction for AlphaTensor-style entangled families rather than only fiber-aligned ones

## Workflow Requirements

- Put all new results in a new phase directory under `outputs/ade3x3_attack/`
- Write human-readable results to that phase's `RESULTS.md`
- If and only if the new phase produces a canon-worthy structural statement, update the canon through `generate_canon_doc.py`
- Do not edit the canonical object directly
- Preserve the repo constraint: do not try to clone or directly compress AlphaTensor's concrete rank-23 solution to a lower rank
- Use AlphaTensor only as evidence for general mechanisms, not as a pattern to imitate term-by-term
- Include at least one wildcard branch alongside structured candidates

## Fast Resume Command

From repo root:

`python "outputs/ade3x3_attack/phase36_kernel_saturation_witness_geometry/phase36_kernel_saturation_witness_geometry.py"`

That should rerun cleanly and reproduce the current Phase 36 artifacts.

## Bottom Line

Phase 35 converted the target into exact saturation `rank(H) = dim ker(Gamma)`.
Phase 36 showed that failure of saturation can be studied as existence of a common witness matrix `C(w)`, and that generic right-inverse lifts saturate while only structured coincidence families fail.

The next agent should push that witness-matrix geometry into an actual obstruction theorem.