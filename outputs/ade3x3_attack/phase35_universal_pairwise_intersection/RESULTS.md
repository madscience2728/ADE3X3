# Phase 35 Results: Universal Pairwise Intersection
Generated: 2026-03-29T17:46:14

## 35a. Track A: W_s = W_t Variety Structure

[MEASURED_FROM_CODE]

For each pair (s,t), the single-pair nullspace inside ker(Gamma) is the exact solution space
of (P_s - P_t) w = 0 with Gamma w = 0. Phase 35 computes primitive rational bases for these
spaces and pushes each basis vector through the complementary pair maps.

| decomposition | pair | pair nullity | all basis vectors hit every complementary pair? | shared W_s rank set |
|---------------|------|--------------|-----------------------------------------------|-------------------|
| strassen_2x2 | 01 | 0 | True | none |
| standard_rank27 | 01 | 9 | True | 1 |
| standard_rank27 | 02 | 9 | True | 1 |
| standard_rank27 | 12 | 9 | True | 1 |
| alphatensor_rank23 | 01 | 8 | True | 1,2,3 |
| alphatensor_rank23 | 02 | 5 | True | 2,3 |
| alphatensor_rank23 | 12 | 6 | True | 2,3 |

The shared property on the known decompositions is negative but useful: every computed basis
vector of a nontrivial single-pair nullspace has nonzero image under every complementary pair
map that completes the full three-channel constraint. So no measured single-pair basis direction
extends to a two-pair defect.

## 35b. Track B: Three-Channel Constraint and Equivalence

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

For n = 3, the canon definition H = [K_0-K_1 | K_1-K_2] together with K_s = P_s - X_0 gives

  H = [P_0-P_1 | P_1-P_2]    and therefore    H^T = [D_01 ; D_12].

Since D_02 = D_01 + D_12, the common kernel of any two pair maps covering all channels is exactly
ker(H^T). Intersecting with ker(Gamma), the exact theorem is

  dim(ker(Gamma) ∩ ker D_st ∩ ker D_s't') = dim(ker(Gamma)) - rank(H)

for any two pairs whose union is {0,1,2}. This is the left-nullity of H inside ker(Gamma).

| decomposition | dim ker(Gamma) | rank(H) | pairwise intersection dim | dim ker(Gamma)-rank(H) | eta_nullity | equals eta_nullity? |
|---------------|----------------|---------|---------------------------|-----------------------|-------------|---------------------|
| strassen_2x2 | 3 | 3 | 0 | 0 | 1 | False |
| standard_rank27 | 18 | 18 | 0 | 0 | 0 | True |
| alphatensor_rank23 | 14 | 14 | 0 | 0 | 4 | False |

This corrects the tempting but false stronger claim from the prompt: pairwise intersection is
not eta_nullity in general. AlphaTensor is the counterexample already in hand: pairwise
intersection dimension 0, but eta_nullity = 4.

## 35c. Track C: Khatri-Rao Rank Analysis

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

Each pair map D_st = (P_s - P_t)^T has entries alpha_k[r,s] beta_k[s,u] - alpha_k[r,t] beta_k[t,u],
so each output row is a difference of two rank-1 outer-product coefficients. The exact ranks are:

| decomposition | pair | full rank of D_st | rank on ker(Gamma) | pair nullity on ker(Gamma) | gap spectrum |
|---------------|------|-------------------|--------------------|---------------------------|-------------|
| strassen_2x2 | 01 | 3 | 3 | 0 | 2 x 2; 10 x 1 |
| standard_rank27 | 01 | 9 | 9 | 9 | 0 x 9; 2 x 9 |
| standard_rank27 | 02 | 9 | 9 | 9 | 0 x 9; 2 x 9 |
| standard_rank27 | 12 | 9 | 9 | 9 | 0 x 9; 2 x 9 |
| alphatensor_rank23 | 01 | 6 | 6 | 8 | 0 x 8; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 0) x 1; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 1) x 1; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 2) x 1; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 3) x 1; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 4) x 1; CRootOf(x**6 - 48*x**5 + 824*x**4 - 6213*x**3 + 20920*x**2 - 29664*x + 13440, 5) x 1 |
| alphatensor_rank23 | 02 | 9 | 9 | 5 | 0 x 5; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 0) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 1) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 2) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 3) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 4) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 5) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 6) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 7) x 1; CRootOf(x**9 - 39*x**8 + 584*x**7 - 4433*x**6 + 18860*x**5 - 46989*x**4 + 69312*x**3 - 58906*x**2 + 26319*x - 4727, 8) x 1 |
| alphatensor_rank23 | 12 | 8 | 8 | 6 | 0 x 6; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 0) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 1) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 2) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 3) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 4) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 5) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 6) x 1; CRootOf(x**8 - 37*x**7 + 508*x**6 - 3441*x**5 + 12749*x**4 - 26787*x**3 + 31389*x**2 - 18732*x + 4292, 7) x 1 |

For Strassen, D_01 already has rank 3 on ker(Gamma), which equals dim ker(Gamma), so there is no
single-pair nullity at all. That is why the Strassen case collapses immediately to positivity.

## 35d. Track D: Standard Proof Template and Entanglement

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

The exact standard-algorithm template remains clean: ker(Gamma) splits into 9 output fibers, each
a 2-dimensional zero-sum plane. On each fiber, every pair form has spectrum {0,2}, and Q_chan acts
as 3 Id. What changes off the standard algorithm is not the theorem statement, but the loss of fiber
separation.

Standard 3x3 pair spectrum: 0 x 9; 2 x 9
Standard 3x3 total spectrum: 3 x 18

| decomposition | pair nullities | sum pair nullities | fiber-aligned maximum | entanglement deficit | nullity range |
|---------------|----------------|--------------------|----------------------|---------------------|---------------|
| standard_rank27 | 9,9,9 | 27 | 27 | 0 | 0 |
| alphatensor_rank23 | 8,5,6 | 19 | 27 | 8 | 3 |
| strassen_2x2 | 0 | 0 | 4 | 4 | 0 |

Measured only: stronger channel entanglement correlates with smaller single-pair nullities
(standard 27, AlphaTensor 19, Strassen 0 in the summed metric above), but the intersection theorem
survives in both the maximally fiber-aligned and entangled cases.

## 35e. Track E: Tensor Rank Obstruction (WILDCARD)

[WILDCARD]

These are not multiplication decompositions. They are random ternary rank-1 collections with gamma
full rank 9 so the restricted gap can be sampled away from the variety.

| random case | trials | min gap | median gap | max gap | min total nullity | max total nullity |
|-------------|--------|---------|------------|---------|-------------------|-------------------|
| random_3x3_R20 | 6 | 1.320360316652 | 1.754768159871 | 3.379897642944 | 0 | 0 |
| random_3x3_R21 | 6 | 0.337546557910 | 1.080227135305 | 3.487990398370 | 0 | 0 |
| random_3x3_R22 | 6 | 0.390501212171 | 1.258679136473 | 2.602992458262 | 0 | 0 |

This wildcard scan should not be read as a lower-bound theorem. It only says the raw spectral gap
outside the multiplication variety does not exhibit an obvious sharp threshold in these random samples.

## 35f. Track F: η_nullity Structural Constraints

[EXACT_DERIVED] / [MEASURED_FROM_CODE]

The row-space overlap heuristic from the prompt is only partly right. For two projected pair maps
A = D_st|ker(Gamma) and B = D_s't'|ker(Gamma), the quantity rank(A)+rank(B)-rank([A;B]) measures
row-space overlap, not kernel intersection. Exact data:

| decomposition | pair A | pair B | rank A | rank B | joint rank | row-space overlap |
|---------------|--------|--------|--------|--------|------------|-------------------|
| standard_rank27 | 01 | 02 | 9 | 9 | 18 | 0 |
| standard_rank27 | 01 | 12 | 9 | 9 | 18 | 0 |
| standard_rank27 | 02 | 12 | 9 | 9 | 18 | 0 |
| alphatensor_rank23 | 01 | 02 | 6 | 9 | 14 | 1 |
| alphatensor_rank23 | 01 | 12 | 6 | 8 | 14 | 0 |
| alphatensor_rank23 | 02 | 12 | 9 | 8 | 14 | 3 |

So zero pairwise kernel intersection does not force zero row-space overlap. AlphaTensor already has
nonzero overlaps for some pair combinations, even though every two-pair kernel intersection is 0.

## 35g. Interpretation

Phase 35 does produce an exact theorem, but it is sharper than the prompt’s provisional chain and
slightly different. The universal pairwise-intersection statement for n = 3 is exactly equivalent to
H saturating ker(Gamma):

  ker(Gamma) ∩ ker D_st ∩ ker D_s't' = {0}  for all covering pair choices
  if and only if rank(H) = dim ker(Gamma) = R - 9.

That is Theorem A in intersection form. It is not the same as eta_nullity = 0, and AlphaTensor proves
the distinction. So the real remaining universal target is still to show that H fills ker(Gamma)
for every valid minimum-rank decomposition.

What Phase 35 adds is structural control around that target. Single-pair nullspaces can be large, but
on all known exact decompositions every basis direction in a single-pair defect is kicked out by the
complementary pair constraints. The standard proof shows the fully fiber-aligned mechanism; AlphaTensor
shows the entangled mechanism can still end in the same zero-intersection theorem even when the pair
nullities become uneven and the row spaces overlap.
