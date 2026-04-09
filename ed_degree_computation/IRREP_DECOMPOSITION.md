# Stabilizer Irrep Decomposition

This note computes the stabilizer action on the 27-dimensional seed space

`V_seed = Mat_3(A) ⊕ Mat_3(B) ⊕ Mat_3(C)`

using the exact factor action implemented in `scripts/poly_rootfind.py::_gf`.

## Main Conclusion

The nontrivial-isotypic idea is real, but the prompt as written still does not produce the reduced ED chart.

Reason:

- for any stabilizer `H`, the decomposition
  `V_seed = ⊕_rho m_rho V_rho`
  always satisfies
  `sum_rho m_rho dim(rho) = dim(V_seed) = 27`.
- so the proposed "full equivariant parameter count per orbit" is automatically `27` for every orbit type if all `H`-types are included.

That does not match the reduced orbit counts `3, 9, 12, 6`, nor does it by itself define the true reduced symmetric-sector chart.

So the irrep decomposition is useful structural data, but it is not yet the missing ED chart. The remaining missing piece is an explicit induced-bundle construction together with the quotient/gauge mechanism that turns those 27 raw seed coordinates into the desired reduced orbit-family coordinates.

## 1. Edge Orbit

Representative: `(0,0,1)`.

The stabilizer has order `8` and is nonabelian. Its conjugacy-class structure matches `D4`:

- classes and traces on `V_seed`:
  - `e`: size `1`, trace `27`
  - `r^2`: size `1`, trace `7`
  - `{r, r^3}`: size `2`, trace `1`
  - reflection class `s`: size `2`, trace `15`
  - reflection class `sr`: size `2`, trace `3`

One explicit generator pair is:

- `r = ((1,0,2), (False,True,False))`, order `4`
- `s = ((0,1,2), (False,True,False))`, order `2`

With the standard real `D4` character table, the seed representation decomposes as:

`V_edge ≅ A1^9 ⊕ A2^0 ⊕ B1^7 ⊕ B2^1 ⊕ E^5`

where `A1, A2, B1, B2` are the four `1`-dimensional irreps and `E` is the `2`-dimensional irrep.

Checks:

- total dimension: `9*1 + 0*1 + 7*1 + 1*1 + 5*2 = 27`
- trivial multiplicity: `9`, exactly matching the stabilizer-fixed seed dimension of the current edge chart.

## 2. Face Orbit

Representative: `(0,1,1)`.

The stabilizer has order `4`, is abelian, and is isomorphic to `V4 = Z2 × Z2`.

Conjugacy classes are singletons because the group is abelian. The traces on `V_seed` are:

- identity: `27`
- one involution: `15`
- two other involutions: `3`, `3`

Hence the four real `1`-dimensional characters occur with multiplicities:

`V_face ≅ χ00^12 ⊕ χ10^9 ⊕ χ01^3 ⊕ χ11^3`

up to relabeling of the nontrivial characters depending on generator choice.

Checks:

- total dimension: `12 + 9 + 3 + 3 = 27`
- trivial multiplicity: `12`, exactly matching the current face fixed-space chart.

## 3. Interior Orbit

Representative: `(1,1,1)`.

The stabilizer has order `6` and class data matching `S3`:

- identity class: size `1`, trace `27`
- transposition class: size `3`, trace `3`
- `3`-cycle class: size `2`, trace `0`

Using the real `S3` irreps `{triv, sign, std}` with dimensions `{1,1,2}`:

`V_interior ≅ triv^6 ⊕ sign^3 ⊕ std^9`

Checks:

- total dimension: `6*1 + 3*1 + 9*2 = 27`
- trivial multiplicity: `6`, exactly matching the current interior fixed-space chart.

## 4. Corner Orbit

Representative: `(0,0,0)`.

Its stabilizer is the full order-48 group. The trivial multiplicity is already known from the Reynolds projection and equals `3`, matching the current corner chart.

A full named irrep decomposition for the entire order-48 group was not completed here, because it is not needed for the present conclusion. What matters for the current question is:

- the trivial-isotypic multiplicity is `3`;
- the remaining `24` dimensions are nontrivial stabilizer types.

## What This Means

The current stabilizer-fixed chart is precisely the trivial-isotypic part of the seed representation.

The nontrivial-isotypic components are real and large:

- edge: `18` nontrivial dimensions
- face: `15` nontrivial dimensions
- interior: `21` nontrivial dimensions
- corner: `24` nontrivial dimensions

But simply declaring "one seed per irrep type" does not yet define a reduced tensor chart. It only reconstructs the full 27-dimensional raw seed representation orbit-by-orbit.

So the real next derivation is:

1. define the correct induced-equivariant multi-seed bundle over each orbit;
2. specify how the `H`-module seed blocks glue into a well-defined unlabeled CP orbit family;
3. identify the quotient/gauge redundancies on that bundle;
4. only then count the true effective parameters and build the ED-critical system.

Until that bundle-level construction is explicit, the irrep decomposition alone does not close the gap to the true ED chart.