# Stabilizer-Fixed Seed Spaces

This note derives the explicit linear seed spaces currently realized by the repo's `Z2 wr S3` factor action, using the same action implemented in `scripts/poly_rootfind.py`.

Important caveat:

- These formulas recover the `eff_params` counts used by the existing symmetric-sector code.
- They are the `Stab(seed)`-fixed seed subspaces for a one-seed-per-orbit transport chart.
- The repo's later algebra notes say that the true equivariant family may be strictly larger than this. So these formulas are an explicit chart for the current seed-space model, but they may still be a proper subfamily of the ultimately correct full equivariant chart.

## Group Action

Let `G = Z2^3 ⋊ S3`, `|G| = 48`.

For a group element `(pi, eps)` and factor triple `(A, B, C)`:

- the support triple `(r, s, u)` is sent to `g.(r,s,u)` by permuting coordinates and optionally swapping `1 <-> 2` in each coordinate;
- the factor action is the one implemented in `scripts/poly_rootfind.py::_gf`.

For an orbit representative `seed`, the stabilizer is

`H_seed = { g in G : g.seed = seed }`.

The current seed chart uses a seed triple lying in the fixed space

`(Mat_3 x Mat_3 x Mat_3)^{H_seed}`.

The orbit chart is then

`phi_seed(p) = sum_{g in T_seed} g.(A_seed(p), B_seed(p), C_seed(p))`

where `T_seed` is any set of transporters, one per orbit member.

## Orbit Table

Repo convention:

- Corner: representative `(0,0,0)`, orbit size `1`, stabilizer size `48`, seed-space dimension `3`
- Edge: representative `(0,0,1)`, orbit size `6`, stabilizer size `8`, seed-space dimension `9`
- Face: representative `(0,1,1)`, orbit size `12`, stabilizer size `4`, seed-space dimension `12`
- Interior: representative `(1,1,1)`, orbit size `8`, stabilizer size `6`, seed-space dimension `6`

Warning on naming:

- Some notes elsewhere swap the labels `Face` and `Interior` while keeping the representatives clear.
- The formulas below are keyed by representative and orbit size, so they are unambiguous.

## 1. Corner `(0,0,0)`

`H_corner = G`.

The fixed-space condition forces

- `A = B = C`
- and the common matrix has the form

```text
Corner(a0, a1, a2) =
[[a0, a1, a1],
 [a1, a2, a2],
 [a1, a2, a2]]
```

So the seed chart is

```text
A_seed = B_seed = C_seed = Corner(a0, a1, a2)
```

with `3` free parameters.

## 2. Edge `(0,0,1)`

A convenient generator description is:

- swap the two zero coordinates;
- flip either zero coordinate by `1 <-> 2`;
- do not flip the nonzero coordinate.

This gives `|H_edge| = 8`.

The fixed seed space splits as:

- `A` in the same 3-parameter corner-type space;
- `B = C` in a 6-parameter row-collapsed space.

Explicitly,

```text
A_seed =
[[a0, a1, a1],
 [a1, a2, a2],
 [a1, a2, a2]]
```

```text
B_seed = C_seed =
[[b00, b01, b02],
 [b10, b11, b12],
 [b10, b11, b12]]
```

Total dimension: `3 + 6 = 9`.

## 3. Face `(0,1,1)`

A convenient generator description is:

- flip the zero coordinate freely;
- swap the two equal nonzero coordinates.

This gives `|H_face| = 4`.

The fixed-space condition gives:

- `A = C`
- `A` has rows `2` and `3` equal in the repo's `0,1,2` indexing,
- `B` is symmetric.

Explicitly,

```text
A_seed = C_seed =
[[a00, a01, a02],
 [a10, a11, a12],
 [a10, a11, a12]]
```

```text
B_seed =
[[b00, b01, b02],
 [b01, b11, b12],
 [b02, b12, b22]]
```

Total dimension: `6 + 6 = 12`.

## 4. Interior `(1,1,1)`

Here the stabilizer is the full permutation group on the three equal coordinates, with no allowed `1 <-> 2` flips if the representative must stay fixed as `(1,1,1)`.

So `|H_interior| = 6`.

The fixed-space condition forces

- `A = B = C`
- and the common matrix is an arbitrary symmetric `3 x 3` matrix.

Explicitly,

```text
Sym(u00, u01, u02, u11, u12, u22) =
[[u00, u01, u02],
 [u01, u11, u12],
 [u02, u12, u22]]
```

with

```text
A_seed = B_seed = C_seed = Sym(...)
```

Total dimension: `6`.

## Effective-Parameter Counts for Orbit Configurations

Using the seed-space dimensions above:

- `[1,12]`: `3 + 12 = 15`
- `[1,6,12]`: `3 + 9 + 12 = 24`
- `[8,12]`: `6 + 12 = 18`
- `[1,8,12]`: `3 + 6 + 12 = 21`
- `[1,6,8,12]`: `3 + 9 + 6 + 12 = 30`

These match the existing orbit-level code for `1, 6, 8, 12` except where external notes intentionally use an approximate `R=22` count.

## What This Gives You

For each chosen orbit configuration, the current explicit polynomial chart is:

1. pick seed parameters in the corresponding linear seed spaces above;
2. transport each seed to one factor triple per orbit member using the repo's factor action;
3. sum the resulting rank-1 terms.

This chart is polynomial in the seed parameters because:

- the seed matrices depend linearly on the chosen coordinates;
- transport is linear;
- each tensor term is trilinear in `(A, B, C)`;
- the orbit sum is linear in those term tensors.

## Remaining Caveat

This note gives the explicit chart for the current stabilizer-fixed seed model.

If the later canon note is correct that the genuinely correct symmetric sector is larger than any single-seed transport model, then the next derivation is not the fixed subspace above but the larger induced-equivariant family on the orbit bundle. In that case this note is still useful as the exact description of the currently implemented model, but it is not yet the final chart needed for the true ED-degree computation.