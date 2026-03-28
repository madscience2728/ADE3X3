# Step 57: Fiber-Group Partition Enumeration with Orbit Budget Filter

[EXACT_DERIVED]

The fiber-partition sweep for R = 9..23 is now exact at the composition and symmetry level.
Ordered 9-part compositions are counted by C(R+8, 8), partition types are the integer partitions
of R into at most 9 parts, and symmetry-inequivalent fiber assignments are reduced under the
S3 x S3 action on the 3x3 output-fiber grid by exact Burnside counting.

The exact no-spreading filter keeps only partition types with within-fiber pair budget
sum_i C(n_i, 2) >= 27. Separately, a conservative exact-3 tiling model tests whether the
partition can be realized by rectangular spreading shapes so that every fiber is touched
exactly three times.

Key rows:

- R=9: ordered=24310, types=30, symmetry classes=768, no-spreading survivors=2, exact-3 survivor assignment orbits=416
- R=22: ordered=5852925, types=732, symmetry classes=165152, no-spreading survivors=676, exact-3 survivor assignment orbits=47
- R=23: ordered=7888725, types=887, symmetry classes=222157, no-spreading survivors=851, exact-3 survivor assignment orbits=23

[INTERPRETATION]

This step does not solve coefficients. It gives the exact discrete search front before the
729-equation feasibility problem. The no-spreading filter is strong only for heavily clustered
partitions; the exact-3 tiling model is stronger on the opposite side because it enforces the
9 fibers x 3 channels picture literally. If exact-3 survivors already appear at small R, then the
partition ansatz alone cannot rule out low rank. If exact-3 survivors disappear at some R, that is
only a failure of this conservative tiling model, not yet a theorem against weighted cancellation.
