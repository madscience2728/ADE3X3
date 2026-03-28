# Step 63: Reverse Engineering + Cancellation Visualization
Generated: 2026-03-28T16:31:27

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

Step 63 first searched for an explicit sub-27 3x3 factorization. Smirnov coefficients were not recovered from the fetched paper metadata, but the public AlphaTensor repository exposes an exact rank-23 3x3x3 factorization in recombination/example.py, and that coefficient table is used here.

## External Source Status

- Smirnov explicit 23-term coefficients: not_recovered_from_available_sources (Search surfaced 23-term 3x3 papers but not an explicit coefficient table in fetched content.)
- AlphaTensor public recombination example: explicit_rank23_coefficients_recovered (Public get_3x3x3_factorization() provides exact 9x23 factor matrices u,v,w.)
- AlphaTensor benchmarking algorithms: 4x4_public_benchmarking_factorizations_only (The public benchmarking factorization module exposes 4x4 rank-49 GPU/TPU algorithms; the explicit 3x3 rank-23 example is in recombination/example.py.)

## Fiber-Mode Profile

- term_count = 23
- gamma_orientation = transpose
- nuisance_rank = 14
- quotient_gain = 9
- nuisance_equals_R_minus_9 = True
- Gamma*Sigma = 3I_9 verified exactly = True
- Gamma*Nuisance = 0 verified exactly = True

## Removal Feasibility

- single-term feasible removals = 0
- pair feasible removals = 0
- best single removal quotient gain = 8 after removing t01
- best pair removal quotient gain = 8 after removing t03,t06

## Commutator Split

- multiplication (3x3): flattening lower bound 9 from ranks (9, 9, 9)
- commutator (3x3): flattening lower bound 8 from ranks (8, 8, 8)
- anticommutator (3x3): flattening lower bound 9 from ranks (9, 9, 9)

[INTERPRETATION]

The main exact finding is that the public rank-23 3x3 decomposition sits exactly on the Step 52 boundary: its nuisance rank can now be measured directly rather than hypothesized. The removal tests then ask the stronger reverse-engineering question the forward search could not ask: whether this known 23-term solution hides a 22-term or 21-term reweightable sub-decomposition. The HTML visualization records the actual cancellation fabric term-by-term rather than only aggregate ranks.