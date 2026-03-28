# Step 46: Same-Fiber Core + 64-Orbit Sub-Algebra Structure
Generated: 2026-03-28T12:43:53

[EXACT_DERIVED]

## Task 1: Same-Fiber Composition

Total same-fiber x same-fiber witness pairs: 675
- Same-fiber outputs: 675 (100.00%)
- Doubly-live different-fiber outputs: 0 (0.00%)
- Outside the 28-orbit core: 0 (0.00%)

| closure_step | set_size | compatible_orbit_pairs | witness_pairs | new_orbits |
|--------------|----------|------------------------|---------------|------------|
| 1 | 10 | 12 | 675 | 0 |

## Task 3: Focused Orbits

Focused orbits: 6
| orbit_id | focus_boundary | fiber_positions | rep |
|----------|----------------|-----------------|-----|
| 0 | both | (0, 0) | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,0]] |
| 1 | c1 | (0, 0) | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[0,1]] |
| 2 | c1 | (0, 0) | CXXC[C[0,0],X[0,0|0,0],X[0,0|0,0],C[1,0]] |
| 30 | both | (0, 1) | CXXC[C[0,0],X[0,0|0,0],X[0,1|1,0],C[0,0]] |
| 132 | c2 | (0, 0) | CXXC[C[0,0],X[0,0|0,1],X[0,0|0,1],C[0,1]] |
| 1057 | c2 | (0, 0) | CXXC[C[0,0],X[1,0|0,0],X[1,0|0,0],C[1,0]] |

Focused x focused witness pairs: 459
- Focused outputs: 459 (100.00%)
- Same-fiber outputs: 459 (100.00%)

## Task 2: 64-Orbit Sub-Algebra Table

- Compatible orbit pairs: 602 / 4096
- Deterministic compatible pairs: 438 (72.76%)
- Mixed compatible pairs: 164 (27.24%)
- Output coverage: 64 / 64 orbits (100.00%)

## Task 4: Generator Analysis

Orbit 0 identity-compatible rows behaving exactly as identity: 18 / 18
Smallest nontrivial orbit singleton-closure seed: 1
Singleton closure terminal status: fixed_point at size 1
Greedy generating set size: 18
Greedy generating set: [1, 2, 4, 5, 6, 50, 52, 54, 125, 126, 127, 143, 191, 354, 980, 982, 984, 1175]

[INTERPRETATION]

The same-fiber layer is more rigid than the full 28-orbit live core, but the decisive
object is the 64-orbit sub-algebra: every later restricted motif lives inside it. The
focused subset isolates the most constrained same-fiber configurations, while the full
64x64 table and greedy closure expose whether the small algebra is monogenic or requires
several independent orbit types to generate.