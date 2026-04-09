# ADE3×3 Handoff Prompt — April 4, 2026

## Project Goal
Find a rank-19 CP decomposition of $T_\text{matmul}$ (9×9×9 matrix multiplication tensor) via the **ADE algebraic framework**: derive a 19-dimensional algebra $\mathcal{A}$ whose intrinsic structure encodes the decomposition.

**Hard constraint:** Do NOT clone or compress AlphaTensor's rank-23 solution. Use AlphaTensor only as evidence for universal mechanisms.

---

## Where We Are

### Phase 1 (Linear BFS) — COMPLETE
Full linear axiom BFS over the 27-dim root space (orbit-reduced from 439 triple-orbit variables). All 12 consistent linear states mapped.

| d.o.f. | Axiom set | Notes |
|--------|-----------|-------|
| 27 | `{}` | root |
| 6 | `{HULL_ZD}` | **Phase 2 target** |
| 3 | `{COMM}` | non-commutative dead end |
| 1 | `{R_BLIND}`, `{COMM, R_BLIND}` | trivial constant algebra |
| 0 | 7 combos | zero algebra only |
| INCONS | all `FIBER_ID` combos | forced elements are not identities |

Key negative: `COMM + C3_normalize → d.o.f.=0` → **target algebra is non-commutative**.

### Phase 2 (Quadratic/Associativity on HULL_ZD) — JUST COMPLETED
Built `CANON_ADE/assoc_variety.py`. Found **1 projective solution** (±antipodal pair on S⁵) in the 6-dim HULL_ZD family:

```
alpha ≈ [0.178, -0.389, -0.589, -0.602, 0.318, 0.082]
```

Properties of the found algebra:
- **Non-commutative** (commutativity error = 1.0, as expected)
- **Associativity loss = 8.9e-15** (machine zero — exact)  
- **Fiber means split into two levels:**
  - Free fibers (size 3): mean ≈ −3/19 ≈ −0.1579
  - Forced fibers (size 1): mean ≈ −1/19 ≈ −0.0526
  - (Solution 2 is the negative, same projective point)
- Saved to `CANON_ADE/assoc_solution_1.npy` and `assoc_solution_2.npy`

The eigenvalue analysis confirmed this is a **genuinely quadratic solution** — not in any linear subspace of the 6-dim family.

---

## Immediate Next Steps (Priority Order)

### 1. Characterize the found algebra
Write `CANON_ADE/characterize_solution.py`:
- Load `assoc_solution_1.npy`
- Compute full multiplication table structure (orbit decomposition of f[i,j,k])
- Check nilpotency: does $e_i^2 = 0$ for all i? What is the nil-index?
- Check left/right ideal structure
- Compute the Jacobson radical
- Check if the algebra is graded by any of the 3 G-orbits of INTERIOR

### 2. C3 fiber normalization check
The C3 constraint requires $\sum_{r: (r,s,u)\in\mathcal{I}} f[\xi, (r,s,u), \zeta] = \delta_{s_\xi, s_\zeta} \cdot \delta_{u_\xi, u_\zeta}$ (gamma normalization).
The fiber sums are currently ≈ −1/19 or −3/19 — not the required delta form.
Question: can the solution be **globally rescaled** or **gauge-transformed** to satisfy C3? Or is C3 violated for this algebra?

### 3. Search for more solutions
The gradient descent found 1 isolated solution from 10 seeds. Run more restarts (1000+) with varied learning rates to confirm it's the unique solution (or find a family):
```bash
cd CANON_ADE
python assoc_variety.py  # already has 100k sample base
# Modify to do 1000 gradient descent restarts
```

### 4. Fix the C5 sedenion bridge
The C5 ZD check is currently broken: the bit-pattern mapping from sedenion indices (0–15) to INTERIOR tuples fails for indices 9–15 (upper sector, coordinates not in {0,1,2}).
Need the correct Cayley-Dickson level-4 construction → INTERIOR coordinate map.
See `CANON_ADE/canon_constraints.py` class `C5_SedenionZDGraph`.

### 5. Update ADE_FRAMEWORK.md
Add Phase 2 results: 1 projective solution found, fiber mean pattern, non-commutative confirmation.

---

## Key Files

| File | Purpose |
|------|---------|
| `CANON_ADE/assoc_variety.py` | Phase 2 solver — just written and run |
| `CANON_ADE/assoc_solution_1.npy` | The found associative algebra (19×19×19 tensor) |
| `CANON_ADE/axiom_bfs.py` | BFS framework; exports `build_canon`, `enc_hullzd`, `orbit_of`, `N`, `N_ORBITS` |
| `CANON_ADE/canon_constraints.py` | C1–C5 constraints; `INTERIOR`, `IDX`, `C3_FiberConstraint`, `C5_SedenionZDGraph` |
| `CANON_ADE/algebra_template.py` | `AlgebraTemplate`; `compute_triple_orbits()` → 439 orbits |
| `CANON_ADE/ADE_FRAMEWORK.md` | Master spec — read this first for full context |
| `CANON_ADE/comm_basis.py` | COMM family extraction and testing |
| `CANON_ADE/candidate_algebra.py` | C4+C2+COMM+R_BLIND(3rd) algebra (constant, trivial) |

---

## Key Numbers
- N = 19 (dimension of algebra)
- 439 = triple orbit count (G-reduced structure constant space)
- 27 = root d.o.f. = |INTERIOR²/G|
- 6 = HULL_ZD d.o.f. (Phase 2 target space)
- 3 = COMM d.o.f. (dead end)
- 34 = distinct symmetric 6×6 quadratic constraint matrices for associativity
- Hull indices {5,7,11,13} → INTERIOR points {(1,0,0),(1,0,2),(1,2,0),(1,2,2)}
- 102/439 orbits zeroed by HULL_ZD; each basis vector: 76/439 nonzero

---

## User Constraints (from memory)
- Do NOT try to clone/compress AlphaTensor's rank-23 solution
- Use AlphaTensor only as evidence for universal structure (the *how*, not the *what*)
- Every new continuation/regularity test should include at least one wildcard branch in addition to named structured candidates
- Prefer universal obstructions, staged diagnostics, symmetry/ker(Γ)/H-structure over solution-specific imitation

---

## Environment
- Workspace: `c:\Users\death\OneDrive\Documents\C4 Website\Github\ADE3X3`
- Python venv: `.venv\Scripts\python.exe`
- Run scripts from `CANON_ADE\` subdirectory (imports are relative)
- All scripts use `sys.path.insert(0, os.path.dirname(__file__))` for local imports
