# BUILD SPEC: R=19 Matrix Multiplication Tensor — Exhaustive Enumeration Pipeline

## READ FIRST — Project Context

You are building an exhaustive combinatorial search pipeline to determine whether the 3×3 matrix multiplication tensor has a rank-19 CP decomposition over small integer coefficients.

**Read every file in `./docs/` before writing any code.** The critical files are:
- `CANON_CONSTRAINTS.md` — the gate hierarchy (Gate 1/2/3), exact constraint formulas
- `SYMMETRY_CANON.md` — the Z₂ ≀ S₃ wreath product symmetry, orbit structure, and the impossibility proof for seed-transport
- `combinatorics_note.md` — the tier-by-tier pruning analysis showing feasibility
- `gate_progress.md` — algebraic experiment results (Gate 2 is free, Gate 3 is the bottleneck)
- `ranks.md` — rank feasibility test results in the symmetric sector
- `canon_index.md` — line-range index into the full canonical object
- `ADE3x3_CANONICAL_OBJECT.md` — the full 16,319-line canonical dossier (ground truth), for refference only, use the canon index to navigate

**This is NOT an optimization problem. Do NOT use gradient descent, LP relaxation, or any continuous optimizer.** The solution (if it exists) lies on a measure-zero algebraic variety. We are performing structured enumeration with algebraic pruning.

---

## HARDWARE & ENVIRONMENT

| Resource | Spec |
|----------|------|
| CPU | AMD Ryzen 9 5900X — 12 cores / 24 threads |
| RAM | 80 GB DDR4 |
| GPU | NVIDIA RTX 3060 — 12 GB VRAM |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Workers | 24 (use all threads) |

**Language:** Python 3.11+ with NumPy, SciPy, and Rich for TUI.
**Performance-critical inner loops:** Use Numba `@njit` with `cache=True` and `parallel=True` where applicable. If a loop body is pure numpy linear algebra (SVD, rank, det), batch it and use `np.linalg.svd` on stacked arrays.
**GPU acceleration (Phase 3 batched rank):** Optional CuPy path for batched SVD on the 3060. Gate behind `--gpu` flag; CPU fallback must always work.

All output files go in `./CANON OPTIMIZER/core/`.

---

## THE MATHEMATICAL PROBLEM

### The Tensor

The 3×3 matrix multiplication tensor T has shape (9, 9, 9) with entries:

```
T[r*3+s, s*3+u, r*3+u] = 1    for all (r, s, u) ∈ {0,1,2}³
```

All other entries are 0. There are exactly 27 nonzero entries.

### What We Seek

Find 19 rank-1 terms, each specified by three 3×3 matrices (α_k, β_k, γ_k), such that:

```
T[i,j,l] = Σ_{k=1}^{19} α_k[i] · β_k[j] · γ_k[l]    for all i,j,l ∈ {0..8}
```

where α_k, β_k are 3×3 matrices (9 entries each) and γ_k is determined by (α_k, β_k) via linear algebra (see Gate 3 below).

### Coordinate Decomposition (from CANON_CONSTRAINTS.md §40)

Each pair (α_k, β_k) generates these coordinate blocks:

```python
# Sigma: fiber-sum (carries the target signal)
# Shape: (R, 9), where R=19
Sigma[k, r*3+u] = sum_s alpha_k[r,s] * beta_k[s,u]    # s ∈ {0,1,2}

# Eta1: first anisotropy
Eta1[k, r*3+u] = alpha_k[r,0]*beta_k[0,u] - alpha_k[r,1]*beta_k[1,u]

# Eta2: second anisotropy
Eta2[k, r*3+u] = alpha_k[r,1]*beta_k[1,u] - alpha_k[r,2]*beta_k[2,u]

# H: combined anisotropy, shape (R, 18)
H[k, :] = [Eta1[k,:], Eta2[k,:]]

# Delta: dead-X coordinates, shape (R, 54)
# For each (r,u) and each pair (s,t) with s≠t:
Delta[k, index(r,s,t,u)] = alpha_k[r,s] * beta_k[t,u]
# There are 9 × 6 = 54 such entries per term

# Nuisance: full nuisance block, shape (R, 72)
Nuisance[k, :] = [H[k,:], Delta[k,:]]
```

### The Master Equation

The 729 tensor equations decompose as:

```
Γ · Σ     = 3 I₉       (81 equations — the target)
Γ · Eta1  = 0           (81 equations)
Γ · Eta2  = 0           (81 equations)
Γ · Delta = 0           (486 equations)
```

Where Γ is 9×19 (this is what we call γ reshaped). **γ is NOT a free variable** — it is determined by solving the linear system above given (α, β).

---

## THE GATE HIERARCHY

### Gate 1: rank(H) = 10

The 19×18 matrix H = [Eta1 | Eta2] must have rank exactly 10.

**Why 10:** The conservation law R + η_nullity = 27 requires η_nullity = 8, which means rank(H) = 18 − 8 = 10. Equivalently, dim(ker Γ) = R − 9 = 10, and H must span all of ker(Γ).

**Pruning power:** For random (α, β) over {-1, 0, 1}, roughly 1 in 10⁶ to 10⁸ configurations achieve rank(H) = 10. This is the primary filter.

**Implementation:** Compute SVD of the current partial H matrix. If rank already exceeds 10 with fewer than 19 terms placed, prune the branch immediately. Use incremental rank tracking: maintain the singular values of H as terms are added one by one.

### Gate 2: Δ ⊂ span(H)

The 19×54 matrix Delta must have all its columns in the column span of H. Equivalently: rank([H | Delta]) = rank(H).

**From gate_progress.md:** This is generically free for R ≤ 21. At generic random points, rk(Nuisance) = rk(H) always. However, on the rank(H)=10 subvariety it may fail for specific configurations. Check it, but expect most Gate 1 survivors to pass.

**Implementation:** After confirming rank(H) = 10, compute rank([H | Delta]). If it exceeds 10, reject.

### Gate 3: Γ Solvability

Given (α, β) passing Gates 1 and 2, solve for Γ:

```
Solve: Γ · [Σ | Nuisance]ᵀ = [3I₉ | 0]ᵀ
Check: rank([Σ | Nuisance]) = rank(Nuisance) + 9 = 19
```

If the augmented rank is 19, Γ exists and is unique. Compute it. Reconstruct the full tensor. Check ‖T − reconstruction‖_∞ = 0 (exact, since we're over integers).

**Implementation:** `np.linalg.lstsq` or exact rational solve. If the system is consistent (residual = 0), we have an exact R=19 decomposition. **This is the finish line.**

---

## THE ENUMERATION PIPELINE

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MASTER COORDINATOR                     │
│  Distributes fiber partitions across 24 workers           │
│  Aggregates results, manages checkpointing                │
│  Rich TUI: live progress, ETA, gate statistics            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐          │
│  │ Worker 1 │  │ Worker 2 │  ...  │Worker 24 │          │
│  │          │  │          │       │          │          │
│  │ Tier 1   │  │ Tier 1   │       │ Tier 1   │          │
│  │ Tier 2   │  │ Tier 2   │       │ Tier 2   │          │
│  │ Tier 3   │  │ Tier 3   │       │ Tier 3   │          │
│  │ Tier 4   │  │ Tier 4   │       │ Tier 4   │          │
│  │ Tier 5   │  │ Tier 5   │       │ Tier 5   │          │
│  │ Tier 6   │  │ Tier 6   │       │ Tier 6   │          │
│  └──────────┘  └──────────┘       └──────────┘          │
│                                                           │
├─────────────────────────────────────────────────────────┤
│                   CHECKPOINT STORE                        │
│  JSON-lines file with completed partitions + survivors    │
│  Resume from any interruption point                       │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
./CANON_OPTIMIZER/core/
├── tensor.py              # T construction, coordinate block computation
├── gates.py               # Gate 1/2/3 check functions
├── fiber.py               # Fiber partition enumeration and per-fiber coupling
├── symmetry.py            # Z₂ ≀ S₃ canonicalization (deduplication only)
├── enumerator.py          # The tree-search engine (Tiers 1-6)
├── worker.py              # Single-worker search loop
├── coordinator.py         # Multi-process coordinator + checkpointing
├── dashboard.py           # Rich TUI dashboard
├── config.py              # Search parameters, coefficient field, hardware settings
├── main.py                # Entry point
└── verify.py              # Independent verification of any solution found
```

---

## FILE-BY-FILE SPECIFICATIONS

### `tensor.py` — Ground Truth Tensor & Coordinate Blocks

```python
"""
Construct the 3×3 matrix multiplication tensor and compute all
coordinate blocks (Sigma, Eta1, Eta2, Delta, H, Nuisance) from
a set of (alpha, beta) factor pairs.

All arrays use int64 when working over integer coefficient fields.
Use exact integer arithmetic throughout — no floats until SVD.
"""
```

**Functions to implement:**

1. `build_tensor() -> np.ndarray` — Returns T as shape (9, 9, 9) int array. T[r*3+s, s*3+u, r*3+u] = 1.

2. `compute_sigma(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray` — Given alpha (R, 3, 3) and beta (R, 3, 3), compute Sigma (R, 9). Entry [k, r*3+u] = Σ_s α_k[r,s]·β_k[s,u].

3. `compute_eta1(alpha, beta) -> np.ndarray` — Shape (R, 9). Entry [k, r*3+u] = α_k[r,0]·β_k[0,u] − α_k[r,1]·β_k[1,u].

4. `compute_eta2(alpha, beta) -> np.ndarray` — Shape (R, 9). Entry [k, r*3+u] = α_k[r,1]·β_k[1,u] − α_k[r,2]·β_k[2,u].

5. `compute_H(alpha, beta) -> np.ndarray` — Shape (R, 18). Horizontal concatenation [Eta1 | Eta2].

6. `compute_delta(alpha, beta) -> np.ndarray` — Shape (R, 54). For each term k, for each (r, u) ∈ {0,1,2}², for each ordered pair (s, t) with s ≠ t (6 pairs), compute α_k[r,s]·β_k[t,u]. Use a consistent indexing: delta_col = r*18 + u*6 + pair_index, where pair_index enumerates (0,1), (0,2), (1,0), (1,2), (2,0), (2,1).

7. `compute_nuisance(alpha, beta) -> np.ndarray` — Shape (R, 72). Horizontal concatenation [H | Delta].

8. `compute_all_blocks(alpha, beta) -> dict` — Returns dict with keys 'Sigma', 'H', 'Delta', 'Nuisance', each as np.ndarray. Compute once, reuse.

9. `reconstruct_tensor(alpha, beta, gamma) -> np.ndarray` — Full reconstruction: T_hat[i,j,l] = Σ_k γ_k.flat[l] · α_k.flat[i] · β_k.flat[j]. Returns shape (9,9,9). Alternatively, use the outer-product-sum formulation.

**Testing:** Include a self-test that verifies the standard R=27 algorithm (α_k = e_r e_s^T, β_k = e_s e_u^T, γ_k = e_r e_u^T for all (r,s,u) ∈ {0,1,2}³) reconstructs T exactly, and that its coordinate blocks satisfy all gate conditions (rank(H)=18, rank(Nuisance)=18, etc.).

---

### `gates.py` — Gate Check Functions

```python
"""
Algebraic gate checks for the R=19 constraint hierarchy.
All functions return (passed: bool, diagnostics: dict).
Use integer/rational arithmetic where possible; float SVD for rank.
"""
```

**Functions:**

1. `check_gate1(H: np.ndarray, target_rank: int = 10) -> tuple[bool, dict]`
   - Compute SVD of H (cast to float64 for SVD).
   - Determine numerical rank using threshold: singular values < 1e-10 × max(sv) are zero.
   - Return `(rank == target_rank, {'rank_H': rank, 'singular_values': sv, 'eta_nullity': 18 - rank})`.
   - Also check conservation: R + (18 - rank) should equal 27.

2. `check_gate2(H: np.ndarray, Delta: np.ndarray) -> tuple[bool, dict]`
   - Compute rank(H) and rank([H | Delta]).
   - Passed iff rank([H | Delta]) == rank(H).
   - Return `(passed, {'rank_H': rk_H, 'rank_nuisance': rk_N, 'delta_leak': rk_N - rk_H})`.

3. `check_gate3(Sigma: np.ndarray, Nuisance: np.ndarray, R: int = 19) -> tuple[bool, dict]`
   - Compute rank([Sigma | Nuisance]) and rank(Nuisance).
   - Passed iff rank([Sigma | Nuisance]) == rank(Nuisance) + 9.
   - If passed, solve for Gamma: build the system `[Sigma | Nuisance]^T @ Gamma^T = [3*I_9 | 0]^T`.
   - Use `np.linalg.lstsq`. Check residual < 1e-12.
   - Return `(passed, {'augmented_rank': aug_rk, 'nuisance_rank': nuis_rk, 'gamma': Gamma_if_solved, 'residual': residual})`.

4. `check_all_gates(alpha, beta) -> tuple[bool, dict]`
   - Compute all blocks via `tensor.compute_all_blocks`.
   - Run Gate 1. If fail, return early.
   - Run Gate 2. If fail, return early.
   - Run Gate 3. If fail, return early.
   - If all pass, reconstruct tensor and verify ‖T − T_hat‖_∞ = 0.
   - Return `(exact_solution_found, full_diagnostics)`.

5. `incremental_rank_check(H_partial: np.ndarray, max_rank: int) -> bool`
   - For use during tree search: given the H matrix built from terms placed so far, check if rank already exceeds max_rank. If so, no extension can bring it back down — prune.
   - This is the key pruning function called millions of times. Must be fast.
   - For small matrices (< 19 rows), use `np.linalg.matrix_rank` or SVD.

---

### `fiber.py` — Fiber Partition Enumeration & Per-Fiber Coupling

```python
"""
Enumerate fiber partitions and valid per-fiber (alpha, beta) configurations.

Each of the 19 terms primarily "serves" one of 9 output fibers.
A fiber is indexed by (r, u) ∈ {0,1,2}², giving 9 fibers.
The fiber assignment determines which entries of alpha and beta are
structurally active for each term.

The term k serving fiber (r, u) contributes:
  Sigma[k, r*3+u] = Σ_s alpha_k[r,s] * beta_k[s,u]  (diagonal contribution)
  Sigma[k, r'*3+u'] for (r',u') ≠ (r,u)              (off-diagonal cross-talk)

For the diagonal to work, we need Σ_k gamma_k * sigma_k = 3 (on-fiber)
and = 0 (off-fiber).
"""
```

**Functions:**

1. `enumerate_fiber_partitions(R: int = 19, n_fibers: int = 9) -> list[tuple]`
   - Enumerate all partitions of R=19 terms into 9 fibers.
   - Each fiber gets at least 1 term (pigeonhole: 19 terms, 9 fibers).
   - Most fibers get 2 terms, one fiber gets 3 (since 19 = 9×2 + 1).
   - Or: some fibers get 1, others get more. Enumerate all compositions of 19 into 9 positive parts.
   - **Symmetry reduction:** The S₃ × S₃ part of the group permutes fibers. Use this to canonicalize partitions. Two partitions that are related by permuting fibers are equivalent. Only keep one representative per equivalence class.
   - Return list of partition tuples, e.g., [(2,2,2,2,2,2,2,2,3), ...] in canonical form.

2. `enumerate_fiber_configs(partition: tuple, coeff_field: list) -> generator`
   - For a given fiber partition and coefficient field (e.g., [-1, 0, 1]):
   - For each fiber (r, u) with n_k terms assigned:
     - Enumerate all n_k-tuples of (alpha_row_r, beta_col_u) vectors where each vector has 3 entries from coeff_field.
     - alpha_row_r is row r of alpha_k (3 entries: alpha_k[r, 0], alpha_k[r, 1], alpha_k[r, 2]).
     - beta_col_u is column u of beta_k (3 entries: beta_k[0, u], beta_k[1, u], beta_k[2, u]).
     - sigma_k = dot(alpha_row_r, beta_col_u) — this is the fiber-sum contribution.
   - **Per-fiber coupling constraint:** The gamma values for terms in this fiber must satisfy gamma_1 * sigma_1 + gamma_2 * sigma_2 = 3 (for the on-diagonal equation) and the off-diagonal contributions must cancel. This constrains which (sigma_1, sigma_2) pairs are valid.
   - **Prune immediately:** If no gamma values over the coefficient field (or rationals) can satisfy the coupling, skip.
   - Yield valid per-fiber configurations.

3. `build_alpha_beta_from_fibers(fiber_configs: dict) -> tuple[np.ndarray, np.ndarray]`
   - Given the per-fiber configurations, assemble the full (19, 3, 3) alpha and beta arrays.
   - Note: each term k serving fiber (r, u) has its active row r of alpha and active column u of beta determined by the fiber config. The OTHER rows of alpha_k and columns of beta_k contribute to cross-talk (off-diagonal Sigma entries and Delta entries).
   - **Critical insight:** The non-active entries of alpha and beta are the primary degrees of freedom for satisfying Gates 1-3. The fiber assignment pins the active entries; the remaining entries must be searched.

---

### `symmetry.py` — Group Action & Canonicalization

```python
"""
Z₂ ≀ S₃ = Z₂³ ⋊ S₃ (order 48) acting on {0,1,2}³.

Used ONLY for deduplication (canonical form), NOT for parameterization.
The seed-transport parameterization is algebraically incompatible with
rk(Σ) = 9 (see SYMMETRY_CANON.md impossibility theorem). We search the
full parameter space and use symmetry only to skip equivalent configurations.

The group acts on the 27 index triples (r, s, u) ∈ {0,1,2}³.
For R=19, the 19 kept triples are those with at least one zero coordinate.
The group permutes these 19 triples among themselves (the partition
19 = 1+6+12 is a union of orbits).
"""
```

**Functions:**

1. `generate_group_elements() -> list[tuple]`
   - Generate all 48 elements of Z₂ ≀ S₃.
   - Each element is (π, ε) where π ∈ S₃ (permutation of 3 coordinates) and ε ∈ Z₂³ (swap flags for each coordinate, swapping 1↔2 and fixing 0).
   - Return as list of (perm, swaps) tuples.

2. `apply_to_triple(g, triple) -> tuple`
   - Apply group element g = (π, ε) to triple (r, s, u):
     - Permute coordinates: (r, s, u) → (x[π[0]], x[π[1]], x[π[2]])
     - Apply swaps: if ε[i] = 1, swap 1↔2 in coordinate i (0 stays fixed)

3. `apply_to_config(g, alpha, beta, gamma) -> tuple`
   - Apply group element to a full R=19 configuration.
   - The group permutes which of the 19 terms is which, AND transforms the factor entries.
   - Return the transformed (alpha', beta', gamma') arrays.

4. `canonicalize(alpha, beta) -> tuple`
   - Apply all 48 group elements to the configuration.
   - For each transformed version, compute a canonical hash (e.g., tuple of sorted flattened entries, or lexicographic comparison).
   - Return the lexicographically smallest version.
   - **This is used to skip duplicate branches in the search tree.**

5. `canonical_hash(alpha, beta) -> int`
   - Fast hash of canonical form for dedup set lookup.

---

### `enumerator.py` — The Tree Search Engine

```python
"""
Core tree-search engine implementing the Tier 1-6 cascade.

Architecture:
- Depth-first search with pruning at each tier
- Each node in the tree = a partial assignment of terms
- Children = extensions by one more term
- Pruning = gate checks on the partial configuration

The search tree has 19 levels (one per term). At each level, we choose
(alpha_k, beta_k) for term k from the coefficient field.
"""
```

**Key design decisions:**

1. **Term ordering:** Process terms in a fixed order that maximizes early pruning. Start with the terms that most constrain rank(H). The Face orbit (12 terms) contributes rank 4 to H per the gate_progress experiments, so its terms are highly constraining. Suggested order:
   - First 3-4 Face terms (establishes rank structure of H early)
   - Then Edge terms
   - Then Corner term
   - Then remaining Face terms

2. **Coefficient field:** Start with `{-1, 0, 1}`. Each alpha_k is a 3×3 matrix with entries from this set: 3⁹ = 19,683 possibilities per alpha. Same for beta. But most are equivalent under row/column scaling symmetries, and many produce degenerate (all-zero) rows.

3. **Pruning checkpoints:**

```
After placing term k:
  ├── Is any row of alpha_k all-zero? → skip (degenerate term)
  ├── Is any column of beta_k all-zero? → skip (degenerate term)
  ├── Compute H so far (k rows of the 19×18 matrix)
  ├── rank(H[0:k+1, :]) > 10? → PRUNE (rank can only increase)
  ├── If k ≥ 10: rank(H[0:k+1, :]) < k+1-9? → PRUNE (too few independent rows)
  ├── k == 18 (all terms placed):
  │     ├── Check Gate 1: rank(H) == 10? → if no, reject
  │     ├── Check Gate 2: rank([H|Δ]) == 10? → if no, reject
  │     ├── Check Gate 3: solve for Γ → if solvable, SOLUTION FOUND
  │     └── Log diagnostics either way
  └── Continue to term k+1
```

4. **Incremental computation:** Don't recompute H from scratch at each node. Maintain a running H matrix and append one row when extending. The SVD/rank check on the partial matrix is the bottleneck — use the smallest matrix possible.

5. **Symmetry deduplication:** Before extending at each level, canonicalize the partial configuration and check against a seen-set. If already explored, prune. Use the canonical_hash function for O(1) lookup.

**Main function:**

```python
def search(coeff_field: list[int],
           fiber_partition: tuple,
           term_order: list[int],
           seen_set: set,
           callback: Callable,  # called on Gate 1 survivors
           progress_callback: Callable  # for dashboard updates
           ) -> list[dict]:
    """
    Returns list of solutions (each a dict with alpha, beta, gamma, diagnostics).
    Empty list if no solution exists for this partition.
    """
```

---

### `worker.py` — Single Worker Process

```python
"""
Worker process that receives a fiber partition (or sub-partition)
and runs the full enumeration pipeline on it.

Communicates with the coordinator via multiprocessing.Queue:
- Receives: work unit (fiber partition + term range to explore)
- Sends: progress updates, gate statistics, solutions, completion signal
"""
```

**Message types sent to coordinator:**

```python
@dataclass
class ProgressUpdate:
    worker_id: int
    partition: tuple
    terms_placed: int
    candidates_checked: int
    gate1_passes: int
    gate2_passes: int
    gate3_passes: int
    solutions_found: int
    elapsed_seconds: float
    estimated_remaining: float  # seconds
    current_branch: str  # human-readable description of current search branch

@dataclass
class SolutionFound:
    worker_id: int
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    fitness: float  # should be 0.0 for exact solution
    diagnostics: dict

@dataclass
class WorkerDone:
    worker_id: int
    partition: tuple
    total_checked: int
    total_gate1: int
    total_gate2: int
    total_gate3: int
    wall_time: float
```

---

### `coordinator.py` — Multi-Process Coordinator

```python
"""
Master coordinator that distributes work across 24 worker processes
and aggregates results.

Uses multiprocessing.Pool or ProcessPoolExecutor.
Manages checkpointing (saves completed partitions to disk).
Handles graceful shutdown on Ctrl+C.
"""
```

**Responsibilities:**

1. **Work distribution:** Generate all fiber partitions. Distribute them round-robin or by estimated difficulty (larger partitions first for load balancing).

2. **Checkpointing:** Every 60 seconds (configurable), write a checkpoint file:
   ```json
   {
     "timestamp": "2026-04-04T12:00:00",
     "coeff_field": [-1, 0, 1],
     "completed_partitions": ["..."],
     "in_progress": ["..."],
     "total_checked": 1234567890,
     "gate1_total": 456,
     "gate2_total": 123,
     "gate3_total": 0,
     "solutions": [],
     "elapsed_hours": 2.5
   }
   ```

3. **Resume:** On startup, load checkpoint and skip completed partitions.

4. **Solution handling:** If any worker finds a solution, immediately:
   - Save it to `./CANON_OPTIMIZER/core/SOLUTION.json`
   - Run `verify.py` to independently confirm
   - Signal all other workers to stop (optional: continue searching for more solutions)
   - Print solution to console with Rich formatting

5. **Graceful shutdown:** On SIGINT/SIGTERM, save checkpoint and exit cleanly.

---

### `dashboard.py` — Rich TUI Dashboard

```python
"""
Real-time dashboard using Rich library.
Updates every 0.5 seconds.
"""
```

**Layout (use Rich Layout + Live display):**

```
╔══════════════════════════════════════════════════════════════════════╗
║  R=19 MATRIX MULTIPLICATION TENSOR — EXHAUSTIVE SEARCH             ║
║  Coefficient field: {-1, 0, 1}                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PROGRESS                                                            ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  67.3%                 ║
║  Partitions: 147 / 219 completed          ETA: 1h 23m 15s           ║
║  Wall time: 2h 47m 02s                    Rate: 1.2M candidates/s   ║
║                                                                      ║
║  GATE STATISTICS                                                     ║
║  ┌─────────────┬──────────────┬───────────┬────────────┐            ║
║  │ Gate        │ Checked      │ Passed    │ Pass Rate  │            ║
║  ├─────────────┼──────────────┼───────────┼────────────┤            ║
║  │ Candidates  │ 847,293,102  │ —         │ —          │            ║
║  │ Gate 1      │ 847,293,102  │ 1,247     │ 1.47e-06   │            ║
║  │ Gate 2      │ 1,247        │ 1,198     │ 96.1%      │            ║
║  │ Gate 3      │ 1,198        │ 0         │ 0.0%       │            ║
║  │ SOLUTIONS   │ —            │ 0         │ —          │            ║
║  └─────────────┴──────────────┴───────────┴────────────┘            ║
║                                                                      ║
║  WORKER STATUS                                                       ║
║  ┌────┬────────────────┬───────────┬──────────┬─────────────┐       ║
║  │ ID │ Partition      │ Progress  │ Rate     │ Gate1 hits  │       ║
║  ├────┼────────────────┼───────────┼──────────┼─────────────┤       ║
║  │  0 │ (2,2,2,2,2,3)  │ 34.2%     │ 52K/s    │ 3           │       ║
║  │  1 │ (2,2,2,2,2,3)  │ 78.1%     │ 48K/s    │ 7           │       ║
║  │  2 │ (2,2,2,2,3,2)  │ 12.7%     │ 55K/s    │ 1           │       ║
║  │ .. │ ...            │ ...       │ ...      │ ...         │       ║
║  │ 23 │ (2,2,3,2,2,2)  │ 91.4%     │ 51K/s    │ 5           │       ║
║  └────┴────────────────┴───────────┴──────────┴─────────────┘       ║
║                                                                      ║
║  BEST NEAR-MISS (if no exact solution yet)                           ║
║  rank(H)=10  rank(N)=10  aug_rank=18  fitness=0.142                 ║
║                                                                      ║
║  CONSERVATION LAW: R + η_null = 19 + 8 = 27 ✓                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Rich components to use:**
- `rich.live.Live` for the auto-updating display
- `rich.table.Table` for gate statistics and worker status
- `rich.progress.Progress` with custom columns for the main progress bar
- `rich.panel.Panel` for the outer frame
- `rich.layout.Layout` for the grid structure
- `rich.text.Text` for colored status indicators (green for pass, red for fail, yellow for in-progress)
- `rich.console.Console` for any logging output below the dashboard

**Color scheme:**
- Gate 1 passes: yellow (interesting but not sufficient)
- Gate 2 passes: cyan (promising)
- Gate 3 passes: bright green + blinking (SOLUTION FOUND)
- Worker idle: dim
- Worker active: bright white
- ETA: magenta

---

### `config.py` — Configuration

```python
"""
Central configuration. All magic numbers in one place.
"""

from dataclasses import dataclass, field

@dataclass
class SearchConfig:
    # Problem parameters
    n: int = 3                          # matrix size (3×3)
    R: int = 19                         # target rank
    target_rank_H: int = 10             # = R - n² + n... = R - 9
    target_nuisance_rank: int = 10      # = target_rank_H when Gate 2 holds
    target_augmented_rank: int = 19     # = R (full rank)
    conservation_target: int = 27       # = n³

    # Coefficient field
    coeff_field: list = field(default_factory=lambda: [-1, 0, 1])

    # Hardware
    n_workers: int = 24
    use_gpu: bool = False               # --gpu flag enables CuPy batched SVD
    gpu_batch_size: int = 4096          # number of H matrices per GPU batch

    # Checkpointing
    checkpoint_interval_seconds: int = 60
    checkpoint_file: str = "./CANON_OPTIMIZER/core/checkpoint.json"
    solution_file: str = "./CANON_OPTIMIZER/core/SOLUTION.json"

    # Pruning thresholds
    rank_tolerance: float = 1e-10       # SVD threshold for numerical rank
    gate3_residual_tolerance: float = 1e-12

    # Dashboard
    dashboard_refresh_rate: float = 0.5  # seconds between Rich updates

    # Logging
    log_file: str = "./CANON_OPTIMIZER/core/search.log"
    log_gate1_survivors: bool = True     # save all Gate 1 survivors to disk
    gate1_survivor_file: str = "./CANON_OPTIMIZER/core/gate1_survivors.jsonl"
```

---

### `main.py` — Entry Point

```python
"""
Entry point for the R=19 exhaustive search.

Usage:
    python main.py                     # full search over {-1, 0, 1}
    python main.py --coeff 2           # search over {-2, -1, 0, 1, 2}
    python main.py --gpu               # enable GPU-accelerated rank checks
    python main.py --resume            # resume from checkpoint
    python main.py --verify FILE       # verify a candidate solution
    python main.py --workers 12        # override worker count
    python main.py --dry-run           # enumerate partitions only, report counts
    python main.py --test              # run self-tests on tensor.py and gates.py
"""
```

**Startup sequence:**

1. Parse CLI args (use `argparse`).
2. Run self-tests (`--test` or always on first run): verify standard R=27 passes all gates, verify a random R=19 config fails, verify group elements have order dividing 48.
3. Enumerate fiber partitions. Print count to console.
4. If `--resume`, load checkpoint and filter out completed partitions.
5. If `--dry-run`, print partition table with estimated candidate counts per partition, then exit.
6. Launch coordinator with Rich dashboard.
7. On completion: print summary statistics. If solution found, print it. If not, print "No R=19 decomposition exists over {coeff_field}."

---

### `verify.py` — Independent Solution Verification

```python
"""
Given a candidate (alpha, beta, gamma), verify independently that it
exactly reconstructs the matrix multiplication tensor.

This file must have ZERO dependencies on the rest of the codebase
(except numpy). It rebuilds T from scratch and checks every entry.
"""
```

**Checks:**
1. Rebuild T from definition.
2. Reconstruct T_hat = Σ_k γ_k ⊗ α_k ⊗ β_k (using explicit triple loop, no library calls beyond numpy).
3. Check T == T_hat entry by entry (exact integer comparison, no floating point).
4. Print "VERIFIED: Exact R=19 decomposition confirmed." or "FAILED: Mismatch at entry (i,j,l)."
5. Additionally verify all gate conditions: rank(H)=10, rank(Nuisance)=10, conservation law.
6. Print the 19 terms in human-readable format.

---

## CRITICAL IMPLEMENTATION NOTES

### 1. Alpha and Beta Are Full 3×3 Matrices

Each term k has alpha_k ∈ ℤ^{3×3} and beta_k ∈ ℤ^{3×3}. That's 18 integers per term, 342 total for 19 terms. The fiber assignment tells you which (r, u) pair this term primarily serves, but ALL 9 entries of alpha_k and ALL 9 entries of beta_k contribute to Sigma, H, and Delta. Do not truncate to just the active row/column.

### 2. The Non-Active Entries Are Where the Magic Happens

A term serving fiber (r, u) contributes Sigma[k, r*3+u] = Σ_s α_k[r,s]·β_k[s,u]. This is determined by row r of alpha and column u of beta (6 numbers). But the OTHER rows of alpha and columns of beta create cross-fiber interference — and it is precisely this interference that must satisfy Gates 1-3. The search over non-active entries is where solutions hide.

### 3. Search Space Reduction Strategy

For the {-1, 0, 1} coefficient field, the naive search space per term is 3^18 ≈ 387M (both alpha and beta). This is too large for brute force over 19 terms.

**Reduction strategy:**
- Fix fiber assignment for all terms (from fiber partition).
- For each term, the active row of alpha has 3³ = 27 options. The active column of beta has 3³ = 27 options. That's 729 (active-entry) options per term.
- The 12 non-active entries (remaining 2 rows of alpha × 3 + remaining 2 columns of beta × 3) have 3^12 ≈ 531K options — but we don't enumerate these all at once.
- **Build the tree incrementally:** Place terms one at a time. After placing each term's active entries, check if the current partial Sigma is compatible. Then enumerate non-active entries with incremental rank pruning on H.
- **Key pruning:** After placing k terms, if rank(H[0:k, :]) > 10, prune immediately. The partial H has k rows and 18 columns. For k ≤ 10, rank can be at most k, so this isn't constraining yet. For k > 10, it becomes progressively more constraining — only configurations where later rows land in the existing column span survive.

### 4. Numba Acceleration for Inner Loops

The rank check is called millions of times. Use this pattern:

```python
@numba.njit(cache=True)
def fast_sigma_row(alpha_row, beta_col):
    """Compute one row of Sigma: sigma = alpha_row @ beta_col (dot product)."""
    return alpha_row[0]*beta_col[0] + alpha_row[1]*beta_col[1] + alpha_row[2]*beta_col[2]

@numba.njit(cache=True)
def fast_eta1_row(alpha_row, beta_col):
    """Compute one row of Eta1."""
    return alpha_row[0]*beta_col[0] - alpha_row[1]*beta_col[1]  # per-entry, 9 values
```

For the SVD/rank computation, you can't JIT that — use batched `np.linalg.svd` on stacked matrices when checking many candidates at once.

### 5. Progress Estimation

For ETA calculation:
- Track candidates checked per second (rolling average over last 60 seconds).
- Estimate total candidates for current partition using the tree structure:
  - At depth k, the branching factor is approximately `|coeff_field|^(entries_per_term) / pruning_factor`.
  - The pruning factor increases with depth (rank constraint gets tighter).
  - Use the observed pruning rate from the first few partitions to calibrate.
- Report ETA per partition and ETA for the full search.

### 6. Memory Management

Working set per worker:
- Current alpha, beta arrays: 19 × 3 × 3 × 8 bytes = 4 KB (negligible)
- H matrix: 19 × 18 × 8 = 3 KB
- Delta matrix: 19 × 54 × 8 = 8 KB
- Dedup set: depends on Gate 1 survivor count, but likely < 1 GB total
- **Total per worker: < 100 MB including overhead**
- **24 workers: < 2.4 GB** — well within 80 GB RAM

### 7. Logging Gate 1 Survivors

Every configuration that passes Gate 1 (rank(H) = 10) is scientifically interesting even if it doesn't pass Gates 2-3. Log these to `gate1_survivors.jsonl`:

```json
{"alpha": [[...]], "beta": [[...]], "rank_H": 10, "rank_N": 11, "delta_leak": 1, "aug_rank": 17, "fitness": 0.234, "partition": [2,2,2,2,2,2,2,2,3]}
```

This data is valuable for understanding the geometry of the near-miss landscape.

---

## SELF-TEST SUITE (run with `--test`)

Implement these tests in each module:

1. **tensor.py tests:**
   - Standard R=27 reconstructs T exactly.
   - Strassen R=7 (2×2 case) reconstructs 2×2 multiplication tensor exactly.
   - Random R=19 config has fitness > 0 (with overwhelming probability).
   - Sigma, H, Delta shapes are correct.

2. **gates.py tests:**
   - Standard R=27: rank(H) = 18, rank(N) = 18, all gates pass.
   - Strassen R=7 (adapted to 2×2): all gates pass.
   - Random R=19: Gate 1 fails (rank(H) = 18 generically, not 10).
   - Construct a known rank-deficient H (e.g., repeat rows): verify incremental_rank_check catches it.

3. **symmetry.py tests:**
   - Group has exactly 48 elements.
   - Every element has order dividing 48 (actually dividing lcm of cycle lengths).
   - The 27 triples partition into orbits of sizes 1, 6, 12, 8.
   - The 19 "boundary" triples (at least one zero) are a union of the first three orbits.
   - Canonicalization is idempotent: canon(canon(x)) == canon(x).
   - Canonicalization is invariant: canon(g·x) == canon(x) for all g.

4. **fiber.py tests:**
   - Number of fiber partitions of 19 into 9 positive parts is correct (compute independently).
   - After symmetry reduction, count decreases.
   - Per-fiber coupling constraint: standard algorithm's fiber assignment satisfies it.

---

## EXECUTION PLAN

**Phase 1 (minutes):** Enumerate and canonicalize all fiber partitions. Print the list with estimated candidate counts. This is the `--dry-run` mode. Should complete in < 1 minute.

**Phase 2 (hours-days):** Full search over {-1, 0, 1} with 24 workers. The combinatorics note estimates this at hours to days depending on the rank filter's selectivity. The dashboard tracks progress in real time.

**Phase 3 (if needed):** If Phase 2 finds no solution, extend to {-2, -1, 0, 1, 2}. This multiplies the search space by ~3^(19×6) / 2^(19×6) ≈ 10^9. Requires deeper pruning or a move to GPU-accelerated batched enumeration. Cross that bridge if we get there.

---

## SUCCESS CRITERIA

- If a solution is found: `SOLUTION.json` contains the exact (alpha, beta, gamma) arrays, verified by `verify.py`.
- If no solution exists over {-1, 0, 1}: the search completes exhaustively, and the checkpoint file records 100% completion with 0 solutions. This is itself a publishable result.
- In either case: `gate1_survivors.jsonl` contains all configurations that achieved rank(H) = 10, providing a complete census of the near-miss landscape over the coefficient field.

---

## DON'T FORGET

- **Read `./docs/CANON_CONSTRAINTS.md` first.** It has the exact formulas for every coordinate block.
- **Read `./docs/gate_progress.md`.** It shows what works and what doesn't at each gate.
- **γ is NOT a free variable.** It is determined by solving Γ·Σ = 3I₉, Γ·Nuisance = 0.
- **Do not use continuous optimization.** This is enumeration with algebraic pruning.
- **Use symmetry for deduplication, not parameterization.**
- **The coefficient field is {-1, 0, 1} initially.** All arithmetic is exact integer.
- **Rich dashboard is mandatory.** The user needs to see progress, gate statistics, and ETA in real time.
- **Checkpoint every 60 seconds.** The search may run for days; it must survive interruptions.
