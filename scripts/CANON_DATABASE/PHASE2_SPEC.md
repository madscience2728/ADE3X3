# PHASE 2 SPEC: Search Engine over TermDB

## READ FIRST

This spec builds the search engine that uses the precomputed `TermDB` (387M records in RAM, ~11 GB) to find rank-19 decompositions of the 3×3 matrix multiplication tensor.

**Read these files before writing code:**
- `./docs/CANON_CONSTRAINTS.md` — Gate hierarchy, exact formulas
- `./docs/gate_progress.md` — Gate 2 is generically free, Gate 3 is the bottleneck
- `./docs/combinatorics_note.md` — Tier-by-tier pruning analysis
- `./CANON_DATABASE/term_db.py` — The TermDB class (already built and tested)

**Architecture:** The TermDB holds all 387M (α,β) pairs over {-1,0,1} with precomputed H-rows and sigma-rows. The search engine's job is to find 19 records from this DB whose combined coordinate blocks satisfy Gates 1-3. The DB's `query_subspace(V)` returns all records with H-row in a given 10-dim subspace in ~2 seconds. This replaces the inner enumeration loop entirely.

---

## HARDWARE

| Resource | Spec |
|----------|------|
| CPU | Ryzen 9 5900X — 24 threads |
| RAM | 80 GB (11 GB used by DB, ~52 GB free) |
| GPU | RTX 3060 12 GB (optional, --gpu flag) |

All output files: `./CANON_OPTIMIZER/core/`

---

## THE THREE-PHASE SEARCH

The search has three nested phases. Each phase produces candidates for the next.

### Overview

```
Phase A: BUILD BASIS (the bottleneck)
  Choose 10 records from DB whose H-rows are linearly independent.
  These span a 10-dim subspace V ⊂ R^18.
  → Produces candidate subspaces V.

Phase B: QUERY + ASSEMBLE (fast, ~2 sec per V)
  Query DB for all records with H-row in V (~150K hits).
  Choose 9 more records from hits.
  → Produces candidate 19-term configurations.

Phase C: GATE CHECK (instant)
  For each 19-term config, check Gates 2 and 3.
  If Gate 3 passes: SOLUTION FOUND.
  → Produces verified solutions (or empty set).
```

---

## PHASE A: BUILD BASIS

### The Challenge

We need 10 records whose H-rows are linearly independent (rank = 10). Naively this is "387M choose 10" which is absurd. The structure that makes it tractable:

1. **Deduplication:** Many records share the same H-row. Only ~X unique H-rows exist (run `db.build_dedup_index()` to find X). Search over unique H-rows, not records.

2. **Incremental independence:** Build the basis one row at a time. At depth k, only H-rows outside the current span survive. As the span grows, fewer H-rows qualify as independent, so the branching factor shrinks.

3. **Early sigma viability:** After choosing k basis terms, check whether the current partial Sigma matrix has "enough diversity" to eventually reach rank([Σ|N]) = 19. If the first k sigma-rows are too degenerate, prune.

4. **Fiber-guided ordering:** Order the basis construction by fiber. Choose the first basis term from fiber (0,0), second from fiber (0,1), etc. This ensures fiber coverage and avoids redundant configurations.

### Algorithm

```python
def build_bases(db: TermDB, config: SearchConfig) -> Generator[BasisResult]:
    """
    Incrementally build 10-dim subspaces by choosing independent H-rows.
    
    Yields BasisResult objects containing:
      - indices: list of 10 record indices
      - V_basis: (10, 18) integer matrix (the H-rows)
      - V_perp: (8, 18) integer nullspace
      - sigma_basis: (10, 9) integer matrix (the sigma-rows)
    """
    
    # Step 1: Get unique H-rows from DB
    unique_H = db.unique_H  # (M, 18) after dedup
    
    # Step 2: Choose first H-row
    for i1 in range(M):
        h1 = unique_H[i1]
        if is_zero(h1): continue
        
        V = h1.reshape(1, 18)  # rank 1
        
        # Step 3: Choose second H-row (independent of first)
        for i2 in range(i1 + 1, M):  # canonical ordering avoids duplicates
            h2 = unique_H[i2]
            if h2_in_span_of_V: continue
            
            V = stack(h1, h2)  # rank 2
            
            # ... continue to depth 10
            
            # At depth 10:
            V_perp = integer_nullspace(V)  # (8, 18)
            yield BasisResult(indices, V, V_perp, sigma_rows)
```

### Critical Pruning Rules

At each depth k (after placing k basis H-rows):

| Check | Condition | Action |
|-------|-----------|--------|
| Independence | rank(V[0:k]) < k | PRUNE — row was dependent |
| Depth budget | k > 10 | STOP — basis complete |
| Sigma diversity | rank(sigma[0:k]) == 0 for k >= 3 | PRUNE — sigma is degenerate |
| Subspace viability | count(unique H-rows independent of V) < (10 - k) | PRUNE — can't reach rank 10 |
| Dependent richness | Quick count of H-rows IN V < 9 | PRUNE — not enough dependents |

**The "dependent richness" check** is the most powerful pruner. At depth k, compute V_perp for the current partial subspace (18-k rows). Query the DB: how many records have H-row in the partial subspace? If this count is already too low (especially accounting for fiber constraints), the subspace can never work.

**Implementation of the dependent richness check:** Don't run a full query (expensive at early depths). Instead, use a FAST PARTIAL CHECK:

```python
def quick_dependent_count(db, V_partial, sample_size=10000):
    """
    Estimate how many DB records have H-row in span(V_partial).
    Uses a random sample for speed at early depths.
    """
    V_perp = integer_nullspace(V_partial)
    if V_perp.shape[0] == 0:
        return db.N_PAIRS  # everything is in span
    
    # At early depths (rank 1-5), V_perp has 13-17 rows.
    # Full query on 387M rows takes ~3-5 sec with that many complement dims.
    # For early pruning, sample:
    sample_idx = np.random.choice(db.N_PAIRS, sample_size, replace=False)
    H_sample = db.H[sample_idx]
    proj = H_sample.astype(np.int16) @ V_perp.astype(np.int16).T
    hit_rate = np.mean(np.all(proj == 0, axis=1))
    return int(hit_rate * db.N_PAIRS)
```

### Reducing the Search Space for Basis Building

**Strategy 1: Fiber-ordered construction**

Assign each basis term to a fiber. Use the fiber partition to determine which fibers get basis terms. For a partition like (2,2,2,2,2,2,2,2,3), each fiber gets 2-3 terms. Of these, typically 1-2 are basis terms and 0-1 are dependent.

A typical assignment: 10 of the 9 fibers contribute 1 basis term each, plus 1 fiber contributes 2 basis terms. This gives a structured enumeration:
- For fiber (0,0): enumerate basis term candidates
- For fiber (0,1): enumerate basis term candidates independent of previous
- ...

**Strategy 2: Restrict basis terms to "high-quality" H-rows**

Many H-rows have small norm or near-zero entries, making them unlikely to contribute useful structure. Pre-filter: keep only H-rows with norm above a threshold, or with entries spanning at least 3 distinct values.

**Strategy 3: Symmetry deduplication**

Use the Z₂ ≀ S₃ group (order 48) to canonicalize partial bases. If two partial bases are related by a group element, only explore one.

### Implementation Notes for Basis Building

The depth-10 tree search is the computational bottleneck. To make it tractable:

1. **Store unique H-rows sorted** by a canonical form (lexicographic on entries). This enables binary search for the next independent row.

2. **Maintain an incremental QR/SVD** of the growing basis. Don't recompute from scratch at each depth.

3. **Use Numba for the independence check:**
```python
@numba.njit(cache=True)
def is_independent(V_basis_rows, new_row):
    """Check if new_row is outside span of V_basis_rows. 
    V_basis_rows: (k, 18), new_row: (18,). Returns bool."""
    # Project new_row onto span of V_basis_rows via Gram-Schmidt
    residual = new_row.copy().astype(np.float64)
    for i in range(V_basis_rows.shape[0]):
        v = V_basis_rows[i].astype(np.float64)
        coeff = np.dot(residual, v) / np.dot(v, v)
        residual -= coeff * v
    return np.dot(residual, residual) > 1e-10
```

4. **Parallelize across first-level branches.** The outermost loop (choosing the first basis H-row) can be split across 24 workers. Each worker gets a range of unique H-rows to start from.

---

## PHASE B: QUERY + ASSEMBLE

Given a basis V (10 × 18 integer matrix) from Phase A:

### Step B1: Query the DB

```python
V_perp = TermDB._integer_nullspace(V)  # (8, 18) int
hits = db.query_subspace(V)  # indices of all records with H-row in V
# Expected: ~100K-200K hits
```

### Step B2: Classify hits by sigma profile

Each hit has a sigma-row (9 entries). Group hits by which fibers they can serve:

```python
def classify_hits(db, hit_indices):
    """
    For each hit, determine which fiber(s) it primarily serves.
    A record "serves" fiber (r, u) if its sigma contribution to that
    fiber is nonzero: sigma[r*3+u] != 0.
    
    Returns: dict mapping fiber_id -> list of record indices
    """
    sigmas = db.sigma[hit_indices]  # (N_hits, 9)
    fiber_map = defaultdict(list)
    for i, idx in enumerate(hit_indices):
        for f in range(9):
            if sigmas[i, f] != 0:
                fiber_map[f].append(idx)
    return fiber_map
```

### Step B3: Assemble 19-term configurations

We have the 10 basis terms (from Phase A) and need 9 dependent terms from the hits.

**Constraint: rank([Σ | N]) = 19**

This is the core solvability check. Build it incrementally:

```python
def assemble_dependents(db, basis_indices, hit_indices, config):
    """
    Given 10 basis terms and a pool of ~150K hit records,
    find 9 dependent terms such that the full 19-term config
    passes Gates 2 and 3.
    
    Uses incremental rank checking on [Σ | N] to prune.
    """
    # Start with the 10 basis terms
    current = list(basis_indices)
    
    # Get the H, Sigma, and Nuisance blocks for basis terms
    H_basis = db.H[basis_indices]              # (10, 18)
    sigma_basis = db.sigma[basis_indices]        # (10, 9)
    delta_basis = db.compute_delta_batch(basis_indices)  # (10, 54)
    N_basis = np.hstack([H_basis, delta_basis])  # (10, 72)
    SN_basis = np.hstack([sigma_basis, N_basis]) # (10, 81)
    # rank(SN_basis) should be approaching 19 as we add dependents
    
    # For each dependent slot (9 slots):
    #   For each candidate from hits:
    #     Add its row to SN
    #     Check if rank increased
    #     If rank == 19 after all 9: check Gate 3 fully
    
    # Recursive search with pruning
    _search_dependents(db, current, hit_indices, SN_basis, results=[])
```

**Incremental rank check for assembly:**

After adding dependent term k (the k-th dependent, so term 10+k overall):
- Compute its Sigma, H, Delta rows
- Append to the running [Σ | N] matrix (now (10+k) × 81)
- Check rank:
  - rank < 10 + k: the new term was linearly dependent in the augmented space → skip
  - rank == 10 + k: good, continue
  - After all 9 dependents (19 rows): rank must be exactly 19

**The rank check on an (m × 81) matrix** with m ≤ 19 is fast (microseconds). This is the inner-loop bottleneck but it's tiny linear algebra.

### Reducing the Dependent Search Space

150K choose 9 is intractable. But:

1. **Fiber bucketing:** Each dependent must serve a specific fiber (or set of fibers). Partition the 150K hits by fiber → ~15K per fiber. Each dependent slot draws from one fiber bucket. This gives ~15K^9 / permutation_factor, still too large.

2. **Incremental rank pruning:** After adding dependent k, if the new row didn't increase rank([Σ|N]), skip it. This eliminates most candidates because most sigma-rows will be in the existing column span.

3. **Sigma rank tracking:** Track rank(Σ_partial) separately. If after adding k dependents, rank(Σ_partial) < k (the sigma matrix is too degenerate), prune. For Gate 3, we ultimately need the sigma columns to contribute 9 independent directions.

4. **Column-space screening:** Before the assembly loop, precompute for each hit: its "sigma innovation" — the component of its sigma-row orthogonal to the current [Σ|N] column space. Rank by innovation magnitude. Try high-innovation records first (greedy heuristic).

5. **Early Gate 2 check:** After adding each dependent, check if its Delta row lies in span(H). If not, reject immediately.

### Assembly Algorithm (Detailed)

```python
def _search_dependents(db, chosen, candidates, SN_matrix, depth, results):
    """
    Recursive DFS to find 9 dependent terms.
    
    Args:
        chosen: list of record indices chosen so far (starts with 10 basis)
        candidates: sorted list of candidate record indices (from DB query)
        SN_matrix: current [Sigma | Nuisance] matrix, (len(chosen), 81)
        depth: current dependent index (0-8)
        results: accumulator for solutions
    """
    if depth == 9:
        # All 19 terms chosen. Final check.
        if np.linalg.matrix_rank(SN_matrix) == 19:
            # GATE 3 PASSED — solve for Gamma
            gamma = solve_gamma(SN_matrix)
            if gamma is not None:
                results.append(Solution(chosen, gamma))
        return
    
    current_rank = np.linalg.matrix_rank(SN_matrix)
    needed_rank = 19
    remaining = 9 - depth
    
    # Pruning: if we can't possibly reach rank 19
    if current_rank + remaining < needed_rank:
        return  # impossible to add enough rank
    
    for i, idx in enumerate(candidates):
        # Skip if already chosen
        if idx in chosen:
            continue
        
        # Compute the new term's contribution
        sigma_row = db.sigma[idx].reshape(1, 9)
        H_row = db.H[idx].reshape(1, 18)
        delta_row = db.compute_delta(idx).reshape(1, 54)
        N_row = np.hstack([H_row, delta_row])
        SN_row = np.hstack([sigma_row, N_row])  # (1, 81)
        
        # Quick Gate 2 check: delta must be in span(H_current)
        # H_current is the H matrix for all chosen terms so far
        # (skip at early depths when span(H) is small)
        
        # Rank check: does this term increase rank?
        SN_new = np.vstack([SN_matrix, SN_row.astype(SN_matrix.dtype)])
        new_rank = np.linalg.matrix_rank(SN_new.astype(np.float64))
        
        if new_rank <= current_rank:
            continue  # didn't contribute, skip
        
        # Recurse
        _search_dependents(
            db, chosen + [idx], candidates[i+1:],  # enforce ordering to avoid permutations
            SN_new, depth + 1, results
        )
```

### Performance Optimization for Assembly

The inner loop calls `np.linalg.matrix_rank` on a ≤19×81 matrix for each candidate. This involves SVD of a tiny matrix — microseconds per call. With ~150K candidates × 9 depths, worst case is ~1.35M rank checks per basis, totaling ~2 seconds. Acceptable.

**Optimization: batch rank checks.** Instead of checking candidates one-by-one, batch them:

```python
# For all candidates at once:
SN_candidates = compute_SN_rows_batch(db, candidate_indices)  # (N_cand, 81)

# For each candidate, check if it increases rank:
# Equivalent to: is SN_row outside the current column span of SN_matrix?
# Project SN_row onto complement of column span.
# Use QR decomposition of SN_matrix.T for efficiency.
Q, R = np.linalg.qr(SN_matrix.T.astype(np.float64))  # Q: (81, current_rank)
residuals = SN_candidates.astype(np.float64) - (SN_candidates.astype(np.float64) @ Q) @ Q.T
innovations = np.linalg.norm(residuals, axis=1)
# Candidates with innovation > threshold increase rank
viable = np.where(innovations > 1e-8)[0]
```

This replaces N_cand individual SVDs with one QR + one batch matmul. For 150K candidates: ~100ms total.

---

## PHASE C: GATE CHECK

For each candidate 19-term configuration from Phase B:

```python
def full_gate_check(db, indices_19):
    """
    Complete gate check on a 19-term configuration.
    Returns (passed, diagnostics).
    """
    # Retrieve all factor matrices
    alphas, betas = db.get_factors_batch(indices_19)  # (19, 3, 3) each
    
    # Compute all blocks
    H = db.H[indices_19]               # (19, 18) — already precomputed
    sigma = db.sigma[indices_19]         # (19, 9)  — already precomputed
    delta = db.compute_delta_batch(indices_19)  # (19, 54)
    nuisance = np.hstack([H, delta])     # (19, 72)
    
    # Gate 1: rank(H) == 10
    rk_H = np.linalg.matrix_rank(H.astype(np.float64))
    if rk_H != 10:
        return False, {'gate': 1, 'rank_H': rk_H}
    
    # Gate 2: rank([H | Delta]) == rank(H)
    rk_N = np.linalg.matrix_rank(nuisance.astype(np.float64))
    if rk_N != rk_H:
        return False, {'gate': 2, 'rank_H': rk_H, 'rank_N': rk_N, 'delta_leak': rk_N - rk_H}
    
    # Gate 3: rank([Sigma | Nuisance]) == 19
    SN = np.hstack([sigma, nuisance])  # (19, 81)
    rk_SN = np.linalg.matrix_rank(SN.astype(np.float64))
    if rk_SN != 19:
        return False, {'gate': 3, 'augmented_rank': rk_SN}
    
    # Gate 3 passed — solve for Gamma
    # System: Gamma @ Sigma = 3*I_9, Gamma @ Nuisance = 0
    # Equivalently: Gamma @ [Sigma | Nuisance] = [3*I_9 | 0]
    # Gamma is 9×19
    target = np.zeros((9, 81), dtype=np.float64)
    target[:, :9] = 3.0 * np.eye(9)
    
    # Solve: Gamma @ SN = target  =>  SN.T @ Gamma.T = target.T
    Gamma_T, residuals, _, _ = np.linalg.lstsq(SN.astype(np.float64).T, target.T, rcond=None)
    Gamma = Gamma_T.T  # (9, 19)
    
    # Verify reconstruction
    # T_hat[i,j,l] = sum_k alpha_k[i] * beta_k[j] * gamma_k[l]
    T = build_multiplication_tensor()
    T_hat = reconstruct(alphas, betas, Gamma)
    
    residual = np.max(np.abs(T - T_hat))
    if residual < 1e-10:
        return True, {
            'gate': 'ALL PASSED',
            'alpha': alphas,
            'beta': betas,
            'gamma': Gamma,
            'residual': residual,
            'rank_H': rk_H,
            'rank_N': rk_N,
            'augmented_rank': rk_SN
        }
    else:
        return False, {'gate': 3, 'solve_residual': residual}


def build_multiplication_tensor():
    """Build the 3×3 matrix multiplication tensor T, shape (9, 9, 9)."""
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[r*3+s, s*3+u, r*3+u] = 1
    return T


def reconstruct(alphas, betas, Gamma):
    """Reconstruct tensor from factors. alphas, betas: (R, 3, 3), Gamma: (9, R)."""
    R = alphas.shape[0]
    T_hat = np.zeros((9, 9, 9), dtype=np.float64)
    for k in range(R):
        a = alphas[k].ravel()  # (9,)
        b = betas[k].ravel()   # (9,)
        g = Gamma[:, k]        # (9,)
        T_hat += np.einsum('i,j,l', a, b, g)
    return T_hat
```

---

## FILE STRUCTURE

```
./CANON_OPTIMIZER/core/
├── search_engine.py      # Phases A + B + C orchestration
├── basis_builder.py      # Phase A: incremental basis construction
├── assembler.py          # Phase B: dependent term assembly
├── gate_check.py         # Phase C: full gate verification
├── search_config.py      # Configuration dataclass
├── search_dashboard.py   # Rich TUI for search progress
└── search_main.py        # Entry point
```

The search engine imports from `../CANON_DATABASE/term_db.py`.

---

## SEARCH_CONFIG

```python
@dataclass
class SearchConfig:
    # Ranks to search
    target_ranks: list = field(default_factory=lambda: [13, 19, 20, 21, 22])
    
    # Coefficient field (determines which TermDB to use)
    coeff_field: list = field(default_factory=lambda: [-1, 0, 1])
    
    # Phase A settings
    max_basis_depth: int = 10           # for R=19
    min_dependent_count: int = 100      # prune if fewer dependents available
    basis_symmetry_dedup: bool = True   # use Z2≀S3 canonicalization
    
    # Phase B settings
    max_assembly_candidates: int = 200_000  # max hits to consider
    batch_rank_check: bool = True            # use QR-based batch innovation check
    innovation_threshold: float = 1e-8       # min norm to count as rank-increasing
    
    # Phase C settings
    gate3_residual_tol: float = 1e-10
    
    # Hardware
    n_workers: int = 24
    checkpoint_interval: int = 60  # seconds
    
    # Paths
    db_path: str = "./CANON_DATABASE/data"
    checkpoint_file: str = "./CANON_OPTIMIZER/core/search_checkpoint.json"
    solution_file: str = "./CANON_OPTIMIZER/core/SOLUTION.json"
    survivors_file: str = "./CANON_OPTIMIZER/core/gate2_survivors.jsonl"
```

---

## SEARCH_MAIN ENTRY POINT

```
Usage:
    python search_main.py                    # full search, all default ranks
    python search_main.py --rank 19          # search R=19 only
    python search_main.py --resume           # resume from checkpoint
    python search_main.py --workers 12       # override worker count
    python search_main.py --db ./data        # override DB path
    python search_main.py --test             # run Phase A/B/C on known examples
    python search_main.py --profile          # profile first 1000 bases, report stats
```

**Startup sequence:**
1. Load TermDB from disk (`TermDB(db_path)`, then `load_to_ram()`)
2. Build dedup index (`db.build_dedup_index()`)
3. For each target rank:
   a. Compute target_rank_H = R - 9
   b. Launch search with Rich dashboard
4. If solution found: save, verify, celebrate

---

## RICH DASHBOARD

```
╔══════════════════════════════════════════════════════════════════════╗
║  R=19 SEARCH — Phase A: Building Bases                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PROGRESS                                                            ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  34.7%                 ║
║  Unique H-rows: 12,847,293     ETA: 3h 12m 45s                      ║
║  Wall time: 1h 41m 02s         Rate: 4,312 bases/s                  ║
║                                                                      ║
║  PHASE A: BASIS BUILDING                                             ║
║  ┌──────────────┬──────────────┬──────────────┐                     ║
║  │ Depth        │ Explored     │ Pruned       │                     ║
║  ├──────────────┼──────────────┼──────────────┤                     ║
║  │ depth 1      │ 12,847,293   │ 1,204        │                     ║
║  │ depth 2      │ 8,412,001    │ 3,891,004    │                     ║
║  │ depth 3      │ 2,104,887    │ 1,987,221    │                     ║
║  │ ...          │ ...          │ ...          │                     ║
║  │ depth 10     │ 47,812       │ 41,003       │                     ║
║  │ BASES BUILT  │ 6,809        │ —            │                     ║
║  └──────────────┴──────────────┴──────────────┘                     ║
║                                                                      ║
║  PHASE B: ASSEMBLY (per basis)                                       ║
║  ┌──────────────┬──────────────┬──────────────┐                     ║
║  │ Metric       │ Current      │ Cumulative   │                     ║
║  ├──────────────┼──────────────┼──────────────┤                     ║
║  │ Bases tested │ 6,809        │ 6,809        │                     ║
║  │ Avg hits     │ 153,412      │ —            │                     ║
║  │ Gate 2 pass  │ 12           │ 847          │                     ║
║  │ Gate 3 pass  │ 0            │ 0            │                     ║
║  └──────────────┴──────────────┴──────────────┘                     ║
║                                                                      ║
║  BEST NEAR-MISS                                                      ║
║  aug_rank=18/19  Σ_rank=8/9  conservation=27 ✓  delta_leak=0       ║
║                                                                      ║
║  WORKERS: 24/24 active                                               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## SELF-TESTS

### Test 1: Standard R=27

The standard 3×3 algorithm has 27 terms with α_k = e_r·e_s^T, β_k = e_s·e_u^T, γ_k = e_r·e_u^T. Build the TermDB records for these 27 terms. Verify:
- All 27 H-rows exist in the DB
- rank(H) = 18
- Choosing any 18 as basis: query returns all 27 as hits (since rank(H) = 18, all rows are in the 18-dim span)
- Gate 3 passes when all 27 are assembled

### Test 2: Synthetic R=10 subproblem

Pick 10 random independent H-rows from DB. Query for hits. Verify:
- Hit count is reasonable (~100K-200K)
- 10 basis rows appear in hits
- Assembly algorithm can be run (may not find Gate 3 solution, but should not crash)

### Test 3: Gate check on known failures

Assemble a random 19-term configuration. Verify:
- Gate 1 fails (rank(H) ≈ 18 generically)
- Gate check returns correct diagnostics

### Test 4: R=13 control

R=13 with target rank(H) = 4. Easier than R=19. Use as control:
- Build 4-dim subspace basis (4 independent H-rows)
- Query for hits with H-row in span
- Assembly: need 9 dependent terms from hits
- Should be significantly easier than R=19 — validates the pipeline

---

## CRITICAL IMPLEMENTATION NOTES

### 1. Delta Recomputation

Delta rows are NOT stored in the DB (saves 21 GB RAM). Recompute on demand:
```python
delta = db.compute_delta(idx)        # single record
deltas = db.compute_delta_batch(indices)  # batch
```
Only called during Phase B assembly and Phase C gate checks — on a few thousand records at most. Cost: negligible.

### 2. Integer Arithmetic Throughout

H-rows, sigma-rows, delta-rows: all int8. Nullspace computation: int64. Matmul queries: int16. The only float arithmetic is in SVD for rank checks and lstsq for Gamma solves. This is exact where it matters and approximate only where it's checked.

### 3. Checkpoint What Matters

Save completed subspaces (Phase A outputs that led to Phase B) and their results. On resume, skip completed subspaces. The checkpoint file should store:
- List of basis H-row tuples (as hashable keys) that have been fully explored
- Cumulative gate statistics
- Best near-miss diagnostics
- Any solutions found

### 4. Solution Verification

When Gate 3 passes, IMMEDIATELY:
1. Save raw solution to SOLUTION.json
2. Run independent verification (reconstruct T from factors, check every entry)
3. Log to console with Rich panel in bright green
4. Optionally: continue searching for more solutions

### 5. The R=13 Control Signal

R=13 has target_rank_H = 4 (much smaller basis). The search space is vastly smaller. If the pipeline can't find anything for R=13 within minutes, something is wrong with the implementation. Use R=13 as a smoke test before committing hours to R=19.

### 6. Parallelization Strategy

Split Phase A across workers by partitioning the first-level choice:
```python
# Worker k handles unique H-rows in range [start_k, end_k)
# Each worker independently builds bases starting from its range
# Workers share a solution queue (multiprocessing.Queue)
# Coordinator collects results and updates dashboard
```

Phase B (assembly) runs inside each worker — no cross-worker communication needed during assembly.

---

## DON'T FORGET

- **TermDB is already built.** Load it, don't regenerate. `db = TermDB("./CANON_DATABASE/data"); db.load_to_ram()`
- **The query is the magic.** `db.query_subspace(V)` replaces all inner-loop enumeration. 2 seconds, exact.
- **Delta is computed on demand.** Not stored. `db.compute_delta_batch(indices)`.
- **R=13 first.** Smoke test the pipeline on the easiest target before R=19.
- **Gate 2 is generically free** (from gate_progress.md). Don't over-invest in Gate 2 pruning.
- **Gate 3 is the real test.** rank([Σ|N]) = 19 is the finish line.
- **Checkpoint every 60 seconds.**
- **Rich dashboard is mandatory.**
