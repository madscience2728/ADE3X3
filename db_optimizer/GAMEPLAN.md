# DB-Optimizer: Two-Phase GPU-Accelerated Rank-19 Search

## Problem Statement

Find an exact (ε → 0) rank-19 CP decomposition of the 3×3 matrix multiplication
tensor T ∈ ℝ^{9×9×9}, where T[3r+u, 3r+s, 3s+u] = 1.0 (27 live entries, 702 dead).

Current best: max_abs = 0.0983 (L-BFGS smooth-max → v2 coordinate descent → EA).
Both the EA and v2 optimizers converge to the same basin floor at ~0.098.
The goal is either to reach ε = 0 (exact decomposition exists) or to
provide strong numerical evidence that rank(T) > 19.

## Architecture

Two-phase asynchronous pipeline with SQLite persistence and GPU-batch evaluation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: GENERATION (CPU)                    │
│                                                                     │
│  Generator Workers (threads, many)                                  │
│  ├── Mutation: support flip, coefficient algebraic snap, Gaussian   │
│  ├── Crossover: uniform term swap between two parents               │
│  ├── Re-injection: pull from shadow archive for basin escape        │
│  └── INSERT INTO candidates (pending) → SQLite                      │
│                                                                     │
│  Selection Queries (CPU, between batches)                           │
│  ├── SELECT parents by virtual island + tournament                  │
│  ├── Prune: DELETE duplicates, blacklisted signatures               │
│  └── Elitism: top-K per virtual island are immortal                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │   SQLite    │
                     │  (K:/ disk) │
                     │             │
                     │ candidates  │  ← main table
                     │ shadow      │  ← basin-escape archive
                     │ blacklist   │  ← known-dead support hashes
                     │ run_log     │  ← audit trail
                     └──────┬──────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│                     PHASE 2: EVALUATION (GPU + CPU)                 │
│                                                                     │
│  Tier 1 — GPU Bulk Screen (FP32, batches of 10K–100K)              │
│  ├── Pack factor matrices → (N, 19, 9) tensors on CUDA             │
│  ├── Batch reconstruct: einsum → (N, 9, 9, 9)                      │
│  ├── Batch residual: candidate - target                             │
│  ├── Batch fitness: max(|residual|) per candidate                   │
│  ├── UPDATE fitness_fp32 back to DB                                 │
│  └── Survivors: WHERE fitness_fp32 < survivor_threshold (e.g. 0.15)│
│                                                                     │
│  Tier 2 — GPU Minimax Sweep (FP32, survivors only, ~1K batch)      │
│  ├── Greedy coordinate descent on GPU                               │
│  │   ├── For each coeff: broadcast N_trials perturbations           │
│  │   ├── Batch incremental residual update                          │
│  │   └── argmin across trials → accept best per candidate           │
│  ├── UPDATE fitness_fp32, updated factor values                     │
│  └── Filter: WHERE fitness_fp32 < refine_threshold (e.g. 0.105)    │
│                                                                     │
│  Tier 3 — CPU FP64 Refinement (top candidates only, ~10–100)       │
│  ├── V2-style minimax sweep (FP64, single-coeff + pair + cross)    │
│  ├── Optional: Newton polish (scipy least_squares)                  │
│  ├── UPDATE fitness_fp64, final factor values                       │
│  └── INSERT best into shadow archive if novel                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Hardware Budget

| Resource  | Available | Allocated         |
|-----------|-----------|-------------------|
| CPU       | 24 logical cores | Gen: 4 threads, Selection: 1 thread, Tier 3: 19 cores |
| RAM       | 80 GB     | SQLite cache: 8 GB, Python heap: 4 GB, OS+headroom: 4 GB, free: 64 GB |
| GPU       | RTX 3060 12 GB VRAM | Tier 1+2 batch eval: ~2 GB, workspace: ~1 GB, free: 9 GB |
| Disk      | K:/ drive | SQLite WAL DB (persistent), checkpoint snapshots |

## SQLite Schema (draft)

```sql
-- Main candidate table
CREATE TABLE candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Factor data: 3 matrices × 19 × 9 = 513 floats, stored as blob
    factors_blob    BLOB NOT NULL,          -- 513 × 8 bytes = 4104 bytes
    support_hash    TEXT NOT NULL,           -- canonical support fingerprint
    support_sig     TEXT NOT NULL,           -- e.g. "(9,9,9)" or "(3,3,3)"
    origin          TEXT NOT NULL,           -- "mutation", "crossover", "inject", "shadow"
    parent_ids      TEXT,                    -- comma-separated parent IDs
    virtual_island  INTEGER NOT NULL,        -- 0..N-1
    -- Fitness (NULL = pending evaluation)
    fitness_fp32    REAL,
    fitness_fp64    REAL,
    fro_residual    REAL,
    -- Lifecycle
    generation      INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/screened/refined/archived/dead
    created_at      REAL NOT NULL,           -- time.time()
    evaluated_at    REAL,
    -- Minimax improvement tracking
    minimax_improved REAL,                   -- delta from pre-minimax fitness
    tier_reached    INTEGER DEFAULT 0        -- 0=pending, 1=GPU screen, 2=GPU minimax, 3=CPU refine
);

CREATE INDEX idx_candidates_status ON candidates(status);
CREATE INDEX idx_candidates_fitness ON candidates(fitness_fp32);
CREATE INDEX idx_candidates_support_hash ON candidates(support_hash);
CREATE INDEX idx_candidates_island_fitness ON candidates(virtual_island, fitness_fp32);
CREATE INDEX idx_candidates_generation ON candidates(generation);

-- Shadow archive: basin-escape memory (persistent across runs)
CREATE TABLE shadow (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factors_blob    BLOB NOT NULL,
    support_hash    TEXT NOT NULL UNIQUE,
    fitness_fp64    REAL NOT NULL,
    origin_run      TEXT,
    archived_at     REAL NOT NULL
);

-- Blacklisted support patterns (known-dead, skip evaluation)
CREATE TABLE blacklist (
    support_hash    TEXT PRIMARY KEY,
    reason          TEXT,
    added_at        REAL NOT NULL
);

-- Run log for audit
CREATE TABLE run_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event           TEXT NOT NULL,
    generation      INTEGER,
    detail_json     TEXT,
    wall_seconds    REAL,
    timestamp       REAL NOT NULL
);
```

## Blob Encoding

Factor matrices are stored as a single contiguous `float64` blob:
```
alpha[0,0..8], alpha[1,0..8], ..., alpha[18,0..8],   (19×9 = 171 floats)
beta[0,0..8],  beta[1,0..8],  ..., beta[18,0..8],    (171 floats)
gamma[0,0..8], gamma[1,0..8], ..., gamma[18,0..8]     (171 floats)
                                                 Total: 513 × 8 = 4104 bytes
```

Encode: `np.concatenate([alpha, beta, gamma]).tobytes()`
Decode: `np.frombuffer(blob, dtype=np.float64).reshape(3, 19, 9)`

This is the dense representation — every candidate is fully dense (9,9,9) like the
L-BFGS solutions. Support masks are derived by thresholding: `|value| > 1e-8`.

---

## Implementation Checklist

### Phase 0: Foundation

- [ ] **0.1** Verify PyTorch CUDA 12.6 is installed and 3060 is visible
  - `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`
  - File: `db_optimizer/tests/test_cuda_available.py`
  
- [ ] **0.2** Create `db_optimizer/` package structure
  ```
  db_optimizer/
  ├── __init__.py
  ├── GAMEPLAN.md           ← this file
  ├── config.py             ← all constants, paths, thresholds
  ├── schema.py             ← SQLite schema creation + migrations
  ├── db.py                 ← connection pool, WAL mode, pragma tuning
  ├── blob.py               ← factor encode/decode, support_hash, support_sig
  ├── tensor.py             ← target tensor, CPU reconstruct, CPU fitness
  ├── gpu_eval.py           ← Tier 1+2 GPU batch evaluation
  ├── cpu_refine.py         ← Tier 3 FP64 minimax + optional Newton polish
  ├── generator.py          ← mutation, crossover, shadow re-injection
  ├── selector.py           ← tournament selection, elitism, pruning queries
  ├── orchestrator.py       ← main loop: gen → eval → select → repeat
  ├── cli.py                ← CLI entry point (argparse)
  ├── tests/
  │   ├── __init__.py
  │   ├── test_cuda_available.py
  │   ├── test_blob_roundtrip.py
  │   ├── test_schema.py
  │   ├── test_tensor_cpu.py
  │   ├── test_gpu_eval.py
  │   ├── test_gpu_minimax.py
  │   ├── test_cpu_refine.py
  │   ├── test_generator.py
  │   ├── test_selector.py
  │   └── test_end_to_end.py
  └── README.md             ← quick-start for running
  ```

- [ ] **0.3** Create `config.py` with all constants
  - `DB_PATH = Path("K:/ade3x3_optimizer/candidates.db")`
  - `RANK = 19`, `DIM = 9`, `N_ENTRIES = 729`
  - `TARGET_TENSOR` (built once)
  - `ALGEBRAIC_LOOKUP` (reuse from step84)
  - GPU batch sizes, thresholds, virtual island count
  - Tier boundaries: `TIER1_SURVIVOR_THRESHOLD = 0.15`
  - Tier boundaries: `TIER2_REFINE_THRESHOLD = 0.105`
  - Backpressure: `MAX_PENDING = 50_000`

- [ ] **0.4** `pytest` baseline — `tests/test_cuda_available.py` passes

### Phase 1: Data Layer

- [ ] **1.1** `blob.py` — encode/decode factor matrices ↔ blob
  - `factors_to_blob(alpha, beta, gamma) → bytes`
  - `blob_to_factors(blob) → (alpha, beta, gamma)` each `(19, 9)` float64
  - `compute_support_hash(alpha, beta, gamma) → str`
  - `compute_support_sig(alpha, beta, gamma) → str`
  - Test: `test_blob_roundtrip.py` — encode → decode → assert allclose

- [ ] **1.2** `schema.py` — create tables + indexes
  - `create_schema(conn)` — idempotent (IF NOT EXISTS)
  - `migrate(conn)` — future schema changes
  - Test: `test_schema.py` — create in `:memory:`, verify tables exist

- [ ] **1.3** `db.py` — connection management
  - `get_connection(db_path) → sqlite3.Connection`
  - WAL mode, `journal_size_limit`, `cache_size = -8000000` (8GB)
  - `synchronous = NORMAL` (not OFF — we want crash safety)
  - `busy_timeout = 5000` ms
  - Thread-safe: one connection per thread
  - Test: `test_schema.py` — open, write, read back

- [ ] **1.4** `tensor.py` — CPU-side tensor operations (reference implementation)
  - `build_target_tensor() → np.ndarray (9,9,9)`
  - `reconstruct(alpha, beta, gamma) → np.ndarray (9,9,9)`
  - `fitness(candidate, target) → float` (max_abs)
  - `frobenius(candidate, target) → float`
  - Test: `test_tensor_cpu.py` — known decomposition → fitness check

- [ ] **1.5** Seed DB with existing candidates
  - Load `optimized_als_r10_at_0.09.json` + all `optimized_*.json`
  - INSERT into `candidates` with `status='refined'` and known fitness
  - INSERT into `shadow` as the baseline archive

### Phase 2: GPU Evaluation

- [ ] **2.1** `gpu_eval.py` — Tier 1: batch fitness on GPU
  - `gpu_batch_fitness(blobs: list[bytes], device) → np.ndarray`
  - Unpack blobs → `(N, 3, 19, 9)` tensor on CUDA
  - `torch.einsum('nra,nrb,nrc->nabc', alpha, beta, gamma)` 
  - `residual = candidate - target.expand(N, ...)`
  - `fitness = residual.abs().flatten(1).max(dim=1).values`
  - Return as numpy float32 array
  - Handle FP32 precision: use `torch.float32` for throughput
  - Test: `test_gpu_eval.py` — compare GPU fitness vs CPU fitness on 100 random candidates, assert `|gpu - cpu| < 1e-5`

- [ ] **2.2** `gpu_eval.py` — Tier 2: batch minimax sweep on GPU
  - `gpu_batch_minimax(factors_batch, target, sweeps=1, fine_range=0.003) → improved_factors_batch, new_fitness`
  - For each coefficient position (513 total, shuffled):
    - Generate N_trials perturbations (algebraic neighbors + fine grid)
    - Broadcast: `(N_candidates, N_trials)` perturbation matrix
    - Incremental residual update (only 1 slice of 9×9 changes per coeff)
    - `argmin` across trials per candidate
    - Accept improvements
  - This is the GPU port of `minimax_local_search()` from step84
  - Test: `test_gpu_minimax.py` — run on 10 candidates, verify fitness decreases

- [ ] **2.3** FP32 vs FP64 validation
  - Run 1000 candidates through both GPU FP32 and CPU FP64
  - Measure max discrepancy
  - If discrepancy > 0.001 at fitness ~0.1: tighten tier thresholds
  - If discrepancy < 1e-4: we can trust GPU screening fully
  - Document results in test

- [ ] **2.4** GPU memory profiling
  - Measure actual VRAM usage for batch sizes: 1K, 10K, 50K, 100K
  - Find the sweet spot: max batch that fits in 10 GB (leave 2 GB headroom)
  - Add `MAX_GPU_BATCH_SIZE` to config

### Phase 3: Generation + Selection

- [ ] **3.1** `generator.py` — candidate generation
  - `mutate_support(factors, rng) → factors` — flip one support index
  - `mutate_coefficient(factors, rng, algebraic_lookup) → factors` — snap to algebraic neighbor
  - `mutate_gaussian(factors, rng, sigma) → factors` — continuous perturbation
  - `crossover(parent_a, parent_b, rng) → child` — uniform term swap
  - `reinject_from_shadow(conn, rng) → factors` — pull old candidate from shadow
  - `generate_batch(parents, n, rng, shadow_conn) → list[factors]`
  - Backpressure: skip generation if `SELECT COUNT(*) FROM candidates WHERE status='pending'` > MAX_PENDING
  - Test: `test_generator.py` — mutations produce valid shapes, crossover preserves rank

- [ ] **3.2** `selector.py` — selection queries
  - `select_parents(conn, island, k, tournament_size) → list[row]`
  - `select_elites(conn, island, k) → list[row]`
  - `prune_duplicates(conn)` — DELETE WHERE support_hash is duplicate, keep best fitness
  - `prune_blacklisted(conn)` — DELETE WHERE support_hash IN blacklist
  - `archive_to_shadow(conn, candidate_row)` — INSERT OR IGNORE into shadow
  - `island_stats(conn) → dict` — per-island best, mean, worst, count
  - Test: `test_selector.py` — insert 100 candidates, verify selection returns correct K

- [ ] **3.3** Virtual island assignment
  - New candidates get `virtual_island = hash(support_hash) % N_ISLANDS`
  - Or: round-robin assignment for diversity
  - Parent selection restricted to same island (with migration probability)

### Phase 4: Orchestrator

- [ ] **4.1** `orchestrator.py` — main loop
  ```python
  def run(config):
      db = open_db(config.db_path)
      create_schema(db)
      seed_initial_candidates(db, config)
      
      for generation in itertools.count():
          # Phase 1: Generate
          parents = select_parents(db, ...)
          children = generate_batch(parents, config.batch_size, rng, db)
          insert_pending(db, children, generation)
          
          # Phase 2: Evaluate
          pending = fetch_pending(db, limit=config.gpu_batch_size)
          # Tier 1: GPU bulk screen
          fitness_fp32 = gpu_batch_fitness(pending)
          update_fitness(db, pending, fitness_fp32, tier=1)
          
          # Tier 2: GPU minimax on survivors
          survivors = fetch_by_threshold(db, config.tier1_threshold)
          improved, new_fitness = gpu_batch_minimax(survivors)
          update_factors_and_fitness(db, improved, new_fitness, tier=2)
          
          # Tier 3: CPU refinement on elite
          elites = fetch_by_threshold(db, config.tier2_threshold)
          for elite in elites:
              refined = cpu_minimax_refine(elite)
              update_refined(db, refined, tier=3)
          
          # Selection + housekeeping
          prune_duplicates(db)
          prune_blacklisted(db)
          archive_novel_elites_to_shadow(db)
          log_generation(db, generation)
          
          # Check termination
          best = get_best(db)
          if best.fitness_fp64 < config.hit_threshold:
              print(f"EXACT HIT at generation {generation}!")
              break
  ```

- [ ] **4.2** Graceful shutdown (Ctrl+C handler)
  - Commit current DB transaction
  - Flush WAL to main DB file
  - Export best candidate as JSON

- [ ] **4.3** Resume from existing DB
  - If DB exists on K:/, pick up from last generation
  - Detect stale `pending` rows (evaluator crashed) → reset to pending

- [ ] **4.4** Live monitoring
  - Log to `run_log` table every generation
  - Periodic console output: gen, best, mean, pending count, eval rate
  - Optional: expose stats via HTTP for the tuner UI to poll

### Phase 5: CLI + Integration

- [ ] **5.1** `cli.py` — command line interface
  ```
  python -m db_optimizer run --db K:/ade3x3_optimizer/candidates.db --generations 10000
  python -m db_optimizer status --db K:/ade3x3_optimizer/candidates.db
  python -m db_optimizer export-best --db K:/ade3x3_optimizer/candidates.db --output best.json
  python -m db_optimizer inject --db K:/ade3x3_optimizer/candidates.db --input optimized_als_r10_at_0.09.json
  python -m db_optimizer blacklist --db K:/ade3x3_optimizer/candidates.db --hash "abc123"
  python -m db_optimizer shadow-stats --db K:/ade3x3_optimizer/candidates.db
  ```

- [ ] **5.2** Seed from existing work
  - Import all `optimized_*.json` from repo root
  - Import warm seeds from `step83b_track2_screening.csv`
  - Import shadow pool from last step84 run (if exists)

- [ ] **5.3** Export compatibility
  - `export-best` writes same JSON format as step84
  - Can be fed back to step84 via `STEP84_INJECT_CANDIDATE`
  - Cross-compatible with `optimize_v2.py` and `optimize_als.py`

### Phase 6: Testing

All tests use `pytest`. Run with: `python -m pytest db_optimizer/tests/ -v`

- [ ] **6.1** `test_cuda_available.py` — GPU visible, FP32+FP64 work, VRAM sufficient
- [ ] **6.2** `test_blob_roundtrip.py` — encode → decode preserves all 513 coefficients
- [ ] **6.3** `test_schema.py` — tables created, indexes exist, CRUD operations work
- [ ] **6.4** `test_tensor_cpu.py` — target tensor correct (27 ones, 702 zeros), known candidate fitness matches
- [ ] **6.5** `test_gpu_eval.py` — GPU fitness within 1e-5 of CPU fitness (FP32), handle edge cases (zero candidate, identity)
- [ ] **6.6** `test_gpu_minimax.py` — fitness improves or stays same after sweep, GPU result within 1e-4 of CPU minimax
- [ ] **6.7** `test_cpu_refine.py` — FP64 minimax matches step84 implementation, pair moves work
- [ ] **6.8** `test_generator.py` — all mutation types produce valid (19,9) factors, crossover preserves term count
- [ ] **6.9** `test_selector.py` — tournament returns correct K, elitism preserves best, prune removes duplicates
- [ ] **6.10** `test_end_to_end.py` — 3-generation run on in-memory DB, fitness improves, no crashes, DB consistent

### Phase 7: Performance Tuning

- [ ] **7.1** Profile GPU batch sizes — find throughput sweet spot (candidates/second)
- [ ] **7.2** Profile generator throughput — ensure gen doesn't bottleneck eval
- [ ] **7.3** SQLite tuning — WAL checkpoint interval, cache warming, VACUUM schedule
- [ ] **7.4** Backpressure calibration — tune MAX_PENDING so gen/eval stay balanced
- [ ] **7.5** Tier threshold tuning — optimize TIER1/TIER2 thresholds for recall vs throughput
- [ ] **7.6** Compare wall-clock: db_optimizer vs step84 EA on same time budget

### Phase 8: Advanced Features (stretch goals)

- [ ] **8.1** Adaptive mutation rates — per-island, based on improvement velocity
- [ ] **8.2** Basin detection — cluster shadow archive by support_hash prefix, detect distinct basins
- [ ] **8.3** Cross-basin injection — pick parents from different basins for crossover
- [ ] **8.4** L-BFGS on GPU — torch.optim.LBFGS with smooth-max objective (port of optimize_als.py)
- [ ] **8.5** Auto-blacklist — if N candidates with same support_hash all stall, blacklist it
- [ ] **8.6** Tuner UI integration — poll db_optimizer stats from step84_tuner.html

---

## Key Design Decisions

1. **SQLite over PostgreSQL/Redis**: Single-writer is fine at our throughput (~100 writes/sec).
   WAL mode gives concurrent reads. No infrastructure to manage. File on K:/ = persistence for free.

2. **Dense encoding (all 9,9,9)**: L-BFGS proved that the best solutions live in dense space.
   Sparse support logic from step84 is dropped. Support hashes still computed for dedup/blacklist.

3. **FP32 for GPU screening, FP64 for CPU refinement**: RTX 3060 has 1/32 FP64 throughput.
   FP32 gives ~7 digits — enough to screen at the 0.1 level. Final refinement needs FP64
   for 0.098-class solutions.

4. **No ALS**: Alternating least squares optimizes Frobenius norm, which actively hurts
   minimax. All local search is minimax-native (greedy coordinate descent).

5. **Newton polish only in Tier 3**: Scipy least_squares is 2s/call on CPU. Only run it
   on the ~10 best candidates per generation, not on all 10K+.

6. **Backpressure via DB count**: Generators check `SELECT COUNT(*) WHERE status='pending'`
   and pause if above threshold. This prevents unbounded queue growth.

## Files Reused from Main Repo

- Target tensor: `matrix_multiplication_tensor(3)` from step63
- Algebraic lookup: `_build_algebraic_lookup()` from step84 (1473 values)
- Support hash: `canonical_support_hash()` from step84_canonical_constraints
- V2 minimax logic: adapted from `optimize_v2.py` (greedy coordinate descent)
- L-BFGS smooth-max: adapted from `optimize_als.py` (stretch goal 8.4)
