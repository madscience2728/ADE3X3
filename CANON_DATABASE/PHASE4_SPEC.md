# PHASE 4 SPEC: AlphaTensor Deletion + Full-Packet Swap Search

## READ FIRST

Read `./docs/CANON_CONSTRAINTS.md` and `./docs/gate_progress.md` for gate definitions.

**Critical finding from Phase 3:** Gate 2 (Δ ⊂ span(H)) cannot be enforced incrementally. Any term that increases rank([Σ|N]) MUST also increase rank(N), violating Gate 2 at every intermediate step. Gate 2 is a *global* property of the full R-term configuration — it can only be checked (and satisfied) at the terminal depth with all R terms present. This kills all incremental assembly strategies.

**New approach:** Work with complete R-term configurations. Score them by gate diagnostics. Improve them by swapping terms from the TermDB.

Two tools, run in order:

1. **AlphaTensor Deletion Search** — trivial enumeration, could solve the problem in minutes
2. **Full-Packet Swap Optimizer** — discrete local search over the 387M-record DB

All output: `./CANON_DATABASE/` (data files) and `./CANON_DATABASE/scripts/` (tools)
TermDB: `./CANON_DATABASE/data/` (load with `TermDB("./CANON_DATABASE/data")`)

---

## TOOL 1: AlphaTensor Deletion Search

### What

The AlphaTensor R=23 decomposition of 3×3 matrix multiplication is known and exact. It has 23 terms with rank(H)=14, rank(N)=14, rank(SN)=23, and satisfies all gates. The coefficients are published.

**Question:** Can you delete 4 terms from AlphaTensor's 23 and get a valid R=19 decomposition?

**Search space:** C(23, 4) = 8,855 possible 4-term deletions. Trivially enumerable.

### The AlphaTensor R=23 Coefficients

The exact coefficients are in `./docs/ADE3x3_CANONICAL_OBJECT.md` at §51 (lines ~14152-14306), titled "REVERSE ENGINEERING + CANCELLATION VISUALIZATION". The 23 rank-1 terms are listed with their (α, β, γ) factors over small integers.

**Read that section and extract the 23 terms.** Each term k has:
- α_k: a 3×3 integer matrix (entries typically in {-1, 0, 1, 2})
- β_k: a 3×3 integer matrix
- γ_k: a 3×3 integer matrix

If the exact coefficients aren't in the canon doc (they may be referenced but stored elsewhere), search the codebase:
```bash
grep -r "AlphaTensor" ./docs/ ./CANON_DATABASE/ ./CANON_OPTIMIZER/ --include="*.py" --include="*.md" -l
```

If coefficients are unavailable locally, search the web for "AlphaTensor 3x3 matrix multiplication rank 23 coefficients" — these were published by DeepMind in the Nature paper supplementary materials (2022).

### Algorithm

```python
"""
alphatensor_delete.py — Exhaustive search over 4-term deletions of AlphaTensor R=23.

For each of the 8,855 ways to delete 4 terms from the 23-term decomposition:
  1. Take the remaining 19 terms
  2. Compute H, Delta, Sigma, Nuisance, [Sigma|Nuisance]
  3. Check Gate 1: rank(H) == 10
  4. Check Gate 2: rank([H|Delta]) == rank(H) (i.e., delta_leak == 0)
  5. Check Gate 3: rank([Sigma|Nuisance]) == 19
  6. If Gate 3 passes: solve for Gamma, verify T_hat == T exactly.

This is 8,855 small linear algebra problems. Total runtime: seconds.
"""
```

### Output

For each deletion, log:
- Which 4 terms were deleted (by index)
- rank(H), rank(N), delta_leak, rank(SN), augmented_gap
- sigma_innovation
- tensor_reconstruction_error (lstsq)
- conservation_violation

Sort by (delta_leak ASC, augmented_gap ASC, reconstruction_error ASC).

Print a Rich table of the top 20 deletions.

If ANY deletion yields delta_leak=0 AND augmented_gap=0: **SOLUTION FOUND.** Save immediately, verify independently, print in bright green.

If no deletion works, print the best near-miss and its diagnostics. This is still a useful result: "R=19 cannot be obtained by 4-term deletion from AlphaTensor R=23."

### File: `alphatensor_delete.py`

Self-contained. No dependency on TermDB (AlphaTensor's coefficients may not be in {-1,0,1} — they're in a slightly larger integer set). Pure numpy + Rich.

---

## TOOL 2: Full-Packet Swap Optimizer

### What

Start with a complete R-term packet (R record indices from TermDB). Improve it by swapping out terms — replacing 1, 2, or 3 records at a time with other records from the DB. Score each configuration by a lexicographic objective that prioritizes the gates in order.

### Scoring Function (Lexicographic)

The score is a tuple, compared lexicographically. LOWER is better for all components.

```python
def score_packet(db, indices):
    """
    Returns a tuple (s1, s2, s3, s4, s5) for lexicographic comparison.
    Lower is better at every position.
    """
    H = db.H[indices].astype(np.float64)
    sigma = db.sigma[indices].astype(np.float64)
    delta = db.compute_delta_batch(indices).astype(np.float64)
    nuisance = np.hstack([H, delta])
    SN = np.hstack([sigma, nuisance])
    
    R = len(indices)
    target_rank_H = R - 9
    
    rank_H = matrix_rank(H)
    rank_N = matrix_rank(nuisance)
    rank_SN = matrix_rank(SN)
    
    delta_leak = rank_N - rank_H
    augmented_gap = R - rank_SN
    sigma_innov = rank_SN - rank_N
    
    # Gate 1 violation: how far is rank(H) from target
    gate1_gap = abs(rank_H - target_rank_H)
    
    # Continuous delta residual (for breaking ties when delta_leak is equal)
    Q_H, _ = np.linalg.qr(H.T)  # (72 or 18 cols...)
    # Actually: delta containment is col(Delta) ⊂ col(H) in R^R
    # Project Delta columns onto col(H) complement
    U_H, _, _ = np.linalg.svd(H, full_matrices=True)
    # Column space of H is spanned by first rank_H left singular vectors
    P = U_H[:, :rank_H]  # (R, rank_H)
    Delta_proj = P @ P.T @ delta  # projection of delta onto col(H)
    delta_resid = np.linalg.norm(delta - Delta_proj)  # Frobenius
    
    # Reconstruction error via lstsq
    target = np.zeros((9, 9 + 72))
    target[:, :9] = 3.0 * np.eye(9)
    Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target.T, rcond=None)
    recon_err = np.max(np.abs(target.T - SN.T @ Gamma_T))
    
    return (
        gate1_gap,       # s1: Gate 1 first — must have correct rank(H)
        delta_leak,      # s2: Gate 2 — minimize leaked dimensions
        delta_resid,     # s3: Gate 2 continuous — minimize residual norm
        augmented_gap,   # s4: Gate 3 — minimize gap to full rank
        recon_err,       # s5: reconstruction quality
    )
```

### Move Types

The optimizer uses three move types, escalating in scope:

**Move 1: Single swap.** Replace 1 term with a random DB record.
- Sample a random index from [0, 387M)
- Replace a random position in the packet
- Score the new packet
- Accept if score improves (strictly lexicographically smaller)

**Move 2: Double swap.** Replace 2 terms simultaneously.
- Sample 2 random DB records
- Replace 2 random positions in the packet
- Score and accept if better

**Move 3: Triple swap.** Replace 3 terms.
- Same pattern, 3 records and 3 positions

**Move selection strategy:**
- Start with single swaps (fast, explores locally)
- After N_stall consecutive non-improving single swaps, escalate to double swaps
- After N_stall doubles, escalate to triples
- After N_stall triples, restart from a fresh random packet
- Configurable: N_stall = 1000 (default)

**Smart replacement sampling (optional optimization):**
Instead of sampling uniformly from 387M records, sample from the *hit pool* of the current packet's H-subspace. This biases toward records that are already H-compatible. But also mix in ~20% fully random records to allow the H-subspace itself to change.

### Parallelization

Each worker runs an independent swap search from a different random starting packet. Workers report their best scores periodically. The coordinator tracks the global best.

- 24 workers, each running independent searches
- Workers communicate via multiprocessing.Queue (score updates only)
- Coordinator logs global best, prints dashboard
- On solution found: save immediately, verify, signal all workers

### Starting Packets

Each worker generates its own random starting packet:

```python
def random_gate1_packet(db, R, target_rank_H):
    """
    Generate a random R-term packet with rank(H) = target_rank_H.
    
    Method: pick target_rank_H independent H-rows (basis),
    then pick (R - target_rank_H) H-rows in their span (dependents).
    Use db.query_subspace to find dependents.
    """
    # Pick random independent H-rows
    basis_indices = []
    while len(basis_indices) < target_rank_H:
        idx = np.random.randint(0, db.N_PAIRS)
        H_trial = db.H[np.array(basis_indices + [idx])]
        if np.linalg.matrix_rank(H_trial.astype(float)) == len(basis_indices) + 1:
            basis_indices.append(idx)
    
    # Query for dependents
    V = db.H[np.array(basis_indices)]
    hits = db.query_subspace(V)
    
    # Pick R - target_rank_H random dependents
    n_dep = R - target_rank_H
    dep_indices = np.random.choice(hits, size=n_dep, replace=False)
    
    return np.concatenate([basis_indices, dep_indices])
```

### Dashboard (Rich)

```
╔══════════════════════════════════════════════════════════════════════╗
║  R=19 SWAP OPTIMIZER — 24 workers                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  GLOBAL BEST SCORE                                                   ║
║  Gate1 gap: 0   Delta leak: 2   Delta resid: 3.412                 ║
║  Aug gap: 3     Recon err: 0.847                                    ║
║  Rank(H)=10 ✓  Rank(N)=12  Rank(SN)=16  σ_innov=4                ║
║                                                                      ║
║  SEARCH STATISTICS                                                   ║
║  Wall time: 12m 34s      Total swaps: 14,847,293                   ║
║  Improvements: 2,847     Restarts: 12                               ║
║  Rate: 19.7K swaps/sec   Stalls: 0/24 workers                      ║
║                                                                      ║
║  SCORE HISTORY (last 10 improvements)                                ║
║  (0, 4, 5.21, 5, 1.02) → (0, 3, 4.87, 4, 0.95) → (0, 2, 3.41,.  ║
║                                                                      ║
║  WORKER STATUS                                                       ║
║  ┌────┬──────────────────────┬────────────┬─────────┐               ║
║  │ ID │ Best score           │ Swaps      │ Stalled │               ║
║  ├────┼──────────────────────┼────────────┼─────────┤               ║
║  │  0 │ (0, 2, 3.41, 3, .8) │ 612,847    │ no      │               ║
║  │  1 │ (0, 3, 4.12, 4, .9) │ 589,221    │ no      │               ║
║  │ .. │ ...                  │ ...        │ ...     │               ║
║  └────┴──────────────────────┴────────────┴─────────┘               ║
║                                                                      ║
║  DELTA LEAK HISTOGRAM (across all workers' current best)             ║
║  0: ░         1: ██       2: ████   3: ████████                     ║
║  4: ██████    5: █████    6: ███    7+: ██                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### File Structure

```
./CANON_OPTIMIZER/core/
├── alphatensor_delete.py    # Tool 1: exhaustive deletion search
├── swap_optimizer.py        # Tool 2: main optimizer
├── swap_worker.py           # Single worker process
├── swap_scoring.py          # Score function (shared by workers)
├── swap_dashboard.py        # Rich TUI
├── swap_config.py           # Configuration
└── swap_main.py             # Entry point
```

### Entry Point

```
Usage:
    python swap_main.py                        # run swap optimizer for R=19
    python swap_main.py --rank 19              # explicit rank
    python swap_main.py --rank 13 19 20        # multiple ranks
    python swap_main.py --workers 12           # override worker count
    python swap_main.py --db ./data            # override DB path
    python swap_main.py --resume               # resume from checkpoint
    python swap_main.py --time-limit 3600      # stop after 1 hour
    python swap_main.py --target-delta-leak 0  # stop when delta_leak reaches 0
```

### Configuration

```python
@dataclass
class SwapConfig:
    R: int = 19
    target_rank_H: int = 10
    n_workers: int = 24
    
    # Move parameters
    stall_threshold: int = 1000     # non-improving swaps before escalation
    single_swap_prob: float = 0.7   # probability of single swap (vs double/triple)
    double_swap_prob: float = 0.2
    triple_swap_prob: float = 0.1
    hit_pool_bias: float = 0.8      # fraction of swaps drawn from hit pool vs random
    
    # Stopping conditions
    time_limit_seconds: int = 0     # 0 = no limit
    target_delta_leak: int = 0      # stop if reached
    max_restarts_per_worker: int = 100
    
    # Checkpointing
    checkpoint_interval: int = 60
    checkpoint_file: str = "./CANON_OPTIMIZER/core/swap_checkpoint.json"
    solution_file: str = "./CANON_OPTIMIZER/core/SOLUTION.json"
    
    # Logging
    log_improvements: bool = True
    log_file: str = "./CANON_OPTIMIZER/core/swap_improvements.jsonl"
    
    # DB
    db_path: str = "./CANON_DATABASE/data"
```

---

## CRITICAL IMPLEMENTATION NOTES

### 1. Score function must be FAST

The score function is called ~20K times/second/worker. It involves:
- 3 matrix_rank calls (H: 19×18, N: 19×72, SN: 19×81)
- 1 SVD for the delta projection
- 1 lstsq for reconstruction error

Total: ~0.05ms per call on these matrix sizes. That's 20K/s per core = 480K/s across 24 workers. This is the performance floor — don't add overhead.

Optimization: cache the SVDs. When doing a single swap (replacing 1 of 19 terms), 18 rows are unchanged. Use rank-1 update formulas for the SVD if possible. But correctness first — optimize later if profiling shows the score function is the bottleneck.

### 2. Don't recompute delta for unchanged terms

When swapping k terms, only recompute delta for the k new terms. Keep the other (R-k) delta rows cached. The delta computation involves `db.compute_delta(idx)` which is cheap but adds up at 20K/s.

```python
# Cache delta rows for the current packet
self.delta_cache = db.compute_delta_batch(self.indices)  # (R, 54)

# On swap at position p with new record idx:
self.delta_cache[p] = db.compute_delta(idx)
```

### 3. Gate 1 maintenance

When swapping a term, rank(H) might change. Reject swaps that break Gate 1 immediately:
```python
H_new = H.copy()
H_new[swap_pos] = db.H[new_idx]
if np.linalg.matrix_rank(H_new.astype(float)) != target_rank_H:
    continue  # skip this swap
```

This is a fast pre-filter that avoids computing the full score for most random swaps.

### 4. Worker isolation

Each worker has its own TermDB instance (memory-mapped, shared pages). No shared state except the output queue. Workers are fully independent — this is embarrassingly parallel.

### 5. Checkpoint format

```json
{
    "rank": 19,
    "global_best_score": [0, 2, 3.412, 3, 0.847],
    "global_best_indices": [1234, 5678, ...],
    "total_swaps": 14847293,
    "total_improvements": 2847,
    "total_restarts": 12,
    "wall_time_seconds": 754,
    "workers": [
        {"id": 0, "best_score": [0, 2, 3.41, 3, 0.8], "swaps": 612847},
        ...
    ]
}
```

### 6. Solution verification

When any score reaches (0, 0, 0.0, 0, 0.0) — Gate 1 perfect, Gate 2 zero leak, Gate 3 zero gap, zero reconstruction error — IMMEDIATELY:
1. Save to SOLUTION.json
2. Run independent verification (rebuild T from scratch, check every entry)
3. Print to console in bright green
4. Signal all workers to stop

---

## RUN ORDER

1. **First: `alphatensor_delete.py`** — 8,855 configurations, seconds to run. If this finds a solution, you're done. If not, you learn the best deletion and its gap from feasibility.

2. **Then: `swap_main.py`** — long-running discrete optimizer. Let it run for hours. The delta_leak histogram across workers tells you whether Gate 2 is approachable (leak decreasing over time) or walled (leak stuck at 4+).

---

## WHAT WE LEARNED FROM PHASES 1-3

- **Gate 2 is the universal bottleneck** over {-1,0,1}. Zero of 28,100 random completions passed it.
- **Gate 2 cannot be enforced incrementally.** Neutral ∩ SN-innovative = empty set (exact, not numerical).
- **Gate 2-compatible records exist** in every hit pool (~6.4% for R=19). The constraint is satisfiable per-record, just not incrementally assemblable.
- **Gate 2 is a global property** of all R terms simultaneously. It can only be checked at full depth.
- **σ_innovation = 0 is a Gate 2 cascade**, not an independent obstruction. Sigma analysis is only meaningful once Gate 2 is resolved.
- **The conservation law holds perfectly** (100% of sampled configs satisfy R + η_null = 27).

## DON'T FORGET

- **AlphaTensor deletion FIRST.** It's trivial and could end the project.
- **Score lexicographically.** Gate 1 > Gate 2 dims > Gate 2 continuous > Gate 3 > reconstruction.
- **No continuous optimization.** All moves are discrete record swaps from the 387M DB.
- **No incremental Gate 2.** Check Gate 2 only on complete R-term packets.
- **Gate 1 is a pre-filter.** Reject any swap that breaks rank(H) = target before scoring.
- **Multiprocessing, not threading.** 24 independent worker processes.
- **Rich dashboard is mandatory.**
- **Checkpoint every 60 seconds.**
