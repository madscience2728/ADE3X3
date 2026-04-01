# ADE3X3 Pipeline Redesign — Phase 2 Handoff

## Status Quo

**Repo:** `C:\Users\madsc\Desktop\Github\ADE3X3`  
**Current pipeline in `main.py`** (`_run_optimizer` function, ~500 lines):

```
Generator (Thread A) → GPU Screen (Thread B, Tier 1) → CPU Refine (Thread C, Tier 3) → DB
```

- **DB writer thread** (`_db_writer_loop`): Single thread owns the SQLite connection. All
  other threads submit ops via `_db_submit(op, args, sync=True/False)` to a `queue.Queue`.
  Writes are fire-and-forget; reads block via `_DbResult.get()`.
- **Activity tracker**: `_set_activity(stage, state, detail)` / `_get_activity()` — exposed
  at `/api/activity`. Dashboard polls every 2s, shows 5 glowing boxes.
- **`_work_available`**: `threading.Event` set by DB writer after inserts/updates, waited on
  by GPU/CPU when idle (replaces blind `sleep()`).
- **Dashboard**: `db_optimizer/static/{dashboard.html, dashboard.css, dashboard.js}` served
  from `main.py`. Has controls, charts, island table, leaderboard, hyperparameter editor.
- **71 tests** in `db_optimizer/tests/` — all passing.

### Key files to modify
| File | Role |
|---|---|
| `main.py` | Pipeline orchestration, HTTP server, dashboard API |
| `db_optimizer/gpu_eval.py` | GPU fitness eval (currently only Tier 1 einsum + max-abs) |
| `db_optimizer/cpu_refine.py` | CPU FP64 minimax coordinate descent |
| `db_optimizer/config.py` | All constants: `RANK=19, DIM=9, GPU_MINIMAX_*`, island config |
| `db_optimizer/generator.py` | Mutation operators (all Python loops, could be GPU-ified later) |
| `db_optimizer/selector.py` | DB query helpers (tournament selection, fetch_pending, etc.) |
| `db_optimizer/schema.py` | SQLite schema (candidates, shadow, blacklist, run_log) |
| `db_optimizer/static/dashboard.html` | Dashboard HTML |
| `db_optimizer/static/dashboard.css` | Dashboard CSS (activity box animations) |
| `db_optimizer/static/dashboard.js` | Dashboard JS (polling, charts, start/stop) |

### Seed files available
- `optimized_als_r10_at_0.09.json` in repo root
- `optim_workers/als_worker_00..23.json` (24 files)
- `optim_workers/v2_worker_00..23.json` (24 files)
- `optim_workers/worker_00..23.json` (24 files)
- Format: `{"terms": [{alpha_support, alpha_values, beta_support, beta_values, gamma_support, gamma_values}, ...]}`

---

## Task 1: Seed Picker in Dashboard

### Backend: `/api/seeds` GET endpoint
Add to `DashboardHandler.do_GET`:
```python
elif path == "/api/seeds":
    seeds = []
    # Scan repo root for optimized_*.json
    for p in sorted(REPO_ROOT.glob("optimized_*.json")):
        seeds.append({"path": str(p.relative_to(REPO_ROOT)), "name": p.stem})
    # Scan optim_workers/
    workers_dir = REPO_ROOT / "optim_workers"
    if workers_dir.exists():
        for p in sorted(workers_dir.glob("*.json")):
            seeds.append({"path": str(p.relative_to(REPO_ROOT)), "name": p.stem})
    self._json_response({"seeds": seeds, "total": len(seeds)})
```

### Backend: `/api/start` POST — accept `seed_mode` and `seed_files`
In config_overrides, accept:
- `"seed_mode": "fresh"` — create empty DB, no seeds
- `"seed_mode": "auto"` — seed from all available files (current behavior)
- `"seed_mode": "pick"` + `"seed_files": ["optim_workers/v2_worker_03.json", ...]`
- `"seed_mode": "resume"` — use existing DB as-is

Pass `seed_mode`/`seed_files` into `_run_optimizer`. Modify the seeding block to respect these.

### Frontend: seed picker UI
In dashboard.html, add a section above the Start button:
```html
<div class="panel__header panel__header--spaced"><h2>Seed Selection</h2></div>
<div class="seed-controls">
    <select id="seed-mode">
        <option value="auto">Auto (all seeds)</option>
        <option value="pick">Pick specific seeds</option>
        <option value="fresh">Fresh start (empty DB)</option>
        <option value="resume">Resume existing DB</option>
    </select>
    <div id="seed-picker" style="display:none">
        <!-- Populated dynamically from /api/seeds -->
        <div id="seed-list" class="seed-checklist"></div>
        <button type="button" class="action-button" id="btn-seed-all">Select All</button>
    </div>
</div>
```
In dashboard.js: fetch `/api/seeds` on load, populate checkboxes, include `seed_mode` +
`seed_files` in the POST body to `/api/start`.

---

## Task 2: Power-Law Islands (Explore/Exploit)

### Concept
Instead of N_ISLANDS equal-sized islands, use **power-law scaling**. Small islands exploit
(tight selection pressure, best candidates only), large islands explore (loose selection,
high diversity).

### New config in `config.py`
```python
# ── Power-law islands ────────────────────────────────────────────────
# Each island has a "size class" controlling selection pressure.
# Population proportional to 2^(island_index + 2): 4, 8, 16, 32, 64, 128, 256, 512
# Fewer islands but exponential gradient from exploit → explore.
ISLAND_SCHEDULE = [4, 8, 16, 32, 64, 128, 256, 512]
N_ISLANDS = len(ISLAND_SCHEDULE)  # 8 islands

# Per-island parameters derived from size class:
#   - Elite K:         max(2, size // 4)
#   - Tournament size: varies from 8 (exploit) → 2 (explore)
#   - Mutation sigma:  varies from 0.001 (exploit) → 0.05 (explore)
#   - P(crossover):    varies from 0.02 (exploit) → 0.30 (explore)
```

### Island assignment
Currently: `v_island = hash(s_hash) % N_ISLANDS`.
Change to weighted distribution — new candidates go to larger islands more often:
```python
weights = np.array(ISLAND_SCHEDULE, dtype=float)
weights /= weights.sum()
v_island = rng.choice(N_ISLANDS, p=weights)
```

### Per-island behavior in generator
```python
def get_island_params(island_idx):
    """Return mutation parameters tuned for this island's role."""
    size = ISLAND_SCHEDULE[island_idx]
    total = sum(ISLAND_SCHEDULE)
    exploit_frac = 1.0 - (size / total)  # small island → high exploit

    return {
        "elite_k": max(2, size // 4),
        "tournament_size": int(2 + 6 * exploit_frac),       # 8 for exploit, 2 for explore
        "p_gaussian_sigma": 0.001 + 0.049 * (1 - exploit_frac),  # tight for exploit
        "p_crossover": 0.02 + 0.28 * (1 - exploit_frac),    # high for explore
        "p_coeff": 0.70 - 0.20 * (1 - exploit_frac),        # dominant for exploit
    }
```

### Migration
Small → large: "escape" — when an exploit island stagnates, its best candidate migrates
to a larger explore island with heavy mutation.
Large → small: "promotion" — when an explore island finds a structurally novel candidate
(new support_hash not seen in small islands), copy it to the smallest island for drilling.

### Dashboard update
- Island table gets a "Role" column showing "Exploit-4" through "Explore-512"
- Per-island chart uses red→blue color gradient (exploit→explore)

---

## Task 3: GPU Minimax Refinement (Tier 2)

### New function in `gpu_eval.py`
```python
def gpu_minimax_refine(
    factors_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    sweeps: int = 1,
    n_trials: int = 32,
    fine_range: float = 0.003,
    device: torch.device | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
```

**Algorithm per sweep:**
For each of the 513 coefficients (3 factors × 19 ranks × 9 dims):
1. Current residual already computed: `R = recon - target` shape `(N, 9, 9, 9)`
2. For this coefficient position `(factor_idx, rank, dim)`:
   - Generate `n_trials` trial values per candidate: linspace around current value ± fine_range,
     plus nearest algebraic values (precomputed on GPU)
   - Shape: `(N, n_trials)`
3. For each trial: compute the rank-1 update `delta_outer` and new residual max-abs
   - This is the key GPU operation: `(N, n_trials)` evaluations simultaneously
   - Use the "rank-1 update trick": only recompute the one term that changed
4. Pick best trial per candidate: `argmin` over trials dimension
5. Accept if improved (vectorized comparison)
6. Update residual tensor in-place

**Memory estimate** for N=10,000 candidates, n_trials=32:
- Factors: 3 × (10K, 19, 9) × 4 bytes = 20 MB
- Residual: (10K, 9, 9, 9) × 4 bytes = 29 MB  
- Trial values: (10K, 32) × 4 bytes = 1.2 MB
- Trial residuals: (10K, 32, 9, 9, 9) × 4 bytes = 933 MB ← too big!

**Fix**: Don't materialize all trial residuals. Instead, compute max-abs incrementally:
```python
# For each coefficient (fi, ri, di):
# old_outer[n, a, b, c] = alpha[n, ri, a] * beta[n, ri, b] * gamma[n, ri, c]
# For each trial t:
#   new_val = trials[n, t]
#   Temporarily set factors[fi][n, ri, di] = new_val
#   new_outer = alpha[n,ri,:] * beta[n,ri,:] * gamma[n,ri,:]  (with the changed coeff)
#   delta = new_outer - old_outer  → shape (N, 9, 9, 9)
#   R_trial = R + delta
#   trial_fitness[n, t] = R_trial.abs().reshape(N, -1).max(dim=1)
```
But delta is rank-1, so it decomposes:
```python
# delta[a,b,c] = new_outer[a,b,c] - old_outer[a,b,c]
# Since only one coeff changed in one factor, the change in the outer product is:
#   If we changed alpha[ri, di]:
#     delta_alpha = new_val - old_val  (scalar per candidate per trial)
#     delta[a,b,c] = delta_alpha * (a==di) * beta[ri,b] * gamma[ri,c]
#     → this is a rank-1 update: delta = delta_alpha * e_di ⊗ beta[ri,:] ⊗ gamma[ri,:]
# So R_trial = R + delta_alpha * e_di ⊗ beta[ri,:] ⊗ gamma[ri,:]
# The max-abs of R_trial can be computed by only looking at the slice R[:, di, :, :] (alpha case)
# Actually no — the residual has contributions from all terms, so we need the full max.
# But we can compute it as: for each trial, add the delta to R and take max.
# Since delta is sparse (only affects one slice), we can batch this efficiently.
```

**Practical GPU implementation** — process one coefficient at a time across all N candidates:
```python
for fi, ri, di in coefficient_order:  # 513 iterations
    # trials shape: (N, n_trials) — generated on GPU
    # old_outer shape: (N, 9, 9, 9) — current rank-1 term
    # For each trial, compute delta_outer and resulting max-abs
    # This is N * n_trials independent max operations on 729-element tensors
    # → (N * 32) = 320K max operations — good GPU utilization
```

**Batch size tuning**: With 12GB VRAM, N=10K candidates × 32 trials is fine.
Process candidates in sub-batches of 10K if more than that pass screening.

### Integration into pipeline
New pipeline stages:
```
Generator → GPU Screen (Tier 1) → GPU Minimax (Tier 2) → CPU Polish (Tier 3) → DB
```

In `main.py`, the current `_gpu_screen_loop` becomes a two-phase function:
1. Fetch pending → GPU batch fitness → filter by TIER1_SURVIVOR_THRESHOLD
2. Take survivors → GPU minimax refine (1-2 sweeps) → filter by TIER2_REFINE_THRESHOLD
3. Put survivors into `gpu_out_queue` for CPU

Add to activity tracking: `"gpu_minimax"` stage with its own box in dashboard.

---

## Task 4: Adaptive CPU Polish with Preemption

### Modified `cpu_minimax_refine` in `cpu_refine.py`
Add a `check_preempt` callback:
```python
def cpu_minimax_refine(
    alpha, beta, gamma,
    sweeps=5,             # default sweeps
    min_sweeps=3,         # always do at least this many
    max_sweeps=20,        # keep going if no preemption
    fine_range=0.003,
    patience=2,
    check_preempt=None,   # callable() → bool, "should I stop early?"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
```

**Logic per sweep:**
```python
for sweep in range(max_sweeps):
    improved = do_one_sweep(...)
    if sweep >= min_sweeps:
        if check_preempt and check_preempt():
            break  # GPU has better candidates waiting
        if not improved:
            stale += 1
            if stale >= patience:
                break
    else:
        if not improved:
            stale += 1
```

### In `main.py`, the CPU refine loop changes:
- Replace `fetch_refine_candidates` DB queries with reading from `gpu_out_queue`
- Each worker gets a `preempt_event` (one per worker via a factory)
- GPU thread sets all preempt events when it finishes a minimax batch
- Workers check their event between sweeps

```python
gpu_out_queue = queue.Queue(maxsize=N_CPU_WORKERS * 4)
preempt_event = threading.Event()

def _cpu_refine_loop():
    with ProcessPoolExecutor(max_workers=N_CPU_WORKERS) as pool:
        while _is_running():
            try:
                row_id, alpha, beta, gamma = gpu_out_queue.get(timeout=0.1)
            except queue.Empty:
                continue  # GPU hasn't produced anything yet

            fut = pool.submit(
                cpu_minimax_refine, alpha, beta, gamma,
                min_sweeps=3, max_sweeps=20,
                # Note: check_preempt can't cross process boundary easily.
                # Instead, use sweeps=min_sweeps when queue is non-empty,
                # sweeps=max_sweeps when queue is empty.
            )
            ...
```

**Simpler preemption (recommended)**: Don't use a callback. Instead, the CPU loop checks
`gpu_out_queue.qsize()` before submitting. If queue has items waiting, use `sweeps=3`.
If queue is empty, use `sweeps=20`. This way workers doing deep polish aren't killed
mid-sweep, but new submissions are adaptive.

---

## Task 5: Dashboard Updates

### Seed picker (Task 1 frontend)
- Add `<select id="seed-mode">` with options: Auto / Pick / Fresh / Resume
- Add `<div id="seed-picker">` with checkboxes populated from `/api/seeds`
- JS: show/hide seed-picker based on select value
- Include `seed_mode` and `seed_files` in POST to `/api/start`

### Explore/exploit slider
- Add slider `<input type="range" id="cfg-exploit-ratio" min="0" max="100">` 
- Label: "Exploit ◄──► Explore"
- This controls the overall balance: shifts the island size schedule
- At 100% exploit: schedule becomes [4, 4, 4, 4, 8, 8, 8, 8] (all small)
- At 100% explore: schedule becomes [64, 128, 256, 512, 512, 512, 512, 512] (all big)
- Default 50%: [4, 8, 16, 32, 64, 128, 256, 512]

### Activity boxes — add 2 new stages
Current 5 boxes: Generator, GPU Screen, CPU Refine, DB I/O, Monitor
Add: **GPU Minimax** (between GPU Screen and CPU Refine)
Total 6 boxes. Update:
- `dashboard.html`: add `<div class="activity-box" id="act-gpu_minimax">`
- `dashboard.js`: add `"gpu_minimax"` to `ACTIVITY_STAGES` array
- `main.py`: add `"gpu_minimax"` to `_pipeline_activity` dict
- `dashboard.css`: grid changes to `grid-template-columns: 1fr 1fr 1fr` for 6 boxes

### Island table enhancement
Add "Role" column and color-code rows:
- Exploit islands (small): warm/red tint
- Explore islands (large): cool/blue tint
- Show island size class in the Role column

---

## Task 6: Config Changes Summary

### `db_optimizer/config.py` additions
```python
# ── Power-law islands ────────────────────────────────────────────────
ISLAND_SCHEDULE = [4, 8, 16, 32, 64, 128, 256, 512]
N_ISLANDS = len(ISLAND_SCHEDULE)  # replaces old N_ISLANDS = 12

# ── GPU minimax (Tier 2) — already partially defined ─────────────────
GPU_MINIMAX_SWEEPS = 1            # already exists
GPU_MINIMAX_FINE_RANGE = 0.003    # already exists
GPU_MINIMAX_N_TRIALS = 32         # already exists
GPU_MINIMAX_BATCH_SIZE = 10_000   # NEW: max candidates per GPU minimax batch

# ── Adaptive CPU polish (Tier 3) ─────────────────────────────────────
CPU_MINIMAX_MIN_SWEEPS = 3        # NEW: always do at least this many
CPU_MINIMAX_MAX_SWEEPS = 20       # NEW: keep going if no better work
CPU_MINIMAX_SWEEPS = 5            # still used as default
```

### `main.py` config_overrides additions
Accept from dashboard:
- `island_schedule` (list of ints)
- `gpu_minimax_sweeps`, `gpu_minimax_batch`
- `cpu_min_sweeps`, `cpu_max_sweeps`
- `seed_mode`, `seed_files`

---

## Implementation Order

1. **Seed picker** (UI + backend) — standalone, no pipeline changes
2. **Power-law islands** (config + generator + selector + dashboard) — modifies island logic
3. **GPU minimax** (`gpu_eval.py` new function + pipeline integration) — the big one
4. **Adaptive CPU** (`cpu_refine.py` + pipeline queue handoff) — depends on GPU minimax
5. **Dashboard updates** (activity boxes, island colors, exploit slider) — cosmetic

Each step should leave tests passing. Run `python -m pytest db_optimizer/tests/ -q --tb=short`
after each change.

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │                  DB Writer Thread               │
                    │  (single SQLite conn, drains _db_queue)         │
                    └────────────▲───────────▲──────────▲─────────────┘
                                 │           │          │
                    _db_submit() │           │          │  _db_submit()
                                 │           │          │
┌──────────────┐   ┌─────────────┴──┐   ┌────┴──────┐  ┌┴──────────────┐
│  Generator   │──▶│  GPU Screen    │──▶│GPU Minimax│─▶│  CPU Polish   │
│  Thread A    │   │  Thread B      │   │(in B)     │  │  Thread C     │
│              │   │  Tier 1: eval  │   │Tier 2:    │  │  Tier 3:      │
│  mutate/xo   │   │  max-abs       │   │coord desc │  │  FP64 refine  │
│  8192/cycle  │   │  100K batch    │   │1-2 sweeps │  │  3-20 sweeps  │
└──────────────┘   └────────────────┘   └───────────┘  └───────────────┘
       │                                       │              │
       │           _work_available.set()       │   gpu_out_queue
       └───────────────────────────────────────┘──────────────┘

Power-Law Islands:
  Island 0: size=4    → Exploit (tight selection, small sigma)
  Island 1: size=8    → Exploit
  Island 2: size=16   → Mixed
  Island 3: size=32   → Mixed
  Island 4: size=64   → Mixed
  Island 5: size=128  → Explore
  Island 6: size=256  → Explore
  Island 7: size=512  → Explore (loose selection, big sigma, high crossover)

Migration: small→large on stagnation, large→small on structural novelty
```
