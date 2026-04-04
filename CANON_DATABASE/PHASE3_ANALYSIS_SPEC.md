# PHASE 3 SPEC: Packet Analysis — Mining the Near-Miss Landscape

## CONTEXT

Stage 1 has harvested ~256+ "packets" per rank for R=13, R=19, and R=20. Each packet is a candidate configuration: a set of R record indices from the TermDB, each with precomputed H-rows, sigma-rows, and (α, β) factors. These packets were found by greedy-random sampling of 10-dim subspaces (for R=19), not exhaustive search.

**None of these packets are solutions.** They are near-misses — configurations that pass Gate 1 (rank(H) = target) or come close, but fail at Gate 2 or Gate 3. The question is: **what do the failures look like, and do they reveal structural obstructions?**

The analysis script does NOT search for solutions. It reads the existing packet files, computes every diagnostic the canon defines, and produces a statistical portrait of the near-miss landscape. This portrait either reveals a new gate (an obstruction that blocks all sampled packets uniformly) or confirms that the landscape is heterogeneous (no universal obstruction, solutions may exist but are sparse).

**Read these files before writing code:**
- `./docs/CANON_CONSTRAINTS.md` — All gate definitions, diagnostic formulas, the conservation law
- `./docs/gate_progress.md` — Known generic ranks per orbit, Gate 2 generically free, Gate 3 bottleneck
- `./docs/SYMMETRY_CANON.md` — The wreath product structure, orbit decomposition
- `./CANON_DATABASE/term_db.py` — TermDB class for retrieving factors and computing blocks

---

## INPUT

Packet files from Stage 1, located in `./CANON_OPTIMIZER/core/` or `./CANON_DATABASE/` (check both). Expected format: JSON-lines or numpy archives, each containing:
- `rank`: the target R
- `basis_indices`: list of record indices (into TermDB) for the basis terms
- `hit_indices`: list of record indices for the dependent terms (may be empty if assembly didn't run)
- `all_indices`: combined basis + dependent indices (the full R-term config)
- Any pre-computed diagnostics from Stage 1

**If the packet format differs from the above, inspect the actual files first and adapt.** Read whatever Stage 1 produced. The packet format was defined by Copilot during Stage 1 implementation — read the Stage 1 code to determine the exact schema.

All output: `./CANON_OPTIMIZER/analysis/`

---

## THE ANALYSIS SCRIPT

### File: `analyze_packets.py`

Single script, self-contained except for TermDB import. Reads all packets, computes all diagnostics, produces:
1. A Rich console report with summary tables
2. A detailed CSV of per-packet diagnostics
3. Statistical summaries and distributions
4. A markdown report with findings and interpretations

```
Usage:
    python analyze_packets.py                      # analyze all available packets
    python analyze_packets.py --rank 19            # analyze R=19 only
    python analyze_packets.py --db ./data           # override DB path
    python analyze_packets.py --output ./analysis   # override output directory
    python analyze_packets.py --verbose             # print per-packet details
```

---

## DIAGNOSTICS TO COMPUTE PER PACKET

For each packet (a set of R record indices), load the TermDB, retrieve the (α, β) factors, and compute everything below. Use `term_db.py` for factor retrieval and `numpy` for linear algebra. All rank computations use SVD with threshold 1e-10 × max(sv).

### Block A: Core Gate Diagnostics

| Metric | Formula | Target | Notes |
|--------|---------|--------|-------|
| `rank_H` | rank of (R, 18) matrix H = [Eta1 \| Eta2] | R - 9 | Gate 1 |
| `rank_N` | rank of (R, 72) matrix [H \| Delta] | = rank_H | Gate 2 |
| `delta_leak` | rank_N - rank_H | 0 | Gate 2 residual |
| `rank_sigma` | rank of (R, 9) matrix Sigma | 9 | Sigma quality |
| `rank_SN` | rank of (R, 81) matrix [Sigma \| Nuisance] | R | Gate 3 (augmented rank) |
| `augmented_gap` | R - rank_SN | 0 | How far from Gate 3 |
| `eta_nullity` | 18 - rank_H | R target: 8 (for R=19) | Conservation law input |
| `conservation` | R + eta_nullity | 27 | Must equal n³ = 27 |
| `conservation_violation` | abs(R + eta_nullity - 27) | 0 | Deviation from law |

### Block B: Sigma Analysis (the Gate 3 bottleneck)

| Metric | Formula | Notes |
|--------|---------|-------|
| `sigma_singular_values` | SVD of Sigma (R, 9) | Full spectrum, all 9 values |
| `sigma_condition_number` | max(sv) / min(nonzero sv) | Conditioning of Sigma |
| `sigma_rank_gap` | 9 - rank_sigma | How many sigma dimensions missing |
| `sigma_in_nuisance_span` | dim(col(Sigma) ∩ col(Nuisance)) | Sigma-nuisance overlap |
| `sigma_independent_of_N` | rank([Sigma \| Nuisance]) - rank(Nuisance) | Independent sigma directions |
| `sigma_innovation` | = sigma_independent_of_N | Must equal 9 for Gate 3 |

**This is the critical diagnostic.** The canon says Gate 3 is the universal bottleneck. `sigma_innovation` measures how many of Sigma's 9 columns contribute directions outside the nuisance span. If this clusters at 7 or 8 across all packets, the missing 1-2 directions are the obstruction.

### Block C: Delta Containment Analysis

| Metric | Formula | Notes |
|--------|---------|-------|
| `delta_rank` | rank of (R, 54) Delta matrix | How many delta directions |
| `delta_in_H_span` | rank(H) - rank([H \| Delta]) == 0 | Boolean: full containment? |
| `delta_residual_frobenius` | ‖Delta - H @ lstsq(H, Delta)‖_F | Continuous leak measure |
| `delta_residual_max` | max entry of above | Worst-case leak |
| `delta_leak_dims` | rank([H \| Delta]) - rank(H) | Number of leaked dimensions |
| `per_channel_leak` | For each (s,t) pair with s≠t: does Delta[s,t] leak outside span(H)? | 6 booleans |

### Block D: Conservation Law and Spectral Analysis

| Metric | Formula | Notes |
|--------|---------|-------|
| `H_singular_values` | Full SVD of H | All min(R, 18) values |
| `H_spectral_gap` | sv[target_rank-1] - sv[target_rank] | Gap at the target rank cutoff |
| `nuisance_singular_values` | SVD of Nuisance (R, 72) | Spectral profile |
| `SN_singular_values` | SVD of [Sigma \| Nuisance] (R, 81) | Full augmented spectrum |
| `SN_spectral_gap` | sv[R-1] if R ≤ min(R,81) | Gap at rank R |

**The spectral gap** tells you whether the rank deficiency is "hard" (big gap between the last nonzero and first zero singular value) or "soft" (gradual decay through the cutoff). Hard gaps mean algebraic obstruction. Soft gaps mean you're close and the sampling might be unlucky.

### Block E: Omega Operator (from CANON_CONSTRAINTS.md §76/§83)

For packets where rank(H) = target and delta_leak = 0:

| Metric | Formula | Notes |
|--------|---------|-------|
| `Z_dim` | dim(ker(H^T) ∩ ker(Delta^T)) | Sigma-silent sector dimension |
| `omega_exists` | Z_dim == 9 | Required for valid decomposition |
| `omega_eigenvalues` | If Z_dim = 9: eigenvalues of 3·(Sigma_Z · Sigma_Z^T)^{-1} | Should be PD |
| `omega_positive_definite` | All eigenvalues > 0? | Must be true for solutions |
| `omega_condition` | max(eig) / min(eig) | How well-conditioned |

The Omega operator is the §83 Synthesis obstruction: if Z_dim < 9 or Omega isn't positive definite, the packet is structurally incompatible with exact multiplication. This is only computable when Gates 1+2 pass.

### Block F: Gamma Solve Attempt

For every packet regardless of gate status:

| Metric | Formula | Notes |
|--------|---------|-------|
| `gamma_lstsq_residual` | ‖[Sigma \| Nuisance]^T @ Gamma^T - [3I_9 \| 0]^T‖_F via lstsq | How close to solvable |
| `gamma_max_entry_residual` | max abs entry of residual | Infinity-norm residual |
| `tensor_reconstruction_error` | ‖T - T_hat‖_∞ after lstsq Gamma | The "fitness" |
| `tensor_reconstruction_rmse` | ‖T - T_hat‖_F / sqrt(729) | RMS fitness |

Even when Gate 3 fails, the least-squares Gamma gives a "best attempt" reconstruction. The residual pattern reveals which tensor entries are hardest to satisfy.

### Block G: Structural Classification

| Metric | Formula | Notes |
|--------|---------|-------|
| `n_zero_alpha_rows` | Count of all-zero rows across all α_k | Degenerate factor count |
| `n_zero_beta_cols` | Count of all-zero columns across all β_k | Degenerate factor count |
| `alpha_density` | Fraction of nonzero entries in α stack | Sparsity measure |
| `beta_density` | Fraction of nonzero entries in β stack | Sparsity measure |
| `n_rank1_terms` | Count of terms where rank(α_k ⊗ β_k outer product) = 1 | Should be all of them (by construction) |
| `fiber_partition` | For each term, which fiber (r,u) it primarily serves | Based on argmax of abs(sigma_row) |
| `fiber_partition_type` | Sorted tuple of fiber term counts | e.g., (1,2,2,2,2,2,2,2,4) |

---

## AGGREGATE STATISTICS

After computing per-packet diagnostics, aggregate across all packets for each rank:

### Distribution Tables

For each numeric diagnostic, report:
- min, max, mean, median, std
- histogram (10 bins)
- count at exact target value (e.g., how many packets have rank_H == 10 exactly)

### Correlation Matrix

Compute pairwise Pearson correlations between all continuous diagnostics:
- rank_H, rank_N, delta_leak, rank_sigma, rank_SN, augmented_gap, sigma_innovation,
  delta_residual_frobenius, gamma_lstsq_residual, tensor_reconstruction_error,
  H_spectral_gap, alpha_density, beta_density

Output as a heatmap-style ASCII table with Rich, and save as CSV.

### Conditional Analysis

**Condition on rank_H = target** (Gate 1 passes):
- Of these, what fraction pass Gate 2 (delta_leak = 0)?
- Of Gate 2 passers, what is the distribution of sigma_innovation?
- Of Gate 2 passers, what is the distribution of augmented_gap?
- What is the maximum sigma_innovation seen? Does it ever reach 9?

**Condition on conservation = 27:**
- What fraction of packets satisfy the conservation law?
- Is there a correlation between conservation violation and tensor_reconstruction_error?

**Condition on sigma_innovation ≥ 8:**
- How many packets are "one sigma dimension away"?
- What do their Omega operators look like?

---

## OUTPUT FILES

### 1. `analysis_report.md` — Human-readable markdown report

Structure:
```markdown
# Packet Analysis Report — R=19 (N=256 packets)
## Executive Summary
[1 paragraph: key finding — is there a universal obstruction or not?]
## Gate Pass Rates
[Table: Gate 1/2/3 pass counts and rates]
## Sigma Innovation Distribution
[The critical plot — histogram of sigma_independent_of_N]
## Conservation Law
[Table: conservation violation statistics]
## Spectral Analysis
[H and SN spectral gap distributions]
## Delta Containment
[Gate 2 conditional analysis]
## Omega Operator
[For Gate 1+2 passers: Omega statistics]
## Gamma Residuals
[Distribution of fitness values]
## Structural Observations
[Fiber partition types, sparsity patterns]
## Conclusion
[Assessment: is R=19 over {-1,0,1} likely obstructed? What would Gate 4 look like?]
```

### 2. `packets_R19_diagnostics.csv` — Full per-packet data

One row per packet, one column per diagnostic. Machine-readable for further analysis.

### 3. `packets_R19_correlations.csv` — Correlation matrix

### 4. Console output — Rich tables

Print a Rich summary to console during execution:

```
╔══════════════════════════════════════════════════════════════════════╗
║  PACKET ANALYSIS — R=19 (256 packets)                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  GATE PASS RATES                                                     ║
║  ┌────────────────┬──────────┬──────────┬──────────────┐            ║
║  │ Gate           │ Tested   │ Passed   │ Rate         │            ║
║  ├────────────────┼──────────┼──────────┼──────────────┤            ║
║  │ Gate 1 (rk H)  │ 256      │ 47       │ 18.4%        │            ║
║  │ Gate 2 (Δ⊂H)  │ 47       │ 45       │ 95.7%        │            ║
║  │ Gate 3 (Γ)     │ 45       │ 0        │ 0.0%         │            ║
║  └────────────────┴──────────┴──────────┴──────────────┘            ║
║                                                                      ║
║  SIGMA INNOVATION (Gate 2 passers only)                              ║
║  ┌─────────┬───────┐                                                ║
║  │ σ_innov │ Count │    ← THIS IS THE KEY TABLE                    ║
║  ├─────────┼───────┤                                                ║
║  │ 5       │ 2     │                                                ║
║  │ 6       │ 8     │                                                ║
║  │ 7       │ 19    │                                                ║
║  │ 8       │ 14    │    ← "one away" — promising or obstructed?    ║
║  │ 9       │ 2     │    ← if ANY hit 9: Gate 3 may be passable!   ║
║  └─────────┴───────┘                                                ║
║                                                                      ║
║  AUGMENTED RANK (all packets)                                        ║
║  max: 18/19   median: 16   mode: 17                                ║
║                                                                      ║
║  CONSERVATION LAW                                                    ║
║  R + η_null = 27: 47/256 (18.4%)   violation range: [0, 8]        ║
║                                                                      ║
║  RECONSTRUCTION ERROR                                                ║
║  min: 0.073   median: 0.412   max: 2.891                           ║
║                                                                      ║
║  SPECTRAL GAP (H, at rank 10 cutoff)                                ║
║  min: 0.001   median: 0.234   max: 1.847                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## CRITICAL IMPLEMENTATION NOTES

### 1. Loading the DB

```python
from term_db import TermDB
db = TermDB("./CANON_DATABASE/data")
db.load_to_ram()
```

All factor retrieval goes through the DB:
```python
alpha, beta = db.get_factors(idx)           # single record
alphas, betas = db.get_factors_batch(indices)  # batch
H_rows = db.H[indices]                       # precomputed
sigma_rows = db.sigma[indices]                # precomputed
deltas = db.compute_delta_batch(indices)      # computed on demand
```

### 2. Computing Nuisance and [Σ|N]

```python
H = db.H[indices].astype(np.float64)              # (R, 18)
sigma = db.sigma[indices].astype(np.float64)        # (R, 9)
delta = db.compute_delta_batch(indices).astype(np.float64)  # (R, 54)
nuisance = np.hstack([H, delta])                     # (R, 72)
SN = np.hstack([sigma, nuisance])                    # (R, 81)
```

### 3. Sigma Innovation Calculation

This is the single most important diagnostic:

```python
# How many independent directions does Sigma add to the Nuisance span?
rank_N = np.linalg.matrix_rank(nuisance)
rank_SN = np.linalg.matrix_rank(SN)
sigma_innovation = rank_SN - rank_N
# Target: 9 (all 9 sigma columns independent of nuisance)
```

### 4. Omega Operator

Only computable when rank(H) = target AND delta_leak = 0:

```python
# Z = ker(H^T) ∩ ker(Delta^T) — the sigma-silent sector
# In R^R, find vectors w such that H^T @ w = 0 and Delta^T @ w = 0
# Equivalently: w ∈ ker(Nuisance^T)
N = nuisance.T  # (72, R)
_, S_N, Vt_N = np.linalg.svd(N, full_matrices=True)
null_start = np.sum(S_N > 1e-10)
Z_basis = Vt_N[null_start:]  # (Z_dim, R) — rows are basis of ker(N^T)

Z_dim = Z_basis.shape[0]
if Z_dim == 9:
    Sigma_Z = Z_basis @ sigma  # (9, 9) — Sigma projected onto Z
    # Omega = 3 * (Sigma_Z @ Sigma_Z^T)^{-1}  ... but check invertibility first
    G = Sigma_Z @ Sigma_Z.T  # Gram matrix
    eigs = np.linalg.eigvalsh(G)
    omega_pd = np.all(eigs > 1e-10)
    if omega_pd:
        Omega = 3.0 * np.linalg.inv(G)
        omega_eigs = np.linalg.eigvalsh(Omega)
```

### 5. Tensor Reconstruction

```python
def build_T():
    T = np.zeros((9, 9, 9))
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[r*3+s, s*3+u, r*3+u] = 1
    return T

def reconstruct(alphas, betas, Gamma):
    """alphas, betas: (R, 3, 3), Gamma: (9, R)"""
    T_hat = np.zeros((9, 9, 9))
    for k in range(alphas.shape[0]):
        T_hat += np.einsum('i,j,l', alphas[k].ravel(), betas[k].ravel(), Gamma[:, k])
    return T_hat

# Least-squares Gamma regardless of gate status
target = np.zeros((9, 81))
target[:, :9] = 3.0 * np.eye(9)
Gamma_T, _, _, _ = np.linalg.lstsq(SN.T, target.T, rcond=None)
Gamma = Gamma_T.T  # (9, R)

T = build_T()
T_hat = reconstruct(alphas, betas, Gamma)
fitness = np.max(np.abs(T - T_hat))
```

### 6. Packet File Discovery

Don't assume a specific filename. Search for packet files:

```python
import glob
packet_files = (
    glob.glob("./CANON_OPTIMIZER/core/*packet*") +
    glob.glob("./CANON_OPTIMIZER/core/*stage1*") +
    glob.glob("./CANON_DATABASE/*packet*") +
    glob.glob("./CANON_DATABASE/*stage1*") +
    glob.glob("./CANON_DATABASE/*harvest*") +
    glob.glob("./CANON_DATABASE/*basis*")
)
```

Read the Stage 1 source code to find the exact output format and location. If files are numpy archives, use `np.load(f, allow_pickle=True)`. If JSON-lines, read line by line.

---

## WHAT WE'RE LOOKING FOR

### Scenario 1: Universal Obstruction

If sigma_innovation never exceeds 7 across ALL packets for R=19, that's a structural ceiling. It means: no matter which 10-dim subspace V you choose, Sigma always has at least 2 directions trapped inside the nuisance span. This would be a **new theorem**: "For R=19 over {-1,0,1}, the sigma image is algebraically constrained to codimension ≥ 2 in the augmented space."

→ **Action:** Characterize the obstruction. Which 2 sigma directions are always trapped? Are they the same directions across all packets, or do they rotate? If same: that's a fixed subspace. If they rotate: the obstruction is more subtle.

### Scenario 2: Soft Ceiling

If sigma_innovation reaches 8 frequently but never 9, we're "one dimension away." The question becomes: is 8→9 an algebraic wall or a combinatorial rarity?

→ **Action:** Focus on sigma_innovation=8 packets. Compute the single missing sigma direction. Is it the same across packets? What property of (α, β) determines whether this direction is capturable?

### Scenario 3: Heterogeneous Landscape

If sigma_innovation varies widely (5 through 9) and some packets reach 9 but fail for other reasons (Gamma residual, reconstruction error), then there is no universal obstruction — the problem is just hard, and more sampling will eventually find a solution.

→ **Action:** Focus on sigma_innovation=9 packets. Why do they fail? Is Gamma not solvable (rank deficiency in [Σ|N])? Or is Gamma solvable but the reconstruction is inexact (rounding / coefficient field limitation)?

### Scenario 4: R-Dependent Behavior

Compare R=13, R=19, R=20 side by side. If R=20 reaches sigma_innovation=9 easily while R=19 doesn't, that tells you R=19 has a specific obstruction that R=20 avoids (or vice versa).

→ **Action:** Build a comparative table across ranks. Identify which gates are rank-dependent and which are universal.

---

## DON'T FORGET

- **Read the actual packet files first.** Don't assume the format — inspect it.
- **Every diagnostic must handle degenerate cases.** Zero matrices, rank 0, empty arrays.
- **Use float64 for all SVD/rank computations.** Cast from int8 before any linear algebra.
- **Log anomalies.** If a packet has unexpected properties (rank_H > target, negative eigenvalues where PD expected, etc.), log it as an anomaly rather than crashing.
- **The Sigma Innovation histogram is the single most important output.** If you produce nothing else, produce that.
- **Rich console output is required** for the summary tables. The user needs to see results at a glance.
- **Markdown report is required** for archival and sharing with collaborators.
- **CSV export is required** for further statistical analysis in other tools.
