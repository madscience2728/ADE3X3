
# ADE & ADEV2: Algebra Discovery Engine — Technical Overview

## 1. What is ADE?

**ADE (Algebra Discovery Engine)** is a Python framework for constructing, analyzing, and searching for finite-dimensional algebras. It provides:

- A core algebra object (`AlgebraState`) representing the structure tensor, grading, and invariants.
- Constraint and invariant checkers for algebraic properties (associativity, commutativity, Jacobi, nilpotency, etc).
- Reference implementations for known algebras (Lie, associative, division, nilpotent, exceptional, etc).
- Baseline verification routines to check invariants for standard algebras.

### 1.1. Core Object: `AlgebraState`
Located in `ade/core/algebra_state.py`, this class encapsulates:
- `n`: dimension
- `base_ring`: typically $\mathbb{R}$ or $\mathbb{C}$
- `name`: label
- `_C_sparse`: sparse structure tensor (dict of $(i,j,k) \to v$)
- `grading`, `Q`, `J`: optional gradings and quadratic forms
- `flags`: dict of algebraic properties (ASSOCIATIVE, COMMUTATIVE, etc)

Key methods:
- `C()`: returns dense $(n,n,n)$ structure tensor
- `set_C(i, j, k, v)`: set structure constant
- `to_dict()` / `from_dict()`: serialization
- `content_hash()`: unique hash for algebra content

### 1.2. Constraint & Invariant Checks
Located in `ade/constraints/checks.py` and `ade/invariants/tier1.py`:
- **Constraint checks**: Functions like `check_associativity`, `check_commutativity`, `check_jacobi`, etc, return diagnostic dicts with residuals and pass/fail.
- **Tier 1 invariants**: Fast $O(n^3)$ computations:
	- Center dimension and basis
	- Killing form, rank, and signature
	- Nilpotency class (lower central series)
	- Associator/defect norm
	- Existence of unit element

### 1.3. Known Algebra Library
Located in `ade/known/algebras.py`, provides constructors for:
- $sl(2,\mathbb{R})$, $so(3,\mathbb{R})$, Heisenberg, Quaternions, Octonions, Clifford, etc.
- Each returns an `AlgebraState` with reference structure constants.

### 1.4. Baseline Verification
`ade/baseline.py` verifies that known algebras reproduce expected invariants and properties.

---

## 2. What is ADEV2?

**ADEV2** is an experimental extension of ADE for exploring "fringe" algebra families and searching for new, exotic algebraic structures. It is intentionally separated from the ADE core for rapid iteration.

### 2.1. Fringe Algebra Families
Located in `adev2/families.py`, implements generators for:
- **Parity-shadow algebra**: Products with parity tags that flip under certain interactions.
- **Nilpotent witness slots**: Adjoin nilpotent variables to track path provenance.
- **Coefficient-splitting algebra**: Asymmetric multiplication encoding two matrix products at once.
- **One-sided zero divisors**: Elements with $ab=0$ but $ba\neq 0$ for routing.
- **Twisted group algebras**: $(\mathbb{Z}/2\mathbb{Z})^n$ with sign cocycles.
- **Signed path algebras**: Path-based algebras with cancellation reservoirs.
- **Radical extensions**: Square-zero radicals attached to twisted-group bases.
- **Random relation algebras**: Sparse, randomly generated short-relation algebras.

Each family is implemented as a function returning an `AlgebraState`, with parameters controlling dimension, width, twist, etc. See `adev2/families.py` for details.

### 2.2. Candidate Specification
`adev2/families.py` defines `CandidateSpec`, a dataclass describing a candidate algebra family and parameters. Specs are used to generate all candidates for a scan.

---

## 3. ADEV2 Scanning Engine

Located in `adev2/engine.py`, the scanning engine:

- **Generates** candidate specs for all parameter combinations (see `build_candidate_specs_slab`).
- **Evaluates** each candidate:
	- Builds the algebra (`build_candidate`)
	- Runs all constraint checks (`verify_all`)
	- Computes tier 1 invariants (`tier1`)
	- Computes fringe metrics (annihilator counts, zero divisors, square-zero fraction, etc)
	- Optionally runs CP-rank search (tensor complexity)
- **Persists** results to SQLite and checkpoint files for resumability (`adev2/persistence.py`).
- **Supports staged pipelines**:
	- **Stage 1**: Broad sweep (fast, wide, shallow)
	- **Stage 2**: Promoted sweep (deep, focused, e.g. with CP-rank)

### 3.1. Evaluation Details
For each candidate, the engine records:
- Candidate ID, content hash, family, parameters, idea references
- Structure tensor stats: dimension, nonzero count, tensor norm
- Tier 1 invariants: center, Killing form, nilpotency, associator norm, unit
- Flags: associativity, commutativity, Jacobi, alternative, flexible, etc
- Fringe metrics: annihilator counts, one/two-sided zero divisors, square-zero fraction, fringe score
- CP-rank diagnostics (if enabled)

### 3.2. Parallelization & Memory
- Uses process pools for parallel candidate evaluation
- Dynamically tunes slab size to fit RAM targets
- Checkpointing and resumability for long scans

### 3.3. Promotion & Staging
- After stage 1, top outliers (by fringe score, nilpotency, etc) are promoted for deeper analysis in stage 2
- Results are sorted and deduplicated by signature

---

## 4. Persistence & Checkpointing

Located in `adev2/persistence.py`:
- Results are written to SQLite for resumability
- Checkpoint chunk files (JSONL) for power-loss/mid-run recovery
- Manifest files track stage progress
- All results are indexed by run ID, stage, candidate ID, and ordinal

---

## 5. Example Workflow

1. **Define candidate specs** (see `adev2/families.py`)
2. **Run scan** using the ADEV2 engine (`adev2/engine.py`)
3. **Analyze results**: invariants, flags, fringe metrics, CP-rank
4. **Promote interesting candidates** for deeper analysis

---

## 6. References & Project Structure

- `100 ideas.md`: Source of many fringe algebra family ideas
- `ade/`: Core engine, algebra objects, invariants, constraints, known algebras
- `adev2/`: Experimental families, scan engine, persistence
- `ADEV2_CHECKLIST.md`: Project checklist and run notes

---

*This document is auto-generated. For implementation details, see the code in `ade/` and `adev2/`.*
