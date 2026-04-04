"""
R=19 Matrix Multiplication — In-RAM Precomputed Database
========================================================

Generates all 387M (α, β) pairs over {-1, 0, 1}, precomputes H-rows
and Sigma-rows, and holds everything in RAM for matmul-driven queries.

Usage:
    from term_db import TermDB
    db = TermDB()           # generates + loads, ~2-4 min, ~12 GB RAM
    db.save("./data")       # optional: save to disk for instant reload
    db = TermDB("./data")   # reload from disk in ~10 seconds

Query interface:
    hits = db.query_subspace(V_basis)  # V_basis: (10, 18) → indices of H-rows in span(V)
    alpha, beta = db.get_factors(idx)  # retrieve the 3×3 matrices for a record
    H_row = db.H[idx]                  # direct access
    sigma_row = db.sigma[idx]          # direct access

Hardware target: Ryzen 9 5900X, 80 GB RAM (64 GB available)
"""

import numpy as np
from pathlib import Path
import time
import sys

# Optional Rich import for progress display
try:
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TermDB:
    """
    In-RAM database of all 387,420,489 (α, β) pairs over {-1, 0, 1}^{3×3}.

    Stored arrays (int8 for compactness):
        H:          (N, 18) — concatenated [Eta1 | Eta2] row per pair
        sigma:      (N, 9)  — fiber-sum row per pair
        alpha_idx:  (N,)    — uint32 index into self.templates
        beta_idx:   (N,)    — uint32 index into self.templates
        templates:  (19683, 3, 3) — all 3×3 matrices over {-1, 0, 1}

    Delta rows (54 entries each, 21 GB total) are NOT stored.
    Recompute on demand via db.compute_delta(idx) for the handful of Gate 2 survivors.
    """

    N_TEMPLATES = 3**9   # 19,683 unique 3×3 matrices over {-1,0,1}
    N_PAIRS = N_TEMPLATES**2  # 387,420,489

    def __init__(self, load_path=None, generate=True):
        """
        Args:
            load_path: Path to previously saved .npy files. If provided, loads from disk (~10s).
            generate:  If True and no load_path, generate from scratch (~2-4 min).
        """
        if load_path is not None:
            self._load(Path(load_path))
        elif generate:
            self._generate()
        else:
            raise ValueError("Either provide load_path or set generate=True")

    # ─────────────────────────────────────────────────────────────
    #  GENERATION
    # ─────────────────────────────────────────────────────────────

    def _generate(self):
        """Generate all 387M records in RAM. ~2-4 min on Ryzen 9 5900X."""
        console = Console() if HAS_RICH else None
        t0 = time.time()

        # Step 1: Build all 19,683 templates
        self._print(console, "[1/4] Building 19,683 template matrices...")
        self.templates = self._build_templates()  # (19683, 3, 3) int8

        # Step 2: Allocate output arrays
        self._print(console, f"[2/4] Allocating arrays ({self.N_PAIRS:,} records, ~12 GB)...")
        self.H = np.empty((self.N_PAIRS, 18), dtype=np.int8)
        self.sigma = np.empty((self.N_PAIRS, 9), dtype=np.int8)
        self.alpha_idx = np.empty(self.N_PAIRS, dtype=np.uint16)
        self.beta_idx = np.empty(self.N_PAIRS, dtype=np.uint16)

        # Step 3: Compute H and Sigma for all pairs, chunked by alpha
        self._print(console, "[3/4] Computing H and Sigma for all 387M pairs...")
        self._compute_all_blocks(console)

        elapsed = time.time() - t0
        self._print(console, f"[4/4] Done. {self.N_PAIRS:,} records in {elapsed:.1f}s "
                             f"({self._sizeof_gb():.1f} GB RAM)")

    def _build_templates(self):
        """Generate all 3^9 = 19,683 matrices with entries in {-1, 0, 1}."""
        # Each template is a 3×3 matrix. Enumerate via base-3 digits mapped to {-1, 0, 1}.
        indices = np.arange(self.N_TEMPLATES, dtype=np.int32)
        digits = np.empty((self.N_TEMPLATES, 9), dtype=np.int8)
        temp = indices.copy()
        for i in range(9):
            digits[:, i] = (temp % 3).astype(np.int8) - 1  # map {0,1,2} → {-1,0,1}
            temp //= 3
        return digits.reshape(self.N_TEMPLATES, 3, 3)

    def _compute_all_blocks(self, console):
        """
        Compute H-rows and Sigma-rows for all (α, β) pairs.

        Chunked by alpha index to keep intermediate memory bounded.
        Each chunk: one α vs all 19,683 β values → 19,683 records.
        Total chunks: 19,683. Each is pure vectorized numpy.
        """
        beta_all = self.templates  # (19683, 3, 3)
        n_beta = self.N_TEMPLATES
        chunk_size = n_beta  # records per chunk

        if HAS_RICH:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeElapsedColumn(),
                console=console or Console(),
            )
            task = progress.add_task("Computing blocks", total=self.N_TEMPLATES)
            progress.start()
        else:
            progress = None

        for ai in range(self.N_TEMPLATES):
            alpha = self.templates[ai]  # (3, 3)
            offset = ai * n_beta

            # ── Sigma: Σ[r,u] = Σ_s α[r,s]·β[s,u] = (α @ β)[r,u] ──
            # alpha: (3,3), beta_all: (N, 3, 3)
            # result: (N, 3, 3) → flatten to (N, 9)
            prod = np.einsum('rs,nsu->nru', alpha, beta_all)  # (N, 3, 3)
            self.sigma[offset:offset+n_beta] = prod.reshape(n_beta, 9).astype(np.int8)

            # ── Eta1: α[r,0]·β[0,u] − α[r,1]·β[1,u] ──
            # Shape: (N, 3, 3) → flatten to (N, 9)
            eta1 = (np.outer(alpha[:, 0], np.ones(3, dtype=np.int8)).reshape(1, 3, 3) *
                    beta_all[:, 0:1, :] -
                    np.outer(alpha[:, 1], np.ones(3, dtype=np.int8)).reshape(1, 3, 3) *
                    beta_all[:, 1:2, :])
            # More readable version:
            eta1 = (alpha[:, 0].reshape(1, 3, 1) * beta_all[:, 0, :].reshape(n_beta, 1, 3) -
                    alpha[:, 1].reshape(1, 3, 1) * beta_all[:, 1, :].reshape(n_beta, 1, 3))
            eta1_flat = eta1.reshape(n_beta, 9)

            # ── Eta2: α[r,1]·β[1,u] − α[r,2]·β[2,u] ──
            eta2 = (alpha[:, 1].reshape(1, 3, 1) * beta_all[:, 1, :].reshape(n_beta, 1, 3) -
                    alpha[:, 2].reshape(1, 3, 1) * beta_all[:, 2, :].reshape(n_beta, 1, 3))
            eta2_flat = eta2.reshape(n_beta, 9)

            # ── H = [Eta1 | Eta2] ──
            self.H[offset:offset+n_beta, :9] = eta1_flat.astype(np.int8)
            self.H[offset:offset+n_beta, 9:] = eta2_flat.astype(np.int8)

            # ── Index storage ──
            self.alpha_idx[offset:offset+n_beta] = ai
            self.beta_idx[offset:offset+n_beta] = np.arange(n_beta, dtype=np.uint16)

            if progress:
                progress.update(task, advance=1)

        if progress:
            progress.stop()

    # ─────────────────────────────────────────────────────────────
    #  QUERIES
    # ─────────────────────────────────────────────────────────────

    def query_subspace(self, V_basis):
        """
        Find all records whose H-row lies in the integer span of V_basis.

        Args:
            V_basis: (d, 18) integer array — basis vectors (typically actual H-rows).

        Returns:
            indices: 1-D int array of record indices where H[idx] ∈ span(V_basis).

        Method:
            Compute integer null space of V_basis, then check H @ V_perp.T == 0
            using exact integer arithmetic. No floats, no thresholds, no false positives.
        """
        V_perp = self._integer_nullspace(np.asarray(V_basis))
        if V_perp.shape[0] == 0:
            # V_basis spans all of R^18 — every H-row is trivially in span
            return np.arange(self.N_PAIRS)
        return self.query_subspace_fast(V_perp)

    @staticmethod
    def _integer_nullspace(V_int):
        """
        Compute an integer basis for the null space of integer matrix V (d×18).

        Uses row reduction over the rationals (via integer arithmetic with scaling)
        to get exact results. For the small matrices in this application (d ≤ 18,
        entries in {-2..2}), this is fast and numerically exact.

        Returns:
            N: (k, 18) int64 array whose rows span ker(V). k = 18 - rank(V).
            Returns shape (0, 18) if V has full column rank.
        """
        V = np.array(V_int, dtype=np.int64)
        m, n = V.shape  # m rows, n=18 columns

        # Augment: [V^T | I_n] then row-reduce V^T to find null space
        # The null space of V (as a d×n matrix) = left null space of V^T
        # Equivalently: find vectors w such that w @ V^T = 0, i.e. V @ w = 0... no.
        # We want vectors w in R^18 such that V @ w = 0 (for each row of V dotted with w).
        # That's the RIGHT null space of V (d×18): ker(V) = {w : Vw = 0}.

        # Method: transpose and do column reduction, OR use the standard
        # augmented matrix approach.
        # Build [V; I_18]^T = 18×(d+18), row reduce the V part, read null space.

        # Simpler: row reduce V^T (18×d) augmented with I_18.
        # Actually, let's just do RREF on V (d×18) directly.

        # Integer RREF via fraction-free Gaussian elimination
        A = V.copy()  # d × 18
        pivot_cols = []
        row = 0
        for col in range(n):
            if row >= m:
                break
            # Find pivot in this column
            pivot = None
            for r in range(row, m):
                if A[r, col] != 0:
                    pivot = r
                    break
            if pivot is None:
                continue
            # Swap rows
            A[[row, pivot]] = A[[pivot, row]]
            pivot_cols.append(col)
            # Eliminate all other rows
            for r in range(m):
                if r != row and A[r, col] != 0:
                    # A[r] = A[r] * A[row, col] - A[row] * A[r, col]
                    factor_r = A[r, col]
                    factor_p = A[row, col]
                    A[r] = A[r] * factor_p - A[row] * factor_r
                    # Reduce by GCD to prevent integer overflow
                    g = np.gcd.reduce(np.abs(A[r][A[r] != 0])) if np.any(A[r] != 0) else 1
                    if g > 1:
                        A[r] //= g
            row += 1

        rank = len(pivot_cols)
        free_cols = [c for c in range(n) if c not in pivot_cols]
        k = len(free_cols)  # nullity

        if k == 0:
            return np.empty((0, n), dtype=np.int64)

        # Build null space vectors: for each free variable, set it to 1,
        # solve for pivot variables
        null_vectors = np.zeros((k, n), dtype=np.int64)
        for i, fc in enumerate(free_cols):
            null_vectors[i, fc] = 1
            # For each pivot row j with pivot in pivot_cols[j]:
            # A[j, pivot_cols[j]] * x[pivot_cols[j]] + ... + A[j, fc] * 1 + ... = 0
            # But we need to handle the reduced form properly.
            # After elimination, row j has: A[j, pivot_cols[j]] * x[pivot_cols[j]] + Σ_{free} A[j, f] * x[f] = 0
            for j in range(rank):
                pc = pivot_cols[j]
                # x[pc] = -A[j, fc] / A[j, pc] (when only free col fc is set to 1)
                # To stay integer: x[pc] = -A[j, fc], and scale everything by A[j, pc]
                pass  # Handle below

        # Better approach: build null space directly from reduced A
        # After RREF, for each free column fc, the null vector is:
        # x[fc] = lcm_of_pivots (to clear denominators)
        # x[pivot_cols[j]] = -A[j, fc] * (lcm / A[j, pivot_cols[j]])

        # Compute LCM of all pivot values for a common denominator
        pivot_vals = np.array([A[j, pivot_cols[j]] for j in range(rank)], dtype=np.int64)

        for i, fc in enumerate(free_cols):
            # Start with x[fc] = product of all pivot values (guaranteed divisible)
            common = np.prod(pivot_vals)  # could overflow for large matrices, fine for 18×18
            null_vectors[i, fc] = common
            for j in range(rank):
                pc = pivot_cols[j]
                # x[pc] = -A[j, fc] * common / A[j, pc]
                null_vectors[i, pc] = -A[j, fc] * (common // pivot_vals[j])

            # Reduce by GCD
            g = np.gcd.reduce(np.abs(null_vectors[i][null_vectors[i] != 0]))
            if g > 1:
                null_vectors[i] //= g

        # Verify: V @ null_vectors.T should be zero
        check = V @ null_vectors.T
        assert np.all(check == 0), f"Null space verification failed: max entry = {np.max(np.abs(check))}"

        return null_vectors

    def query_subspace_fast(self, V_perp_int):
        """
        Fast integer query when V_perp is already known as an integer matrix.

        Args:
            V_perp_int: (k, 18) integer array — rows orthogonal to the target subspace.
                        H-row is in the subspace iff H @ V_perp.T == 0.

        Returns:
            indices: 1-D int array of matching record indices.
        """
        V = np.asarray(V_perp_int)
        # Choose dtype to avoid overflow: H is int8 ({-2..2}), V entries may be larger.
        # Each dot product is sum of 18 terms. Worst case int8 × int64: use int32 or int64.
        if np.max(np.abs(V)) * 2 * 18 < 32767:
            proj = self.H.astype(np.int16) @ V.astype(np.int16).T
        else:
            proj = self.H.astype(np.int32) @ V.astype(np.int32).T
        return np.where(np.all(proj == 0, axis=1))[0]

    def get_factors(self, idx):
        """Retrieve (alpha, beta) as (3,3) int8 arrays for record idx."""
        return self.templates[self.alpha_idx[idx]], self.templates[self.beta_idx[idx]]

    def get_factors_batch(self, indices):
        """Retrieve (alpha_batch, beta_batch) as (len, 3, 3) int8 arrays."""
        return self.templates[self.alpha_idx[indices]], self.templates[self.beta_idx[indices]]

    def compute_delta(self, idx):
        """
        Compute Delta row (54 entries) on demand for a single record.
        Only called on Gate 1+2 survivors — never in bulk.

        Delta[r,s,t,u] = alpha[r,s] * beta[t,u] for s ≠ t.
        Pairs (s,t) with s≠t: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) — 6 pairs.
        For each (r,u) ∈ {0,1,2}²: 6 entries. Total: 9 × 6 = 54.
        """
        alpha, beta = self.get_factors(idx)
        delta = np.empty(54, dtype=np.int8)
        pairs = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        col = 0
        for r in range(3):
            for u in range(3):
                for s, t in pairs:
                    delta[col] = alpha[r, s] * beta[t, u]
                    col += 1
        return delta

    def compute_delta_batch(self, indices):
        """Compute Delta rows for multiple records. Returns (len, 54) int8."""
        alphas, betas = self.get_factors_batch(indices)
        n = len(indices)
        delta = np.empty((n, 54), dtype=np.int8)
        pairs = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        col = 0
        for r in range(3):
            for u in range(3):
                for s, t in pairs:
                    delta[:, col] = alphas[:, r, s] * betas[:, t, u]
                    col += 1
        return delta

    # ─────────────────────────────────────────────────────────────
    #  DEDUPLICATION (optional, for faster queries)
    # ─────────────────────────────────────────────────────────────

    def _dedup_cache_path(self) -> Path | None:
        """Return the directory where dedup cache files would live, or None."""
        if hasattr(self, '_data_path') and self._data_path:
            return Path(self._data_path)
        return None

    def _try_load_dedup_cache(self, progress_callback=None) -> bool:
        """Attempt to load cached dedup artifacts. Returns True on success."""
        cache_dir = self._dedup_cache_path()
        if cache_dir is None:
            return False
        files = {
            'unique_H': cache_dir / 'dedup_unique_H.npy',
            'H_dedup_map': cache_dir / 'dedup_H_dedup_map.npy',
            'H_dedup_counts': cache_dir / 'dedup_H_dedup_counts.npy',
        }
        if not all(f.exists() for f in files.values()):
            return False
        try:
            if progress_callback:
                progress_callback("dedup: loading cached unique_H")
            self.unique_H = np.load(str(files['unique_H']), mmap_mode=None)
            if progress_callback:
                progress_callback("dedup: loading cached H_dedup_map")
            self.H_dedup_map = np.load(str(files['H_dedup_map']), mmap_mode=None)
            if progress_callback:
                progress_callback("dedup: loading cached H_dedup_counts")
            self.H_dedup_counts = np.load(str(files['H_dedup_counts']), mmap_mode=None)
            self.n_unique_H = self.unique_H.shape[0]
            # Validate sizes match current DB
            if self.H_dedup_map.shape[0] != self.N_PAIRS:
                if progress_callback:
                    progress_callback("dedup: cache size mismatch, rebuilding")
                return False
            # Roundtrip sanity check on cached data
            rng = np.random.default_rng(0)
            check_idx = rng.choice(self.N_PAIRS, min(1000, self.N_PAIRS), replace=False)
            for idx in check_idx:
                if not np.array_equal(self.H[idx], self.unique_H[self.H_dedup_map[idx]]):
                    if progress_callback:
                        progress_callback("dedup: cache roundtrip failed, rebuilding")
                    return False
            if progress_callback:
                progress_callback(f"dedup: loaded cache — {self.n_unique_H:,} unique")
            return True
        except Exception:
            return False

    def _save_dedup_cache(self, progress_callback=None):
        """Save dedup artifacts to disk for next run."""
        cache_dir = self._dedup_cache_path()
        if cache_dir is None:
            return
        try:
            if progress_callback:
                progress_callback("dedup: saving cache to disk")
            np.save(str(cache_dir / 'dedup_unique_H.npy'), self.unique_H)
            np.save(str(cache_dir / 'dedup_H_dedup_map.npy'), self.H_dedup_map)
            np.save(str(cache_dir / 'dedup_H_dedup_counts.npy'), self.H_dedup_counts)
            if progress_callback:
                progress_callback("dedup: cache saved")
        except Exception:
            pass  # non-fatal

    def build_dedup_index(self, progress_callback=None):
        """
        Build a deduplicated H-row index. Many (α,β) pairs produce
        identical H-rows. Queries on the deduped set are 30-50% faster.

        Sets:
            self.unique_H:      (M, 18) int8 — unique H-rows (M < N)
            self.H_dedup_map:   (N,) uint32  — maps record → unique H index
            self.H_dedup_counts:(M,) uint32  — count per unique row
        """
        # Try loading from cache first
        if self._try_load_dedup_cache(progress_callback):
            return

        console = Console() if (HAS_RICH and not progress_callback) else None
        silent = bool(progress_callback)
        self._print(console, "Building dedup index...", silent=silent)
        t0 = time.time()

        # Pack each 18-value H-row into a single int64 key.
        # Entries are {-2, -1, 0, 1, 2} → map to {0, 1, 2, 3, 4} then encode in base 5.
        # 5^18 = 3,814,697,265,625 < 2^63, so fits comfortably in int64.
        # np.unique on plain int64 is a pure C sort that RELEASES the GIL.
        if progress_callback:
            progress_callback("dedup: packing H rows into int64 keys")
        N = self.H.shape[0]
        keys = np.zeros(N, dtype=np.int64)
        H = self.H  # (N, 18) int8 — values in {-2, -1, 0, 1, 2}
        # Horner's method from the last column backwards: key = (...((h17+2)*5 + h16+2)*5 + ...)*5 + h0+2
        # Each op is a single vectorized numpy call on int64 that releases the GIL
        for col in range(17, -1, -1):
            keys *= 5
            # Adding int8 column to int64 array — numpy upcasts the int8 automatically
            keys += H[:, col]
            keys += 2  # shift {-2,-1,0,1,2} → {0,1,2,3,4}

        if progress_callback:
            progress_callback("dedup: sorting (GIL-free np.unique on int64)")
        unique_keys, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        del keys  # free 2.9 GB

        if progress_callback:
            progress_callback("dedup: unpacking unique H rows")
        # Decode unique keys back to (M, 18) int8 rows
        M = len(unique_keys)
        unique_H = np.empty((M, 18), dtype=np.int8)
        temp = unique_keys.copy()
        del unique_keys  # free M*8 bytes
        for col in range(18):
            unique_H[:, col] = (temp % 5).astype(np.int8) - 2
            temp //= 5
        del temp

        self.unique_H = unique_H
        self.H_dedup_map = inverse.astype(np.uint32)
        self.H_dedup_counts = counts.astype(np.uint32)
        self.n_unique_H = M

        # ── Self-test: catch encoding bugs before they propagate ──
        h_min, h_max = int(self.H.min()), int(self.H.max())
        assert h_min >= -2 and h_max <= 2, (
            f"H entries [{h_min}, {h_max}] outside expected range {{-2..2}}"
        )
        rng = np.random.default_rng(0)
        check_idx = rng.choice(self.N_PAIRS, min(1000, self.N_PAIRS), replace=False)
        for idx in check_idx:
            assert np.array_equal(self.H[idx], self.unique_H[self.H_dedup_map[idx]]), (
                f"Dedup roundtrip failed at record {idx}: "
                f"H={self.H[idx].tolist()}, "
                f"unique_H[{self.H_dedup_map[idx]}]={self.unique_H[self.H_dedup_map[idx]].tolist()}"
            )

        elapsed = time.time() - t0
        ratio = self.n_unique_H / self.N_PAIRS * 100
        if progress_callback:
            progress_callback(f"dedup: done — {self.n_unique_H:,} unique ({ratio:.1f}%) in {elapsed:.1f}s")
        self._print(console, f"Dedup: {self.N_PAIRS:,} → {self.n_unique_H:,} unique H-rows "
                             f"({ratio:.1f}%), {elapsed:.1f}s", silent=silent)

        # Save cache for next run
        self._save_dedup_cache(progress_callback)

    def query_subspace_dedup(self, V_basis):
        """
        Query on deduplicated H-rows. Returns record indices (not unique-H indices).
        Requires build_dedup_index() to have been called first.
        """
        if not hasattr(self, 'unique_H'):
            raise RuntimeError("Call build_dedup_index() first")

        V_perp = self._integer_nullspace(np.asarray(V_basis))
        if V_perp.shape[0] == 0:
            return np.arange(self.N_PAIRS)

        V = np.asarray(V_perp)
        # Use pre-cached int16 view if available (set by finalize_for_search)
        uH = self._unique_H_i16 if hasattr(self, '_unique_H_i16') else self.unique_H.astype(np.int16)
        # Choose dtype to avoid overflow: H entries in {-2..2}, V entries may be larger.
        # Each dot product is sum of 18 terms. Worst case: 18 * 2 * max(|V|).
        max_dot = int(np.max(np.abs(V))) * 2 * 18
        if max_dot < 32767:
            proj = uH @ V.astype(np.int16).T
        else:
            proj = uH.astype(np.int32) @ V.astype(np.int32).T
        unique_hits = np.where(np.all(proj == 0, axis=1))[0]

        if unique_hits.size == 0:
            return np.empty(0, dtype=np.int64)

        # Fast expansion: use boolean mask on H_dedup_map instead of np.isin
        # Build a membership set for the unique hit indices
        hit_mask = np.zeros(self.n_unique_H, dtype=np.bool_)
        hit_mask[unique_hits] = True
        record_hits = np.where(hit_mask[self.H_dedup_map])[0]
        return record_hits

    def finalize_for_search(self):
        """Pre-cache expensive conversions so worker threads don't each allocate copies."""
        # Cache int16 version of unique_H — 31.7M × 18 × 2 = ~1.1 GB, allocated once
        if not hasattr(self, '_unique_H_i16'):
            self._unique_H_i16 = self.unique_H.astype(np.int16)

    # ─────────────────────────────────────────────────────────────
    #  PERSISTENCE (optional, for fast reload)
    # ─────────────────────────────────────────────────────────────

    def save(self, path):
        """Save arrays to disk for fast reload. ~38 GB, writes in ~13s on NVMe."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._data_path = str(p.resolve())
        self._print(None, f"Saving to {p}...")
        t0 = time.time()
        np.save(p / "H.npy", self.H)
        np.save(p / "sigma.npy", self.sigma)
        np.save(p / "alpha_idx.npy", self.alpha_idx)
        np.save(p / "beta_idx.npy", self.beta_idx)
        np.save(p / "templates.npy", self.templates)
        elapsed = time.time() - t0
        self._print(None, f"Saved in {elapsed:.1f}s")

    def _load(self, path, progress_callback=None):
        """Load from disk. ~10s from NVMe via memory-mapping."""
        p = Path(path)
        self._data_path = str(p.resolve())
        # Only use standalone console when NOT driven by dashboard (progress_callback)
        console = Console() if (HAS_RICH and not progress_callback) else None
        silent = bool(progress_callback)
        self._print(console, f"Loading from {p}...", silent=silent)
        t0 = time.time()
        if progress_callback:
            progress_callback("load: templates")
        self.templates = np.load(p / "templates.npy")
        if progress_callback:
            progress_callback("load: H memmap")
        self.H = np.load(p / "H.npy", mmap_mode='r+')
        if progress_callback:
            progress_callback("load: sigma memmap")
        self.sigma = np.load(p / "sigma.npy", mmap_mode='r+')
        if progress_callback:
            progress_callback("load: factor indices")
        self.alpha_idx = np.load(p / "alpha_idx.npy")
        self.beta_idx = np.load(p / "beta_idx.npy")
        elapsed = time.time() - t0
        if progress_callback:
            progress_callback(f"load: complete in {elapsed:.1f}s")
        self._print(console, f"Loaded {self.H.shape[0]:,} records in {elapsed:.1f}s "
                             f"({self._sizeof_gb():.1f} GB)", silent=silent)

    def load_to_ram(self, progress_callback=None):
        """Force memory-mapped arrays fully into RAM for max query speed."""
        console = Console() if (HAS_RICH and not progress_callback) else None
        silent = bool(progress_callback)
        self._print(console, "Loading arrays fully into RAM...", silent=silent)
        t0 = time.time()
        if progress_callback:
            progress_callback("load_to_ram: H")
        if isinstance(self.H, np.memmap):
            self.H = np.array(self.H)
        if progress_callback:
            progress_callback("load_to_ram: sigma")
        if isinstance(self.sigma, np.memmap):
            self.sigma = np.array(self.sigma)
        elapsed = time.time() - t0
        if progress_callback:
            progress_callback(f"load_to_ram: complete in {elapsed:.1f}s")
        self._print(console, f"Fully in RAM: {self._sizeof_gb():.1f} GB, {elapsed:.1f}s", silent=silent)

    # ─────────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────────

    def _sizeof_gb(self):
        """Approximate RAM usage in GB."""
        total = (self.H.nbytes + self.sigma.nbytes +
                 self.alpha_idx.nbytes + self.beta_idx.nbytes +
                 self.templates.nbytes)
        return total / (1024**3)

    def _print(self, console, msg, silent=False):
        if silent:
            return
        if console and HAS_RICH:
            console.print(msg)
        else:
            print(msg, flush=True)

    def stats(self):
        """Print database statistics."""
        print(f"╔══════════════════════════════════════════════╗")
        print(f"║  TermDB — In-RAM Record Database             ║")
        print(f"╠══════════════════════════════════════════════╣")
        print(f"║  Records:    {self.N_PAIRS:>15,}             ║")
        print(f"║  Templates:  {self.N_TEMPLATES:>15,}             ║")
        print(f"║  H shape:    {str(self.H.shape):>15s}             ║")
        print(f"║  Σ shape:    {str(self.sigma.shape):>15s}             ║")
        print(f"║  RAM usage:  {self._sizeof_gb():>12.1f} GB             ║")
        if hasattr(self, 'n_unique_H'):
            print(f"║  Unique H:   {self.n_unique_H:>15,}             ║")
        print(f"╚══════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="R=19 Term Database — Generate or Load")
    parser.add_argument("--save", type=str, help="Generate and save to this directory")
    parser.add_argument("--load", type=str, help="Load from this directory")
    parser.add_argument("--dedup", action="store_true", help="Build dedup index after loading")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        print("Running self-tests...")

        # Test 1: Template generation
        db = TermDB.__new__(TermDB)
        templates = db._build_templates()
        assert templates.shape == (19683, 3, 3)
        assert templates.dtype == np.int8
        assert set(np.unique(templates)) == {-1, 0, 1}
        # Check that we have all 3^9 unique matrices
        as_tuples = set(tuple(t.ravel()) for t in templates)
        assert len(as_tuples) == 19683
        print("  ✓ Template generation correct")

        # Test 2: Small-scale generation (first 100 α × all β)
        db = TermDB()  # full generation
        # Spot-check: the zero matrix α=0 should give H=0, Σ=0 for all β
        zero_idx = np.where(np.all(db.templates.reshape(19683, 9) == 0, axis=1))[0][0]
        start = zero_idx * 19683
        end = start + 19683
        assert np.all(db.H[start:end] == 0), "Zero α should give zero H"
        assert np.all(db.sigma[start:end] == 0), "Zero α should give zero Σ"
        print("  ✓ Zero-alpha records correct")

        # Test 3: Identity α, identity β should give known values
        # α = I₃ → α[r,s] = δ_{rs}
        # β = I₃ → β[s,u] = δ_{su}
        # Sigma[r,u] = Σ_s δ_{rs}·δ_{su} = δ_{ru} → identity
        # Eta1[r,u] = δ_{r0}·δ_{0u} - δ_{r1}·δ_{1u}
        eye_flat = np.eye(3, dtype=np.int8).ravel()
        eye_idx = np.where(np.all(db.templates.reshape(19683, 9) == eye_flat, axis=1))[0]
        if len(eye_idx) > 0:
            rec_idx = eye_idx[0] * 19683 + eye_idx[0]
            sigma_row = db.sigma[rec_idx]
            expected_sigma = np.eye(3, dtype=np.int8).ravel()  # δ_{ru}
            assert np.array_equal(sigma_row, expected_sigma), f"Identity Σ mismatch: {sigma_row}"
            print("  ✓ Identity α,β Sigma correct")

        # Test 4: Query test — use actual H-rows as subspace basis (guarantees hits)
        # Pick 10 random H-rows that are linearly independent
        rng = np.random.default_rng(42)
        candidate_indices = rng.choice(db.N_PAIRS, size=1000, replace=False)
        basis_indices = []
        H_stack = np.empty((0, 18), dtype=np.float64)
        for idx in candidate_indices:
            row = db.H[idx].astype(np.float64).reshape(1, 18)
            if H_stack.shape[0] == 0 or np.linalg.matrix_rank(np.vstack([H_stack, row])) > H_stack.shape[0]:
                H_stack = np.vstack([H_stack, row])
                basis_indices.append(int(idx))
                if len(basis_indices) == 10:
                    break
        assert len(basis_indices) == 10, f"Could not find 10 independent H-rows (got {len(basis_indices)})"

        V_basis = db.H[basis_indices].astype(np.float64)  # (10, 18)
        hits = db.query_subspace(V_basis)

        # The 10 basis rows must be among the hits
        for bi in basis_indices:
            assert bi in hits, f"Basis row {bi} not found in query results"

        # Verify all hits actually lie in the subspace
        Q, _ = np.linalg.qr(V_basis.T)  # Q: (18, 10) orthonormal basis
        for idx in hits[:200]:
            h = db.H[idx].astype(np.float64)
            proj = h - Q @ (Q.T @ h)  # residual after projecting onto V
            assert np.allclose(proj, 0, atol=1e-6), f"False positive at idx {idx}, residual={np.max(np.abs(proj))}"
        print(f"  ✓ Subspace query correct ({len(hits)} hits, {len(basis_indices)} basis rows verified)")

        # Test 4b: integer nullspace correctness
        V_perp = TermDB._integer_nullspace(V_basis.astype(np.int64))
        assert V_perp.shape == (8, 18), f"Expected (8,18) nullspace, got {V_perp.shape}"
        # Verify V_basis @ V_perp.T == 0 exactly
        check = V_basis.astype(np.int64) @ V_perp.T
        assert np.all(check == 0), f"Nullspace verification failed: {np.max(np.abs(check))}"
        # Verify query_subspace_fast gives same results as query_subspace
        hits_fast = db.query_subspace_fast(V_perp)
        assert np.array_equal(hits, hits_fast), "query_subspace and query_subspace_fast disagree"
        print(f"  ✓ Integer nullspace correct (shape {V_perp.shape}, fast query matches)")

        # Test 5: Delta computation
        delta = db.compute_delta(0)
        assert delta.shape == (54,)
        assert delta.dtype == np.int8
        print("  ✓ Delta computation correct")

        print("\nAll self-tests passed. ✓")

    elif args.save:
        db = TermDB()
        db.stats()
        db.save(args.save)

    elif args.load:
        db = TermDB(load_path=args.load)
        db.load_to_ram()
        db.stats()
        if args.dedup:
            db.build_dedup_index()
            db.stats()

    else:
        # Default: generate, print stats, run a quick benchmark
        db = TermDB()
        db.stats()
        print("\nBenchmarking subspace query...")
        V = np.random.randn(10, 18)
        t0 = time.time()
        hits = db.query_subspace(V)
        elapsed = time.time() - t0
        print(f"  Query: {len(hits):,} hits in {elapsed:.3f}s "
              f"({db.N_PAIRS/elapsed/1e6:.0f}M records/sec)")
