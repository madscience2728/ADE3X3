"""Blob encoding/decoding for factor matrices, plus support hashing."""

import hashlib
from itertools import permutations

import numpy as np

from .config import RANK, DIM, N_PARAMS, BLOB_BYTES

_S3_ELEMENTS = list(permutations(range(3)))


def factors_to_blob(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> bytes:
    """Encode three (RANK, DIM) factor matrices into a single 4104-byte blob."""
    assert alpha.shape == (RANK, DIM), f"alpha shape {alpha.shape} != ({RANK}, {DIM})"
    assert beta.shape == (RANK, DIM), f"beta shape {beta.shape} != ({RANK}, {DIM})"
    assert gamma.shape == (RANK, DIM), f"gamma shape {gamma.shape} != ({RANK}, {DIM})"
    return np.concatenate([
        alpha.astype(np.float64).ravel(),
        beta.astype(np.float64).ravel(),
        gamma.astype(np.float64).ravel(),
    ]).tobytes()


def blob_to_factors(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a 4104-byte blob into three (RANK, DIM) factor matrices."""
    assert len(blob) == BLOB_BYTES, f"blob length {len(blob)} != {BLOB_BYTES}"
    flat = np.frombuffer(blob, dtype=np.float64)
    assert flat.shape[0] == N_PARAMS
    parts = flat.reshape(3, RANK, DIM)
    return parts[0].copy(), parts[1].copy(), parts[2].copy()


def bulk_blobs_to_stacked(blobs: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulk-decode N blobs into three (N, RANK, DIM) contiguous arrays.

    ~10-50x faster than calling blob_to_factors in a loop because it
    does one memcpy + one reshape instead of N function calls with 3N copies.
    """
    N = len(blobs)
    if N == 0:
        empty = np.empty((0, RANK, DIM), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    buf = b"".join(blobs)
    flat = np.frombuffer(buf, dtype=np.float64).copy()  # writable
    stacked = flat.reshape(N, 3, RANK, DIM)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def factors_from_json(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load factor matrices from the JSON format used by optimize_v2 / step84.

    JSON has {"terms": [{"alpha_support": [...], "alpha_values": [...], ...}, ...]}.
    Returns dense (RANK, DIM) arrays.
    """
    terms = data["terms"]
    assert len(terms) == RANK, f"Expected {RANK} terms, got {len(terms)}"
    alpha = np.zeros((RANK, DIM), dtype=np.float64)
    beta = np.zeros((RANK, DIM), dtype=np.float64)
    gamma = np.zeros((RANK, DIM), dtype=np.float64)
    for i, term in enumerate(terms):
        for idx, val in zip(term["alpha_support"], term["alpha_values"]):
            alpha[i, idx] = val
        for idx, val in zip(term["beta_support"], term["beta_values"]):
            beta[i, idx] = val
        for idx, val in zip(term["gamma_support"], term["gamma_values"]):
            gamma[i, idx] = val
    return alpha, beta, gamma


def factors_to_json(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> dict:
    """Export factor matrices to the JSON format used by optimize_v2 / step84."""
    terms = []
    for i in range(RANK):
        a_sup = [int(j) for j in range(DIM) if abs(alpha[i, j]) > 1e-15]
        b_sup = [int(j) for j in range(DIM) if abs(beta[i, j]) > 1e-15]
        g_sup = [int(j) for j in range(DIM) if abs(gamma[i, j]) > 1e-15]
        terms.append({
            "alpha_support": a_sup,
            "alpha_values": [float(alpha[i, j]) for j in a_sup],
            "beta_support": b_sup,
            "beta_values": [float(beta[i, j]) for j in b_sup],
            "gamma_support": g_sup,
            "gamma_values": [float(gamma[i, j]) for j in g_sup],
        })
    return {"terms": terms}


def compute_support_hash(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                         threshold: float = 1e-8) -> str:
    """Compute canonical support hash (S3×S3×S3 orbit representative).

    For dense (9,9,9) solutions all supports are identical under every
    permutation, so the slow 216-iteration canonical loop is unnecessary.
    In that case we hash the quantised coefficients instead — fast and
    still groups near-identical solutions together while keeping genuinely
    different coefficient patterns as separate candidates.
    """
    # Fast path: detect all-dense
    all_dense = True
    for i in range(alpha.shape[0]):
        if (np.sum(np.abs(alpha[i]) > threshold) < DIM or
            np.sum(np.abs(beta[i]) > threshold) < DIM or
            np.sum(np.abs(gamma[i]) > threshold) < DIM):
            all_dense = False
            break

    if all_dense:
        # Quantise to 0.01 precision — groups very nearby solutions
        q = np.concatenate([alpha.ravel(), beta.ravel(), gamma.ravel()])
        q = np.round(q, 2)
        return hashlib.sha256(q.tobytes()).hexdigest()[:16]

    # Slow path: full S3×S3×S3 canonical form for sparse solutions
    supports = []
    for i in range(alpha.shape[0]):
        a_sup = tuple(int(j) for j in range(DIM) if abs(alpha[i, j]) > threshold)
        b_sup = tuple(int(j) for j in range(DIM) if abs(beta[i, j]) > threshold)
        g_sup = tuple(int(j) for j in range(DIM) if abs(gamma[i, j]) > threshold)
        supports.append((a_sup, b_sup, g_sup))

    best = None
    for pi_rA in _S3_ELEMENTS:
        for pi_shared in _S3_ELEMENTS:
            for pi_cB in _S3_ELEMENTS:
                transformed = []
                for a_sup, b_sup, g_sup in supports:
                    new_a = tuple(sorted(3 * pi_rA[i // 3] + pi_shared[i % 3] for i in a_sup))
                    new_b = tuple(sorted(3 * pi_shared[i // 3] + pi_cB[i % 3] for i in b_sup))
                    new_g = tuple(sorted(3 * pi_rA[i // 3] + pi_cB[i % 3] for i in g_sup))
                    transformed.append((new_a, new_b, new_g))
                transformed.sort()
                if best is None or transformed < best:
                    best = transformed

    raw = ";".join(f"{a}|{b}|{g}" for a, b, g in best)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_support_sig(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                        threshold: float = 1e-8) -> str:
    """Compute support signature like '(9,9,9)' — sizes of support per factor."""
    # Fast path for all-dense
    all_dense = True
    for i in range(alpha.shape[0]):
        if (np.sum(np.abs(alpha[i]) > threshold) < DIM or
            np.sum(np.abs(beta[i]) > threshold) < DIM or
            np.sum(np.abs(gamma[i]) > threshold) < DIM):
            all_dense = False
            break
    if all_dense:
        return f"({DIM},{DIM},{DIM})x{alpha.shape[0]}"

    a_sizes = sorted(int(np.sum(np.abs(alpha[i]) > threshold)) for i in range(alpha.shape[0]))
    b_sizes = sorted(int(np.sum(np.abs(beta[i]) > threshold)) for i in range(beta.shape[0]))
    g_sizes = sorted(int(np.sum(np.abs(gamma[i]) > threshold)) for i in range(gamma.shape[0]))
    return f"A{a_sizes}B{b_sizes}G{g_sizes}"
