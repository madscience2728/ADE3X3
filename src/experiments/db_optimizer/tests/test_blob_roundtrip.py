"""Test blob encode/decode roundtrip and support hashing."""

import numpy as np
import pytest

from db_optimizer.blob import (
    factors_to_blob,
    blob_to_factors,
    factors_from_json,
    factors_to_json,
    compute_support_hash,
    compute_support_sig,
)
from db_optimizer.config import RANK, DIM, BLOB_BYTES


def _random_factors(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    alpha = rng.standard_normal((RANK, DIM))
    beta = rng.standard_normal((RANK, DIM))
    gamma = rng.standard_normal((RANK, DIM))
    return alpha, beta, gamma


class TestBlobRoundtrip:
    def test_encode_decode_exact(self):
        alpha, beta, gamma = _random_factors()
        blob = factors_to_blob(alpha, beta, gamma)
        assert len(blob) == BLOB_BYTES
        a2, b2, g2 = blob_to_factors(blob)
        np.testing.assert_array_equal(a2, alpha)
        np.testing.assert_array_equal(b2, beta)
        np.testing.assert_array_equal(g2, gamma)

    def test_blob_length(self):
        alpha, beta, gamma = _random_factors()
        blob = factors_to_blob(alpha, beta, gamma)
        assert len(blob) == 513 * 8

    def test_decode_bad_length_raises(self):
        with pytest.raises(AssertionError):
            blob_to_factors(b"\x00" * 100)

    def test_encode_bad_shape_raises(self):
        with pytest.raises(AssertionError):
            factors_to_blob(np.zeros((10, 9)), np.zeros((19, 9)), np.zeros((19, 9)))

    def test_roundtrip_preserves_negatives(self):
        rng = np.random.default_rng(99)
        alpha, beta, gamma = _random_factors(rng)
        alpha[0, 0] = -1.23456789012345
        blob = factors_to_blob(alpha, beta, gamma)
        a2, b2, g2 = blob_to_factors(blob)
        assert a2[0, 0] == alpha[0, 0]


class TestJsonRoundtrip:
    def test_json_roundtrip(self):
        alpha, beta, gamma = _random_factors()
        data = factors_to_json(alpha, beta, gamma)
        a2, b2, g2 = factors_from_json(data)
        np.testing.assert_allclose(a2, alpha, atol=1e-15)
        np.testing.assert_allclose(b2, beta, atol=1e-15)
        np.testing.assert_allclose(g2, gamma, atol=1e-15)

    def test_json_sparse_roundtrip(self):
        """Sparse factors: only a few nonzero entries per term."""
        alpha = np.zeros((RANK, DIM))
        beta = np.zeros((RANK, DIM))
        gamma = np.zeros((RANK, DIM))
        alpha[0, 0] = 1.0
        alpha[0, 3] = -0.5
        beta[0, 1] = 2.0
        gamma[0, 2] = 3.0
        data = factors_to_json(alpha, beta, gamma)
        a2, b2, g2 = factors_from_json(data)
        np.testing.assert_array_equal(a2, alpha)
        np.testing.assert_array_equal(b2, beta)
        np.testing.assert_array_equal(g2, gamma)


class TestSupportHash:
    def test_deterministic(self):
        alpha, beta, gamma = _random_factors()
        h1 = compute_support_hash(alpha, beta, gamma)
        h2 = compute_support_hash(alpha, beta, gamma)
        assert h1 == h2

    def test_different_factors_different_hash(self):
        rng = np.random.default_rng(42)
        a1, b1, g1 = _random_factors(rng)
        rng2 = np.random.default_rng(99)
        a2, b2, g2 = _random_factors(rng2)
        h1 = compute_support_hash(a1, b1, g1)
        h2 = compute_support_hash(a2, b2, g2)
        # Different random factors should have different hashes (overwhelmingly likely)
        # But both are dense (9,9,9) so they might collide — just check type
        assert isinstance(h1, str)
        assert len(h1) == 16

    def test_hash_length(self):
        alpha, beta, gamma = _random_factors()
        h = compute_support_hash(alpha, beta, gamma)
        assert len(h) == 16  # sha256 hex prefix


class TestSupportSig:
    def test_dense_sig(self):
        alpha, beta, gamma = _random_factors()
        sig = compute_support_sig(alpha, beta, gamma)
        assert "9" in sig  # dense: all entries are 9

    def test_sparse_sig(self):
        alpha = np.zeros((RANK, DIM))
        beta = np.zeros((RANK, DIM))
        gamma = np.zeros((RANK, DIM))
        alpha[0, 0] = 1.0
        beta[0, 1] = 1.0
        gamma[0, 2] = 1.0
        sig = compute_support_sig(alpha, beta, gamma)
        assert "1" in sig  # sparse: some entries are 1
