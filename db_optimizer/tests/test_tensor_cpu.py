"""Test CPU tensor operations: target construction, reconstruction, fitness."""

import numpy as np
import pytest

from db_optimizer.config import TARGET_TENSOR, DIM, RANK
from db_optimizer.tensor import reconstruct, fitness_maxabs, fitness_frobenius


class TestTargetTensor:
    def test_shape(self):
        assert TARGET_TENSOR.shape == (9, 9, 9)

    def test_dtype(self):
        assert TARGET_TENSOR.dtype == np.float64

    def test_27_live_entries(self):
        assert np.sum(TARGET_TENSOR == 1.0) == 27

    def test_702_dead_entries(self):
        assert np.sum(TARGET_TENSOR == 0.0) == 702

    def test_total_entries(self):
        assert TARGET_TENSOR.size == 729

    def test_live_positions(self):
        """Verify T[3r+s, 3s+u, 3r+u] = 1 for all r,s,u in {0,1,2}."""
        for r in range(3):
            for s in range(3):
                for u in range(3):
                    a = 3 * r + s
                    b = 3 * s + u
                    c = 3 * r + u
                    assert TARGET_TENSOR[a, b, c] == 1.0, f"T[{a},{b},{c}] != 1"

    def test_frobenius_norm(self):
        assert np.linalg.norm(TARGET_TENSOR) == pytest.approx(np.sqrt(27.0))


class TestReconstruct:
    def test_zero_factors(self):
        alpha = np.zeros((RANK, DIM))
        beta = np.zeros((RANK, DIM))
        gamma = np.zeros((RANK, DIM))
        R = reconstruct(alpha, beta, gamma)
        np.testing.assert_array_equal(R, np.zeros((9, 9, 9)))

    def test_rank1_identity(self):
        """Single rank-1 term: a ⊗ b ⊗ c."""
        alpha = np.zeros((1, DIM))
        beta = np.zeros((1, DIM))
        gamma = np.zeros((1, DIM))
        alpha[0, 0] = 1.0
        beta[0, 0] = 1.0
        gamma[0, 0] = 1.0
        R = reconstruct(alpha, beta, gamma)
        assert R[0, 0, 0] == 1.0
        assert R.sum() == 1.0


class TestFitness:
    def test_zero_decomposition_fitness(self):
        """Zero factors should give fitness = 1.0 (max entry in target)."""
        alpha = np.zeros((RANK, DIM))
        beta = np.zeros((RANK, DIM))
        gamma = np.zeros((RANK, DIM))
        f = fitness_maxabs(alpha, beta, gamma)
        assert f == pytest.approx(1.0)

    def test_exact_decomposition_fitness_zero(self):
        """If we reconstruct exactly, fitness should be 0."""
        # Use the 27-term trivial decomposition
        alpha = np.zeros((27, DIM))
        beta = np.zeros((27, DIM))
        gamma = np.zeros((27, DIM))
        idx = 0
        for r in range(3):
            for s in range(3):
                for u in range(3):
                    alpha[idx, 3 * r + s] = 1.0
                    beta[idx, 3 * s + u] = 1.0
                    gamma[idx, 3 * r + u] = 1.0
                    idx += 1
        f = fitness_maxabs(alpha, beta, gamma)
        assert f == pytest.approx(0.0, abs=1e-14)

    def test_frobenius_zero_factors(self):
        alpha = np.zeros((RANK, DIM))
        beta = np.zeros((RANK, DIM))
        gamma = np.zeros((RANK, DIM))
        f = fitness_frobenius(alpha, beta, gamma)
        assert f == pytest.approx(np.sqrt(27.0))

    def test_fitness_positive(self):
        """Random factors should give positive fitness."""
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal((RANK, DIM))
        beta = rng.standard_normal((RANK, DIM))
        gamma = rng.standard_normal((RANK, DIM))
        f = fitness_maxabs(alpha, beta, gamma)
        assert f > 0
