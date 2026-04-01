"""Test CPU FP64 minimax refinement."""

import numpy as np
import pytest

from db_optimizer.config import RANK, DIM
from db_optimizer.tensor import fitness_maxabs
from db_optimizer.cpu_refine import cpu_minimax_refine


class TestCPURefine:
    def test_refine_improves_or_same(self):
        """Refinement should never make fitness worse."""
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal((RANK, DIM)) * 0.3
        beta = rng.standard_normal((RANK, DIM)) * 0.3
        gamma = rng.standard_normal((RANK, DIM)) * 0.3

        before = fitness_maxabs(alpha, beta, gamma)
        a2, b2, g2, after = cpu_minimax_refine(
            alpha, beta, gamma, sweeps=1, fine_range=0.01
        )
        assert after <= before + 1e-10  # should not worsen

    def test_output_shapes(self):
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal((RANK, DIM))
        beta = rng.standard_normal((RANK, DIM))
        gamma = rng.standard_normal((RANK, DIM))

        a2, b2, g2, fit = cpu_minimax_refine(alpha, beta, gamma, sweeps=1)
        assert a2.shape == (RANK, DIM)
        assert b2.shape == (RANK, DIM)
        assert g2.shape == (RANK, DIM)
        assert isinstance(fit, float)
        assert fit > 0

    def test_does_not_mutate_input(self):
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal((RANK, DIM))
        orig = alpha.copy()
        cpu_minimax_refine(alpha, np.zeros((RANK, DIM)), np.zeros((RANK, DIM)), sweeps=1)
        np.testing.assert_array_equal(alpha, orig)
