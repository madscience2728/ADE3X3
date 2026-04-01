"""Test GPU batch evaluation against CPU reference."""

import numpy as np
import pytest
import torch

from db_optimizer.config import RANK, DIM
from db_optimizer.tensor import fitness_maxabs, fitness_frobenius
from db_optimizer.gpu_eval import gpu_batch_fitness, gpu_batch_frobenius
from db_optimizer.blob import factors_to_blob
from db_optimizer.gpu_eval import gpu_batch_fitness_from_blobs


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _random_factors(rng, n=1):
    """Generate n random factor triples."""
    return [
        (rng.standard_normal((RANK, DIM)),
         rng.standard_normal((RANK, DIM)),
         rng.standard_normal((RANK, DIM)))
        for _ in range(n)
    ]


class TestGPUBatchFitness:
    def test_empty_batch(self):
        result = gpu_batch_fitness([])
        assert result.shape == (0,)

    def test_single_candidate(self):
        rng = np.random.default_rng(42)
        factors = _random_factors(rng, 1)
        gpu_fit = gpu_batch_fitness(factors)
        cpu_fit = fitness_maxabs(*factors[0])
        assert abs(gpu_fit[0] - cpu_fit) < 1e-3  # FP32 tolerance

    def test_batch_100(self):
        rng = np.random.default_rng(123)
        factors = _random_factors(rng, 100)
        gpu_fits = gpu_batch_fitness(factors)
        assert gpu_fits.shape == (100,)

        cpu_fits = np.array([fitness_maxabs(*f) for f in factors])
        # FP32 vs FP64: allow 1e-3 relative tolerance at typical magnitude ~10
        np.testing.assert_allclose(gpu_fits, cpu_fits, rtol=1e-3, atol=1e-4)

    def test_zero_factors_gives_one(self):
        factors = [(np.zeros((RANK, DIM)), np.zeros((RANK, DIM)), np.zeros((RANK, DIM)))]
        gpu_fit = gpu_batch_fitness(factors)
        assert abs(gpu_fit[0] - 1.0) < 1e-5

    def test_from_blobs(self):
        rng = np.random.default_rng(77)
        factors = _random_factors(rng, 10)
        blobs = [factors_to_blob(*f) for f in factors]
        gpu_fit_blobs = gpu_batch_fitness_from_blobs(blobs)
        gpu_fit_direct = gpu_batch_fitness(factors)
        np.testing.assert_allclose(gpu_fit_blobs, gpu_fit_direct, atol=1e-6)


class TestGPUBatchFrobenius:
    def test_batch_10(self):
        rng = np.random.default_rng(55)
        factors = _random_factors(rng, 10)
        gpu_fro = gpu_batch_frobenius(factors)
        cpu_fro = np.array([fitness_frobenius(*f) for f in factors])
        np.testing.assert_allclose(gpu_fro, cpu_fro, rtol=1e-3, atol=1e-3)

    def test_zero_factors_frobenius(self):
        factors = [(np.zeros((RANK, DIM)), np.zeros((RANK, DIM)), np.zeros((RANK, DIM)))]
        gpu_fro = gpu_batch_frobenius(factors)
        assert abs(gpu_fro[0] - np.sqrt(27.0)) < 1e-3
