"""Test candidate generation: mutations, crossover, batch generation."""

import numpy as np
import pytest

from db_optimizer.config import RANK, DIM
from db_optimizer.generator import (
    mutate_coefficient,
    mutate_gaussian,
    mutate_support_flip,
    crossover,
    generate_batch,
)


def _random_factors(rng):
    return (
        rng.standard_normal((RANK, DIM)),
        rng.standard_normal((RANK, DIM)),
        rng.standard_normal((RANK, DIM)),
    )


class TestMutateCoefficient:
    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_coefficient(*parent, rng)
        assert a.shape == (RANK, DIM)
        assert b.shape == (RANK, DIM)
        assert g.shape == (RANK, DIM)

    def test_at_least_one_change(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_coefficient(*parent, rng)
        changed = (
            not np.array_equal(a, parent[0]) or
            not np.array_equal(b, parent[1]) or
            not np.array_equal(g, parent[2])
        )
        assert changed

    def test_does_not_mutate_parent(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        orig_a = parent[0].copy()
        mutate_coefficient(*parent, rng)
        np.testing.assert_array_equal(parent[0], orig_a)


class TestMutateGaussian:
    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_gaussian(*parent, rng)
        assert a.shape == (RANK, DIM)

    def test_small_perturbation(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_gaussian(*parent, rng, sigma=0.001)
        # Most entries should be unchanged
        total_diff = (
            np.sum(a != parent[0]) +
            np.sum(b != parent[1]) +
            np.sum(g != parent[2])
        )
        assert total_diff <= 15  # at most 5 perturbations × 3 factors


class TestMutateSupportFlip:
    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_support_flip(*parent, rng)
        assert a.shape == (RANK, DIM)

    def test_one_entry_changes(self):
        rng = np.random.default_rng(42)
        parent = _random_factors(rng)
        a, b, g = mutate_support_flip(*parent, rng)
        diffs = (
            np.sum(a != parent[0]) +
            np.sum(b != parent[1]) +
            np.sum(g != parent[2])
        )
        assert diffs == 1


class TestCrossover:
    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        pa = _random_factors(rng)
        pb = _random_factors(rng)
        a, b, g = crossover(pa, pb, rng)
        assert a.shape == (RANK, DIM)

    def test_terms_from_parents(self):
        rng = np.random.default_rng(42)
        pa = _random_factors(rng)
        pb = _random_factors(rng)
        a, b, g = crossover(pa, pb, rng)
        # Each term row should match either parent A or parent B
        for i in range(RANK):
            from_a = np.array_equal(a[i], pa[0][i])
            from_b = np.array_equal(a[i], pb[0][i])
            assert from_a or from_b


class TestGenerateBatch:
    def test_batch_size(self):
        rng = np.random.default_rng(42)
        parents = [_random_factors(rng) for _ in range(5)]
        children = generate_batch(parents, 20, rng)
        assert len(children) == 20

    def test_origin_labels(self):
        rng = np.random.default_rng(42)
        parents = [_random_factors(rng) for _ in range(5)]
        children = generate_batch(parents, 100, rng)
        origins = set(c[3] for c in children)
        assert len(origins) >= 2  # should have at least two mutation types

    def test_all_valid_shapes(self):
        rng = np.random.default_rng(42)
        parents = [_random_factors(rng) for _ in range(5)]
        children = generate_batch(parents, 50, rng)
        for a, b, g, origin in children:
            assert a.shape == (RANK, DIM)
            assert b.shape == (RANK, DIM)
            assert g.shape == (RANK, DIM)
            assert isinstance(origin, str)
