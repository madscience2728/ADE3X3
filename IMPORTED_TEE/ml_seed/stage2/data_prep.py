from __future__ import annotations

import itertools
import hashlib
import random
from collections import Counter

import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.schema import TrainingSample

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def load_samples(paths: list[str]) -> list[TrainingSample]:
    samples_by_key: dict[str, TrainingSample] = {}
    for path in paths:
        for sample in DatasetWriter.load_all(path):
            key = hashlib.md5(np.asarray(sample.image, dtype=np.float32).tobytes()).hexdigest()
            incumbent = samples_by_key.get(key)
            if incumbent is None:
                samples_by_key[key] = sample
                continue
            incumbent_score = (incumbent.res_final, incumbent.rank, 0 if incumbent.res_k is not None else 1)
            candidate_score = (sample.res_final, sample.rank, 0 if sample.res_k is not None else 1)
            if candidate_score < incumbent_score:
                samples_by_key[key] = sample
    return list(samples_by_key.values())


def residual_histogram(samples: list[TrainingSample], bins: int = 20) -> dict:
    if not samples:
        return {"bins": [], "counts": []}
    residuals = np.array([sample.res_final for sample in samples], dtype=np.float64)
    counts, edges = np.histogram(residuals, bins=bins)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "min": float(residuals.min()),
        "max": float(residuals.max()),
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
    }


def build_rank_groups(samples: list[TrainingSample]) -> dict[int, list[TrainingSample]]:
    by_rank: dict[int, list[TrainingSample]] = {}
    for sample in samples:
        by_rank.setdefault(sample.rank, []).append(sample)
    return by_rank


def summarize_pairs(
    by_rank: dict[int, list[TrainingSample]],
    margin: float,
    max_pairs_per_rank: int,
) -> dict:
    pair_counts: dict[int, int] = {}
    usable_counts: dict[int, int] = {}
    sampled_pair_counts: dict[int, int] = {}
    for rank, group in by_rank.items():
        total_pairs = len(group) * (len(group) - 1) // 2
        pair_counts[rank] = total_pairs
        sampled_pair_counts[rank] = min(total_pairs, max_pairs_per_rank)
        usable = 0
        for sample_a, sample_b in itertools.combinations(group, 2):
            if abs(sample_a.res_final - sample_b.res_final) >= margin:
                usable += 1
        usable_counts[rank] = usable
    total_pairs = sum(pair_counts.values())
    usable_pairs = sum(usable_counts.values())
    return {
        "pair_counts": pair_counts,
        "sampled_pair_counts": sampled_pair_counts,
        "usable_pair_counts": usable_counts,
        "total_pairs": total_pairs,
        "usable_pairs": usable_pairs,
        "usable_fraction": (usable_pairs / total_pairs) if total_pairs else 0.0,
    }


def dataset_summary(
    paths: list[str],
    margin: float = 0.1,
    max_pairs_per_rank: int = 10_000,
) -> dict:
    samples = load_samples(paths)
    by_rank = build_rank_groups(samples)
    return {
        "n_samples": len(samples),
        "rank_counts": dict(Counter(sample.rank for sample in samples)),
        "residual_histogram": residual_histogram(samples),
        "pair_summary": summarize_pairs(by_rank, margin, max_pairs_per_rank),
    }


class PairDataset(Dataset):
    """Same-rank pair dataset for pairwise residual ordering."""

    def __init__(self, paths: list[str], max_pairs_per_rank: int = 10_000):
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for ml_seed.stage2.data_prep. Install torch in the active environment."
            ) from _TORCH_IMPORT_ERROR

        samples = load_samples(paths)
        by_rank = build_rank_groups(samples)
        self.pairs: list[tuple[TrainingSample, TrainingSample]] = []
        self.rank_summary = {rank: len(group) for rank, group in by_rank.items()}

        for rank, group in by_rank.items():
            all_pairs = list(itertools.combinations(range(len(group)), 2))
            if len(all_pairs) > max_pairs_per_rank:
                all_pairs = random.sample(all_pairs, max_pairs_per_rank)
            for i, j in all_pairs:
                self.pairs.append((group[i], group[j]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sample_a, sample_b = self.pairs[idx]
        image_a = torch.from_numpy(np.array(sample_a.image, copy=True)).unsqueeze(0)
        image_b = torch.from_numpy(np.array(sample_b.image, copy=True)).unsqueeze(0)
        res_a = torch.tensor(sample_a.res_final, dtype=torch.float32)
        res_b = torch.tensor(sample_b.res_final, dtype=torch.float32)
        rank = torch.tensor(sample_a.rank, dtype=torch.int64)
        return image_a, image_b, res_a, res_b, rank