from __future__ import annotations

import argparse
import numpy as np

from ml_seed.dataset import DatasetWriter
from ml_seed.stage4.inject import inject
from ml_seed.stage4.invert import image_to_C

from bini_slime.slime_search import compute_algebra_rank


def eval_seed(C: np.ndarray, rank: int, restarts: int = 3) -> float:
    residual, _best_rank, _factors = compute_algebra_rank(C, ranks=[rank], n_restarts=restarts)
    return float(residual)


def run(n_seeds: int = 20, rank: int = 19) -> None:
    print(f"\nBenchmark - {n_seeds} seeds per source, rank={rank}\n{'=' * 56}")

    manifest = inject(n_seeds=n_seeds, n_steps=200)
    generated_residuals: list[float] = []
    for entry in manifest:
        with np.load(entry["path"]) as data:
            residual = eval_seed(data["C"], rank)
        generated_residuals.append(residual)
        print(f"  gen   score={entry['score_final']:.4f} -> res={residual:.4f}")

    rng = np.random.default_rng(42)
    random_residuals = [eval_seed(rng.standard_normal((9, 9, 9)), rank) for _ in range(n_seeds)]

    samples = DatasetWriter.load_all("ml_seed/data/samples.jsonl")
    rank_samples = [sample for sample in samples if sample.rank == rank]
    rank_samples.sort(key=lambda sample: sample.res_final)
    historical_residuals: list[float] = []
    for sample in rank_samples[:n_seeds]:
        historical_residuals.append(eval_seed(image_to_C(sample.image), rank))

    print(f"\n{'Source':<16} {'mean res':>10} {'min res':>10} {'median':>10}")
    print("-" * 50)
    for name, values in (
        ("generated", generated_residuals),
        ("random", random_residuals),
        ("historical", historical_residuals),
    ):
        if values:
            print(f"{name:<16} {np.mean(values):>10.4f} {np.min(values):>10.4f} {np.median(values):>10.4f}")

    print()
    if generated_residuals and np.mean(generated_residuals) < np.mean(random_residuals):
        print("Generator beats random on mean residual.")
    else:
        print("Generator does not yet beat random on mean residual.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generated seeds against baselines.")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--rank", type=int, default=19)
    args = parser.parse_args()
    run(n_seeds=args.n_seeds, rank=args.rank)


if __name__ == "__main__":
    main()