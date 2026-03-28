from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import permutations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPORTS_DIR = REPO_ROOT / "outputs" / "exports"

from src.ade3x3.steps.ade3x3_step11_full_raw_9slot_warehouse import ADE3x3RawWarehouse

S3 = tuple(permutations(range(3)))
RAW_ACTIONS = tuple(
    tuple(3 * row_perm[digit // 3] + col_perm[digit % 3] for digit in range(9))
    for row_perm in S3
    for col_perm in S3
)
FULL_COMPATIBLE_RAW_ACTIONS = tuple(
    tuple(3 * row_perm[digit // 3] + col_perm[digit % 3] for digit in range(9))
    for row_perm in S3
    for _shared_perm in S3
    for col_perm in S3
)


def decode_digits(tuple_id: int) -> list[int]:
    digits = [0] * 9
    value = tuple_id
    for digit_idx in range(8, -1, -1):
        digits[digit_idx] = value % 9
        value //= 9
    return digits


def digits_to_id(digits: list[int]) -> int:
    result = 0
    for digit in digits:
        result = result * 9 + digit
    return result


def decode_checksum(start_id: int, count: int) -> tuple[int, int]:
    checksum = 0
    last_id = start_id + count
    for tuple_id in range(start_id, last_id):
        value = tuple_id
        digit_sum = 0
        for _ in range(9):
            digit_sum += value % 9
            value //= 9
        checksum += digit_sum
    return count, checksum


def canonical36_checksum(start_id: int, count: int) -> tuple[int, int]:
    checksum = 0
    last_id = start_id + count
    for tuple_id in range(start_id, last_id):
        digits = decode_digits(tuple_id)
        min_id = tuple_id
        distinct_count = len(set(digits))
        for action in RAW_ACTIONS:
            transformed = [action[digit] for digit in digits]
            transformed_id = digits_to_id(transformed)
            if transformed_id < min_id:
                min_id = transformed_id
        checksum += (min_id % 1_000_003) + distinct_count
    return count, checksum


def canonical216_checksum(start_id: int, count: int) -> tuple[int, int]:
    checksum = 0
    last_id = start_id + count
    for tuple_id in range(start_id, last_id):
        digits = decode_digits(tuple_id)
        min_id = tuple_id
        distinct_count = len(set(digits))
        for action in FULL_COMPATIBLE_RAW_ACTIONS:
            transformed = [action[digit] for digit in digits]
            transformed_id = digits_to_id(transformed)
            if transformed_id < min_id:
                min_id = transformed_id
        checksum += (min_id % 1_000_003) + distinct_count
    return count, checksum


def canonical216_grouped_summary(start_id: int, count: int) -> tuple[int, int, list[tuple[int, int]]]:
    checksum = 0
    canonical_counts: dict[int, int] = {}
    last_id = start_id + count
    for tuple_id in range(start_id, last_id):
        digits = decode_digits(tuple_id)
        min_id = tuple_id
        distinct_count = len(set(digits))
        for action in FULL_COMPATIBLE_RAW_ACTIONS:
            transformed = [action[digit] for digit in digits]
            transformed_id = digits_to_id(transformed)
            if transformed_id < min_id:
                min_id = transformed_id
        canonical_counts[min_id] = canonical_counts.get(min_id, 0) + 1
        checksum += (min_id % 1_000_003) + distinct_count
    return count, checksum, sorted(canonical_counts.items())


def append_log_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_job(job: tuple[int, int, str]) -> tuple[int, int]:
    start_id, count, mode = job
    if mode == "decode":
        return decode_checksum(start_id, count)
    if mode == "canonical36":
        return canonical36_checksum(start_id, count)
    if mode == "canonical216":
        return canonical216_checksum(start_id, count)
    if mode == "canonical216_grouped":
        return canonical216_grouped_summary(start_id, count)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke benchmark for a full arity-9 raw-layer scan.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Worker process count")
    parser.add_argument("--ids-per-worker", type=int, default=250_000, help="Benchmark IDs assigned to each worker")
    parser.add_argument(
        "--mode",
        choices=("decode", "canonical36", "canonical216", "canonical216_grouped"),
        default="canonical216",
        help="Benchmark kernel: light decode scan, reduced 36-action raw canonicalization, full 216-action compatible canonicalization, or builder-like grouped canonicalization",
    )
    parser.add_argument(
        "--log-csv",
        default=str(EXPORTS_DIR / "arity9_smokescreen_log.csv"),
        help="CSV file to append benchmark results to",
    )
    args = parser.parse_args()

    workers = max(1, args.workers)
    ids_per_worker = max(1, args.ids_per_worker)

    warehouse = ADE3x3RawWarehouse()
    warehouse.build_all_layers()
    full_count = warehouse.layers[9].tuple_count

    total_sample_ids = workers * ids_per_worker
    if total_sample_ids > full_count:
        total_sample_ids = full_count
        ids_per_worker = max(1, full_count // workers)

    jobs = []
    next_id = 0
    for worker_idx in range(workers):
        remaining = total_sample_ids - next_id
        if remaining <= 0:
            break
        chunk = ids_per_worker if worker_idx < workers - 1 else remaining
        chunk = min(chunk, remaining)
        jobs.append((next_id, chunk, args.mode))
        next_id += chunk

    print(f"Arity-9 full-layer size: {full_count:,}")
    print(f"Workers: {workers}")
    print(f"Smoke sample IDs: {sum(chunk for _, chunk, _ in jobs):,}")
    print(f"IDs per worker target: {ids_per_worker:,}")
    if args.mode == "decode":
        print("Benchmark operation: full 9-digit base-9 decode plus checksum accumulation")
    elif args.mode == "canonical36":
        print("Benchmark operation: 9-digit decode, 36 raw symmetry actions, canonical representative minimization, and checksum accumulation")
    elif args.mode == "canonical216":
        print("Benchmark operation: 9-digit decode, full 216 compatible-action loop, canonical representative minimization, and checksum accumulation")
    else:
        print("Benchmark operation: 9-digit decode, full 216 compatible-action loop, per-worker orbit-style hash grouping, parent merge, and checksum accumulation")

    start = time.perf_counter()
    total_processed = 0
    total_checksum = 0
    merged_unique = None
    merged_total = None
    merged_top_bucket = None
    merged_mean_bucket = None
    merged_top5 = None
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if args.mode == "canonical216_grouped":
            merged_counts: dict[int, int] = {}
            for processed, checksum, grouped_items in executor.map(run_job, jobs):
                total_processed += processed
                total_checksum += checksum
                for canonical_id, local_count in grouped_items:
                    merged_counts[canonical_id] = merged_counts.get(canonical_id, 0) + local_count
            merged_unique = len(merged_counts)
            merged_total = sum(merged_counts.values())
            merged_top_bucket = max(merged_counts.values()) if merged_counts else 0
            merged_mean_bucket = (merged_total / merged_unique) if merged_unique else 0.0
            top5 = sorted(merged_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            merged_top5 = "; ".join(f"{canonical_id}:{bucket_size}" for canonical_id, bucket_size in top5)
        else:
            for processed, checksum in executor.map(run_job, jobs):
                total_processed += processed
                total_checksum += checksum
    elapsed = time.perf_counter() - start

    ids_per_second = total_processed / elapsed if elapsed > 0 else 0.0
    estimated_seconds = full_count / ids_per_second if ids_per_second > 0 else float("inf")
    estimated_minutes = estimated_seconds / 60.0
    estimated_hours = estimated_minutes / 60.0

    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Observed throughput: {ids_per_second:,.0f} IDs/sec")
    print(f"Estimated full arity-9 scan time: {estimated_seconds:,.1f} sec")
    print(f"Estimated full arity-9 scan time: {estimated_minutes:,.1f} min")
    print(f"Estimated full arity-9 scan time: {estimated_hours:,.2f} hr")
    if merged_unique is not None and merged_total is not None:
        print(f"Merged unique canonical IDs in sample: {merged_unique:,}")
        print(f"Merged grouped total in sample: {merged_total:,}")
        print(f"Largest canonical bucket in sample: {merged_top_bucket}")
        print(f"Mean bucket size in sample: {merged_mean_bucket:.3f}")
        print(f"Top-5 canonical buckets in sample: {merged_top5}")
    print(f"Checksum guard: {total_checksum}")

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "workers": workers,
        "ids_per_worker": ids_per_worker,
        "sample_ids": total_processed,
        "elapsed_seconds": f"{elapsed:.6f}",
        "throughput_ids_per_second": f"{ids_per_second:.3f}",
        "estimated_full_seconds": f"{estimated_seconds:.3f}",
        "estimated_full_minutes": f"{estimated_minutes:.3f}",
        "estimated_full_hours": f"{estimated_hours:.6f}",
        "checksum_guard": total_checksum,
        "sample_unique_canonical_ids": merged_unique if merged_unique is not None else "",
        "sample_total_grouped_ids": merged_total if merged_total is not None else "",
        "sample_top_bucket": merged_top_bucket if merged_top_bucket is not None else "",
        "sample_mean_bucket_size": f"{merged_mean_bucket:.6f}" if merged_mean_bucket is not None else "",
        "sample_top5_buckets": merged_top5 if merged_top5 is not None else "",
    }
    append_log_row(Path(args.log_csv), log_row)
    print(f"Log appended to: {args.log_csv}")


if __name__ == "__main__":
    main()