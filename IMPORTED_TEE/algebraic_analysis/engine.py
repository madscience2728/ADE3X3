"""ADEV2 scanning engine.

This module turns fringe-family generators into a reproducible exploration pass:
generate candidate algebras, compute diagnostics, optionally run CP-rank search,
and persist incremental JSON output for long scans.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ade.constraints.checks import verify_all
from ade.invariants.tier1 import tier1
from adev2.families import CandidateSpec, build_candidate, build_candidate_specs_slab, default_candidate_specs, total_candidate_count
from adev2.persistence import ADEV2RunStore
from matmul_search.algebra_complexity import compute_algebra_rank

try:
    import ctypes
    from ctypes import wintypes
except ImportError:
    ctypes = None
    wintypes = None


@dataclass
class ADEV2Result:
    candidate_id: str
    content_hash: str
    signature: dict[str, Any]
    spec: CandidateSpec
    dim: int
    nnz: int
    tensor_norm: float
    tier1: dict[str, Any]
    flags: dict[str, bool]
    fringe_metrics: dict[str, Any]
    cp_rank: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "signature": self.signature,
            "family": self.spec.family,
            "params": self.spec.params,
            "idea_refs": list(self.spec.idea_refs),
            "label": self.spec.label,
            "dim": self.dim,
            "nnz": self.nnz,
            "tensor_norm": self.tensor_norm,
            "tier1": self.tier1,
            "flags": self.flags,
            "fringe_metrics": self.fringe_metrics,
            "cp_rank": self.cp_rank,
        }


def _bool_flags(verification: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {name: bool(result["verified"]) for name, result in verification.items()}


def _basis_annihilator_counts(tensor: np.ndarray, tol: float = 1e-12) -> dict[str, int]:
    dim = tensor.shape[0]
    left_zero = 0
    right_zero = 0
    two_sided = 0
    for idx in range(dim):
        left = float(np.linalg.norm(tensor[idx, :, :]))
        right = float(np.linalg.norm(tensor[:, idx, :]))
        if left <= tol:
            left_zero += 1
        if right <= tol:
            right_zero += 1
        if left <= tol and right <= tol:
            two_sided += 1
    return {
        "basis_left_annihilators": left_zero,
        "basis_right_annihilators": right_zero,
        "basis_two_sided_annihilators": two_sided,
    }


def _one_sided_zero_divisor_pairs(tensor: np.ndarray, tol: float = 1e-12) -> dict[str, int]:
    dim = tensor.shape[0]
    left_only = 0
    right_only = 0
    symmetric_zero = 0
    for left in range(dim):
        for right in range(dim):
            forward = float(np.linalg.norm(tensor[left, right, :]))
            backward = float(np.linalg.norm(tensor[right, left, :]))
            if forward <= tol and backward > tol:
                left_only += 1
            elif forward > tol and backward <= tol:
                right_only += 1
            elif forward <= tol and backward <= tol:
                symmetric_zero += 1
    return {
        "one_sided_left_zero_pairs": left_only,
        "one_sided_right_zero_pairs": right_only,
        "two_sided_zero_pairs": symmetric_zero,
    }


def _square_zero_fraction(tensor: np.ndarray, tol: float = 1e-12) -> float:
    dim = tensor.shape[0]
    square_zero = 0
    for idx in range(dim):
        if float(np.linalg.norm(tensor[idx, idx, :])) <= tol:
            square_zero += 1
    return square_zero / max(dim, 1)


def _fringe_score(flags: dict[str, bool], metrics: dict[str, Any]) -> float:
    score = 0.0
    if not flags.get("ASSOCIATIVE", False):
        score += 1.5
    if not flags.get("COMMUTATIVE", False):
        score += 1.0
    if not flags.get("FLEXIBLE", False):
        score += 0.5
    score += 0.25 * metrics["basis_two_sided_annihilators"]
    score += 0.02 * (metrics["one_sided_left_zero_pairs"] + metrics["one_sided_right_zero_pairs"])
    score += metrics["square_zero_fraction"]
    return float(score)


def _candidate_id(spec: CandidateSpec) -> str:
    if not spec.params:
        return spec.family
    params = "__".join(f"{key}-{spec.params[key]}" for key in sorted(spec.params))
    return f"{spec.family}__{params}"


def _candidate_signature(
    dim: int,
    nnz: int,
    flags: dict[str, bool],
    tier1_summary: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dim": dim,
        "nnz": nnz,
        "flags_true": sorted(name for name, value in flags.items() if value),
        "killing_signature": list(tier1_summary["killing_signature"]),
        "killing_rank": int(tier1_summary["killing_rank"]),
        "nilpotency_class": int(tier1_summary["nilpotency_class"]),
        "semisimple": bool(tier1_summary["semisimple"]),
        "dim_center": int(tier1_summary["dim_center"]),
        "basis_two_sided_annihilators": int(metrics["basis_two_sided_annihilators"]),
        "square_zero_fraction": float(metrics["square_zero_fraction"]),
    }


def evaluate_candidate(
    spec: CandidateSpec,
    rank_cap: int | None,
    rank_restarts: int,
    rank_als_iters: int,
    rank_lbfgs_maxiter: int,
    workers: int | None,
) -> ADEV2Result:
    algebra = build_candidate(spec)
    verification = verify_all(algebra)
    flags = _bool_flags(verification)
    tier1_summary = tier1(algebra)
    tensor = algebra.C()
    metrics: dict[str, Any] = {}
    metrics.update(_basis_annihilator_counts(tensor))
    metrics.update(_one_sided_zero_divisor_pairs(tensor))
    metrics["square_zero_fraction"] = _square_zero_fraction(tensor)
    metrics["fringe_score"] = _fringe_score(flags, metrics)
    signature = _candidate_signature(algebra.n, algebra.nnz, flags, tier1_summary, metrics)

    cp_rank: dict[str, Any] | None = None
    if rank_cap is not None:
        complexity = compute_algebra_rank(
            algebra,
            max_rank=rank_cap,
            n_restarts=rank_restarts,
            als_iters=rank_als_iters,
            lbfgs_maxiter=rank_lbfgs_maxiter,
            max_workers=workers,
            verbose=False,
        )
        cp_rank = complexity.to_dict()

    return ADEV2Result(
        candidate_id=_candidate_id(spec),
        content_hash=algebra.content_hash(),
        signature=signature,
        spec=spec,
        dim=algebra.n,
        nnz=algebra.nnz,
        tensor_norm=float(np.linalg.norm(tensor)),
        tier1={
            "dim_center": int(tier1_summary["dim_center"]),
            "killing_rank": int(tier1_summary["killing_rank"]),
            "killing_signature": list(tier1_summary["killing_signature"]),
            "nilpotency_class": int(tier1_summary["nilpotency_class"]),
            "semisimple": bool(tier1_summary["semisimple"]),
            "assoc_defect_norm": float(tier1_summary["assoc_defect_norm"]),
        },
        flags=flags,
        fringe_metrics=metrics,
        cp_rank=cp_rank,
    )


def _evaluate_candidate_star(args: tuple[CandidateSpec, int | None, int, int, int, int | None]) -> ADEV2Result:
    spec, rank_cap, rank_restarts, rank_als_iters, rank_lbfgs_maxiter, workers = args
    return evaluate_candidate(
        spec,
        rank_cap=rank_cap,
        rank_restarts=rank_restarts,
        rank_als_iters=rank_als_iters,
        rank_lbfgs_maxiter=rank_lbfgs_maxiter,
        workers=workers,
    )


def _worker_init() -> None:
    """Worker initializer for Windows-safe process-pool shutdown.

    Workers ignore console interrupts so shutdown is orchestrated by the parent,
    which explicitly terminates the pool on interruption or stop.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


def _evaluate_candidate_chunk(
    args: tuple[list[tuple[CandidateSpec, int | None, int, int, int, int | None]],]
) -> list[ADEV2Result]:
    (items,) = args
    return [_evaluate_candidate_star(item) for item in items]


def _shutdown_executor(executor: ProcessPoolExecutor, future_map: dict[Any, int]) -> None:
    for future in future_map:
        future.cancel()
    executor.shutdown(cancel_futures=True, wait=False)
    for worker_process in getattr(executor, "_processes", {}).values():
        try:
            if worker_process.is_alive():
                worker_process.terminate()
        except Exception:
            pass


def _result_sort_key(item: ADEV2Result) -> tuple[Any, ...]:
    cp_upper = item.cp_rank["cp_rank_upper"] if item.cp_rank is not None else None
    if cp_upper is None:
        cp_upper = math.inf
    residual = item.cp_rank["residual"] if item.cp_rank is not None else math.inf
    return (cp_upper, residual, -item.fringe_metrics["fringe_score"], item.dim, item.spec.label)


def _promote_results(results: list[ADEV2Result], promote_top_k: int) -> list[ADEV2Result]:
    ordered = sorted(
        results,
        key=lambda item: (
            -item.fringe_metrics["fringe_score"],
            item.signature["nilpotency_class"],
            -item.signature["basis_two_sided_annihilators"],
            item.dim,
            item.spec.label,
        ),
    )
    promoted: list[ADEV2Result] = []
    seen_signatures: set[str] = set()
    for item in ordered:
        signature_key = json.dumps(item.signature, sort_keys=True)
        if signature_key in seen_signatures:
            continue
        promoted.append(item)
        seen_signatures.add(signature_key)
        if len(promoted) >= promote_top_k:
            break
    return promoted


def _build_scan_meta(
    max_dim: int,
    rank_cap: int | None,
    rank_restarts: int,
    rank_als_iters: int,
    rank_lbfgs_maxiter: int,
    completed: int,
    total: int,
    elapsed_sec: float,
    workers: int | None,
    candidate_workers: int | None,
    candidate_slab_size: int | None,
    families: list[str] | None,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "max_dim": max_dim,
        "rank_cap": rank_cap,
        "rank_restarts": rank_restarts,
        "rank_als_iters": rank_als_iters,
        "rank_lbfgs_maxiter": rank_lbfgs_maxiter,
        "workers": workers,
        "candidate_workers": candidate_workers,
        "candidate_slab_size": candidate_slab_size,
        "families": families,
        "limit": limit,
        "completed": completed,
        "total": total,
        "elapsed_sec": elapsed_sec,
    }


def _process_rss_bytes(pid: int) -> int | None:
    if pid <= 0:
        return None
    if os.name == "nt" and ctypes is not None and wintypes is not None:
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        process_query_information = 0x0400
        process_vm_read = 0x0010
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    return None


def _total_rss_bytes(extra_pids: list[int] | None = None) -> int | None:
    total = 0
    seen: set[int] = set()
    for pid in [os.getpid(), *(extra_pids or [])]:
        if pid in seen:
            continue
        seen.add(pid)
        rss = _process_rss_bytes(pid)
        if rss is None:
            continue
        total += rss
    return total or None


def _retune_candidate_slab_size(
    current_slab_size: int,
    actual_candidates: int,
    remaining_candidates: int,
    peak_rss_bytes: int | None,
    ram_target_gb: float | None,
    verbose: bool = False,
) -> int:
    if ram_target_gb is None or ram_target_gb <= 0 or peak_rss_bytes is None or peak_rss_bytes <= 0:
        return current_slab_size
    target_bytes = int(ram_target_gb * (1024 ** 3))
    if target_bytes <= 0:
        return current_slab_size
    basis = max(actual_candidates, 1)
    scaled = int(round(basis * (target_bytes / peak_rss_bytes)))
    min_slab = min(1024, max(remaining_candidates, 1))
    max_growth = max(current_slab_size * 4, basis)
    max_shrink = max(current_slab_size // 4, 1)
    next_slab_size = max(max_shrink, min(scaled, max_growth, max(remaining_candidates, 1)))
    next_slab_size = max(min_slab, next_slab_size)
    if verbose:
        print(
            f"[adev2] memory retune: peak_rss_gb={peak_rss_bytes / (1024 ** 3):.2f} target_gb={ram_target_gb:.2f} next_slab_size={next_slab_size}",
            flush=True,
        )
        if next_slab_size >= remaining_candidates and peak_rss_bytes < target_bytes:
            print(
                "[adev2] memory retune: remaining corpus is too small/light to reach the RAM target with current per-candidate density",
                flush=True,
            )
    return next_slab_size


def _derive_candidate_slab_size(total_candidates: int, candidate_slab_size: int | None, ram_target_gb: float | None) -> int:
    if candidate_slab_size is not None and candidate_slab_size > 0:
        return min(candidate_slab_size, max(total_candidates, 1))
    if ram_target_gb is not None and ram_target_gb > 0:
        estimated_bytes_per_candidate = 256 * 1024
        slab = int((ram_target_gb * (1024 ** 3)) / estimated_bytes_per_candidate)
        return max(1024, min(slab, max(total_candidates, 1)))
    return max(total_candidates, 1)


def _derive_chunk_size(task_count: int, candidate_workers: int, rank_cap: int | None) -> int:
    if task_count <= 0:
        return 1
    if rank_cap is None:
        return max(256, min(1024, math.ceil(task_count / max(candidate_workers, 1))))
    target_chunks = max(candidate_workers * 4, 1)
    return max(8, min(128, math.ceil(task_count / target_chunks)))


def _format_family_counts(family_counts: Counter[str], top_k: int = 4) -> str:
    if not family_counts:
        return "none"
    return ", ".join(f"{family}={count}" for family, count in family_counts.most_common(top_k))


def _scan_specs(
    specs: list[CandidateSpec],
    rank_cap: int | None,
    rank_restarts: int,
    rank_als_iters: int,
    rank_lbfgs_maxiter: int,
    workers: int | None,
    candidate_workers: int | None,
    started: float,
    max_dim: int,
    stage_total_candidates: int,
    families: list[str] | None,
    limit: int | None,
    output_path: Path | None = None,
    store: ADEV2RunStore | None = None,
    stage_name: str = "main",
    verbose: bool = False,
    memory_report: dict[str, Any] | None = None,
) -> list[ADEV2Result]:
    results: list[ADEV2Result] = []
    completed_base = 0
    peak_rss_bytes = _total_rss_bytes()
    slab_started = time.perf_counter()
    family_counts: Counter[str] = Counter()
    if store is not None:
        completed_base = store.start_stage(stage_name, stage_total_candidates)
        completed_ids = store.load_completed_candidate_ids(stage_name)
        pending_before_filter = len(specs)
        specs = [spec for spec in specs if _candidate_id(spec) not in completed_ids]
        skipped_in_slab = pending_before_filter - len(specs)
        if verbose and skipped_in_slab:
            print(
                f"[adev2] resume detected: stage={stage_name} skipping {skipped_in_slab} already-persisted candidates in this slab",
                flush=True,
            )
    if verbose:
        print(
            f"[adev2] evaluating slab: stage={stage_name} pending_candidates={len(specs)} candidate_workers={candidate_workers or 1} rank_cap={rank_cap}",
            flush=True,
        )
    if candidate_workers is None or candidate_workers <= 1 or len(specs) <= 1:
        for index, spec in enumerate(specs, start=completed_base + 1):
            result = evaluate_candidate(
                spec,
                rank_cap=rank_cap,
                rank_restarts=rank_restarts,
                rank_als_iters=rank_als_iters,
                rank_lbfgs_maxiter=rank_lbfgs_maxiter,
                workers=workers,
            )
            results.append(result)
            family_counts[result.spec.family] += 1
            if store is not None:
                store.append_result(stage_name, index, result.to_dict())
            current_rss = _total_rss_bytes()
            if current_rss is not None:
                peak_rss_bytes = max(peak_rss_bytes or 0, current_rss)
            if verbose and ((index - completed_base) % max(32, (store.checkpoint_every if store is not None else 128)) == 0 or index == stage_total_candidates):
                elapsed = max(time.perf_counter() - slab_started, 1e-9)
                slab_rate = (index - completed_base) / elapsed
                print(
                    f"[adev2] progress: stage={stage_name} completed={index}/{stage_total_candidates} slab_rate={slab_rate:.1f}/s rss_gb={((peak_rss_bytes or 0) / (1024 ** 3)):.2f}",
                    flush=True,
                )
            if output_path is not None:
                if store is not None:
                    continue
                payload = {
                    "meta": _build_scan_meta(
                        max_dim=max_dim,
                        rank_cap=rank_cap,
                        rank_restarts=rank_restarts,
                        rank_als_iters=rank_als_iters,
                        rank_lbfgs_maxiter=rank_lbfgs_maxiter,
                        completed=index,
                        total=stage_total_candidates,
                        elapsed_sec=time.perf_counter() - started,
                        workers=workers,
                        candidate_workers=candidate_workers,
                        candidate_slab_size=None,
                        families=families,
                        limit=limit,
                    ),
                    "results": [item.to_dict() for item in sorted(results, key=_result_sort_key)],
                }
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return results

    task_args = [
        (spec, rank_cap, rank_restarts, rank_als_iters, rank_lbfgs_maxiter, workers)
        for spec in specs
    ]
    chunk_size = _derive_chunk_size(len(task_args), max(candidate_workers, 1), rank_cap)
    chunks = [task_args[offset:offset + chunk_size] for offset in range(0, len(task_args), chunk_size)]
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=candidate_workers,
        initializer=_worker_init,
        mp_context=mp_context,
    ) as executor:
        future_map = {
            executor.submit(_evaluate_candidate_chunk, (chunk,)): chunk_index
            for chunk_index, chunk in enumerate(chunks)
        }
        completed = 0
        try:
            for future in as_completed(future_map):
                chunk_results = future.result()
                for result in chunk_results:
                    completed += 1
                    results.append(result)
                    family_counts[result.spec.family] += 1
                    if store is not None:
                        store.append_result(stage_name, completed_base + completed, result.to_dict())
                worker_pids = [process.pid for process in getattr(executor, "_processes", {}).values() if process.pid is not None]
                current_rss = _total_rss_bytes(worker_pids)
                if current_rss is not None:
                    peak_rss_bytes = max(peak_rss_bytes or 0, current_rss)
                if output_path is not None:
                    if store is not None:
                        continue
                    payload = {
                        "meta": _build_scan_meta(
                            max_dim=max_dim,
                            rank_cap=rank_cap,
                            rank_restarts=rank_restarts,
                            rank_als_iters=rank_als_iters,
                            rank_lbfgs_maxiter=rank_lbfgs_maxiter,
                            completed=completed_base + completed,
                            total=stage_total_candidates,
                            elapsed_sec=time.perf_counter() - started,
                            workers=workers,
                            candidate_workers=candidate_workers,
                            candidate_slab_size=None,
                            families=families,
                            limit=limit,
                        ),
                        "results": [item.to_dict() for item in sorted(results, key=_result_sort_key)],
                    }
                    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if verbose:
                    elapsed = max(time.perf_counter() - slab_started, 1e-9)
                    slab_rate = completed / elapsed
                    print(
                        f"[adev2] progress: stage={stage_name} completed={completed_base + completed}/{stage_total_candidates} chunk_results={len(chunk_results)} slab_rate={slab_rate:.1f}/s rss_gb={((peak_rss_bytes or 0) / (1024 ** 3)):.2f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            if store is not None:
                store.flush_stage(stage_name)
            _shutdown_executor(executor, future_map)
            raise
    if store is not None:
        store.finalize_stage(stage_name)
    if memory_report is not None:
        memory_report["peak_rss_bytes"] = peak_rss_bytes
        memory_report["slab_elapsed_sec"] = time.perf_counter() - slab_started
        memory_report["family_counts"] = dict(family_counts)
    return results


def run_adev2_scan(
    output_path: Path,
    max_dim: int = 9,
    rank_cap: int | None = 12,
    rank_restarts: int = 6,
    rank_als_iters: int = 150,
    rank_lbfgs_maxiter: int = 600,
    workers: int | None = None,
    candidate_workers: int | None = None,
    limit: int | None = None,
    families: list[str] | None = None,
    random_radical_samples: int = 0,
    random_relation_samples: int = 0,
    random_seed: int = 0,
    db_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    run_id: str | None = None,
    checkpoint_every: int = 256,
    candidate_slab_size: int | None = None,
    ram_target_gb: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    total_candidates = total_candidate_count(
        max_dim=max_dim,
        random_radical_samples=random_radical_samples,
        random_relation_samples=random_relation_samples,
        random_seed=random_seed,
    )
    if limit is not None:
        total_candidates = min(total_candidates, limit)

    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = ADEV2RunStore(
        db_path=db_path or output_path.with_suffix(".sqlite"),
        checkpoint_dir=checkpoint_dir or (output_path.parent / f"{output_path.stem}_checkpoints"),
        run_id=run_id or f"adev2-scan-{int(time.time())}",
        mode="scan",
        checkpoint_every=checkpoint_every,
        verbose=verbose,
        config={
            "max_dim": max_dim,
            "rank_cap": rank_cap,
            "rank_restarts": rank_restarts,
            "rank_als_iters": rank_als_iters,
            "rank_lbfgs_maxiter": rank_lbfgs_maxiter,
            "workers": workers,
            "candidate_workers": candidate_workers,
            "families": families,
            "limit": limit,
            "random_radical_samples": random_radical_samples,
            "random_relation_samples": random_relation_samples,
            "random_seed": random_seed,
            "candidate_slab_size": candidate_slab_size,
            "ram_target_gb": ram_target_gb,
        },
    )
    effective_candidate_workers = candidate_workers if candidate_workers is not None else workers
    if rank_cap is not None:
        effective_candidate_workers = 1
    slab_size = _derive_candidate_slab_size(total_candidates, candidate_slab_size, ram_target_gb)
    peak_rss_bytes = 0
    if verbose:
        print(
            f"[adev2] scan initialized: total_candidates={total_candidates} slab_size={slab_size} candidate_workers={effective_candidate_workers} workers={workers} ram_target_gb={ram_target_gb}",
            flush=True,
        )
    results: list[ADEV2Result] = []
    overall_family_counts: Counter[str] = Counter()
    offset = 0
    slab_index = 0
    scan_started = time.perf_counter()
    while offset < total_candidates:
        slab_index += 1
        slab_count = min(slab_size, total_candidates - offset)
        if verbose:
            print(
                f"[adev2] preparing slab: index={slab_index} offset={offset} requested_candidates={slab_count}",
                flush=True,
            )
        slab_specs = build_candidate_specs_slab(
            max_dim=max_dim,
            random_radical_samples=random_radical_samples,
            random_relation_samples=random_relation_samples,
            random_seed=random_seed,
            offset=offset,
            max_count=slab_count,
        )
        if families:
            family_set = set(families)
            slab_specs = [spec for spec in slab_specs if spec.family in family_set]
        if verbose:
            print(
                f"[adev2] slab prepared: index={slab_index} actual_candidates={len(slab_specs)} offset_range=[{offset},{offset + slab_count})",
                flush=True,
            )
        if not slab_specs:
            offset += slab_size
            continue
        memory_report: dict[str, Any] = {}
        results.extend(
            _scan_specs(
                slab_specs,
                rank_cap=rank_cap,
                rank_restarts=rank_restarts,
                rank_als_iters=rank_als_iters,
                rank_lbfgs_maxiter=rank_lbfgs_maxiter,
                workers=workers,
                candidate_workers=effective_candidate_workers,
                started=started,
                max_dim=max_dim,
                stage_total_candidates=total_candidates,
                families=families,
                limit=limit,
                output_path=output_path,
                store=store,
                stage_name="main",
                verbose=verbose,
                memory_report=memory_report,
            )
        )
        slab_peak_rss = int(memory_report.get("peak_rss_bytes", 0) or 0)
        slab_elapsed_sec = float(memory_report.get("slab_elapsed_sec", 0.0) or 0.0)
        slab_family_counts = Counter(memory_report.get("family_counts", {}))
        overall_family_counts.update(slab_family_counts)
        peak_rss_bytes = max(peak_rss_bytes, slab_peak_rss)
        if verbose:
            completed_now = store._stage_state.get("main", {}).get("completed", 0)
            overall_rate = completed_now / max(time.perf_counter() - scan_started, 1e-9)
            slab_rate = len(slab_specs) / max(slab_elapsed_sec, 1e-9)
            print(
                f"[adev2] slab complete: index={slab_index} cumulative_completed={completed_now}/{total_candidates} slab_rate={slab_rate:.1f}/s overall_rate={overall_rate:.1f}/s slab_peak_rss_gb={slab_peak_rss / (1024 ** 3):.2f} slab_families=[{_format_family_counts(slab_family_counts)}] overall_families=[{_format_family_counts(overall_family_counts)}]",
                flush=True,
            )
        offset += slab_count
        remaining_candidates = total_candidates - offset
        if remaining_candidates > 0 and candidate_slab_size is None and ram_target_gb is not None and rank_cap is None:
            slab_size = _retune_candidate_slab_size(
                current_slab_size=slab_count,
                actual_candidates=len(slab_specs),
                remaining_candidates=remaining_candidates,
                peak_rss_bytes=slab_peak_rss,
                ram_target_gb=ram_target_gb,
                verbose=verbose,
            )
    all_results = store.load_stage_results("main")
    payload = {
        "meta": _build_scan_meta(
            max_dim=max_dim,
            rank_cap=rank_cap,
            rank_restarts=rank_restarts,
            rank_als_iters=rank_als_iters,
            rank_lbfgs_maxiter=rank_lbfgs_maxiter,
            completed=len(all_results),
            total=total_candidates,
            elapsed_sec=time.perf_counter() - started,
            workers=workers,
            candidate_workers=effective_candidate_workers,
            candidate_slab_size=slab_size,
            families=families,
            limit=limit,
        ),
        "memory": {
            "peak_total_rss_bytes": peak_rss_bytes,
            "peak_total_rss_gb": peak_rss_bytes / (1024 ** 3),
            "ram_target_gb": ram_target_gb,
        },
        "family_counts": dict(overall_family_counts),
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    store.finalize_run(status="completed")
    store.close()
    return payload


def run_adev2_staged_scan(
    output_path: Path,
    max_dim: int = 9,
    families: list[str] | None = None,
    limit: int | None = None,
    stage1_rank_cap: int | None = None,
    stage1_restarts: int = 0,
    stage1_als_iters: int = 0,
    stage1_lbfgs_maxiter: int = 0,
    stage2_rank_cap: int | None = 12,
    stage2_restarts: int = 6,
    stage2_als_iters: int = 150,
    stage2_lbfgs_maxiter: int = 600,
    workers: int | None = None,
    candidate_workers: int | None = None,
    promote_top_k: int = 8,
    stage2_time_budget_sec: float | None = None,
    random_radical_samples: int = 0,
    random_relation_samples: int = 0,
    random_seed: int = 0,
    db_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    run_id: str | None = None,
    checkpoint_every: int = 256,
    candidate_slab_size: int | None = None,
    ram_target_gb: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    total_candidates = total_candidate_count(
        max_dim=max_dim,
        random_radical_samples=random_radical_samples,
        random_relation_samples=random_relation_samples,
        random_seed=random_seed,
    )
    if limit is not None:
        total_candidates = min(total_candidates, limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    store = ADEV2RunStore(
        db_path=db_path or output_path.with_suffix(".sqlite"),
        checkpoint_dir=checkpoint_dir or (output_path.parent / f"{output_path.stem}_checkpoints"),
        run_id=run_id or f"adev2-staged-{int(time.time())}",
        mode="staged",
        checkpoint_every=checkpoint_every,
        verbose=verbose,
        config={
            "max_dim": max_dim,
            "families": families,
            "limit": limit,
            "stage1_rank_cap": stage1_rank_cap,
            "stage1_restarts": stage1_restarts,
            "stage1_als_iters": stage1_als_iters,
            "stage1_lbfgs_maxiter": stage1_lbfgs_maxiter,
            "stage2_rank_cap": stage2_rank_cap,
            "stage2_restarts": stage2_restarts,
            "stage2_als_iters": stage2_als_iters,
            "stage2_lbfgs_maxiter": stage2_lbfgs_maxiter,
            "workers": workers,
            "candidate_workers": candidate_workers,
            "promote_top_k": promote_top_k,
            "stage2_time_budget_sec": stage2_time_budget_sec,
            "random_radical_samples": random_radical_samples,
            "random_relation_samples": random_relation_samples,
            "random_seed": random_seed,
            "candidate_slab_size": candidate_slab_size,
            "ram_target_gb": ram_target_gb,
        },
    )
    effective_candidate_workers = candidate_workers if candidate_workers is not None else workers
    slab_size = _derive_candidate_slab_size(total_candidates, candidate_slab_size, ram_target_gb)
    peak_stage1_rss_bytes = 0
    if verbose:
        print(
            f"[adev2] staged scan initialized: total_candidates={total_candidates} stage1_slab_size={slab_size} candidate_workers={effective_candidate_workers} workers={workers} ram_target_gb={ram_target_gb}",
            flush=True,
        )
    stage1_results: list[ADEV2Result] = []
    stage1_family_counts: Counter[str] = Counter()
    offset = 0
    slab_index = 0
    stage1_started = time.perf_counter()
    while offset < total_candidates:
        slab_index += 1
        slab_count = min(slab_size, total_candidates - offset)
        if verbose:
            print(
                f"[adev2] preparing stage1 slab: index={slab_index} offset={offset} requested_candidates={slab_count}",
                flush=True,
            )
        slab_specs = build_candidate_specs_slab(
            max_dim=max_dim,
            random_radical_samples=random_radical_samples,
            random_relation_samples=random_relation_samples,
            random_seed=random_seed,
            offset=offset,
            max_count=slab_count,
        )
        if families:
            family_set = set(families)
            slab_specs = [spec for spec in slab_specs if spec.family in family_set]
        if verbose:
            print(
                f"[adev2] stage1 slab prepared: index={slab_index} actual_candidates={len(slab_specs)} offset_range=[{offset},{offset + slab_count})",
                flush=True,
            )
        if not slab_specs:
            offset += slab_size
            continue
        memory_report: dict[str, Any] = {}
        stage1_results.extend(
            _scan_specs(
                slab_specs,
                rank_cap=stage1_rank_cap,
                rank_restarts=stage1_restarts,
                rank_als_iters=stage1_als_iters,
                rank_lbfgs_maxiter=stage1_lbfgs_maxiter,
                workers=workers,
                candidate_workers=effective_candidate_workers,
                started=started,
                max_dim=max_dim,
                stage_total_candidates=total_candidates,
                families=families,
                limit=limit,
                store=store,
                stage_name="stage1",
                verbose=verbose,
                memory_report=memory_report,
            )
        )
        slab_peak_rss = int(memory_report.get("peak_rss_bytes", 0) or 0)
        slab_elapsed_sec = float(memory_report.get("slab_elapsed_sec", 0.0) or 0.0)
        slab_family_counts = Counter(memory_report.get("family_counts", {}))
        stage1_family_counts.update(slab_family_counts)
        peak_stage1_rss_bytes = max(peak_stage1_rss_bytes, slab_peak_rss)
        if verbose:
            completed_now = store._stage_state.get("stage1", {}).get("completed", 0)
            overall_rate = completed_now / max(time.perf_counter() - stage1_started, 1e-9)
            slab_rate = len(slab_specs) / max(slab_elapsed_sec, 1e-9)
            print(
                f"[adev2] stage1 slab complete: index={slab_index} cumulative_completed={completed_now}/{total_candidates} slab_rate={slab_rate:.1f}/s overall_rate={overall_rate:.1f}/s slab_peak_rss_gb={slab_peak_rss / (1024 ** 3):.2f} slab_families=[{_format_family_counts(slab_family_counts)}] overall_families=[{_format_family_counts(stage1_family_counts)}]",
                flush=True,
            )
        offset += slab_count
        remaining_candidates = total_candidates - offset
        if remaining_candidates > 0 and candidate_slab_size is None and ram_target_gb is not None and stage1_rank_cap is None:
            slab_size = _retune_candidate_slab_size(
                current_slab_size=slab_count,
                actual_candidates=len(slab_specs),
                remaining_candidates=remaining_candidates,
                peak_rss_bytes=slab_peak_rss,
                ram_target_gb=ram_target_gb,
                verbose=verbose,
            )

    promoted = _promote_results(stage1_results, promote_top_k=min(promote_top_k, len(stage1_results)))
    if verbose:
        print(f"[adev2] promotion complete: promoted={len(promoted)} requested={promote_top_k}", flush=True)
    stage2_results: list[ADEV2Result] = []
    store.start_stage("stage2", len(promoted))
    completed_stage2 = len(store.load_completed_candidate_ids("stage2"))
    pending_promoted = [item for item in promoted if item.candidate_id not in store.load_completed_candidate_ids("stage2")]
    if verbose:
        print(
            f"[adev2] stage2 starting: pending_candidates={len(pending_promoted)} completed={completed_stage2}/{len(promoted)}",
            flush=True,
        )
    for item in pending_promoted:
        if stage2_time_budget_sec is not None and (time.perf_counter() - started) >= stage2_time_budget_sec:
            if verbose:
                print("[adev2] stage2 time budget reached; stopping promoted follow-up", flush=True)
            break
        result = evaluate_candidate(
            item.spec,
            rank_cap=stage2_rank_cap,
            rank_restarts=stage2_restarts,
            rank_als_iters=stage2_als_iters,
            rank_lbfgs_maxiter=stage2_lbfgs_maxiter,
            workers=workers,
        )
        stage2_results.append(result)
        completed_stage2 += 1
        store.append_result("stage2", completed_stage2, result.to_dict())
        if verbose:
            print(
                f"[adev2] stage2 progress: completed={completed_stage2}/{len(promoted)} candidate={result.candidate_id}",
                flush=True,
            )
    store.finalize_stage("stage2")
    stage1_payload = store.load_stage_results("stage1")
    stage2_payload = store.load_stage_results("stage2")

    payload = {
        "meta": {
            "mode": "staged",
            "elapsed_sec": time.perf_counter() - started,
            "promote_top_k": promote_top_k,
            "stage2_time_budget_sec": stage2_time_budget_sec,
            "max_dim": max_dim,
            "families": families,
            "limit": limit,
            "workers": workers,
            "candidate_workers": candidate_workers if candidate_workers is not None else workers,
            "peak_stage1_rss_gb": peak_stage1_rss_bytes / (1024 ** 3),
            "ram_target_gb": ram_target_gb,
            "stage1_family_counts": dict(stage1_family_counts),
        },
        "stage1": {
            "meta": _build_scan_meta(
                max_dim=max_dim,
                rank_cap=stage1_rank_cap,
                rank_restarts=stage1_restarts,
                rank_als_iters=stage1_als_iters,
                rank_lbfgs_maxiter=stage1_lbfgs_maxiter,
                completed=len(stage1_payload),
                total=total_candidates,
                elapsed_sec=time.perf_counter() - started,
                workers=workers,
                candidate_workers=effective_candidate_workers,
                candidate_slab_size=slab_size,
                families=families,
                limit=limit,
            ),
            "results": sorted(stage1_payload, key=lambda result: (-result["fringe_metrics"]["fringe_score"], result["dim"], result["label"])),
        },
        "stage2": {
            "meta": _build_scan_meta(
                max_dim=max_dim,
                rank_cap=stage2_rank_cap,
                rank_restarts=stage2_restarts,
                rank_als_iters=stage2_als_iters,
                rank_lbfgs_maxiter=stage2_lbfgs_maxiter,
                completed=len(stage2_payload),
                total=len(promoted),
                elapsed_sec=time.perf_counter() - started,
                workers=workers,
                candidate_workers=1,
                candidate_slab_size=slab_size,
                families=families,
                limit=limit,
            ),
            "promoted_candidate_ids": [item.candidate_id for item in promoted],
            "results": stage2_payload,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    store.finalize_run(status="completed")
    store.close()
    return payload