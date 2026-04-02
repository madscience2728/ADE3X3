#!/usr/bin/env python
"""GPU profiling script — find the sweet spot for GPU utilization.

Tests various batch sizes and operations to determine:
1. Max batch size for residual computation
2. Max batch size for Jacobian computation
3. Max batch size for smooth Adam descent
4. Memory usage at each level
5. Throughput (candidates/sec) at each level
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T_np

R, D = RANK, DIM
N_ENTRIES = D ** 3
N_PARAMS = 3 * R * D

device = torch.device("cuda", 0)
T_gpu = torch.tensor(T_np, dtype=torch.float64, device=device)
T_flat = T_gpu.reshape(-1)


def make_batch(n):
    return torch.randn(n, N_PARAMS, dtype=torch.float64, device=device) * 0.1


def residual_batch(xs):
    n = xs.shape[0]
    a = xs[:, :R*D].reshape(n, R, D)
    b = xs[:, R*D:2*R*D].reshape(n, R, D)
    g = xs[:, 2*R*D:].reshape(n, R, D)
    T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
    return T_hat.reshape(n, -1) - T_flat.unsqueeze(0)


def maxabs_batch(xs):
    return residual_batch(xs).abs().max(dim=1).values


def smooth_maxabs(xs, beta=200.0):
    res = residual_batch(xs)
    abs_res = res.abs()
    return (torch.logsumexp(beta * abs_res, dim=1) - np.log(N_ENTRIES)) / beta


def smooth_descent(xs, steps, lr=5e-5, beta=200.0):
    xs = xs.clone().requires_grad_(True)
    opt = torch.optim.Adam([xs], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = smooth_maxabs(xs, beta).sum()
        loss.backward()
        opt.step()
    return xs.detach()


def profile_op(name, fn, batch_sizes, reps=5):
    """Profile an operation at various batch sizes."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'Batch':>8}  {'Time(ms)':>10}  {'Cands/s':>10}  {'GPU Mem(MB)':>12}  {'GPU%':>6}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*6}")

    results = []
    for n in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6

        try:
            # Warmup
            xs = make_batch(n)
            fn(xs)
            torch.cuda.synchronize()

            # Timed
            times = []
            for _ in range(reps):
                xs = make_batch(n)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn(xs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            avg_ms = np.mean(times) * 1000
            cands_per_sec = n / np.mean(times)
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            
            # Estimate GPU utilization based on memory pressure
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
            mem_pct = peak_mem / total_mem * 100

            print(f"  {n:>8}  {avg_ms:>10.1f}  {cands_per_sec:>10.0f}  {peak_mem:>12.0f}  {mem_pct:>5.1f}%")
            results.append((n, avg_ms, cands_per_sec, peak_mem, mem_pct))

            del xs
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {n:>8}  {'OOM':>10}")
            del xs
            torch.cuda.empty_cache()
            break

    return results


def profile_smooth_descent(batch_sizes, steps=50, reps=3):
    """Profile smooth Adam descent (the main GPU workload)."""
    print(f"\n{'='*70}")
    print(f"  Smooth Adam Descent ({steps} steps)")
    print(f"{'='*70}")
    print(f"  {'Batch':>8}  {'Time(ms)':>10}  {'Cands/s':>10}  {'GPU Mem(MB)':>12}  {'GPU%':>6}  {'Fit improvd':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*6}  {'-'*12}")

    results = []
    for n in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Load the actual best solution as base
            import json
            best_path = Path(__file__).resolve().parent.parent / "slp_turbo_best.json"
            if best_path.exists():
                d = json.load(open(best_path))
                x0 = np.concatenate([
                    np.array(d["alpha"]).ravel(),
                    np.array(d["beta"]).ravel(),
                    np.array(d["gamma"]).ravel()
                ])
                base = torch.tensor(x0, dtype=torch.float64, device=device)
            else:
                base = torch.randn(N_PARAMS, dtype=torch.float64, device=device) * 0.1

            # Create candidates near base
            noise = torch.randn(n, N_PARAMS, dtype=torch.float64, device=device) * 0.001
            xs = base.unsqueeze(0) + noise

            # Measure initial fitness
            fit_before = maxabs_batch(xs).cpu().numpy()

            # Warmup
            _ = smooth_descent(xs[:min(4, n)], steps=2)
            torch.cuda.synchronize()

            # Timed
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            xs_out = smooth_descent(xs, steps=steps)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            fit_after = maxabs_batch(xs_out).cpu().numpy()
            improved = np.sum(fit_after < fit_before)

            avg_ms = dt * 1000
            cands_per_sec = n / dt
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
            mem_pct = peak_mem / total_mem * 100

            print(f"  {n:>8}  {avg_ms:>10.0f}  {cands_per_sec:>10.0f}  {peak_mem:>12.0f}  {mem_pct:>5.1f}%  {improved:>5}/{n}")
            results.append((n, avg_ms, cands_per_sec, peak_mem, mem_pct, improved))

            del xs, xs_out, noise
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {n:>8}  {'OOM':>10}")
            torch.cuda.empty_cache()
            break

    return results


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Tensor: {D}×{D}×{D} = {N_ENTRIES} entries, rank-{R}, {N_PARAMS} params")

    # Batch sizes to test
    small = [32, 64, 128, 256, 512, 1024, 2048]
    medium = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    large = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # 1. Residual only (lightweight)
    profile_op("Residual (einsum only)", residual_batch, large, reps=10)

    # 2. MaxAbs (residual + reduce)
    profile_op("MaxAbs (residual + abs + max)", maxabs_batch, large, reps=10)

    # 3. Smooth MaxAbs (logsumexp — forward only, no backward)
    profile_op("Smooth MaxAbs (logsumexp, no grad)", smooth_maxabs, medium, reps=10)

    # 4. Smooth descent with Adam (the real workload)
    for steps in [20, 50, 100]:
        profile_smooth_descent(small, steps=steps, reps=2)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print("  Target: ~90% GPU memory utilization during smooth descent")
    print("  Use the largest batch that fits + highest step count for throughput")
