"""Tensor rank search for matrix multiplication tensors — Parallel Edition.

Each restart runs ALS warm-start + L-BFGS-B in a worker process.
Dense print output so stalls are immediately visible at every stage.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class RankSearchResult:
    n: int
    rank_target: int
    rank_naive: int
    best_residual: float
    success: bool
    n_restarts: int
    elapsed_sec: float
    factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    algorithm_beats_naive: bool


def _reconstruct(U, V, W):
    """T_hat = sum_t u_t ⊗ v_t ⊗ w_t. U,V,W each (r,d)."""
    return np.einsum("ra,rb,rc->abc", U, V, W)


def _best_factors_path(n: int, rank: int) -> Path:
    return ROOT / "matmul_search" / "results" / f"factors_n{n}_r{rank}_best.npz"


def _save_best_factor_snapshot(params: np.ndarray, rank: int, shape: tuple[int, int, int], residual: float) -> Path:
    d1, d2, d3 = shape
    U = params[:rank * d1].reshape(rank, d1)
    V = params[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
    W = params[rank * (d1 + d2):].reshape(rank, d3)
    n = int(round(d1 ** 0.5))
    out_path = _best_factors_path(n, rank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.{os.getpid()}.{uuid.uuid4().hex}{out_path.suffix}.tmp")
    with open(tmp_path, "wb") as handle:
        np.savez(
            handle,
            U=U,
            V=V,
            W=W,
            n=np.array(n),
            rank=np.array(rank),
            residual=np.array(residual),
        )
    os.replace(tmp_path, out_path)
    return out_path


def _loss_and_grad(params, T, r):
    d1, d2, d3 = T.shape
    U = params[:r * d1].reshape(r, d1)
    V = params[r * d1:r * (d1 + d2)].reshape(r, d2)
    W = params[r * (d1 + d2):].reshape(r, d3)
    R = _reconstruct(U, V, W) - T
    loss = 0.5 * float(np.dot(R.ravel(), R.ravel()))
    dU = np.einsum("abc,rb,rc->ra", R, V, W)
    dV = np.einsum("abc,ra,rc->rb", R, U, W)
    dW = np.einsum("abc,ra,rb->rc", R, U, V)
    return loss, np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])


def _als_step(T, U, V, W):
    """One ALS round. All factors (r,d)."""
    r = U.shape[0]
    d1, d2, d3 = T.shape
    reg = 1e-10 * np.eye(r)

    def _update(T_mode, F1, F2):
        KR = np.einsum("ra,rb->rab", F1, F2).reshape(r, -1)
        gram = KR @ KR.T + reg
        rhs = T_mode @ KR.T
        return np.linalg.lstsq(gram, rhs.T, rcond=None)[0]

    T1 = T.reshape(d1, d2 * d3)
    U = _update(T1, V, W)

    T2 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
    V = _update(T2, U, W)

    T3 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)
    W = _update(T3, U, V)

    return U, V, W


def _single_restart(args):
    T_flat, shape, rank, als_iters, lbfgs_maxiter, seed, idx = args
    pid = os.getpid()
    t0 = time.time()
    T = T_flat.reshape(shape)
    d1, d2, d3 = shape
    rng = np.random.default_rng(seed)
    scale = max((float(np.linalg.norm(T)) / rank) ** (1 / 3), 0.1)

    U = rng.standard_normal((rank, d1)) * scale
    V = rng.standard_normal((rank, d2)) * scale
    W = rng.standard_normal((rank, d3)) * scale

    print(
        f"  [PID {pid}] restart {idx + 1:3d} | init_scale={scale:.4f} | "
        f"ALS starting ({als_iters} iters)...",
        flush=True,
    )

    for it in range(als_iters):
        try:
            U, V, W = _als_step(T, U, V, W)
        except Exception as exc:
            print(f"  [PID {pid}] restart {idx + 1:3d} | ALS iter {it} CRASH: {exc}", flush=True)
            break
        if it % 100 == 0 or it == als_iters - 1:
            res = float(np.linalg.norm(T - _reconstruct(U, V, W)))
            print(f"  [PID {pid}] restart {idx + 1:3d} | ALS iter {it:4d} | res={res:.4e}", flush=True)
            if res < 1e-6:
                print(f"  [PID {pid}] restart {idx + 1:3d} | ALS early-converge", flush=True)
                break

    pre_res = float(np.linalg.norm(T - _reconstruct(U, V, W)))
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    print(
        f"  [PID {pid}] restart {idx + 1:3d} | ALS done | pre-LBFGS res={pre_res:.4e} | "
        f"L-BFGS-B starting (maxiter={lbfgs_maxiter})...",
        flush=True,
    )

    iters = [0]

    def cb(xk):
        iters[0] += 1
        if iters[0] % 200 == 0:
            r_ = rank
            U_ = xk[:r_ * d1].reshape(r_, d1)
            V_ = xk[r_ * d1:r_ * (d1 + d2)].reshape(r_, d2)
            W_ = xk[r_ * (d1 + d2):].reshape(r_, d3)
            res_ = float(np.linalg.norm(T - _reconstruct(U_, V_, W_)))
            print(f"  [PID {pid}] restart {idx + 1:3d} | LBFGS iter {iters[0]:5d} | res={res_:.4e}", flush=True)

    try:
        opt = minimize(
            _loss_and_grad,
            x0,
            args=(T, rank),
            method="L-BFGS-B",
            jac=True,
            callback=cb,
            options={"maxiter": lbfgs_maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        residual = float(np.sqrt(max(2.0 * opt.fun, 0.0)))
        params = opt.x
        msg = getattr(opt, "message", str(getattr(opt, "status", "")))
    except Exception as exc:
        print(f"  [PID {pid}] restart {idx + 1:3d} | L-BFGS-B CRASH: {exc}", flush=True)
        traceback.print_exc()
        residual = pre_res
        params = x0
        msg = f"EXCEPTION: {exc}"

    elapsed = time.time() - t0
    print(
        f"  [PID {pid}] restart {idx + 1:3d} | DONE | res={residual:.4e} | "
        f"elapsed={elapsed:.1f}s | {msg}",
        flush=True,
    )
    return {
        "idx": idx,
        "residual": residual,
        "params": params.tolist(),
        "elapsed": elapsed,
        "pid": pid,
        "lbfgs_iters": iters[0],
    }


def _structured_restart(args):
    T_flat, shape, U_init, V_init, W_init, desc, del_res, idx, rank, lbfgs_maxiter = args
    pid = os.getpid()
    t0 = time.time()
    T = T_flat.reshape(shape)
    d1, d2, d3 = shape

    pre_res = float(np.linalg.norm(T - _reconstruct(U_init, V_init, W_init)))
    print(
        f"  [PID {pid}] struct restart {idx + 1:3d} | {desc[:40]} | "
        f"del_res={del_res:.4e} | actual_start_res={pre_res:.4e}",
        flush=True,
    )

    x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
    iters = [0]

    def cb(xk):
        iters[0] += 1
        if iters[0] % 500 == 0:
            U_ = xk[:rank * d1].reshape(rank, d1)
            V_ = xk[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
            W_ = xk[rank * (d1 + d2):].reshape(rank, d3)
            res_ = float(np.linalg.norm(T - _reconstruct(U_, V_, W_)))
            print(f"  [PID {pid}] struct {idx + 1:3d} | LBFGS {iters[0]:5d} | res={res_:.4e}", flush=True)

    try:
        opt = minimize(
            _loss_and_grad,
            x0,
            args=(T, rank),
            method="L-BFGS-B",
            jac=True,
            callback=cb,
            options={"maxiter": lbfgs_maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        residual = float(np.sqrt(max(2.0 * opt.fun, 0.0)))
        params = opt.x
        msg = getattr(opt, "message", str(getattr(opt, "status", "")))
    except Exception as exc:
        residual = pre_res
        params = x0
        msg = f"EXCEPTION: {exc}"

    elapsed = time.time() - t0
    print(
        f"  [PID {pid}] struct {idx + 1:3d} | DONE | res={residual:.4e} | "
        f"elapsed={elapsed:.1f}s | {msg}",
        flush=True,
    )
    return {
        "idx": idx,
        "residual": residual,
        "params": params.tolist(),
        "elapsed": elapsed,
        "pid": pid,
        "desc": desc,
    }


def search_rank_from_init(
    T: np.ndarray,
    rank: int,
    U_init: np.ndarray,
    V_init: np.ndarray,
    W_init: np.ndarray,
    n_restarts: int = 20,
    lbfgs_maxiter: int = 2000,
    tol: float = 1e-5,
    rng_seed: int = 42,
    max_workers: int | None = None,
    callback: Callable | None = None,
) -> RankSearchResult:
    d1, d2, d3 = T.shape
    n_workers = max_workers or os.cpu_count() or 4
    T_flat = T.ravel()
    t_start = time.time()

    init_result = _structured_restart(
        (T_flat, T.shape, U_init, V_init, W_init, "structured_init", 0.0, 0, rank, lbfgs_maxiter)
    )
    best_residual = float(init_result["residual"])
    best_params = np.array(init_result["params"])
    completed = 1

    print(
        f"\n  [STRUCT_INIT_SEARCH] seeded run complete | best_res={best_residual:.4e} | "
        f"remaining_random_restarts={max(n_restarts - 1, 0)}",
        flush=True,
    )
    best_path = _save_best_factor_snapshot(best_params, rank, T.shape, best_residual)
    print(f"  [STRUCT_INIT_SEARCH] best factors saved -> {best_path}", flush=True)

    if callback:
        should_stop = callback(0, best_residual, best_params.copy())
        if should_stop:
            elapsed = time.time() - t_start
            U = best_params[:rank * d1].reshape(rank, d1)
            V = best_params[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
            W = best_params[rank * (d1 + d2):].reshape(rank, d3)
            naive = int(round(d1 ** 1.5))
            return RankSearchResult(
                n=int(round(d1 ** 0.5)),
                rank_target=rank,
                rank_naive=naive,
                best_residual=best_residual,
                success=best_residual < tol,
                n_restarts=completed,
                elapsed_sec=elapsed,
                factors=(U, V, W),
                algorithm_beats_naive=(best_residual < tol and rank < naive),
            )

    if best_residual >= tol and n_restarts > 1:
        jobs = [(T_flat, T.shape, rank, 0, lbfgs_maxiter, rng_seed + i, i + 1) for i in range(n_restarts - 1)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fmap = {ex.submit(_single_restart, job): job[-1] for job in jobs}
            print(f"  [STRUCT_INIT_SEARCH] {len(jobs)} random fallback futures submitted\n", flush=True)
            for fut in as_completed(fmap):
                idx = fmap[fut]
                completed += 1
                try:
                    result = fut.result()
                    residual = float(result["residual"])
                    print(
                        f"  [STRUCT_INIT_SEARCH] restart {idx + 1:3d} returned | res={residual:.4e} | "
                        f"done={completed}/{n_restarts}",
                        flush=True,
                    )
                    if residual < best_residual:
                        best_residual = residual
                        best_params = np.array(result["params"])
                        best_path = _save_best_factor_snapshot(best_params, rank, T.shape, best_residual)
                        print(f"  [STRUCT_INIT_SEARCH] *** NEW BEST {best_residual:.4e} ***", flush=True)
                        print(f"  [STRUCT_INIT_SEARCH] best factors saved -> {best_path}", flush=True)
                        if callback:
                            should_stop = callback(idx, residual, best_params.copy())
                            if should_stop:
                                for pending in fmap:
                                    pending.cancel()
                                break
                    if best_residual < tol:
                        for pending in fmap:
                            pending.cancel()
                        break
                except Exception as exc:
                    print(f"  [STRUCT_INIT_SEARCH] restart {idx + 1} FUTURE EXCEPTION: {exc}", flush=True)

    elapsed = time.time() - t_start
    success = best_residual < tol
    U = best_params[:rank * d1].reshape(rank, d1)
    V = best_params[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
    W = best_params[rank * (d1 + d2):].reshape(rank, d3)
    naive = int(round(d1 ** 1.5))
    return RankSearchResult(
        n=int(round(d1 ** 0.5)),
        rank_target=rank,
        rank_naive=naive,
        best_residual=best_residual,
        success=success,
        n_restarts=completed,
        elapsed_sec=elapsed,
        factors=(U, V, W),
        algorithm_beats_naive=success and rank < naive,
    )


def search_rank(
    T: np.ndarray,
    rank: int,
    n_restarts: int = 20,
    als_iters: int = 400,
    lbfgs_maxiter: int = 2000,
    tol: float = 1e-5,
    rng_seed: int = 42,
    max_workers: int | None = None,
    verbose: bool = True,
    callback: Callable | None = None,
) -> RankSearchResult:
    d1, d2, d3 = T.shape
    n_workers = max_workers or os.cpu_count() or 4
    T_flat = T.ravel()

    best_residual = np.inf
    best_params = None
    t_start = time.time()
    completed = 0

    print(
        f"\n  [COORDINATOR] {n_restarts} restarts | {n_workers} workers | "
        f"rank={rank} | shape={T.shape} | "
        f"params/restart={rank * (d1 + d2 + d3)} | tol={tol:.0e}",
        flush=True,
    )

    jobs = [(T_flat, T.shape, rank, als_iters, lbfgs_maxiter, rng_seed + i, i) for i in range(n_restarts)]

    # On Windows, ProcessPoolExecutor pipe handles are inherited by other
    # multiprocessing pools (e.g. the CRT search), causing OSError: handle is
    # closed when both run simultaneously.  This is a Python 3.11 bug fixed in
    # 3.12.  ThreadPoolExecutor is safe because numpy/scipy release the GIL
    # for their C extensions, giving similar parallel throughput without the
    # handle-inheritance issue.
    # Set RANK_FORCE_PROCESSES=1 to override on Windows machines that are NOT
    # running a CRT pool simultaneously (avoids MKL thread-contention with
    # ThreadPoolExecutor which tanks CPU utilisation on many-core workstations).
    _force_proc = os.environ.get("RANK_FORCE_PROCESSES", "0").strip() == "1"
    ExecutorClass = ProcessPoolExecutor if (_force_proc or sys.platform != "win32") else ThreadPoolExecutor
    with ExecutorClass(max_workers=n_workers) as ex:
        fmap = {ex.submit(_single_restart, j): j[-1] for j in jobs}
        print(f"  [COORDINATOR] {n_restarts} futures submitted — waiting...\n", flush=True)

        for fut in as_completed(fmap):
            idx = fmap[fut]
            completed += 1
            wall = time.time() - t_start
            try:
                r = fut.result()
                res = r["residual"]
                print(
                    f"  [COORDINATOR] restart {idx + 1:3d} returned | res={res:.4e} | "
                    f"pid={r['pid']} | restart_wall={r['elapsed']:.1f}s | "
                    f"total_wall={wall:.1f}s | done={completed}/{n_restarts}",
                    flush=True,
                )

                if res < best_residual:
                    best_residual = res
                    best_params = np.array(r["params"])
                    best_path = _save_best_factor_snapshot(best_params, rank, T.shape, best_residual)
                    print(f"  [COORDINATOR] *** NEW BEST {best_residual:.4e} (restart {idx + 1}) ***", flush=True)
                    print(f"  [COORDINATOR] best factors saved -> {best_path}", flush=True)

                    if callback:
                        should_stop = callback(idx, res, best_params.copy())
                        if should_stop:
                            print("  [COORDINATOR] stop requested by callback — cancelling remaining futures", flush=True)
                            for f in fmap:
                                f.cancel()
                            break

                if best_residual < tol:
                    print("  [COORDINATOR] ✓ SOLVED — cancelling remaining futures", flush=True)
                    for f in fmap:
                        f.cancel()
                    break

            except Exception as exc:
                print(f"  [COORDINATOR] restart {idx + 1} FUTURE EXCEPTION: {exc}", flush=True)

    elapsed = time.time() - t_start
    success = best_residual < tol
    print(
        f"\n  [COORDINATOR] COMPLETE | best_res={best_residual:.4e} | "
        f"success={success} | wall={elapsed:.1f}s | {completed}/{n_restarts} finished",
        flush=True,
    )

    factors = None
    if best_params is not None:
        U = best_params[:rank * d1].reshape(rank, d1)
        V = best_params[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
        W = best_params[rank * (d1 + d2):].reshape(rank, d3)
        factors = (U, V, W)

    naive = int(round(d1 ** 1.5))
    return RankSearchResult(
        n=int(round(d1 ** 0.5)),
        rank_target=rank,
        rank_naive=naive,
        best_residual=best_residual,
        success=success,
        n_restarts=n_restarts,
        elapsed_sec=elapsed,
        factors=factors,
        algorithm_beats_naive=success and rank < naive,
    )


def extract_algorithm(factors, n, tol=1e-4):
    U, V, W = factors
    r = U.shape[0]
    mults = []
    for t in range(r):
        u = U[t].reshape(n, n).copy()
        u[np.abs(u) < tol] = 0
        v = V[t].reshape(n, n).copy()
        v[np.abs(v) < tol] = 0
        w = W[t].reshape(n, n).copy()
        w[np.abs(w) < tol] = 0
        mults.append({"t": t + 1, "A_coeffs": u.tolist(), "B_coeffs": v.tolist(), "C_update": w.tolist()})
    return {"n": n, "rank": r, "multiplications": mults}


def search_rank_structured(
    T: np.ndarray,
    U_base: np.ndarray,
    V_base: np.ndarray,
    W_base: np.ndarray,
    rank: int,
    n_noise_variants: int = 5,
    noise_eps: float = 0.01,
    lbfgs_maxiter: int = 10000,
    tol: float = 1e-5,
    max_workers: int | None = None,
    rng_seed: int = 42,
) -> RankSearchResult:
    from matmul_search.structured_init import deletion_inits

    d1, d2, d3 = T.shape
    n_workers = max_workers or os.cpu_count() or 4
    T_flat = T.ravel()

    inits = list(
        deletion_inits(
            T,
            U_base,
            V_base,
            W_base,
            n_noise_variants=n_noise_variants,
            noise_eps=noise_eps,
            rng_seed=rng_seed,
        )
    )

    n_restarts = len(inits)
    print(
        f"\n  [STRUCT_SEARCH] {n_restarts} structured inits "
        f"({U_base.shape[0]} deletions x {1 + n_noise_variants} variants)",
        flush=True,
    )
    print(f"  [STRUCT_SEARCH] {n_workers} workers | rank={rank} | tol={tol:.0e}", flush=True)

    best_residual = np.inf
    best_params = None
    t_start = time.time()
    completed = 0

    jobs = [
        (T_flat, T.shape, U_init, V_init, W_init, desc, del_res, idx, rank, lbfgs_maxiter)
        for idx, (U_init, V_init, W_init, desc, del_res) in enumerate(inits)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fmap = {ex.submit(_structured_restart, job): job[7] for job in jobs}
        print(f"  [STRUCT_SEARCH] {n_restarts} futures submitted\n", flush=True)

        for fut in as_completed(fmap):
            idx = fmap[fut]
            completed += 1
            wall = time.time() - t_start
            try:
                result = fut.result()
                residual = result["residual"]
                print(
                    f"  [STRUCT_SEARCH] restart {idx + 1:3d} returned | "
                    f"res={residual:.4e} | wall={wall:.1f}s | {completed}/{n_restarts}",
                    flush=True,
                )

                if residual < best_residual:
                    best_residual = residual
                    best_params = np.array(result["params"])
                    best_path = _save_best_factor_snapshot(best_params, rank, T.shape, best_residual)
                    print(f"  [STRUCT_SEARCH] *** NEW BEST {best_residual:.4e} ***", flush=True)
                    print(f"  [STRUCT_SEARCH] best factors saved -> {best_path}", flush=True)

                if best_residual < tol:
                    print("  [STRUCT_SEARCH] ✓ SOLVED | cancelling remaining", flush=True)
                    for pending in fmap:
                        pending.cancel()
                    break
            except Exception as exc:
                print(f"  [STRUCT_SEARCH] restart {idx + 1} EXCEPTION: {exc}", flush=True)

    elapsed = time.time() - t_start
    success = best_residual < tol
    print(
        f"\n  [STRUCT_SEARCH] COMPLETE | best_res={best_residual:.4e} | "
        f"success={success} | wall={elapsed:.1f}s",
        flush=True,
    )

    factors = None
    if best_params is not None:
        U_f = best_params[:rank * d1].reshape(rank, d1)
        V_f = best_params[rank * d1:rank * (d1 + d2)].reshape(rank, d2)
        W_f = best_params[rank * (d1 + d2):].reshape(rank, d3)
        factors = (U_f, V_f, W_f)

    naive = int(round(d1 ** 1.5))
    return RankSearchResult(
        n=int(round(d1 ** 0.5)),
        rank_target=rank,
        rank_naive=naive,
        best_residual=best_residual,
        success=success,
        n_restarts=n_restarts,
        elapsed_sec=elapsed,
        factors=factors,
        algorithm_beats_naive=success and rank < naive,
    )