#!/usr/bin/env python
"""GPU-accelerated SLP Turbo — Maximum throughput batched minimax optimizer.

Architecture (two-tier concurrent pipeline):
  Tier 1 — GPU smooth descent (runs WHILE CPU does LP):
    Log-sum-exp smooth minimax + Adam on 512 candidates simultaneously
    Explores the landscape broadly, feeds improved candidates to Tier 2
  Tier 2 — CPU exact LP (SLP):
    Active-set LP on top candidates, trust-region managed
    Precise minimax descent on the most promising candidates

Pipeline overlap per iteration:
  1. GPU: compute residuals + Jacobians for SLP workers → transfer to CPU
  2. CONCURRENT:
     a. CPU: 22-process pool solves N LPs
     b. GPU: runs K steps of smooth Adam descent on 512 exploration candidates
  3. GPU: verify LP candidates, merge best GPU-explored candidates
  4. Trust-region update, repeat
"""

import argparse
import json
import sys
import time
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from db_optimizer.config import RANK, DIM, TARGET_TENSOR as T_np

R, D = RANK, DIM
N_ENTRIES = D ** 3   # 729
N_PARAMS = 3 * R * D  # 513

console = Console()

# ── I/O ──────────────────────────────────────────────────────────────

def load_factors(path):
    with open(path) as f:
        data = json.load(f)
    if "alpha" in data:
        a = np.array(data["alpha"], dtype=np.float64)
        b = np.array(data["beta"], dtype=np.float64)
        g = np.array(data["gamma"], dtype=np.float64)
        fit = data.get("fitness", None)
    else:
        from db_optimizer.blob import factors_from_json
        a, b, g = factors_from_json(data)
        fit = data.get("fitness", None)
    return a, b, g, fit


def save_factors(alpha, beta, gamma, fit, path):
    data = {"fitness": float(fit), "rank": R, "dim": D,
            "alpha": alpha.tolist(), "beta": beta.tolist(), "gamma": gamma.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def pack(a, b, g):
    return np.concatenate([a.ravel(), b.ravel(), g.ravel()])


def unpack(x):
    return (x[:R*D].reshape(R, D),
            x[R*D:2*R*D].reshape(R, D),
            x[2*R*D:].reshape(R, D))


# ── GPU kernels ──────────────────────────────────────────────────────

class GPUBatchSLP:
    """Batched residual + Jacobian on GPU. Pre-allocates all buffers."""

    def __init__(self, device, max_workers):
        self.device = device
        self.max_workers = max_workers
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.T_flat = self.T.reshape(-1)

        # Pre-allocate GPU buffers for max_workers
        self._J = torch.zeros(max_workers, N_ENTRIES, N_PARAMS,
                              dtype=torch.float64, device=device)
        self._res = torch.zeros(max_workers, N_ENTRIES,
                                dtype=torch.float64, device=device)

        # Pre-allocate pinned CPU buffers for async transfer
        self._res_pin = torch.zeros(max_workers, N_ENTRIES,
                                    dtype=torch.float64).pin_memory()
        self._J_pin = torch.zeros(max_workers, N_ENTRIES, N_PARAMS,
                                  dtype=torch.float64).pin_memory()

        # CUDA stream for async transfer
        self._transfer_stream = (torch.cuda.Stream(device)
                                 if device.type == 'cuda' else None)

        # Precompute index arrays for Jacobian scatter
        ai_range = torch.arange(D, device=device)
        self._alpha_row_idx = []
        self._alpha_col_idx = []
        for ai in range(D):
            self._alpha_row_idx.append(
                torch.arange(ai * D * D, (ai + 1) * D * D, device=device))
            self._alpha_col_idx.append(
                torch.arange(ai, R * D, D, device=device))

        self._beta_row_idx = []
        self._beta_col_idx = []
        for bi in range(D):
            self._beta_row_idx.append(
                (ai_range[:, None] * D * D + bi * D + ai_range[None, :]).reshape(-1))
            self._beta_col_idx.append(R * D + torch.arange(R, device=device) * D + bi)

        self._gamma_row_idx = []
        self._gamma_col_idx = []
        for ci in range(D):
            self._gamma_row_idx.append(
                (ai_range[:, None] * D * D + ai_range[None, :] * D + ci).reshape(-1))
            self._gamma_col_idx.append(2 * R * D + torch.arange(R, device=device) * D + ci)

    def residual_batch(self, xs_gpu, n=None):
        """Compute residuals in-place. xs_gpu: (N, 513) on device. Returns (N, 729) view."""
        if n is None:
            n = xs_gpu.shape[0]
        a = xs_gpu[:n, :R*D].reshape(n, R, D)
        b = xs_gpu[:n, R*D:2*R*D].reshape(n, R, D)
        g = xs_gpu[:n, 2*R*D:].reshape(n, R, D)
        T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
        self._res[:n] = T_hat.reshape(n, -1) - self.T_flat.unsqueeze(0)
        return self._res[:n]

    def maxabs_batch(self, xs_gpu, n=None):
        """Compute max|residual| for arbitrary batch. Does NOT use pre-allocated buffers."""
        if n is None:
            n = xs_gpu.shape[0]
        a = xs_gpu[:n, :R*D].reshape(n, R, D)
        b = xs_gpu[:n, R*D:2*R*D].reshape(n, R, D)
        g = xs_gpu[:n, 2*R*D:].reshape(n, R, D)
        T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
        res = T_hat.reshape(n, -1) - self.T_flat.unsqueeze(0)
        return res.abs().max(dim=1).values

    def jacobian_batch(self, xs_gpu, n=None):
        """Compute Jacobians in-place. Returns (N, 729, 513) view."""
        if n is None:
            n = xs_gpu.shape[0]
        a = xs_gpu[:n, :R*D].reshape(n, R, D)
        b = xs_gpu[:n, R*D:2*R*D].reshape(n, R, D)
        g = xs_gpu[:n, 2*R*D:].reshape(n, R, D)

        J = self._J[:n]
        J.zero_()

        bg = torch.einsum('nkb,nkc->nkbc', b, g).reshape(n, R, D*D)
        for ai in range(D):
            rows = self._alpha_row_idx[ai]
            cols = self._alpha_col_idx[ai]
            J[:, rows[:, None], cols[None, :]] += bg.transpose(1, 2)

        ag = torch.einsum('nka,nkc->nkac', a, g).reshape(n, R, D*D)
        for bi in range(D):
            rows = self._beta_row_idx[bi]
            cols = self._beta_col_idx[bi]
            J[:, rows[:, None], cols[None, :]] += ag.transpose(1, 2)

        ab = torch.einsum('nka,nkb->nkab', a, b).reshape(n, R, D*D)
        for ci in range(D):
            rows = self._gamma_row_idx[ci]
            cols = self._gamma_col_idx[ci]
            J[:, rows[:, None], cols[None, :]] += ab.transpose(1, 2)

        return J

    def compute_and_transfer(self, xs_gpu, n):
        """GPU compute + async pinned-memory transfer. Returns (res_np, J_np)."""
        # Compute on GPU
        self.residual_batch(xs_gpu, n)
        self.jacobian_batch(xs_gpu, n)

        # Async copy to pinned memory
        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                self._res_pin[:n].copy_(self._res[:n])
                self._J_pin[:n].copy_(self._J[:n])
            self._transfer_stream.synchronize()
        else:
            self._res_pin[:n].copy_(self._res[:n])
            self._J_pin[:n].copy_(self._J[:n])

        return self._res_pin[:n].numpy(), self._J_pin[:n].numpy()


# ── GPU Smooth Minimax Engine (runs concurrently with CPU LP) ────────

class GPUSmoothDescent:
    """Batched log-sum-exp smooth minimax + Adam on GPU.

    Optimizes smooth_max(|res|) ≈ max(|res|) using differentiable surrogate.
    Runs on a large pool of candidates to explore broadly while CPU does LP.
    """

    def __init__(self, device, n_candidates, T_tensor):
        self.device = device
        self.n_candidates = n_candidates
        self.T = T_tensor  # (9,9,9) on device
        self.T_flat = T_tensor.reshape(-1)

    @torch.no_grad()
    def _maxabs_batch(self, xs):
        """Non-differentiable true maxabs for tracking."""
        n = xs.shape[0]
        a = xs[:, :R*D].reshape(n, R, D)
        b = xs[:, R*D:2*R*D].reshape(n, R, D)
        g = xs[:, 2*R*D:].reshape(n, R, D)
        T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
        return (T_hat.reshape(n, -1) - self.T_flat).abs().max(dim=1).values

    def smooth_maxabs(self, xs, beta=200.0):
        """Differentiable smooth-max approximation: log-sum-exp of |residuals|.

        beta controls sharpness: higher = closer to true max but harder gradients.
        """
        n = xs.shape[0]
        a = xs[:, :R*D].reshape(n, R, D)
        b = xs[:, R*D:2*R*D].reshape(n, R, D)
        g = xs[:, 2*R*D:].reshape(n, R, D)
        T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
        res = (T_hat.reshape(n, -1) - self.T_flat)
        abs_res = res.abs()
        # log-sum-exp: log(mean(exp(beta * |r|))) / beta
        return (torch.logsumexp(beta * abs_res, dim=1) - np.log(N_ENTRIES)) / beta

    def descend(self, seeds_gpu, steps=30, lr=1e-4, beta=200.0):
        """Run Adam descent on smooth surrogate. Returns improved xs and true maxabs fits.

        seeds_gpu: (N, 513) float64 on device
        Returns: (xs_improved, fits) both on device
        """
        n = seeds_gpu.shape[0]
        xs = seeds_gpu.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([xs], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.smooth_maxabs(xs, beta=beta)
            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()

            # Increase beta over steps for sharper approximation
            if step == steps // 2:
                beta = min(beta * 2, 1000.0)

        xs_out = xs.detach()
        fits = self._maxabs_batch(xs_out)
        return xs_out, fits


# ── CPU LP solver (multiprocessing-compatible) ──────────────────────

def _lp_worker_solve(args_tuple):
    """Top-level function for multiprocessing — solves one LP."""
    res_np, J_np, trust, active_k = args_tuple
    return solve_lp_active(res_np, J_np, trust, active_k)


def solve_lp_active(res_np, J_np, trust, active_k=500):
    """Active-set LP on CPU. Returns (dx, t_pred) or (None, None)."""
    abs_res = np.abs(res_np)
    active_idx = np.argpartition(abs_res, -active_k)[-active_k:]

    dx, t_pred = _solve_lp(res_np, J_np, trust, active_idx)
    if dx is None:
        return None, None

    # Verify active set sufficiency
    new_res_lin = res_np + J_np @ dx
    if np.max(np.abs(new_res_lin)) <= t_pred * 1.001:
        return dx, t_pred

    # Fallback to full LP
    return _solve_lp(res_np, J_np, trust, np.arange(N_ENTRIES))


def _solve_lp(res_np, J_np, trust, idx):
    n_active = len(idx)
    J_sub = J_np[idx]
    r_sub = res_np[idx]

    ones_col = -np.ones((n_active, 1))
    A_ub = np.vstack([
        np.hstack([J_sub, ones_col]),
        np.hstack([-J_sub, ones_col])
    ])
    b_ub = np.concatenate([-r_sub, r_sub])

    c_obj = np.zeros(N_PARAMS + 1)
    c_obj[N_PARAMS] = 1.0

    bounds = [(-trust, trust)] * N_PARAMS + [(0, None)]

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'time_limit': 30})

    if result.success:
        return result.x[:N_PARAMS], result.x[N_PARAMS]
    return None, None


# ── Perturbation ─────────────────────────────────────────────────────

def make_seeds(x0, n_workers, scale, rng):
    seeds = [x0.copy()]
    for i in range(1, n_workers):
        if i % 3 == 0:
            # Targeted: perturb params most sensitive at worst entries
            a, b, g = unpack(x0)
            res = (np.einsum('ra,rb,rc->abc', a, b, g) - T_np).ravel()
            abs_res = np.abs(res)
            worst = np.argsort(abs_res)[-20:]
            J_local = np.zeros((len(worst), N_PARAMS))
            bg = np.einsum('kb,kc->kbc', b, g).reshape(R, D*D)
            ag = np.einsum('ka,kc->kac', a, g).reshape(R, D*D)
            ab = np.einsum('ka,kb->kab', a, b).reshape(R, D*D)
            # Simplified sensitivity: sum |J| over worst entries
            sensitivity = np.zeros(N_PARAMS)
            for wi, entry in enumerate(worst):
                ai_e, bc = divmod(entry, D*D)
                bi_e, ci_e = divmod(bc, D)
                for k in range(R):
                    sensitivity[k*D + ai_e] += abs(b[k, bi_e] * g[k, ci_e])
                    sensitivity[R*D + k*D + bi_e] += abs(a[k, ai_e] * g[k, ci_e])
                    sensitivity[2*R*D + k*D + ci_e] += abs(a[k, ai_e] * b[k, bi_e])
            sensitivity /= sensitivity.max() + 1e-30
            noise = rng.normal(0, scale, size=x0.shape) * sensitivity
            seeds.append(x0 + noise)
        elif i % 3 == 1:
            seeds.append(x0 + rng.normal(0, scale * 0.5, size=x0.shape))
        else:
            seeds.append(x0 + rng.normal(0, scale * 2.0, size=x0.shape))
    return seeds


# ── Plateau detection ────────────────────────────────────────────────

class PlateauDetector:
    """Track fitness history and detect stagnation."""

    def __init__(self, window=50, threshold=1e-9, warn_after=100):
        self.history = []          # (iter, best_fitness)
        self.window = window
        self.threshold = threshold
        self.warn_after = warn_after
        self.warnings = []
        self.plateau_since = None

    def update(self, iteration, best_fit):
        self.history.append((iteration, best_fit))
        if len(self.history) < self.window:
            return None

        recent = self.history[-self.window:]
        improvement = recent[0][1] - recent[-1][1]

        if improvement < self.threshold:
            if self.plateau_since is None:
                self.plateau_since = recent[0][0]
            plateau_len = iteration - self.plateau_since
            if plateau_len >= self.warn_after:
                msg = (f"PLATEAU DETECTED: no improvement > {self.threshold:.0e} "
                       f"in last {plateau_len} iters (fit={best_fit:.10f})")
                if not self.warnings or self.warnings[-1] != msg:
                    self.warnings.append(msg)
                return msg
        else:
            self.plateau_since = None
        return None

    @property
    def is_plateau(self):
        if self.plateau_since is None:
            return False
        if not self.history:
            return False
        return (self.history[-1][0] - self.plateau_since) >= self.warn_after


# ── Main SLP loop ────────────────────────────────────────────────────

def build_status_table(
    rnd, n_rounds, iteration, max_iters, workers_fits, workers_accepts,
    workers_trusts, best_fit, best_worker, elapsed, plateau_det, lp_time,
    start_fit, gpu_explore_inj=0, gpu_explore_imp=0
):
    """Build a rich Table showing current state."""
    table = Table(title=f"GPU SLP Turbo — Round {rnd}/{n_rounds}", expand=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    improvement = start_fit - best_fit
    table.add_row("Iteration", f"{iteration}/{max_iters}")
    table.add_row("Best Fitness", f"[bold green]{best_fit:.10f}[/]")
    table.add_row("Start Fitness", f"{start_fit:.10f}")
    table.add_row("Improvement", f"[cyan]{improvement:.2e}[/]" if improvement > 0 else "[red]none[/]")
    table.add_row("Best Worker", f"W{best_worker:02d}")
    table.add_row("Elapsed", f"{elapsed:.0f}s")
    table.add_row("LP solve time", f"{lp_time:.1f}s (last batch)")

    # Worker summary
    active = sum(1 for t in workers_trusts if t > 1e-8)
    total_accepts = sum(workers_accepts)
    table.add_row("Active workers", f"{active}/{len(workers_fits)}")
    table.add_row("Total accepts", f"{total_accepts}")

    # GPU exploration
    if gpu_explore_inj > 0 or gpu_explore_imp > 0:
        table.add_row("GPU explore", f"[magenta]{gpu_explore_inj} injected, {gpu_explore_imp} improved best[/]")

    # Top 5 workers
    sorted_idx = sorted(range(len(workers_fits)), key=lambda i: workers_fits[i])
    top5 = ", ".join(f"W{i:02d}={workers_fits[i]:.8f}" for i in sorted_idx[:5])
    table.add_row("Top 5", top5)

    # Plateau warning
    if plateau_det.is_plateau:
        table.add_row("[bold red]⚠ PLATEAU[/]",
                       f"[bold red]{plateau_det.warnings[-1] if plateau_det.warnings else 'stagnating'}[/]")

    return table


def run_slp_turbo(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[bold green]GPU:[/] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        console.print("[bold yellow]WARNING: No CUDA GPU — running on CPU[/]")

    # Check float64 support
    test = torch.zeros(1, dtype=torch.float64, device=device)
    console.print(f"[bold]float64 on {device}:[/] OK")

    # Load input
    src = Path(args.input)
    if not src.exists():
        src = Path(__file__).resolve().parent / args.input
    a, b, g, loaded_fit = load_factors(src)
    x0 = pack(a, b, g)
    verified_fit = float(np.max(np.abs(np.einsum('ra,rb,rc->abc', a, b, g) - T_np)))

    console.print(f"[bold]Loaded:[/] {src}")
    console.print(f"[bold]Fitness:[/] {verified_fit:.10f}  (file says {loaded_fit})")
    console.print(f"[bold]Workers:[/] {args.workers}  |  [bold]Iters:[/] {args.max_iters}  |  "
                  f"[bold]Active-K:[/] {args.active_k}  |  [bold]Rounds:[/] {args.rounds}")
    console.print(f"[bold]GPU Explore:[/] {args.gpu_explore} candidates × {args.gpu_smooth_steps} steps "
                  f"every {args.gpu_explore_interval} iters  |  lr={args.gpu_smooth_lr}")
    console.print()

    gpu_slp = GPUBatchSLP(device, args.workers)
    gpu_smooth = GPUSmoothDescent(device, args.gpu_explore, gpu_slp.T)
    rng = np.random.default_rng(args.seed)

    overall_best_fit = verified_fit
    overall_best_x = x0.copy()
    start_fit = verified_fit

    for rnd in range(1, args.rounds + 1):
        console.rule(f"[bold]Round {rnd}/{args.rounds}[/]")

        # Create seeds
        seeds = make_seeds(overall_best_x, args.workers, args.perturb_scale, rng)
        N = len(seeds)

        # Worker state (numpy, mirrored to GPU each iter)
        xs = np.stack(seeds)  # (N, 513)
        current_fits = np.array([float(np.max(np.abs(
            np.einsum('ra,rb,rc->abc', *unpack(s)) - T_np))) for s in seeds])
        best_xs = xs.copy()
        best_fits = current_fits.copy()
        trusts = np.full(N, args.trust_init)
        accepts = np.zeros(N, dtype=int)
        rejects = np.zeros(N, dtype=int)
        alive = np.ones(N, dtype=bool)

        round_best_fit = float(best_fits.min())
        round_best_worker = int(best_fits.argmin())
        plateau_det = PlateauDetector(window=50, threshold=1e-9, warn_after=100)
        round_start = time.time()
        last_lp_time = 0.0

        console.print(f"  Seed fitnesses: min={current_fits.min():.10f}  max={current_fits.max():.10f}")

        # Resident GPU tensor for all workers
        xs_gpu = torch.tensor(xs, dtype=torch.float64, device=device)

        # Process pool for LP solves — bypass GIL completely
        lp_pool = ProcessPoolExecutor(max_workers=min(N, 22))

        # Stats for GPU exploration
        gpu_explore_improvements = 0
        gpu_explore_injections = 0

        with Live(console=console, refresh_per_second=2) as live:
            for it in range(1, args.max_iters + 1):
                alive_idx = np.where(alive)[0]
                if len(alive_idx) == 0:
                    break
                M = len(alive_idx)

                # 1. GPU: compute residuals + Jacobians, async transfer to pinned CPU
                xs_alive_gpu = xs_gpu[torch.tensor(alive_idx, device=device)]
                res_cpu, J_cpu = gpu_slp.compute_and_transfer(xs_alive_gpu, M)

                # 2. TRULY CONCURRENT: CPU LP in processes + GPU explore in thread
                lp_start = time.time()

                # Launch CPU LP solves (runs in separate processes, no GIL)
                lp_args = [
                    (res_cpu[i].copy(), J_cpu[i].copy(), trusts[alive_idx[i]], args.active_k)
                    for i in range(M)
                ]
                lp_future = lp_pool.map(_lp_worker_solve, lp_args)

                # Launch GPU exploration in background thread (torch releases GIL)
                gpu_result = [None]  # mutable container for thread result
                gpu_thread = None
                if args.gpu_explore > 0 and it % args.gpu_explore_interval == 0:
                    # Snapshot best_xs for thread safety
                    best_xs_snap = best_xs.copy()
                    best_fits_snap = best_fits.copy()

                    def _gpu_explore():
                        all_by_fit = sorted(range(N), key=lambda i: best_fits_snap[i])
                        n_elite = min(16, N)
                        elite_idx = all_by_fit[:n_elite]
                        children_per = args.gpu_explore // n_elite
                        explore_seeds = []
                        for rank_i, ei in enumerate(elite_idx):
                            base = torch.tensor(best_xs_snap[ei], dtype=torch.float64, device=device)
                            scale = args.perturb_scale * (0.3 + 0.7 * rank_i / max(n_elite - 1, 1))
                            noise = torch.randn(children_per, N_PARAMS,
                                                dtype=torch.float64, device=device) * scale
                            explore_seeds.append(base.unsqueeze(0) + noise)
                        explore_gpu = torch.cat(explore_seeds, dim=0)
                        explore_out, explore_fits = gpu_smooth.descend(
                            explore_gpu, steps=args.gpu_smooth_steps,
                            lr=args.gpu_smooth_lr, beta=200.0)
                        gpu_result[0] = (explore_out, explore_fits)

                    gpu_thread = threading.Thread(target=_gpu_explore, daemon=True)
                    gpu_thread.start()

                # Collect LP results (runs while GPU thread is working)
                lp_results_list = list(lp_future)

                last_lp_time = time.time() - lp_start

                # 3. Build candidate steps and verify on GPU
                candidate_xs_np = []
                candidate_map = []
                for i, wi in enumerate(alive_idx):
                    dx, t_pred = lp_results_list[i]
                    if dx is None:
                        trusts[wi] = max(trusts[wi] / 4, args.trust_min)
                        if trusts[wi] <= args.trust_min:
                            alive[wi] = False
                        continue

                    pred_imp = current_fits[wi] - t_pred
                    if pred_imp < 1e-14:
                        alive[wi] = False
                        continue

                    candidate_xs_np.append(xs[wi] + dx)
                    candidate_map.append((i, wi, t_pred))

                # GPU batch verification
                if candidate_xs_np:
                    cands_gpu = torch.tensor(np.stack(candidate_xs_np),
                                             dtype=torch.float64, device=device)
                    new_fits = gpu_slp.maxabs_batch(cands_gpu).cpu().numpy()

                    for j, (idx_a, wi, t_pred) in enumerate(candidate_map):
                        new_fit = float(new_fits[j])
                        pred_imp = current_fits[wi] - t_pred
                        actual_imp = current_fits[wi] - new_fit
                        rho = actual_imp / pred_imp if pred_imp > 1e-30 else 0

                        if rho >= args.eta_accept and actual_imp > 0:
                            xs[wi] = candidate_xs_np[j]
                            xs_gpu[wi] = cands_gpu[j]
                            current_fits[wi] = new_fit
                            accepts[wi] += 1

                            if new_fit < best_fits[wi]:
                                best_xs[wi] = xs[wi].copy()
                                best_fits[wi] = new_fit

                            if rho > 0.75:
                                trusts[wi] = min(trusts[wi] * 2, args.trust_max)
                            elif rho > 0.5:
                                trusts[wi] = min(trusts[wi] * 1.5, args.trust_max)
                        else:
                            rejects[wi] += 1
                            trusts[wi] = max(trusts[wi] / 2, args.trust_min)
                            if trusts[wi] <= args.trust_min:
                                alive[wi] = False

                    del cands_gpu

                # 4. Join GPU exploration thread, inject results
                if gpu_thread is not None:
                    gpu_thread.join()
                    if gpu_result[0] is not None:
                        explore_out, explore_fits = gpu_result[0]
                        explore_fits_np = explore_fits.cpu().numpy()
                        explore_out_np = explore_out.cpu().numpy()

                        sorted_explore = np.argsort(explore_fits_np)
                        replaceable = sorted(range(N), key=lambda i: current_fits[i], reverse=True)
                        injected_set = set()
                        for bei in sorted_explore:
                            e_fit = float(explore_fits_np[bei])
                            for wi in replaceable:
                                if wi in injected_set:
                                    continue
                                if e_fit < current_fits[wi]:
                                    xs[wi] = explore_out_np[bei]
                                    xs_gpu[wi] = explore_out[bei]
                                    current_fits[wi] = e_fit
                                    if e_fit < best_fits[wi]:
                                        best_xs[wi] = xs[wi].copy()
                                        best_fits[wi] = e_fit
                                    alive[wi] = True
                                    trusts[wi] = args.trust_init
                                    gpu_explore_injections += 1
                                    if e_fit < round_best_fit:
                                        gpu_explore_improvements += 1
                                    injected_set.add(wi)
                                    break

                        del explore_out

                # Update round best
                cur_best_idx = int(best_fits.argmin())
                round_best_fit = float(best_fits[cur_best_idx])
                round_best_worker = cur_best_idx

                # Plateau detection
                plateau_msg = plateau_det.update(it, round_best_fit)

                # Update display
                if it % 2 == 0 or plateau_msg:
                    elapsed = time.time() - round_start
                    table = build_status_table(
                        rnd, args.rounds, it, args.max_iters,
                        best_fits.tolist(), accepts.tolist(), trusts.tolist(),
                        round_best_fit, round_best_worker, elapsed,
                        plateau_det, last_lp_time, start_fit,
                        gpu_explore_injections, gpu_explore_improvements
                    )
                    live.update(table)

                if plateau_msg:
                    console.print(f"  [bold red]⚠ {plateau_msg}[/]")

        lp_pool.shutdown(wait=True)

        # Round summary
        console.print()
        sorted_w = sorted(range(N), key=lambda i: best_fits[i])
        for wi in sorted_w[:5]:
            console.print(f"  W{wi:02d}: {best_fits[wi]:.10f}  "
                          f"(acc={accepts[wi]}, rej={rejects[wi]})")
        if N > 5:
            console.print(f"  ... worst: {best_fits[sorted_w[-1]]:.10f}")

        # Update global best
        if round_best_fit < overall_best_fit:
            overall_best_fit = round_best_fit
            overall_best_x = best_xs[round_best_worker].copy()
            a_b, b_b, g_b = unpack(overall_best_x)
            save_factors(a_b, b_b, g_b, overall_best_fit, args.out)
            console.print(f"  [bold green]>> NEW BEST: {overall_best_fit:.10f} "
                          f"(W{round_best_worker:02d}) → saved {args.out}[/]")
        else:
            console.print(f"  No improvement (best remains {overall_best_fit:.10f})")

        # Save top-3 for diversity
        for rank_i, wi in enumerate(sorted_w[:3]):
            a_r, b_r, g_r = unpack(best_xs[wi])
            save_factors(a_r, b_r, g_r, float(best_fits[wi]),
                         args.out.replace(".json", f"_top{rank_i}.json"))

    # Final
    console.rule("[bold]RESULT[/]")
    console.print(f"  Input:  {start_fit:.10f}")
    console.print(f"  Final:  [bold green]{overall_best_fit:.10f}[/]")
    imp = start_fit - overall_best_fit
    if imp > 0:
        console.print(f"  Improvement: [cyan]{imp:.2e}[/]")
    console.print(f"  Saved: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="GPU SLP Turbo: batched GPU Jacobian + parallel CPU LP")
    parser.add_argument("--input", default="slp_turbo_best.json")
    parser.add_argument("--out", default="slp_turbo_best.json")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--trust-init", type=float, default=0.003)
    parser.add_argument("--trust-max", type=float, default=0.5)
    parser.add_argument("--trust-min", type=float, default=1e-8)
    parser.add_argument("--eta-accept", type=float, default=0.01)
    parser.add_argument("--active-k", type=int, default=500)
    parser.add_argument("--perturb-scale", type=float, default=0.001)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-explore", type=int, default=150_000,
                        help="Number of GPU smooth-descent candidates (~10.5GB at 150K)")
    parser.add_argument("--gpu-explore-interval", type=int, default=3,
                        help="Run GPU exploration every N iterations")
    parser.add_argument("--gpu-smooth-steps", type=int, default=50,
                        help="Adam steps per GPU exploration burst (~33s for 150K)")
    parser.add_argument("--gpu-smooth-lr", type=float, default=5e-5,
                        help="Adam learning rate for GPU smooth descent")
    args = parser.parse_args()
    run_slp_turbo(args)


if __name__ == "__main__":
    main()
