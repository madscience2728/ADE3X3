"""GPU smooth minimax engine — runs continuously in its own thread.

Pulls elite seeds from pool, runs large-batch Adam descent on log-sum-exp
surrogate, deposits improved candidates back into pool. Never blocks CPU.
"""

import threading
import time

import numpy as np
import torch

from .config import R, D, N_ENTRIES, N_PARAMS, T_NP, LIVE_IDX, DEAD_IDX


class GPUEngine:
    """Continuous GPU smooth descent running in a daemon thread."""

    def __init__(self, device, pool, args):
        self.device = device
        self.pool = pool
        self.args = args

        self.T = torch.tensor(T_NP, dtype=torch.float64, device=device)
        self.T_flat = self.T.reshape(-1)

        # Build tier weights on GPU
        w = np.ones(N_ENTRIES, dtype=np.float64)
        w[LIVE_IDX] = args.live_weight
        w[DEAD_IDX] = args.dead_weight
        self.tier_weights = torch.tensor(w, dtype=torch.float64, device=device)

        self._stop = threading.Event()
        self._thread = None

        # Stats (read by display)
        self.bursts = 0
        self.total_injected = 0
        self.last_burst_time = 0.0
        self.best_gpu_fit = float('inf')
        self.state = "idle"

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="gpu-engine")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)

    def _run_loop(self):
        """Continuous loop: seed → descend → inject → repeat."""
        while not self._stop.is_set():
            self.state = "seeding"
            t0 = time.time()

            # Pull elites from pool
            elite_xs, elite_fits = self.pool.get_elite_seeds(n_elite=16)
            n_elite = len(elite_xs)
            if n_elite == 0:
                self._stop.wait(1.0)
                continue

            # Build seeds on GPU — wider perturbation for real exploration
            children_per = self.args.gpu_batch // n_elite
            seeds_list = []
            for rank_i in range(n_elite):
                base = torch.tensor(elite_xs[rank_i], dtype=torch.float64, device=self.device)
                # Tier 0 (best): perturb_scale, Tier n_elite-1 (worst): 10x wider
                scale = self.args.perturb_scale * (1.0 + 9.0 * rank_i / max(n_elite - 1, 1))
                noise = torch.randn(children_per, N_PARAMS,
                                    dtype=torch.float64, device=self.device) * scale
                seeds_list.append(base.unsqueeze(0) + noise)
            xs_gpu = torch.cat(seeds_list, dim=0)

            # Smooth descent
            self.state = f"descending {xs_gpu.shape[0]} candidates"
            xs_out = self._smooth_descent(xs_gpu, self.args.gpu_steps,
                                          self.args.gpu_lr, self.args.gpu_beta)

            # Evaluate true maxabs
            self.state = "evaluating"
            fits_gpu = self._maxabs_batch(xs_out)
            fits_np = fits_gpu.cpu().numpy()
            xs_np = xs_out.cpu().numpy()

            # Sort best-first and inject top candidates
            order = np.argsort(fits_np)
            # Only send candidates that are actually good (top 1% or up to N slots)
            n_inject = min(len(order), self.pool.n, max(self.pool.n, 100))
            top_xs = xs_np[order[:n_inject]]
            top_fits = fits_np[order[:n_inject]]

            self.state = "injecting"
            injected = self.pool.inject_gpu_candidates(top_xs, top_fits)

            # Update stats
            self.bursts += 1
            self.total_injected += injected
            self.last_burst_time = time.time() - t0
            self.best_gpu_fit = min(self.best_gpu_fit, float(top_fits[0]))

            # Free GPU memory for next burst
            del xs_gpu, xs_out, fits_gpu
            torch.cuda.empty_cache()

            self.state = "idle"

    def _smooth_descent(self, xs, steps, lr, beta):
        """Adam on log-sum-exp smooth minimax surrogate with tier weights."""
        xs = xs.clone().requires_grad_(True)
        opt = torch.optim.Adam([xs], lr=lr)
        w = self.tier_weights  # (729,)
        sparsity_lam = self.args.sparsity_lambda

        for step in range(steps):
            if self._stop.is_set():
                break
            opt.zero_grad()
            n = xs.shape[0]
            a = xs[:, :R*D].reshape(n, R, D)
            b = xs[:, R*D:2*R*D].reshape(n, R, D)
            g = xs[:, 2*R*D:].reshape(n, R, D)
            T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
            res = (T_hat.reshape(n, -1) - self.T_flat).abs()
            # Apply tier weights: live entries weighted higher
            weighted_res = res * w.unsqueeze(0)
            loss = (torch.logsumexp(beta * weighted_res, dim=1) - np.log(N_ENTRIES)) / beta

            # Optional L1 sparsity penalty on factor entries
            if sparsity_lam > 0:
                loss = loss + sparsity_lam * xs.abs().sum(dim=1)

            loss.sum().backward()
            opt.step()

            # Sharpen beta halfway through
            if step == steps // 2:
                beta = min(beta * 2, 1000.0)

        return xs.detach()

    @torch.no_grad()
    def _maxabs_batch(self, xs):
        """True maxabs fitness (non-differentiable)."""
        n = xs.shape[0]
        a = xs[:, :R*D].reshape(n, R, D)
        b = xs[:, R*D:2*R*D].reshape(n, R, D)
        g = xs[:, 2*R*D:].reshape(n, R, D)
        T_hat = torch.einsum('nra,nrb,nrc->nabc', a, b, g)
        return (T_hat.reshape(n, -1) - self.T_flat).abs().max(dim=1).values
