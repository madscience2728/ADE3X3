"""Thread-safe in-RAM candidate pool — the central hub.

All engines read from and write to this pool. No engine blocks another.
"""

import threading
import time

import numpy as np

from .config import N_PARAMS


class CandidatePool:
    """Thread-safe store of N candidates with fitness tracking.

    Layout:
      xs:           (N, 513) float64 — current parameter vectors
      best_xs:      (N, 513) float64 — best-ever per slot
      fits:         (N,) float64     — current fitness
      best_fits:    (N,) float64     — best-ever fitness per slot
      trusts:       (N,) float64     — trust radii for SLP
      alive:        (N,) bool        — whether slot is active for SLP
      accepts:      (N,) int         — accept count per slot
      rejects:      (N,) int         — reject count per slot
    """

    def __init__(self, n_slots, trust_init=0.003, on_new_best=None):
        self.n = n_slots
        self._lock = threading.Lock()

        self.xs = np.zeros((n_slots, N_PARAMS), dtype=np.float64)
        self.best_xs = np.zeros((n_slots, N_PARAMS), dtype=np.float64)
        self.fits = np.full(n_slots, np.inf)
        self.best_fits = np.full(n_slots, np.inf)
        self.trusts = np.full(n_slots, trust_init)
        self.alive = np.ones(n_slots, dtype=bool)
        self.accepts = np.zeros(n_slots, dtype=int)
        self.rejects = np.zeros(n_slots, dtype=int)

        self._trust_init = trust_init
        self.global_best_fit = np.inf
        self.global_best_x = np.zeros(N_PARAMS, dtype=np.float64)
        self._on_new_best = on_new_best  # callback(x, fit) called on improvement

        # Stats
        self.gpu_injections = 0
        self.gpu_improvements = 0  # improved global best
        self.cpu_iterations = 0
        self.start_time = time.time()

    def seed(self, idx, x, fit):
        """Set initial candidate (no lock needed before engines start)."""
        self.xs[idx] = x
        self.best_xs[idx] = x
        self.fits[idx] = fit
        self.best_fits[idx] = fit
        if fit < self.global_best_fit:
            self.global_best_fit = fit
            self.global_best_x = x.copy()
            self._fire_new_best(x, fit)

    def _fire_new_best(self, x, fit):
        """Call the on_new_best callback (outside lock to avoid deadlock)."""
        if self._on_new_best is not None:
            try:
                self._on_new_best(x.copy(), float(fit))
            except Exception:
                pass  # never let callback errors crash the pool

    # ── CPU SLP interface ─────────────────────────────────────────────

    def get_alive_snapshot(self):
        """Return (alive_indices, xs_copy, fits_copy, trusts_copy) — read snapshot."""
        with self._lock:
            alive_idx = np.where(self.alive)[0]
            return (alive_idx.copy(),
                    self.xs[alive_idx].copy(),
                    self.fits[alive_idx].copy(),
                    self.trusts[alive_idx].copy())

    def apply_lp_results(self, alive_idx, new_xs, new_fits, accepted, trust_updates, args):
        """Apply LP step results back to the pool."""
        with self._lock:
            new_best_x, new_best_fit = None, None
            for i, wi in enumerate(alive_idx):
                if accepted[i]:
                    self.xs[wi] = new_xs[i]
                    self.fits[wi] = new_fits[i]
                    self.accepts[wi] += 1
                    if new_fits[i] < self.best_fits[wi]:
                        self.best_xs[wi] = new_xs[i].copy()
                        self.best_fits[wi] = new_fits[i]
                    if new_fits[i] < self.global_best_fit:
                        self.global_best_fit = new_fits[i]
                        self.global_best_x = new_xs[i].copy()
                        new_best_x, new_best_fit = new_xs[i], new_fits[i]
                else:
                    self.rejects[wi] += 1

                self.trusts[wi] = trust_updates[i]
                if self.trusts[wi] <= args.trust_min:
                    self.alive[wi] = False

            self.cpu_iterations += 1

        # Fire callback outside lock
        if new_best_x is not None:
            self._fire_new_best(new_best_x, new_best_fit)

    def kill_slot(self, wi):
        """Mark a slot as dead (no improvement possible)."""
        with self._lock:
            self.alive[wi] = False

    # ── GPU exploration interface ─────────────────────────────────────

    def get_elite_seeds(self, n_elite=16):
        """Return (elite_xs, elite_fits) from best_xs, sorted by best_fits."""
        with self._lock:
            order = np.argsort(self.best_fits)[:n_elite]
            return self.best_xs[order].copy(), self.best_fits[order].copy()

    def inject_gpu_candidates(self, new_xs, new_fits, replace_frac=0.25):
        """Inject best GPU candidates into worst worker slots unconditionally.

        Always replaces the bottom `replace_frac` of workers with the best GPU
        candidates, so GPU exploration feeds diverse starting points into LP.
        new_xs: (K, 513), new_fits: (K,) — already sorted best-first.
        Returns number of injections.
        """
        new_best_x, new_best_fit = None, None
        with self._lock:
            # How many slots to forcibly replace
            n_replace = max(1, int(self.n * replace_frac))
            n_replace = min(n_replace, len(new_fits))

            # Worst workers go first (sorted worst-fitness-first)
            slot_order = np.argsort(self.fits)[::-1][:n_replace]

            injected = 0
            for i, wi in enumerate(slot_order):
                if i >= len(new_fits):
                    break
                self.xs[wi] = new_xs[i]
                self.fits[wi] = new_fits[i]
                if new_fits[i] < self.best_fits[wi]:
                    self.best_xs[wi] = new_xs[i].copy()
                    self.best_fits[wi] = new_fits[i]
                if new_fits[i] < self.global_best_fit:
                    self.global_best_fit = new_fits[i]
                    self.global_best_x = new_xs[i].copy()
                    self.gpu_improvements += 1
                    new_best_x, new_best_fit = new_xs[i], new_fits[i]
                self.alive[wi] = True
                self.trusts[wi] = self._trust_init
                injected += 1

            self.gpu_injections += injected

        # Fire callback outside lock
        if new_best_x is not None:
            self._fire_new_best(new_best_x, new_best_fit)

        return injected

    # ── Display interface ─────────────────────────────────────────────

    def snapshot(self):
        """Return a dict of current stats for display (non-blocking copy)."""
        with self._lock:
            return {
                "global_best_fit": self.global_best_fit,
                "best_fits": self.best_fits.copy(),
                "fits": self.fits.copy(),
                "alive": self.alive.copy(),
                "accepts": self.accepts.copy(),
                "rejects": self.rejects.copy(),
                "trusts": self.trusts.copy(),
                "gpu_injections": self.gpu_injections,
                "gpu_improvements": self.gpu_improvements,
                "cpu_iterations": self.cpu_iterations,
                "elapsed": time.time() - self.start_time,
            }
