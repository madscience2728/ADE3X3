"""
Master coordinator that distributes work across worker processes
and aggregates results. Manages checkpointing and graceful shutdown.
"""

import json
import time
import signal
import os
from multiprocessing import Process, Queue, Event
from typing import List, Optional
from datetime import datetime

from .config import SearchConfig
from .worker import run_worker, ProgressUpdate, SolutionFound, WorkerDone
from .fiber import enumerate_fiber_partitions


def _worker_loop(wid, work_queue, result_queue, stop_event):
    """Worker loop: pull (config, partition) from queue until empty."""
    while not stop_event.is_set():
        try:
            cfg, partition = work_queue.get_nowait()
        except Exception:
            break
        run_worker(wid, cfg, partition, result_queue, stop_event)


class Coordinator:
    def __init__(self, ranks: List[int], config: SearchConfig,
                 dashboard=None):
        self.ranks = ranks
        self.base_config = config
        self.dashboard = dashboard
        self.result_queue = Queue()
        self.stop_event = Event()
        self.workers: List[Process] = []
        self.completed_ranks = set()
        self.solutions = {}  # R -> list of solutions
        self.stats = {}  # R -> aggregated stats
        self.start_time = None

        # Per-rank tracking
        for R in ranks:
            self.solutions[R] = []
            self.stats[R] = {
                'candidates_checked': 0,
                'gate1_passes': 0,
                'gate2_passes': 0,
                'gate3_passes': 0,
                'solutions_found': 0,
                'workers_done': 0,
                'workers_active': 0,
            }

    def _make_config_for_rank(self, R: int) -> SearchConfig:
        """Create a SearchConfig for a specific R value."""
        cfg = SearchConfig(
            R=R,
            n=self.base_config.n,
            coeff_field=list(self.base_config.coeff_field),
            n_workers=self.base_config.n_workers,
            use_gpu=self.base_config.use_gpu,
            rank_tolerance=self.base_config.rank_tolerance,
            gate3_residual_tolerance=self.base_config.gate3_residual_tolerance,
            checkpoint_file=self.base_config.checkpoint_file,
            solution_file=self.base_config.solution_file,
            log_file=self.base_config.log_file,
            log_gate1_survivors=self.base_config.log_gate1_survivors,
            gate1_survivor_file=self.base_config.gate1_survivor_file,
        )
        return cfg

    def load_checkpoint(self) -> bool:
        """Load checkpoint file if it exists. Returns True if loaded."""
        cp_file = self.base_config.checkpoint_file
        if not os.path.exists(cp_file):
            return False
        try:
            with open(cp_file, 'r') as f:
                cp = json.load(f)
            self.completed_ranks = set(cp.get('completed_ranks', []))
            for R_str, sols in cp.get('solutions', {}).items():
                R = int(R_str)
                if R in self.solutions:
                    self.solutions[R] = sols
            return True
        except Exception:
            return False

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        cp_file = self.base_config.checkpoint_file
        os.makedirs(os.path.dirname(cp_file), exist_ok=True)
        cp = {
            'timestamp': datetime.now().isoformat(),
            'coeff_field': list(self.base_config.coeff_field),
            'completed_ranks': list(self.completed_ranks),
            'stats': {str(R): s for R, s in self.stats.items()},
            'solutions': {str(R): [] for R in self.ranks},  # can't serialize numpy easily
            'elapsed_seconds': time.perf_counter() - self.start_time if self.start_time else 0,
        }
        with open(cp_file, 'w') as f:
            json.dump(cp, f, indent=2)

    def save_solution(self, R: int, sol: SolutionFound):
        """Save a solution immediately."""
        sol_file = self.base_config.solution_file
        os.makedirs(os.path.dirname(sol_file), exist_ok=True)
        sol_data = {
            'R': R,
            'alpha': sol.alpha.tolist() if sol.alpha is not None else None,
            'beta': sol.beta.tolist() if sol.beta is not None else None,
            'gamma': sol.gamma.tolist() if sol.gamma is not None else None,
            'residual': float(sol.residual) if sol.residual is not None else None,
            'timestamp': datetime.now().isoformat(),
        }
        # Append R to filename if multiple ranks
        out_file = sol_file.replace('.json', f'_R{R}.json')
        with open(out_file, 'w') as f:
            json.dump(sol_data, f, indent=2)

    def _process_messages(self, timeout: float = 0.1):
        """Drain result queue and process messages."""
        while True:
            try:
                msg = self.result_queue.get(timeout=timeout)
            except Exception:
                break

            if isinstance(msg, SolutionFound):
                self.solutions[msg.R].append(msg)
                self.stats[msg.R]['solutions_found'] += 1
                self.save_solution(msg.R, msg)
                if self.dashboard:
                    self.dashboard.on_solution(msg)

            elif isinstance(msg, WorkerDone):
                s = self.stats[msg.R]
                s['candidates_checked'] += msg.total_checked
                s['gate1_passes'] += msg.total_gate1
                s['gate2_passes'] += msg.total_gate2
                s['gate3_passes'] += msg.total_gate3
                s['workers_done'] += 1
                if '_completed_units' in self.stats:
                    self.stats['_completed_units'] += 1
                if self.dashboard:
                    self.dashboard.on_worker_done(msg)

            elif isinstance(msg, ProgressUpdate):
                if self.dashboard:
                    self.dashboard.on_progress(msg)

    def run(self, resume: bool = False):
        """Main coordinator loop. Distributes partitions across workers."""
        self.start_time = time.perf_counter()

        if resume:
            self.load_checkpoint()

        # Set up signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)

        def _shutdown(signum, frame):
            self.stop_event.set()
            self.save_checkpoint()
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, _shutdown)

        # Build work queue: (R, partition) for each rank × partition
        pending_ranks = [R for R in self.ranks if R not in self.completed_ranks]
        if not pending_ranks:
            print("All ranks already completed (from checkpoint).")
            return

        work_units = []
        for R in pending_ranks:
            partitions = enumerate_fiber_partitions(R, 9)
            for part in partitions:
                work_units.append((R, part))

        completed_units = set(self.completed_partitions) if hasattr(self, 'completed_partitions') else set()
        work_units = [(R, p) for R, p in work_units if (R, p) not in completed_units]

        total_units = len(work_units)
        self.stats['_total_units'] = total_units
        self.stats['_completed_units'] = 0

        print(f"Work units: {total_units} partitions across {len(pending_ranks)} ranks")
        print(f"Workers: {self.base_config.n_workers}")

        # Use a work queue — feed partitions to workers as they finish
        work_queue = Queue()
        for R, part in work_units:
            cfg = self._make_config_for_rank(R)
            work_queue.put((cfg, part))

        n_workers = min(self.base_config.n_workers, total_units)

        for wid in range(n_workers):
            p = Process(
                target=_worker_loop,
                args=(wid, work_queue, self.result_queue, self.stop_event),
                daemon=True,
            )
            self.workers.append(p)

        for p in self.workers:
            p.start()

        if self.dashboard:
            self.dashboard.start(self.ranks, self.stats)

        # Main loop: process messages and update dashboard
        last_checkpoint = time.perf_counter()
        while any(p.is_alive() for p in self.workers):
            if self.stop_event.is_set():
                break

            self._process_messages(timeout=0.5)

            # Periodic checkpoint
            now = time.perf_counter()
            if now - last_checkpoint > self.base_config.checkpoint_interval_seconds:
                self.save_checkpoint()
                last_checkpoint = now

            if self.dashboard:
                elapsed = now - self.start_time
                self.dashboard.update(self.stats, elapsed)

        # Final drain
        self._process_messages(timeout=1.0)
        self.save_checkpoint()

        # Wait for workers
        for p in self.workers:
            p.join(timeout=5)

        if self.dashboard:
            self.dashboard.stop()

        # Restore signal handler
        signal.signal(signal.SIGINT, original_sigint)

    def print_summary(self):
        """Print final results."""
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        print(f"\n{'='*70}")
        print(f"  SEARCH COMPLETE — Wall time: {elapsed:.1f}s")
        print(f"  Coefficient field: {{{', '.join(str(c) for c in self.base_config.coeff_field)}}}")
        print(f"{'='*70}")

        for R in sorted(self.ranks):
            s = self.stats[R]
            print(f"\n  R = {R}:")
            print(f"    Candidates checked: {s['candidates_checked']:,}")
            print(f"    Gate 1 passes:      {s['gate1_passes']:,}")
            print(f"    Gate 2 passes:      {s['gate2_passes']:,}")
            print(f"    Gate 3 passes:      {s['gate3_passes']:,}")
            print(f"    Solutions found:     {s['solutions_found']}")

            if self.solutions[R]:
                print(f"    *** SOLUTION FOUND for R={R}! ***")
                print(f"    Saved to: {self.base_config.solution_file.replace('.json', f'_R{R}.json')}")
            else:
                print(f"    No R={R} decomposition over "
                      f"{{{', '.join(str(c) for c in self.base_config.coeff_field)}}}")

        print(f"\n{'='*70}")
