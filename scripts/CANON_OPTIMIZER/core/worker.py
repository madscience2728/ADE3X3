"""
Worker process that receives a work unit and runs enumeration.
Communicates with coordinator via multiprocessing.Queue.
"""

import numpy as np
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Queue

from .enumerator import search_partition, SearchStats, _build_fiber_assignment
from .config import SearchConfig


@dataclass
class ProgressUpdate:
    worker_id: int
    R: int
    terms_placed: int
    candidates_checked: int
    gate1_passes: int
    gate2_passes: int
    gate3_passes: int
    solutions_found: int
    elapsed_seconds: float
    message: str = ""


@dataclass
class SolutionFound:
    worker_id: int
    R: int
    alpha: np.ndarray
    beta: np.ndarray
    gamma: Optional[np.ndarray]
    residual: Optional[float]


@dataclass
class WorkerDone:
    worker_id: int
    R: int
    total_checked: int
    total_gate1: int
    total_gate2: int
    total_gate3: int
    solutions_found: int
    wall_time: float


def run_worker(worker_id: int, config: SearchConfig,
               partition: tuple,
               result_queue: Queue, stop_event=None):
    """Worker entry point. Searches a single fiber partition in a subprocess."""
    t0 = time.perf_counter()

    def stop_flag():
        if stop_event is not None:
            return stop_event.is_set()
        return False

    def progress_cb(msg):
        try:
            result_queue.put_nowait(ProgressUpdate(
                worker_id=worker_id,
                R=config.R,
                terms_placed=0,
                candidates_checked=0,
                gate1_passes=0,
                gate2_passes=0,
                gate3_passes=0,
                solutions_found=0,
                elapsed_seconds=time.perf_counter() - t0,
                message=str(msg),
            ))
        except Exception:
            pass

    def gate_callback(gate_name, alpha, beta, stats):
        if gate_name == 'gate1':
            try:
                result_queue.put_nowait(ProgressUpdate(
                    worker_id=worker_id,
                    R=config.R,
                    terms_placed=config.R,
                    candidates_checked=stats.candidates_checked,
                    gate1_passes=stats.gate1_passes,
                    gate2_passes=stats.gate2_passes,
                    gate3_passes=stats.gate3_passes,
                    solutions_found=stats.solutions_found,
                    elapsed_seconds=time.perf_counter() - t0,
                    message=f"Gate 1 pass #{stats.gate1_passes}",
                ))
            except Exception:
                pass

    fiber_assign = _build_fiber_assignment(partition)

    try:
        solutions, stats = search_partition(
            config=config,
            partition=partition,
            fiber_assign=fiber_assign,
            callback=gate_callback,
            progress_callback=progress_cb,
            stop_flag=stop_flag,
        )

        for sol in solutions:
            result_queue.put(SolutionFound(
                worker_id=worker_id,
                R=config.R,
                alpha=sol['alpha'],
                beta=sol['beta'],
                gamma=sol.get('gamma'),
                residual=sol.get('residual'),
            ))

    except Exception as e:
        progress_cb(f"ERROR: {traceback.format_exc()}")
        solutions = []
        stats = SearchStats()

    wall = time.perf_counter() - t0
    result_queue.put(WorkerDone(
        worker_id=worker_id,
        R=config.R,
        total_checked=stats.candidates_checked,
        total_gate1=stats.gate1_passes,
        total_gate2=stats.gate2_passes,
        total_gate3=stats.gate3_passes,
        solutions_found=stats.solutions_found,
        wall_time=wall,
    ))
