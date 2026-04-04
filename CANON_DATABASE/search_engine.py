from __future__ import annotations

import json
import hashlib
import multiprocessing
import tempfile
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from assembler import DependentAssembler
from basis_builder import BasisBuilder
from gate_check import save_solution, verify_phase2_solution
from term_db import TermDB


def _get_ram_gb() -> tuple[float, float]:
    """Return (used_gb, available_gb) for the current process + system."""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        used = proc.memory_info().rss / (1024**3)
        avail = psutil.virtual_memory().available / (1024**3)
        return used, avail
    return 0.0, 0.0


# ── Worker process globals (set by _worker_init) ──
_worker_db: TermDB | None = None
_worker_config = None


def _worker_init(db_path: str, config):
    """Initializer for each worker process — loads a read-only mmap'd DB."""
    global _worker_db, _worker_config
    _worker_config = config
    _worker_db = TermDB.__new__(TermDB)
    _worker_db._load(db_path, progress_callback=lambda _: None, readonly=True)


def _worker_process_basis(args: tuple) -> tuple[list[dict], dict, str | None]:
    """Runs in a child process — full GIL-free parallelism."""
    rank, basis_indices, basis_V_basis, basis_dependent_estimate, hit_record_indices = args
    db = _worker_db
    config = _worker_config

    assembler = DependentAssembler(db, config)
    solutions = assembler.assemble(rank, basis_indices, hit_record_indices)

    stats = {
        "bases_tested": 1,
        "hits": len(hit_record_indices),
        "gate2_passes": assembler.stats.gate2_passes,
        "gate3_passes": assembler.stats.gate3_passes,
        "configs_tested": assembler.stats.configs_tested,
    }

    near_miss = None
    if not solutions:
        near_miss = f"hits={len(hit_record_indices):,}, dep_est={basis_dependent_estimate:,}"

    return solutions, stats, near_miss


PACKET_DIR = "packets"


class SearchEngine:
    def __init__(self, config, dashboard=None):
        self.config = config
        self.dashboard = dashboard
        self.db = None
        self.state = {
            "current_rank": None,
            "phase": "init",
            "elapsed": 0.0,
            "basis": {},
            "assembly": {},
            "near_miss": "-",
            "workers_active": 0,
        }
        self.start_time = 0.0
        self.solutions: dict[int, list[dict]] = {}
        self._lock = threading.Lock()

    def load_db(self) -> None:
        """Load DB with all blocking ops in a background thread so the TUI stays alive."""
        from search_dashboard import LOAD_STAGES, RAM_STAGES, DEDUP_STAGES, REP_STAGES

        self._db_error: Exception | None = None

        def _do_load():
            """Runs in a daemon thread — all numpy blocking happens here."""
            try:
                self.db = TermDB.__new__(TermDB)
                self.db._load(self.config.db_path, progress_callback=self._set_db_status)
                if self.config.load_db_to_ram:
                    self.db.load_to_ram(progress_callback=self._set_db_status)
                self.db.build_dedup_index(progress_callback=self._set_db_status)

                # Representative index: for each unique H-row, find any one record index.
                self._set_db_status("rep: building representative index")
                n_unique = self.db.n_unique_H
                rep_cache = Path(self.config.db_path) / 'dedup_unique_record_index.npy'
                if rep_cache.exists():
                    self._set_db_status("rep: loading cached representative index")
                    rep = np.load(str(rep_cache))
                    if rep.shape[0] == n_unique:
                        self.db.unique_record_index = rep
                        self._set_db_status(f"rep: loaded cache — {n_unique:,} unique H rows")
                    else:
                        rep = None  # size mismatch, rebuild
                else:
                    rep = None
                if not hasattr(self.db, 'unique_record_index'):
                    rep = np.full(n_unique, self.db.H.shape[0], dtype=np.int64)
                    record_indices = np.arange(self.db.H.shape[0], dtype=np.int64)
                    np.minimum.at(rep, self.db.H_dedup_map, record_indices)
                    del record_indices  # free 2.9 GB
                    self.db.unique_record_index = rep
                    try:
                        np.save(str(rep_cache), rep)
                    except Exception:
                        pass
                    self._set_db_status(f"rep: done — {n_unique:,} unique H rows")

                # Pre-cache int16 unique_H so workers don't each allocate 1.1GB copies
                self._set_db_status("finalizing: caching int16 unique_H")
                self.db.finalize_for_search()
                self._set_db_status(f"ready — {n_unique:,} unique H rows")
            except Exception as exc:
                self._db_error = exc

        # Set up stage progress on the dashboard
        all_stages = LOAD_STAGES + (RAM_STAGES if self.config.load_db_to_ram else []) + DEDUP_STAGES + REP_STAGES
        if self.dashboard:
            self.dashboard.set_stages(all_stages)

        self.state["phase"] = "db-load"
        self.state["db_status"] = "starting"
        self._update_dashboard()

        worker = threading.Thread(target=_do_load, daemon=True)
        worker.start()

        # Manual refresh loop — keeps heartbeat, wall time, and progress bar alive
        while worker.is_alive():
            if self.dashboard:
                self.dashboard.refresh()
            time.sleep(0.1)  # releases GIL so numpy in the worker can run

        worker.join()
        if self._db_error:
            raise self._db_error

        self.state["phase"] = "db-ready"
        self.state["db_status"] = f"ready ({self.db.n_unique_H:,} unique H rows)"
        if self.dashboard:
            self.dashboard.set_stages([])  # clear stage bar
        self._update_dashboard()
        if self.dashboard:
            self.dashboard.refresh()

    def run(self, resume: bool = False) -> dict[int, list[dict]]:
        self.config.validate()
        self.start_time = time.perf_counter()
        if self.dashboard:
            self.dashboard.start(self.state)

        self.load_db()

        self._run_error: Exception | None = None
        self._run_done = False

        def _search_loop():
            try:
                self._search_all_ranks(resume)
            except Exception as exc:
                self._run_error = exc
            finally:
                self._run_done = True

        worker = threading.Thread(target=_search_loop, daemon=True)
        worker.start()

        # Main thread: refresh dashboard + sample RAM until search finishes
        while worker.is_alive():
            used, avail = _get_ram_gb()
            with self._lock:
                self.state["ram_used_gb"] = used
                self.state["ram_avail_gb"] = avail
            if self.dashboard:
                self.dashboard.refresh()
            time.sleep(0.15)

        worker.join()

        if self.dashboard:
            self.dashboard.stop()

        if self._run_error:
            raise self._run_error
        return self.solutions

    def _search_all_ranks(self, resume: bool) -> None:
        """Runs entirely in a background thread. Workers are child processes."""
        checkpoint = self._load_checkpoint() if resume else {}
        n_workers = max(1, self.config.n_workers)
        ram_floor_gb = self.config.ram_floor_gb

        for rank in self.config.target_ranks:
            self.solutions.setdefault(rank, [])
            with self._lock:
                self.state["current_rank"] = rank
                self.state["phase"] = "basis"
                self.state["assembly"] = {}

            basis_builder = BasisBuilder(self.db, self.config, rank, rng=np.random.default_rng(rank))
            completed_keys = set(checkpoint.get(str(rank), []))
            last_checkpoint = time.perf_counter()

            # ProcessPoolExecutor: each child mmaps the DB independently (shares OS pages)
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(self.config.db_path, self.config),
                mp_context=multiprocessing.get_context('spawn'),
            ) as pool:
                futures: dict = {}
                basis_exhausted = False
                basis_iter = basis_builder.build_bases()

                def _submit_next():
                    nonlocal basis_exhausted
                    while not basis_exhausted:
                        with self._lock:
                            self.state["basis"] = basis_builder.stats.__dict__.copy()
                        try:
                            basis = next(basis_iter)
                        except StopIteration:
                            basis_exhausted = True
                            return False
                        basis_key = ",".join(map(str, basis.indices))
                        if basis_key in completed_keys:
                            continue

                        # Query hits in the coordinator (has dedup index in RAM)
                        hit_indices = self.db.query_subspace_dedup(basis.V_basis)
                        basis_record_indices = self._expand_basis_indices(basis.indices)

                        # Ship only small arrays to child process
                        args = (
                            rank,
                            basis_record_indices,
                            basis.V_basis,
                            basis.dependent_estimate,
                            hit_indices,
                        )
                        fut = pool.submit(_worker_process_basis, args)
                        futures[fut] = (basis, basis_key)
                        return True
                    return False

                for _ in range(n_workers):
                    if not _submit_next():
                        break
                    with self._lock:
                        self.state["workers_active"] = len(futures)
                        self.state["phase"] = "search" if futures else "basis"

                done_this_rank = False
                while futures and not done_this_rank:
                    newly_done = [f for f in list(futures) if f.done()]

                    if not newly_done:
                        time.sleep(0.05)
                        continue

                    for fut in newly_done:
                        basis, basis_key = futures.pop(fut)

                        try:
                            worker_solutions, worker_stats, near_miss_msg = fut.result()
                        except Exception as exc:
                            with self._lock:
                                self.state["near_miss"] = f"ERROR: {exc}"
                            _submit_next()
                            continue

                        with self._lock:
                            completed_keys.add(basis_key)
                            self.state["basis"] = basis_builder.stats.__dict__.copy()
                            self.state["workers_active"] = len(futures)
                            self.state["phase"] = "search"

                            # Merge worker stats
                            shared = self.state.get("assembly", {})
                            shared["bases_tested"] = shared.get("bases_tested", 0) + 1
                            old_avg = shared.get("avg_hits", 0.0)
                            n = shared.get("bases_tested", 1)
                            shared["avg_hits"] = ((n - 1) * old_avg + float(worker_stats["hits"])) / n
                            shared["gate2_passes"] = shared.get("gate2_passes", 0) + worker_stats["gate2_passes"]
                            shared["gate3_passes"] = shared.get("gate3_passes", 0) + worker_stats["gate3_passes"]
                            shared["configs_tested"] = shared.get("configs_tested", 0) + worker_stats["configs_tested"]
                            self.state["assembly"] = shared

                            if worker_solutions:
                                for solution in worker_solutions:
                                    save_solution(solution, self._solution_path(rank))
                                    verify_phase2_solution(solution)
                                    self.solutions[rank].append(solution)
                                    if len(self.solutions[rank]) >= self.config.max_solutions_per_rank:
                                        break
                            elif near_miss_msg:
                                self.state["near_miss"] = near_miss_msg

                            if time.perf_counter() - last_checkpoint >= self.config.checkpoint_interval:
                                checkpoint[str(rank)] = sorted(completed_keys)
                                self._save_checkpoint(checkpoint)
                                last_checkpoint = time.perf_counter()

                        if len(self.solutions[rank]) >= self.config.max_solutions_per_rank:
                            for f in futures:
                                f.cancel()
                            futures.clear()
                            done_this_rank = True
                            break

                        # RAM throttle
                        _, avail = _get_ram_gb()
                        if avail >= ram_floor_gb:
                            _submit_next()
                        else:
                            with self._lock:
                                self.state["near_miss"] = f"RAM throttle: {avail:.1f}GB free < {ram_floor_gb}GB floor"

                    with self._lock:
                        self.state["workers_active"] = len(futures)

            checkpoint[str(rank)] = sorted(completed_keys)
            self._save_checkpoint(checkpoint)

    def _expand_basis_indices(self, unique_indices: list[int]) -> list[int]:
        return [int(self.db.unique_record_index[unique_idx]) for unique_idx in unique_indices]

    def _solution_path(self, rank: int) -> str:
        path = Path(self.config.solution_file)
        return str(path.with_name(f"{path.stem}_R{rank}{path.suffix}"))

    def _load_checkpoint(self) -> dict:
        path = Path(self.config.checkpoint_file)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_checkpoint(self, checkpoint: dict) -> None:
        path = Path(self.config.checkpoint_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2)

    def _update_dashboard(self) -> None:
        self.state["elapsed"] = time.perf_counter() - self.start_time
        used, avail = _get_ram_gb()
        self.state["ram_used_gb"] = used
        self.state["ram_avail_gb"] = avail
        if self.dashboard:
            self.dashboard.update(self.state)

    def _set_db_status(self, message: str) -> None:
        self.state["db_status"] = message
        # Determine phase from the message prefix
        if message.startswith("load_to_ram:"):
            self.state["phase"] = "db-ram"
        elif message.startswith("dedup:"):
            self.state["phase"] = "db-dedup"
        elif message.startswith("rep:"):
            self.state["phase"] = "db-index"
        elif message.startswith("load:"):
            self.state["phase"] = "db-load"
        # Advance stage progress bar
        if self.dashboard:
            self.dashboard.advance_stage(message.split("(")[0].strip())


# ═══════════════════════════════════════════════════════════════
#  STAGE 1 — Harvest: basis builder → query → save work packets
# ═══════════════════════════════════════════════════════════════

class HarvestEngine(SearchEngine):
    """Stage 1: generate bases, query hits, write .npz work packets to disk.

    Reuses SearchEngine's load_db and dashboard wiring.  The run() method
    iterates ranks, builds bases, queries the dedup index, and writes one
    .npz packet per basis to  <db_path>/../packets/R<rank>/basis_NNNN.npz.
    """

    def run(self, resume: bool = False) -> dict[int, int]:
        """Returns {rank: packets_written}."""
        self.config.validate()
        self.start_time = time.perf_counter()
        if self.dashboard:
            self.dashboard.start(self.state)

        self.load_db()

        self._run_error: Exception | None = None
        result: dict[int, int] = {}

        def _harvest_loop():
            try:
                for rank in self.config.target_ranks:
                    result[rank] = self._harvest_rank(rank)
            except Exception as exc:
                self._run_error = exc

        worker = threading.Thread(target=_harvest_loop, daemon=True)
        worker.start()

        while worker.is_alive():
            used, avail = _get_ram_gb()
            with self._lock:
                self.state["ram_used_gb"] = used
                self.state["ram_avail_gb"] = avail
            if self.dashboard:
                self.dashboard.refresh()
            time.sleep(0.15)

        worker.join()
        if self.dashboard:
            self.dashboard.stop()
        if self._run_error:
            raise self._run_error
        return result

    def _harvest_rank(self, rank: int) -> int:
        with self._lock:
            self.state["current_rank"] = rank
            self.state["phase"] = "harvest"
            self.state["assembly"] = {}

        packet_dir = Path(self.config.db_path).parent / PACKET_DIR / f"R{rank}"
        packet_dir.mkdir(parents=True, exist_ok=True)

        # Content-keyed: filename encodes sorted basis unique indices
        existing = set(p.stem for p in packet_dir.glob("basis_*.npz"))

        basis_builder = BasisBuilder(
            self.db, self.config, rank, rng=np.random.default_rng(rank)
        )
        count = 0
        total_hits = 0

        for basis in basis_builder.build_bases():
            if count >= self.config.max_bases_per_rank:
                break

            with self._lock:
                self.state["basis"] = basis_builder.stats.__dict__.copy()

            # Content-based key: hash of sorted unique indices → deterministic name
            idx_key = ",".join(str(i) for i in sorted(basis.indices))
            basis_hash = hashlib.sha256(idx_key.encode()).hexdigest()[:12]
            packet_name = f"basis_{count:04d}_{basis_hash}"
            if packet_name in existing:
                count += 1
                continue

            hit_indices = self.db.query_subspace_dedup(basis.V_basis)
            basis_record_indices = self._expand_basis_indices(basis.indices)

            # Atomic write: temp file → rename (prevents corrupt partial writes)
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".npz", dir=str(packet_dir)
            )
            os.close(tmp_fd)
            try:
                np.savez_compressed(
                    tmp_path,
                    rank=np.array([rank], dtype=np.int32),
                    basis_unique_indices=np.array(basis.indices, dtype=np.int64),
                    basis_record_indices=np.array(basis_record_indices, dtype=np.int64),
                    V_basis=basis.V_basis,
                    V_perp=basis.V_perp,
                    sigma_basis=basis.sigma_basis,
                    dependent_estimate=np.array([basis.dependent_estimate], dtype=np.int64),
                    hit_record_indices=hit_indices.astype(np.int64),
                )
                final_path = packet_dir / f"{packet_name}.npz"
                os.replace(tmp_path, str(final_path))
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            total_hits += len(hit_indices)
            count += 1

            with self._lock:
                self.state["near_miss"] = (
                    f"R{rank} pkt {count}: {len(hit_indices):,} hits  "
                    f"(avg {total_hits / count:,.0f})"
                )

        return count


# ═══════════════════════════════════════════════════════════════
#  STAGE 2 — Assemble: load packets → assembly DFS → gate check
# ═══════════════════════════════════════════════════════════════

def _s2_worker_init(db_path: str, config):
    global _worker_db, _worker_config
    _worker_config = config
    _worker_db = TermDB.__new__(TermDB)
    # readonly=True: use mmap_mode='r' so all worker processes share OS pages
    # (avoids N x 11 GB RAM copies on Windows). silent via no-op callback.
    _worker_db._load(db_path, progress_callback=lambda _: None, readonly=True)


def _s2_worker_process_packet(packet_path: str) -> tuple[str, list[dict], dict, str | None]:
    db = _worker_db
    config = _worker_config

    data = np.load(packet_path, allow_pickle=False)
    rank = int(data["rank"][0])
    basis_record_indices = data["basis_record_indices"].tolist()
    hit_record_indices = data["hit_record_indices"]
    dependent_estimate = int(data["dependent_estimate"][0])

    assembler = DependentAssembler(db, config)
    solutions = assembler.assemble(rank, basis_record_indices, hit_record_indices)

    stats = {
        "bases_tested": 1,
        "hits": len(hit_record_indices),
        "gate2_passes": assembler.stats.gate2_passes,
        "gate3_passes": assembler.stats.gate3_passes,
        "configs_tested": assembler.stats.configs_tested,
    }

    near_miss = None
    if not solutions:
        near_miss = f"hits={len(hit_record_indices):,}, dep_est={dependent_estimate:,}"

    return packet_path, solutions, stats, near_miss


class AssembleEngine:
    """Stage 2: load work packets from disk, run assembly + gates in parallel."""

    def __init__(self, config, dashboard=None):
        self.config = config
        self.dashboard = dashboard
        self.state = {
            "current_rank": None,
            "phase": "init",
            "elapsed": 0.0,
            "basis": {},
            "assembly": {},
            "near_miss": "-",
            "workers_active": 0,
        }
        self.start_time = 0.0
        self.solutions: dict[int, list[dict]] = {}
        self._lock = threading.Lock()

    def run(self) -> dict[int, list[dict]]:
        self.config.validate()
        self.start_time = time.perf_counter()
        if self.dashboard:
            self.dashboard.start(self.state)

        self._run_error: Exception | None = None

        def _assemble_loop():
            try:
                for rank in self.config.target_ranks:
                    self._assemble_rank(rank)
            except Exception as exc:
                self._run_error = exc

        worker = threading.Thread(target=_assemble_loop, daemon=True)
        worker.start()

        while worker.is_alive():
            used, avail = _get_ram_gb()
            with self._lock:
                self.state["ram_used_gb"] = used
                self.state["ram_avail_gb"] = avail
                self.state["elapsed"] = time.perf_counter() - self.start_time
            if self.dashboard:
                self.dashboard.refresh()
            time.sleep(0.15)

        worker.join()
        if self.dashboard:
            self.dashboard.stop()
        if self._run_error:
            raise self._run_error
        return self.solutions

    def _assemble_rank(self, rank: int) -> None:
        self.solutions.setdefault(rank, [])

        packet_dir = Path(self.config.db_path).parent / PACKET_DIR / f"R{rank}"
        if not packet_dir.exists():
            return

        packets = sorted(packet_dir.glob("basis_*.npz"))
        if not packets:
            return

        with self._lock:
            self.state["current_rank"] = rank
            self.state["phase"] = "assemble"
            self.state["assembly"] = {}
            self.state["basis"] = {
                "seeds_total": len(packets),
                "seeds_processed": 0,
                "built_bases": len(packets),
            }

        # Track completed packets via marker files
        done_dir = packet_dir / "done"
        done_dir.mkdir(exist_ok=True)
        done_set = set(p.stem for p in done_dir.glob("*.done"))
        pending = [p for p in packets if p.stem not in done_set]

        if not pending:
            with self._lock:
                self.state["basis"]["seeds_processed"] = len(packets)
            return

        n_workers = max(1, self.config.n_workers)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_s2_worker_init,
            initargs=(self.config.db_path, self.config),
            mp_context=multiprocessing.get_context('spawn'),
        ) as pool:
            futures: dict = {}
            packet_iter = iter(pending)

            def _submit_next() -> bool:
                try:
                    pkt = next(packet_iter)
                except StopIteration:
                    return False
                fut = pool.submit(_s2_worker_process_packet, str(pkt))
                futures[fut] = pkt
                return True

            for _ in range(min(n_workers, len(pending))):
                _submit_next()

            with self._lock:
                self.state["workers_active"] = len(futures)

            done_this_rank = False
            while futures and not done_this_rank:
                newly_done = [f for f in list(futures) if f.done()]
                if not newly_done:
                    time.sleep(0.05)
                    continue

                for fut in newly_done:
                    pkt = futures.pop(fut)

                    try:
                        packet_path, worker_solutions, worker_stats, near_miss_msg = fut.result()
                    except Exception as exc:
                        with self._lock:
                            self.state["near_miss"] = f"ERROR: {pkt.name}: {exc}"
                        _submit_next()
                        continue

                    (done_dir / f"{pkt.stem}.done").touch()

                    with self._lock:
                        self.state["basis"]["seeds_processed"] = (
                            self.state["basis"].get("seeds_processed", 0) + 1
                        )
                        self.state["workers_active"] = len(futures)

                        shared = self.state.get("assembly", {})
                        shared["bases_tested"] = shared.get("bases_tested", 0) + 1
                        n = shared["bases_tested"]
                        old_avg = shared.get("avg_hits", 0.0)
                        shared["avg_hits"] = ((n - 1) * old_avg + float(worker_stats["hits"])) / n
                        shared["gate2_passes"] = shared.get("gate2_passes", 0) + worker_stats["gate2_passes"]
                        shared["gate3_passes"] = shared.get("gate3_passes", 0) + worker_stats["gate3_passes"]
                        shared["configs_tested"] = shared.get("configs_tested", 0) + worker_stats["configs_tested"]
                        self.state["assembly"] = shared

                        if worker_solutions:
                            for solution in worker_solutions:
                                sol_path = self._solution_path(rank)
                                save_solution(solution, sol_path)
                                verify_phase2_solution(solution)
                                self.solutions[rank].append(solution)
                                if len(self.solutions[rank]) >= self.config.max_solutions_per_rank:
                                    break
                        elif near_miss_msg:
                            self.state["near_miss"] = near_miss_msg

                    if len(self.solutions[rank]) >= self.config.max_solutions_per_rank:
                        for f in futures:
                            f.cancel()
                        futures.clear()
                        done_this_rank = True
                        break

                    _submit_next()

                with self._lock:
                    self.state["workers_active"] = len(futures)

    def _solution_path(self, rank: int) -> str:
        path = Path(self.config.solution_file)
        return str(path.with_name(f"{path.stem}_R{rank}{path.suffix}"))