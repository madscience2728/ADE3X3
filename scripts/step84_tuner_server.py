from __future__ import annotations

import csv
import json
import mimetypes
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
HOST = "127.0.0.1"
PORT = 8765
STEP84_SCRIPT = REPO_ROOT / "src" / "ade3x3" / "steps" / "ade3x3_step84_metaheuristic_rank19_search.py"
DEFAULT_EXPORTS = REPO_ROOT / "outputs" / "exports"
BATCH_EXPORTS = DEFAULT_EXPORTS / "step84_batches"
LOG_PATH = DEFAULT_EXPORTS / "step84_evolution_log.csv"
SUMMARY_PATH = DEFAULT_EXPORTS / "step84_summary.json"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def tail_list(items: deque[str], limit: int = 120) -> list[str]:
    return list(items)[-limit:]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def history_rows(log_path: Path) -> dict[int, dict]:
    if not log_path.exists():
        return {}

    per_generation: dict[int, dict] = {}
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            generation = parse_int(row.get("generation"))
            if generation is None:
                continue
            slot = per_generation.setdefault(
                generation,
                {
                    "generation": generation,
                    "best_fitness": None,
                    "mean_fitness_sum": 0.0,
                    "mean_fitness_count": 0,
                    "worst_fitness": None,
                    "wall_seconds_cumulative": None,
                    "signature_333_count": 0,
                    "shadow_pool_size": 0,
                },
            )

            best_fitness = parse_float(row.get("best_fitness"))
            if best_fitness is not None:
                current_best = slot["best_fitness"]
                slot["best_fitness"] = best_fitness if current_best is None else min(current_best, best_fitness)

            mean_fitness = parse_float(row.get("mean_fitness"))
            if mean_fitness is not None:
                slot["mean_fitness_sum"] += mean_fitness
                slot["mean_fitness_count"] += 1

            worst_fitness = parse_float(row.get("worst_fitness"))
            if worst_fitness is not None:
                current_worst = slot["worst_fitness"]
                slot["worst_fitness"] = worst_fitness if current_worst is None else max(current_worst, worst_fitness)

            wall_seconds = parse_float(row.get("wall_seconds_cumulative"))
            if wall_seconds is not None:
                current_wall = slot["wall_seconds_cumulative"]
                slot["wall_seconds_cumulative"] = wall_seconds if current_wall is None else max(current_wall, wall_seconds)

            signature_count = parse_int(row.get("signature_333_count"))
            if signature_count is not None:
                slot["signature_333_count"] += signature_count

            shadow_pool_size = parse_int(row.get("shadow_pool_size"))
            if shadow_pool_size is not None:
                slot["shadow_pool_size"] += shadow_pool_size

    return per_generation


def aggregate_history(log_paths: list[Path], limit: int = 600) -> list[dict]:
    combined: dict[int, dict] = {}
    for log_path in log_paths:
        for generation, row in history_rows(log_path).items():
            slot = combined.setdefault(
                generation,
                {
                    "generation": generation,
                    "best_fitness": None,
                    "mean_fitness_sum": 0.0,
                    "mean_fitness_count": 0,
                    "worst_fitness": None,
                    "wall_seconds_cumulative": None,
                    "signature_333_count": 0,
                    "shadow_pool_size": 0,
                },
            )
            best_fitness = row.get("best_fitness")
            if best_fitness is not None:
                current_best = slot["best_fitness"]
                slot["best_fitness"] = best_fitness if current_best is None else min(current_best, best_fitness)
            if row.get("mean_fitness_count"):
                slot["mean_fitness_sum"] += row["mean_fitness_sum"]
                slot["mean_fitness_count"] += row["mean_fitness_count"]
            worst_fitness = row.get("worst_fitness")
            if worst_fitness is not None:
                current_worst = slot["worst_fitness"]
                slot["worst_fitness"] = worst_fitness if current_worst is None else max(current_worst, worst_fitness)
            wall_seconds = row.get("wall_seconds_cumulative")
            if wall_seconds is not None:
                current_wall = slot["wall_seconds_cumulative"]
                slot["wall_seconds_cumulative"] = wall_seconds if current_wall is None else max(current_wall, wall_seconds)
            slot["signature_333_count"] += row.get("signature_333_count", 0)
            slot["shadow_pool_size"] += row.get("shadow_pool_size", 0)

    history = []
    for generation in sorted(combined):
        slot = combined[generation]
        mean_fitness = None
        if slot["mean_fitness_count"]:
            mean_fitness = slot["mean_fitness_sum"] / slot["mean_fitness_count"]
        history.append(
            {
                "generation": generation,
                "best_fitness": slot["best_fitness"],
                "mean_fitness": mean_fitness,
                "worst_fitness": slot["worst_fitness"],
                "wall_seconds_cumulative": slot["wall_seconds_cumulative"],
                "signature_333_count": slot["signature_333_count"],
                "shadow_pool_size": slot["shadow_pool_size"],
            }
        )
    return history[-limit:]


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def latest_checkpoint(export_dir: Path) -> Path | None:
    candidates = sorted(export_dir.glob("step84_checkpoint_gen*.json"))
    if not candidates:
        return None
    return candidates[-1]


@dataclass
class RunCopy:
    index: int
    export_dir: Path
    log_path: Path
    summary_path: Path
    process: subprocess.Popen
    seed: int
    stdout_lines: deque[str] = field(default_factory=lambda: deque(maxlen=180))
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=120))
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None


@dataclass
class RunManager:
    copies: list[RunCopy] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float | None = None
    finished_at: float | None = None
    env: dict[str, str] = field(default_factory=dict)
    batch_dir: Path | None = None
    continued_from_previous: bool = False
    resumed_copy_count: int = 0

    def _reader(self, stream, buffer: deque[str], prefix: str) -> None:
        for line in iter(stream.readline, ""):
            text = line.rstrip("\r\n")
            if text:
                with self.lock:
                    buffer.append(f"{prefix}{text}")
        stream.close()

    def is_running(self) -> bool:
        return any(copy.process.poll() is None for copy in self.copies)

    def start(self, env_overrides: dict[str, str], copies: int = 1, fresh_start: bool = False) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Step 84 is already running")
            if copies <= 0:
                raise RuntimeError("Copy count must be positive")

            previous_copies = list(self.copies)
            previous_batch_dir = self.batch_dir
            explicit_resume = bool(str(env_overrides.get("STEP84_RESUME_CHECKPOINT", "")).strip()) and not fresh_start

            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.update(env_overrides)

            self.started_at = time.time()
            self.finished_at = None
            self.env = dict(env_overrides)
            self.copies = []
            self.continued_from_previous = False
            self.resumed_copy_count = 0

            BATCH_EXPORTS.mkdir(parents=True, exist_ok=True)
            auto_continue = previous_copies and not explicit_resume and not fresh_start
            if auto_continue and previous_batch_dir is not None:
                self.batch_dir = previous_batch_dir
                self.continued_from_previous = True
            else:
                batch_stamp = time.strftime("%Y%m%d_%H%M%S")
                self.batch_dir = BATCH_EXPORTS / f"run_{batch_stamp}_{int(self.started_at * 1000) % 1000:03d}"
                self.batch_dir.mkdir(parents=True, exist_ok=True)

            assert self.batch_dir is not None
            self.batch_dir.mkdir(parents=True, exist_ok=True)

            base_seed_raw = env_overrides.get("STEP84_SEED")
            base_seed = int(base_seed_raw) if base_seed_raw is not None and str(base_seed_raw).strip() else 8401001

            for index in range(copies):
                export_dir = self.batch_dir / f"copy_{index + 1:03d}"
                export_dir.mkdir(parents=True, exist_ok=True)
                copy_env = env.copy()
                copy_env["STEP84_EXPORTS_DIR"] = str(export_dir)
                copy_env["STEP84_SEED"] = str(base_seed + index)
                if explicit_resume:
                    copy_env["STEP84_RESUME_CHECKPOINT"] = str(env_overrides["STEP84_RESUME_CHECKPOINT"])
                elif auto_continue and index < len(previous_copies):
                    checkpoint_path = latest_checkpoint(previous_copies[index].export_dir)
                    if checkpoint_path is not None:
                        copy_env["STEP84_RESUME_CHECKPOINT"] = str(checkpoint_path)
                        self.resumed_copy_count += 1
                process = subprocess.Popen(
                    [str(PYTHON_EXE), str(STEP84_SCRIPT)],
                    cwd=str(REPO_ROOT),
                    env=copy_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                copy = RunCopy(
                    index=index + 1,
                    export_dir=export_dir,
                    log_path=export_dir / "step84_evolution_log.csv",
                    summary_path=export_dir / "step84_summary.json",
                    process=process,
                    seed=base_seed + index,
                    started_at=time.time(),
                )
                self.copies.append(copy)

                assert process.stdout is not None
                assert process.stderr is not None
                prefix = f"[copy {index + 1:02d}] "
                threading.Thread(target=self._reader, args=(process.stdout, copy.stdout_lines, prefix), daemon=True).start()
                threading.Thread(target=self._reader, args=(process.stderr, copy.stderr_lines, prefix), daemon=True).start()
                threading.Thread(target=self._watcher, args=(copy,), daemon=True).start()

    def _watcher(self, copy: RunCopy) -> None:
        code = copy.process.wait()
        with self.lock:
            copy.exit_code = code
            copy.finished_at = time.time()
            if not self.is_running():
                self.finished_at = time.time()

    def stop(self) -> None:
        with self.lock:
            for copy in self.copies:
                if copy.process.poll() is None:
                    copy.process.terminate()

    def payload(self) -> dict:
        with self.lock:
            running = self.is_running()
            env = dict(self.env)
            batch_dir = str(self.batch_dir) if self.batch_dir else None
            started_at = self.started_at
            finished_at = self.finished_at
            copies = list(self.copies)

        log_paths: list[Path] = []
        summaries: list[dict] = []
        stdout: list[str] = []
        stderr: list[str] = []
        copy_payloads: list[dict] = []
        for copy in copies:
            log_paths.append(copy.log_path)
            summary = load_summary(copy.summary_path)
            if summary is not None:
                summaries.append({**summary, "copy_index": copy.index, "seed": copy.seed, "export_dir": str(copy.export_dir)})
            stdout.extend(tail_list(copy.stdout_lines, 8))
            stderr.extend(tail_list(copy.stderr_lines, 6))
            copy_payloads.append(
                {
                    "index": copy.index,
                    "pid": copy.process.pid,
                    "running": copy.process.poll() is None,
                    "exit_code": copy.exit_code,
                    "seed": copy.seed,
                    "export_dir": str(copy.export_dir),
                    "best_fitness_ever": summary.get("best_fitness_ever") if summary else None,
                    "best_support_signature": summary.get("best_support_signature") if summary else None,
                    "total_generations": summary.get("total_generations") if summary else None,
                }
            )

        history = aggregate_history(log_paths if log_paths else [LOG_PATH])
        latest = history[-1] if history else None
        best_summary = None
        if summaries:
            best_summary = min(
                summaries,
                key=lambda item: (
                    item.get("best_fitness_ever") if item.get("best_fitness_ever") is not None else float("inf"),
                    item.get("total_wall_seconds") if item.get("total_wall_seconds") is not None else float("inf"),
                ),
            )
        elif not copies:
            best_summary = load_summary(SUMMARY_PATH)

        return {
            "running": running,
            "started_at": started_at,
            "finished_at": finished_at,
            "env": env,
            "batch_dir": batch_dir,
            "continued_from_previous": self.continued_from_previous,
            "resumed_copy_count": self.resumed_copy_count,
            "copy_count": len(copy_payloads),
            "running_count": sum(1 for copy in copy_payloads if copy["running"]),
            "copies": copy_payloads,
            "stdout_tail": stdout[-120:],
            "stderr_tail": stderr[-80:],
            "history": history,
            "latest": latest,
            "summary": best_summary,
        }


RUN_MANAGER = RunManager()


class Step84TunerHandler(BaseHTTPRequestHandler):
    server_version = "Step84Tuner/1.0"

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        relative = path.lstrip("/") or "step84_tuner.html"
        file_path = (DOCS_DIR / relative).resolve()
        if DOCS_DIR not in file_path.parents and file_path != DOCS_DIR:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(RUN_MANAGER.payload())
            return
        if parsed.path in {"/", "/step84_tuner.html", "/step84_tuner.css", "/step84_tuner.js"}:
            target = "/step84_tuner.html" if parsed.path == "/" else parsed.path
            self._serve_static(target)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run/start":
            try:
                payload = self._read_json_body()
                env = {str(key): str(value) for key, value in payload.get("env", {}).items()}
                copies = int(payload.get("copies", 1))
                fresh_start = bool(payload.get("fresh_start", False))
                RUN_MANAGER.start(env, copies=copies, fresh_start=fresh_start)
            except RuntimeError as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.CONFLICT)
                return
            except Exception as error:  # pragma: no cover
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "state": RUN_MANAGER.payload()})
            return

        if parsed.path == "/api/run/stop":
            RUN_MANAGER.stop()
            self._send_json({"ok": True, "state": RUN_MANAGER.payload()})
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    if not PYTHON_EXE.exists():
        raise SystemExit(f"Missing Python executable: {PYTHON_EXE}")
    if not STEP84_SCRIPT.exists():
        raise SystemExit(f"Missing Step 84 script: {STEP84_SCRIPT}")

    server = ThreadingHTTPServer((HOST, PORT), Step84TunerHandler)
    print(f"Step 84 tuner server running at http://{HOST}:{PORT}/step84_tuner.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUN_MANAGER.stop()
        server.server_close()


if __name__ == "__main__":
    main()