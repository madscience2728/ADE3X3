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
LOG_PATH = REPO_ROOT / "outputs" / "exports" / "step84_evolution_log.csv"
SUMMARY_PATH = REPO_ROOT / "outputs" / "exports" / "step84_summary.json"
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


def aggregate_history(log_path: Path, limit: int = 600) -> list[dict]:
    if not log_path.exists():
        return []

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

    history = []
    for generation in sorted(per_generation):
        slot = per_generation[generation]
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


@dataclass
class RunManager:
    process: subprocess.Popen | None = None
    stdout_lines: deque[str] = field(default_factory=lambda: deque(maxlen=600))
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=300))
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    def _reader(self, stream, buffer: deque[str]) -> None:
        for line in iter(stream.readline, ""):
            text = line.rstrip("\r\n")
            if text:
                with self.lock:
                    buffer.append(text)
        stream.close()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, env_overrides: dict[str, str]) -> None:
        with self.lock:
            if self.is_running():
                raise RuntimeError("Step 84 is already running")

            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.update(env_overrides)

            self.stdout_lines.clear()
            self.stderr_lines.clear()
            self.started_at = time.time()
            self.finished_at = None
            self.exit_code = None
            self.env = dict(env_overrides)

            self.process = subprocess.Popen(
                [str(PYTHON_EXE), str(STEP84_SCRIPT)],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            assert self.process.stdout is not None
            assert self.process.stderr is not None
            threading.Thread(target=self._reader, args=(self.process.stdout, self.stdout_lines), daemon=True).start()
            threading.Thread(target=self._reader, args=(self.process.stderr, self.stderr_lines), daemon=True).start()
            threading.Thread(target=self._watcher, daemon=True).start()

    def _watcher(self) -> None:
        process = self.process
        if process is None:
            return
        code = process.wait()
        with self.lock:
            self.exit_code = code
            self.finished_at = time.time()

    def stop(self) -> None:
        with self.lock:
            if not self.is_running():
                return
            assert self.process is not None
            self.process.terminate()

    def payload(self) -> dict:
        with self.lock:
            running = self.is_running()
            process = self.process
            pid = process.pid if process else None
            started_at = self.started_at
            finished_at = self.finished_at
            exit_code = self.exit_code
            env = dict(self.env)
            stdout = tail_list(self.stdout_lines)
            stderr = tail_list(self.stderr_lines)

        history = aggregate_history(LOG_PATH)
        summary = load_summary(SUMMARY_PATH)

        latest = history[-1] if history else None
        return {
            "running": running,
            "pid": pid,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "env": env,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "history": history,
            "latest": latest,
            "summary": summary,
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
                RUN_MANAGER.start(env)
            except RuntimeError as error:
                self._send_json({"ok": False, "error": str(error)}, HTTPStatus.CONFLICT)
                return
            except Exception as error:  # pragma: no cover - defensive path for UI use
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