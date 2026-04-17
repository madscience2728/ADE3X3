"""
fusion_search/serve.py — HTTP server for the λ landscape dashboard.

Endpoints:
  GET  /                      — HTML dashboard (ui.html)
  GET  /api/results           — full results.jsonl as JSON array, sorted by λ
  GET  /api/status            — {"running": bool, "pid": int|null, "n_evaluated": int}
  POST /api/start             — launch lambda_search.py in background
  POST /api/stop              — terminate running lambda_search.py
  POST /api/archive           — copy results.jsonl → results_YYYYMMDD_HHMMSS.jsonl
  POST /api/clear             — delete results.jsonl (refused if search is running)
  GET  /api/stream            — SSE stream: pushes updated results every 3 s
"""
from __future__ import annotations

import http.server
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs

# Use the threading mixin so every request gets its own thread.
# This is essential: the SSE /api/stream handler blocks indefinitely,
# and without threading it locks out all POST requests (Start button).
class _ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

PORT = 8780
SEARCH_MODULE    = "fusion_search.lambda_search"
MS_MODULE        = "fusion_search.multi_sector"
ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).parent
RESULTS_PATH = HERE / "results.jsonl"
MS_RESULTS_PATH  = HERE / "data" / "multi_sector_sweep.jsonl"
MS_STATUS_PATH   = HERE / "data" / "multi_sector_status.json"
UI_HTML = HERE / "ui.html"
PID_PATH = HERE / "serve.pid"


def _resolve_python() -> pathlib.Path:
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
        pathlib.Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            return c
    return pathlib.Path(sys.executable)


PYTHON = _resolve_python()

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()
_started_at: float | None = None
_exit_code: int | None = None

# Multi-sector search process state
_ms_proc: subprocess.Popen | None = None
_ms_proc_lock = threading.Lock()
_ms_started_at: float | None = None
_ms_exit_code: int | None = None


# ---------------------------------------------------------------------------
# Results reader
# ---------------------------------------------------------------------------

def _load_results() -> list[dict]:
    records: dict[float, dict] = {}
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = round(float(rec["lambda"]), 6)
                records[key] = rec
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return sorted(records.values(), key=lambda r: r["lambda"])


def _load_ms_results() -> list[dict]:
    """Load multi-sector sweep records (all phases, not deduplicated)."""
    records: list[dict] = []
    if not MS_RESULTS_PATH.exists():
        return records
    with MS_RESULTS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _ms_status() -> dict:
    global _ms_proc, _ms_exit_code
    with _ms_proc_lock:
        if _ms_proc is not None:
            code = _ms_proc.poll()
            if code is not None:
                _ms_exit_code = int(code)
                _ms_proc = None
        running = _ms_proc is not None
        pid = _ms_proc.pid if _ms_proc is not None else None

    n_pts = len(_load_ms_results())

    # Read persisted status JSON if present
    best_residual = None
    best_A1 = None
    broke_floor = None
    if MS_STATUS_PATH.exists():
        try:
            st = json.loads(MS_STATUS_PATH.read_text(encoding="utf-8"))
            fst = st.get("fusion", {})
            best_residual = fst.get("best_residual")
            best_A1       = fst.get("best_A1")
            broke_floor   = fst.get("broke_floor")
        except Exception:
            pass

    return {
        "running":       running,
        "pid":           pid,
        "n_evaluated":   n_pts,
        "exit_code":     _ms_exit_code,
        "started_at":    _ms_started_at,
        "best_residual": best_residual,
        "best_A1":       best_A1,
        "broke_floor":   broke_floor,
    }


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _search_status() -> dict:
    global _proc, _exit_code
    with _proc_lock:
        if _proc is not None:
            code = _proc.poll()
            if code is not None:
                _exit_code = int(code)
                _proc = None
        running = _proc is not None
        pid = _proc.pid if _proc is not None else None
    results = _load_results()
    return {
        "running": running,
        "pid": pid,
        "n_evaluated": len(results),
        "exit_code": _exit_code,
        "started_at": _started_at,
    }


def _start_search(grid_lo: float = -5.0, grid_hi: float = 5.0,
                  grid_step: float = 0.1, restarts: int = 20,
                  workers: int | None = None) -> dict:
    global _proc, _started_at, _exit_code
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return {"status": "already_running", "pid": _proc.pid}
        cmd = [
            str(PYTHON), "-m", SEARCH_MODULE,
            "--grid-lo", str(grid_lo),
            "--grid-hi", str(grid_hi),
            "--grid-step", str(grid_step),
            "--restarts", str(restarts),
            "--out", str(RESULTS_PATH),
        ]
        if workers is not None:
            cmd += ["--workers", str(workers)]
        kwargs: dict = {"cwd": str(ROOT)}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        _proc = subprocess.Popen(cmd, **kwargs)
        _started_at = time.time()
        _exit_code = None
    return {"status": "started", "pid": _proc.pid}


def _archive_results() -> dict:
    if not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0:
        return {"status": "empty", "message": "Nothing to archive — results.jsonl is empty or missing."}
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = RESULTS_PATH.parent / f"results_{ts}.jsonl"
    import shutil
    shutil.copy2(RESULTS_PATH, dest)
    return {"status": "ok", "archived_to": dest.name}


def _clear_results() -> dict:
    """Delete results.jsonl so the next run starts fresh. Refused if search is running."""
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return {"status": "error", "message": "Stop the search before clearing data."}
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    return {"status": "ok"}


def _kill_tree(pid: int) -> None:
    """
    On Windows, taskkill /F /T kills the whole job tree in one syscall.
    On POSIX,  the lambda_search process was started with start_new_session=True
    so all its workers share a process group; killpg terminates them in one shot.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def _stop_search() -> dict:
    global _proc
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            return {"status": "not_running"}
        pid = _proc.pid
        _kill_tree(pid)
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait(timeout=5)
        except Exception:
            pass
        _proc = None
    return {"status": "stopped", "pid": pid}


def _start_ms_search(
    lam: float = -0.46,
    n_random: int = 500,
    n_refine_starts: int = 10,
    restarts_random: int = 2,
    restarts_refine: int = 4,
    refine_maxfev: int = 40,
    workers: int | None = None,
) -> dict:
    global _ms_proc, _ms_started_at, _ms_exit_code
    with _ms_proc_lock:
        if _ms_proc is not None and _ms_proc.poll() is None:
            return {"status": "already_running", "pid": _ms_proc.pid}
        cmd = [
            str(PYTHON), "-m", MS_MODULE,
            "--lam",             str(lam),
            "--random",          str(n_random),
            "--refine",          str(n_refine_starts),
            "--restarts-random", str(restarts_random),
            "--restarts-refine", str(restarts_refine),
            "--refine-maxfev",   str(refine_maxfev),
            "--out",             str(MS_RESULTS_PATH),
        ]
        if workers is not None:
            cmd += ["--workers", str(workers)]
        kwargs: dict = {"cwd": str(ROOT)}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        _ms_proc = subprocess.Popen(cmd, **kwargs)
        _ms_started_at = time.time()
        _ms_exit_code = None
    return {"status": "started", "pid": _ms_proc.pid}


def _stop_ms_search() -> dict:
    global _ms_proc
    with _ms_proc_lock:
        if _ms_proc is None or _ms_proc.poll() is not None:
            return {"status": "not_running"}
        pid = _ms_proc.pid
        _kill_tree(pid)
        try:
            _ms_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ms_proc.kill()
            _ms_proc.wait(timeout=5)
        except Exception:
            pass
        _ms_proc = None
    return {"status": "stopped", "pid": pid}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args) -> None:  # suppress default access log
        pass

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last = ""
        try:
            while True:
                data = json.dumps({
                    "results":   _load_results(),
                    "status":    _search_status(),
                    "ms_status": _ms_status(),
                })
                if data != last:
                    msg = f"data: {data}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                    last = data
                time.sleep(3)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path in ("", "/"):
            html = UI_HTML.read_bytes() if UI_HTML.exists() else b"<h1>ui.html not found</h1>"
            self._send_html(html)

        elif path == "/api/results":
            self._send_json(_load_results())

        elif path == "/api/status":
            self._send_json(_search_status())

        elif path == "/api/ms-results":
            self._send_json(_load_ms_results())

        elif path == "/api/ms-status":
            self._send_json(_ms_status())

        elif path == "/api/stream":
            self._send_sse_stream()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(body) if body else {}
        except json.JSONDecodeError:
            params = {}

        if path == "/api/start":
            workers_raw = params.get("workers")
            result = _start_search(
                grid_lo=float(params.get("grid_lo", -5.0)),
                grid_hi=float(params.get("grid_hi", 5.0)),
                grid_step=float(params.get("grid_step", 0.1)),
                restarts=int(params.get("restarts", 20)),
                workers=int(workers_raw) if workers_raw is not None else None,
            )
            self._send_json(result)

        elif path == "/api/stop":
            self._send_json(_stop_search())

        elif path == "/api/start_ms":
            w_raw = params.get("workers")
            result = _start_ms_search(
                lam=float(params.get("lam", -0.46)),
                n_random=int(params.get("n_random", 500)),
                n_refine_starts=int(params.get("n_refine", 10)),
                restarts_random=int(params.get("restarts_random", 2)),
                restarts_refine=int(params.get("restarts_refine", 4)),
                refine_maxfev=int(params.get("refine_maxfev", 40)),
                workers=int(w_raw) if w_raw is not None else None,
            )
            self._send_json(result)

        elif path == "/api/stop_ms":
            self._send_json(_stop_ms_search())

        elif path == "/api/archive":
            self._send_json(_archive_results())

        elif path == "/api/clear":
            self._send_json(_clear_results())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(port: int = PORT, open_browser: bool = True) -> None:
    PID_PATH.write_text(
        json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8"
    )

    server = _ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"[fusion-ui] listening at {url}", flush=True)

    if open_browser:
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    def _cleanup(*_):
        _stop_search()
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    import atexit
    atexit.register(_cleanup)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    main(port=args.port, open_browser=not args.no_browser)
