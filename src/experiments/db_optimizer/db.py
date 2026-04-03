"""SQLite connection management with WAL mode and pragma tuning."""

import sqlite3
from pathlib import Path


def get_connection(db_path: Path | str, *, read_only: bool = False, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a tuned SQLite connection.

    - WAL journal mode for concurrent reads
    - Large cache (8 GB by default)
    - Normal synchronous (crash-safe but not paranoid)
    - 30 s busy timeout for writer contention
    """
    db_path = Path(db_path)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    uri = f"file:{db_path}"
    if read_only:
        uri += "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -8000000")  # 8 GB
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 1073741824")  # 1 GB mmap

    return conn


def get_memory_connection() -> sqlite3.Connection:
    """Open an in-memory SQLite connection (for tests)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn
