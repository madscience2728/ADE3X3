"""Selection queries: tournament, elitism, pruning, archival."""

import sqlite3
import time

from .blob import blob_to_factors, factors_to_blob, compute_support_hash


def select_parents(
    conn: sqlite3.Connection,
    island: int,
    k: int,
    tournament_size: int = 5,
) -> list[sqlite3.Row]:
    """Tournament selection: pick k parents from a virtual island."""
    rows = conn.execute(
        """SELECT * FROM candidates
           WHERE virtual_island = ? AND fitness_fp32 IS NOT NULL
           ORDER BY fitness_fp32 ASC
           LIMIT 500""",
        (island,),
    ).fetchall()
    if not rows:
        return []

    import numpy as np
    rng = np.random.default_rng()
    selected = []
    for _ in range(k):
        contenders_idx = rng.choice(len(rows), size=min(tournament_size, len(rows)), replace=False)
        best = min(contenders_idx, key=lambda i: rows[i]["fitness_fp32"])
        selected.append(rows[best])
    return selected


def select_elites(conn: sqlite3.Connection, island: int, k: int) -> list[sqlite3.Row]:
    """Return the top-k candidates by fitness_fp32 from an island."""
    return conn.execute(
        """SELECT * FROM candidates
           WHERE virtual_island = ? AND fitness_fp32 IS NOT NULL
           ORDER BY fitness_fp32 ASC
           LIMIT ?""",
        (island, k),
    ).fetchall()


def select_global_elites(conn: sqlite3.Connection, k: int) -> list[sqlite3.Row]:
    """Return the top-k candidates across all islands."""
    return conn.execute(
        """SELECT * FROM candidates
           WHERE fitness_fp32 IS NOT NULL
           ORDER BY fitness_fp32 ASC
           LIMIT ?""",
        (k,),
    ).fetchall()


def fetch_pending(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Fetch pending candidates for evaluation."""
    return conn.execute(
        """SELECT id, factors_blob, virtual_island FROM candidates
           WHERE status = 'pending'
           ORDER BY id ASC
           LIMIT ?""",
        (limit,),
    ).fetchall()


def update_fitness_batch(
    conn: sqlite3.Connection,
    ids: list[int],
    fitness_fp32: list[float],
    tier: int = 1,
) -> None:
    """Batch update fitness_fp32 and status after GPU screening."""
    now = time.time()
    status = "screened" if tier == 1 else "refined"
    conn.executemany(
        """UPDATE candidates
           SET fitness_fp32 = ?, status = ?, tier_reached = ?, evaluated_at = ?
           WHERE id = ?""",
        [(float(f), status, tier, now, cid) for cid, f in zip(ids, fitness_fp32)],
    )
    conn.commit()


def fetch_survivors(conn: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    """Fetch candidates that passed Tier 1 screening."""
    return conn.execute(
        """SELECT id, factors_blob FROM candidates
           WHERE status = 'screened' AND fitness_fp32 < ? AND tier_reached = 1
           ORDER BY fitness_fp32 ASC""",
        (threshold,),
    ).fetchall()


def fetch_refine_candidates(conn: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    """Fetch candidates for CPU refinement, preferring GPU-minimax'd (tier 2)."""
    return conn.execute(
        """SELECT id, factors_blob, fitness_fp32, virtual_island FROM candidates
           WHERE fitness_fp32 IS NOT NULL AND fitness_fp32 < ? AND tier_reached < 3
           ORDER BY tier_reached DESC, fitness_fp32 ASC
           LIMIT 100""",
        (threshold,),
    ).fetchall()


def update_refined(
    conn: sqlite3.Connection,
    cid: int,
    factors_blob: bytes,
    fitness_fp64: float,
    fro_residual: float,
) -> None:
    """Update a candidate after Tier 3 CPU refinement."""
    now = time.time()
    conn.execute(
        """UPDATE candidates
           SET factors_blob = ?, fitness_fp64 = ?, fro_residual = ?,
               status = 'refined', tier_reached = 3, evaluated_at = ?
           WHERE id = ?""",
        (factors_blob, fitness_fp64, fro_residual, now, cid),
    )
    conn.commit()


def prune_duplicates(conn: sqlite3.Connection) -> int:
    """Delete duplicate candidates (same support_hash), keeping the best fitness."""
    result = conn.execute(
        """DELETE FROM candidates
           WHERE id NOT IN (
               SELECT id FROM (
                   SELECT id, ROW_NUMBER() OVER (
                       PARTITION BY support_hash
                       ORDER BY COALESCE(fitness_fp32, 999) ASC
                   ) AS rn
                   FROM candidates
               ) WHERE rn = 1
           )"""
    )
    conn.commit()
    return result.rowcount


def prune_blacklisted(conn: sqlite3.Connection) -> int:
    """Delete candidates whose support_hash is blacklisted."""
    result = conn.execute(
        """DELETE FROM candidates
           WHERE support_hash IN (SELECT support_hash FROM blacklist)"""
    )
    conn.commit()
    return result.rowcount


def archive_to_shadow(conn: sqlite3.Connection, cid: int) -> None:
    """Copy a candidate to the shadow archive (INSERT OR IGNORE)."""
    row = conn.execute(
        "SELECT factors_blob, support_hash, fitness_fp64 FROM candidates WHERE id = ?",
        (cid,),
    ).fetchone()


def fetch_shadow_pool(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    """Fetch top shadow candidates for reinject diversity."""
    return conn.execute(
        "SELECT factors_blob FROM shadow ORDER BY fitness_fp64 ASC LIMIT ?",
        (limit,),
    ).fetchall()
    if row and row["fitness_fp64"] is not None:
        conn.execute(
            """INSERT OR IGNORE INTO shadow
               (factors_blob, support_hash, fitness_fp64, archived_at)
               VALUES (?, ?, ?, ?)""",
            (row["factors_blob"], row["support_hash"], row["fitness_fp64"], time.time()),
        )
        conn.commit()


def pending_count(conn: sqlite3.Connection) -> int:
    """Count pending candidates (for backpressure)."""
    row = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status = 'pending'").fetchone()
    return row["cnt"]


def island_counts(conn: sqlite3.Connection, n_islands: int) -> list[int]:
    """Return total population count per island, including pending rows."""
    counts = [0] * n_islands
    rows = conn.execute(
        """SELECT virtual_island, COUNT(*) as cnt
           FROM candidates
           GROUP BY virtual_island"""
    ).fetchall()
    for row in rows:
        island = row["virtual_island"]
        if 0 <= island < n_islands:
            counts[island] = row["cnt"]
    return counts


def island_stats(conn: sqlite3.Connection, n_islands: int) -> list[dict]:
    """Per-island statistics."""
    stats = []
    for island in range(n_islands):
        row = conn.execute(
            """SELECT
                 COUNT(*) as cnt,
                 SUM(CASE WHEN fitness_fp32 IS NOT NULL THEN 1 ELSE 0 END) as scored,
                 MIN(fitness_fp32) as best,
                 AVG(fitness_fp32) as mean,
                 MAX(fitness_fp32) as worst
               FROM candidates
               WHERE virtual_island = ?""",
            (island,),
        ).fetchone()
        stats.append({
            "island": island,
            "count": row["cnt"],
            "scored": row["scored"] or 0,
            "best": row["best"],
            "mean": row["mean"],
            "worst": row["worst"],
        })
    return stats


def insert_candidates(
    conn: sqlite3.Connection,
    candidates: list[tuple[bytes, str, str, str, int, int]],
) -> None:
    """Batch insert new candidates.

    Each tuple: (factors_blob, support_hash, support_sig, origin, virtual_island, generation)
    """
    now = time.time()
    conn.executemany(
        """INSERT INTO candidates
           (factors_blob, support_hash, support_sig, origin, virtual_island,
            generation, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
        [(*c, now) for c in candidates],
    )
    conn.commit()


def log_event(
    conn: sqlite3.Connection,
    event: str,
    generation: int | None = None,
    detail_json: str | None = None,
    wall_seconds: float | None = None,
) -> None:
    """Insert a run_log entry."""
    conn.execute(
        """INSERT INTO run_log (event, generation, detail_json, wall_seconds, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (event, generation, detail_json, wall_seconds, time.time()),
    )
    conn.commit()


# ── Tier 2 GPU minimax helpers ───────────────────────────────────────

def update_minimax_batch(
    conn: sqlite3.Connection,
    ids: list[int],
    blobs: list[bytes],
    fitness_list: list[float],
) -> None:
    """Batch update candidates after GPU minimax (Tier 2)."""
    now = time.time()
    conn.executemany(
        """UPDATE candidates
           SET factors_blob = ?, minimax_improved = ?, tier_reached = 2, evaluated_at = ?
           WHERE id = ?""",
        [(blob, float(f), now, cid) for cid, blob, f in zip(ids, blobs, fitness_list)],
    )
    conn.commit()


# ── Power-law island migration ───────────────────────────────────────

def promote_best(
    conn: sqlite3.Connection,
    src_island: int,
    dst_island: int,
    k: int = 1,
    dst_capacity: int | None = None,
) -> int:
    """Promote the best k candidates from a larger island to a smaller one."""
    if dst_capacity is not None:
        current = conn.execute(
            "SELECT COUNT(*) as cnt FROM candidates WHERE virtual_island = ?",
            (dst_island,),
        ).fetchone()["cnt"]
        k = min(k, max(0, dst_capacity - current))
        if k <= 0:
            return 0
    rows = conn.execute(
        """SELECT id FROM candidates
           WHERE virtual_island = ? AND fitness_fp32 IS NOT NULL
           ORDER BY COALESCE(fitness_fp64, fitness_fp32) ASC
           LIMIT ?""",
        (src_island, k),
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    conn.executemany(
        "UPDATE candidates SET virtual_island = ? WHERE id = ?",
        [(dst_island, cid) for cid in ids],
    )
    conn.commit()
    return len(ids)


def demote_random(
    conn: sqlite3.Connection,
    src_island: int,
    dst_island: int,
    k: int = 1,
    dst_capacity: int | None = None,
) -> int:
    """Demote k random candidates from a smaller island to a larger one (diversity injection)."""
    if dst_capacity is not None:
        current = conn.execute(
            "SELECT COUNT(*) as cnt FROM candidates WHERE virtual_island = ?",
            (dst_island,),
        ).fetchone()["cnt"]
        k = min(k, max(0, dst_capacity - current))
        if k <= 0:
            return 0
    rows = conn.execute(
        """SELECT id FROM candidates
           WHERE virtual_island = ? AND fitness_fp32 IS NOT NULL
           ORDER BY RANDOM()
           LIMIT ?""",
        (src_island, k),
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    conn.executemany(
        "UPDATE candidates SET virtual_island = ? WHERE id = ?",
        [(dst_island, cid) for cid in ids],
    )
    conn.commit()
    return len(ids)
