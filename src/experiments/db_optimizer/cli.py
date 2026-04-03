"""CLI entry point for db_optimizer."""

import argparse
import json
import sys
from pathlib import Path

from .config import DB_PATH, N_ISLANDS


def cmd_run(args):
    from .orchestrator import run
    seed_paths = None
    if args.seed_dir:
        seed_dir = Path(args.seed_dir)
        seed_paths = sorted(seed_dir.glob("optimized_*.json"))
        if not seed_paths:
            print(f"[warn] No optimized_*.json files found in {seed_dir}")
    run(
        db_path=args.db,
        max_generations=args.generations,
        seed_paths=seed_paths,
        verbose=not args.quiet,
    )


def cmd_status(args):
    from .db import get_connection
    conn = get_connection(args.db, read_only=True)
    total = conn.execute("SELECT COUNT(*) as cnt FROM candidates").fetchone()["cnt"]
    pending = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='pending'").fetchone()["cnt"]
    screened = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='screened'").fetchone()["cnt"]
    refined = conn.execute("SELECT COUNT(*) as cnt FROM candidates WHERE status='refined'").fetchone()["cnt"]

    best = conn.execute(
        "SELECT MIN(COALESCE(fitness_fp64, fitness_fp32)) as best FROM candidates"
    ).fetchone()["best"]

    shadow_count = conn.execute("SELECT COUNT(*) as cnt FROM shadow").fetchone()["cnt"]
    blacklist_count = conn.execute("SELECT COUNT(*) as cnt FROM blacklist").fetchone()["cnt"]

    print(f"Database: {args.db}")
    print(f"Total candidates: {total}")
    print(f"  pending:  {pending}")
    print(f"  screened: {screened}")
    print(f"  refined:  {refined}")
    print(f"Best fitness: {best}")
    print(f"Shadow archive: {shadow_count}")
    print(f"Blacklisted: {blacklist_count}")

    # Per-island stats
    from .selector import island_stats
    stats = island_stats(conn, N_ISLANDS)
    print(f"\nIsland stats:")
    for s in stats:
        if s["count"] > 0:
            print(f"  Island {s['island']}: n={s['count']} best={s['best']:.6f} mean={s['mean']:.6f}")
    conn.close()


def cmd_export_best(args):
    from .db import get_connection
    from .blob import blob_to_factors, factors_to_json
    conn = get_connection(args.db, read_only=True)
    row = conn.execute(
        """SELECT factors_blob, COALESCE(fitness_fp64, fitness_fp32) as fit
           FROM candidates
           WHERE fitness_fp32 IS NOT NULL
           ORDER BY COALESCE(fitness_fp64, fitness_fp32) ASC
           LIMIT 1"""
    ).fetchone()
    if row is None:
        print("No evaluated candidates in database.")
        sys.exit(1)
    alpha, beta, gamma = blob_to_factors(row["factors_blob"])
    data = factors_to_json(alpha, beta, gamma)
    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported best candidate (fitness={row['fit']:.10f}) → {out}")
    conn.close()


def cmd_inject(args):
    from .db import get_connection
    from .schema import create_schema
    from .orchestrator import seed_from_json
    conn = get_connection(args.db)
    create_schema(conn)
    paths = [Path(p) for p in args.input]
    count = seed_from_json(conn, paths)
    print(f"Injected {count} candidates into {args.db}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        prog="db_optimizer",
        description="Two-phase GPU-accelerated rank-19 CP decomposition search",
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the optimizer")
    p_run.add_argument("--generations", type=int, default=10_000)
    p_run.add_argument("--seed-dir", type=str, help="Directory with optimized_*.json files to seed")
    p_run.add_argument("--quiet", action="store_true")

    # status
    sub.add_parser("status", help="Show database status")

    # export-best
    p_export = sub.add_parser("export-best", help="Export best candidate as JSON")
    p_export.add_argument("--output", type=str, default="best_candidate.json")

    # inject
    p_inject = sub.add_parser("inject", help="Inject JSON candidates into DB")
    p_inject.add_argument("--input", nargs="+", required=True, help="JSON file paths")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "export-best":
        cmd_export_best(args)
    elif args.command == "inject":
        cmd_inject(args)


if __name__ == "__main__":
    main()
