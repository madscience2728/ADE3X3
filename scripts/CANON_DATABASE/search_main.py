from __future__ import annotations

import argparse
import time
from pathlib import Path

from search_config import SearchConfig
from search_dashboard import create_search_dashboard
from search_engine import AssembleEngine, HarvestEngine, SearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 DB-driven search over TermDB")
    parser.add_argument("--rank", type=int, action="append", dest="ranks", help="Target rank to search")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=24, help="Worker count placeholder for future parallelism")
    parser.add_argument("--db", type=str, default="./CANON_DATABASE/data", help="Path to on-disk TermDB files")
    parser.add_argument("--test", action="store_true", help="Run lightweight Phase 2 smoke tests")
    parser.add_argument("--profile", action="store_true", help="Profile only the first configured bases")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the Rich dashboard")
    parser.add_argument("--stage1", action="store_true", help="Stage 1 only: harvest bases + query hits → save .npz packets")
    parser.add_argument("--stage2", action="store_true", help="Stage 2 only: load packets → assembly + gate check (parallel)")
    parser.add_argument("--seeds", type=int, default=None, help="Override basis_seed_count")
    parser.add_argument("--max-bases", type=int, default=None, help="Override max_bases_per_rank")
    return parser.parse_args()


def run_smoke_test(workspace_root: Path, use_dashboard: bool = True) -> None:
    config = SearchConfig(target_ranks=[13], max_bases_per_rank=1, basis_seed_count=8, db_path="./CANON_DATABASE/data")
    config.resolve_paths(workspace_root)
    dashboard = create_search_dashboard(config.dashboard_refresh_rate) if use_dashboard else None
    engine = SearchEngine(config=config, dashboard=dashboard)
    if dashboard:
        engine.start_time = time.perf_counter()
        dashboard.start(engine.state)
    engine.load_db()
    if dashboard:
        dashboard.stop()
    if not hasattr(engine.db, "unique_H") or engine.db.unique_H.shape[0] == 0:
        raise RuntimeError("TermDB dedup index is empty")
    print("Phase 2 smoke test: DB loaded and dedup index built")


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[1]

    if args.test:
        run_smoke_test(workspace_root, use_dashboard=not args.no_dashboard)
        return

    ranks = args.ranks if args.ranks else [13, 19, 20, 21, 22]
    config = SearchConfig(target_ranks=ranks, n_workers=args.workers, db_path=args.db)
    config.resolve_paths(workspace_root)
    if args.profile:
        config.max_bases_per_rank = min(config.max_bases_per_rank, 8)
    if args.seeds is not None:
        config.basis_seed_count = args.seeds
    if args.max_bases is not None:
        config.max_bases_per_rank = args.max_bases

    dashboard = None if args.no_dashboard else create_search_dashboard(config.dashboard_refresh_rate)

    if args.stage1:
        engine = HarvestEngine(config=config, dashboard=dashboard)
        result = engine.run(resume=args.resume)
        print("\nStage 1 (harvest) complete — packets written:")
        for rank, count in result.items():
            print(f"  R={rank}: {count} packet(s)")
        return

    if args.stage2:
        engine = AssembleEngine(config=config, dashboard=dashboard)
        solutions = engine.run()
        print("\nStage 2 (assemble) complete:")
        for rank in ranks:
            print(f"  R={rank}: {len(solutions.get(rank, []))} solution(s)")
        return

    # Default: legacy combined mode
    engine = SearchEngine(config=config, dashboard=dashboard)
    solutions = engine.run(resume=args.resume)

    print("Phase 2 search complete")
    for rank in ranks:
        print(f"  R={rank}: {len(solutions.get(rank, []))} solution(s)")


if __name__ == "__main__":
    main()