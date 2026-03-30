from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outputs.ade3x3_attack.attack_common import tensor_residual_stats, terms_from_decomposition  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute residuals for saved decomposition JSON files and rank them by actual tensor residual.",
    )
    parser.add_argument(
        "glob",
        nargs="?",
        default="experiments/wing it/best_decompositions/*.json",
        help="Glob pattern, relative to repo root, selecting decomposition JSON files.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Only include payloads whose metadata.rank matches this value.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of rows to print after sorting by actual max-abs residual.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a plain-text table.",
    )
    parser.add_argument(
        "--mismatch-threshold",
        type=float,
        default=1e-8,
        help="Absolute tolerance above which reported and recomputed residuals are flagged as mismatched.",
    )
    return parser


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_rows(glob_pattern: str, rank_filter: int | None, mismatch_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(REPO_ROOT.glob(glob_pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) or {}
        rank_value = metadata.get("rank", payload.get("term_count"))
        if rank_filter is not None and rank_value != rank_filter:
            continue

        terms = terms_from_decomposition(payload)
        actual_max_abs, actual_loss = tensor_residual_stats(terms)
        reported_max_abs = float(payload.get("max_abs_residual", metadata.get("final_max_abs", float("nan"))))
        reported_loss = float(payload.get("loss_value", metadata.get("final_loss", float("nan"))))
        delta = abs(actual_max_abs - reported_max_abs)
        rows.append(
            {
                "path": relative_path(path),
                "search_id": metadata.get("search_id", payload.get("decomposition_id", path.stem)),
                "rank": rank_value,
                "strategy": metadata.get("strategy", payload.get("source_detail", "")),
                "init_detail": metadata.get("init_detail", ""),
                "reported_max_abs": reported_max_abs,
                "actual_max_abs": actual_max_abs,
                "reported_loss": reported_loss,
                "actual_loss": actual_loss,
                "max_abs_delta": delta,
                "mismatch": delta > mismatch_threshold,
            }
        )
    rows.sort(key=lambda row: (float(row["actual_max_abs"]), float(row["actual_loss"]), str(row["path"])))
    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'actual_max_abs':>16}  {'reported_max_abs':>16}  {'delta':>12}  {'rank':>4}  "
        f"{'strategy':<18}  {'search_id':<20}  path"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{float(row['actual_max_abs']):16.9g}  "
            f"{float(row['reported_max_abs']):16.9g}  "
            f"{float(row['max_abs_delta']):12.3g}  "
            f"{int(row['rank']):4d}  "
            f"{str(row['strategy']):<18.18}  "
            f"{str(row['search_id']):<20.20}  "
            f"{row['path']}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = load_rows(args.glob, args.rank, args.mismatch_threshold)
    if not rows:
        raise SystemExit("No matching decomposition JSON files found.")

    top_rows = rows[: max(1, args.top)]
    mismatch_count = sum(1 for row in rows if bool(row["mismatch"]))

    if args.json:
        print(
            json.dumps(
                {
                    "glob": args.glob,
                    "rank_filter": args.rank,
                    "file_count": len(rows),
                    "mismatch_count": mismatch_count,
                    "top_rows": top_rows,
                },
                indent=2,
            )
        )
        return

    print(f"Verified {len(rows)} decomposition files from {args.glob}")
    print(f"Residual metadata mismatches above {args.mismatch_threshold:g}: {mismatch_count}")
    print_table(top_rows)


if __name__ == "__main__":
    main()