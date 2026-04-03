"""
main.py - CLI for the symmetry-reduced CANON OPTIMIZER pivot.

Examples:
  python "CANON OPTIMIZER/main.py" --verify-only
  python "CANON OPTIMIZER/main.py" --runs 20 --maxiter 5000 --out "CANON OPTIMIZER/result.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from symmetry_solver import SearchConfig, run_search, save_result, verify_symmetry_setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symmetry-reduced R=19 solver")
    parser.add_argument("--verify-only", action="store_true", help="Run exact symmetry checks and exit.")
    parser.add_argument("--runs", type=int, default=1, help="Number of random restarts.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--seed-scale", type=float, default=0.3, help="Gaussian initialization scale.")
    parser.add_argument("--maxiter", type=int, default=500, help="Scipy optimizer max iterations per run.")
    parser.add_argument("--log-every", type=int, default=10, help="Record a callback entry every N accepted iterations.")
    parser.add_argument("--method", type=str, default="L-BFGS-B", help="Scipy method. Default: L-BFGS-B.")
    parser.add_argument("--out", type=str, default="CANON OPTIMIZER/result.json", help="JSON output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.verify_only:
        verification = verify_symmetry_setup()
        print(f"group_size={verification['group_size']}")
        print(f"orbit_sizes={verification['orbit_sizes']}")
        print(f"stabilizer_sizes={verification['stabilizer_sizes']}")
        print(f"kept_term_count={verification['kept_term_count']}")
        print(f"deleted_term_count={verification['deleted_term_count']}")
        print(f"tensor_equation_orbit_count_z2={verification['tensor_equation_orbit_count']}")
        print(f"tensor_equation_orbit_count_full={verification['tensor_equation_orbit_count_full_group']}")
        print(f"max_standard_equivariance_error={verification['max_standard_equivariance_error']:.3e}")
        print(f"verification_passed={verification['verification_passed']}")
        return 0 if verification["verification_passed"] else 1

    config = SearchConfig(
        runs=args.runs,
        seed=args.seed,
        seed_scale=args.seed_scale,
        maxiter=args.maxiter,
        log_every=args.log_every,
        method=args.method,
    )
    result = run_search(config)
    save_result(result, args.out)

    diagnostics = result["diagnostics"]
    print(f"saved={args.out}")
    print(f"max_abs_residual={diagnostics['max_abs_residual']:.6e}")
    print(f"loss_fro={diagnostics['loss_fro']:.6e}")
    print(f"rank_H={diagnostics['rank_H']}")
    print(f"conservation={diagnostics['conservation']}")
    print(f"optimizer_success={diagnostics['optimizer_success']}")
    print(f"optimizer_message={diagnostics['optimizer_message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())