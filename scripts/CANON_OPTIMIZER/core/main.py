"""
Entry point for the exhaustive enumeration search.

Usage:
    python -m core.main                       # full search, default ranks [13,19,20,21,22], 24 workers
    python -m core.main --ranks 19            # search only R=19
    python -m core.main --coeff 2             # search over {-2,-1,0,1,2}
    python -m core.main --gpu                 # enable GPU (future)
    python -m core.main --resume              # resume from checkpoint
    python -m core.main --verify FILE         # verify a candidate solution
    python -m core.main --workers 12          # override worker count
    python -m core.main --dry-run             # partition counts only
    python -m core.main --test                # run self-tests
"""

import argparse
import sys
import os

# Add parent directory to path so core package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="R-rank exhaustive enumeration for 3x3 matrix multiplication tensor")
    parser.add_argument('--ranks', type=int, nargs='+', default=None,
                        help='Target ranks to search (default: 13 19 20 21 22)')
    parser.add_argument('--coeff', type=int, default=1,
                        help='Max absolute coefficient (default: 1 for {-1,0,1})')
    parser.add_argument('--workers', type=int, default=24,
                        help='Number of worker processes (default: 24)')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration (future)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--verify', type=str, default=None,
                        help='Verify a candidate solution JSON file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print partition counts and exit')
    parser.add_argument('--test', action='store_true',
                        help='Run self-tests')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable Rich TUI dashboard')
    return parser.parse_args()


def run_self_tests():
    """Run all module self-tests."""
    print("Running self-tests...\n")

    from core.tensor import self_test as test_tensor
    test_tensor()

    from core.gates import self_test as test_gates
    test_gates()

    from core.fiber import self_test as test_fiber
    test_fiber()

    from core.symmetry import self_test as test_symmetry
    test_symmetry()

    print("\nAll self-tests passed!")


def run_verify(filepath: str):
    """Run independent verification on a solution file."""
    from core.verify import main as verify_main
    sys.argv = ['verify.py', filepath]
    verify_main()


def run_dry_run(ranks, coeff_field):
    """Print partition info and exit."""
    from core.fiber import enumerate_fiber_partitions, enumerate_nonzero_vectors

    n_vecs = len(enumerate_nonzero_vectors(coeff_field, 3))
    print(f"Coefficient field: {{{', '.join(str(c) for c in coeff_field)}}}")
    print(f"Non-zero 3-vectors: {n_vecs}")
    print(f"(alpha_row, beta_col) pairs per term: {n_vecs * n_vecs}")
    print()

    for R in sorted(ranks):
        parts = enumerate_fiber_partitions(R, 9)
        target_rk = R - 9
        print(f"  R = {R}: target rank(H) = {target_rk}")
        print(f"    Fiber partitions (canonical): {len(parts)}")
        for p in parts[:5]:
            print(f"      {p}")
        if len(parts) > 5:
            print(f"      ... ({len(parts) - 5} more)")
        print()


def run_search(ranks, config, resume, use_dashboard):
    """Launch the full search."""
    from core.coordinator import Coordinator

    dashboard = None
    if use_dashboard:
        try:
            from core.dashboard import create_dashboard
            dashboard = create_dashboard(config.dashboard_refresh_rate)
        except ImportError:
            print("Rich not available, running without dashboard")

    coord = Coordinator(ranks=ranks, config=config, dashboard=dashboard)
    coord.run(resume=resume)
    coord.print_summary()


def main():
    args = parse_args()

    # Handle special modes
    if args.test:
        run_self_tests()
        return

    if args.verify:
        run_verify(args.verify)
        return

    # Build coefficient field
    max_coeff = args.coeff
    coeff_field = list(range(-max_coeff, max_coeff + 1))

    # Determine ranks
    ranks = args.ranks if args.ranks else [13, 19, 20, 21, 22]

    if args.dry_run:
        run_dry_run(ranks, coeff_field)
        return

    # Build config
    from core.config import SearchConfig
    config = SearchConfig(
        coeff_field=coeff_field,
        n_workers=args.workers,
        use_gpu=args.gpu,
    )

    print(f"ADE3x3 Exhaustive Enumeration Pipeline")
    print(f"  Ranks: {ranks}")
    print(f"  Coefficient field: {{{', '.join(str(c) for c in coeff_field)}}}")
    print(f"  Workers: {config.n_workers}")
    print(f"  GPU: {'enabled' if config.use_gpu else 'disabled'}")
    if args.resume:
        print(f"  Resuming from checkpoint")
    print()

    run_search(ranks, config, args.resume, not args.no_dashboard)


if __name__ == "__main__":
    main()
