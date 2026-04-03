"""
main.py — Entry point for the Canon Optimizer (Grassmannian solver).

Usage:
    python "CANON OPTIMIZER/main.py" [options]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from grassmannian import GrassmannianSolver
from tensor import build_fiber_coordinates


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _save(result, path):
    out = {
        'alpha': result['alpha'].tolist(),
        'beta':  result['beta'].tolist(),
        'gamma': result['gamma'].tolist(),
        'diagnostics': _to_serializable(result.get('diagnostics', {})),
        'success': result.get('success', False),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def _load_warm_start(path):
    with open(path) as f:
        d = json.load(f)
    return {
        'alpha': np.array(d['alpha']),
        'beta':  np.array(d['beta']),
        'gamma': np.array(d.get('gamma', np.zeros_like(d['alpha']))),
    }


def main():
    parser = argparse.ArgumentParser(description='Canon Optimizer — Grassmannian solver')
    parser.add_argument('--R', type=int, default=19, help='Target rank')
    parser.add_argument('--max-outer', type=int, default=500)
    parser.add_argument('--max-inner', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', type=str, default=None, help='Warm-start from JSON')
    parser.add_argument('--out', type=str, default='CANON OPTIMIZER/result.json')
    parser.add_argument('--log-every', type=int, default=10)
    args = parser.parse_args()

    warm = _load_warm_start(args.load) if args.load else None

    global_best = None

    for run in range(args.runs):
        seed = args.seed + run
        print(f'\n{"="*60}')
        print(f'Run {run+1}/{args.runs}  seed={seed}')
        print(f'{"="*60}')

        solver = GrassmannianSolver(R=args.R)
        result = solver.search(
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            lr=args.lr,
            seed=seed,
            warm_start=warm,
        )

        fit = result.get('fitness', float('inf'))
        print(f'\nRun {run+1} best fitness: {fit:.6f}')

        if global_best is None or fit < global_best.get('fitness', float('inf')):
            global_best = result

    print(f'\n{"="*60}')
    print(f'Global best fitness: {global_best["fitness"]:.6f}')
    d = global_best.get('diagnostics', {})
    print(f'  rank(H)={d.get("rank_H","?")}  '
          f'rank(N)={d.get("rank_nuisance","?")}  '
          f'conservation={d.get("conservation","?")}')

    _save(global_best, args.out)
    print(f'Saved to {args.out}')

    # Also save best separately
    best_path = args.out.replace('.json', '_best.json')
    _save(global_best, best_path)


if __name__ == '__main__':
    main()
