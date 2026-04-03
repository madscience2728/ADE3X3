"""
main.py — Entry point for the canonized R=19 optimizer.

Usage:
    python -m rank19_optimizer.main [options]
    python rank19_optimizer/main.py [options]

Init strategy:
    1. Random α, β in N(0, 0.3).
    2. Iteratively project each β_k to Gate-1 feasibility:
           β_new = β_old - pinv(Perp_A) @ (Perp_A @ β_old)
       This preserves ‖β‖ ≈ 0.3; after ~5 rounds rank(H) == 10.
    3. Initial Γ via exact solve.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Support both  `python main.py`  and  `python -m rank19_optimizer.main`
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tensor import build_fiber_coordinates
from gates  import (compute_V10, compute_jacobians_beta,
                    compute_jacobians_alpha, project_to_gate1, solve_gamma)
from optimizer import optimize


# ── Init ──────────────────────────────────────────────────────────────────────

def _init_gate1_feasible(R: int, seed: int, init_rounds: int = 10) -> tuple:
    """Random init projected to Gate-1 feasibility."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, 0.3, (R, 3, 3))
    beta  = rng.normal(0, 0.3, (R, 3, 3))

    for _ in range(init_rounds):
        coords = build_fiber_coordinates(alpha, beta)
        V10    = compute_V10(coords['H'])

        for k in range(R):
            A_k, _, _ = compute_jacobians_beta(alpha[k])
            b_new     = project_to_gate1(beta[k].ravel(), A_k, V10)
            beta[k]   = b_new.reshape(3, 3)

    # Verify scale didn't collapse
    max_b = float(np.max(np.abs(beta)))
    coords = build_fiber_coordinates(alpha, beta)
    rk_H   = int(np.linalg.matrix_rank(coords['H'], tol=1e-10))
    print(f'  init: max|β|={max_b:.4f}  rank(H)={rk_H}'
          + ('  ✓' if rk_H == 10 and max_b > 0.05 else '  ⚠ SCALE COLLAPSED' if max_b < 0.05 else ''))

    gamma0, _ = solve_gamma(coords['Sigma'], coords['Nuisance'])
    gamma     = gamma0.T.reshape(R, 3, 3)
    return alpha, beta, gamma


def _load_json_init(path: str) -> tuple:
    with open(path) as f:
        d = json.load(f)
    alpha = np.array(d['alpha'])
    beta  = np.array(d['beta'])
    gamma = np.array(d['gamma'])
    return alpha, beta, gamma


# ── Serialisation ─────────────────────────────────────────────────────────────

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


def _save(result: dict, path: str):
    out = {
        'alpha':       result['alpha'].tolist(),
        'beta':        result['beta'].tolist(),
        'gamma':       result['gamma'].tolist(),
        'diagnostics': _to_serializable(result['diagnostics']),
        'success':     result['success'],
        'iterations':  result['iterations'],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved → {path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Canonized R=19 optimizer (Gate 1-3)')
    p.add_argument('--runs',        type=int,   default=1,           help='Independent restarts')
    p.add_argument('--max-iter',    type=int,   default=5000,        help='Max iterations per run')
    p.add_argument('--seed',        type=int,   default=42,          help='Base RNG seed')
    p.add_argument('--log-every',   type=int,   default=1,           help='Log frequency')
    p.add_argument('--gate2',       action='store_true',             help='Enable Gate 2 constraints in LP')
    p.add_argument('--out',         type=str,   default='rank19_optimizer/result.json',
                                                                      help='Output path')
    p.add_argument('--load',        type=str,   default=None,        help='Warm-start from JSON')
    p.add_argument('--init-rounds', type=int,   default=10,          help='Gate-1 init rounds')
    return p.parse_args()


def main():
    args = parse_args()
    R    = 19

    best_result  = None
    best_fitness = float('inf')

    for run in range(args.runs):
        seed = args.seed + run
        print(f'\n{"─"*60}')
        print(f'Run {run+1}/{args.runs}   seed={seed}')
        print(f'{"─"*60}')

        if args.load is not None and run == 0:
            print(f'Loading warm start from {args.load}')
            alpha0, beta0, gamma0 = _load_json_init(args.load)
        else:
            alpha0, beta0, gamma0 = _init_gate1_feasible(R, seed, args.init_rounds)

        t0 = time.time()
        result = optimize(
            alpha0, beta0, gamma0,
            max_iter    = args.max_iter,
            log_every   = args.log_every,
            verbose     = True,
        )
        elapsed = time.time() - t0
        fit = result['diagnostics']['fitness']
        print(f'\nRun {run+1} done in {elapsed:.1f}s  '
              f'fitness={fit:.5e}  success={result["success"]}')

        if fit < best_fitness:
            best_fitness = fit
            best_result  = result

        if result['success']:
            break

    if best_result is not None:
        out_path = args.out
        if args.runs > 1:
            stem = out_path.replace('.json', '')
            out_path = f'{stem}_best.json'
        _save(best_result, out_path)


if __name__ == '__main__':
    main()
