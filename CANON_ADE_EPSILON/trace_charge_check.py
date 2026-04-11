"""
Task 2 — Trace charge verification.

For every saved result, compute the Noether trace charge:
  Q = Σ_k tr(α̂_k) · tr(β̂_k) · tr(γ̂_k)
where α̂_k is α_k reshaped as 3×3.

Prediction: Q ≈ 3 universally (= n for n×n matmul).
"""
import numpy as np
import json
import os
import glob

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
SEARCH_DIRS = [
    os.path.join(RESULTS_ROOT, 'gated_search'),
    os.path.join(RESULTS_ROOT, 'cliff_search'),
    RESULTS_ROOT,  # top-level files
]


def compute_trace_charge(alpha, beta, gamma):
    """Q = Σ_k tr(α_k.reshape(3,3)) · tr(β_k.reshape(3,3)) · tr(γ_k.reshape(3,3))"""
    R = len(alpha)
    Q = 0.0
    for k in range(R):
        ta = np.trace(np.array(alpha[k]).reshape(3, 3))
        tb = np.trace(np.array(beta[k]).reshape(3, 3))
        tg = np.trace(np.array(gamma[k]).reshape(3, 3))
        Q += ta * tb * tg
    return Q


def main():
    print("="*80)
    print("TRACE CHARGE VERIFICATION — Q = Σ tr(α̂_k)·tr(β̂_k)·tr(γ̂_k)")
    print("="*80)
    print(f"\n{'File':<55} {'R':>3} {'εF':>12} {'Q':>12} {'|Q-3|':>12}")
    print("-"*96)

    files_checked = 0
    all_results = []

    for search_dir in SEARCH_DIRS:
        if not os.path.isdir(search_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(search_dir, '*.json'))):
            try:
                with open(fpath) as fp:
                    data = json.load(fp)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if 'alpha' not in data or 'beta' not in data or 'gamma' not in data:
                continue

            R = data.get('rank', '?')
            eps = data.get('frobenius', float('nan'))
            Q = compute_trace_charge(data['alpha'], data['beta'], data['gamma'])
            delta = abs(Q - 3.0)

            rel_path = os.path.relpath(fpath, os.path.dirname(__file__))
            print(f"{rel_path:<55} {R:>3} {eps:12.6f} {Q:12.6f} {delta:12.6e}")
            files_checked += 1
            all_results.append({
                'file': rel_path, 'rank': R, 'frobenius': eps,
                'trace_charge_Q': Q, 'abs_Q_minus_3': delta,
            })

    print(f"\n  Files checked: {files_checked}")
    if all_results:
        deltas = [r['abs_Q_minus_3'] for r in all_results]
        print(f"  Max |Q-3|: {max(deltas):.6e}")
        print(f"  Mean |Q-3|: {np.mean(deltas):.6e}")
        print(f"  Min |Q-3|: {min(deltas):.6e}")

        q_vals = [r['trace_charge_Q'] for r in all_results]
        print(f"\n  Q range: [{min(q_vals):.6f}, {max(q_vals):.6f}]")

        if max(deltas) < 0.01:
            print("\n  *** CONFIRMED: Q ≈ 3 universally — trace charge is conserved ***")
        elif max(deltas) < 0.1:
            print("\n  *** LIKELY: Q ≈ 3 with small deviations ***")
        else:
            print("\n  *** MIXED: Some files deviate significantly from Q=3 ***")
            for r in all_results:
                if r['abs_Q_minus_3'] > 0.1:
                    print(f"      Outlier: {r['file']}  R={r['rank']}  Q={r['trace_charge_Q']:.6f}")

    # Save
    outpath = os.path.join(RESULTS_ROOT, 'trace_charge_report.json')
    with open(outpath, 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n  Report saved to {outpath}")


if __name__ == '__main__':
    main()
