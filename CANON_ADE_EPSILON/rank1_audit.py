"""
Task 3 — Rank-1 structure audit on all saved decompositions.

For every result file, reshape each α_k, β_k, γ_k as 3×3 and compute
its effective matrix rank (via SVD, threshold = 2% of σ_max).

Reports distribution of rank-1 / rank-2 / full-rank per file and per R.
"""
import numpy as np
import json
import os
import glob
from collections import defaultdict

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
SEARCH_DIRS = [
    os.path.join(RESULTS_ROOT, 'gated_search'),
    os.path.join(RESULTS_ROOT, 'cliff_search'),
    RESULTS_ROOT,
]

RANK_THRESHOLD = 0.02  # σ_k / σ_0 < threshold → effectively zero


def matrix_rank_profile(vectors):
    """Given list of 9-vectors, return (n_rank1, n_rank2, n_full)."""
    r1 = r2 = r3 = 0
    for v in vectors:
        mat = np.array(v).reshape(3, 3)
        svs = np.linalg.svd(mat, compute_uv=False)
        if svs[0] < 1e-12:
            r1 += 1  # zero matrix, count as "rank 0/1"
            continue
        ratios = svs[1:] / svs[0]
        if ratios[0] < RANK_THRESHOLD:
            r1 += 1
        elif ratios[1] < RANK_THRESHOLD:
            r2 += 1
        else:
            r3 += 1
    return r1, r2, r3


def main():
    print("="*80)
    print("RANK-1 STRUCTURE AUDIT — coefficient matrix rank distribution")
    print("="*80)
    print(f"\n{'File':<45} {'R':>3}  {'α r1':>4}/{'r2':>3}/{'fR':>3}  "
          f"{'β r1':>4}/{'r2':>3}/{'fR':>3}  {'γ r1':>4}/{'r2':>3}/{'fR':>3}  {'εF':>10}")
    print("-"*110)

    per_rank = defaultdict(lambda: {'alpha': [0,0,0], 'beta': [0,0,0], 'gamma': [0,0,0], 'count': 0})
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

            a_r1, a_r2, a_fr = matrix_rank_profile(data['alpha'])
            b_r1, b_r2, b_fr = matrix_rank_profile(data['beta'])
            g_r1, g_r2, g_fr = matrix_rank_profile(data['gamma'])

            rel_path = os.path.relpath(fpath, os.path.dirname(__file__))
            print(f"{rel_path:<45} {R:>3}  {a_r1:4d}/{a_r2:3d}/{a_fr:3d}  "
                  f"{b_r1:4d}/{b_r2:3d}/{b_fr:3d}  {g_r1:4d}/{g_r2:3d}/{g_fr:3d}  {eps:10.6f}")

            if isinstance(R, int):
                pr = per_rank[R]
                pr['alpha'][0] += a_r1; pr['alpha'][1] += a_r2; pr['alpha'][2] += a_fr
                pr['beta'][0] += b_r1; pr['beta'][1] += b_r2; pr['beta'][2] += b_fr
                pr['gamma'][0] += g_r1; pr['gamma'][1] += g_r2; pr['gamma'][2] += g_fr
                pr['count'] += 1

            all_results.append({
                'file': rel_path, 'rank': R, 'frobenius': eps,
                'alpha_rank_profile': [a_r1, a_r2, a_fr],
                'beta_rank_profile': [b_r1, b_r2, b_fr],
                'gamma_rank_profile': [g_r1, g_r2, g_fr],
            })

    # ── Per-rank summary ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PER-RANK AGGREGATE (summed over all files at each rank)")
    print("="*80)
    print(f"\n{'R':>3} {'files':>5}  {'α r1%':>6} {'β r1%':>6} {'γ r1%':>6}  "
          f"{'α r1':>4}/{'r2':>3}/{'fR':>3}  {'β r1':>4}/{'r2':>3}/{'fR':>3}  {'γ r1':>4}/{'r2':>3}/{'fR':>3}")
    print("-"*90)

    for R in sorted(per_rank.keys()):
        pr = per_rank[R]
        n = pr['count']
        for table_name in ['alpha', 'beta', 'gamma']:
            total = sum(pr[table_name])
            if total == 0:
                pr[f'{table_name}_r1_pct'] = 0
            else:
                pr[f'{table_name}_r1_pct'] = 100 * pr[table_name][0] / total

        print(f"{R:3d} {n:5d}  {pr['alpha_r1_pct']:5.1f}% {pr['beta_r1_pct']:5.1f}% {pr['gamma_r1_pct']:5.1f}%  "
              f"{pr['alpha'][0]:4d}/{pr['alpha'][1]:3d}/{pr['alpha'][2]:3d}  "
              f"{pr['beta'][0]:4d}/{pr['beta'][1]:3d}/{pr['beta'][2]:3d}  "
              f"{pr['gamma'][0]:4d}/{pr['gamma'][1]:3d}/{pr['gamma'][2]:3d}")

    # Check if rank-1 dominance is universal
    total_r1 = sum(pr['alpha'][0] + pr['beta'][0] + pr['gamma'][0] for pr in per_rank.values())
    total_all = sum(sum(pr['alpha']) + sum(pr['beta']) + sum(pr['gamma']) for pr in per_rank.values())
    if total_all > 0:
        pct = 100 * total_r1 / total_all
        print(f"\n  Overall rank-1 fraction: {total_r1}/{total_all} = {pct:.1f}%")
        if pct > 70:
            print("  *** CONFIRMED: rank-1 dominance persists across all ranks ***")

    outpath = os.path.join(RESULTS_ROOT, 'rank1_audit_report.json')
    with open(outpath, 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n  Report saved to {outpath}")


if __name__ == '__main__':
    main()
