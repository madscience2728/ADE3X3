import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor_core import build_T_matmul
from budget import budget_E_norm, budget_frob
from irrep_decomp import build_basis_vectors, build_E_from_coords, E_norm_from_coords
from als_search import als_decompose, anti_isometric_score, run_all_ranks


BASIS_LABELS = (
    ['G_inv_O0', 'G_inv_O1', 'G_inv_O2', 'G_inv_O3']
    + ['O1_anti_0', 'O1_anti_1', 'O1_anti_2']
    + ['O2_anti_0']
    + [f'O3_anti_{i}' for i in range(7)]
)


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def main():
    print("=" * 70)
    print("STEP 5 VERIFICATION — ALS Decomposition on Perturbed Target")
    print("=" * 70)

    T = build_T_matmul()
    eps = 2.0 ** -23

    # ------------------------------------------------------------------ #
    # 1. Baseline ALS on unperturbed T_matmul
    # ------------------------------------------------------------------ #
    print("\n--- Check 1: Baseline ALS on unperturbed T_matmul ---")
    R_all = [13, 15, 17, 18, 19]
    baselines = {}
    hdr = f"  {'R':>4}  {'baseline_residual':>18}  {'budget_frob':>14}  {'ratio (res/bgt)':>16}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for R in R_all:
        result = als_decompose(T, R, n_restarts=10)
        res = result['best_residual']
        bf = budget_frob(R, 23, 1.0)
        ratio = res / bf if bf > 0 else float('inf')
        baselines[R] = res
        print(f"  {R:>4}  {res:>18.6e}  {bf:>14.6e}  {ratio:>16.4f}")

    # ------------------------------------------------------------------ #
    # 2. Anti-isometric score checks
    # ------------------------------------------------------------------ #
    print("\n--- Check 2: Anti-isometric score ---")
    score_T = anti_isometric_score(T)
    print(f"  Anti-isometric score of T_matmul: {score_T:.6e}")
    check("anti_isometric_score(T_matmul) == 0.0 (to 1e-10)",
          score_T < 1e-10, f"{score_T:.2e}")

    # Perturb with a single non-invariant coord
    coords_ni = np.zeros(15)
    coords_ni[4] = 1.0  # first non-invariant coord (O1_anti_0)
    E_ni = build_E_from_coords(coords_ni)
    T_ni = T - eps * E_ni
    score_ni = anti_isometric_score(T_ni)
    print(f"  Anti-isometric score after non-inv perturbation (O1_anti_0, coeff=1): {score_ni:.6e}")
    check("non-invariant perturbation increases anti-isometric score",
          score_ni > score_T, f"{score_ni:.2e} > {score_T:.2e}")

    # ------------------------------------------------------------------ #
    # 3. Search for R in [17, 18, 19]
    # ------------------------------------------------------------------ #
    print("\n--- Check 3: Searching E for R in [17, 18, 19] (2000 samples each) ---")
    R_search = [17, 18, 19]
    results = run_all_ranks(R_search, b=23, M=1.0)

    hdr2 = (f"  {'R':>4}  {'baseline':>12}  {'best_res':>12}  {'budget_frob':>12}  "
            f"{'improvement':>12}  {'success':>8}  {'anti_score':>10}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for res in results:
        R = res['R']
        base = res['baseline_residual']
        best = res['best_residual']
        bf = res['budget_frob']
        impr = base / best if best > 1e-20 else float('inf')
        print(f"  {R:>4}  {base:>12.4e}  {best:>12.4e}  {bf:>12.4e}  "
              f"  {impr:>11.4f}x  {str(res['success']):>8}  {res['anti_score_at_best']:>10.4e}")

    # ------------------------------------------------------------------ #
    # 4. Best coords analysis
    # ------------------------------------------------------------------ #
    print("\n--- Check 4: Best coords analysis ---")
    best_overall = min(results, key=lambda x: x['best_residual'])
    print(f"  Best result: R={best_overall['R']}, residual={best_overall['best_residual']:.4e}")
    coords = np.array(best_overall['best_coords'])
    print(f"  Basis component magnitudes:")
    indexed = sorted(enumerate(coords), key=lambda x: abs(x[1]), reverse=True)
    for idx, val in indexed[:10]:
        print(f"    [{idx:>2}] {BASIS_LABELS[idx]:>15}  coeff = {val:+.6f}")

    # ------------------------------------------------------------------ #
    # 5. Save results
    # ------------------------------------------------------------------ #
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'step5_als.json')
    to_save = []
    for res in results:
        entry = {k: v for k, v in res.items()}
        to_save.append(entry)
    with open(out_path, 'w') as f:
        json.dump(to_save, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
