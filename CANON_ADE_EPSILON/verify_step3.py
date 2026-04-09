import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distance_search import run_all, search_rank


def main():
    print("=" * 70)
    print("STEP 3 VERIFICATION  Distance Search C1 (corrected formulation)")
    print("=" * 70)

    # 1. Run search for all target ranks
    R_list = [13, 15, 17, 18, 19]
    print(f"\nRunning constrained grid+refine search for R in {R_list}  (b=23, M=1.0)...")
    results = run_all(R_list, b=23, M=1.0)

    # 2. Per-R detailed output
    print("\n--- Per-R Results ---")
    for res in results:
        R = res["R"]
        print(f"\n  R={R}")
        print(f"    best_score (sigma_{{R+1}})  : {res['best_score']:.6e}")
        print(f"    E_norm  (||E||_F)          : {res['E_norm']:.6f}")
        print(f"    budget_E_norm              : {res['budget_E_norm']:.4f}")
        print(f"    absolute_perturbation      : {res['absolute_perturbation']:.6e}")
        print(f"    loop_closed                : {res['loop_closed']}")
        w = res["best_w"]
        print(f"    best_w                     : ({w[0]:.5f}, {w[1]:.5f}, {w[2]:.5f}, {w[3]:.5f})")

    # 4. Sanity check: score should decrease as R increases
    print("\n--- Sanity Check: score(R=19) <= score(R=13) ---")
    score_by_R = {res["R"]: res["best_score"] for res in results}
    s13 = score_by_R[13]
    s19 = score_by_R[19]
    print(f"  score(R=13) = {s13:.6e}")
    print(f"  score(R=19) = {s19:.6e}")
    if s19 <= s13:
        print("[PASS] score decreases with R (higher rank easier to approach)")
    else:
        print("[WARN] score does NOT decrease with R  G-invariant family cannot")
        print("       easily reach rank-R variety for any R in this range.")
        print("       This is itself a finding: the orbit structure resists perturbation.")

    # 3. Ranked table by best_score ascending
    ranked = sorted(results, key=lambda x: x["best_score"])
    print("\n--- Ranked by best_score ascending (lowest = closest to rank-R variety) ---")
    hdr = f"  {'R':>4}  {'best_score':>14}  {'E_norm':>10}  {'budget_E_norm':>14}  {'abs_perturb':>14}  loop_closed"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for res in ranked:
        print(f"  {res['R']:>4}  {res['best_score']:>14.6e}  "
              f"{res['E_norm']:>10.4f}  {res['budget_E_norm']:>14.4f}  "
              f"{res['absolute_perturbation']:>14.6e}  {res['loop_closed']}")

    # 5. Save JSON
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "step3_distance.json")
    to_save = [
        {
            "R": res["R"],
            "best_score": res["best_score"],
            "best_w": res["best_w"],
            "E_norm": res["E_norm"],
            "budget_E_norm": res["budget_E_norm"],
            "loop_closed": res["loop_closed"],
        }
        for res in results
    ]
    with open(out_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
