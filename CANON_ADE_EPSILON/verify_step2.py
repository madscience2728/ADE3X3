import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from budget import (
    PRECISION_PROFILES,
    delta_per_element,
    budget_frob,
    budget_E_norm,
    is_loop_closed,
    summary_table,
)

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition

def main():
    print("=" * 70)
    print("STEP 2 VERIFICATION — Bit Budget Formula")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Check 1: exact values for b=23, M=1, R=19
    # ------------------------------------------------------------------ #
    b, M, R = 23, 1.0, 19
    expected_delta = 19 * 2**-23
    expected_frob  = 3 * 19 * 2**-23

    got_delta = delta_per_element(R, b, M)
    got_frob  = budget_frob(R, b, M)

    print("\n--- Check 1: Exact values (R=19, b=23, M=1) ---")
    print(f"  expected delta_per_element : {expected_delta:.6e}")
    print(f"  got      delta_per_element : {got_delta:.6e}")
    print(f"  expected budget_frob       : {expected_frob:.6e}")
    print(f"  got      budget_frob       : {got_frob:.6e}")

    check("delta_per_element(19,23,1) == 19*2^-23",
          got_delta == expected_delta,
          f"{got_delta:.6e}")
    check("budget_frob(19,23,1) == 3*19*2^-23",
          got_frob == expected_frob,
          f"{got_frob:.6e}")

    # ------------------------------------------------------------------ #
    # Check 2: summary table
    # ------------------------------------------------------------------ #
    R_list = [13, 15, 17, 18, 19]
    rows = summary_table(R_list, b=23, M=1.0)

    print("\n--- Check 2: Summary Table (b=23 / float32, M=1.0) ---")
    hdr = f"{'R':>4}  {'b':>4}  {'M':>4}  {'eps':>12}  {'delta/elem':>14}  {'budget_frob':>14}  {'budget_E_norm':>14}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(f"{row['R']:>4}  {row['b']:>4}  {row['M']:>4.1f}  "
              f"{row['eps']:>12.6e}  {row['delta_per_element']:>14.6e}  "
              f"{row['budget_frob']:>14.6e}  {row['budget_E_norm']:>14.6f}")

    # ------------------------------------------------------------------ #
    # Check 3: budget_E_norm is independent of b
    # ------------------------------------------------------------------ #
    print("\n--- Check 3: budget_E_norm independence from b (R=19, M=1) ---")
    norms = {}
    for label, bval in PRECISION_PROFILES.items():
        n = budget_E_norm(19, bval, 1.0)
        norms[label] = n
        print(f"  {label:10s}  b={bval:2d}  budget_E_norm = {n:.6f}")

    all_same = len(set(norms.values())) == 1
    check("budget_E_norm identical across all precision profiles",
          all_same,
          f"unique values: {set(norms.values())}")

    # ------------------------------------------------------------------ #
    # Check 4: feedback loop status
    # ------------------------------------------------------------------ #
    print("\n--- Check 4: Feedback loop (E_norm=3.0 placeholder, b=23, M=1) ---")
    E_norm_placeholder = 3.0
    for Rv in [13, 15, 17, 18, 19]:
        closed = is_loop_closed(Rv, 23, 1.0, E_norm_placeholder)
        bEN = budget_E_norm(Rv, 23, 1.0)
        status = "CLOSED" if closed else "OPEN  "
        print(f"  R={Rv:2d}  budget_E_norm={bEN:6.2f}  E_norm={E_norm_placeholder:.2f}  loop={status}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
