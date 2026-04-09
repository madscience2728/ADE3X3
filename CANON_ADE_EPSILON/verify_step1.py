import sys
import numpy as np

sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')

from tensor_core import build_T_matmul, G_ORBITS, build_G_invariant_E, frobenius

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition

def main():
    print("=" * 60)
    print("STEP 1 VERIFICATION")
    print("=" * 60)

    T = build_T_matmul()

    # Check 1: exactly 27 nonzero entries
    nnz = int(np.count_nonzero(T))
    check("T_matmul has exactly 27 nonzero entries", nnz == 27, f"found {nnz}")

    # Check 2: orbit sizes sum to 27
    sizes = {k: len(v) for k, v in G_ORBITS.items()}
    total = sum(sizes.values())
    check(
        "Orbit sizes sum to 27",
        total == 27 and sizes['O0'] == 1 and sizes['O1'] == 6
            and sizes['O2'] == 12 and sizes['O3'] == 8,
        f"O0={sizes['O0']}, O1={sizes['O1']}, O2={sizes['O2']}, O3={sizes['O3']}, total={total}"
    )

    # Check 3: E(1,1,1,1) == T_matmul elementwise
    E = build_G_invariant_E(1.0, 1.0, 1.0, 1.0)
    diff = np.max(np.abs(E - T))
    check(
        "build_G_invariant_E(1,1,1,1) equals T_matmul elementwise",
        diff < 1e-14,
        f"max |E-T| = {diff:.2e}"
    )

    # Check 4: frobenius(T, T) == 0
    fval = frobenius(T, T)
    check(
        "frobenius(T_matmul, T_matmul) == 0.0",
        fval == 0.0,
        f"value = {fval}"
    )

    print("=" * 60)

if __name__ == "__main__":
    main()
