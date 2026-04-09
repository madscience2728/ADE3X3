import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor_core import build_T_matmul, build_G_invariant_E, G_ORBITS
from budget import budget_E_norm
from irrep_decomp import (
    build_basis_vectors, build_E_from_coords, E_norm_from_coords,
    project_to_G_invariant, project_to_non_invariant,
    spectral_analysis, O1_PAIRS, O2_PAIRS, O3_TERMS,
)

ORBIT_SIZES = {'O0': 1, 'O1': 6, 'O2': 12, 'O3': 8}


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def main():
    print("=" * 70)
    print("STEP 4 VERIFICATION — Irrep Decomposition (D3)")
    print("=" * 70)

    basis = build_basis_vectors()
    all_vecs = basis['all']  # 15 vectors

    # ------------------------------------------------------------------ #
    # Check 1a: G-invariant vectors have correct nonzero count
    # ------------------------------------------------------------------ #
    print("\n--- Check 1a: G_inv basis vectors have correct nonzero count ---")
    labels_Ginv = ['O0', 'O1', 'O2', 'O3']
    for i, (key, v) in enumerate(zip(labels_Ginv, basis['G_inv'])):
        nnz = int(np.count_nonzero(v))
        expected = ORBIT_SIZES[key]
        check(f"G_inv[{key}] has {expected} nonzero entries", nnz == expected,
              f"got {nnz}")

    # ------------------------------------------------------------------ #
    # Check 1b: Anti vectors sum to 0 over their orbit
    # ------------------------------------------------------------------ #
    print("\n--- Check 1b: Anti vectors sum to 0 over their orbit ---")
    for i, v in enumerate(basis['O1_anti']):
        orbit_sum = float(sum(v[3*r+s, 3*s+u, 3*r+u] for (r, s, u) in G_ORBITS['O1']))
        check(f"O1_anti[{i}] orbit sum == 0", abs(orbit_sum) < 1e-12, f"{orbit_sum:.2e}")

    for i, v in enumerate(basis['O2_anti']):
        orbit_sum = float(sum(v[3*r+s, 3*s+u, 3*r+u] for (r, s, u) in G_ORBITS['O2']))
        check(f"O2_anti[{i}] orbit sum == 0", abs(orbit_sum) < 1e-12, f"{orbit_sum:.2e}")

    for i, v in enumerate(basis['O3_anti']):
        orbit_sum = float(sum(v[3*r+s, 3*s+u, 3*r+u] for (r, s, u) in O3_TERMS))
        check(f"O3_anti[{i}] orbit sum == 0", abs(orbit_sum) < 1e-12, f"{orbit_sum:.2e}")

    # ------------------------------------------------------------------ #
    # Check 1c: All 15 basis vectors mutually orthogonal
    # ------------------------------------------------------------------ #
    print("\n--- Check 1c: All 15 basis vectors mutually orthogonal ---")
    max_err = 0.0
    failures = []
    for i in range(15):
        for j in range(i + 1, 15):
            dot = float(np.dot(all_vecs[i].ravel(), all_vecs[j].ravel()))
            if abs(dot) > max_err:
                max_err = abs(dot)
            if abs(dot) > 1e-10:
                failures.append((i, j, dot))
    check("All 15 pairs orthogonal (tol 1e-10)", len(failures) == 0,
          f"max |dot| = {max_err:.2e}, violations = {len(failures)}")
    if failures:
        for (i, j, dot) in failures[:5]:
            print(f"  basis[{i}] · basis[{j}] = {dot:.4e}")

    # ------------------------------------------------------------------ #
    # Check 2: Cross-check G_inv[O0] with tensor_core
    # ------------------------------------------------------------------ #
    print("\n--- Check 2: G_inv basis cross-check with tensor_core ---")
    from_core = build_G_invariant_E(1, 0, 0, 0)  # O0 indicator
    from_basis = basis['G_inv'][0]
    diff = float(np.max(np.abs(from_core - from_basis)))
    check("build_G_invariant_E(1,0,0,0) == G_inv['O0'] basis vector",
          diff < 1e-14, f"max diff = {diff:.2e}")

    # ------------------------------------------------------------------ #
    # Check 3: Spectral analysis of T_matmul
    # ------------------------------------------------------------------ #
    print("\n--- Check 3: Spectral analysis of T_matmul ---")
    T = build_T_matmul()
    spec = spectral_analysis(T)
    sqrt3 = math.sqrt(3.0)

    for mode_key in ('mode0_svs', 'mode1_svs', 'mode2_svs'):
        svs = spec[mode_key]
        sv_str = "  ".join(f"{s:.6f}" for s in svs)
        print(f"  {mode_key}: [{sv_str}]")

    sv0 = spec['mode0_svs']
    print(f"\n  √3 = {sqrt3:.10f}")
    # Find which sv indices are closest to sqrt3
    for idx, sv in enumerate(sv0):
        if abs(sv - sqrt3) < 0.01:
            print(f"  mode0: sigma[{idx}] = {sv:.10f}  (distance from √3: {sv - sqrt3:+.2e})")
    print(f"  sqrt3_gap (sigma[3] - √3) = {spec['sqrt3_gap']:+.6e}")

    # ------------------------------------------------------------------ #
    # Check 4: G-invariant E cannot push sigma[3] below √3
    # ------------------------------------------------------------------ #
    print("\n--- Check 4: G-invariant E cannot push below √3 ---")
    budget_en = budget_E_norm(19, 23, 1.0)  # = 57
    eps = 2.0 ** -23
    rng = np.random.default_rng(42)
    min_gap_ginv = math.inf
    for _ in range(100):
        # Sample random G-invariant coords (first 4 only)
        coords = np.zeros(15)
        coords[:4] = rng.uniform(-10, 10, 4)
        E = build_E_from_coords(coords)
        # Scale to fit in budget
        en = float(np.linalg.norm(E))
        if en > budget_en:
            E = E * (budget_en / en)
        T_p = T - eps * E
        gap = spectral_analysis(T_p)['sqrt3_gap']
        if gap < min_gap_ginv:
            min_gap_ginv = gap
    print(f"  G-invariant E: min sqrt3_gap across 100 samples = {min_gap_ginv:+.6e}")
    check("G-invariant E: min_gap >= -1e-10 (cannot push below √3)",
          min_gap_ginv >= -1e-10, f"{min_gap_ginv:+.6e}")

    # ------------------------------------------------------------------ #
    # Check 5: Non-invariant E — can it push below √3?
    # ------------------------------------------------------------------ #
    print("\n--- Check 5: Non-invariant E — can it push below √3? ---")
    min_gap_ni = math.inf
    max_gap_ni = -math.inf
    below_count = 0
    for _ in range(100):
        # Sample random non-invariant coords (last 11 only)
        coords = np.zeros(15)
        coords[4:] = rng.uniform(-10, 10, 11)
        E = build_E_from_coords(coords)
        en = float(np.linalg.norm(E))
        if en > budget_en:
            E = E * (budget_en / en)
        T_p = T - eps * E
        gap = spectral_analysis(T_p)['sqrt3_gap']
        if gap < min_gap_ni:
            min_gap_ni = gap
        if gap > max_gap_ni:
            max_gap_ni = gap
        if gap < 0:
            below_count += 1
    print(f"  Non-invariant E: min sqrt3_gap = {min_gap_ni:+.6e}")
    print(f"  Non-invariant E: max sqrt3_gap = {max_gap_ni:+.6e}")
    print(f"  Samples that pushed below √3: {below_count}/100")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'step4_irrep.json')
    out = {
        'basis_orthogonality_max_error': max_err,
        'T_matmul_mode0_svs': [float(x) for x in spec['mode0_svs']],
        'sqrt3_position': int(np.argmin(np.abs(np.array(spec['mode0_svs']) - sqrt3))),
        'G_inv_min_gap': min_gap_ginv,
        'non_inv_min_gap': min_gap_ni,
        'non_inv_samples_below_sqrt3': below_count,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
