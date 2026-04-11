"""
r19_kernel_attack.py — Systematic R=19 kernel configuration attack.

Enumerates ALL G-irrep kernel shapes of dimension 10 in ℝ^19,
solves each in parallel across 24 cores, validates against A1–A5.

Two-phase approach:
  Phase 1 (TRIAGE):  3 seeds per shape × 526 shapes = 1578 work units → 24 cores
  Phase 2 (DEEP):    50 seeds per promising shape (res < 0.1) → 24 cores

"Every kernel shape gets its day in court." — Thread 2
"""
import sys, os, json, time, warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize

NCORES = 24
R = 19
TARGET_KERNEL_DIM = R - 9  # = 10


# ═══════════════════════════════════════════════════════════════
# WORKER: pure function, no imports from CANON_ADE at module level
# (avoids pickling issues with multiprocessing)
# ═══════════════════════════════════════════════════════════════

def _solve_one(args):
    """
    Single work unit: (shape_idx, combo, kernel_basis_flat, R, active_dim, seed)
    Returns dict with shape_idx, seed, residual, params (or None).
    """
    shape_idx, combo, kb_flat, kb_shape, R, seed = args

    kernel_basis = kb_flat.reshape(kb_shape)
    target_dim = kernel_basis.shape[1]
    U, s, Vt = np.linalg.svd(kernel_basis, full_matrices=True)
    active_basis = U[:, target_dim:]  # (R, 9)

    # Build T_matmul inline to avoid cross-process import
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(3):
        for ss in range(3):
            for u in range(3):
                T[3*r+ss, 3*ss+u, 3*r+u] = 1.0

    def objective_and_grad(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)
        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        diff = T_recon - T
        obj = 0.5 * np.sum(diff ** 2)
        dA = np.einsum('ijk,tj,tk->ti', diff, B, C)
        dB = np.einsum('ijk,ti,tk->tj', diff, A, C)
        dC = np.einsum('ijk,ti,tj->tk', diff, A, B)
        grad = np.concatenate([
            (active_basis.T @ dA).ravel(),
            (active_basis.T @ dB).ravel(),
            (active_basis.T @ dC).ravel(),
        ])
        return obj, grad

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(243) * 0.5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda p: objective_and_grad(p), x0, jac=True,
                           method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
        # Compute final residual
        M_A = res.x[:81].reshape(9, 9)
        M_B = res.x[81:162].reshape(9, 9)
        M_C = res.x[162:243].reshape(9, 9)
        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C
        T_recon = np.einsum('ti,tj,tk->ijk', A, B, C)
        final_res = float(np.max(np.abs(T_recon - T)))
        return {
            'shape_idx': shape_idx, 'seed': seed,
            'residual': final_res, 'params': res.x,
        }
    except Exception as e:
        return {
            'shape_idx': shape_idx, 'seed': seed,
            'residual': float('inf'), 'params': None,
        }


def main():
    freeze_support()
    from CANON_ADE.axiom_engine import tensor_core as tc
    from CANON_ADE.axiom_engine.algebra_constructor import (
        decompose_irreps_clean,
        enumerate_kernel_shapes,
    )
    from CANON_ADE.axiom_engine.axiom_evaluators import (
        eval_A1_conservation,
        eval_A2_parity,
        eval_A3_fiber,
        eval_A4_relation_module,
        eval_A5_coord_liberation,
    )

    print("=" * 70)
    print(f"R=19 KERNEL CONFIGURATION ATTACK — 24 CORES")
    print(f"Target: rank-19 decomposition of T_matmul(3×3)")
    print("=" * 70)

    # §1 Decompose
    print("\n§1  G-IRREP DECOMPOSITION OF ℝ^19")
    irreps = decompose_irreps_clean(R=19)
    dims = [d for d, _ in irreps]
    print(f"  {len(irreps)} irreps, dims = {dims}, sum = {sum(dims)}")
    assert sum(dims) == 19

    # §2 Enumerate kernel shapes
    print(f"\n§2  KERNEL SHAPE ENUMERATION (target dim = {TARGET_KERNEL_DIM})")
    kernel_shapes = enumerate_kernel_shapes(irreps, TARGET_KERNEL_DIM)
    print(f"  {len(kernel_shapes)} valid kernel shapes")

    # Pre-compute kernel bases (serializable numpy arrays)
    kernel_bases = []
    for combo in kernel_shapes:
        kb = np.hstack([irreps[i][1] for i in combo])
        kernel_bases.append(kb)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: TRIAGE — 3 seeds per shape, all 24 cores
    # ═══════════════════════════════════════════════════════════
    TRIAGE_SEEDS = 3
    n_shapes = len(kernel_shapes)
    work_units = []
    for i, combo in enumerate(kernel_shapes):
        kb = kernel_bases[i]
        for seed in range(TRIAGE_SEEDS):
            work_units.append((i, list(combo), kb.ravel(), kb.shape, R, seed * 1000 + i))

    print(f"\n§3  PHASE 1: TRIAGE — {n_shapes} shapes × {TRIAGE_SEEDS} seeds = {len(work_units)} jobs on {NCORES} cores")
    print("-" * 70)

    t0 = time.time()
    triage_results = {}  # shape_idx → best residual & params
    done_count = 0

    with ProcessPoolExecutor(max_workers=NCORES) as pool:
        futures = {pool.submit(_solve_one, wu): wu[0] for wu in work_units}
        for future in as_completed(futures):
            result = future.result()
            idx = result['shape_idx']
            done_count += 1

            if idx not in triage_results or result['residual'] < triage_results[idx]['residual']:
                triage_results[idx] = result

            if done_count % 100 == 0 or done_count == len(work_units):
                elapsed = time.time() - t0
                best_so_far = min(r['residual'] for r in triage_results.values())
                exact_so_far = sum(1 for r in triage_results.values() if r['residual'] < 1e-10)
                print(f"  [{done_count:5d}/{len(work_units)}] {elapsed:6.1f}s  "
                      f"best={best_so_far:.3e}  exact={exact_so_far}")

    phase1_time = time.time() - t0

    # Sort by residual
    sorted_shapes = sorted(triage_results.items(), key=lambda x: x[1]['residual'])

    # Print triage results
    print(f"\n  Phase 1 complete: {phase1_time:.1f}s")
    exact_count = sum(1 for _, r in sorted_shapes if r['residual'] < 1e-10)
    close_count = sum(1 for _, r in sorted_shapes if 1e-10 <= r['residual'] < 1e-3)
    warm_count = sum(1 for _, r in sorted_shapes if 1e-3 <= r['residual'] < 0.1)
    print(f"  ★ EXACT  (res < 1e-10): {exact_count}")
    print(f"  ◆ CLOSE  (res < 1e-3):  {close_count}")
    print(f"  ◇ WARM   (res < 0.1):   {warm_count}")
    print(f"  · FAR    (res ≥ 0.1):   {n_shapes - exact_count - close_count - warm_count}")

    print(f"\n  Top 20 shapes:")
    for rank, (idx, r) in enumerate(sorted_shapes[:20]):
        combo_dims = [dims[j] for j in kernel_shapes[idx]]
        tag = "★" if r['residual'] < 1e-10 else ("◆" if r['residual'] < 1e-3 else "◇" if r['residual'] < 0.1 else "·")
        print(f"    {rank+1:3d}. shape[{idx:3d}] dims={combo_dims}  res={r['residual']:.6e}  {tag}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: DEEP DIVE — 50 seeds on promising shapes (res < 0.1)
    # ═══════════════════════════════════════════════════════════
    promising = [(idx, r) for idx, r in sorted_shapes if r['residual'] < 0.1 and r['residual'] >= 1e-10]
    # Also re-run exact ones with more seeds to find MORE exact solutions
    already_exact = [(idx, r) for idx, r in sorted_shapes if r['residual'] < 1e-10]

    DEEP_SEEDS = 50
    deep_targets = [idx for idx, _ in promising]
    # For already-exact shapes, just keep the result, no need to re-run

    if deep_targets:
        deep_work = []
        for idx in deep_targets:
            kb = kernel_bases[idx]
            for seed in range(DEEP_SEEDS):
                deep_work.append((idx, list(kernel_shapes[idx]), kb.ravel(), kb.shape, R, 10000 + seed * 1000 + idx))

        print(f"\n§4  PHASE 2: DEEP DIVE — {len(deep_targets)} shapes × {DEEP_SEEDS} seeds = {len(deep_work)} jobs")
        print("-" * 70)

        t1 = time.time()
        deep_results = {idx: triage_results[idx] for idx in deep_targets}  # start from triage best
        done_count = 0

        with ProcessPoolExecutor(max_workers=NCORES) as pool:
            futures = {pool.submit(_solve_one, wu): wu[0] for wu in deep_work}
            for future in as_completed(futures):
                result = future.result()
                idx = result['shape_idx']
                done_count += 1

                if result['residual'] < deep_results[idx]['residual']:
                    deep_results[idx] = result

                if done_count % 200 == 0 or done_count == len(deep_work):
                    elapsed = time.time() - t1
                    new_exact = sum(1 for r in deep_results.values() if r['residual'] < 1e-10)
                    print(f"  [{done_count:5d}/{len(deep_work)}] {elapsed:6.1f}s  new_exact={new_exact}")

        phase2_time = time.time() - t1
        print(f"\n  Phase 2 complete: {phase2_time:.1f}s")

        # Merge deep results back
        for idx, r in deep_results.items():
            if r['residual'] < triage_results[idx]['residual']:
                triage_results[idx] = r
    else:
        print(f"\n§4  PHASE 2: SKIPPED — no warm shapes to deep-dive")
        phase2_time = 0

    # ═══════════════════════════════════════════════════════════
    # §5: AXIOM VALIDATION on all solutions with res < 1e-3
    # ═══════════════════════════════════════════════════════════
    print(f"\n§5  AXIOM VALIDATION")
    print("-" * 70)

    # Re-sort after phase 2
    sorted_shapes = sorted(triage_results.items(), key=lambda x: x[1]['residual'])
    exact_count = sum(1 for _, r in sorted_shapes if r['residual'] < 1e-10)
    close_count = sum(1 for _, r in sorted_shapes if 1e-10 <= r['residual'] < 1e-3)
    print(f"  Final: {exact_count} exact, {close_count} close")

    validated = []
    for idx, r in sorted_shapes:
        if r['residual'] >= 1e-3 or r['params'] is None:
            continue

        params = r['params']
        combo = kernel_shapes[idx]
        kb = kernel_bases[idx]
        target_dim = kb.shape[1]
        U, s, Vt = np.linalg.svd(kb, full_matrices=True)
        active_basis = U[:, target_dim:]

        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)
        alpha = (active_basis @ M_A).reshape(R, 3, 3)
        beta = (active_basis @ M_B).reshape(R, 3, 3)
        gamma = (active_basis @ M_C).reshape(R, 3, 3)

        Sigma, H, Delta = tc.compute_step51(alpha, beta)

        a1 = eval_A1_conservation(Sigma, H, Delta, R)
        a2 = eval_A2_parity(alpha, beta, gamma, R)
        a3 = eval_A3_fiber(Sigma, R)
        a4 = eval_A4_relation_module(alpha, beta, gamma, R)
        a5 = eval_A5_coord_liberation(Sigma, H, Delta, R)

        n_pass = sum([
            a1['A1_pass'], a2['A2_pass'], a3['A3_achievable'],
            a4['A4_generic_position'], a5['A5_gates_compatible']
        ])

        combo_dims = [dims[j] for j in combo]
        print(f"\n  shape[{idx:3d}] dims={combo_dims}  res={r['residual']:.3e}  AXIOMS={n_pass}/5")
        print(f"    A1(conservation): {'✓' if a1['A1_pass'] else '✗'}  rank(H)={a1['A1_rank_H']} (target {R-9})")
        print(f"    A2(parity):       {'✓' if a2['A2_pass'] else '✗'}  fitness={a2['A2_fitness']:.3e}")
        print(f"    A3(fiber):        {'✓' if a3['A3_achievable'] else '✗'}  rank(Σ)={a3['A3_rank_sigma']}")
        print(f"    A4(relation):     {'✓' if a4['A4_generic_position'] else '✗'}  dim(K_A)={a4['A4_dim_KA']}, dim(K_B)={a4['A4_dim_KB']}, dim(K_C)={a4['A4_dim_KC']}")
        print(f"    A5(liberation):   {'✓' if a5['A5_gates_compatible'] else '✗'}  gap={a5['A5_aug_gap']} gate2={a5['A5_gate2']} gate3={a5['A5_gate3']}")

        validated.append({
            'shape_idx': int(idx),
            'combo': list(combo),
            'combo_dims': combo_dims,
            'residual': float(r['residual']),
            'n_pass': n_pass,
            'A1_pass': a1['A1_pass'], 'A1_rank_H': a1['A1_rank_H'],
            'A2_pass': a2['A2_pass'], 'A2_fitness': float(a2['A2_fitness']),
            'A3_achievable': a3['A3_achievable'], 'A3_rank_sigma': a3['A3_rank_sigma'],
            'A4_generic': a4['A4_generic_position'],
            'A4_dims': [a4['A4_dim_KA'], a4['A4_dim_KB'], a4['A4_dim_KC']],
            'A5_compatible': a5['A5_gates_compatible'],
            'A5_aug_gap': a5['A5_aug_gap'],
        })

    # ═══════════════════════════════════════════════════════════
    # §6: SUMMARY
    # ═══════════════════════════════════════════════════════════
    total_time = phase1_time + phase2_time
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {total_time:.1f}s total ({phase1_time:.1f}s triage + {phase2_time:.1f}s deep)")
    print(f"{'=' * 70}")
    print(f"  Total kernel shapes:     {n_shapes}")
    print(f"  Exact (res < 1e-10):     {exact_count}")
    print(f"  Close (res < 1e-3):      {close_count}")
    print(f"  Validated:               {len(validated)}")

    if validated:
        best_v = max(validated, key=lambda x: (x['n_pass'], -x['residual']))
        print(f"\n  BEST CANDIDATE:")
        print(f"    Shape[{best_v['shape_idx']}]: {best_v['combo']}  dims={best_v['combo_dims']}")
        print(f"    Residual: {best_v['residual']:.3e}")
        print(f"    Axioms: {best_v['n_pass']}/5")

        # Axiom-pass distribution
        from collections import Counter
        pass_dist = Counter(v['n_pass'] for v in validated)
        print(f"\n  Axiom-pass distribution:")
        for k in sorted(pass_dist.keys(), reverse=True):
            print(f"    {k}/5 axioms: {pass_dist[k]} shapes")

    # Save
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'R': R,
        'ncores': NCORES,
        'n_irreps': len(irreps),
        'irrep_dims': dims,
        'n_kernel_shapes': n_shapes,
        'triage_seeds': TRIAGE_SEEDS,
        'deep_seeds': DEEP_SEEDS,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'n_exact': exact_count,
        'n_close': close_count,
        'best_residual': float(sorted_shapes[0][1]['residual']) if sorted_shapes else float('inf'),
        'all_residuals': [
            {'shape_idx': int(idx), 'combo': list(kernel_shapes[idx]),
             'combo_dims': [dims[j] for j in kernel_shapes[idx]],
             'residual': float(r['residual'])}
            for idx, r in sorted_shapes
        ],
        'validated': validated,
    }
    outpath = os.path.join(os.path.dirname(__file__), 'results', 'r19_kernel_attack.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")


if __name__ == '__main__':
    main()
