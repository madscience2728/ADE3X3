"""
multi_rank_attack.py — Full-parallel multi-rank kernel attack.

ALL ranks R=9..22, ALL kernel shapes, ALL seeds submitted to a single
24-core pool simultaneously. No sequential rank-by-rank bottleneck.

Phase 1: 3 seeds/shape across all ranks (triage)
Phase 2: 50 seeds on warm shapes (deep dive)
Then: axiom validation on solutions with res < 1e-3.
"""
import sys, os, json, time, warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from itertools import combinations as combs
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize

NCORES = 24
TRIAGE_SEEDS = 3
DEEP_SEEDS = 50


def _solve_one(args):
    """Worker: (run_key, shape_idx, kb_flat, kb_shape, seed) -> result tuple."""
    run_key, shape_idx, kb_flat, kb_shape, seed = args
    import numpy as np
    from scipy.optimize import minimize
    import warnings

    kernel_basis = kb_flat.reshape(kb_shape)
    R = kb_shape[0]
    target_dim = kernel_basis.shape[1]

    if target_dim > 0:
        U, s, Vt = np.linalg.svd(kernel_basis, full_matrices=True)
        active_basis = U[:, target_dim:]
    else:
        active_basis = np.eye(R, dtype=np.float64)

    T = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[3*r+s, 3*s+u, 3*r+u] = 1.0

    def obj_grad(params):
        M_A = params[:81].reshape(9, 9)
        M_B = params[81:162].reshape(9, 9)
        M_C = params[162:243].reshape(9, 9)
        A = active_basis @ M_A
        B = active_basis @ M_B
        C = active_basis @ M_C
        T_r = np.einsum('ti,tj,tk->ijk', A, B, C)
        d = T_r - T
        obj = 0.5 * np.sum(d * d)
        dA = np.einsum('ijk,tj,tk->ti', d, B, C)
        dB = np.einsum('ijk,ti,tk->tj', d, A, C)
        dC = np.einsum('ijk,ti,tj->tk', d, A, B)
        g = np.concatenate([(active_basis.T @ dA).ravel(),
                            (active_basis.T @ dB).ravel(),
                            (active_basis.T @ dC).ravel()])
        return obj, g

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(243) * 0.5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda p: obj_grad(p), x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-12})
        p = res.x
        A = active_basis @ p[:81].reshape(9, 9)
        B = active_basis @ p[81:162].reshape(9, 9)
        C = active_basis @ p[162:243].reshape(9, 9)
        T_r = np.einsum('ti,tj,tk->ijk', A, B, C)
        fr = float(np.max(np.abs(T_r - T)))
        return (run_key, shape_idx, fr, p)
    except Exception:
        return (run_key, shape_idx, float('inf'), None)


def main():
    freeze_support()
    from CANON_ADE.axiom_engine import tensor_core as tc
    from CANON_ADE.axiom_engine.algebra_constructor import (
        decompose_irreps_clean, enumerate_kernel_shapes,
    )
    from CANON_ADE.axiom_engine.axiom_evaluators import (
        eval_A1_conservation, eval_A2_parity, eval_A3_fiber,
        eval_A4_relation_module, eval_A5_coord_liberation,
    )

    print("=" * 70)
    print(f"MULTI-RANK KERNEL ATTACK — R=9..22 — {NCORES} CORES — FULL PARALLEL")
    print("=" * 70)

    # ─── Build support sets ───
    orbit_list = [('O0', tc.ORBIT_0), ('O1', tc.ORBIT_1),
                  ('O2', tc.ORBIT_2), ('O3', tc.ORBIT_3)]
    support_sets = {}
    seen = set()

    def add(R, kept, label):
        k = tuple(sorted(kept))
        if k in seen: return
        seen.add(k)
        support_sets.setdefault(R, []).append((sorted(kept), label))

    for n in range(1, 5):
        for sub in combs(range(4), n):
            kept, names = [], []
            for i in sub:
                kept.extend(orbit_list[i][1]); names.append(orbit_list[i][0])
            add(len(kept), kept, '+'.join(names))

    # Only G-stable (orbit union) supports — H-stable would create
    # millions of jobs with no mathematical advantage.
    # G-stable ranks in [9,22]: {9, 12, 13, 14, 19, 20, 21}

    # ─── Pre-compute all decompositions ───
    run_meta = {}
    total_triage_jobs = 0

    print(f"\n§1  ENUMERATION")
    print("-" * 70)
    for R in range(9, 23):
        if R not in support_sets:
            continue
        for kept, label in support_sets[R]:
            kernel_dim = R - 9
            run_key = f"R{R}_{label}"

            irreps = decompose_irreps_clean(R=R, kept=kept)
            dims = [d for d, _ in irreps]
            if sum(dims) != R:
                print(f"  {run_key}: irrep mismatch ({sum(dims)}!={R}), skip")
                continue

            if kernel_dim == 0:
                ks = [()]
                kbs = [np.zeros((R, 0), dtype=np.float64)]
            elif kernel_dim < 0:
                continue
            else:
                ks = enumerate_kernel_shapes(irreps, kernel_dim)
                if not ks:
                    print(f"  {run_key}: no kernel shapes for dim={kernel_dim}, skip")
                    continue
                kbs = [np.hstack([irreps[i][1] for i in c]) for c in ks]

            run_meta[run_key] = {
                'R': R, 'label': label, 'kept': kept, 'irreps': irreps,
                'dims': dims, 'kernel_shapes': ks, 'kernel_bases': kbs,
            }
            n = len(ks)
            total_triage_jobs += n * TRIAGE_SEEDS
            print(f"  {run_key:<30s}  shapes={n:5d}  triage_jobs={n*TRIAGE_SEEDS:6d}")

    print(f"\n  Total runs: {len(run_meta)}")
    print(f"  Total Phase 1 jobs: {total_triage_jobs}")

    # ─── PHASE 1: ALL TRIAGE IN ONE POOL ───
    print(f"\n§2  PHASE 1: TRIAGE — {total_triage_jobs} jobs on {NCORES} cores")
    print("=" * 70)

    all_work = []
    for run_key, m in run_meta.items():
        for i, kb in enumerate(m['kernel_bases']):
            for s in range(TRIAGE_SEEDS):
                seed = s * 7919 + i * 97 + m['R'] * 1000003
                all_work.append((run_key, i, kb.ravel(), kb.shape, seed))

    grand_t0 = time.time()
    best_per = defaultdict(dict)  # best_per[run_key][shape_idx] = (res, params)
    done = 0

    with ProcessPoolExecutor(max_workers=NCORES) as pool:
        futs = {pool.submit(_solve_one, w): w for w in all_work}
        for f in as_completed(futs):
            rk, idx, res, params = f.result()
            done += 1
            if idx not in best_per[rk] or res < best_per[rk][idx][0]:
                best_per[rk][idx] = (res, params)
            if done % 500 == 0 or done == len(all_work):
                elapsed = time.time() - grand_t0
                ne = sum(1 for sh in best_per.values() for r, _ in sh.values() if r < 1e-10)
                print(f"  [{done:6d}/{len(all_work)}] {elapsed:7.1f}s  total_exact={ne}", flush=True)

    phase1_time = time.time() - grand_t0
    print(f"\n  Phase 1 done: {phase1_time:.1f}s")

    # Triage summary
    print(f"\n  TRIAGE SUMMARY:")
    for rk in sorted(run_meta.keys(), key=lambda k: (run_meta[k]['R'], k)):
        if rk not in best_per: continue
        shapes = best_per[rk]
        best = min(r for r, _ in shapes.values())
        nexact = sum(1 for r, _ in shapes.values() if r < 1e-10)
        nclose = sum(1 for r, _ in shapes.values() if 1e-10 <= r < 1e-3)
        nwarm = sum(1 for r, _ in shapes.values() if 1e-3 <= r < 0.1)
        R = run_meta[rk]['R']
        tag = " ★★★" if nexact > 0 else (" ◆" if nclose > 0 else (" ◇" if nwarm > 0 else ""))
        print(f"    R={R:2d} {run_meta[rk]['label']:<25s} shapes={len(shapes):5d}  "
              f"exact={nexact:3d} close={nclose:3d} warm={nwarm:3d}  best={best:.3e}{tag}")

    # ─── PHASE 2: DEEP DIVE ALL WARM SHAPES ───
    deep_work = []
    for rk, shapes in best_per.items():
        m = run_meta[rk]
        for idx, (res, _) in shapes.items():
            if 1e-10 <= res < 0.1:
                kb = m['kernel_bases'][idx]
                for s in range(DEEP_SEEDS):
                    seed = 50000 + s * 7919 + idx * 97 + m['R'] * 1000003
                    deep_work.append((rk, idx, kb.ravel(), kb.shape, seed))

    if deep_work:
        print(f"\n§3  PHASE 2: DEEP DIVE — {len(deep_work)} jobs on {NCORES} cores")
        print("=" * 70)
        t1 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=NCORES) as pool:
            futs = {pool.submit(_solve_one, w): w for w in deep_work}
            for f in as_completed(futs):
                rk, idx, res, params = f.result()
                done += 1
                if res < best_per[rk][idx][0]:
                    best_per[rk][idx] = (res, params)
                if done % 500 == 0 or done == len(deep_work):
                    elapsed = time.time() - t1
                    ne = sum(1 for sh in best_per.values() for r, _ in sh.values() if r < 1e-10)
                    print(f"  [{done:6d}/{len(deep_work)}] {elapsed:7.1f}s  total_exact={ne}", flush=True)
        phase2_time = time.time() - t1
        print(f"\n  Phase 2 done: {phase2_time:.1f}s")
    else:
        print(f"\n§3  PHASE 2: SKIPPED — no warm shapes")
        phase2_time = 0

    # ─── §4 AXIOM VALIDATION ───
    print(f"\n§4  AXIOM VALIDATION")
    print("=" * 70)

    grand_results = {}
    for rk in sorted(run_meta.keys(), key=lambda k: (run_meta[k]['R'], k)):
        if rk not in best_per: continue
        m = run_meta[rk]
        R, dims, ks, kbs = m['R'], m['dims'], m['kernel_shapes'], m['kernel_bases']

        sorted_shapes = sorted(best_per[rk].items(), key=lambda x: x[1][0])
        exact_count = sum(1 for _, (r, _) in sorted_shapes if r < 1e-10)
        close_count = sum(1 for _, (r, _) in sorted_shapes if 1e-10 <= r < 1e-3)

        validated = []
        for idx, (res, params) in sorted_shapes:
            if res >= 1e-3 or params is None: continue

            kb = kbs[idx]
            if kb.shape[1] > 0:
                U, s, Vt = np.linalg.svd(kb, full_matrices=True)
                ab = U[:, kb.shape[1]:]
            else:
                ab = np.eye(R, dtype=np.float64)

            alpha = (ab @ params[:81].reshape(9, 9)).reshape(R, 3, 3)
            beta = (ab @ params[81:162].reshape(9, 9)).reshape(R, 3, 3)
            gamma = (ab @ params[162:243].reshape(9, 9)).reshape(R, 3, 3)
            Sigma, H, Delta = tc.compute_step51(alpha, beta)

            a1 = eval_A1_conservation(Sigma, H, Delta, R)
            a2 = eval_A2_parity(alpha, beta, gamma, R)
            a3 = eval_A3_fiber(Sigma, R)
            a4 = eval_A4_relation_module(alpha, beta, gamma, R)
            a5 = eval_A5_coord_liberation(Sigma, H, Delta, R)

            np_ = sum([a1['A1_pass'], a2['A2_pass'], a3['A3_achievable'],
                       a4['A4_generic_position'], a5['A5_gates_compatible']])

            cd = [dims[j] for j in ks[idx]] if ks[idx] else []
            print(f"  R={R:2d} {m['label']:<20s} shape[{idx}] dims={cd} res={res:.3e} -> {np_}/5  "
                  f"A1={'Y' if a1['A1_pass'] else 'N'} A2={'Y' if a2['A2_pass'] else 'N'} "
                  f"A3={'Y' if a3['A3_achievable'] else 'N'} A4={'Y' if a4['A4_generic_position'] else 'N'} "
                  f"A5={'Y' if a5['A5_gates_compatible'] else 'N'}(gap={a5['A5_aug_gap']})")

            validated.append({
                'shape_idx': int(idx), 'combo': list(ks[idx]), 'combo_dims': cd,
                'residual': float(res), 'n_pass': np_,
                'A1': a1['A1_pass'], 'A2': a2['A2_pass'], 'A3': a3['A3_achievable'],
                'A4': a4['A4_generic_position'], 'A5': a5['A5_gates_compatible'],
                'A5_gap': a5['A5_aug_gap'], 'A1_rank_H': a1['A1_rank_H'],
            })

        grand_results[rk] = {
            'R': R, 'support': m['label'], 'n_shapes': len(ks),
            'irrep_dims': dims, 'n_exact': exact_count, 'n_close': close_count,
            'best_residual': float(sorted_shapes[0][1][0]) if sorted_shapes else float('inf'),
            'validated': validated,
        }

    # ─── GRAND SUMMARY ───
    total_time = time.time() - grand_t0
    print(f"\n{'='*70}")
    print(f"GRAND SUMMARY — {total_time:.1f}s ({phase1_time:.1f}s triage + {phase2_time:.1f}s deep)")
    print(f"{'='*70}")
    print(f"  {'R':>3s}  {'Support':<25s}  {'Shapes':>7s}  {'Exact':>6s}  {'Close':>6s}  {'Best Res':>12s}")
    print(f"  {'-'*3}  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*12}")
    for key in sorted(grand_results.keys(), key=lambda k: (grand_results[k]['R'], k)):
        g = grand_results[key]
        tag = " ★★★" if g['n_exact'] > 0 else (" ◆" if g['n_close'] > 0 else "")
        print(f"  {g['R']:3d}  {g['support']:<25s}  {g['n_shapes']:7d}  "
              f"{g['n_exact']:6d}  {g['n_close']:6d}  {g['best_residual']:12.3e}{tag}")

    exact_ranks = sorted(set(g['R'] for g in grand_results.values() if g['n_exact'] > 0))
    if exact_ranks:
        print(f"\n  ★ RANKS WITH EXACT SOLUTIONS: {exact_ranks}")
        print(f"  ★ LOWEST EXACT RANK: R = {min(exact_ranks)}")
    else:
        close_ranks = sorted(set(g['R'] for g in grand_results.values() if g['n_close'] > 0))
        if close_ranks:
            print(f"\n  ◆ RANKS WITH CLOSE SOLUTIONS: {close_ranks}")
        print(f"\n  No exact solutions found at any rank.")

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'ncores': NCORES, 'triage_seeds': TRIAGE_SEEDS, 'deep_seeds': DEEP_SEEDS,
        'phase1_time': phase1_time, 'phase2_time': phase2_time,
        'total_time': total_time, 'grand_results': grand_results,
    }
    outpath = os.path.join(os.path.dirname(__file__), 'results', 'multi_rank_kernel_attack.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved -> {outpath}")


if __name__ == '__main__':
    main()
