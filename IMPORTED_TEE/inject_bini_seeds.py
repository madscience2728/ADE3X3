"""Inject 3 near-associative n=9 Bini seeds into the DB and gen 12 seed list.

Constructs M(3,R) + ε_i * dC_i for three random perturbation directions,
tuning ε_i so that assoc_defect_norm ∈ {0.025, 0.055, 0.085} — all in the
near-assoc Bini zone [0.01, 0.10].  These are inserted into the DB tagged
with BINI_SEED flag, then appended to gen 12's seeds_hashes_json so gen 13
picks them up as additional starting seeds.

Usage:
    python matmul_search/inject_bini_seeds.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ade.constraints.checks import verify_all
from ade.core.algebra_state import AlgebraState
from ade.db.algebra_db import AlgebraDB
from ade.invariants.tier1 import compute_associativity_defect, tier1
from matmul_search.matmul_tensor import matrix_algebra

# Target assoc_defect values — spread across [0.01,0.10], well-separated (>0.02)
TARGETS = [0.025, 0.055, 0.085]
RNG_SEEDS = [1701, 1702, 1703]


def _tier1_fingerprint(A: AlgebraState) -> tuple:
    """Compute tier-1-only fingerprint for n=9 (H² is too expensive)."""
    t1 = tier1(A)
    return (
        A.n,
        A.base_ring,
        t1["dim_center"],
        t1["killing_signature"],
        t1["nilpotency_class"],
        t1["assoc_defect_norm"],
        -1, -1, -1,    # placeholders for dim_Der, aut_proxy, dim_H2
    )


def _tune_epsilon(C_base: np.ndarray, dC: np.ndarray, target: float,
                  rtol: float = 0.01, max_iter: int = 60) -> tuple[float, float]:
    """Binary-search epsilon so assoc_defect_norm ≈ target (within rtol * target)."""
    eps_lo, eps_hi = 0.0, 10.0

    # First find an upper bracket where defect > target
    A_probe = AlgebraState(C_base.shape[0], "R", "probe")
    for _ in range(50):
        C_probe = C_base + eps_hi * dC
        A_probe.set_C_dense(C_probe)
        if compute_associativity_defect(A_probe)["assoc_defect_norm"] > target:
            break
        eps_hi *= 2.0

    for _ in range(max_iter):
        eps = (eps_lo + eps_hi) / 2.0
        C_new = C_base + eps * dC
        A_probe.set_C_dense(C_new)
        defect = compute_associativity_defect(A_probe)["assoc_defect_norm"]
        if defect < target:
            eps_lo = eps
        else:
            eps_hi = eps
        if abs(defect - target) < rtol * target:
            return eps, defect

    eps = (eps_lo + eps_hi) / 2.0
    C_new = C_base + eps * dC
    A_probe.set_C_dense(C_new)
    defect = compute_associativity_defect(A_probe)["assoc_defect_norm"]
    return eps, defect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be inserted without writing to DB.")
    args = parser.parse_args()

    print("=" * 60)
    print("Bini seed injection: near-assoc n=9 algebras")
    print("=" * 60)

    M3R = matrix_algebra(3)
    C_base = M3R.C().copy()
    n = C_base.shape[0]   # 9
    print(f"M(3,R) tensor loaded: shape={C_base.shape}, "
          f"||C||_F={np.linalg.norm(C_base):.4f}\n")

    db = AlgebraDB()
    new_hashes: list[str] = []
    seed_names: list[str] = []

    for idx, (target, rng_seed) in enumerate(zip(TARGETS, RNG_SEEDS), start=1):
        name = f"bini_seed_{idx}"
        print(f"--- {name} | target assoc_defect = {target:.3f} | rng_seed={rng_seed} ---")

        rng = np.random.default_rng(rng_seed)
        dC_raw = rng.standard_normal(C_base.shape)
        dC = dC_raw / np.linalg.norm(dC_raw)  # unit-norm perturbation direction

        eps, actual_defect = _tune_epsilon(C_base, dC, target)
        print(f"  epsilon = {eps:.6f}  |  actual assoc_defect = {actual_defect:.6f}")

        C_new = C_base + eps * dC
        A = AlgebraState(n, "R", name)
        A.set_C_dense(C_new)

        # Compute tier-1 fingerprint (H² too expensive for n=9)
        fp = _tier1_fingerprint(A)
        print(f"  fingerprint: {fp[:6]}...")  # first 6 elements

        # Compute constraint flags
        flags_raw = verify_all(A, tol=1e-8)
        flag_set: dict[str, bool] = {k: bool(v["verified"]) for k, v in flags_raw.items()}
        flag_set["FROBENIUS"] = False  # not checked by verify_all, default False
        active_flags = [k for k, v in flag_set.items() if v]
        print(f"  flags: {active_flags if active_flags else ['(none)']}")

        if args.dry_run:
            print(f"  [DRY-RUN] would insert '{name}' — skipping DB write.")
            new_hashes.append(f"DRYRUN_{idx}")
            seed_names.append(name)
            print()
            continue

        fph = db.insert(
            name, A, fp, flag_set,
            generation=12,      # mark as part of gen 12 seed pool
            is_known=False,
            depth=0,
            seed_name="M(3,R)+perturbation",
            rng_seed=rng_seed,
            score=0.0,
        )
        new_hashes.append(fph)
        seed_names.append(name)
        print(f"  fp_hash = {fph}")
        print()

    if args.dry_run:
        print("\n[DRY-RUN] No DB writes performed.")
        return

    # ----------------------------------------------------------------
    # Append to gen 12's seeds_hashes_json
    # ----------------------------------------------------------------
    cur = db._con.execute(
        "SELECT seeds_hashes_json FROM gen_log WHERE gen_idx=12"
    ).fetchone()

    if cur is None:
        print("WARNING: gen_idx=12 not found in gen_log — cannot append seeds.")
        print("Run gen 12 first, or manually insert a gen_log row.")
        return

    existing = json.loads(cur[0]) if cur[0] else []
    print(f"Gen 12 seeds_hashes_json: {len(existing)} hashes before injection.")

    # Avoid duplicates
    already = set(existing)
    to_add = [h for h in new_hashes if h not in already]
    updated = existing + to_add

    db._con.execute(
        "UPDATE gen_log SET seeds_hashes_json=? WHERE gen_idx=12",
        (json.dumps(updated),),
    )
    db._con.commit()

    print(f"Gen 12 seeds_hashes_json: {len(updated)} hashes after injection "
          f"(added {len(to_add)} new).")
    print("\nInjected Bini seeds:")
    for name, fph in zip(seed_names, new_hashes):
        print(f"  {name:25s}  fp_hash={fph}")
    print("\nDone. Gen 13 will now start with these near-assoc n=9 seeds.")


if __name__ == "__main__":
    main()
