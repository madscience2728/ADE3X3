"""
Gate-3 Free Sweep
=================
Tests the unconditional Gate-3 criterion (augmented rank) without requiring
Gate 1 or 2 first. For each 19-term draw, computes:

  aug_gap = rank([Sigma | Nuisance]) - rank(Nuisance)

Gate 3 requires aug_gap = 9 (Sigma spans 9 new dimensions beyond nuisance).

Also records rank(H) and rank([H|Delta]) so we can identify which draws
satisfy Gates 1 and/or 2 and compare their aug_gap distribution.

Draws:
  - Random draws from full TermDB (uniform over 387M records)
  - Draws from existing R=19 packet files (basis + random pool)
"""
import numpy as np
from pathlib import Path

DB = Path("./CANON_DATABASE/data")
print("Loading TermDB ...", end=" ", flush=True)
H_db     = np.load(DB / "H.npy",        mmap_mode="r")
sigma_db = np.load(DB / "sigma.npy",    mmap_mode="r")
alpha_db = np.load(DB / "alpha_idx.npy", mmap_mode="r")
beta_db  = np.load(DB / "beta_idx.npy",  mmap_mode="r")
templates = np.load(DB / "templates.npy")
N = H_db.shape[0]
print(f"done. N={N:,}")

RNG = np.random.default_rng(7)
RANK_CONFIGS = {13: 4, 19: 10, 20: 11}

def stable_rank(M, rel_tol=1e-9, abs_tol=1e-12):
    sv = np.linalg.svd(M.astype(float), compute_uv=False)
    if len(sv) == 0 or sv[0] == 0:
        return 0
    return int(np.sum(sv > max(abs_tol, rel_tol * sv[0])))

def build_delta(idx):
    alphas = templates[alpha_db[idx]]
    betas  = templates[beta_db[idx]]
    pairs  = [(s,t) for s in range(3) for t in range(3) if s != t]
    Delta  = np.zeros((len(idx), 54), dtype=float)
    for ci, (s,t) in enumerate(pairs):
        Delta[:, ci*9:(ci+1)*9] = (
            alphas[:,:,s][:,:,None] * betas[:,t,:][:,None,:]
        ).reshape(len(idx), 9)
    return Delta

def diagnose(idx, target_H):
    """Returns (rH, delta_leak, aug_gap, rS_quotient)."""
    idx = np.asarray(idx)
    H     = H_db[idx].astype(float)
    Sigma = sigma_db[idx].astype(float)
    Delta = build_delta(idx)

    Nuisance = np.hstack([H, Delta])
    rH  = stable_rank(H)
    rN  = stable_rank(Nuisance)
    delta_leak = rN - rH

    aug = np.hstack([Sigma, Nuisance])
    rAN = stable_rank(aug)
    aug_gap = rAN - rN

    rS_q = None
    if rH == target_H:
        _, sv, Vt = np.linalg.svd(H.T, full_matrices=True)
        tol = max(1e-12, 1e-9 * sv[0])
        r = int(np.sum(sv > tol))
        U = Vt[r:]
        rS_q = stable_rank(U @ Sigma)

    return rH, delta_leak, aug_gap, rS_q

for R, TARGET_H in sorted(RANK_CONFIGS.items()):
    print(f"\n{'='*60}")
    print(f"=== R={R}  target rank(H)={TARGET_H} ===")

    # ── Sweep 1: pure random draws ──────────────────────────────────
    N_RANDOM = 5000
    print(f"\n  Sweep 1: {N_RANDOM} random {R}-term draws from TermDB")
    aug_gap_hist    = {}
    gate1_aug_hist  = {}
    gate12_aug_hist = {}
    gate12_rSq_hist = {}
    n_g1 = n_g12 = 0

    for _ in range(N_RANDOM):
        idx = RNG.integers(0, N, size=R)
        rH, dl, ag, rSq = diagnose(idx, TARGET_H)
        aug_gap_hist[ag] = aug_gap_hist.get(ag, 0) + 1
        if rH == TARGET_H:
            n_g1 += 1
            gate1_aug_hist[ag] = gate1_aug_hist.get(ag, 0) + 1
            if dl == 0:
                n_g12 += 1
                gate12_aug_hist[ag] = gate12_aug_hist.get(ag, 0) + 1
                if rSq is not None:
                    gate12_rSq_hist[rSq] = gate12_rSq_hist.get(rSq, 0) + 1
                if ag == 9:
                    print(f"  *** ALL THREE GATES — random draw ***")

    print(f"  Gate-1: {n_g1}  Gate-1+2: {n_g12}")
    print(f"  aug_gap (all draws): { {k: aug_gap_hist[k] for k in sorted(aug_gap_hist)} }")
    if gate1_aug_hist:
        print(f"  aug_gap | Gate-1:    { {k: gate1_aug_hist[k] for k in sorted(gate1_aug_hist)} }")
    if gate12_aug_hist:
        print(f"  aug_gap | Gate-1+2:  { {k: gate12_aug_hist[k] for k in sorted(gate12_aug_hist)} }")
        print(f"  rS_quot | Gate-1+2:  { {k: gate12_rSq_hist[k] for k in sorted(gate12_rSq_hist)} }")

    # ── Sweep 2: packet-file draws ───────────────────────────────────
    packet_dir = Path(f"./CANON_DATABASE/packets/R{R}")
    if not packet_dir.exists():
        print(f"\n  Sweep 2: no packet dir found, skipping")
        continue
    files = sorted(packet_dir.glob("*.npz"))
    print(f"\n  Sweep 2: {len(files)} packet files × 4 draws")

    pkt_aug_hist = {}
    pkt_g1_aug = {}
    pkt_g12_aug = {}
    pkt_g12_rSq = {}
    pkt_n_g1 = pkt_n_g12 = pkt_n_tested = 0

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            basis = d["basis_record_indices"]
            hits  = d["hit_record_indices"]
        except Exception:
            continue
        need = R - len(basis)
        if len(hits) < need:
            continue
        for draw in range(4):
            extra = hits[:need] if draw == 0 else hits[RNG.integers(0, len(hits), size=need)]
            idx = np.concatenate([basis, extra])
            rH, dl, ag, rSq = diagnose(idx, TARGET_H)
            pkt_n_tested += 1
            pkt_aug_hist[ag] = pkt_aug_hist.get(ag, 0) + 1
            if rH == TARGET_H:
                pkt_n_g1 += 1
                pkt_g1_aug[ag] = pkt_g1_aug.get(ag, 0) + 1
                if dl == 0:
                    pkt_n_g12 += 1
                    pkt_g12_aug[ag] = pkt_g12_aug.get(ag, 0) + 1
                    if rSq is not None:
                        pkt_g12_rSq[rSq] = pkt_g12_rSq.get(rSq, 0) + 1
                    if ag == 9:
                        print(f"  *** ALL THREE GATES — packet draw *** {f.name}")

    print(f"  Draws: {pkt_n_tested}  Gate-1: {pkt_n_g1}  Gate-1+2: {pkt_n_g12}")
    print(f"  aug_gap | Gate-1:   { {k: pkt_g1_aug[k] for k in sorted(pkt_g1_aug)} }")
    if pkt_g12_aug:
        print(f"  aug_gap | G1+2:    { {k: pkt_g12_aug[k] for k in sorted(pkt_g12_aug)} }")
        print(f"  rS_quot | G1+2:    { {k: pkt_g12_rSq[k] for k in sorted(pkt_g12_rSq)} }")

print("\n=== Done ===")
