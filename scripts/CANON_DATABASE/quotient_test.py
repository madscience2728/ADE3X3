"""
Quotient Obstruction Test
=========================
Tests whether det(U @ Sigma) = 0 is a universal obstruction at Gate 1,
and whether any Gate-1 packet also satisfies Gate 3 (rank-9 quotient sigma).
Then: for Gate-1+Gate-3 survivors, check Gate 2 (delta_leak).
Runs for R=13, R=19, R=20.
"""

import numpy as np
import json
from pathlib import Path

DB_PATH = Path("./CANON_DATABASE/data")

# ── load DB ──────────────────────────────────────────────────────────────────
print("Loading TermDB arrays ...", end=" ", flush=True)
H_db     = np.load(DB_PATH / "H.npy",        mmap_mode="r")   # (387M, 18) int8
sigma_db = np.load(DB_PATH / "sigma.npy",    mmap_mode="r")   # (387M,  9) int8
alpha_db = np.load(DB_PATH / "alpha_idx.npy", mmap_mode="r")  # (387M,) uint16
beta_db  = np.load(DB_PATH / "beta_idx.npy",  mmap_mode="r")  # (387M,) uint16
templates = np.load(DB_PATH / "templates.npy")                 # (19683, 3, 3) int8
print(f"done. N={H_db.shape[0]:,}")

# R → target rank(H)  (from ranks.md / CANON_CONSTRAINTS.md)
TARGET_H = {13: 4, 19: 10, 20: 11}
RNG = np.random.default_rng(42)

# ── helper ────────────────────────────────────────────────────────────────────
def rank(M, rel_tol=1e-9, abs_tol=1e-12):
    sv = np.linalg.svd(M.astype(float), compute_uv=False)
    if len(sv) == 0 or sv[0] == 0:
        return 0
    threshold = max(abs_tol, rel_tol * sv[0])
    return int(np.sum(sv > threshold))

def col_perp(M):
    """Orthonormal basis for the left nullspace (row space complement) of M.
    M shape: (R, d). Returns U shape (9, R) where 9 = R - rank(M)."""
    _, s, Vt = np.linalg.svd(M.T.astype(float), full_matrices=True)
    n = M.shape[0]
    tol = 1e-9 * (s[0] if len(s) > 0 else 1.0)
    r = int(np.sum(s > tol))
    # null space of M.T = rows of Vt[r:]  → shape (n-r, n)
    return Vt[r:]   # (n-r, n)  i.e. U s.t. U @ M = 0 (approximately)

def test_packet(indices, tgt, label=""):
    idx = np.array(indices)
    R     = len(idx)
    H     = H_db[idx].astype(float)      # (R, 18)
    Sigma = sigma_db[idx].astype(float)  # (R,  9)

    rH = rank(H)
    if rH != tgt:
        return rH, None, None

    U = col_perp(H)          # (R-tgt, R)
    null_dim = R - tgt
    S = U @ Sigma            # (null_dim, 9)
    rS = rank(S)
    # det only defined when S is square
    det_S = abs(np.linalg.det(S)) if S.shape[0] == S.shape[1] else float('nan')
    return rH, rS, det_S

# ── Test 1: best Phase-4 packet (R=19 specific) ─────────────────────────────
with open("./CANON_DATABASE/swap_checkpoint.json") as f:
    ckpt = json.load(f)

best_idx = ckpt["global_best_indices"]
TGT19 = TARGET_H[19]
rH, rS, det_S = test_packet(best_idx, TGT19, "Phase-4 best")
det_str = f"{det_S:.3e}" if not (det_S != det_S) else "N/A"
print(f"\n=== Phase-4 best packet (R=19) ===")
print(f"  rank(H)          = {rH}  (target {TGT19})")
print(f"  rank(U @ Sigma)  = {rS}  (need 9 for Gate 3)")
print(f"  |det(U @ Sigma)| = {det_str}")
if rS is None:
    print("  [Gate 1 not satisfied — can't test]")
elif rS < 9:
    print("  ✗  Gate 3 BLOCKED in this K")
else:
    print("  ✓  Gate 3 OPEN in this K")

# ── Test 2: sweep packets for each rank ──────────────────────────────────────
def compute_rd_rs(idx, tgt):
    """Returns (rH, r_delta_quotient, r_sigma_quotient) using packet's own U."""
    idx = np.array(idx)
    R_size = len(idx)
    H     = H_db[idx].astype(float)
    Sigma = sigma_db[idx].astype(float)
    rH = rank(H)
    if rH != tgt:
        return rH, None, None
    U = col_perp(H)                     # (R_size - tgt, R_size)
    alphas = templates[alpha_db[idx]]   # (R_size, 3, 3)
    betas  = templates[beta_db[idx]]    # (R_size, 3, 3)
    pairs  = [(s,t) for s in range(3) for t in range(3) if s != t]
    Delta  = np.zeros((R_size, 54), dtype=float)
    for ci, (s, t) in enumerate(pairs):
        a_col = alphas[:, :, s]
        b_row = betas[:, t, :]
        Delta[:, ci*9:(ci+1)*9] = (a_col[:, :, None] * b_row[:, None, :]).reshape(R_size, 9)
    rD = rank(U @ Delta)
    rS = rank(U @ Sigma)
    return rH, rD, rS

for R, TGT in sorted(TARGET_H.items()):
    packet_dir = Path(f"./CANON_DATABASE/packets/R{R}")
    if not packet_dir.exists():
        print(f"\n=== R={R}: packet dir not found, skipping ===")
        continue
    files = sorted(packet_dir.glob("*.npz"))
    null_dim = R - TGT  # dimension of K^perp
    print(f"\n=== R={R}  target rank(H)={TGT}  null_dim={null_dim} ===")
    print(f"  Found {len(files)} packet files")

    n_tested = 0
    rd_rs_hist = {}   # (r_delta, r_sigma) → count

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
            if draw == 0:
                idx = np.concatenate([basis, hits[:need]])
            else:
                extra = hits[RNG.integers(0, len(hits), size=need)]
                idx = np.concatenate([basis, extra])

            rH, rD, rS = compute_rd_rs(idx, TGT)
            if rH != TGT:
                continue
            n_tested += 1

            key = (rD, rS)
            rd_rs_hist[key] = rd_rs_hist.get(key, 0) + 1
            if key == (0, 9):
                print(f"  *** (0,9) HIT — all three gates *** file={f.name}")

    print(f"  Gate-1 packets tested : {n_tested}")
    if n_tested:
        print(f"  (r_Delta, r_Sigma) histogram  [target = (0,9)]:")
        for key in sorted(rd_rs_hist):
            flag = " <-- TARGET" if key == (0,9) else ""
            print(f"    {key}: {rd_rs_hist[key]}{flag}")

        hits_rd0 = {k: v for k, v in rd_rs_hist.items() if k[0] == 0}
        hits_rs9 = {k: v for k, v in rd_rs_hist.items() if k[1] == 9}
        best_rs_at_rd0 = max((k[1] for k in hits_rd0), default=None)
        best_rd_at_rs9 = min((k[0] for k in hits_rs9), default=None)

        print(f"  r_Delta=0 packets : {sum(hits_rd0.values())}  best r_Sigma there: {best_rs_at_rd0}")
        print(f"  r_Sigma=9 packets : {sum(hits_rs9.values())}  best r_Delta there: {best_rd_at_rs9}")
        if (0, 9) in rd_rs_hist:
            print("  FEASIBLE REGION FOUND — all three gates simultaneously satisfied")
        else:
            print("  No (0,9) found — Gates 2+3 not simultaneously achieved")

