"""
delta_census.py
===============
Census of  Delta = C_+^2 - 4*D*C_-  for 3x3 real matrices sampled from
several populations, plus a two-branch recovery log for the Stratum-3
formula derived in channel_stratification.py.

Populations
-----------
  sq_generic      A,B  in R^{3x3}, iid Gaussian
  sq_commuting    A,B  diagonal (Delta = 0 always)
  sq_symmetric    A=A^T, B=B^T, Gaussian symmetric
  sq_rank1_A      rank(A)=1, B generic
  sq_rank1_AB     rank(A)=rank(B)=1
  sq_near_comm    B = A + 0.1 * rand  (small perturbation off commuting)
  generic_triple  D, Cp, Cm drawn independently  (no square-stratum structure)

Recovery strategies
-------------------
  diag_entry  --  formula applied to each diagonal entry of P0, P1
                  independently.  Exact iff D,Cp,Cm are diagonal.
  eig_based   --  jointly diagonalise P0 (= D+Cp+Cm) when D,Cp,Cm
                  commute pairwise.  Exact for all commuting pairs.

Output
------
  stdout          --  rank table, fingerprint table, success table
  delta_census_successes.json  --  up to MAX_SAVED successful cases
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from numpy.linalg import norm

try:
    from scipy.linalg import sqrtm as mat_sqrtm
    _HAVE_SQRTM = True
except ImportError:
    _HAVE_SQRTM = False

# ─── constants ────────────────────────────────────────────────────────────────
N         = 3
K_PER_POP = 2000
MAX_SAVED = 500
ATOL_RNK  = 1e-9   # singular-value floor for rank
ATOL_SYM  = 1e-8   # ||M - M^T|| / ||M|| threshold
ATOL_COM  = 1e-8   # relative commutator norm threshold
ATOL_SUC  = 1e-8   # recovery error threshold for "success"
SPARSE_FRAC = 0.5  # fraction of near-zero entries to call sparse

RNG = np.random.default_rng(0)

# DFT-3 roots
omega  = np.exp(2j * np.pi / 3)        # primitive cube root of unity
omega2 = np.conj(omega)                 # omega^2
alpha  = 1.0 - omega2                   # 1 - omega^2  (~1.5 + 0.866j)
beta   = 1.0 - omega                    # 1 - omega    (~1.5 - 0.866j)


# ─── linear algebra helpers ───────────────────────────────────────────────────

def sv_rank(M, atol=ATOL_RNK):
    sv = np.linalg.svd(M, compute_uv=False)
    ref = sv[0] if sv[0] > atol else 1.0
    return int(np.sum(sv > atol * ref))


def is_sym(M, atol=ATOL_SYM):
    return bool(norm(M - M.T) < atol * max(1.0, norm(M)))


def rel_comm(A, B):
    """Relative Frobenius norm of [A,B]."""
    return float(norm(A @ B - B @ A) / max(1.0, norm(A), norm(B)))


def approx_commutes(A, B, atol=ATOL_COM):
    return rel_comm(A, B) < atol


def is_sparse(M, frac=SPARSE_FRAC):
    return float(np.mean(np.abs(M) < 1e-10)) > frac


def is_diagonal(M, atol=1e-10):
    """True iff all off-diagonal entries are negligible."""
    return bool(norm(M - np.diag(np.diag(M))) < atol * max(1.0, norm(M)))


def has_repeated_rows_or_cols(M, atol=1e-8):
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if norm(M[i] - M[j]) < atol or norm(M[:, i] - M[:, j]) < atol:
                return True
    return False


def fingerprint_tags(X, Y, D, Cp, Cm):
    """
    Return a sorted tuple of string tags describing the structural class
    of the sample.  X,Y are the generating matrices (A,B on the square
    stratum; stand-ins for generic_triple).
    """
    tags = []
    if is_diagonal(X):               tags.append("X_diag")
    elif is_sym(X):                  tags.append("X_sym")
    if is_diagonal(Y):               tags.append("Y_diag")
    elif is_sym(Y):                  tags.append("Y_sym")
    rX = sv_rank(X);  rY = sv_rank(Y)
    if rX < N:                       tags.append("X_lowrank")
    if rY < N:                       tags.append("Y_lowrank")
    if rX == 1:                      tags.append("X_rank1")
    if rY == 1:                      tags.append("Y_rank1")
    if is_sparse(X) and not is_diagonal(X): tags.append("X_sparse")
    if is_sparse(Y) and not is_diagonal(Y): tags.append("Y_sparse")
    if approx_commutes(X, Y):        tags.append("XY_comm")
    if has_repeated_rows_or_cols(X): tags.append("X_rep")
    if has_repeated_rows_or_cols(Y): tags.append("Y_rep")
    if (approx_commutes(D, Cp) and
            approx_commutes(D, Cm) and
            approx_commutes(Cp, Cm)):
        tags.append("DCpCm_comm")
    return tuple(sorted(tags))


# ─── channel algebra ──────────────────────────────────────────────────────────

def square_channels(A, B):
    return A @ A, A @ B + B @ A, B @ B


def delta_mat(D, Cp, Cm):
    return Cp @ Cp - 4.0 * D @ Cm


def pack_obs(D, Cp, Cm):
    P0 = D + Cp + Cm
    P1 = D + omega2 * Cp + omega  * Cm
    return P0, P1


# ─── Stratum-3 scalar quadratic ───────────────────────────────────────────────

def _stratum3_scalar(p0_vec, p1_vec):
    """
    Apply   3β²·c_-² + 2α²·S·c_- - T² = 0   element-wise.
    Inputs: 1-D complex arrays of scalar observations.
    Returns (D_plus, D_minus) as 1-D arrays.
    """
    S    = p0_vec + p1_vec
    T    = p0_vec - p1_vec
    a2   = 3.0 * beta ** 2
    bv   = 2.0 * alpha ** 2 * S
    cv   = -(T * T)
    disc = np.sqrt(bv * bv - 4.0 * a2 * cv)

    results = []
    for sign in (+1, -1):
        Cm_c = (-bv + sign * disc) / (2.0 * a2)
        Cp_c = (T - beta * Cm_c) / alpha
        D_c  = (S + omega * Cp_c + omega2 * Cm_c) / 2.0
        results.append(D_c)
    return results[0], results[1]   # D_plus, D_minus


# ─── Recovery strategy 1: diagonal-entry ─────────────────────────────────────

def recover_diag_entry(P0, P1, D_true):
    """
    Apply the Stratum-3 scalar formula to the diagonal entries of P0, P1
    independently.  Reconstructs D as a diagonal matrix.
    Exact when D,Cp,Cm are diagonal; approximate (usually wrong) otherwise.

    Returns (err_plus, err_minus, D_rec_best, all_real).
    """
    p0d = np.diag(P0)
    p1d = np.diag(P1)
    Dp_vec, Dm_vec = _stratum3_scalar(p0d, p1d)
    D_rec_p = np.diag(Dp_vec.real)
    D_rec_m = np.diag(Dm_vec.real)
    ep = float(norm(D_rec_p - D_true))
    em = float(norm(D_rec_m - D_true))
    # all_real: imaginary residual is small
    im_p = float(np.max(np.abs(Dp_vec.imag)))
    im_m = float(np.max(np.abs(Dm_vec.imag)))
    all_real = max(im_p, im_m) < 1e-6
    best = D_rec_p if ep <= em else D_rec_m
    return ep, em, best, all_real


# ─── Recovery strategy 2: eigenvalue-based ───────────────────────────────────

def recover_eig_based(P0, P1, D_true, Cp_true, Cm_true):
    """
    Jointly diagonalise P0 when D,Cp,Cm commute pairwise.
    Returns (err_plus, err_minus, D_rec_best, A_rec, B_rec) or None.
    A_rec, B_rec computed via sqrtm when scipy is available.
    """
    if not (approx_commutes(D_true, Cp_true) and
            approx_commutes(D_true, Cm_true) and
            approx_commutes(Cp_true, Cm_true)):
        return None

    try:
        # Eigenvectors of P0 serve as shared basis (P0 commutes with P1 here)
        _, V = np.linalg.eig(P0)
        if abs(np.linalg.det(V)) < 1e-12:
            return None
        Vinv = np.linalg.inv(V)
        p0d = np.diag(Vinv @ P0 @ V)
        p1d = np.diag(Vinv @ P1 @ V)

        Dp_vec, Dm_vec = _stratum3_scalar(p0d, p1d)
        D_rec_p = (V @ np.diag(Dp_vec) @ Vinv).real
        D_rec_m = (V @ np.diag(Dm_vec) @ Vinv).real
        ep = float(norm(D_rec_p - D_true))
        em = float(norm(D_rec_m - D_true))
        best_D = D_rec_p if ep <= em else D_rec_m

        # Optionally recover A = sqrtm(D_rec), B = sqrtm(Cm_rec)
        A_rec = B_rec = None
        if _HAVE_SQRTM and ep < ATOL_SUC:
            # Recover Cm from the winning Cm_vec
            best_Cm_vec = np.diag(Vinv @ Cp_true @ V if ep <= em else Vinv @ Cm_true @ V)
            # Simpler: reconstruct Cm from the formula
            bv = 2.0 * alpha ** 2 * (p0d + p1d)
            cv = -((p0d - p1d) ** 2)
            a2 = 3.0 * beta ** 2
            disc = np.sqrt(bv * bv - 4.0 * a2 * cv)
            Cm_vec = ((-bv + disc) if ep <= em else (-bv - disc)) / (2.0 * a2)
            Cm_rec = (V @ np.diag(Cm_vec) @ Vinv).real
            try:
                A_rec = mat_sqrtm(best_D).real
                B_rec = mat_sqrtm(Cm_rec).real
            except Exception:
                pass

        return ep, em, best_D, A_rec, B_rec

    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return None


# ─── census state ─────────────────────────────────────────────────────────────

rank_counts  = defaultdict(Counter)   # pop -> Counter[int delta_rank]
fp_counts    = defaultdict(Counter)   # pop -> Counter[fingerprint tuple]
suc_diag     = Counter()              # diagonal-entry success per pop
suc_eig      = Counter()              # eig-based success per pop
total_cnt    = Counter()              # total samples per pop
saved        = []                     # list of dicts for successful cases
fp_rank_joint = Counter()             # (fingerprint_tuple, delta_rank) -> count


def _to_real_list(M):
    return [[float(v.real) for v in row] for row in M]


def process(pop, X, Y, D, Cp, Cm, has_XY=True):
    """Record one sample.  X,Y are A,B on square stratum, or stand-ins."""
    Delta  = delta_mat(D, Cp, Cm)
    dr     = sv_rank(Delta)
    fp     = fingerprint_tags(X, Y, D, Cp, Cm)
    P0, P1 = pack_obs(D, Cp, Cm)

    # ── diagonal-entry recovery ──────────────────────────────────────────────
    ep_de, em_de, D_de, all_real = recover_diag_entry(P0, P1, D)
    ok_de = min(ep_de, em_de) < ATOL_SUC

    # ── eigenvalue-based recovery ────────────────────────────────────────────
    eig_res = recover_eig_based(P0, P1, D, Cp, Cm)
    ok_eig  = False
    ep_eig  = em_eig = None
    A_rec   = B_rec  = None
    if eig_res is not None:
        ep_eig, em_eig, D_eig_best, A_rec, B_rec = eig_res
        ok_eig = min(ep_eig, em_eig) < ATOL_SUC

    # ── tally ────────────────────────────────────────────────────────────────
    rank_counts[pop][dr] += 1
    fp_counts[pop][fp]   += 1
    total_cnt[pop]        += 1
    fp_rank_joint[(fp, dr)] += 1
    if ok_de:  suc_diag[pop] += 1
    if ok_eig: suc_eig[pop]  += 1

    # ── save ─────────────────────────────────────────────────────────────────
    if (ok_de or ok_eig) and len(saved) < MAX_SAVED:
        rec = {
            "population":   pop,
            "fingerprint":  list(fp),
            "delta_rank":   dr,
            # structural properties
            "rank_X":       int(sv_rank(X)),
            "rank_Y":       int(sv_rank(Y)),
            "X_sym":        bool(is_sym(X)),
            "Y_sym":        bool(is_sym(Y)),
            "XY_comm":      bool(approx_commutes(X, Y)),
            "XY_rel_comm":  float(rel_comm(X, Y)),
            "DCpCm_comm":   bool("DCpCm_comm" in fp),
            # recovery errors
            "diag_entry": {
                "error_plus":  ep_de,
                "error_minus": em_de,
                "success":     ok_de,
                "all_real":    all_real,
            },
            "eig_based": {
                "error_plus":  float(ep_eig) if ep_eig is not None else None,
                "error_minus": float(em_eig) if em_eig is not None else None,
                "success":     ok_eig,
            },
            # matrices
            "X":     _to_real_list(X),
            "Y":     _to_real_list(Y),
            "D":     _to_real_list(D),
            "Cp":    _to_real_list(Cp),
            "Cm":    _to_real_list(Cm),
            "Delta": _to_real_list(Delta),
        }
        if A_rec is not None:
            rec["A_rec"] = _to_real_list(A_rec)
        if B_rec is not None:
            rec["B_rec"] = _to_real_list(B_rec)
        saved.append(rec)


# ─── sampling helpers ─────────────────────────────────────────────────────────

def rand_mat():    return RNG.standard_normal((N, N))
def rand_sym():    M = RNG.standard_normal((N, N)); return (M + M.T) / 2.0
def rand_diag():   return np.diag(RNG.standard_normal(N))
def rand_rank1():  return np.outer(RNG.standard_normal(N), RNG.standard_normal(N))


# ─── main loop ────────────────────────────────────────────────────────────────

POPS_ORDER = [
    "sq_generic",
    "sq_commuting",
    "sq_symmetric",
    "sq_rank1_A",
    "sq_rank1_AB",
    "sq_near_comm",
    "generic_triple",
]

print("Running delta census …")
for pop in POPS_ORDER:
    n = K_PER_POP
    print(f"  {pop:<18}  ({n} samples) …", flush=True)
    for _ in range(n):
        if pop == "sq_generic":
            A, B = rand_mat(), rand_mat()
        elif pop == "sq_commuting":
            A, B = rand_diag(), rand_diag()
        elif pop == "sq_symmetric":
            A, B = rand_sym(), rand_sym()
        elif pop == "sq_rank1_A":
            A, B = rand_rank1(), rand_mat()
        elif pop == "sq_rank1_AB":
            A, B = rand_rank1(), rand_rank1()
        elif pop == "sq_near_comm":
            A = rand_mat();  B = A + 0.1 * rand_mat()
        elif pop == "generic_triple":
            D  = rand_mat(); Cp = rand_mat(); Cm = rand_mat()
            process(pop, D, Cm, D, Cp, Cm, has_XY=False)
            continue

        D, Cp, Cm = square_channels(A, B)
        process(pop, A, B, D, Cp, Cm)


# ─── printable rank table ─────────────────────────────────────────────────────

SEP  = "=" * 82
SEP2 = "-" * 82

print(f"\n{SEP}")
print("DELTA RANK CENSUS   (Delta = C_+^2 - 4*D*C_-,  N=3,  K={K} per pop)".format(
    K=K_PER_POP))
print(SEP)
print(f"{'Population':<20} {'Total':>6}  "
      f"{'rk=0':>7}  {'rk=1':>7}  {'rk=2':>7}  {'rk=3':>7}  "
      f"{'suc_diag':>10}  {'suc_eig':>9}")
print(SEP2)
for pop in POPS_ORDER:
    t   = total_cnt[pop]
    rc  = rank_counts[pop]
    pct = lambda k: f"{rc[k]:>5}({100*rc[k]/t:4.1f}%)"
    print(f"{pop:<20} {t:>6}  "
          f"{pct(0):>12}  {pct(1):>12}  {pct(2):>12}  {pct(3):>12}  "
          f"{suc_diag[pop]:>10}  {suc_eig[pop]:>9}")


# ─── fingerprint table ────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("TOP STRUCTURAL FINGERPRINTS PER POPULATION")
print(SEP)
for pop in POPS_ORDER:
    t = total_cnt[pop]
    print(f"\n  [{pop}]  (total {t})")
    for fp, cnt in fp_counts[pop].most_common(10):
        label = ", ".join(fp) if fp else "(no tags)"
        print(f"    {cnt:>5}  ({100*cnt/t:5.1f}%)  {label}")


# ─── success breakdown by fingerprint ────────────────────────────────────────

print(f"\n{SEP}")
print("SUCCESS FINGERPRINTS  (all saved cases, both recovery strategies)")
print(SEP)

fp_suc = Counter(tuple(c["fingerprint"]) for c in saved)
fp_tot = sum((fp_counts[p] for p in POPS_ORDER), Counter())

print(f"\n  {'Fingerprint':<55} {'suc':>5}  {'ttl':>7}  {'rate':>6}")
print("  " + "-" * 75)
for fp, suc in fp_suc.most_common(25):
    tot  = fp_tot.get(fp, 0)
    rate = suc / tot if tot else 0.0
    label = ", ".join(fp) if fp else "(no tags)"
    print(f"  {label:<55} {suc:>5}  {tot:>7}  {rate:>6.3f}")


# ─── per-strategy breakdown ───────────────────────────────────────────────────

print(f"\n{SEP}")
print("PER-STRATEGY BREAKDOWN  (diag_entry vs eig_based)")
print(SEP)

suc_both = suc_only_de = suc_only_eig = suc_neither = 0
for c in saved:
    de  = c["diag_entry"]["success"]
    eig = c["eig_based"]["success"]
    if de and eig:      suc_both     += 1
    elif de:            suc_only_de  += 1
    elif eig:           suc_only_eig += 1
    else:               suc_neither  += 1   # shouldn't happen (we filter on save)

total_saved = len(saved)
tot_succ    = sum(suc_diag[p] + (suc_eig[p] if p != "sq_commuting" else 0)
                  for p in POPS_ORDER)  # rough total
print(f"  Total saved cases            : {total_saved}")
print(f"  Both diag+eig succeeded      : {suc_both}")
print(f"  Only diag_entry succeeded    : {suc_only_de}")
print(f"  Only eig_based succeeded     : {suc_only_eig}")
print(f"  Saved but neither flagged    : {suc_neither}  (unexpected)")

if _HAVE_SQRTM:
    n_with_AB = sum(1 for c in saved if "A_rec" in c)
    print(f"  Cases with recovered A,B     : {n_with_AB}  (via scipy.linalg.sqrtm)")
else:
    print(f"  Recovered A,B                : not available (scipy not found)")


# ─── delta rank conditioned on structural tag  (uses full fp×rank joint) ──────

print(f"\n{SEP}")
print("DELTA RANK CONDITIONED ON STRUCTURAL TAG  (all samples)")
print(SEP)

PROPS_TO_CHECK = [
    "XY_comm", "DCpCm_comm",
    "X_diag", "Y_diag",
    "X_sym",  "Y_sym",
    "X_rank1", "Y_rank1",
    "X_lowrank", "Y_lowrank",
]

for prop in PROPS_TO_CHECK:
    rk_with    = Counter()
    rk_without = Counter()
    for (fp, rk), cnt in fp_rank_joint.items():
        if prop in fp:
            rk_with[rk]    += cnt
        else:
            rk_without[rk] += cnt
    n_with    = sum(rk_with.values())
    n_without = sum(rk_without.values())
    if n_with == 0:
        continue
    print(f"\n  tag={prop!r}  (n_with={n_with}, n_without={n_without})")
    for rk in range(N + 1):
        w  = rk_with.get(rk, 0)
        wo = rk_without.get(rk, 0)
        pw  = f"{100*w /n_with   :5.1f}%" if n_with    else "   n/a"
        pwo = f"{100*wo/n_without:5.1f}%" if n_without else "   n/a"
        print(f"    rk={rk}:  with={w:>5}({pw}),  without={wo:>5}({pwo})")


# ─── persist ─────────────────────────────────────────────────────────────────

out_path = Path("delta_census_successes.json")
with out_path.open("w") as fh:
    json.dump(saved, fh, indent=2)

print(f"\n{SEP}")
print(f"Saved {len(saved)} successful recovery cases  -->  {out_path}")
print(f"scipy.linalg.sqrtm available: {_HAVE_SQRTM}")
print("Done.")
