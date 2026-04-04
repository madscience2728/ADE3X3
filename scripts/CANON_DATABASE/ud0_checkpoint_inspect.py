import json, numpy as np, sys
sys.path.insert(0, 'CANON_DATABASE')
from ud0_optimizer import _DB, col_perp_basis, fast_rank, score_packet

with open('CANON_DATABASE/ud0_checkpoint.json') as f:
    ckpt = json.load(f)
print('Current checkpoint:')
print(f"  score      = {ckpt['score']}")
print(f"  rk_ud0     = {ckpt['rk_ud0']}")
print(f"  elapsed    = {ckpt['elapsed_seconds']:.1f}s")

idx = np.array(ckpt['indices'], dtype=np.int64)
db = _DB()
H, sigma, delta, D0 = db.get_blocks(idx)
U = col_perp_basis(H)
UD0 = U @ D0
print(f"\n  rank(H)        = {fast_rank(H)}  (Gate 1 target: 10)")
N = np.hstack([H, delta])
print(f"  rank(Nuis)     = {fast_rank(N)}  (Gate 2: want = rank(H))")
print(f"  rank(UD0)      = {fast_rank(UD0)}  (Gate 3: want 9)")
print(f"  ||UD0||_F      = {np.linalg.norm(UD0):.4f}")
print(f"  aug_gap_H      = {fast_rank(np.hstack([H, sigma])) - fast_rank(H)}")
print(f"\n  Singular values of UD0:")
sv = np.linalg.svd(UD0, compute_uv=False)
for i,s in enumerate(sv):
    bar = '#' * int(s * 20)
    print(f"    sv[{i}] = {s:.4f}  {bar}")
