import json, math, numpy as np, glob
from pathlib import Path

DEAD_PAIRS = [(s,t) for s in range(3) for t in range(3) if s != t]

def compute_coords(alpha_flat, beta_flat):
    R = alpha_flat.shape[0]
    A = alpha_flat.reshape(R,3,3)
    B = beta_flat.reshape(R,3,3)
    Sigma = np.zeros((R,9)); Eta1 = np.zeros((R,9)); Eta2 = np.zeros((R,9))
    for k in range(R):
        for r in range(3):
            for u in range(3):
                idx = r*3+u
                s0 = A[k,r,0]*B[k,0,u]; s1 = A[k,r,1]*B[k,1,u]; s2 = A[k,r,2]*B[k,2,u]
                Sigma[k,idx] = s0+s1+s2; Eta1[k,idx] = s0-s1; Eta2[k,idx] = s1-s2
    H = np.hstack([Eta1, Eta2])
    Delta = np.zeros((R,54))
    for k in range(R):
        col = 0
        for s,t in DEAD_PAIRS:
            for r in range(3):
                for u in range(3):
                    Delta[k,col] = A[k,r,s]*B[k,t,u]; col += 1
    return Sigma, H, Delta

def stable_rank(M, tol=1e-9):
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[0] < 1e-15: return 0
    return int(np.sum(sv > tol*sv[0]))

def solve_gamma(Sigma):
    GT, _, _, _ = np.linalg.lstsq(Sigma.T, 3*np.eye(9), rcond=None)
    return GT.T  # (9,R)

def analyse(name, R, alpha_flat, beta_flat, gamma_flat=None):
    Sigma, H, Delta = compute_coords(alpha_flat, beta_flat)
    rH = stable_rank(H)
    rN = stable_rank(np.hstack([H, Delta]))

    if gamma_flat is not None:
        Gamma = gamma_flat.T  # (9,R)
    else:
        Gamma = solve_gamma(Sigma)

    lk = float(np.linalg.norm(Gamma @ Delta))
    lk_H = float(np.linalg.norm(Gamma @ H))

    budgets = [('bfloat16',7), ('float16',10), ('float32',23)]
    budget_vals = {lbl: 3*R*(2**-b) for lbl,b in budgets}

    print(f'\n=== {name} | R={R} ===')
    print(f'  rank(H)             = {rH}   (target: {R-9} for exact, 18 for generic)')
    print(f'  rank(Nuisance)      = {rN}')
    print(f'  ||Gamma @ Delta||_F = {lk:.6e}   (Delta leakage)')
    print(f'  ||Gamma @ H||_F     = {lk_H:.6e}   (H leakage, ~0 if exact)')
    for lbl,b in budgets:
        bf = budget_vals[lbl]
        closed = lk <= bf
        print(f'  {lbl:10s}  budget={bf:.4e}  lk/budget={lk/bf:.4f}  closed={closed}')
    return {'name':name,'R':R,'rank_H':rH,'rank_Nuisance':rN,'leakage':lk,
            'leakage_H':lk_H,
            'budget_bfloat16':budget_vals['bfloat16'],
            'budget_float16':budget_vals['float16'],
            'budget_float32':budget_vals['float32']}

def load_sparse(terms):
    R = len(terms)
    a = np.zeros((R,9)); b = np.zeros((R,9)); g = np.zeros((R,9))
    for k,t in enumerate(terms):
        for i,v in zip(t['alpha_support'], t['alpha_values']): a[k,i] = v
        for i,v in zip(t['beta_support'],  t['beta_values']):  b[k,i] = v
        for i,v in zip(t['gamma_support'], t['gamma_values']): g[k,i] = v
    return a, b, g

results = []

# --- slp_best_at_0.074 ---
with open('tests/slp_best_at_0.074.json') as f: d = json.load(f)
a = np.array(d['alpha']); b = np.array(d['beta']); g = np.array(d['gamma'])
results.append(analyse('slp_best_at_0.074', d['rank'], a, b, g))

# --- optimized_als_r10_at_0.09 ---
with open('tests/optimized_als_r10_at_0.09.json') as f: d2 = json.load(f)
a2,b2,g2 = load_sparse(d2['terms'])
results.append(analyse('optimized_als_r10_at_0.09', len(d2['terms']), a2, b2, g2))

# --- optimized_candidate_r7_at_0.4586 ---
with open('tests/optimized_candidate_r7_at_0.4586.json') as f: d3 = json.load(f)
a3,b3,g3 = load_sparse(d3['terms'])
results.append(analyse('optimized_candidate_r7_at_0.4586', len(d3['terms']), a3, b3, g3))

# --- phase4 gradient search best decompositions ---
phase4_files = sorted(glob.glob('outputs/ade3x3_attack/phase4_gradient_search/best_decompositions/*.json'))
for fpath in phase4_files:
    with open(fpath) as f: dp = json.load(f)
    if 'terms' not in dp:
        continue
    terms = dp['terms']
    if not terms:
        continue
    t0 = terms[0]
    R4 = len(terms)
    if 'alpha' in t0:  # dense: term has 'alpha', 'beta', 'gamma' as lists
        ap = np.array([t['alpha'] for t in terms])
        bp = np.array([t['beta']  for t in terms])
        gp = np.array([t['gamma'] for t in terms])
        results.append(analyse(Path(fpath).stem, R4, ap, bp, gp))
    elif 'alpha_support' in t0:
        at,bt,gt = load_sparse(terms)
        results.append(analyse(Path(fpath).stem, R4, at, bt, gt))

# --- Summary ---
print('\n\n' + '='*90)
print('SUMMARY — sorted by leakage ascending')
print('='*90)
hdr = f"{'name':45}  {'R':>3}  {'rank_H':>6}  {'leakage':>12}  {'bf16 closed':>11}  {'f16 closed':>10}"
print(hdr)
print('-'*len(hdr))
for r in sorted(results, key=lambda x: x['leakage']):
    bf16 = r['leakage'] <= r['budget_bfloat16']
    f16  = r['leakage'] <= r['budget_float16']
    print(f"{r['name']:45}  {r['R']:>3}  {r['rank_H']:>6}  {r['leakage']:>12.4e}  {str(bf16):>11}  {str(f16):>10}")
