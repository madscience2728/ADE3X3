"""Verify ChatGPT's hand-written candidate JSON."""
import json, numpy as np

with open('chat_gpt.json') as f:
    data = json.load(f)

terms = data['terms']
print(f"Term count: {len(terms)}\n")

# Check for length mismatches
total_vars = 0
for i, t in enumerate(terms):
    al = len(t['alpha_support']); av = len(t['alpha_values'])
    bl = len(t['beta_support']);  bv = len(t['beta_values'])
    gl = len(t['gamma_support']); gv = len(t['gamma_values'])
    total_vars += av + bv + gv
    ok = (al==av) and (bl==bv) and (gl==gv)
    tag = "OK" if ok else "MISMATCH"
    print(f"  term {i+1:2d}: alpha({al},{av}) beta({bl},{bv}) gamma({gl},{gv})  {tag}")
    if not ok:
        print(f"    *** alpha off by {av-al}, beta off by {bv-bl}, gamma off by {gv-gl}")

print(f"\nTotal variables: {total_vars}")

# Check support indices valid 0-8
for i, t in enumerate(terms):
    for key in ['alpha_support','beta_support','gamma_support']:
        for idx in t[key]:
            if idx < 0 or idx > 8:
                print(f"  BAD INDEX term {i+1} {key}: {idx}")

# Support signatures
sigs = [(len(t['alpha_support']), len(t['beta_support']), len(t['gamma_support'])) for t in terms]
print(f"\nSupport signatures: {sigs}")

# Reconstruct
T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

candidate = np.zeros((9,9,9))
for term in terms:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i,v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i,v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i,v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate += np.einsum('c,a,b->cab', g, a, b)

R = candidate - T
max_abs = np.max(np.abs(R))
fro = np.linalg.norm(R)
T_hat_fro2 = np.sum(candidate**2)
R_fro2 = np.sum(R**2)
inner = np.sum(R * candidate)

print(f"\nmax_abs = {max_abs:.6f}")
print(f"fro     = {fro:.6f}")
print(f"||R||^2 + ||That||^2 = {R_fro2 + T_hat_fro2:.2f}  (should be ~27)")
print(f"<R, That> = {inner:.4f}  (should be ~0 at ALS minima)")

# Compare supports to baseline
with open('outputs/exports/step84_batches/run_20260331_111612_906/copy_008/step84_best_individual.json') as f:
    baseline = json.load(f)

print("\n=== SUPPORT COMPARISON vs baseline ===")
support_match = True
for i, (bt, ct) in enumerate(zip(baseline['terms'], terms)):
    for key in ['alpha_support','beta_support','gamma_support']:
        if bt[key] != ct[key]:
            print(f"  DIFFERENT term {i+1} {key}: baseline={bt[key]} chatgpt={ct[key]}")
            support_match = False
if support_match:
    print("  All supports MATCH baseline")
else:
    print("  Supports DIFFER from baseline")

# Worst entries
flat = R.ravel()
worst_idx = np.argsort(np.abs(flat))[-15:][::-1]
print("\n=== 15 WORST RESIDUAL ENTRIES ===")
for idx in worst_idx:
    c,a,b = np.unravel_index(idx, (9,9,9))
    live = T[c,a,b] == 1.0
    tag = "LIVE" if live else "dead"
    print(f"  R[{c},{a},{b}] = {R[c,a,b]:+.6f}  ({tag})")

# ChatGPT claimed max_abs = 0.2336 — check
print(f"\n=== VERDICT ===")
print(f"ChatGPT claimed max_abs = 0.2336")
print(f"Actual max_abs  = {max_abs:.6f}")
if max_abs < 0.5:
    print("IMPROVEMENT over baseline (0.500)")
else:
    print("NO IMPROVEMENT over baseline (0.500)")
