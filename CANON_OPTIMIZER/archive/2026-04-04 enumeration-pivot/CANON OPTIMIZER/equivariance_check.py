"""
FULL EQUIVARIANT PARAMETERIZATION.

The constraint is: for g in the group, if g maps term t to term t',
then factors(t') = g · factors(t).

This means:
1. Pick ONE representative t0 per orbit
2. Choose factors at t0 FREELY (subject to stabilizer constraint below)
3. Transport to all orbit members via chosen transporters
4. Stabilizer constraint: for h in Stab(t0), h·factors(t0) = factors(t0)
   ... WRONG! That's the seed-transport model.

Actually, the constraint is weaker. The group permutes the R terms.
The TENSOR T = Σ_k α_k ⊗ β_k ⊗ γ_k must satisfy g·T = T.
This does NOT require g to map each individual term to another term.
It requires the SUM to be invariant.

Let me reconsider. The standard algorithm has R=27 terms where
each term is e_r ⊗ e_s ⊗ e_u (ignoring the factor structure for
a moment). The group permutes the triples (r,s,u). If we label
the terms by triples, then the group maps term (r,s,u) to term
(r',s',u'). So the terms ARE permuted.

But the factors α, β, γ at each term need to be such that:
  g · (α_t, β_t, γ_t) = (α_{g·t}, β_{g·t}, γ_{g·t})

This IS the seed-transport model. So why doesn't the standard
algorithm fit?

Let me re-check: for the standard algorithm, does the group
action on factors map term t's factors to term g·t's factors?
"""

import numpy as np
from itertools import permutations, product as iproduct
import sys

def flush(*a, **kw): print(*a, **kw); sys.stdout.flush()

def _swap12(x): return x if x == 0 else 3 - x
_S3 = list(permutations(range(3)))
_GROUP = [(pi, eps) for pi in _S3 for eps in iproduct([False, True], repeat=3)]

def _apply_triple(pi, eps, t):
    x = [t[pi[i]] for i in range(3)]
    return tuple(_swap12(x[i]) if eps[i] else x[i] for i in range(3))

def _apply_factors_fixed(pi, eps, a, b, g):
    def _perm_mat(e): return np.eye(3)[[0,2,1],:] if e else np.eye(3)
    P = [_perm_mat(eps[i]) for i in range(3)]
    pair_to_factor = {(0,1): a, (1,2): b, (0,2): g}
    new_factors = {}
    for (i,j), name in [((0,1),'a'), ((1,2),'b'), ((0,2),'g')]:
        oi, oj = pi[i], pi[j]
        key = (min(oi,oj), max(oi,oj))
        F = pair_to_factor[key]
        F_oriented = F.copy() if oi <= oj else F.T
        new_factors[name] = P[i] @ F_oriented @ P[j].T
    return new_factors['a'], new_factors['b'], new_factors['g']

def e(i): v = np.zeros(3); v[i] = 1; return v

flush("="*90)
flush("VERIFICATION: Does the standard algorithm satisfy g·factors(t) = factors(g·t)?")
flush("="*90)

# For every group element g and every term t in the standard algorithm,
# check whether applying g to t's factors gives (g·t)'s factors.

all_triples = [(r,s,u) for r in range(3) for s in range(3) for u in range(3)]
std_factors = {}
for t in all_triples:
    r,s,u = t
    std_factors[t] = (np.outer(e(r),e(s)), np.outer(e(s),e(u)), np.outer(e(r),e(u)))

n_checked = 0
n_fail = 0

for pi, eps in _GROUP:
    for t in all_triples:
        a_t, b_t, g_t = std_factors[t]
        t_new = _apply_triple(pi, eps, t)
        
        # Apply g to t's factors
        a_new, b_new, g_new = _apply_factors_fixed(pi, eps, a_t, b_t, g_t)
        
        # Expected: factors of t_new
        a_exp, b_exp, g_exp = std_factors[t_new]
        
        err = max(np.linalg.norm(a_new - a_exp),
                  np.linalg.norm(b_new - b_exp),
                  np.linalg.norm(g_new - g_exp))
        
        n_checked += 1
        if err > 1e-10:
            n_fail += 1
            if n_fail <= 5:
                flush(f"  FAIL: g=({list(pi)},{[int(x) for x in eps]}) on {t}→{t_new}  err={err:.2e}")
                flush(f"    α_got: {a_new.ravel().tolist()}")
                flush(f"    α_exp: {a_exp.ravel().tolist()}")

flush(f"\nChecked {n_checked} (group × terms) pairs: {n_fail} failures")

if n_fail == 0:
    flush("✓ Standard algorithm IS equivariant under the group action!")
    flush("  → Bug must be in how we build M_a, M_b from seed transport.")
else:
    flush(f"✗ {n_fail} equivariance violations found.")
    flush("  → The group action on factors is WRONG, or the standard algorithm")
    flush("    is NOT equivariant under Z₂≀S₃.")
    
    # Let's check: maybe the GROUP itself is wrong?
    # Verify the group acts on triples correctly by checking it's a group action
    flush("\n  Verifying group action on triples is well-defined...")
    for g1 in _GROUP[:5]:
        for g2 in _GROUP[:5]:
            # g1 ∘ g2 should equal their composition
            pi1, eps1 = g1
            pi2, eps2 = g2
            for t in [(0,0,1), (0,1,1), (1,1,1)]:
                t12 = _apply_triple(pi1, eps1, _apply_triple(pi2, eps2, t))
                # Composition: first apply g2, then g1
                # pi_comp[i] = pi1[pi2[i]], eps_comp[i] = eps1[i] XOR eps2[pi1[i]]... 
                # Too complex. Just check associativity numerically.
                pass
    
    # More useful: check which specific terms/group elements fail
    flush("\n  Failure pattern analysis:")
    fail_by_pi = {}
    for pi, eps in _GROUP:
        for t in all_triples:
            a_t, b_t, g_t = std_factors[t]
            t_new = _apply_triple(pi, eps, t)
            a_new, b_new, g_new = _apply_factors_fixed(pi, eps, a_t, b_t, g_t)
            a_exp, b_exp, g_exp = std_factors[t_new]
            err = max(np.linalg.norm(a_new-a_exp), np.linalg.norm(b_new-b_exp), np.linalg.norm(g_new-g_exp))
            if err > 1e-10:
                key = tuple(pi)
                fail_by_pi[key] = fail_by_pi.get(key, 0) + 1
    
    flush(f"  Failures by π: {dict(sorted(fail_by_pi.items()))}")
    flush(f"  Note: identity π=(0,1,2) failures: {fail_by_pi.get((0,1,2), 0)}")
    flush(f"  Non-identity π failures: {sum(v for k,v in fail_by_pi.items() if k != (0,1,2))}")
