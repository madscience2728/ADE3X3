import numpy as np

def build_T_matmul() -> np.ndarray:
    T = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                T[3*r + s, 3*s + u, 3*r + u] = 1.0
    return T

# Classify the 27 nonzero entries into 4 G-orbits based on how many
# indices in (r,s,u) are equal to each other:
#   O0: r==s==u  (diagonal, 3 entries... wait, let me use the actual orbit definition)
# The prompt defines orbits by number of zeros in (r,s,u) treated as a triple:
#   O0: all three equal (r==s==u) => unique representative, but size 3? 
# Re-reading: "27 nonzero entries split into 4 G-orbits:
#   O0: 1 term  (0,0,0)
#   O1: 6 terms (exactly two zeros in triple)
#   O2: 12 terms (exactly one zero in triple)
#   O3: 8 terms  (no zeros in triple)"
# "two zeros" means two of (r,s,u) = 0; "no zeros" means r,s,u all in {1,2}
# But sizes: two zeros -> C(3,1)*2^1... choose which is nonzero (3 choices) * nonzero val (2) = 6. Yes.
# one zero -> C(3,1)*2^2 = 3*4 = 12. Yes.
# no zeros -> 2^3 = 8. Yes.
# zero zeros in "all zero" sense + exactly r==s==u==0 is 1. Yes, total = 1+6+12+8 = 27. 

def _classify_orbits():
    O0, O1, O2, O3 = [], [], [], []
    for r in range(3):
        for s in range(3):
            for u in range(3):
                zeros = (r == 0) + (s == 0) + (u == 0)
                if zeros == 3:
                    O0.append((r, s, u))
                elif zeros == 2:
                    O1.append((r, s, u))
                elif zeros == 1:
                    O2.append((r, s, u))
                else:
                    O3.append((r, s, u))
    return {'O0': O0, 'O1': O1, 'O2': O2, 'O3': O3}

G_ORBITS = _classify_orbits()

def build_G_invariant_E(w0: float, w1: float, w2: float, w3: float) -> np.ndarray:
    E = np.zeros((9, 9, 9), dtype=np.float64)
    weights = {'O0': w0, 'O1': w1, 'O2': w2, 'O3': w3}
    for key, triples in G_ORBITS.items():
        w = weights[key]
        for (r, s, u) in triples:
            E[3*r + s, 3*s + u, 3*r + u] = w
    return E

def perturbed_target(eps: float, w0: float, w1: float, w2: float, w3: float) -> np.ndarray:
    T = build_T_matmul()
    E = build_G_invariant_E(w0, w1, w2, w3)
    return T - eps * E

def frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B))
