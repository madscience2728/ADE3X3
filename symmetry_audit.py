"""
Symmetry audit of the rank-22 approximate 3x3 matmul decomposition.
Hunting for hidden structure in alpha, beta, gamma tables.

Emmy Noether's theorem: every continuous symmetry => conserved quantity.
For tensor decompositions: symmetries of T_matmul that are preserved 
in the decomposition constrain the solution variety.
"""
import numpy as np
from itertools import permutations

# ---- Alpha table (22x9) ----
alpha = np.array([
    [-0.2284, -0.5528,  0.2374,  0.0113,  0.0870,  0.0026,  0.6499, -1.3969, -1.4142],
    [ 3.8509, -0.5374,  1.3732, -2.6627, -2.4152,  2.7859,  2.7148, -0.1642,  0.6812],
    [-4.2364,  1.0560, -2.1338,  0.5331, -0.1295,  0.2638, -2.8022,  0.6980, -1.4117],
    [-1.0657,  0.2630, -0.5377,  0.0863, -0.0233,  0.0462,  1.4683, -0.3635,  0.7395],
    [-0.5630, -0.6416,  0.7646,  0.0530,  0.0604, -0.0719,  0.3796,  0.4325, -0.5155],
    [ 0.3682,  0.4198, -0.5002, -2.6741, -3.0449,  3.6279,  0.4455,  0.5077, -0.6050],
    [-0.1234,  0.6711,  0.3693,  0.0384, -0.1619, -0.1019, -0.7649,  5.0575,  2.5130],
    [-0.6412,  4.1188,  2.0764,  0.0793, -0.4986, -0.2508, -0.4192,  2.6937,  1.3573],
    [-2.4739,  0.6651, -1.3110,  0.3012, -0.0932,  0.1759, -1.6984,  0.3689, -0.7830],
    [ 0.3623,  0.4121, -0.4909, -2.7489, -3.1299,  3.7291,  0.4476,  0.5093, -0.6068],
    [-1.5913,  0.3785, -0.7850,  0.1339, -0.0270,  0.0596,  2.1843, -0.5250,  1.0787],
    [-0.6712,  0.1633, -0.3368,  4.9892, -1.2451,  2.4861, -0.8219,  0.2024, -0.4114],
    [ 0.0960, -0.6938, -0.3301, -0.0208,  0.1797,  0.0776,  0.7876, -4.9705, -2.5285],
    [-0.6896,  4.4320,  2.2345,  0.0876, -0.5724, -0.2901, -0.4509,  2.9013,  1.4623],
    [ 0.6682, -0.1669,  0.3327, -4.9561,  1.2050, -2.4889,  0.8173, -0.2017,  0.4089],
    [ 4.0924, -0.9986,  2.0325, -0.5101,  0.1280, -0.2579,  2.7065, -0.6605,  1.3452],
    [-0.5093,  3.2327,  1.6399,  0.0565, -0.3389, -0.1781,  0.3514, -2.2098, -1.1260],
    [-0.3943, -0.4494,  0.5356,  0.2584,  0.2942, -0.3506,  0.4620,  0.5265, -0.6274],
    [ 1.0833, -3.5215, -1.4279, -4.1993,  1.4432, -1.8529,  1.0256, -2.3824, -0.7751],
    [ 0.4853, -3.1389, -1.5772, -0.0423,  0.2873,  0.1419, -0.3246,  2.1086,  1.0569],
    [-0.6859,  2.2056,  0.8908,  2.6737, -0.8569,  1.2165, -0.1956, -1.4287, -0.9933],
    [ 2.0876, -0.5220,  1.0565, -0.2043,  0.0517, -0.1042, -1.4023,  0.3514, -0.7098],
])

beta = np.array([
    [-0.6360,  0.6054,  0.2761, -0.5915,  0.5649,  0.2574,  0.9806, -0.9319, -0.4251],
    [-1.0062,  0.4766, -3.0497, -0.9470,  0.4335, -2.8437,  1.5668, -0.7123,  4.6962],
    [ 0.9059, -0.4037,  2.7011,  0.8425, -0.3891,  2.5359, -1.3906,  0.6468, -4.1936],
    [ 1.0899, -1.1162, -0.5108, -0.1651,  0.1691,  0.0773,  0.6640, -0.6800, -0.3112],
    [ 0.6583, -0.5361, -0.2003,  0.6064, -0.4940, -0.1839, -1.0004,  0.8154,  0.3050],
    [ 1.8524,  1.4931,  1.3547,  1.7318,  1.3955,  1.2680, -2.8628, -2.3070, -2.0964],
    [-1.8096,  1.7588,  0.8148, -2.4703,  2.0739,  0.8458,  2.4017, -2.4924, -1.2113],
    [-0.3880, -0.3380, -0.2384,  3.0244,  2.6362,  1.8568,  2.2524,  1.9634,  1.3826],
    [ 0.9696, -0.6107,  1.3000,  0.3796, -0.1495,  1.3858, -0.4532,  0.1077, -2.3460],
    [-0.8361, -1.9856,  1.7453, -0.7753, -1.8436,  1.6226,  1.2804,  3.0434, -2.6773],
    [-1.8144,  1.4484,  0.6543,  0.2748, -0.2194, -0.0991, -1.1053,  0.8823,  0.3986],
    [ 0.1804, -0.0946,  0.5637, -1.4364,  0.6614, -4.3202, -1.0726,  0.4853, -3.2105],
    [ 1.9362, -1.7785, -0.8135,  2.1322, -2.5410, -1.1378, -2.8218,  2.3045,  1.0669],
    [-0.1489, -0.3959,  0.3871,  1.1980,  3.1734, -3.0906,  0.8956,  2.3716, -2.3087],
    [ 0.1734, -0.0917,  0.5432, -1.4631,  0.6737, -4.4005, -1.1002,  0.4978, -3.2930],
    [ 0.8810, -0.3927,  2.6272,  0.8079, -0.3732,  2.4318, -1.3296,  0.6185, -4.0099],
    [ 0.2910, -0.3248, -0.0002, -2.4221,  2.6612,  0.0422, -1.8181,  1.9939,  0.0352],
    [ 0.2853,  0.4938,  0.2416,  0.3120,  0.4233,  0.2058, -0.5794, -0.6392, -0.3185],
    [-0.2274,  0.0895, -0.6571,  1.7329, -0.7910,  5.1996,  1.2868, -0.5979,  3.8797],
    [-0.3344,  0.2578,  0.2302,  2.4999, -1.9263, -1.7597,  1.8514, -1.4265, -1.3072],
    [-0.0798,  0.0176, -0.4194,  1.0382, -0.4735,  3.1470,  0.8126, -0.3844,  2.3322],
    [-1.7275,  1.4078,  0.5254,  0.2617, -0.2132, -0.0795, -1.0524,  0.8576,  0.3201],
])

gamma = np.array([
    [-0.5911,  0.4872,  0.2711,  0.8850, -0.7297, -0.4060, -1.0005,  0.8244,  0.4589],
    [ 3.4292,  4.3845, -0.4668,  0.8693,  1.1031, -0.1023,  2.4395,  3.1234, -0.3428],
    [-0.9436, -2.4157,  2.9454, -0.2419, -0.7158,  0.9684, -0.7027, -2.4045,  3.5296],
    [ 0.7196,  0.7954,  0.2347, -0.0852, -0.0940, -0.0279, -1.1038, -1.2203, -0.3599],
    [-0.7299,  0.6016,  0.3348, -0.1839,  0.1514,  0.0843, -0.5198,  0.4285,  0.2384],
    [ 2.8023,  3.5809, -0.3826,  0.6907,  0.8904, -0.1064,  2.0047,  2.5591, -0.2732],
    [-0.3968,  0.5742,  0.2196, -4.2235,  3.4511,  1.9327, -0.3578, -0.0883,  0.1054],
    [ 0.3931,  0.5243, -0.0389,  3.3187,  4.4759, -0.4233,  0.0622,  0.0833, -0.0025],
    [-0.2706, -0.5759,  1.7117, -0.0562,  0.1381, -0.1622, -0.1201,  1.3087, -2.3830],
    [ 2.7180,  3.4752, -0.3707,  0.6940,  0.8912, -0.1070,  1.9309,  2.4685, -0.2626],
    [ 0.4024,  0.9397, -1.1948, -0.0476, -0.1110,  0.1411, -0.6172, -1.4418,  1.8334],
    [-0.0409, -0.3831,  0.5186, -0.3849, -3.3378,  4.5241, -0.0082, -0.0650,  0.0881],
    [ 0.3163, -0.5009, -0.1816,  4.2092, -3.4402, -1.9263,  0.2303,  0.1826, -0.0486],
    [-0.4280, -0.4328,  0.0644, -3.9325, -3.9952,  0.7010, -0.0787, -0.0791,  0.0086],
    [ 0.3461,  0.2665,  0.4556,  3.1464,  2.4343,  4.1653,  0.0631,  0.0485,  0.0876],
    [ 2.3712,  1.6448,  3.3001,  0.6025,  0.3944,  0.8876,  1.6778,  1.0240,  2.6486],
    [ 0.3291, -0.2713, -0.1509,  3.1469, -2.5937, -1.4434,  0.0640, -0.0529, -0.0293],
    [-0.1811, -0.3585,  0.0054,  0.0516,  0.0190, -0.0144,  0.4618,  0.4086, -0.0911],
    [ 0.5169,  0.5402, -0.0961,  4.7148,  4.8835, -0.8230,  0.0989,  0.1108, -0.0388],
    [-0.3658,  0.3016,  0.1677, -3.0888,  2.5458,  1.4168, -0.0570,  0.0471,  0.0260],
    [ 0.3219, -0.2960, -0.1019,  2.9338, -2.4144, -1.3510,  0.0688, -0.0093, -0.1023],
    [ 0.0332,  0.7486, -1.5649, -0.0039, -0.0884,  0.1849, -0.0510, -1.1486,  2.4010],
])

R = 22
print("="*80)
print("SYMMETRY AUDIT OF RANK-22 APPROXIMATE 3x3 MATMUL DECOMPOSITION")
print("="*80)

# ============================================================
# 1. TRANSPOSE SYMMETRY: T_matmul has (A,B) -> (B^T, A^T) symmetry
#    i.e. C=AB => C^T = B^T A^T
#    On flattened indices, transpose of a 3x3 is permutation P_T
#    P_T maps index (i,j) -> (j,i), i.e. flat index 3i+j -> 3j+i
# ============================================================
print("\n" + "="*80)
print("1. TRANSPOSE SYMMETRY: T(A,B)^T = T(B^T, A^T)")
print("="*80)

# Transpose permutation on 9-vectors (3x3 flattened row-major)
P_T = [0, 3, 6, 1, 4, 7, 2, 5, 8]  # (i,j)->(j,i)

alpha_T = alpha[:, P_T]  # alpha columns permuted by transpose
beta_T = beta[:, P_T]

# If the decomposition respects transpose symmetry, there should be a 
# permutation sigma of {1..22} such that:
#   alpha[sigma(k)] ≈ c_k * beta_T[k] (up to scaling)
#   beta[sigma(k)] ≈ c_k * alpha_T[k]
#   gamma[sigma(k)] ≈ (1/c_k) * gamma_T[k]

# For each term k, find best matching term k' where alpha[k'] ~ beta_T[k]
print("\nSearching for transpose-paired terms (alpha[k'] ~ beta^T[k]):")
transpose_pairs = []
for k in range(R):
    bT_k = beta[k, P_T]  # transpose-permuted beta[k]
    bT_k_norm = bT_k / (np.linalg.norm(bT_k) + 1e-30)
    
    best_match = -1
    best_cos = 0
    best_scale = 0
    for kp in range(R):
        a_kp = alpha[kp]
        a_kp_norm = a_kp / (np.linalg.norm(a_kp) + 1e-30)
        cos = abs(np.dot(bT_k_norm, a_kp_norm))
        if cos > best_cos:
            best_cos = cos
            best_match = kp
            best_scale = np.dot(bT_k, a_kp) / (np.dot(a_kp, a_kp) + 1e-30)
    
    if best_cos > 0.99:
        transpose_pairs.append((k, best_match, best_cos, best_scale))
        print(f"  term {k+1:2d} <-> term {best_match+1:2d}  cos={best_cos:.6f}  scale={best_scale:.4f}")

if not transpose_pairs:
    print("  No transpose-paired terms found (cos > 0.99)")
    print("  => Decomposition BREAKS transpose symmetry of T_matmul")

# ============================================================
# 2. ROW/COLUMN STRUCTURE: Each 9-vector can be reshaped to 3x3.
#    Check if alpha[k] reshaped has row/column rank structure.
# ============================================================
print("\n" + "="*80)
print("2. RANK STRUCTURE OF RESHAPED COEFFICIENT MATRICES")
print("="*80)

for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    rank1_count = 0
    rank2_count = 0
    for k in range(R):
        mat = table[k].reshape(3, 3)
        svs = np.linalg.svd(mat, compute_uv=False)
        if svs[1] / svs[0] < 0.02:
            rank1_count += 1
        elif svs[2] / svs[0] < 0.02:
            rank2_count += 1
    print(f"  {name}: {rank1_count} rank-1, {rank2_count} rank-2, {R - rank1_count - rank2_count} full-rank (as 3x3)")

# ============================================================
# 3. NEAR-DUPLICATE / PROPORTIONAL TERM PAIRS
#    If alpha[k] ~ c * alpha[k'], this signals gauge redundancy
# ============================================================
print("\n" + "="*80)
print("3. NEAR-PROPORTIONAL PAIRS IN EACH TABLE")
print("="*80)

for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    pairs = []
    for i in range(R):
        for j in range(i+1, R):
            vi = table[i] / (np.linalg.norm(table[i]) + 1e-30)
            vj = table[j] / (np.linalg.norm(table[j]) + 1e-30)
            cos = abs(np.dot(vi, vj))
            if cos > 0.98:
                pairs.append((i+1, j+1, cos))
    if pairs:
        print(f"  {name}: {len(pairs)} near-proportional pairs:")
        for i, j, c in pairs:
            print(f"    terms {i} & {j}: |cos| = {c:.6f}")
    else:
        print(f"  {name}: no near-proportional pairs (|cos| > 0.98)")

# ============================================================
# 4. S3 PERMUTATION SYMMETRY ON ROWS/COLUMNS
#    The 3x3 matmul tensor has S3 x S3 x S3 symmetry (row/col/output perms)
#    Check if any S3 action maps the term set to itself
# ============================================================
print("\n" + "="*80)
print("4. S3 PERMUTATION SYMMETRY (row permutations of 3x3)")
print("="*80)

def perm_matrix_on_flat(perm):
    """Given a permutation of {0,1,2} (row permutation of 3x3),
    return the corresponding permutation of the 9 flat indices."""
    P = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            old_idx = 3*i + j
            new_idx = 3*perm[i] + j
            P[new_idx, old_idx] = 1
    return P

# For each S3 permutation (on rows of A), check if it permutes the term set
for perm in permutations([0, 1, 2]):
    if perm == (0, 1, 2):
        continue
    P = perm_matrix_on_flat(perm)
    # Under row perm of A: alpha -> P @ alpha^T, gamma -> P @ gamma^T
    alpha_perm = (P @ alpha.T).T
    gamma_perm = (P @ gamma.T).T
    
    # Check if alpha_perm is a row-permutation of alpha (up to scaling)
    matched = 0
    for k in range(R):
        v = alpha_perm[k] / (np.linalg.norm(alpha_perm[k]) + 1e-30)
        for kp in range(R):
            vp = alpha[kp] / (np.linalg.norm(alpha[kp]) + 1e-30)
            if abs(np.dot(v, vp)) > 0.99:
                matched += 1
                break
    if matched >= R - 2:
        print(f"  Row perm {perm}: {matched}/{R} alpha terms matched => APPROXIMATE S3 SYMMETRY")

print("  (If nothing printed, no S3 row-perm symmetry in the decomposition)")

# ============================================================
# 5. NOETHER-STYLE CONSERVED QUANTITIES
#    Check sum rules: do columns of alpha/beta/gamma sum to special values?
#    For exact matmul tensor, sum of all entries = trace(AB) structure
# ============================================================
print("\n" + "="*80)
print("5. CONSERVED QUANTITIES / SUM RULES (Noether perspective)")
print("="*80)

print("\n  Column sums (sum over 22 terms for each of 9 positions):")
for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    csums = table.sum(axis=0)
    print(f"    {name}: [{', '.join(f'{x:+.4f}' for x in csums)}]")
    print(f"           norm = {np.linalg.norm(csums):.4f}")

print("\n  Row sums (sum of 9 entries for each term):")
for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    rsums = table.sum(axis=1)
    print(f"    {name} row sums: [{', '.join(f'{x:+.4f}' for x in rsums)}]")

# Check: alpha_row_sum * beta_row_sum should relate to trace structure
print("\n  Product of row sums (alpha * beta per term):")
a_rs = alpha.sum(axis=1)
b_rs = beta.sum(axis=1)
g_rs = gamma.sum(axis=1)
ab_rs = a_rs * b_rs
print(f"    alpha*beta row-sum products: [{', '.join(f'{x:+.4f}' for x in ab_rs)}]")
print(f"    Weighted sum (gamma_rowsum * alpha_rowsum * beta_rowsum):")
trace_check = np.sum(g_rs * ab_rs)
print(f"    = {trace_check:.6f}")
print(f"    (For exact trace: should reconstruct trace(AB) = sum of diagonal entries)")

# ============================================================
# 6. GRAM MATRIX STRUCTURE (inner products between terms)
# ============================================================
print("\n" + "="*80)
print("6. GRAM MATRIX EIGENSPECTRUM")
print("="*80)

for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    G = table @ table.T  # 22x22 Gram matrix
    eigs = np.linalg.eigvalsh(G)[::-1]
    # Effective rank
    eigs_pos = eigs[eigs > 1e-10]
    eff_rank = len(eigs_pos)
    print(f"  {name} Gram: effective rank = {eff_rank}/9 (max possible)")
    print(f"    top 5 eigenvalues: {eigs[:5].round(3)}")
    print(f"    condition number (pos eigs): {eigs_pos[0]/eigs_pos[-1]:.1f}")

# ============================================================
# 7. TENSOR RECONSTRUCTION & RESIDUAL STRUCTURE
# ============================================================
print("\n" + "="*80)
print("7. RECONSTRUCTED TENSOR RESIDUAL ANALYSIS")
print("="*80)

# Build the exact 3x3 matmul tensor (9x9x9)
T_exact = np.zeros((9, 9, 9))
for i in range(3):
    for j in range(3):
        for k in range(3):
            # C[i,k] = sum_j A[i,j] * B[j,k]
            a_idx = 3*i + j  # A[i,j]
            b_idx = 3*j + k  # B[j,k]
            c_idx = 3*i + k  # C[i,k]
            T_exact[a_idx, b_idx, c_idx] += 1

# Reconstruct from decomposition
T_approx = np.zeros((9, 9, 9))
for k in range(R):
    T_approx += np.einsum('i,j,k->ijk', alpha[k], beta[k], gamma[k])

residual = T_exact - T_approx
print(f"  Frobenius norm of residual: {np.linalg.norm(residual):.6e}")
print(f"  Max absolute residual: {np.max(np.abs(residual)):.6e}")

# Check residual structure: is it low-rank?
res_flat = residual.reshape(9, 81)
svs = np.linalg.svd(res_flat, compute_uv=False)
print(f"  Residual (mode-1 unfolding) singular values:")
print(f"    {svs[:9].round(6)}")
print(f"  Residual effective rank (sv > 1e-4): {np.sum(svs > 1e-4)}")

# ============================================================
# 8. PAIRED TERM ANALYSIS (alpha/beta exchange symmetry)
# ============================================================
print("\n" + "="*80)
print("8. ALPHA-BETA EXCHANGE SYMMETRY PER TERM")
print("="*80)
print("  For each term k, check if alpha[k] ~ beta[k] (self-symmetric term)")
for k in range(R):
    a_n = alpha[k] / (np.linalg.norm(alpha[k]) + 1e-30)
    b_n = beta[k] / (np.linalg.norm(beta[k]) + 1e-30)
    cos = np.dot(a_n, b_n)
    if abs(cos) > 0.95:
        print(f"  Term {k+1}: cos(alpha, beta) = {cos:.4f} => SELF-SYMMETRIC")

# ============================================================
# 9. RATIONAL APPROXIMATION TEST (algebraic structure)
# ============================================================
print("\n" + "="*80)
print("9. RATIO ANALYSIS BETWEEN NEAR-PROPORTIONAL TERMS")
print("="*80)

# Check ratios between terms that look paired
for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    for i in range(R):
        for j in range(i+1, R):
            vi = table[i]
            vj = table[j]
            # Check if vi ~ c * vj
            nj = np.dot(vj, vj)
            if nj < 1e-10:
                continue
            c = np.dot(vi, vj) / nj
            resid = np.linalg.norm(vi - c * vj) / (np.linalg.norm(vi) + 1e-30)
            if resid < 0.05:
                print(f"  {name}[{i+1}] ≈ {c:.6f} * {name}[{j+1}]  (resid={resid:.4f})")

# ============================================================
# 10. BLOCK DIAGONAL / CLUSTER STRUCTURE
# ============================================================
print("\n" + "="*80)
print("10. CLUSTER STRUCTURE (cosine similarity graph)")
print("="*80)

# Build cosine similarity matrix for gamma (output side)
from collections import defaultdict

for name, table in [("gamma", gamma)]:
    cos_mat = np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            ni = np.linalg.norm(table[i])
            nj = np.linalg.norm(table[j])
            if ni > 1e-10 and nj > 1e-10:
                cos_mat[i,j] = np.dot(table[i], table[j]) / (ni * nj)
    
    # Simple clustering: group terms with |cos| > 0.9  
    visited = set()
    clusters = []
    for i in range(R):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, R):
            if j not in visited and abs(cos_mat[i,j]) > 0.9:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    
    if clusters:
        print(f"  {name} clusters (|cos| > 0.9):")
        for cl in clusters:
            print(f"    terms {[k+1 for k in cl]}")
    else:
        print(f"  No strong clusters in {name}")

# ============================================================
# 11. 3x3 RESHAPE AND TRACE/DETERMINANT INVARIANTS
# ============================================================
print("\n" + "="*80)
print("11. INVARIANTS OF RESHAPED 3x3 COEFFICIENT MATRICES")
print("="*80)

for name, table in [("alpha", alpha), ("beta", beta), ("gamma", gamma)]:
    traces = []
    dets = []
    for k in range(R):
        mat = table[k].reshape(3, 3)
        traces.append(np.trace(mat))
        dets.append(np.linalg.det(mat))
    print(f"\n  {name} traces: [{', '.join(f'{x:+.4f}' for x in traces)}]")
    print(f"  {name} trace sum: {sum(traces):.6f}")
    print(f"  {name} det sum: {sum(dets):.6f}")

# For exact matmul, the diagonal contribution is special
# trace(alpha[k].reshape(3,3)) relates to how term k contributes to diagonal of C
print(f"\n  Sum of (tr(alpha_k) * tr(beta_k) * tr(gamma_k)):")
total = 0
for k in range(R):
    ta = np.trace(alpha[k].reshape(3,3))
    tb = np.trace(beta[k].reshape(3,3))
    tg = np.trace(gamma[k].reshape(3,3))
    total += ta * tb * tg
print(f"    = {total:.6f}")
print(f"    (For exact tensor, trace(T,diag) = 3)")

# ============================================================
# 12. CYCLIC INDEX SYMMETRY (i,j,k -> j,k,i rotation on 3×3)
# ============================================================
print("\n" + "="*80)
print("12. CYCLIC / COMMUTATOR STRUCTURE")
print("="*80)

# For 3x3 matmul: T is NOT cyclic symmetric, but T + T_cyclic might be
# Check: is the residual antisymmetric under transpose?
res_T = np.zeros((9,9,9))
for a_idx in range(9):
    for b_idx in range(9):
        for c_idx in range(9):
            # Transpose: (a,b,c) -> (b_T, a_T, c_T)
            i_a, j_a = divmod(a_idx, 3)
            i_b, j_b = divmod(b_idx, 3)
            i_c, j_c = divmod(c_idx, 3)
            a_T = 3*j_a + i_a
            b_T = 3*j_b + i_b
            c_T = 3*j_c + i_c
            res_T[a_idx, b_idx, c_idx] = residual[b_T, a_T, c_T]

sym_part = (residual + res_T) / 2
anti_part = (residual - res_T) / 2
print(f"  Residual under transpose exchange:")
print(f"    Symmetric part norm: {np.linalg.norm(sym_part):.6e}")
print(f"    Antisymmetric part norm: {np.linalg.norm(anti_part):.6e}")

if np.linalg.norm(sym_part) < 0.1 * np.linalg.norm(anti_part):
    print("    => Residual is predominantly ANTISYMMETRIC (commutator-like)")
    print("    => The decomposition captures the symmetric part of T_matmul exactly!")
    print("    => NOETHER IMPLICATION: The approximate decomposition preserves a hidden")
    print("       AB+BA (anticommutator) conserved quantity.")
elif np.linalg.norm(anti_part) < 0.1 * np.linalg.norm(sym_part):
    print("    => Residual is predominantly SYMMETRIC")
else:
    ratio = np.linalg.norm(sym_part) / (np.linalg.norm(anti_part) + 1e-30)
    print(f"    => Mixed: sym/anti ratio = {ratio:.4f}")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
