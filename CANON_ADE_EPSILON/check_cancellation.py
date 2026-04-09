import json, numpy as np

def compute_Sigma(alpha_flat, beta_flat):
    R = alpha_flat.shape[0]
    A = alpha_flat.reshape(R,3,3); B = beta_flat.reshape(R,3,3)
    Sigma = np.zeros((R,9))
    for k in range(R):
        for r in range(3):
            for u in range(3):
                Sigma[k, r*3+u] = sum(A[k,r,s]*B[k,s,u] for s in range(3))
    return Sigma

with open('outputs/ade3x3_attack/phase4_gradient_search/best_decompositions/10_border23_0000.json') as f:
    d = json.load(f)
terms = d['terms']
R = len(terms)
alpha = np.array([t['alpha'] for t in terms])
beta  = np.array([t['beta']  for t in terms])
gamma = np.array([t['gamma'] for t in terms]).reshape(R, 9)

Sigma = compute_Sigma(alpha, beta)
U, sv_s, Vt = np.linalg.svd(Sigma, full_matrices=False)

Gamma_stored = gamma.T  # shape 9xR

# Check fiber-sum constraint
fiber = Gamma_stored @ Sigma
print("Gamma_stored @ Sigma (should be 3*I9):")
print(np.round(fiber, 4))
print(f"Deviation from 3*I9: {np.linalg.norm(fiber - 3*np.eye(9)):.6f}")
print()

# Sigma = U S Vt.  Gamma.Sigma = 3I => (Gamma.U).S.Vt = 3I => Gamma.U = 3.Vt^T / S
# So Gamma.U[:,i] carries the weight for Sigma direction i; loading = ||( Gamma_stored @ U )[:,i]||
GammaU = Gamma_stored @ U  # 9x9
Gamma_in_sigma_basis = GammaU  # column i is loading on Sigma left-singular direction i
print("Gamma_stored loading on each Sigma singular direction:")
print("  dir   sigma       gamma_loading   sigma*loading")
for i in range(9):
    loading = np.linalg.norm(Gamma_in_sigma_basis[:, i])
    print(f"  {i:3d}   {sv_s[i]:9.6f}   {loading:13.6f}   {sv_s[i]*loading:13.6f}")
