import numpy as np
import os, glob

def build_T333():
    n = 3
    dim = n * n
    T = np.zeros((dim, dim, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                alpha = i * n + j
                beta  = j * n + k
                gamma = i * n + k
                T[alpha, beta, gamma] = 1.0
    return T

T = build_T333()
results_dir = r"C:\Users\madsc\Desktop\Github\ADE3X3\IMPORTED_TEE\matmul_search_results"
files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))

print(f"Found {len(files)} factor files\n")

for f in files:
    try:
        data = np.load(f)
        keys = list(data.keys())
        name = os.path.basename(f)
        
        if "U" in keys and "V" in keys and "W" in keys:
            U, V, W = data["U"], data["V"], data["W"]
            R_hat = np.einsum("ra,rb,rc->abc", U, V, W)
            residual = np.linalg.norm(R_hat - T) / np.linalg.norm(T)
            rank = U.shape[0]
            print(f"{name}: rank={rank}, shape U={U.shape}, rel_residual={residual:.6e}")
            
            if rank in [19, 20, 21, 22, 23]:
                norms = np.sqrt(np.sum(U**2, axis=1) * np.sum(V**2, axis=1) * np.sum(W**2, axis=1))
                print(f"  Term norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
                print(f"  Abs residual: {np.linalg.norm(R_hat - T):.6e}")
        else:
            print(f"{name}: keys={keys}")
    except Exception as e:
        print(f"Error processing {f}: {e}")
