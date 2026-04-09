"""Levenberg-Marquardt on R=22 with analytic Jacobian."""
import numpy as np, json
from scipy.optimize import least_squares

T = np.zeros((9,9,9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+s, s*3+u, r*3+u] = 1.0

R, DIM = 22, 9

def residuals(x):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    return (np.einsum('ki,kj,kl->ijl', A, B, C) - T).ravel()  # (729,)

def jacobian(x):
    A = x[:R*DIM].reshape(R, DIM)
    B = x[R*DIM:2*R*DIM].reshape(R, DIM)
    C = x[2*R*DIM:].reshape(R, DIM)
    J = np.zeros((729, R * 3 * DIM))
    # E[i,j,l] = sum_k A[k,i]*B[k,j]*C[k,l] - T[i,j,l]
    # flat index: n = i*81 + j*9 + l
    for k in range(R):
        # dE/dA[k,i0]: nonzero at n = i0*81 + j*9 + l, all j,l
        for i0 in range(DIM):
            n_start = i0 * 81
            J[n_start:n_start+81, k*DIM + i0] = np.outer(B[k], C[k]).ravel()
        # dE/dB[k,j0]: nonzero at n = i*81 + j0*9 + l, all i,l
        for j0 in range(DIM):
            for i0 in range(DIM):
                n_start = i0*81 + j0*9
                J[n_start:n_start+9, R*DIM + k*DIM + j0] = A[k, i0] * C[k]
        # dE/dC[k,l0]: nonzero at n = i*81 + j*9 + l0, all i,j
        for l0 in range(DIM):
            for i0 in range(DIM):
                for j0 in range(DIM):
                    n = i0*81 + j0*9 + l0
                    J[n, 2*R*DIM + k*DIM + l0] += A[k, i0] * B[k, j0]
    return J

with open('CANON_ADE_EPSILON/results/rank22_best.json') as f:
    d = json.load(f)
x0 = np.concatenate([np.array(d['alpha']).ravel(), np.array(d['beta']).ravel(), np.array(d['gamma']).ravel()])
frob0 = np.linalg.norm(residuals(x0))
print(f'Starting ε = {frob0:.8f}  (ε²={frob0**2:.2e})')

# Quick Jacobian sanity check (finite diff on first 5 params)
eps_fd = 1e-6
r0 = residuals(x0)
J0 = jacobian(x0)
for i in [0, 1, 50, 200, 500]:
    xp = x0.copy(); xp[i] += eps_fd
    fd = (residuals(xp) - r0) / eps_fd
    err = np.max(np.abs(fd - J0[:, i]))
    print(f'  Jac col {i}: max_err={err:.2e}')

print('\nRunning LM...')
res = least_squares(residuals, x0, jac=jacobian, method='lm',
                    ftol=1e-15, xtol=1e-15, gtol=1e-15,
                    max_nfev=200000, verbose=2)

frob1 = np.linalg.norm(res.fun)
print(f'\nFinal ε = {frob1:.8f}  (ε²={frob1**2:.2e})')
print(f'Status: {res.message}')

A = res.x[:R*DIM].reshape(R,DIM)
B = res.x[R*DIM:2*R*DIM].reshape(R,DIM)
C = res.x[2*R*DIM:].reshape(R,DIM)
with open('CANON_ADE_EPSILON/results/rank22_best.json','w') as f:
    json.dump({'rank':R,'frobenius':frob1,'frob_sq':frob1**2,
               'alpha':A.tolist(),'beta':B.tolist(),'gamma':C.tolist()}, f)
print('Saved.')
