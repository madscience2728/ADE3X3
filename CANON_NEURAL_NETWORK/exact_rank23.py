"""exact_rank23.py — Get exact rank-23 via gradient descent with adaptive restarts.

The ALS+LBFGS approach plateaus at ~1e-2 because ALS gets stuck in the 
swamp (flat region near degenerate decompositions). 

New strategy:
1. Use ADAM optimizer (PyTorch) instead of ALS — handles saddle points better
2. Much longer optimization (50k+ steps)
3. Learning rate warmup + cosine decay
4. Multiple random seeds, keep the winner
5. If we get below 1e-4, switch to high-precision LBFGS for final polish
"""
import torch
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Build T333
def build_T333():
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[3*i+j, 3*j+k, 3*i+k] = 1.0
    return T

T_np = build_T333()
T = torch.tensor(T_np, dtype=torch.float64, device=device)
T_NORM = T.norm().item()

def try_rank(rank, seed, steps=80000, lr=3e-3):
    """Single optimization run at given rank."""
    torch.manual_seed(seed)
    
    # Xavier-like init scaled for tensor decomp
    scale = (T_NORM / rank) ** (1/3) * 0.5
    U = torch.randn(rank, 9, dtype=torch.float64, device=device) * scale
    V = torch.randn(rank, 9, dtype=torch.float64, device=device) * scale  
    W = torch.randn(rank, 9, dtype=torch.float64, device=device) * scale
    
    U = torch.nn.Parameter(U)
    V = torch.nn.Parameter(V)
    W = torch.nn.Parameter(W)
    
    optimizer = torch.optim.Adam([U, V, W], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    
    best_res = float("inf")
    best_state = None
    
    for step in range(steps):
        optimizer.zero_grad()
        R = torch.einsum("ra,rb,rc->abc", U, V, W)
        loss = 0.5 * ((R - T) ** 2).sum()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([U, V, W], 10.0)
        
        optimizer.step()
        scheduler.step()
        
        res = (R - T).norm().item()
        if res < best_res:
            best_res = res
            best_state = (U.data.clone(), V.data.clone(), W.data.clone())
        
        if step % 5000 == 0:
            print(f"    step {step:6d}: loss={loss.item():.6e} res={res:.6e} best={best_res:.6e} lr={scheduler.get_last_lr()[0]:.2e}")
        
        if best_res < 1e-10:
            print(f"    CONVERGED at step {step}")
            break
    
    return best_state, best_res


def lbfgs_polish(U_t, V_t, W_t, maxiter=5000):
    """Final polish with LBFGS."""
    U = torch.nn.Parameter(U_t.clone())
    V = torch.nn.Parameter(V_t.clone())
    W = torch.nn.Parameter(W_t.clone())
    
    optimizer = torch.optim.LBFGS([U, V, W], lr=1.0, max_iter=20, 
                                   history_size=50, tolerance_grad=1e-16, tolerance_change=1e-16)
    
    best_res = float("inf")
    best_state = None
    
    for i in range(maxiter // 20):
        def closure():
            optimizer.zero_grad()
            R = torch.einsum("ra,rb,rc->abc", U, V, W)
            loss = 0.5 * ((R - T) ** 2).sum()
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        res = (torch.einsum("ra,rb,rc->abc", U, V, W) - T).norm().item()
        
        if res < best_res:
            best_res = res
            best_state = (U.data.clone(), V.data.clone(), W.data.clone())
        
        if i % 10 == 0:
            print(f"    LBFGS iter {i*20:5d}: res={res:.6e} best={best_res:.6e}")
        
        if best_res < 1e-12:
            break
    
    return best_state, best_res


def main():
    print("="*60)
    print("EXACT RANK SEARCH FOR T_{3,3,3}")
    print("="*60)
    
    for rank in [23, 22, 21, 20]:
        print(f"\n{'='*60}")
        print(f"RANK {rank}")
        print(f"{'='*60}")
        
        best_res = float("inf")
        best_state = None
        t0 = time.time()
        
        for seed in range(50):
            print(f"\n  Seed {seed}:")
            state, res = try_rank(rank, seed, steps=50000, lr=3e-3)
            
            if res < best_res:
                best_res = res
                best_state = state
            
            print(f"  -> res={res:.6e}  BEST={best_res:.6e}  [{time.time()-t0:.0f}s]")
            
            # If close, try LBFGS polish
            if res < 0.1:
                print(f"  Polishing with LBFGS...")
                polished, pol_res = lbfgs_polish(*state)
                if pol_res < best_res:
                    best_res = pol_res
                    best_state = polished
                print(f"  -> polished: {pol_res:.6e}  BEST={best_res:.6e}")
            
            if best_res < 1e-8:
                print(f"\n  ** EXACT rank-{rank} found! **")
                break
        
        # Save best
        U, V, W = [x.cpu().numpy() for x in best_state]
        np.savez(f"exact_r{rank}.npz", U=U, V=V, W=W, residual=np.array([best_res]))
        
        tag = "EXACT" if best_res < 1e-6 else "APPROXIMATE"
        print(f"\n  RANK {rank} RESULT: {best_res:.6e} [{tag}]  ({time.time()-t0:.0f}s)")
        
        if best_res > 0.5:
            print(f"  Cannot find rank-{rank}. Stopping descent.")
            break

if __name__ == "__main__":
    main()
