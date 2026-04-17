"""
algebraic_net.py — Learn-to-snap: weights ARE algebraic by construction.

Each weight is parameterized as:
    w = sign * (numerator / denominator) ^ (1 / root)

where sign ∈ {-1, 0, +1}, numerator ∈ {0..10}, denominator ∈ {1..10}, root ∈ {1,2,3,4}.

Learnable parameters are logits over each categorical choice.
Straight-through Gumbel-softmax makes it differentiable.
Temperature anneals from τ_start → τ_end over training.

No snapping needed — the weights are algebraic at every step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Algebraic weight layer
# ---------------------------------------------------------------------------

# Categorical choices
SIGNS = torch.tensor([-1.0, 0.0, 1.0])          # 3 choices
NUMERATORS = torch.arange(0, 11, dtype=torch.float32)  # 0..10, 11 choices
DENOMINATORS = torch.arange(1, 11, dtype=torch.float32) # 1..10, 10 choices
ROOTS = torch.arange(1, 5, dtype=torch.float32)         # 1..4,  4 choices


def gumbel_softmax_hard(logits, tau):
    """Straight-through Gumbel-softmax: forward uses argmax, backward uses soft."""
    soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    # Straight-through: hard in forward, soft gradient in backward
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
    return hard - soft.detach() + soft


class AlgebraicWeight(nn.Module):
    """
    A tensor of algebraic weights, each parameterized by 4 sets of logits.
    Shape: (*shape) weights, each = sign * (num/den)^(1/root).
    """
    def __init__(self, shape):
        super().__init__()
        n = math.prod(shape)
        self.shape_ = shape
        self.n = n

        # Logits for each categorical variable
        self.sign_logits = nn.Parameter(torch.randn(n, len(SIGNS)) * 0.1)
        self.num_logits = nn.Parameter(torch.randn(n, len(NUMERATORS)) * 0.1)
        self.den_logits = nn.Parameter(torch.randn(n, len(DENOMINATORS)) * 0.1)
        self.root_logits = nn.Parameter(torch.randn(n, len(ROOTS)) * 0.1)

    def forward(self, tau=1.0):
        dev = self.sign_logits.device

        signs = SIGNS.to(dev)
        nums = NUMERATORS.to(dev)
        dens = DENOMINATORS.to(dev)
        roots = ROOTS.to(dev)

        # Gumbel-softmax selections (straight-through)
        s = gumbel_softmax_hard(self.sign_logits, tau)   # (n, 3)
        n = gumbel_softmax_hard(self.num_logits, tau)     # (n, 11)
        d = gumbel_softmax_hard(self.den_logits, tau)     # (n, 10)
        r = gumbel_softmax_hard(self.root_logits, tau)    # (n, 4)

        # Compute values
        sign_val = (s * signs).sum(-1)           # (n,)
        num_val = (n * nums).sum(-1)             # (n,)
        den_val = (d * dens).sum(-1)             # (n,)
        root_val = (r * roots).sum(-1)           # (n,)

        # w = sign * (num/den)^(1/root)
        ratio = num_val / den_val.clamp(min=1.0)
        w = sign_val * ratio.clamp(min=0.0).pow(1.0 / root_val.clamp(min=1.0))

        return w.view(self.shape_)

    def get_discrete(self):
        """Return the hard-argmax discrete values (no Gumbel noise)."""
        dev = self.sign_logits.device
        signs = SIGNS.to(dev)
        nums = NUMERATORS.to(dev)
        dens = DENOMINATORS.to(dev)
        roots = ROOTS.to(dev)

        s_idx = self.sign_logits.argmax(-1)
        n_idx = self.num_logits.argmax(-1)
        d_idx = self.den_logits.argmax(-1)
        r_idx = self.root_logits.argmax(-1)

        sign_val = signs[s_idx]
        num_val = nums[n_idx]
        den_val = dens[d_idx]
        root_val = roots[r_idx]

        ratio = num_val / den_val.clamp(min=1.0)
        w = sign_val * ratio.clamp(min=0.0).pow(1.0 / root_val.clamp(min=1.0))

        return w.view(self.shape_), (s_idx, n_idx, d_idx, r_idx)

    def label(self, idx):
        """Human-readable label for weight at flat index idx."""
        dev = self.sign_logits.device
        s = SIGNS[self.sign_logits[idx].argmax()].item()
        n = NUMERATORS[self.num_logits[idx].argmax()].item()
        d = DENOMINATORS[self.den_logits[idx].argmax()].item()
        r = ROOTS[self.root_logits[idx].argmax()].item()

        if s == 0 or n == 0:
            return "0"

        sign_str = "-" if s < 0 else ""
        n, d, r = int(n), int(d), int(r)

        # Simplify
        if d == 1:
            base = str(n)
        else:
            from math import gcd
            g = gcd(n, d)
            base = f"{n // g}/{d // g}"

        if r == 1:
            return f"{sign_str}{base}"
        elif r == 2:
            return f"{sign_str}√({base})"
        elif r == 3:
            return f"{sign_str}∛({base})"
        else:
            return f"{sign_str}∜({base})"


# ---------------------------------------------------------------------------
# Algebraic bilinear decomposition
# ---------------------------------------------------------------------------

class AlgebraicDecomp(nn.Module):
    """
    Bilinear decomposition C_hat = W @ ((U@A) * (V@B))
    where U, V, W are AlgebraicWeight tensors.
    
    Every weight is algebraic by construction. No snapping needed.
    """
    def __init__(self, N, d_in, d_out):
        super().__init__()
        self.N = N
        self.d_in = d_in
        self.d_out = d_out

        self.U = AlgebraicWeight((N, d_in))
        self.V = AlgebraicWeight((N, d_in))
        self.W = AlgebraicWeight((d_out, N))

    def forward(self, A, B, tau=1.0):
        U = self.U(tau)  # (N, d_in)
        V = self.V(tau)  # (N, d_in)
        W = self.W(tau)  # (d_out, N)

        p = A @ U.T      # (batch, N)
        q = B @ V.T      # (batch, N)
        m = p * q         # (batch, N)
        return m @ W.T    # (batch, d_out)

    def get_UVW(self):
        """Get discrete U, V, W matrices."""
        U, _ = self.U.get_discrete()
        V, _ = self.V.get_discrete()
        W, _ = self.W.get_discrete()
        return U, V, W

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def print_labels(self, header=""):
        """Print algebraic labels for all weights."""
        if header:
            print(f"\n  {header}")
        for name, alg in [("U", self.U), ("V", self.V), ("W", self.W)]:
            vals, _ = alg.get_discrete()
            print(f"  {name}:")
            for i in range(vals.shape[0]):
                labels = [alg.label(i * vals.shape[1] + j) for j in range(vals.shape[1])]
                print(f"    ch{i}: [{', '.join(labels)}]")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_algebraic(
    N=7, mat_size=2, steps=5000,
    tau_start=2.0, tau_end=0.1,
    lr=3e-3, batch_size=4096,
    log_every=100, snapshot_every=500,
    seed=0, device="cpu",
):
    if device == "cpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)

    d_in = mat_size * mat_size
    d_out = mat_size * mat_size
    model = AlgebraicDecomp(N, d_in, d_out).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    print(f"\n{'='*70}")
    print(f"  ALGEBRAIC NET  |  {mat_size}x{mat_size} N={N}")
    print(f"  params={model.param_count()}  τ: {tau_start} → {tau_end}")
    print(f"  steps={steps}  lr={lr}  seed={seed}")
    print(f"{'='*70}")

    log = []

    for step in range(1, steps + 1):
        # Temperature schedule (exponential decay)
        frac = (step - 1) / max(steps - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** frac

        model.train()
        A = torch.randn(batch_size, d_in, device=dev)
        B = torch.randn(batch_size, d_in, device=dev)
        C = torch.bmm(
            A.view(-1, mat_size, mat_size),
            B.view(-1, mat_size, mat_size),
        ).view(-1, d_out)

        C_hat = model(A, B, tau=tau)
        err_vec = (C_hat - C).norm(dim=1) / C.norm(dim=1).clamp(min=1e-12)
        loss = err_vec.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                Av = torch.randn(8192, d_in, device=dev)
                Bv = torch.randn(8192, d_in, device=dev)
                Cv = torch.bmm(
                    Av.view(-1, mat_size, mat_size),
                    Bv.view(-1, mat_size, mat_size),
                ).view(-1, d_out)

                # Use hard argmax for eval (no Gumbel noise)
                U, V, W = model.get_UVW()
                p = Av @ U.T
                q = Bv @ V.T
                m = p * q
                C_h = m @ W.T
                val_err = ((C_h - Cv).norm(dim=1) / Cv.norm(dim=1).clamp(min=1e-12)).mean().item()

            bits = -math.log2(val_err) if val_err > 0 else 99
            entry = {"step": step, "tau": tau, "train_err": loss.item(),
                     "val_err": val_err, "bits": bits}
            log.append(entry)
            print(f"  step {step:5d}  τ={tau:.3f}  train={loss.item():.3e}  "
                  f"val(discrete)={val_err:.3e}  bits={bits:.1f}")

        if step % snapshot_every == 0:
            model.eval()
            model.print_labels(f"--- SNAPSHOT step {step} ---")

    return model, log


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_algebraic(model, mat_size=2):
    """Verify the discrete decomposition."""
    from crystallize import verify_decomposition, strassen_UVW

    U, V, W = model.get_UVW()
    err = verify_decomposition(U, V, W, n=mat_size)
    bits = -math.log2(err) if err > 0 else 99
    status = "★ EXACT" if err < 1e-6 else ("~ near" if err < 0.01 else "✗ fail")

    print(f"\n{'='*70}")
    print(f"  ALGEBRAIC VERIFICATION")
    print(f"{'='*70}")
    print(f"  Discrete decomposition error: {err:.3e}  bits={bits:.1f}  {status}")
    model.print_labels("Final decomposition:")

    if mat_size == 2:
        U_s, V_s, W_s = strassen_UVW()
        serr = verify_decomposition(U_s, V_s, W_s, n=2)
        print(f"\n  Strassen sanity: error={serr:.3e}")

    return err


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  LEARN-TO-SNAP: Algebraic Weight Network")
    print("  Weights = sign * (num/den)^(1/root)  with Gumbel-softmax")
    print("=" * 70)

    best_err = float("inf")
    best_model = None

    for seed in range(5):
        model, log = train_algebraic(
            N=7, mat_size=2, steps=5000,
            tau_start=2.0, tau_end=0.05,
            lr=3e-3, seed=seed,
            log_every=200, snapshot_every=1000,
        )
        err = verify_algebraic(model, mat_size=2)
        if err < best_err:
            best_err = err
            best_model = model

    print(f"\n{'='*70}")
    print(f"  BEST RESULT: error={best_err:.3e}")
    print(f"{'='*70}")
    if best_model is not None:
        best_model.print_labels("Best decomposition:")
