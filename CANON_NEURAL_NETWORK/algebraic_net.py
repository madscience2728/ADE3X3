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


def gumbel_softmax_hard(logits, tau):
    """Straight-through Gumbel-softmax: forward uses argmax, backward uses soft."""
    soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    # Straight-through: hard in forward, soft gradient in backward
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
    return hard - soft.detach() + soft


def gumbel_softmax_soft(logits, tau):
    """Soft Gumbel-softmax (no straight-through). For blending values smoothly."""
    return F.gumbel_softmax(logits, tau=tau, hard=False)


class AlgebraicWeight(nn.Module):
    """
    A tensor of algebraic weights, each parameterized by 4 sets of logits.
    Shape: (*shape) weights, each = sign * (num/den)^(1/root).
    
    Strategy: precompute ALL possible values into a table, then each weight
    selects from the table via Gumbel-softmax. This avoids numerical issues
    with differentiating through pow/log.
    """
    def __init__(self, shape):
        super().__init__()
        n = math.prod(shape)
        self.shape_ = shape
        self.n = n

        # Build value table: all sign * (num/den)^(1/root) combinations
        vals = []
        labels = []
        for s in [-1.0, 0.0, 1.0]:
            for ni in range(11):      # num 0..10
                for di in range(1, 11):  # den 1..10
                    for ri in range(1, 5):  # root 1..4
                        v = s * (ni / di) ** (1.0 / ri) if ni > 0 else 0.0
                        vals.append(v)
                        labels.append((s, ni, di, ri))
        # Deduplicate (keep first occurrence)
        seen = {}
        unique_vals = []
        unique_labels = []
        for v, lab in zip(vals, labels):
            key = round(v, 10)
            if key not in seen:
                seen[key] = len(unique_vals)
                unique_vals.append(v)
                unique_labels.append(lab)
        
        self.register_buffer("value_table", torch.tensor(unique_vals, dtype=torch.float32))
        self._labels = unique_labels
        V = len(unique_vals)
        
        # Each weight: logits over the value table
        self.logits = nn.Parameter(torch.randn(n, V) * 0.1)

    def forward(self, tau=1.0):
        # Soft selection from value table
        weights = gumbel_softmax_hard(self.logits, tau)  # (n, V)
        w = (weights * self.value_table).sum(-1)          # (n,)
        return w.view(self.shape_)

    def get_discrete(self):
        """Return the hard-argmax discrete values."""
        idx = self.logits.argmax(-1)  # (n,)
        w = self.value_table[idx]
        return w.view(self.shape_), idx

    def label(self, flat_idx):
        """Human-readable label for weight at flat index."""
        idx = self.logits[flat_idx].argmax().item()
        s, n, d, r = self._labels[idx]
        
        if s == 0 or n == 0:
            return "0"

        sign_str = "-" if s < 0 else ""
        from math import gcd
        g = gcd(n, d)
        n_s, d_s = n // g, d // g

        if d_s == 1:
            base = str(n_s)
        else:
            base = f"{n_s}/{d_s}"

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
    seed=0, device="auto",
):
    if device == "auto":
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
