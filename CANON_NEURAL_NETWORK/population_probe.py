"""
population_probe.py — Computational Probe of the Tensor Rank Landscape.

Generates a large population of tiny networks. Varies architecture
systematically. Trains each briefly. Measures convergence rate.
Fast convergers are structurally aligned with the problem.

The search space is program space, not decomposition space. (Turing)
We're measuring which computational structures are commensurate
with the problem geometry. (Strassen)

Architecture grammar (LeCun/Hinton):
  - Bilinear bottleneck is FIXED: C_hat = W @ ((U@A) * (V@B))
    Cost: 15N params for 3×3, 12N for 2×2
  - Selector/encoder varies within remaining param budget
  - Degenerate cases included as controls (Hinton)

Usage:
    python population_probe.py --problem 2x2 --budget 200 --population 100 --steps 200
    python population_probe.py --problem 3x3 --budget 500 --N 23 --population 200 --steps 200
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

PROBLEMS = {
    "2x2": {"input_dim": 4, "output_dim": 4, "products_dim": 16, "mat_size": 2},
    "3x3": {"input_dim": 9, "output_dim": 9, "products_dim": 81, "mat_size": 3},
}


def sample_batch(problem: str, batch_size: int, device: torch.device):
    cfg = PROBLEMS[problem]
    n = cfg["mat_size"]
    d = n * n
    A = torch.randn(batch_size, d, dtype=torch.float64)
    B = torch.randn(batch_size, d, dtype=torch.float64)
    C = torch.bmm(A.view(-1, n, n), B.view(-1, n, n)).view(-1, d)
    return A.float().to(device), B.float().to(device), C.float().to(device)


def relative_error(C_pred, C_true, d: int):
    diff = (C_pred - C_true).view(-1, d)
    norm = C_true.view(-1, d).norm(dim=1).clamp(min=1e-12)
    return (diff.norm(dim=1) / norm).mean()


# ---------------------------------------------------------------------------
# Architecture grammar
# ---------------------------------------------------------------------------

@dataclass
class ArchSpec:
    """Hashable architecture specification."""
    name: str
    N: int
    selector_type: str   # "none" | "linear" | "mlp" | "attn" | "diagonal"
    selector_width: int
    selector_depth: int
    selector_activation: str  # "gelu" | "relu" | "tanh" | "none"
    use_products: bool   # True = expanded products, False = raw concat
    head_depth: int      # 0 = linear heads, 1+ = MLP heads

    @property
    def fingerprint(self) -> str:
        s = f"{self.name}|N{self.N}|{self.selector_type}|w{self.selector_width}|" \
            f"d{self.selector_depth}|{self.selector_activation}|" \
            f"prod{self.use_products}|hd{self.head_depth}"
        return hashlib.md5(s.encode()).hexdigest()[:12]


ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}


# ---------------------------------------------------------------------------
# Tiny network builder
# ---------------------------------------------------------------------------

class TinyBilinear(nn.Module):
    """Configurable tiny bilinear network from ArchSpec."""

    def __init__(self, spec: ArchSpec, problem: str = "3x3"):
        super().__init__()
        cfg = PROBLEMS[problem]
        self.d_in = cfg["input_dim"]
        self.d_out = cfg["output_dim"]
        self.d_prod = cfg["products_dim"]
        self.N = spec.N
        self.spec = spec

        # Input: either expanded products or raw concat
        if spec.use_products:
            enc_input = self.d_prod
        else:
            enc_input = 2 * self.d_in

        # Build selector (encoder → latent that generates U, V, W)
        if spec.selector_type == "none":
            # No selector — direct decomposition (control)
            self.selector = None
            self.U = nn.Parameter(torch.randn(spec.N, self.d_in) * 0.1)
            self.V = nn.Parameter(torch.randn(spec.N, self.d_in) * 0.1)
            self.W = nn.Parameter(torch.randn(self.d_out, spec.N) * 0.1)
        elif spec.selector_type == "diagonal":
            # Diagonal selector — minimal capacity control (Hinton)
            self.selector = None
            self.diag_enc = nn.Parameter(torch.randn(enc_input) * 0.1)
            latent_dim = min(spec.selector_width, enc_input)
            self.proj = nn.Linear(enc_input, latent_dim)
            self._build_heads(latent_dim, spec)
        else:
            # MLP / attention selector
            act = ACTIVATIONS[spec.selector_activation]
            layers = [nn.Linear(enc_input, spec.selector_width), act()]
            for _ in range(spec.selector_depth - 1):
                layers += [nn.Linear(spec.selector_width, spec.selector_width), act()]
            latent_dim = spec.selector_width
            self.selector = nn.Sequential(*layers)
            self._build_heads(latent_dim, spec)

        self._init_weights()

    def _build_heads(self, latent_dim, spec):
        """Build U, V, W heads from latent."""
        if spec.head_depth == 0:
            self.head_U = nn.Linear(latent_dim, spec.N * self.d_in)
            self.head_V = nn.Linear(latent_dim, spec.N * self.d_in)
            self.head_W = nn.Linear(latent_dim, self.d_out * spec.N)
        else:
            hw = max(latent_dim, 16)
            for name, out_dim in [("head_U", spec.N * self.d_in),
                                   ("head_V", spec.N * self.d_in),
                                   ("head_W", self.d_out * spec.N)]:
                layers = [nn.Linear(latent_dim, hw), nn.GELU()]
                for _ in range(spec.head_depth - 1):
                    layers += [nn.Linear(hw, hw), nn.GELU()]
                layers.append(nn.Linear(hw, out_dim))
                setattr(self, name, nn.Sequential(*layers))

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "head" in name and ("weight" in name) and p.dim() == 2:
                # Small init on head output layers
                if p.shape[0] > p.shape[1]:
                    nn.init.normal_(p, std=0.01)

    def _compute_products(self, A, B):
        return (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, self.d_prod)

    def forward(self, A, B):
        batch = A.shape[0]

        if self.spec.selector_type == "none":
            # Direct decomposition — no encoder
            p = A @ self.U.T
            q = B @ self.V.T
            m = p * q
            return m @ self.W.T

        # Compute input representation
        if self.spec.use_products:
            x = self._compute_products(A, B)
        else:
            x = torch.cat([A, B], dim=1)

        if self.spec.selector_type == "diagonal":
            x = x * self.diag_enc
            latent = self.proj(x)
        else:
            latent = self.selector(x)

        # Generate U, V, W from latent
        U = self.head_U(latent).view(batch, self.spec.N, self.d_in)
        V = self.head_V(latent).view(batch, self.spec.N, self.d_in)
        W = self.head_W(latent).view(batch, self.d_out, self.spec.N)

        # Bilinear bottleneck
        p = torch.bmm(U, A.unsqueeze(2)).squeeze(2)   # (batch, N)
        q = torch.bmm(V, B.unsqueeze(2)).squeeze(2)   # (batch, N)
        m = p * q                                       # (batch, N)
        return torch.bmm(W, m.unsqueeze(2)).squeeze(2)  # (batch, d_out)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Architecture sampler (LeCun/Hinton grammar)
# ---------------------------------------------------------------------------

def sample_architecture(N: int, budget: int, problem: str = "3x3",
                        rng: random.Random = None) -> Optional[ArchSpec]:
    """Sample a random tiny architecture within param budget."""
    if rng is None:
        rng = random.Random()

    cfg = PROBLEMS[problem]
    d = cfg["input_dim"]
    d_out = cfg["output_dim"]

    # Bilinear cost: N * (d + d + d_out) for direct, more for hypernetwork
    bilinear_direct = N * (d + d + d_out)

    # Selector types weighted to include controls (Hinton)
    selector_type = rng.choices(
        ["none", "linear", "mlp", "diagonal"],
        weights=[0.15, 0.25, 0.45, 0.15],
    )[0]

    if selector_type == "none":
        # Direct decomposition — all budget is bilinear
        if bilinear_direct > budget:
            return None
        return ArchSpec(
            name=f"direct_N{N}",
            N=N, selector_type="none",
            selector_width=0, selector_depth=0,
            selector_activation="none",
            use_products=False, head_depth=0,
        )

    # Remaining budget for selector + heads
    use_products = rng.random() < 0.6
    head_depth = rng.choices([0, 1], weights=[0.6, 0.4])[0]
    selector_activation = rng.choice(["gelu", "relu", "tanh"])

    if selector_type == "linear":
        selector_depth = 1
    elif selector_type == "diagonal":
        selector_depth = 1
    else:
        selector_depth = rng.randint(1, 3)

    # Find largest selector width that fits budget
    best_spec = None
    for width in [8, 12, 16, 24, 32, 48, 64]:
        spec = ArchSpec(
            name=f"{selector_type}_w{width}_d{selector_depth}_N{N}",
            N=N, selector_type=selector_type,
            selector_width=width, selector_depth=selector_depth,
            selector_activation=selector_activation,
            use_products=use_products, head_depth=head_depth,
        )
        try:
            model = TinyBilinear(spec, problem)
            if model.param_count() <= budget:
                best_spec = spec
            else:
                break
        except Exception:
            continue

    return best_spec


# ---------------------------------------------------------------------------
# Run log (Emmy + Feynman schema)
# ---------------------------------------------------------------------------

THRESHOLDS = [0.1, 0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7]


@dataclass
class ProbeLog:
    arch_fingerprint: str
    arch_spec: dict
    param_count: int
    N: int
    problem: str
    seed: int
    # Convergence
    final_error: float = float("inf")
    best_error: float = float("inf")
    converged: bool = False
    convergence_step: int = -1
    thresholds: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)
    # Weight analysis
    weight_stats: dict = field(default_factory=dict)
    # Feynman's failure fields
    failure_mode: str = ""
    per_output_error: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)
    vanishing_layers: list = field(default_factory=list)
    elapsed_s: float = 0.0


def near_integer_score(params):
    stats = {}
    for name, p in params.items():
        d = p.detach().cpu().float()
        residuals = (d - d.round()).abs()
        stats[name] = {
            "mean_residual": residuals.mean().item(),
            "frac_within_0.1": (residuals < 0.1).float().mean().item(),
        }
    return stats


# ---------------------------------------------------------------------------
# Train one tiny network
# ---------------------------------------------------------------------------

def train_probe(spec: ArchSpec, problem: str, seed: int,
                steps: int = 200, batch_size: int = 2048,
                lr: float = 1e-2, device: str = "auto") -> ProbeLog:
    """Train a tiny network for a few steps. Measure everything."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)

    cfg = PROBLEMS[problem]
    d_out = cfg["output_dim"]

    model = TinyBilinear(spec, problem).to(dev)

    plog = ProbeLog(
        arch_fingerprint=spec.fingerprint,
        arch_spec=asdict(spec),
        param_count=model.param_count(),
        N=spec.N, problem=problem, seed=seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        A, B, C = sample_batch(problem, batch_size, dev)
        C_hat = model(A, B)
        loss = relative_error(C_hat, C, d_out)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Log every 10 steps
        if step % 10 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                Av, Bv, Cv = sample_batch(problem, 2048, dev)
                err = relative_error(model(Av, Bv), Cv, d_out).item()

            plog.curve.append({"step": step, "error": err})
            if err < plog.best_error:
                plog.best_error = err
            for t in THRESHOLDS:
                key = f"{t:.0e}"
                if key not in plog.thresholds and err < t:
                    plog.thresholds[key] = step

        # Gradient norms at end
        if step == steps:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    plog.gradient_norms.append({
                        "name": name, "norm": p.grad.norm().item()
                    })
                    if p.grad.norm().item() < 1e-6:
                        plog.vanishing_layers.append(name)

    plog.elapsed_s = time.time() - t0
    plog.final_error = err

    # Per-output error
    model.eval()
    with torch.no_grad():
        Av, Bv, Cv = sample_batch(problem, 4096, dev)
        C_hat_v = model(Av, Bv)
        diff = (C_hat_v - Cv).abs()
        scale = Cv.view(-1, d_out).norm(dim=1, keepdim=True).clamp(min=1e-12)
        plog.per_output_error = (diff / scale).mean(dim=0).tolist()

    # Convergence / failure
    if plog.best_error < 1e-5:
        plog.converged = True
        plog.convergence_step = plog.thresholds.get("1e-05", steps)
    else:
        errors = [c["error"] for c in plog.curve]
        if len(errors) > 5:
            early = sum(errors[:3]) / 3
            late = sum(errors[-3:]) / 3
            if late > 0.5:
                plog.failure_mode = "no_convergence"
            elif late > early * 0.9:
                plog.failure_mode = "plateau"
            else:
                plog.failure_mode = "slow_descent"

    # Weight stats
    plog.weight_stats = near_integer_score(dict(model.named_parameters()))

    return plog


# ---------------------------------------------------------------------------
# Population sweep
# ---------------------------------------------------------------------------

def run_population(problem: str = "3x3", N: int = 23, budget: int = 500,
                   population: int = 100, steps: int = 200, seeds: int = 1,
                   device: str = "auto"):
    """Sample and evaluate a population of tiny architectures."""

    rng = random.Random(42)
    results = []
    attempted = 0
    skipped = 0

    print(f"\n{'='*70}")
    print(f"  POPULATION PROBE  |  {problem}  N={N}  budget={budget}  pop={population}")
    print(f"  steps={steps}  seeds={seeds}")
    print(f"{'='*70}\n")

    seen_fingerprints = set()

    while len(results) < population * seeds and attempted < population * 10:
        attempted += 1
        spec = sample_architecture(N, budget, problem, rng)
        if spec is None:
            skipped += 1
            continue
        if spec.fingerprint in seen_fingerprints and seeds == 1:
            continue
        seen_fingerprints.add(spec.fingerprint)

        for s in range(seeds):
            plog = train_probe(spec, problem, seed=s, steps=steps, device=device)
            bits = -math.log2(plog.best_error) if plog.best_error > 0 else 99
            status = "✓" if plog.converged else f"✗ {plog.failure_mode}"
            print(f"  [{len(results)+1:>4d}/{population*seeds}]  "
                  f"{spec.name:<35s}  params={plog.param_count:>4d}  "
                  f"best={plog.best_error:.3e}  bits={bits:.1f}  {status}  "
                  f"{plog.elapsed_s:.1f}s")
            results.append(asdict(plog))

    # Sort by best_error
    results.sort(key=lambda r: r["best_error"])

    # Summary
    print(f"\n{'='*70}")
    print(f"  TOP 10 ARCHITECTURES")
    print(f"{'='*70}")
    for i, r in enumerate(results[:10]):
        bits = -math.log2(r["best_error"]) if r["best_error"] > 0 else 99
        print(f"  {i+1:>2d}. {r['arch_spec']['name']:<35s}  "
              f"params={r['param_count']:>4d}  best={r['best_error']:.3e}  bits={bits:.1f}")

    # Failure distribution (Feynman)
    print(f"\n  FAILURE DISTRIBUTION:")
    modes = {}
    for r in results:
        m = r["failure_mode"] if r["failure_mode"] else "converged"
        modes[m] = modes.get(m, 0) + 1
    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
        print(f"    {mode}: {count}")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"population_{problem}_N{N}")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "population_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")
    print(f"  {len(results)} runs, {skipped} specs skipped (over budget)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Population probe of tensor rank landscape")
    parser.add_argument("--problem", choices=["2x2", "3x3"], default="3x3")
    parser.add_argument("--N", type=int, default=23)
    parser.add_argument("--budget", type=int, default=500,
                        help="Max parameter count per network")
    parser.add_argument("--population", type=int, default=100,
                        help="Number of distinct architectures to sample")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps per network (keep small)")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_population(
        problem=args.problem, N=args.N, budget=args.budget,
        population=args.population, steps=args.steps,
        seeds=args.seeds, device=args.device,
    )


if __name__ == "__main__":
    main()
