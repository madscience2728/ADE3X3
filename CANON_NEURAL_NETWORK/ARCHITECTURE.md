# CANON_NEURAL_NETWORK — Architecture Specification
## The Keth-Varai Machine

---

## Core Hypothesis

The naive rank-27 and even the known rank-23 (AlphaTensor) decompositions of $T_\text{matmul}$ are searched in the **standard coordinate basis** of the 9-entry matrix representation.

The standard basis is arbitrary. It is a human cognitive artifact (entry-wise thinking). The Keth-Varai never use it.

**Hypothesis:** There exists a nonlinear change-of-basis (a learned encoder) such that, in the transformed representation, the bilinear rank of $T_\text{matmul}$ to machine epsilon ($\leq \epsilon_{\text{float32}} \approx 10^{-7}$) is **lower than in the standard basis**.

If true: the minimum achievable $N$ in the learned-basis bottleneck is a new empirical lower bound candidate — and the encoder weights are the explicit change-of-basis, auditable after training.

This is Arc θ from the xenobiology pipeline: **Build the machine. Let it tell us the rank.**

---

## Architecture

```
Input: A ∈ R^9, B ∈ R^9 (flattened 3×3 matrices)

┌─────────────────────────────────┐
│  Coordinate Transformer         │
│  φ_A = f_enc(A)  ∈ R^d         │
│  φ_B = g_enc(B)  ∈ R^d         │  ← shared or separate MLP
└─────────────────────────────────┘
           │                │
           ▼                ▼
      ã = [φ_A, A]     b̃ = [φ_B, B]      ← latent mixed with raw (R^{d+9})
           │                │
           └───────┬────────┘
                   ▼
        Bilinear Bottleneck (N channels)
        m_k = (u_k · ã)(v_k · b̃)  for k = 1..N
        u_k, v_k ∈ R^{d+9}  (learned linear projections)
                   │
                   ▼
            Decoder: W ∈ R^{9×N}
            Ĉ = W · [m_1, ..., m_N]
                   │
                   ▼
            Output: Ĉ ∈ R^9  (predicted flatten(A·B))
```

### Components

**Encoder** (`f_enc`, `g_enc`): Small MLP, same weights for both (the operation
is symmetric under relabeling if we want, but A-arm and B-arm are structurally
different so separate weights are used). Output dim `d` is a hyperparameter (default: 32).

**Mix**: Concatenate `[φ_A, A]` — the latent knows the structure; the raw keeps
the coordinate information. This is the "mixed with original inputs" step.

**Bilinear Bottleneck**: Each channel is exactly a rank-1 bilinear form
$m_k = (u_k^\top \tilde{a})(v_k^\top \tilde{b})$. This is architecturally enforced,
not approximated. No activations inside this layer.

**Decoder**: A single linear map. No nonlinearity — the channel outputs $m_k$ are
scalars and their linear combination must reconstruct $\hat{C}$.

---

## Why This is Novel

Standard bilinear bottleneck (what everyone has tried):

```
[A, B] → Linear → [p_k × q_k] → Linear → C
```

This is CP decomposition with SGD. `frob_search.py` already does this better (L-BFGS-B).

**This architecture adds**: a nonlinear coordinate transformer applied *before*
the bilinear layer. The encoder does not compute products — it transforms the
representation space. The bilinear rank is then measured in the *learned* basis.

The encoder may discover that, in the right basis, fewer than 23 bilinear
terms suffice to approximate $C = AB$ to float32 precision.

This connects to: Arc η (characteristic-dependent rank), Arc ζ (hidden symmetry),
and §3.6 of the xenobiology doc — the rank depends on how you count, and counting
requires choosing a basis.

---

## Training

**Dataset**: Random 3×3 matrices sampled from $\mathcal{N}(0, 1)$.
Target: $C = A \cdot B$ (exact numpy matmul, float64, then cast to float32).
Split: train/val/test (no data reuse for reporting).

**Loss**: MSE on flattened $\hat{C}$ vs $C$.

**Threshold**: Training is considered successful at rank $N$ if test Frobenius error
$\|C - \hat{C}\|_F \leq \epsilon_{\text{float32}} \approx 1.19 \times 10^{-7}$
normalized by $\|C\|_F$ (relative error).

**Regime sweep**: Start at $N=27$ (control). Reduce by 1. At each $N$, run
$k$ seeds (default: $k=10$). Report best and median relative Frobenius error.

---

## Protocol

### Phase 1 — Sanity (N=27)
- Build the full pipeline
- Train until relative error $< 10^{-6}$
- Verify: encoder output should be approximately identity (trivial transform)
- Log: final Frobenius error, convergence curve

### Phase 2 — Squeeze (N=26 down to N=19)
- Reduce $N$ by 1 each step
- At each $N$: 10 seeds, best error logged
- Stop criterion: best error over 10 seeds $> 10^{-5}$ (three orders above threshold)
- Record the **first $N$ that fails**

### Phase 3 — Interrogation (after minimum $N$ found)
- Project the learned encoder basis back to standard coordinates
- Check if the basis vectors align with known orbit structure (O0, O1, O2, O3)
- Check if the bottleneck weight matrices $U, V, W$ factor into anything interpretable
- Compare to AlphaTensor's R=23 factors (structural, not numerical clone)

---

## Hyperparameters (defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d` (encoder output dim) | 32 | Try 16, 64 also |
| `encoder_depth` | 2 | Hidden layers |
| `encoder_width` | 64 | Hidden units per layer |
| `N` (bilinear channels) | 27→sweep | Main variable |
| `lr` | 1e-3 | Adam |
| `batch_size` | 2048 | Random pairs each batch |
| `max_steps` | 50000 | Per run |
| `seeds_per_N` | 10 | For sweep |

---

## Files to Build

```
CANON_NEURAL_NETWORK/
    ARCHITECTURE.md         ← this file
    model.py                ← BilinearBottleneck and KethVaraiMachine classes
    train.py                ← single run, configurable N and seed
    sweep.py                ← loop N from 27 down, multi-seed, log results
    data.py                 ← matrix pair generator
    results/                ← training logs, sweep table
```

---

## Open Questions (tracked, not resolved)

1. Does the encoder converge to a **symmetric** basis at low N, or does it break symmetry? 
   (If it breaks: evidence for Arc ζ — hidden non-body symmetry)

2. At what N does the error first fail? Is it $\geq 23$ (consistent with known bounds)?
   Or can it reach $< 23$? (Would be a major finding.)

3. Does the encoder with $d=0$ (no latent, raw input only) find the same minimum N as
   a plain CP search? (Sanity check that encoder is doing real work)

4. Are the bilinear weight vectors $u_k, v_k$ near-integer or near-rational after training?
   (Connects to Smirnov coefficient structure)
