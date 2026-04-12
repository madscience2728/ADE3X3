# CANON_NEURAL_NETWORK — Architecture Specification
## The Keth-Varai Machine

---

## Core Hypothesis

The naive rank-27 and even the known rank-23 (AlphaTensor) decompositions of $T_\text{matmul}$ are searched as **fixed** decompositions — the same coefficient vectors for every input pair.

**Hypothesis:** There exists a nonlinear, input-dependent selection of bilinear
decomposition coefficients such that, for each matrix pair $(A, B)$, the bilinear
rank required to reconstruct $C = AB$ to machine epsilon is **lower than any
fixed decomposition can achieve**.

A joint encoder sees both $A$ and $B$ and generates the weight matrices (the
"tuning knobs") of a bilinear bottleneck. The encoder output **never enters the
data path** — $A$ and $B$ interact only through the bilinear layer. This makes $N$
(the number of bilinear channels) an honest rank count.

This is a **hypernetwork**: the encoder learns the structure of the problem and
uses that to parameterize the computation, without performing the computation itself.

This is Arc θ from the xenobiology pipeline: **Build the machine. Let it tell us the rank.**

---

## Architecture

```
         A ∈ R^9              B ∈ R^9
         │                      │
         ├──────────┐           │
         │          ▼           │
         │   ┌────────────┐    │
         │   │  encoder   │◄───┤    MLP: cat(A,B) ∈ R^18 → latent ∈ R^L
         │   │  (joint)   │    │
         │   └─────┬──────┘    │
         │         │           │
         │         ▼           │
         │   ┌────────────┐    │
         │   │ hyper-heads │    │    latent → U(N×9), V(N×9), W(9×N)
         │   └──┬──┬──┬───┘    │
         │      U  V  W        │
         │      │  │  │        │
         ▼      ▼  │  │        ▼
       p = UA      │  │      q = VB     (linear projections, no activation)
         │         │  │        │
         └────┬────┘  │        │
              │       │        │
           p ⊙ q      │    (element-wise multiply)
              │       │
              ▼       │
         m ∈ R^N      │    m_k = (u_k·A)(v_k·B)  rank-1 bilinear per channel
              │       │
              ▼       │
        Ĉ = W·m  ◄───┘     (linear decode, no bias)
              │
              ▼
         Ĉ ∈ R^9            predicted flatten(A·B)
```

### Components

**Encoder** (joint): MLP that sees `cat(A, B)` ∈ R^18 and produces a latent
vector ∈ R^latent_dim. This is the "Keth-Varai neural sheet" — it learns the
structural relationship between A and B (the tuning knobs), not the answer.

**Hyper-heads**: Three linear projections from latent space to the weight matrices
U ∈ R^{N×9}, V ∈ R^{N×9}, W ∈ R^{9×N}. Generated per-sample.

**Bilinear Bottleneck**: Each channel is exactly a rank-1 bilinear form
$m_k = (u_k^\top A)(v_k^\top B)$. Architecturally enforced — no activations,
no latent in the data path. The encoder generates the weights but never touches
A or B directly in the computation.

**Decoder**: $\hat{C} = W \cdot m$. Linear, no bias. Every output component must
come from the N bilinear channels.

### Anti-cheat Guarantee

The encoder cannot smuggle the answer through the bottleneck because:
1. The latent vector parameterizes U, V, W — it does not enter the bilinear computation.
2. A and B enter as raw 9-vectors into the bilinear layer.
3. Their only interaction is $m_k = (u_k \cdot A)(v_k \cdot B)$ — rank-1 bilinear.
4. The decoder sees only the N scalars $m_k$, not the latent and not A or B.

$N$ is an honest count of rank-1 bilinear forms.

---

## Why This is Novel

Standard bilinear bottleneck (what everyone has tried):

```
[A, B] → Linear → [p_k × q_k] → Linear → C
```

This is CP decomposition with SGD. `frob_search.py` already does this better (L-BFGS-B).

**This architecture adds**: a hypernetwork that generates input-dependent
decomposition coefficients. The encoder sees the specific (A, B) pair and chooses
the best bilinear decomposition for THAT pair. If some pairs admit lower-rank
decompositions, the encoder can exploit that.

The key difference from a fixed decomposition: the encoder can learn that the
*structure* of the problem varies with the input, and adapt accordingly. If the
minimum N is the same for all inputs, that is itself a finding (universality).

This connects to: Arc η (characteristic-dependent rank), Arc ζ (hidden symmetry),
Arc θ (build the machine, let it tell us the rank), and the Keth-Varai body plan
(encoder = neural sheet, bottleneck = bridge between lobes).

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
- Train until relative error $< 10^{-5}$
- Verify: at N=27 the hypernetwork should converge easily
- Log: final Frobenius error, convergence curve

### Phase 2 — Squeeze (N=26 down to N=19)
- Reduce $N$ by 1 each step
- At each $N$: 5 seeds, best error logged
- Stop criterion: best error over all seeds $> 5 \times 10^{-3}$
- Record the **first $N$ that fails**

### Phase 3 — Interrogation (after minimum $N$ found)
- Sample many (A, B) pairs and extract the generated U, V, W matrices
- Check if the generated decompositions cluster (few archetypes) or vary continuously
- Check if U, V vectors align with known orbit structure (O0, O1, O2, O3)
- Compare to AlphaTensor's R=23 factors (structural, not numerical clone)
- Key question: does the encoder produce the SAME decomposition for all inputs?
  If yes → the rank is universal. If no → input-dependent rank, novel finding.

---

## Hyperparameters (defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `latent_dim` | 64 | Encoder output dim. Try 32, 128 also |
| `encoder_depth` | 3 | Hidden layers in joint encoder |
| `encoder_width` | 128 | Hidden units per layer |
| `N` (bilinear channels) | 27→sweep | Main variable |
| `lr` | 3e-4 | AdamW (weight decay on encoder only) |
| `batch_size` | 8192 | Random pairs each batch |
| `max_steps` | 200000 | Per run |
| `seeds_per_N` | 5 | For sweep |
| `input_noise_std` | 0.05 | Annealed to 0; prevents encoder exploiting precision |
| `lambda_entropy` | 1e-3 | Channel entropy reg; prevents load concentration |

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

1. Does the encoder generate the **same** U, V, W for all (A, B) pairs, or do the
   weights vary per input? If constant → the rank is universal and the hypernetwork
   has learned a fixed decomposition. If varying → input-dependent rank, novel finding.

2. At what N does the error first fail? Is it $\geq 23$ (consistent with known bounds)?
   Or can it reach $< 23$? (Would be a major finding.)

3. Do the generated weight vectors $u_k, v_k$ cluster into a small number of archetypes?
   (Would suggest discrete structural modes in the decomposition space.)

4. Are the generated $u_k, v_k$ near-integer or near-rational?
   (Connects to Smirnov coefficient structure.)

5. Does the hypernetwork find a lower N than a fixed-weight model (plain CP search)?
   If yes: evidence that input-dependent decomposition is genuinely more efficient.
   If no: the minimum rank is universal, and the hypernetwork collapses to fixed weights.
