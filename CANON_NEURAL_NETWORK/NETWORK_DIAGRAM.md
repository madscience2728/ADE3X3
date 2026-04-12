# Keth-Varai Machine — Network Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                    ENCODER (11M params)                 │
                         │                                                         │
  A ∈ ℝ⁹ ──┐             │   ┌──────────────────┐                                  │
            ├─ cat ──────►│   │  Linear(18→576)   │                                  │
  B ∈ ℝ⁹ ──┘   ℝ¹⁸      │   │     + GELU        │                                  │
                         │   └────────┬─────────┘                                  │
                         │            │  ℝ⁵⁷⁶                                      │
                         │            ▼                                            │
                         │   ┌────────────────────────────────────────────┐         │
                         │   │          16 × ResBlock + 4 × AttnBlock    │         │
                         │   │                                            │         │
                         │   │  ResBlock ×4:                              │         │
                         │   │    ┌───────────────────────────┐           │         │
                         │   │    │ ╭─────────────────────╮   │           │         │
                         │   │    │ │ LayerNorm(576)       │   │           │         │
                         │   │    │ │ Linear(576→576)      │   │           │         │
                         │   │    │ │ GELU                 │   │           │         │
                         │   │    │ │ Dropout(0.1)         │   │           │         │
                         │   │    │ │ Linear(576→576)      │   │           │         │
                         │   │    │ │ Dropout(0.1)         │   │           │         │
                         │   │    │ ╰──────────┬──────────╯   │           │         │
                         │   │    │      x ────(+)────► out   │           │         │
                         │   │    └───────────────────────────┘           │         │
                         │   │                 │                          │         │
                         │   │  AttnBlock (after every 4th ResBlock):     │         │
                         │   │    ┌───────────────────────────┐           │         │
                         │   │    │  reshape ℝ⁵⁷⁶ → 9 × ℝ⁶⁴  │           │         │
                         │   │    │  ╭─────────────────────╮  │           │         │
                         │   │    │  │ LayerNorm(64)        │  │           │         │
                         │   │    │  │ MultiHeadAttn        │  │           │         │
                         │   │    │  │   4 heads × 16 dim   │  │           │         │
                         │   │    │  │ Dropout(0.1)         │  │           │         │
                         │   │    │  ╰──────────┬──────────╯  │           │         │
                         │   │    │       x ────(+)────► out  │           │         │
                         │   │    │  reshape 9 × ℝ⁶⁴ → ℝ⁵⁷⁶  │           │         │
                         │   │    └───────────────────────────┘           │         │
                         │   │                                            │         │
                         │   │  Pattern repeats 4×:                       │         │
                         │   │    [Res][Res][Res][Res][Attn]              │         │
                         │   │    [Res][Res][Res][Res][Attn]              │         │
                         │   │    [Res][Res][Res][Res][Attn]              │         │
                         │   │    [Res][Res][Res][Res][Attn]              │         │
                         │   └────────────────────────────────────────────┘         │
                         │            │  ℝ⁵⁷⁶                                      │
                         │            ▼                                            │
                         │   ┌──────────────────┐                                  │
                         │   │  LayerNorm(576)   │                                  │
                         │   │  Linear(576→256)  │                                  │
                         │   └────────┬─────────┘                                  │
                         │            │  ℝ²⁵⁶  (latent)                            │
                         └────────────┼────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
     │  head_U          │    │  head_V          │    │  head_W          │
     │  Linear(256→N×9) │    │  Linear(256→N×9) │    │  Linear(256→9×N) │
     └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
              │ U ∈ ℝᴺˣ⁹            │ V ∈ ℝᴺˣ⁹            │ W ∈ ℝ⁹ˣᴺ
              │                      │                      │
══════════════╪══════════════════════╪══════════════════════╪══════════════
              │      BILINEAR BOTTLENECK (anti-cheat barrier)│
              │      No encoder output crosses this line     │
              │                      │                      │
              ▼                      ▼                      │
     ┌─────────────────┐    ┌─────────────────┐            │
     │   p = U @ A      │    │   q = V @ B      │            │
     │   ℝᴺ             │    │   ℝᴺ             │            │
     └────────┬────────┘    └────────┬────────┘            │
              │                      │                      │
              └──────────┬───────────┘                      │
                         │                                  │
                         ▼                                  │
                ┌─────────────────┐                         │
                │   m = p ⊙ q      │                         │
                │   ℝᴺ (rank-1     │                         │
                │   bilinear/ch)   │                         │
                └────────┬────────┘                         │
                         │                                  │
                         ▼                                  │
                ┌─────────────────┐                         │
                │   Ĉ = W @ m     │◄────────────────────────┘
                │   ℝ⁹ (no bias)  │
                └────────┬────────┘
                         │
                         ▼
                    Ĉ ∈ ℝ⁹
              (predicted C = A×B)
```

## Key Properties

| Property | Value |
|---|---|
| Total params (N=19) | 11.0M |
| Encoder params | 10.9M |
| Head params | 131k |
| Encoder width | 576 (= 9 tokens × 64 dims) |
| Latent dim | 256 |
| ResBlocks | 16 |
| AttnBlocks | 4 (after every 4th ResBlock) |
| Attention heads | 4 × 16 dim per token |
| Dropout | 0.1 everywhere |

## Anti-Cheat Guarantee

The encoder output (latent ∈ ℝ²⁵⁶) generates U, V, W but **never enters the data path**.
Raw A and B pass through the bilinear bottleneck using U, V, W as weights.
Each channel k computes: `mₖ = (uₖ · A)(vₖ · B)` — a rank-1 bilinear form.
N channels = bilinear rank N. If the network achieves low error at rank N, it has found an honest rank-N decomposition.

## Attention Interpretation

The 9 tokens in each AttnBlock correspond to the 9 positions of a 3×3 matrix.
Attention can learn structural relationships like "entry (i,j) should attend to row-i and column-j entries" — which is literally the structure of matrix multiplication.
