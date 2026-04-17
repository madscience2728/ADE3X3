"""
model.py — The Keth-Varai Machine (Hypernetwork variant).

Architecture (see ARCHITECTURE.md):

  Input: A ∈ R^9, B ∈ R^9

  Encoder (joint — sees both A and B):
    latent = encoder(cat(A, B))  ∈ R^latent_dim

  Hyper-heads (latent → weight matrices for bilinear layer):
    U = head_U(latent)  ∈ R^{N × 9}     (reshaped)
    V = head_V(latent)  ∈ R^{N × 9}     (reshaped)
    W = head_W(latent)  ∈ R^{9 × N}     (reshaped)

  Bilinear Bottleneck (N channels, input-dependent weights):
    p = U @ A    ∈ R^N       (linear projection, no activation)
    q = V @ B    ∈ R^N       (linear projection, no activation)
    m = p ⊙ q    ∈ R^N       (element-wise — each channel is rank-1 bilinear)

  Decoder:
    Ĉ = W @ m   ∈ R^9       (linear, no bias)

The encoder output NEVER enters the data path. It only generates weights.
A and B interact ONLY through the bilinear bottleneck.
N is an honest rank count: N rank-1 bilinear forms, architecturally enforced.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → Add."""

    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttnBlock(nn.Module):
    """Pre-norm self-attention residual block over n_tokens tokens."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 n_tokens: int = 9):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_model // self.n_tokens
        assert d_model % self.n_tokens == 0, f"encoder_width must be divisible by {n_tokens}, got {d_model}"
        self.norm = nn.LayerNorm(self.d_token)
        self.attn = nn.MultiheadAttention(
            self.d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) → (batch, 9, d_token)
        batch = x.shape[0]
        tokens = x.view(batch, self.n_tokens, self.d_token)
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + self.dropout(attn_out)
        return tokens.view(batch, -1)  # back to (batch, d_model)


class KethVaraiMachine(nn.Module):
    """
    Hypernetwork: encoder(A, B) → tuning knobs (U, V, W),
    then A and B pass through the bilinear bottleneck defined by those knobs.

    The encoder learns the structure of the problem.
    The bottleneck enforces honest bilinear rank N.
    The encoder cannot cheat — its output never touches the data path.
    """

    def __init__(
        self,
        N: int,
        latent_dim: int = 256,
        encoder_depth: int = 16,
        encoder_width: int = 576,
        dropout: float = 0.1,
        n_heads: int = 4,
        attn_every: int = 4,
        expanded_products: bool = True,
        linear_heads: bool = False,
        head_depth: int = 1,
        head_width: int = 256,
    ):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim
        self.expanded_products = expanded_products
        self.linear_heads = linear_heads

        # Input dimension: 81 bilinear products (expanded) or 18 raw entries
        if expanded_products:
            input_dim = 81  # A[i,s] * B[t,j] for all i,s,t,j ∈ {0,1,2}
            n_tokens = 81
        else:
            input_dim = 18
            n_tokens = 9
        self.n_tokens = n_tokens

        # Joint encoder: input projection + residual blocks + output projection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_width),
            nn.GELU(),
        )

        # Interleaved ResBlocks + attention: 1 AttnBlock every `attn_every` ResBlocks
        blocks = []
        for i in range(encoder_depth):
            blocks.append(ResBlock(encoder_width, dropout))
            if attn_every > 0 and (i + 1) % attn_every == 0:
                blocks.append(AttnBlock(encoder_width, n_heads=n_heads, dropout=dropout,
                                        n_tokens=n_tokens))

        self.encoder_blocks = nn.Sequential(*blocks)

        self.encoder_head = nn.Sequential(
            nn.LayerNorm(encoder_width),
            nn.Linear(encoder_width, latent_dim),
        )

        # Hyper-heads: latent → weight matrices for the bilinear layer
        # U ∈ R^{N×9}: projects A into N channels
        # V ∈ R^{N×9}: projects B into N channels
        # W ∈ R^{9×N}: decodes N channel outputs to 9 output entries
        self.head_U = self._build_head(latent_dim, N * 9, head_depth, head_width, linear_heads)
        self.head_V = self._build_head(latent_dim, N * 9, head_depth, head_width, linear_heads)
        self.head_W = self._build_head(latent_dim, 9 * N, head_depth, head_width, linear_heads)

        self._init_weights()

    @staticmethod
    def _build_head(in_dim: int, out_dim: int, depth: int, width: int,
                    linear: bool) -> nn.Module:
        """Build a hyper-head with configurable depth.
        depth=0 or linear=True → single Linear (no nonlinearity).
        depth=1 → Linear→GELU→Linear (shallow nonlinear, default nano).
        depth=2+ → deep MLP.
        """
        if linear or depth == 0:
            return nn.Linear(in_dim, out_dim)
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, out_dim))
        return nn.Sequential(*layers)

    def _compute_products(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute all 81 bilinear products A[i,s] * B[t,j].
        A, B: (batch, 9) — flattened 3×3 matrices.
        Returns: (batch, 81) — products ordered as (i,s,t,j) in row-major.

        The 27 products where s==t are the Sigma (live) coordinates.
        The 54 products where s!=t are the Delta (dead-X) coordinates.
        The attention learns which products to combine and cancel.
        """
        # A[i,s] = A_flat[3*i + s], B[t,j] = B_flat[3*t + j]
        # Product[i,s,t,j] = A[3*i+s] * B[3*t+j]
        # Reshape: A → (batch, 9, 1), B → (batch, 1, 9) → outer product (batch, 9, 9) → flatten to 81
        return (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, 81)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Input projection → residual blocks → output projection."""
        if self.expanded_products:
            # x is cat(A, B) with shape (batch, 18); compute 81 products
            A, B = x[:, :9], x[:, 9:]
            x = self._compute_products(A, B)
        h = self.encoder(x)
        h = self.encoder_blocks(h)
        return self.encoder_head(h)

    def _init_weights(self):
        for head in [self.head_U, self.head_V, self.head_W]:
            if isinstance(head, nn.Linear):
                # Pure linear head — small init
                nn.init.normal_(head.weight, std=0.01)
                nn.init.zeros_(head.bias)
            else:
                # Sequential — small init on last layer
                last = head[-1]
                nn.init.normal_(last.weight, std=0.01)
                nn.init.zeros_(last.bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A, B: (batch, 9)
        Returns Ĉ: (batch, 9)
        """
        batch = A.shape[0]

        # Encoder: learn the tuning knobs for this (A, B) pair
        latent = self._encode(torch.cat([A, B], dim=1))  # (batch, latent_dim)

        # Generate weight matrices (per-sample)
        U = self.head_U(latent).view(batch, self.N, 9)   # (batch, N, 9)
        V = self.head_V(latent).view(batch, self.N, 9)   # (batch, N, 9)
        W = self.head_W(latent).view(batch, 9, self.N)   # (batch, 9, N)

        # Bilinear bottleneck: raw A and B only — no latent in the data path
        p = torch.bmm(U, A.unsqueeze(2)).squeeze(2)  # (batch, N)
        q = torch.bmm(V, B.unsqueeze(2)).squeeze(2)  # (batch, N)
        m = p * q                                      # (batch, N) — rank-1 bilinear per channel

        # Decode: W @ m, no bias
        C_hat = torch.bmm(W, m.unsqueeze(2)).squeeze(2)  # (batch, 9)
        return C_hat

    def channel_norms(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Per-channel importance for a given batch: ||u_k|| * ||v_k|| * ||w_k||.
        Returns (batch, N).
        """
        batch = A.shape[0]
        with torch.no_grad():
            latent = self._encode(torch.cat([A, B], dim=1))
            U = self.head_U(latent).view(batch, self.N, 9)
            V = self.head_V(latent).view(batch, self.N, 9)
            W = self.head_W(latent).view(batch, 9, self.N)
            u_norms = U.norm(dim=2)           # (batch, N)
            v_norms = V.norm(dim=2)           # (batch, N)
            w_norms = W.norm(dim=1)           # (batch, N)
            return u_norms * v_norms * w_norms

    def extra_repr(self) -> str:
        return f"N={self.N}, latent_dim={self.latent_dim}"
