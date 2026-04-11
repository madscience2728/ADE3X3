"""
model.py — The Keth-Varai Machine.

Architecture (see ARCHITECTURE.md):

  Input: A ∈ R^9, B ∈ R^9

  Encoder:
    φ_A = f_enc(A)  ∈ R^d
    φ_B = g_enc(B)  ∈ R^d

  Mix:
    ã = cat(φ_A, A)  ∈ R^{d+9}
    b̃ = cat(φ_B, B)  ∈ R^{d+9}

  Bilinear Bottleneck (N channels):
    m_k = (u_k · ã)(v_k · b̃)    k = 1..N
    U ∈ R^{N × (d+9)},  V ∈ R^{N × (d+9)}

  Decoder:
    Ĉ = W · m  ∈ R^9
    W ∈ R^{9 × N}

The bilinear structure is exact: no activations inside the bottleneck.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP: R^9 → R^d"""

    def __init__(self, d: int, depth: int = 3, width: int = 128, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(9, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(width, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KethVaraiMachine(nn.Module):
    """
    Full pipeline: (A, B) → Ĉ ≈ A·B
    with a learned-basis bilinear bottleneck of rank N.
    """

    def __init__(self, N: int, d: int = 64, encoder_depth: int = 3, encoder_width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.N = N
        self.d = d
        mixed_dim = d + 9  # size of ã and b̃

        # Separate encoders for A-arm and B-arm (structurally different roles)
        self.enc_a = Encoder(d, depth=encoder_depth, width=encoder_width, dropout=dropout)
        self.enc_b = Encoder(d, depth=encoder_depth, width=encoder_width, dropout=dropout)

        # Bilinear bottleneck: U projects ã, V projects b̃
        # Each row u_k / v_k is a linear functional on the mixed representation
        self.U = nn.Linear(mixed_dim, N, bias=False)
        self.V = nn.Linear(mixed_dim, N, bias=False)

        # Decoder: linear recombination of N scalar channel outputs → 9 outputs
        self.W = nn.Linear(N, 9, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Small init on bottleneck to avoid scale explosion
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.normal_(self.W.weight, std=0.1)
        nn.init.zeros_(self.W.bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A, B: (batch, 9)
        Returns Ĉ: (batch, 9)
        """
        # Encode
        phi_a = self.enc_a(A)   # (batch, d)
        phi_b = self.enc_b(B)   # (batch, d)

        # Mix latent with raw input
        a_tilde = torch.cat([phi_a, A], dim=1)  # (batch, d+9)
        b_tilde = torch.cat([phi_b, B], dim=1)  # (batch, d+9)

        # Bilinear bottleneck: element-wise product of two linear projections
        p = self.U(a_tilde)  # (batch, N)
        q = self.V(b_tilde)  # (batch, N)
        m = p * q            # (batch, N)  — each channel is a rank-1 bilinear form

        # Decode
        return self.W(m)     # (batch, 9)

    def channel_norms(self) -> torch.Tensor:
        """
        For each channel k, return ||u_k||_2 * ||v_k||_2 * ||w_k||_2
        as a proxy for channel importance. Useful for pruning analysis.
        """
        u_norms = self.U.weight.norm(dim=1)   # (N,)
        v_norms = self.V.weight.norm(dim=1)   # (N,)
        w_norms = self.W.weight.norm(dim=0)   # (N,)
        return u_norms * v_norms * w_norms

    def extra_repr(self) -> str:
        return f"N={self.N}, d={self.d}, mixed_dim={self.d + 9}"
