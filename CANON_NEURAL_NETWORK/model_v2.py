"""
model_v2.py — Binary Keth-Varai Machine (H-Bomb variant).

All four anti-cheat mechanisms enforced:

  1. BIT SHUFFLE: Inputs grouped by bit position, not by entry.
     The network sees [bit0 of all 18 entries, bit1 of all 18, ...].
     Reconstructing a single float requires gathering from n_bits positions.

  2. LOCAL FIRST LAYER: Each group of neurons only sees bits at the same
     position across entries (18 bits per group). Cross-entry logic happens
     before within-entry reconstruction can.

  3. BOTTLENECK < 18: After the local layer, all groups are concatenated
     and squeezed through a channel narrower than 18. The network literally
     cannot reconstruct all 18 floats — forced to compress at the bit level.

  4. BINARY ACTIVATIONS + STE: Every hidden unit uses sign activation with
     straight-through estimator. The network is a Boolean circuit — it cannot
     form float-valued intermediates anywhere in the encoder.

The bilinear bottleneck (U, V, W) stays real-valued — that's the honest
structural constraint, and U/V/W must be precise to represent a real
decomposition of matrix multiplication.

Architecture:

  Input: A ∈ R^9, B ∈ R^9
    → quantize to n_bits fixed-point each → 18 * n_bits binary values
    → shuffle to group by bit position

  Binary Encoder:
    → local first layer: n_bits groups of 18 → group_width each (binary)
    → concat: n_bits * group_width
    → bottleneck: → bottleneck_dim (binary), bottleneck_dim < 18
    → binary hidden layers (LayerNorm → Linear → Sign_STE) × depth
    → real output projection → latent_dim (NO sign activation)

  Hyper-heads: latent → U, V, W  (same as v1)
  Bilinear Bottleneck: p=Ua, q=Vb, m=p⊙q, Ĉ=Wm  (same as v1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Straight-Through Estimator for sign binarization
# ---------------------------------------------------------------------------

class SignSTE(torch.autograd.Function):
    """Sign activation: forward → {-1, +1}, backward → straight-through."""

    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float() * 2 - 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


# ---------------------------------------------------------------------------
# Binary building blocks
# ---------------------------------------------------------------------------

class BinaryLinear(nn.Module):
    """Linear → LayerNorm → Sign_STE."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sign_ste(self.norm(self.linear(x)))


class LocalBitLayer(nn.Module):
    """
    Local first layer: independently processes each bit-position group.

    Input shape: (batch, n_bits, 18) — bits grouped by position.
    Output shape: (batch, n_bits * group_width) — flattened.

    Each of n_bits groups: Linear(18 → group_width) → LayerNorm → Sign_STE.
    No cross-group interaction — forces bit-level processing.
    """

    def __init__(self, n_bits: int, n_entries: int, group_width: int):
        super().__init__()
        self.n_bits = n_bits
        self.n_entries = n_entries
        self.group_width = group_width
        # One linear+norm per bit position
        self.groups = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_entries, group_width),
                nn.LayerNorm(group_width),
            )
            for _ in range(n_bits)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bits, n_entries)
        outs = []
        for i, group in enumerate(self.groups):
            outs.append(sign_ste(group(x[:, i, :])))  # (batch, group_width)
        return torch.cat(outs, dim=1)  # (batch, n_bits * group_width)


# ---------------------------------------------------------------------------
# Input encoding: float → fixed-point binary → bit-shuffled
# ---------------------------------------------------------------------------

class BinaryInputEncoder(nn.Module):
    """
    Converts 18 float entries to bit-shuffled binary representation.

    Each float is clamped to [lo, hi], quantized to n_bits fixed-point,
    then bits are grouped by position (bit 0 of all entries first, then
    bit 1, etc.) to prevent trivial float reconstruction.

    Forward pass: quantize + binarize (non-differentiable but inputs are
    data, not parameters — no gradient needed through this layer).
    """

    def __init__(self, n_entries: int = 18, n_bits: int = 8,
                 lo: float = -4.0, hi: float = 4.0):
        super().__init__()
        self.n_entries = n_entries
        self.n_bits = n_bits
        self.register_buffer('lo', torch.tensor(lo))
        self.register_buffer('hi', torch.tensor(hi))
        # Bit extraction masks: [1, 2, 4, 8, ...]
        self.register_buffer('bit_masks',
                             torch.tensor([1 << b for b in range(n_bits)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_entries) floats
        Returns: (batch, n_bits, n_entries) binary in {-1, +1}
        """
        # Quantize to integer range [0, 2^n_bits - 1]
        x_clamped = x.clamp(self.lo.item(), self.hi.item())
        x_norm = (x_clamped - self.lo) / (self.hi - self.lo)  # [0, 1]
        max_val = (1 << self.n_bits) - 1
        x_int = (x_norm * max_val).round().long()  # (batch, n_entries)

        # Extract bits: (batch, n_entries, n_bits)
        bits = (x_int.unsqueeze(-1) & self.bit_masks.unsqueeze(0).unsqueeze(0)) > 0

        # Transpose to (batch, n_bits, n_entries) — grouped by bit position
        bits = bits.permute(0, 2, 1).float()

        # Map {0, 1} → {-1, +1} for sign-activation compatibility
        return bits * 2 - 1


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BinaryKethVarai(nn.Module):
    """
    H-Bomb variant: binary encoder with all four anti-cheat mechanisms.

    The encoder is a Boolean circuit (sign activations throughout).
    The bilinear bottleneck is real-valued (structural honesty preserved).
    """

    def __init__(
        self,
        N: int,
        latent_dim: int = 128,
        n_bits: int = 8,
        group_width: int = 16,
        bottleneck_dim: int = 12,
        binary_depth: int = 6,
        binary_width: int = 128,
        input_lo: float = -4.0,
        input_hi: float = 4.0,
    ):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim
        self.n_bits = n_bits
        self.bottleneck_dim = bottleneck_dim

        # Input encoding: float → binary bits, shuffled by position
        self.input_encoder = BinaryInputEncoder(
            n_entries=18, n_bits=n_bits, lo=input_lo, hi=input_hi
        )

        # Anti-cheat 2: Local first layer (per-bit-position groups)
        self.local_layer = LocalBitLayer(n_bits, 18, group_width)

        # Anti-cheat 3: Bottleneck below reconstruction capacity
        local_out_dim = n_bits * group_width
        self.bottleneck = BinaryLinear(local_out_dim, bottleneck_dim)

        # Anti-cheat 4: Binary hidden layers (Sign_STE throughout)
        layers = [BinaryLinear(bottleneck_dim, binary_width)]
        for _ in range(binary_depth - 1):
            layers.append(BinaryLinear(binary_width, binary_width))
        self.binary_trunk = nn.Sequential(*layers)

        # Transition to real: final projection to latent_dim (NO sign activation)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(binary_width),
            nn.Linear(binary_width, latent_dim),
        )

        # Hyper-heads: latent → U, V, W (identical to v1)
        head_hidden = latent_dim * 2
        self.head_U = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, N * 9),
        )
        self.head_V = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, N * 9),
        )
        self.head_W = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 9 * N),
        )

        self._init_weights()

    def _init_weights(self):
        for head in [self.head_U, self.head_V, self.head_W]:
            last = head[-1]
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)

    def _encode(self, AB: torch.Tensor) -> torch.Tensor:
        """Full binary encoder pipeline: float input → real latent."""
        A = AB[:, :9]
        B = AB[:, 9:]

        # Binarize both inputs, concat along entry dimension
        # A_bits, B_bits: (batch, n_bits, 9) each
        A_bits = self.input_encoder(A)  # (batch, n_bits, 9)
        B_bits = self.input_encoder(B)  # (batch, n_bits, 9)

        # Concat entries: (batch, n_bits, 18) — bit-position grouping preserved
        bits = torch.cat([A_bits, B_bits], dim=2)

        # Local layer: per-bit-position processing → (batch, n_bits * group_width)
        h = self.local_layer(bits)

        # Bottleneck: squeeze below float-reconstruction capacity
        h = self.bottleneck(h)

        # Binary trunk: deep Boolean circuit
        h = self.binary_trunk(h)

        # Transition to real-valued latent
        return self.to_latent(h)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A, B: (batch, 9)
        Returns Ĉ: (batch, 9)
        """
        batch = A.shape[0]

        latent = self._encode(torch.cat([A, B], dim=1))

        U = self.head_U(latent).view(batch, self.N, 9)
        V = self.head_V(latent).view(batch, self.N, 9)
        W = self.head_W(latent).view(batch, 9, self.N)

        p = torch.bmm(U, A.unsqueeze(2)).squeeze(2)
        q = torch.bmm(V, B.unsqueeze(2)).squeeze(2)
        m = p * q

        C_hat = torch.bmm(W, m.unsqueeze(2)).squeeze(2)
        return C_hat

    def channel_norms(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        batch = A.shape[0]
        with torch.no_grad():
            latent = self._encode(torch.cat([A, B], dim=1))
            U = self.head_U(latent).view(batch, self.N, 9)
            V = self.head_V(latent).view(batch, self.N, 9)
            W = self.head_W(latent).view(batch, 9, self.N)
            u_norms = U.norm(dim=2)
            v_norms = V.norm(dim=2)
            w_norms = W.norm(dim=1)
            return u_norms * v_norms * w_norms

    def extra_repr(self) -> str:
        return (f"N={self.N}, latent_dim={self.latent_dim}, "
                f"n_bits={self.n_bits}, bottleneck_dim={self.bottleneck_dim}")
