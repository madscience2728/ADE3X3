"""
model_kolmogorov.py -- The Bundle Machine.

Architecture derived from the LeCun-Hinton-Strassen-Noether analysis:

  Stage 1: Strategy Encoder (manifold decoder)
    Input: (A, B) -> 81 bilinear products
    Output: strategy code s in R^K_s (information-bottlenecked)
    The bottleneck minimizes I(s; A,B) subject to I(s; C) being sufficient.

  Stage 2: Execution Head (section evaluator)
    Input: s (strategy code only -- NO direct access to A, B)
    Output: U(s), V(s), W(s) -- the bilinear form parameters

  Stage 3: Bilinear Executor
    Input: A, B, U, V, W
    Output: C_hat = W @ ((U @ A) * (V @ B))

Key insight from the N=1 analysis:
  W is a SECTION of a bundle over the input manifold.
  The encoder computes which section to evaluate.
  The strategy code is the fiber coordinate.

Objective:
  min E[||C - C_hat||^2]
    + lambda_s * K_s          (strategy complexity: dim of bottleneck)
    + lambda_ib * I(s; A,B)   (information bottleneck on strategy)
    + lambda_n * N             (execution rank)

The IB penalty is implemented as a variational bound (VIB):
  s = mu(x) + sigma(x) * epsilon,  epsilon ~ N(0, I)
  KL(q(s|x) || p(s)) as the IB penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class BundleMachine(nn.Module):
    """
    Two-stage architecture with information bottleneck on strategy code.

    Stage 1: Encoder -> bottlenecked strategy code s
    Stage 2: Execution heads generate U, V, W from s alone
    Stage 3: Bilinear executor: C_hat = W @ ((U@A) * (V@B))

    The strategy code s never sees raw A, B -- only the encoder's
    compressed representation. This enforces separation between
    "what computation to do" and "doing it."
    """

    def __init__(
        self,
        N: int = 1,
        strategy_dim: int = 9,       # K_s: bottleneck dimension
        encoder_width: int = 192,
        encoder_depth: int = 2,
        head_width: int = 256,
        head_depth: int = 1,
        dropout: float = 0.1,
        stochastic: bool = True,      # Use VIB (variational information bottleneck)
    ):
        super().__init__()
        self.N = N
        self.strategy_dim = strategy_dim
        self.stochastic = stochastic

        # Stage 1: Encoder (sees 81 bilinear products -> strategy code)
        input_dim = 81  # A[i,s] * B[t,j]
        layers = [nn.Linear(input_dim, encoder_width), nn.GELU()]
        for _ in range(encoder_depth):
            layers.append(ResBlock(encoder_width, dropout))
        layers.append(nn.LayerNorm(encoder_width))
        self.encoder = nn.Sequential(*layers)

        # VIB: encoder -> mu, log_var -> reparametrize -> s
        if stochastic:
            self.fc_mu = nn.Linear(encoder_width, strategy_dim)
            self.fc_logvar = nn.Linear(encoder_width, strategy_dim)
        else:
            self.fc_s = nn.Linear(encoder_width, strategy_dim)

        # Stage 2: Execution heads (strategy code -> U, V, W)
        # These see ONLY s -- no A, B leakage
        self.head_U = self._build_head(strategy_dim, N * 9, head_depth, head_width)
        self.head_V = self._build_head(strategy_dim, N * 9, head_depth, head_width)
        self.head_W = self._build_head(strategy_dim, 9 * N, head_depth, head_width)

        self._init_weights()

    @staticmethod
    def _build_head(in_dim, out_dim, depth, width):
        if depth == 0:
            return nn.Linear(in_dim, out_dim)
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, out_dim))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for head in [self.head_U, self.head_V, self.head_W]:
            last = head[-1] if isinstance(head, nn.Sequential) else head
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
        if self.stochastic:
            nn.init.zeros_(self.fc_mu.weight)
            nn.init.zeros_(self.fc_mu.bias)
            nn.init.constant_(self.fc_logvar.bias, -2.0)  # start with small variance

    def _compute_products(self, A, B):
        return (A.unsqueeze(2) * B.unsqueeze(1)).view(-1, 81)

    def encode(self, A, B):
        """
        Encode (A, B) -> strategy code s.
        Returns: s, kl_loss
          s: (batch, strategy_dim)
          kl_loss: scalar KL divergence (0 if deterministic)
        """
        x = self._compute_products(A, B)
        h = self.encoder(x)

        if self.stochastic:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            # Reparametrization trick
            if self.training:
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                s = mu + std * eps
            else:
                s = mu
            # KL divergence: KL(N(mu, sigma^2) || N(0, I))
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            return s, kl
        else:
            s = self.fc_s(h)
            return s, torch.tensor(0.0, device=A.device)

    def forward(self, A, B):
        """
        Full forward: encode -> generate weights -> bilinear execute.
        Returns: C_hat, kl_loss
        """
        batch = A.shape[0]
        s, kl_loss = self.encode(A, B)

        # Stage 2: Generate bilinear form from strategy code alone
        U = self.head_U(s).view(batch, self.N, 9)
        V = self.head_V(s).view(batch, self.N, 9)
        W = self.head_W(s).view(batch, 9, self.N)

        # Stage 3: Execute bilinear form
        p = torch.bmm(U, A.unsqueeze(2)).squeeze(2)  # (batch, N)
        q = torch.bmm(V, B.unsqueeze(2)).squeeze(2)  # (batch, N)
        m = p * q
        C_hat = torch.bmm(W, m.unsqueeze(2)).squeeze(2)  # (batch, 9)

        return C_hat, kl_loss

    def channel_norms(self, A, B):
        batch = A.shape[0]
        with torch.no_grad():
            s, _ = self.encode(A, B)
            U = self.head_U(s).view(batch, self.N, 9)
            V = self.head_V(s).view(batch, self.N, 9)
            W = self.head_W(s).view(batch, 9, self.N)
            u_norms = U.norm(dim=2)
            v_norms = V.norm(dim=2)
            w_norms = W.norm(dim=1)
            return u_norms * v_norms * w_norms
