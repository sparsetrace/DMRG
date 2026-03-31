# mhdm.py
# Minimal PyTorch MHDM blocks for ViT integration.
# - No RoPE
# - No beta / temperature
# - Symmetric DM-style logits (distance kernel)
# - Channel attention is "transpose + same mechanism" (defaults to 1 head)

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def _to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    (B, N, D) -> (B, H, N, d)
    """
    B, N, D = x.shape
    if D % num_heads != 0:
        raise ValueError(f"dim={D} not divisible by num_heads={num_heads}")
    d = D // num_heads
    return x.view(B, N, num_heads, d).transpose(1, 2)  # (B, H, N, d)


def _from_heads(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, N, d) -> (B, N, D)
    """
    B, H, N, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, N, H * d)


class MHDM(nn.Module):
    """
    Token diffusion-style attention with tied Q=K projection.

    Input:  x (B, N, D)
    Output: y (B, N, D)

    Projections:
      qk = W_qk x
      v  = W_v  x

    Symmetric distance logits:
      dot_ij  = <qk_i, qk_j>
      logits_ij = 2*dot_ij - ||qk_i||^2 - ||qk_j||^2 = -||qk_i - qk_j||^2

    This is symmetric in (i, j).
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qk_proj = nn.Linear(dim, dim, bias=bias)  # tied Q=K
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qk = self.qk_proj(x)  # (B, N, D)
        v = self.v_proj(x)    # (B, N, D)

        qk = _to_heads(qk, self.num_heads)  # (B, H, N, d)
        v = _to_heads(v, self.num_heads)    # (B, H, N, d)

        # dot: (B, H, N, N)
        dot = torch.matmul(qk, qk.transpose(-1, -2))

        # norms: (B, H, N)
        q2 = (qk * qk).sum(dim=-1)

        # symmetric distance logits: 2*dot - q2_i - q2_j
        logits = 2.0 * dot - q2.unsqueeze(-1) - q2.unsqueeze(-2)  # (B, H, N, N)

        attn = F.softmax(logits, dim=-1)
        w = torch.matmul(attn, v)  # (B, H, N, d)

        out = _from_heads(w)       # (B, N, D)
        out = self.out_proj(out)   # (B, N, D)
        return out


class MHDM_(nn.Module):
    """
    Channel version: transpose token/channel axes and run the same MHDM mechanism.

    Input:  x (B, N, D)
    Process:
      y = x^T -> (B, D, N)
      attention over "sequence" length D (channels), embedding dim N
      y -> (B, D, N)
      transpose back -> (B, N, D)

    IMPORTANT:
      For ViT at 224 with cls token: N = 197.
      197 is prime-ish for head splitting, so default num_heads=1 here to keep it simple.

    If you *really* want multiple heads, you must ensure N % num_heads == 0.
    """

    def __init__(self, num_heads: int = 1, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.bias = bias

        # We don't know N (token length) until forward().
        # We'll create projections lazily with in_features=N, out_features=N.
        self.qk_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None

    def _maybe_init(self, N: int, device: torch.device, dtype: torch.dtype):
        if self.qk_proj is None:
            self.qk_proj = nn.Linear(N, N, bias=self.bias).to(device=device, dtype=dtype)
            self.v_proj = nn.Linear(N, N, bias=self.bias).to(device=device, dtype=dtype)
            self.out_proj = nn.Linear(N, N, bias=self.bias).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        self._maybe_init(N, x.device, x.dtype)
        assert self.qk_proj is not None and self.v_proj is not None and self.out_proj is not None

        # (B, N, D) -> (B, D, N)
        y = x.transpose(1, 2)

        # projections along last dim (N)
        qk = self.qk_proj(y)  # (B, D, N)
        v = self.v_proj(y)    # (B, D, N)

        # treat channels as sequence: seq_len = D, embed_dim = N
        qk = _to_heads(qk, self.num_heads)  # (B, H, D, n)
        v = _to_heads(v, self.num_heads)    # (B, H, D, n)

        dot = torch.matmul(qk, qk.transpose(-1, -2))      # (B, H, D, D)
        q2 = (qk * qk).sum(dim=-1)                        # (B, H, D)
        logits = 2.0 * dot - q2.unsqueeze(-1) - q2.unsqueeze(-2)  # symmetric (B,H,D,D)

        attn = F.softmax(logits, dim=-1)
        w = torch.matmul(attn, v)                         # (B, H, D, n)

        y = _from_heads(w)                                # (B, D, N)
        y = self.out_proj(y)                              # (B, D, N)

        return y.transpose(1, 2)                          # (B, N, D)

class ChannelDiffusion(nn.Module):
    """
    Learns a feature-space transform W (D->D) and computes diffusion logits
    as negative squared Euclidean distances BETWEEN CHANNEL VECTORS across tokens.

    Input:  x (B, N, D)
    Output: y (B, N, D)

    Params depend on D, not on N.
    """
    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False, temperature: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads

        # This is your W (per token): Q = X @ W_qk
        # It's D->D, independent of N.
        self.qk_proj = nn.Linear(dim, dim, bias=bias)  # tied Q=K projection like your MHDM
        self.v_proj  = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        # Optional per-head temperature (stabilizes softmax)
        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1)) if temperature else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qk = self.qk_proj(x)  # (B, N, D)  => Q = R W
        v  = self.v_proj(x)   # (B, N, D)

        # Channels-as-sequence: transpose to (B, D, N)
        qk = qk.transpose(1, 2)   # (B, D, N)
        v  = v.transpose(1, 2)    # (B, D, N)

        # Head split along channels: (B, H, Dh, N)
        qk = qk.reshape(B, self.num_heads, self.dh, N)
        v  = v.reshape(B, self.num_heads, self.dh, N)

        # Compute logits_{c,c'} = -||qk_c - qk_c'||^2
        # Using: -||a-b||^2 = 2 a·b - ||a||^2 - ||b||^2
        dot = torch.matmul(qk, qk.transpose(-1, -2))         # (B, H, Dh, Dh)
        q2  = (qk * qk).sum(dim=-1)                          # (B, H, Dh)
        logits = 2.0 * dot - q2.unsqueeze(-1) - q2.unsqueeze(-2)  # (B,H,Dh,Dh)

        # Scale for stability (since dot sums over N)
        logits = logits / math.sqrt(N)
        if self.tau is not None:
            logits = logits * self.tau

        attn = F.softmax(logits, dim=-1)                     # (B, H, Dh, Dh)

        # Diffuse values across channels: (B,H,Dh,Dh) @ (B,H,Dh,N) -> (B,H,Dh,N)
        w = torch.matmul(attn, v)                            # (B, H, Dh, N)

        # Merge heads back: (B, D, N) -> (B, N, D)
        out = w.reshape(B, D, N).transpose(1, 2).contiguous()
        out = self.out_proj(out)
        return out

class Diffusion_BlockX(nn.Module):
    """
    Minimal ViT-friendly block:
      x = x + MHDM(LN(x))
      x = x + MHDM_(LN(x))

    No FFN/MLP branch (matching your current "attention + channel attention" idea).
    """

    def __init__(self, dim: int, num_heads: int, channel_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHDM(dim=dim, num_heads=num_heads, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.ch_attn = MHDM_(num_heads=channel_heads, bias=False)

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ch_attn(self.norm2(x)))
        return x

class Diffusion_Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, channel_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHDM(dim=dim, num_heads=num_heads, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.ch_attn = ChannelDiffusion(dim=dim, num_heads=channel_heads, bias=False, temperature=True)

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Optional but often stabilizes training when swapping blocks:
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x))) * self.gamma1
        x = x + self.drop(self.ch_attn(self.norm2(x))) * self.gamma2
        return x

#if __name__ == "__main__":
#    torch.manual_seed(0)
#    B, N, D = 2, 197, 768
#    x = torch.randn(B, N, D)#
#
#    blk = Diffusion_Block(dim=D, num_heads=12, channel_heads=1)
#    y = blk(x)
#    print("in:", x.shape, "out:", y.shape)
