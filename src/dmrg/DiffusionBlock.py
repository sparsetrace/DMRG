"""
DiffusionBlock.py
=================

A ViT/DMRG-compatible diffusion-style transformer block.

This block is intended to be used as a drop-in replacement block for the
current DMRG.py workflow. It preserves the external ViT hidden-state shape:

    input:  (B, N, D)
    output: (B, N, D)

Design goals
------------
- Compatible with Hugging Face ViT residual streams
- Compatible with the current DMRG block-construction API
- No lazy parameter creation based on token length
- Accepts both:
    - hidden_size / num_attention_heads
    - dim / num_heads
- Stable during block swapping via small learnable residual scales

Structure
---------
x = x + gamma1 * TokenDiffusion( LN(x) )
x = x + gamma2 * ChannelDiffusion( LN(x) )

Notes
-----
- This is not a standard ViT block because it does not include an FFN/MLP branch.
- It is closer to your diffusion-style block idea from mhdm.py, adapted to be
  friendlier to DMRG integration.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    (B, N, D) -> (B, H, N, d)
    """
    b, n, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(f"hidden dimension {d_model} is not divisible by num_heads={num_heads}")
    d_head = d_model // num_heads
    return x.view(b, n, num_heads, d_head).transpose(1, 2).contiguous()


def _from_heads(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, N, d) -> (B, N, D)
    """
    b, h, n, d_head = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * d_head)


class TokenDiffusion(nn.Module):
    """
    Token diffusion-style attention with tied Q=K projection.

    Input:  x (B, N, D)
    Output: y (B, N, D)

    Logits are symmetric negative squared distances in projected space:
        logits_ij = -||q_i - q_j||^2
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        bias: bool = False,
        temperature: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads

        self.qk_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1)) if temperature else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        qk = self.qk_proj(x)
        v = self.v_proj(x)

        qk = _to_heads(qk, self.num_heads)  # (B, H, N, d_h)
        v = _to_heads(v, self.num_heads)    # (B, H, N, d_h)

        dot = torch.matmul(qk, qk.transpose(-1, -2))  # (B, H, N, N)
        q2 = (qk * qk).sum(dim=-1)                    # (B, H, N)

        logits = 2.0 * dot - q2.unsqueeze(-1) - q2.unsqueeze(-2)
        logits = logits / math.sqrt(qk.shape[-1])

        if self.tau is not None:
            logits = logits * self.tau

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)                   # (B, H, N, d_h)

        out = _from_heads(out)                        # (B, N, D)
        out = self.out_proj(out)
        return out


class ChannelDiffusion(nn.Module):
    """
    Channel diffusion over the feature dimension.

    Input:  x (B, N, D)
    Output: y (B, N, D)

    Channels are treated as the "sequence" being diffused, while token index N
    acts as the embedding axis for the distance computation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        bias: bool = False,
        temperature: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads

        self.qk_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1)) if temperature else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        qk = self.qk_proj(x)          # (B, N, D)
        v = self.v_proj(x)            # (B, N, D)

        qk = qk.transpose(1, 2)       # (B, D, N)
        v = v.transpose(1, 2)         # (B, D, N)

        qk = qk.reshape(b, self.num_heads, self.d_head, n)
        v = v.reshape(b, self.num_heads, self.d_head, n)

        dot = torch.matmul(qk, qk.transpose(-1, -2))  # (B, H, d_h, d_h)
        q2 = (qk * qk).sum(dim=-1)                    # (B, H, d_h)

        logits = 2.0 * dot - q2.unsqueeze(-1) - q2.unsqueeze(-2)
        logits = logits / math.sqrt(n)

        if self.tau is not None:
            logits = logits * self.tau

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)                   # (B, H, d_h, N)

        out = out.reshape(b, d, n).transpose(1, 2).contiguous()  # (B, N, D)
        out = self.out_proj(out)
        return out


class DiffusionBlock(nn.Module):
    """
    DMRG-compatible diffusion-style block for ViT replacement.

    Compatible constructor names:
      - hidden_size / num_attention_heads   (preferred for current DMRG.py)
      - dim / num_heads                     (aliases)

    Forward accepts:
      - x
      - hidden_states
    and ignores extra kwargs so it plays nicely with adapters.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        *,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        channel_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        temperature: bool = True,
        init_scale: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__()

        dim = hidden_size if hidden_size is not None else dim
        num_heads = num_attention_heads if num_attention_heads is not None else num_heads

        if dim is None:
            raise ValueError("DiffusionBlock needs `hidden_size` or `dim`.")
        if num_heads is None:
            raise ValueError("DiffusionBlock needs `num_attention_heads` or `num_heads`.")

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if dim % channel_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by channel_heads={channel_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.channel_heads = channel_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TokenDiffusion(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            temperature=temperature,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ch_attn = ChannelDiffusion(
            dim=dim,
            num_heads=channel_heads,
            bias=bias,
            temperature=temperature,
        )

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Small residual scales help when swapping random blocks into a pretrained model.
        self.gamma1 = nn.Parameter(init_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(dim))

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        *,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states is not None:
            x = hidden_states
        if x is None:
            raise ValueError("DiffusionBlock.forward expected `x` or `hidden_states`.")

        x = x + self.drop(self.attn(self.norm1(x))) * self.gamma1
        x = x + self.drop(self.ch_attn(self.norm2(x))) * self.gamma2
        return x


# Optional alias if you want to preserve your earlier naming style.
Diffusion_Block = DiffusionBlock
