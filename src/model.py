"""Tiny decoder-only transformer for command parsing.

Adapted from esp32-tinyllm's TinyLM. The five-arm ablation machinery and the
Per-Layer Embeddings path are gone: PLE's advantage comes from a large
vocabulary (+0.098 nats at 32,768 against +0.025 at 4,096), and this model has
1,024 tokens. What remains is the plain core, which is all the task needs.

Every op here must stay reproducible by `firmware/common/llm.h`: split-half
RoPE, SiLU-SwiGLU, RMSNorm, tied embedding. Change one side, change the other,
then re-export and re-run verify.c.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 1024
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ffn_hidden: int = 128
    seq_len: int = 96
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    # Tying is inherited from esp32-tinyllm, where it saved 3.1M parameters on a
    # 32,768 vocabulary. Here it saves 65,536 -- but the input and output
    # vocabularies are nearly disjoint (symbols are only ever emitted, English
    # words almost only ever read), so one shared space may be the wrong
    # constraint rather than a free saving. Measured, not assumed: see RESULTS.

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # Split-half, matching llm.h. Interleaved would be numerically different.
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        """`loss_mask` selects the positions that count.

        Only the emitted command is a target. Training the model to predict the
        user's own words would spend most of its capacity modelling English it
        will never need to produce -- the prompt is context, not output.
        """
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self.head(self.out_norm(x))

        if targets is None:
            return logits, None

        flat = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), targets.reshape(-1),
            reduction="none")
        if loss_mask is None:
            return logits, flat.mean()
        w = loss_mask.reshape(-1).float()
        return logits, (flat * w).sum() / w.sum().clamp(min=1)

    def param_count(self) -> dict:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue                       # tied weights counted once
            seen.add(id(p))
            total += p.numel()
        emb = self.tok_emb.weight.numel()
        return {"total": total, "embedding_tied_head": emb, "blocks": total - emb}


def make_model(cfg: Config, verbose: bool = True) -> TinyLM:
    model = TinyLM(cfg)
    if verbose:
        c = model.param_count()
        print(f"vocab={cfg.vocab_size} d_model={cfg.d_model} layers={cfg.n_layers} "
              f"ffn={cfg.ffn_hidden} seq={cfg.seq_len}")
        print(f"params: {c['total']:,} total "
              f"({c['blocks']:,} blocks + {c['embedding_tied_head']:,} tied emb/head)"
              f"  -> int8 {c['total'] / 1024:.0f} KB")
    return model
