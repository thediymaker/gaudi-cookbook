# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision tower, ported from the transformers reference `Glm5NextVisionModel`.

Ported from the reference rather than from vLLM's GLM-4.1V tower, which differs in four places that all fail quietly:
q_norm / k_norm on the attention heads, biases on qkv and proj, clamped SwiGLU in the blocks and in the patch merger,
and a GELU between the merger's norm and its gated MLP. Parameter names equal the reference's, so loading is a prefix
strip. With explicit-matmul attention the port matched the reference bit for bit on CPU; the SDPA path used here
differs only by accumulation order.

The tower is replicated on every rank (about 1.1 GiB in bf16) and runs eagerly: vllm-gaudi's vision graph wrap only
applies to Gemma 3 and to towers named `vision_tower`. Images cost compile time through the prompt-length buckets they
create, not through image geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float | None) -> torch.Tensor:
    """`silu(clamp(gate, max=L)) * clamp(up, -L, L)`, as the reference."""
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class Glm5NextVisionRMSNorm(nn.Module):
    """RMSNorm in fp32, cast back, as the reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x32.to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """fp32 rotary on [S, H, D], as the reference."""
    qd, kd = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.to(qd), k_out.to(kd)


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config, bias: bool = True):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, swiglu_limit: float, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.gelu(self.post_projection_norm(x))  # the reference's act1, before the gated MLP
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        bias = bool(getattr(config, "attention_bias", True))
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.scaling = self.head_dim**-0.5
        self.q_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _attend(self, q, k, v, cu_seqlens):
        """q, k, v [S, H, D] -> [S, H, D]. One image is one segment and skips the data-dependent split."""
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        if cu_seqlens is None or cu_seqlens.numel() <= 2:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scaling
            )
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outs = [
                F.scaled_dot_product_attention(
                    qq, kk, vv, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scaling
                )
                for qq, kk, vv in zip(
                    torch.split(q, lengths, dim=2),
                    torch.split(k, lengths, dim=2),
                    torch.split(v, lengths, dim=2),
                    strict=False,
                )
            ]
            out = torch.cat(outs, dim=2)
        return out.squeeze(0).transpose(0, 1)

    def forward(self, x, cu_seqlens, position_embeddings):
        s = x.shape[0]
        qkv = self.qkv(x).reshape(s, 3, self.num_heads, -1)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = _apply_rope_vision(q, k, cos, sin)
        out = self._attend(q, k, v, cu_seqlens)
        return self.proj(out.reshape(s, -1))


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config, bias=bool(getattr(config, "attention_bias", True)))

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        return x + self.mlp(self.norm2(x))


class Glm5NextVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        k = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


def vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Per-patch (h, w) ids in merge-block order, as `get_vision_position_ids` in the reference: the downsample later
    folds each merge_size x merge_size neighbourhood into one token, and the rotary ids must follow that order."""
    out = []
    m = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        for pos in (hpos, wpos):
            pos = pos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            out.append(pos.repeat(t))
    return torch.stack([torch.cat(out[0::2]), torch.cat(out[1::2])], dim=-1)


def vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Segment boundaries, as `get_vision_cu_seqlens` in the reference."""
    per = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    return F.pad(per, (1, 0), value=0)


class Glm5NextVisionModel(nn.Module):
    """Returns the merger output (the reference's `pooler_output`), which is what the language model receives."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.merger = Glm5NextVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            swiglu_limit=config.swiglu_limit,
        )
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = Glm5NextVisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = vision_position_ids(grid_thw, self.spatial_merge_size).to(pixel_values.device)
        cu = vision_cu_seqlens(grid_thw).to(pixel_values.device)

        x = self.patch_embed(pixel_values)
        rot = self.rotary_pos_emb(pos_ids)
        emb = torch.cat((rot, rot), dim=-1)
        pos_emb = (emb.cos(), emb.sin())
        for blk in self.blocks:
            x = blk(x, cu, pos_emb)

        x = self.post_layernorm(x)
        m = self.spatial_merge_size
        x = x.view(-1, m, m, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.config.out_hidden_size)
        return self.merger(x)
