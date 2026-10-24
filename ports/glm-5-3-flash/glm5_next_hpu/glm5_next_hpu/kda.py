# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) kernels for Gaudi, in PyTorch.

KDA is a gated delta rule with a PER-CHANNEL decay, `g: [B, T, H, K]`. vllm-gaudi's gated-delta-rule kernel takes one
scalar decay per head, and upstream's chunked KDA kernels are Triton, which Gaudi does not run. 34 of the 45 layers
of GLM-5.3-Flash use this.

Numerics, in the order they bit this port:

* Decay masks. The reference materialises `exp(g_i - g_j)` as [B, H, NC, C, C, D] (8.6 GB at 4K tokens). Factoring
  it as `exp(g_i) * exp(-g_j)` overflows fp32: the within-chunk cumulative decay reaches about -300 with the gate's
  lower bound of -5. `_gauged_scores` anchors the exponent at a sub-block boundary so both factors lie in (0, 1].
* The UT transform `(I - A)^-1`. Nilpotent doubling is exact in algebra and cancels catastrophically in fp32 on the
  real model's A (rows 32 to 63 of every chunk wrong). `_ut_inverse` uses blocked forward substitution (16 x 16
  blocks, four sequential stages), with doubling only inside a diagonal block where it is stable.
* Rank. Every tensor stays at rank 4 or less. The graph compiler rejects higher-rank tensors in a graph that also
  holds the MoE op ("ndim is currently not supported for MOE").
* Constants. `eye` and the masks must be module buffers owned by the layer (see `make_kda_consts`). Built inside the
  forward they are frozen into the first captured graph, which turns later prompt buckets into garbage or crashes.

Prefill carries the state between 64-token chunks either with the reference loop (`hpu_chunk_kda`) or with a
log-depth scan (`hpu_chunk_kda_scan`, switch GLM5_NEXT_KDA_SCAN=1). Both share the chunk-independent half.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

__all__ = ["KdaConsts", "make_kda_consts", "hpu_chunk_kda", "hpu_chunk_kda_scan", "hpu_recurrent_kda", "hpu_kda_gate"]


class KdaConsts(NamedTuple):
    """`eye(chunk)`, `keep(sub, strict)`, `keep(sub, with diagonal)`."""

    eye: torch.Tensor
    keep_strict: torch.Tensor
    keep_diag: torch.Tensor


def make_kda_consts(chunk_size: int, sub_block: int, dtype: torch.dtype, device) -> KdaConsts:
    """Build the constants. Call from a module __init__ and register them as buffers, never from a forward."""
    return KdaConsts(
        eye=torch.eye(chunk_size, dtype=dtype, device=device),
        keep_strict=1.0 - torch.ones(sub_block, sub_block, dtype=dtype, device=device).triu(0),
        keep_diag=1.0 - torch.ones(sub_block, sub_block, dtype=dtype, device=device).triu(1),
    )


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x / sqrt(sum(x^2) + eps)`: eps inside the sqrt, as the reference (not `F.normalize`)."""
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def _ut_inverse_doubling(a: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """(I - a)^-1 for strictly lower-triangular `a` by nilpotent doubling.

    Exact in algebra, but each step forms a^(2^k), which is far larger than the sum, so it is only accurate for a
    small block: used for the 16 x 16 diagonal blocks (about 5e-5 on real inputs), never for a whole 64-token chunk.
    """
    c = a.shape[-1]
    s = eye + a
    x = a
    for _ in range(max(0, math.ceil(math.log2(c)) - 1)):
        x = x @ x
        s = s + s @ x
    return s


def _ut_inverse(a: torch.Tensor, eye: torch.Tensor, block: int = 16) -> torch.Tensor:
    """(I - a)^-1 for strictly lower-triangular `a` by blocked forward substitution.

        S_ii = (I - A_ii)^-1,    S_ij = S_ii @ sum_{k=j}^{i-1} A_ik @ S_kj  (i > j)

    Worst row error on the real A, fp32 against float64: 6.7e-4 here, 8.2 with whole-chunk doubling.
    """
    c = a.shape[-1]
    if c <= block:
        return _ut_inverse_doubling(a, eye)
    nb = c // block
    if nb * block != c:
        raise ValueError(f"chunk {c} not divisible by inverse block {block}")

    eye_b = eye[:block, :block]
    lead = a.shape[:-2]
    a4 = a.reshape(-1, c, c)
    n = a4.shape[0]

    def blk(i, j):
        return a4[:, i * block : (i + 1) * block, j * block : (j + 1) * block]

    dia = [_ut_inverse_doubling(blk(i, i).contiguous(), eye_b) for i in range(nb)]
    s = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        s[i][i] = dia[i]
        for j in range(i - 1, -1, -1):
            acc = blk(i, j) @ s[j][j]
            for k in range(j + 1, i):
                acc = acc + blk(i, k) @ s[k][j]
            s[i][j] = dia[i] @ acc

    zero = a4.new_zeros(n, block, block)
    rows = [torch.cat([s[i][j] if j <= i else zero for j in range(nb)], dim=-1) for i in range(nb)]
    return torch.cat(rows, dim=-2).reshape(*lead, c, c)


def _gauged_scores(
    a: torch.Tensor, b: torch.Tensor, g: torch.Tensor, sub: int, strict: bool, keep: torch.Tensor
) -> torch.Tensor:
    """S[i, j] = sum_c a[i,c] * b[j,c] * exp(g[i,c] - g[j,c]), lower-triangular (strict: j < i, else j <= i).

    `g` is the within-chunk cumulative decay (non-positive, decreasing). For row block I the anchor is g at its first
    row; for every earlier j, exp(g_i - g_j) = exp(g_i - g_ref) * exp(g_ref - g_j) with both factors in (0, 1], so the
    off-diagonal part is one matmul. Rows are joined with `cat` and masks applied by multiplication: in-place slice
    writes and `torch.where` fall back to the CPU during graph capture.
    """
    *lead, c, d = a.shape
    if c % sub:
        raise ValueError(f"chunk {c} not divisible by sub-block {sub}")
    nb = c // sub
    m = 1
    for x in lead:
        m *= x
    a = a.reshape(m, c, d)
    b = b.reshape(m, c, d)
    g = g.reshape(m, c, d)

    rows = []
    for i_blk in range(nb):
        i0, i1 = i_blk * sub, (i_blk + 1) * sub
        g_ref = g[:, i0 : i0 + 1, :]
        gi = g[:, i0:i1, :]
        dexp = (gi.unsqueeze(2) - gi.unsqueeze(1)).clamp(max=0.0).exp()
        mid = (a[:, i0:i1, :].unsqueeze(2) * b[:, i0:i1, :].unsqueeze(1) * dexp).sum(-1) * keep

        pieces = []
        if i0 > 0:
            row = a[:, i0:i1, :] * (gi - g_ref).clamp(max=0.0).exp()
            col = b[:, :i0, :] * (g_ref - g[:, :i0, :]).clamp(max=0.0).exp()
            pieces.append(row @ col.transpose(-1, -2))
        pieces.append(mid)
        if i1 < c:
            pieces.append(a.new_zeros(m, sub, c - i1))
        rows.append(torch.cat(pieces, dim=-1))

    return torch.cat(rows, dim=-2).reshape(*lead, c, c)


def _chunk_prep(query, key, value, g, beta, chunk_size, sub_block, use_qk_l2norm_in_kernel, consts):
    """The chunk-independent half of prefill, batched over every chunk.

    Returns (v_hat, k_cumdecay, intra, q_dec, k_dec, g_last, t, dtype); per-chunk tensors are [B, H, NC, C, D].
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    t, k_dim = key.shape[2], key.shape[3]
    pad = (chunk_size - t % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad)) * (k_dim**-0.5)
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)
    ]
    g = g.cumsum(dim=-2)

    if consts is None:
        # Eager callers only (CPU tests): under graph capture the layer must pass its buffers.
        consts = make_kda_consts(chunk_size, sub_block, torch.float32, query.device)
    a_mat = -_gauged_scores(k_beta, key, g, sub=sub_block, strict=True, keep=consts.keep_strict)
    ut = _ut_inverse(a_mat, consts.eye)
    v_hat = ut @ v_beta
    k_cumdecay = ut @ (k_beta * g.exp())
    intra = _gauged_scores(query, key, g, sub=sub_block, strict=False, keep=consts.keep_diag)
    g_last = g[:, :, :, -1:, :]
    q_dec = query * g.exp()
    k_dec = key * (g_last - g).exp()
    return v_hat, k_cumdecay, intra, q_dec, k_dec, g_last, t, initial_dtype


def _initial_state(initial_state, b_sz, h, k_dim, v_dim, device):
    if initial_state is None:
        return torch.zeros(b_sz, h, k_dim, v_dim, dtype=torch.float32, device=device)
    return initial_state.to(torch.float32)


def hpu_chunk_kda(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = 64,
    sub_block: int = 16,
    initial_state=None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    consts: KdaConsts | None = None,
):
    """Prefill KDA with the reference state carry, one loop iteration per chunk.

    q, k, v, g: [B, T, H, D] (g already gated), beta: [B, T, H]. Returns (out [B, T, H, V], final state [B, H, K, V]).
    """
    v_hat, k_cumdecay, intra, q_dec, k_dec, g_last, t, dtype = _chunk_prep(
        query, key, value, g, beta, chunk_size, sub_block, use_qk_l2norm_in_kernel, consts
    )
    b_sz, h, n_chunks, _, k_dim = k_dec.shape
    v_dim = v_hat.shape[-1]
    state = _initial_state(initial_state, b_sz, h, k_dim, v_dim, q_dec.device)
    outs = []
    for i in range(n_chunks):
        v_new = v_hat[:, :, i] - k_cumdecay[:, :, i] @ state
        outs.append(q_dec[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * g_last[:, :, i].squeeze(-2).unsqueeze(-1).exp() + k_dec[:, :, i].transpose(-1, -2) @ v_new

    core = torch.stack(outs, dim=2).reshape(b_sz, h, -1, v_dim)
    core = core[:, :, :t].transpose(1, 2).contiguous().to(dtype)
    return core, (state if output_final_state else None)


def scan_carry(v_hat, k_cumdecay, intra, q_dec, k_dec, g_last, state, eye_k):
    """Every chunk's output and the final state, with the carry as a Hillis-Steele scan.

    Per chunk the carry is affine in the state, S[i+1] = M[i] S[i] + B[i], with M[i] = diag(gamma[i]) - k_dec[i]^T kc[i]
    and B[i] = k_dec[i]^T v_hat[i]. Affine maps compose associatively, so the states entering all chunks take
    ceil(log2(NC)) levels of batched matmuls (7 at 128 chunks) instead of NC dependent steps. Every M[i] is a product of
    contractions (||k|| = 1, beta in (0, 1), exp(g) <= 1), so the partial products never grow and nothing cancels.

    Shapes (m = B * H): v_hat [m, NC, C, V], k_cumdecay / q_dec / k_dec [m, NC, C, K], intra [m, NC, C, C],
    g_last [m, NC, 1, K], state [m, K, V], eye_k [K, K]. Returns (o [m, NC, C, V], final [m, K, V]).
    """
    nc = k_dec.shape[1]
    kd_t = k_dec.transpose(-1, -2)
    gam = g_last.exp().transpose(-1, -2)
    mat = eye_k * gam - kd_t @ k_cumdecay
    add = kd_t @ v_hat
    s0 = state.unsqueeze(1)
    # fold the initial state into chunk 0, so after the scan add[i] is the state after chunk i
    add = torch.cat([mat[:, :1] @ s0 + add[:, :1], add[:, 1:]], dim=1)
    d = 1
    while d < nc:
        tail = mat[:, d:]
        add = torch.cat([add[:, :d], tail @ add[:, : nc - d] + add[:, d:]], dim=1)
        if 2 * d < nc:
            mat = torch.cat([mat[:, :d], tail @ mat[:, : nc - d]], dim=1)
        d *= 2
    s_in = torch.cat([s0, add[:, : nc - 1]], dim=1)
    o = (q_dec - intra @ k_cumdecay) @ s_in + intra @ v_hat
    return o, add[:, nc - 1]


def hpu_chunk_kda_scan(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = 64,
    sub_block: int = 16,
    initial_state=None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    consts: KdaConsts | None = None,
    eye_k: torch.Tensor | None = None,
):
    """`hpu_chunk_kda` with the scan carry. `eye_k` is the layer's eye(K) buffer (required under graph capture).

    In lazy mode the loop above is recorded on the host op by op: about 2,100 ops per KDA layer for an 8,192-token
    chunk. The scan records about 400.
    """
    v_hat, k_cumdecay, intra, q_dec, k_dec, g_last, t, dtype = _chunk_prep(
        query, key, value, g, beta, chunk_size, sub_block, use_qk_l2norm_in_kernel, consts
    )
    b_sz, h, n_chunks, _, k_dim = k_dec.shape
    v_dim = v_hat.shape[-1]
    if eye_k is None:
        eye_k = torch.eye(k_dim, dtype=torch.float32, device=q_dec.device)
    state = _initial_state(initial_state, b_sz, h, k_dim, v_dim, q_dec.device)
    m = b_sz * h

    def flat(x):
        return x.reshape(m, n_chunks, x.shape[-2], x.shape[-1])

    o, final = scan_carry(
        flat(v_hat),
        flat(k_cumdecay),
        flat(intra),
        flat(q_dec),
        flat(k_dec),
        flat(g_last),
        state.reshape(m, k_dim, v_dim),
        eye_k,
    )
    core = o.reshape(b_sz, h, -1, v_dim)
    core = core[:, :, :t].transpose(1, 2).contiguous().to(dtype)
    final = final.reshape(b_sz, h, k_dim, v_dim)
    return core, (final if output_final_state else None)


def hpu_recurrent_kda(
    query, key, value, g, beta, initial_state=None, output_final_state: bool = True, use_qk_l2norm_in_kernel=True
):
    """Decode KDA. q, k, v, g: [B, T, H, D], beta: [B, T, H]. The per-channel decay scales the state's key axis."""
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    b_sz, t, h, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (k_dim**-0.5)
    state = _initial_state(initial_state, b_sz, h, k_dim, v_dim, query.device)
    outs = []
    for i in range(t):
        k_i = key[:, i]
        state = state * g[:, i].unsqueeze(-1).exp()
        kv_mem = (state * k_i.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, i] - kv_mem) * beta[:, i].unsqueeze(-1)
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * query[:, i].unsqueeze(-1)).sum(dim=-2))

    return torch.stack(outs, dim=1).to(initial_dtype), (state if output_final_state else None)


def hpu_kda_gate(
    raw_g: torch.Tensor, a_log: torch.Tensor, head_dim: int, g_bias: torch.Tensor | None, lower_bound: float
) -> torch.Tensor:
    """GLM's bounded-sigmoid forget gate: `lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))`.

    `raw_g` is [..., H*D]; returns [..., H, D]. This is not vLLM's `fused_kda_gate`, which is Kimi Linear's softplus
    gate: run on GLM weights that loads and talks fluently while retention over 64 tokens is off by twelve orders of
    magnitude.
    """
    lead = raw_g.shape[:-1]
    g = raw_g.float()
    if g_bias is not None:
        g = g + g_bias.float().reshape(-1)
    g = g.view(*lead, -1, head_dim)
    decay = torch.exp(a_log.float().reshape(1, -1, 1) if a_log.dim() == 1 else a_log.float())
    while decay.dim() < g.dim():
        decay = decay.unsqueeze(0)
    return lower_bound * torch.sigmoid(decay * g)
