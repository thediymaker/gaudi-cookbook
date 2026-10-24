# SPDX-License-Identifier: Apache-2.0
"""DeepSeek sparse attention (DSA) for GLM-5.x on vllm-gaudi's HPU MLA path.

GLM-5.x is trained with DSA: a lightning indexer scores every earlier token and each query attends only to its top
``index_topk`` (2048) keys. vllm-gaudi serves these layers as dense MLA (``patches/glm_dsa.py`` turns the native
sparse path off, it does not run under this configuration). Dense attention is exact up to 2048 keys and an
approximation beyond. With the accurate FusedSDPA softmax (``patches/fsdpa_softmax_mode.py``) dense attention
passes the long-context needles to 128K; this module adds the real top-k selection and makes long contexts faster.

Switch: ``GLM_DSA_SPARSE=1`` (off: dense MLA everywhere). Below :data:`MIN_KEYS` keys per sequence attention stays
dense with the accurate softmax; above it:

* Prefill indexer: each TP rank scores 1/world of the query rows with fp32 scores, finds every row's
  ``index_topk``-th score by bisection (reproduces the exact top-k selection on GLM-5.3 scores), and the
  bit-packed selections are all-gathered, so all ranks attend over the same keys.
* Prefill attention, decided per indexer group (one full indexer layer and the shared layers after it): when 64
  neighbouring queries select overlapping keys (early layers), each block gathers the union of its keys from the
  latent cache and runs absorbed-MLA FusedSDPA on it; when selections are scattered (the union approaches the whole
  context) the selection is a mask on dense FusedSDPA with causal row-block trimming, which is then cheaper.
* Decode: each sequence gathers exactly its ``index_topk`` latent rows (fp32 softmax), so decode cost no longer
  grows with context. Steps with several query rows per sequence and batched prefills use the selection as a mask
  on the dense paths instead.

The indexer keys live in one bf16 buffer per full indexer layer, sized like the MLA cache and allocated after vLLM's
memory profiling, so serve with ``--gpu-memory-utilization 0.86`` and keep prefills with more than 2048 tokens out
of HPU graphs (``--max-cudagraph-capture-size 2048``).
"""

import math
import os

import torch

ENABLED = os.environ.get("GLM_DSA_SPARSE", "0") == "1"
TAG = "[glm-dsa]"

NEG = -1.0e30
# Attention stays dense (exact up to index_topk keys, and fast) until a prefill chunk (context + chunk, bucket
# padded) or a decode batch (per sequence) has more keys than this.
MIN_KEYS = 32768
# Scratch for the indexer scores of one (row block, head group): rows x heads x keys x 2 bytes. Not seen by vLLM's
# memory profiling, so keep it small.
ROW_BUDGET_BYTES = 1 << 30
# Row blocks shrink to 512 before heads are split: top-k uses the TPCs badly on fewer rows.
MAX_ROWS = 2048
# Queries per gathered key set in the sparse prefill attention.
UNION_ROWS = 64
# Gather a layer group only when its padded union is at most this fraction of the keys, and at most GATHER_MAX_UCAP.
GATHER_MAX_FRAC = 0.5
GATHER_MAX_UCAP = 32768
BISECT_ITERS = 20
# FusedSDPA softmax mode of the sparse prefill attention: "fast" is badly wrong for some rows of deep GLM layers.
SOFTMAX_MODE = "None"
# Gathered latent rows per FusedSDPA call are capped (one 9.7 GB gather hung all eight cards into the driver's 600 s
# "no progress" timeout).
ATTN_BUDGET_BYTES = 512 << 20
DENSE_ROW_BLOCKS = 4
# Above this many keys a dense-masked prefill keeps the selection bit-packed and builds each row block's mask on
# the fly (a [8192, 262144] bf16 mask is 4.3 GB).
FULLMASK_MAX_KEYS = 131072

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA as _FSDPA
except ImportError:  # CPU tests
    _FSDPA = None

_STATE = {"sel": None, "active": None}
_INFO = {"topk": None, "ready": False}
_ORIG_FORWARD_IMPL = None


def _graph_break():
    """Lazy mode: cut the graph around every row block of the prefill indexer. Unrolled into one graph per step,
    the row blocks of 21 layers made graphs whose first compile ran into the 600 s timeout. Never inside an HPU
    graph capture."""
    try:
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore
    except ImportError:
        return
    if ht.utils.internal.is_lazy() and not ht.hpu.is_current_stream_capturing():
        htcore.mark_step()


# ---------------------------------------------------------------------------------------------------------------------
# Core: plain torch, no vLLM imports (tests/test_dsa_core.py runs it on the CPU)


def kth_largest(s, k, m=16384):
    """k-th largest value per row of s [R, N]. Two top-k passes over groups of m are ~3x faster on Gaudi 2 than one
    over the full row, and exact: every global top-k element is inside its group's top-k."""
    r, n = s.shape
    if n <= 2 * m:
        return torch.topk(s, k, dim=-1).values[:, -1:]
    pad = (-n) % m
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=-float("inf"))
    c = torch.topk(s.view(r, -1, m), k, dim=-1).values.reshape(r, -1)
    return torch.topk(c, k, dim=-1).values[:, -1:]


def topk_indices(s, k, m=16384):
    """Exact top-k (values, indices) per row of s [R, N], two passes for long rows."""
    r, n = s.shape
    if n <= 2 * m:
        return torch.topk(s, k, dim=-1)
    pad = (-n) % m
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=-float("inf"))
    g = (n + pad) // m
    v1, i1 = torch.topk(s.view(r, g, m), k, dim=-1)
    i1 = i1 + (torch.arange(g, device=s.device, dtype=i1.dtype) * m).view(1, g, 1)
    v2, i2 = torch.topk(v1.reshape(r, g * k), k, dim=-1)
    return v2, torch.gather(i1.reshape(r, g * k), 1, i2)


def prefill_mask_bias(q_idx, w, k_all, attn_bias, topk):
    """Selection as a mask (batched prefill). q_idx [B,Q,Hi,D], w [B,Q,Hi], k_all [B,KV,D], attn_bias [B,1,Q,KV]
    (0 / -inf). Returns attn_bias with every visible key outside the query's top-k set to NEG."""
    b_, q_, hi, d = q_idx.shape
    kv = k_all.shape[1]
    if kv <= topk:
        return attn_bias
    rows = min(q_, MAX_ROWS)
    while rows > 512 and rows * hi * kv * 2 > ROW_BUDGET_BYTES:
        rows //= 2
    hc = max(1, min(hi, ROW_BUDGET_BYTES // (rows * kv * 2)))
    neg = torch.tensor(NEG, dtype=attn_bias.dtype, device=attn_bias.device)
    out = []
    _graph_break()
    for b in range(b_):
        kt = k_all[b].transpose(0, 1)
        parts = []
        for r0 in range(0, q_, rows):
            r1 = min(q_, r0 + rows)
            n = r1 - r0
            acc = None
            for h0 in range(0, hi, hc):
                h1 = min(hi, h0 + hc)
                s = torch.matmul(q_idx[b, r0:r1, h0:h1].reshape(n * (h1 - h0), d), kt)
                s = torch.relu_(s).view(n, h1 - h0, kv)
                s = torch.bmm(w[b, r0:r1, h0:h1].to(s.dtype).unsqueeze(1), s).squeeze(1).float()
                acc = s if acc is None else acc + s
            bias_r = attn_bias[b, 0, r0:r1]
            s = acc.to(bias_r.dtype) + bias_r
            parts.append(torch.where(s >= kth_largest(s, topk), bias_r, neg))
            _graph_break()
        out.append(torch.cat(parts, 0) if len(parts) > 1 else parts[0])
    res = torch.stack(out, 0).unsqueeze(1)
    _graph_break()
    return res


def decode_mask_bias(q_idx, w, k_blocks, block_groups, block_bias, topk):
    """Selection as a block mask (decode with several query rows per sequence). q_idx [bs,Hi,D], w [bs,Hi],
    k_blocks [nb,T,D], block_groups [nb] (row id, -1 padding), block_bias [nb,T] (0 / -inf)."""
    bs = q_idx.shape[0]
    nb, t, _ = k_blocks.shape
    if nb * t <= topk:
        return block_bias
    g = block_groups.to(torch.long)
    valid = (g >= 0) & (g < bs)
    gc = g.clamp(0, bs - 1)
    s = torch.relu(torch.bmm(q_idx.index_select(0, gc), k_blocks.transpose(1, 2)))
    s = torch.bmm(w.index_select(0, gc).to(s.dtype).unsqueeze(1), s).squeeze(1).float()
    s = s + block_bias.float()
    s = torch.where(valid.unsqueeze(-1), s, -float("inf"))
    seq = torch.arange(bs, device=s.device, dtype=torch.long)
    member = (gc.unsqueeze(0) == seq.unsqueeze(1)) & valid.unsqueeze(0)
    dense = torch.where(member.unsqueeze(-1), s.unsqueeze(0), -float("inf"))
    thr = kth_largest(dense.reshape(bs, nb * t), topk)[:, 0]
    thr_b = thr.index_select(0, gc).unsqueeze(-1)
    # Padding blocks keep their bias: flat paged attention rescales each block by its group's sum, a padding block's
    # group sum is 0, and a fully masked padding block would compute 0/0 and poison the batch.
    keep = torch.where((s >= thr_b) | (~valid).unsqueeze(-1), 0.0, NEG)
    return (block_bias.float() + keep).to(block_bias.dtype)


def causal_bias(bsz, q_len, seq_lens, dtype, device):
    """The no-context prompt bias vllm-gaudi builds when FusedSDPA is not used."""
    len_mask = (
        torch.arange(0, q_len, device=device, dtype=torch.int32)
        .view(1, q_len)
        .ge(seq_lens.unsqueeze(-1))
        .view(bsz, 1, 1, q_len)
    )
    causal = torch.triu(torch.ones((bsz, 1, q_len, q_len), device=device, dtype=torch.bool), diagonal=1)
    mask = causal.logical_or(len_mask)
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)


def prefill_visible(r0, r1, kv, cb, ctx_len, seq_len, device):
    """Visibility [R, KV] of prefill rows r0..r1 of one sequence in the HPU prefill key layout: cb context slots
    (valid below ctx_len), then the chunk (causal, valid below seq_len)."""
    j = torch.arange(kv, device=device, dtype=torch.int32).view(1, kv)
    r = torch.arange(r0, r1, device=device, dtype=torch.int32).view(-1, 1)
    jc = j - cb
    return (j < ctx_len) | ((jc >= 0) & (jc <= r) & (jc < seq_len))


def bisect_threshold(acc, ok, k, iters=BISECT_ITERS):
    """Per-row threshold t with count(acc >= t) >= k, as tight as `iters` halvings allow; -inf when a row has at
    most k valid keys (keep them all)."""
    big = torch.tensor(float("inf"), device=acc.device, dtype=acc.dtype)
    lo = torch.where(ok, acc, big).amin(1, keepdim=True)
    hi = torch.where(ok, acc, -big).amax(1, keepdim=True)
    nvalid = ok.sum(1, keepdim=True)
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        ge = (acc >= mid).sum(1, keepdim=True) >= k
        lo = torch.where(ge, mid, lo)
        hi = torch.where(ge, hi, mid)
    return torch.where(nvalid <= k, -big, lo)


def prefill_select(q_idx, w, kt, r0, r1, cb, ctx_len, seq_len, topk, rr):
    """Indexer selection for prefill rows r0..r1 of one sequence. q_idx [Q,Hi,D], w [Q,Hi], kt [D,KV]. Returns
    sel [n, KV] (row keeps key) and u [n/rr, KV] (union over each block of rr rows)."""
    _, hi, d = q_idx.shape
    kv = kt.shape[1]
    rows = min(r1 - r0, MAX_ROWS)
    while rows > 512 and rows * hi * kv * 2 > ROW_BUDGET_BYTES:
        rows //= 2
    rows = max(rr, rows - rows % rr)
    hc = max(1, min(hi, ROW_BUDGET_BYTES // (rows * kv * 2)))
    sels, us = [], []
    _graph_break()
    for a in range(r0, r1, rows):
        b = min(r1, a + rows)
        n = b - a
        acc = None
        for h0 in range(0, hi, hc):
            h1 = min(hi, h0 + hc)
            s = torch.matmul(q_idx[a:b, h0:h1].reshape(n * (h1 - h0), d), kt)
            s = torch.relu_(s).view(n, h1 - h0, kv)
            s = torch.bmm(w[a:b, h0:h1].to(s.dtype).unsqueeze(1), s).squeeze(1).float()
            acc = s if acc is None else acc + s
        ok = prefill_visible(a, b, kv, cb, ctx_len, seq_len, acc.device)
        row = torch.arange(a, b, device=acc.device, dtype=torch.int32).view(-1, 1)
        ok = ok & (row < seq_len)  # bucket-padding rows select nothing
        acc = torch.where(ok, acc, -float("inf"))  # invisible keys never count toward the top-k
        sel = (acc >= bisect_threshold(acc, ok, topk)) & ok
        sels.append(sel)
        us.append(sel.view(n // rr, rr, kv).any(1))
        _graph_break()
    if len(sels) == 1:
        return sels[0], us[0]
    return torch.cat(sels, 0), torch.cat(us, 0)


def union_bucket(n, topk):
    """Static bucket for a union size: multiples of topk/2 up to 4*topk, then powers of 2."""
    step = topk // 2
    if n <= 4 * topk:
        return max(topk, -(-n // step) * step)
    c = 4 * topk
    while c < n:
        c *= 2
    return c


def compact(sel, u, rr, ucap, dtype):
    """Per block of rr rows: cl [nb, ucap] = its union keys in key order (padded with key 0), bias [nb*rr, ucap] =
    0 where the row selected that key, NEG elsewhere and on padding."""
    nb, kv = u.shape
    dev = u.device
    ui = u.to(torch.int32)
    cnt = ui.sum(1, keepdim=True)
    pos = torch.cumsum(ui, 1, dtype=torch.int32) - 1
    tgt = torch.where(u, pos, ucap).to(torch.long)
    src = torch.arange(kv, device=dev, dtype=torch.int32).view(1, kv).expand(nb, kv).contiguous()
    cl = torch.zeros(nb, ucap + 1, dtype=torch.int32, device=dev).scatter_(1, tgt, src)[:, :ucap].contiguous()
    g = torch.gather(sel.view(nb, rr, kv), 2, cl.to(torch.long).unsqueeze(1).expand(nb, rr, ucap))
    live = torch.arange(ucap, device=dev, dtype=torch.int32).view(1, 1, ucap) < cnt.view(nb, 1, 1)
    bias = torch.where(g & live, 0.0, NEG).to(dtype).view(nb * rr, ucap)
    return cl, bias


def pack_bits(sel):
    """[n, KV] bool -> [n, KV/8] uint8 (KV % 8 == 0)."""
    n, kv = sel.shape
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.bfloat16, device=sel.device).view(8, 1)
    return torch.matmul(sel.reshape(n * kv // 8, 8).to(torch.bfloat16), wts).view(n, kv // 8).to(torch.uint8)


def unpack_bits(p, kv):
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=p.device)
    return (torch.bitwise_and(p.unsqueeze(-1), wts) != 0).view(p.shape[0], kv)


def sparse_prefill_attn(q_abs, kv_flat, phys, bias, scale, rr, lora):
    """Prefill MLA over gathered union blocks with FusedSDPA in absorbed form: head dim L+P, one latent 'kv head'
    shared by the H query heads, V = the latent's first L dims. q_abs [Q,H,C], kv_flat [S,C], phys [nb,ucap],
    bias [Q,ucap]. Returns [Q,H,lora]."""
    _, h, c = q_abs.shape
    nb, ucap = phys.shape
    g = max(1, min(nb, ATTN_BUDGET_BYTES // (ucap * c * 2)))
    outs = []
    for b0 in range(0, nb, g):
        b1 = min(nb, b0 + g)
        n = b1 - b0
        lat = kv_flat.index_select(0, phys[b0:b1].reshape(-1)).view(n, 1, ucap, c)
        qb = q_abs[b0 * rr : b1 * rr].view(n, rr, h, c).transpose(1, 2)
        o = _FSDPA.apply(
            qb,
            lat,
            lat,
            bias[b0 * rr : b1 * rr].view(n, 1, rr, ucap),
            0.0,
            False,
            scale,
            SOFTMAX_MODE,
            True,
            None,
            "right",
            False,
            False,
            (-1, -1),
            None,
        )
        outs.append(o[..., :lora].transpose(1, 2).reshape(n * rr, h, lora))
        if g < nb:
            _graph_break()
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def dense_masked_attn(q, k, v, mask, cb, scale, packed=None):
    """Dense attention under a selection mask, one sequence. q [Q,H,D], k/v [KV,H,D] (cb context slots + Q chunk
    rows), mask [Q,KV] (0 keep / NEG drop). Query rows go in blocks that read keys only up to the block's last row
    (the chunk part is causal), which skips about 3/8 of the first chunk's work."""
    q_, h, d = q.shape
    nb = DENSE_ROW_BLOCKS
    while nb > 1 and (q_ % nb or (q_ // nb) % 128):
        nb //= 2
    rb = q_ // nb
    outs = []
    for i in range(nb):
        r0, r1 = i * rb, (i + 1) * rb
        kend = cb + r1
        if packed is not None:
            kvp = packed.shape[1] * 8
            m = torch.where(unpack_bits(packed[r0:r1], kvp)[:, :kend], 0.0, NEG).to(q.dtype)
        else:
            m = mask[r0:r1, :kend].contiguous()
        o = _FSDPA.apply(
            q[r0:r1].unsqueeze(0).transpose(1, 2),
            k[:kend].unsqueeze(0).transpose(1, 2),
            v[:kend].unsqueeze(0).transpose(1, 2),
            m.view(1, 1, rb, kend),
            0.0,
            False,
            scale,
            SOFTMAX_MODE,
            True,
            None,
            "right",
            False,
            False,
            (-1, -1),
            None,
        )
        outs.append(o.transpose(1, 2)[0])
        if packed is not None:
            _graph_break()
    return torch.cat(outs, 0) if nb > 1 else outs[0]


def decode_topk(q_idx, w, k_blocks, block_groups, block_bias, topk):
    """Decode indexer: q_idx [bs,Hi,D], w [bs,Hi], k_blocks [nb,T,D], block_groups [nb] (sequence id, -1 padding),
    block_bias [nb,T]. Returns idx [bs, topk] int32: flat positions (block_pos*T + t) in the block list, -1 = none."""
    bs = q_idx.shape[0]
    nb, t, _ = k_blocks.shape
    g = block_groups.to(torch.long)
    valid = (g >= 0) & (g < bs)
    gc = g.clamp(0, bs - 1)
    s = torch.relu(torch.bmm(q_idx.index_select(0, gc), k_blocks.transpose(1, 2)))
    s = torch.bmm(w.index_select(0, gc).to(s.dtype).unsqueeze(1), s).squeeze(1).float() + block_bias.float()
    s = torch.where(valid.unsqueeze(-1), s, -float("inf"))
    seq = torch.arange(bs, device=s.device, dtype=torch.long)
    member = (gc.unsqueeze(0) == seq.unsqueeze(1)) & valid.unsqueeze(0)
    dense = torch.where(member.unsqueeze(-1), s.unsqueeze(0), -float("inf"))
    v, i = topk_indices(dense.reshape(bs, nb * t), topk)
    return torch.where(v > -float("inf"), i.to(torch.int32), -1)


def sparse_decode_attn(q_abs, kv_flat, phys, bias, scale, lora):
    """q_abs [bs,H,C], phys [bs,k] latent rows, bias [bs,k] (0 / NEG). fp32 softmax."""
    bs, _, c = q_abs.shape
    k = phys.shape[1]
    lat = kv_flat.index_select(0, phys.reshape(-1)).view(bs, k, c)
    s = torch.bmm(q_abs * scale, lat.transpose(1, 2)).float() + bias.unsqueeze(1)
    p = torch.softmax(s, dim=-1).to(lat.dtype)
    return torch.bmm(p, lat[..., :lora])


# ---------------------------------------------------------------------------------------------------------------------
# Glue: vllm-gaudi's HPU MLA wrapper and attention


def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _all_gather(x):
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x.contiguous(), dim=0)


def _index_qkw(ind, rope, hidden_states, q_c, positions):
    """vLLM Indexer.forward up to the scores, in bf16."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    q_c = q_c.reshape(-1, q_c.shape[-1])
    q, _ = ind.wq_b(q_c)
    q = q.view(-1, ind.n_head, ind.head_dim)
    kw, _ = ind.wk_weights_proj(hidden_states)
    k = ind.k_norm(kw[:, : ind.head_dim])
    w = kw[:, ind.head_dim :]
    q_pe, q_nope = torch.split(q, [ind.rope_dim, ind.head_dim - ind.rope_dim], dim=-1)
    k_pe, k_nope = torch.split(k, [ind.rope_dim, ind.head_dim - ind.rope_dim], dim=-1)
    q_pe, k_pe = rope(positions, q_pe.contiguous(), k_pe.unsqueeze(1).contiguous())
    q = torch.cat([q_pe.reshape(-1, ind.n_head, ind.rope_dim), q_nope], dim=-1)
    k = torch.cat([k_pe.reshape(-1, ind.rope_dim), k_nope], dim=-1)
    # vLLM's positive per-head constants do not change the top-k; kept so scores match vLLM's scale
    w = w * (ind.softmax_scale * ind.n_head**-0.5)
    return q, k, w


def alloc_buffers(model, kv_caches):
    """Runner hook after the KV cache is bound: one indexer-key buffer per full indexer layer."""
    if not ENABLED:
        return
    from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper
    from vllm_gaudi.extension.runtime import get_config

    if get_config().defrag:
        raise RuntimeError(f"{TAG} GLM_DSA_SPARSE=1 needs KV defragmentation off: it moves blocks, not these buffers")
    n, nbytes = 0, 0
    for m in model.modules():
        if isinstance(m, HPUMultiHeadLatentAttentionWrapper) and getattr(m, "indexer", None) is not None:
            kv = kv_caches.get(m.mla_attn.layer_name)
            if kv is None:
                continue
            kc = kv[0] if isinstance(kv, tuple | list) else kv
            m._glm_dsa_buf = torch.zeros(kc.shape[0], m.indexer.head_dim, dtype=kc.dtype, device=kc.device)
            n += 1
            nbytes += m._glm_dsa_buf.numel() * m._glm_dsa_buf.element_size()
            _INFO["topk"] = int(m.topk_tokens)
    if not n:
        raise RuntimeError(f"{TAG} GLM_DSA_SPARSE=1 but the model has no MLA layer with an indexer")
    _INFO["ready"] = True
    print(f"{TAG} indexer-key buffers for {n} full layers ({nbytes / 2**30:.2f} GiB per rank)", flush=True)


def skip_prompt_bias(md, batch_size, seq_len):
    """Runner hook: the sparse prefill builds its own masks, so the dense prompt bias [bs,1,Q,ctx+Q] is not needed
    for a bs-1 prefill with more than MIN_KEYS keys."""
    if not (ENABLED and _INFO["ready"]) or batch_size != 1:
        return False
    nctx = md.block_list.shape[0] if md.block_list is not None else 0
    return nctx * md.block_size + seq_len > max(_INFO["topk"], MIN_KEYS)


def _batched_prefill_bias(self, md, q_i, k_i, w, hidden_states):
    buf = self._glm_dsa_buf
    bsz = md.seq_lens_tensor.shape[0]
    q_len = q_i.shape[0] // bsz
    hi, d = q_i.shape[1], q_i.shape[2]
    q_i = q_i.view(bsz, q_len, hi, d)
    w = w.view(bsz, q_len, hi)
    cur = k_i.to(buf.dtype).view(-1, d)
    if md.block_list is not None:
        past = buf.unflatten(0, (-1, md.block_size)).index_select(0, md.block_list).reshape(-1, d)
        k_all = torch.cat((past, cur), 0).view(bsz, -1, d)
        attn_bias = md.attn_bias
    else:
        k_all = cur.view(bsz, q_len, d)
        attn_bias = causal_bias(bsz, q_len, md.seq_lens_tensor, hidden_states.dtype, hidden_states.device)
    if attn_bias is None or k_all.shape[1] <= self.topk_tokens:
        return None
    return prefill_mask_bias(q_i, w, k_all, attn_bias, self.topk_tokens)


def _full_layer(self, md, hidden_states, q_c, positions):
    """Full indexer layer. Returns the selection the following shared layers reuse: a dict (sparse prefill,
    dense-masked prefill, sparse decode), a mask tensor (batched prefill, several query rows per sequence in decode)
    or None (dense attention is exact or the cheaper choice)."""
    q_i, k_i, w = _index_qkw(self.indexer, self.indexer_rope_emb, hidden_states, q_c, positions)
    buf = self._glm_dsa_buf
    buf.index_put_((md.slot_mapping.flatten(),), k_i.to(buf.dtype))
    topk = self.topk_tokens
    bs_ = md.block_size
    d = q_i.shape[2]
    if md.is_prompt:
        bsz = md.seq_lens_tensor.shape[0]
        q_len = q_i.shape[0] // bsz
        nctx = md.block_list.shape[0] // bsz if md.block_list is not None else 0
        kv = nctx * bs_ + q_len
        if kv <= max(topk, MIN_KEYS):
            return None
        if bsz != 1:
            return _batched_prefill_bias(self, md, q_i, k_i, w, hidden_states)
        cur = k_i.to(buf.dtype).view(-1, d)
        bl = None
        if nctx:
            bl = md.block_list.to(torch.long).clamp(min=0)
            past = buf.unflatten(0, (-1, bs_)).index_select(0, bl).reshape(-1, d)
            kt = torch.cat((past, cur), 0).transpose(0, 1).contiguous()
        else:
            kt = cur.transpose(0, 1).contiguous()
        ctx_len = md.context_lens_tensor[:1].to(torch.int32)
        seq_len = md.seq_lens_tensor[:1].to(torch.int32)
        world, rank = _tp()
        split = world > 1 and q_len % world == 0  # each rank scores 1/world of the rows
        n = q_len // world if split else q_len
        rr = UNION_ROWS
        while n % rr:
            rr //= 2
        r0 = rank * n if split else 0
        sel, u = prefill_select(q_i, w, kt, r0, r0 + n, nctx * bs_, ctx_len, seq_len, topk, rr)
        lmax = u.to(torch.int32).sum(1, keepdim=True).amax(0, keepdim=True).to(torch.float32)
        gmax = _all_gather(lmax) if split else lmax
        ucap = union_bucket(max(int(gmax.amax().cpu().item()), 1), topk)  # host sync: the static union bucket
        if ucap <= GATHER_MAX_FRAC * kv and ucap <= GATHER_MAX_UCAP:
            cl, bias = compact(sel, u, rr, ucap, hidden_states.dtype)
            if split:
                cl, bias = _all_gather(cl), _all_gather(bias)
            cur_slots = md.slot_mapping.flatten().to(torch.long).clamp(min=0)
            if nctx:
                ctx_slots = bl.view(-1, 1) * bs_ + torch.arange(bs_, device=bl.device, dtype=torch.long).view(1, -1)
                vslot = torch.cat((ctx_slots.reshape(-1), cur_slots), 0)
            else:
                vslot = cur_slots
            phys = vslot.index_select(0, cl.reshape(-1).to(torch.long)).view(cl.shape)
            _graph_break()
            return {"kind": "prefill", "phys": phys, "bias": bias, "rr": rr}
        if kv > FULLMASK_MAX_KEYS:
            packed = pack_bits(sel)
            del sel, u
            if split:
                packed = _all_gather(packed)
            _graph_break()
            return {"kind": "prefill_dense", "mask": None, "packed": packed, "cb": nctx * bs_, "bl": bl}
        if split:
            sel = unpack_bits(_all_gather(pack_bits(sel)), kv)
        mask = torch.where(sel, 0.0, NEG).to(hidden_states.dtype)
        del sel, u
        _graph_break()
        return {"kind": "prefill_dense", "mask": mask, "packed": None, "cb": nctx * bs_, "bl": bl}
    nb = md.block_list.shape[0]
    bs = md.block_mapping.shape[1] if md.block_mapping is not None else q_i.shape[0]
    if nb * bs_ <= topk or nb * bs_ <= MIN_KEYS * bs:
        return None
    kb = buf.unflatten(0, (-1, bs_)).index_select(0, md.block_list)
    if q_i.shape[0] != bs:
        return decode_mask_bias(q_i, w, kb, md.block_groups, md.attn_bias, topk)
    idx = decode_topk(q_i, w, kb, md.block_groups, md.attn_bias, topk)
    ok = idx >= 0
    ic = idx.clamp(min=0).to(torch.long)
    blk = md.block_list.to(torch.long).index_select(0, (ic // bs_).reshape(-1)).view(ic.shape).clamp(min=0)
    phys = blk * bs_ + ic % bs_
    return {"kind": "decode", "phys": phys, "bias": torch.where(ok, 0.0, NEG).to(torch.float32)}


def _wrapper_forward(self, positions, hidden_states, llama_4_scaling=None):
    """MultiHeadLatentAttentionWrapper.forward (the MLA part GLM uses), plus the indexer."""
    from vllm.forward_context import get_forward_context

    q_c = None
    if self.q_lora_rank is not None:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]
    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim :], k_pe)

    sel = None
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = md.get(self.mla_attn.layer_name)
    if md is not None and q_c is not None:
        if getattr(self, "indexer", None) is not None and getattr(self, "_glm_dsa_buf", None) is not None:
            sel = _full_layer(self, md, hidden_states, q_c, positions)
            _STATE["sel"] = sel
        elif getattr(self, "indexer", None) is None and getattr(self, "skip_topk", False):
            sel = _STATE["sel"]  # shared layer: reuse the group's selection
    if llama_4_scaling is not None:
        q *= llama_4_scaling
    _STATE["active"] = sel
    try:
        attn_out = self.mla_attn(
            q, kv_c_normed, k_pe, output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim)
        )
    finally:
        _STATE["active"] = None
    return self.o_proj(attn_out)[0]


def _prefill_dense_impl(self, q, k_c_normed, k_pe, kv_cache, md, sel):
    """Prefill whose layer group is dense-masked: write the latent cache, decompress K/V for every visible slot (as
    HPUMLAImpl.forward_mha does) and run dense attention under the selection mask."""
    impl = self.impl
    slot_mapping = md.slot_mapping.flatten() if md.slot_mapping is not None else None
    latent = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
    latent = latent.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
    impl.latent_cache_k(latent, kv_cache[0], slot_mapping)
    if sel["cb"]:
        past = impl.latent_cache_k.fetch_from_cache(kv_cache[0].unflatten(0, (-1, md.block_size)), sel["bl"])
        latent = torch.concat((past.view(-1, past.shape[-1]), latent), dim=0)
    k_c, k_r = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_nope = impl.kv_b_proj(k_c)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_r.view(-1, 1, self.qk_rope_head_dim).expand((*k_nope.shape[:-1], -1))), dim=-1)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    if self.qk_head_dim > self.v_head_dim:
        v = torch.nn.functional.pad(v, (0, self.qk_head_dim - self.v_head_dim))
    o = dense_masked_attn(q, k, v.contiguous(), sel["mask"], sel["cb"], self.scale, packed=sel["packed"])
    return o[..., : self.v_head_dim].reshape(-1, self.num_heads * self.v_head_dim)


def _forward_impl(
    self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
):
    """HPUMLAAttention.forward_impl: the sparse paths for a dict selection, the original otherwise (dense, or a mask
    tensor that forward_mha / forward_mqa below substitute for the attention bias)."""
    sel = _STATE["active"]
    if isinstance(sel, dict) and sel["kind"] == "prefill_dense":
        return _prefill_dense_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, sel)
    if not isinstance(sel, dict):
        return _ORIG_FORWARD_IMPL(
            self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )
    slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
    latent = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
    latent = latent.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
    self.impl.latent_cache_k(latent, kv_cache[0], slot_mapping)
    kc = kv_cache[0]
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ql = torch.bmm(q_nope.transpose(0, 1).contiguous(), self.W_UK_T).transpose(0, 1)
    q_abs = torch.cat((ql, q_pe), dim=-1).contiguous()
    if sel["kind"] == "prefill":
        o = sparse_prefill_attn(q_abs, kc, sel["phys"], sel["bias"], self.scale, sel["rr"], self.kv_lora_rank)
    else:
        o = sparse_decode_attn(q_abs, kc, sel["phys"], sel["bias"], self.scale, self.kv_lora_rank)
    return self._v_up_proj(o)


def _forward_mha(self, q, latent_vec_k, k_cache, attn_metadata):
    """HPUMLAImpl.forward_mha with a mask selection substituted for the prompt bias."""
    from vllm_gaudi.extension import ops

    if attn_metadata.block_list is not None:
        current = latent_vec_k
        if isinstance(k_cache, tuple):
            k_cache = k_cache[0]
        past = self.latent_cache_k.fetch_from_cache(
            k_cache.unflatten(0, (-1, attn_metadata.block_size)), attn_metadata.block_list
        )
        latent_vec_k = torch.concat((past.view(-1, past.shape[-1]), current), dim=0)
    k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
    kv_nope = self.kv_b_proj(k_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
    batch_size = 1 if self.use_merged_prefill else attn_metadata.seq_lens_tensor.shape[0]
    q = q.view(batch_size, -1, self.num_heads, self.qk_head_dim)
    k = k.view(batch_size, -1, self.num_heads, self.qk_head_dim)
    v = v.view(batch_size, -1, self.num_heads, self.v_head_dim)
    if self.qk_head_dim > self.v_head_dim:
        pad = torch.zeros(*v.shape[:-1], q.shape[-1] - v.shape[-1], device=v.device, dtype=v.dtype)
        v_padded = torch.cat((v, pad), dim=-1)
    else:
        v_padded = v
    sel = _STATE["active"]
    output = ops.prompt_attention(
        impl=self.prefill_impl,
        query=q,
        key=k,
        value=v_padded,
        is_causal=True,
        attn_bias=sel if sel is not None else attn_metadata.attn_bias,
        position_bias=None,
        valid_seq_lengths=attn_metadata.seq_lens_tensor,
        scale=self.scale,
        matmul_qk_op=self.matmul_qk,
        softmax_op=self.softmax,
        matmul_av_op=self.matmul_av,
        keys_fetch_func=self.latent_cache_k.fetch_from_cache,
        values_fetch_func=None,
        fsdpa_op=self.fused_scaled_dot_product_attention,
    )
    output = output.view(batch_size, -1, self.num_heads, q.shape[-1])[..., : v.shape[-1]]
    return output.reshape(-1, self.num_heads * v.shape[-1])


def _forward_mqa(self, q_nope, q_pe, k_cache, attn_metadata):
    """HPUMLAImpl.forward_mqa with a mask selection substituted for the block bias."""
    from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention

    if isinstance(k_cache, tuple):
        k_cache = k_cache[0]
    query = torch.cat([q_nope, q_pe], dim=-1)
    key_cache = k_cache.unsqueeze(1) if k_cache is not None else None
    sel = _STATE["active"]
    return HPUPagedAttention.forward_decode(
        query=query,
        key_cache=key_cache,
        value_cache=None,
        block_list=attn_metadata.block_list,
        block_mapping=attn_metadata.block_mapping,
        block_bias=sel if sel is not None else attn_metadata.attn_bias,
        block_groups=attn_metadata.block_groups,
        block_size=attn_metadata.block_size,
        scale=self.scale,
        matmul_qk_op=self.matmul_qk,
        matmul_av_op=self.matmul_av,
        batch2block_matmul_op=self.batch2block_matmul,
        block2batch_matmul_op=self.block2batch_matmul,
        keys_fetch_func=self.latent_cache_k.fetch_from_cache,
        values_fetch_func=None,
        kv_lora_rank=self.kv_lora_rank,
    )


def install(wrapper_cls, attention_cls):
    """Swap in the DSA methods (called at the end of vllm_gaudi/attention/oot_mla.py)."""
    global _ORIG_FORWARD_IMPL
    if not ENABLED or _ORIG_FORWARD_IMPL is not None:
        return
    from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl

    HPUMLAImpl.forward_mha = _forward_mha
    HPUMLAImpl.forward_mqa = _forward_mqa
    _ORIG_FORWARD_IMPL = attention_cls.forward_impl
    wrapper_cls.forward = _wrapper_forward
    attention_cls.forward_impl = _forward_impl
    print(f"{TAG} sparse attention on above {MIN_KEYS} keys per sequence", flush=True)
