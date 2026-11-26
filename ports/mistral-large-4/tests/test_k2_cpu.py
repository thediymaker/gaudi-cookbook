"""HPU-free unit test for ML4_ATTN_KERNEL (mistral_large4_hpu.k2attn). torch CPU only, no device opened.

  1. k2_g2 emulated on CPU from the cache bytes, step for step (the memoised addressing of k2attn._addr; per key the
     tag check against (owner, clamped position) with the INT_MIN "not wanted" sentinel; int8 x fp32 tile scale in
     fp32, round-to-nearest-even to bf16 by integer ops, k_pe bf16 -> fp32) equals Fp8LatentCache.read(...)[0].float()
     and its valid mask bit for bit: scattered block tables, epochs, stale tags, -1 keys, padding requests, 1-8 rows.
  2. k2attn.absorbed's math on the emulated rows equals ML4Attention.absorbed_mm bit for bit (CPU, same ops).
  3. eligible(): off by default and outside the served codec / decode rows / DCP.
Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<pkg> python tests/test_k2_cpu.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
for k, v in {"ML4_KV_FP8": "1", "ML4_KV_CODEC": "i8-t128-f32", "ML4_TAG_CHECK": "mask"}.items():
    os.environ[k] = v
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml4_hpu"))

import torch  # noqa: E402

from mistral_large4_hpu import attention as A, k2attn, k1ix  # noqa: E402
from mistral_large4_hpu.paged import BatchCtx, latent_cache, latent_width  # noqa: E402

BS, K = 128, 2048
INT_MIN = torch.iinfo(torch.int32).min


def emu_k2_g2(tags, data, idx, prow, pown, tagcheck=1):
    """The kernel, per key: rows fp32 [R, K, 576], valid bool [R, K]."""
    b = data.view(torch.uint8)[prow.long()]  # [R, K, 656] the row's bytes
    q = b[..., :512].contiguous().view(torch.int8).to(torch.int32)
    sc = b[..., 512:528].contiguous().view(torch.float32)  # [R, K, 4]
    f = q.to(torch.float32) * sc.repeat_interleave(128, -1)  # fp32 product (RNE)
    v = f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    v = (v + 0x7FFF + ((v >> 16) & 1)) & 0xFFFF0000  # RNE to bf16, in the fp32 container
    c = (v.to(torch.int64) - ((v >> 31) << 32)).to(torch.int32).view(torch.float32)
    kpe = b[..., 528:656].contiguous().view(torch.bfloat16).to(torch.float32)
    t = tags.view(-1, 2)[prow.long()]
    ok = (t[..., 0] == pown) & (t[..., 1] == idx) if tagcheck else torch.ones_like(pown, dtype=torch.bool)
    ok = ok & (pown != INT_MIN)
    return torch.cat([c, kpe], -1), ok


def world(lens, S, seed, pad=False, stale=0.0, invalid=0.0):
    g = torch.Generator().manual_seed(seed)
    B = len(lens) + (1 if pad else 0)
    nb = max(-(-L // BS) for L in lens)
    pool = sum(-(-L // BS) for L in lens) + 5
    perm = torch.randperm(pool - 1, generator=g) + 1
    bt = torch.full((B, nb), -1, dtype=torch.long)
    ep = torch.zeros(B, nb, dtype=torch.int32)
    pos = torch.full((B, S), -1, dtype=torch.long)
    uid = torch.full((B,), -1, dtype=torch.int32)
    tags = torch.full((pool * BS, 2), -1, dtype=torch.int32)
    j = 0
    for i, L in enumerate(lens):
        n = -(-L // BS)
        bt[i, :n] = perm[j : j + n]
        j += n
        ep[i, :n] = torch.randint(1, 1 << 20, (n,), generator=g, dtype=torch.int32)
        pos[i] = torch.arange(L - S, L)
        uid[i] = i + 1
        for bi in range(n):
            p0, p1 = bi * BS, min(L, (bi + 1) * BS)
            r = torch.arange(int(bt[i, bi]) * BS, int(bt[i, bi]) * BS + p1 - p0)
            tags[r, 0] = int(ep[i, bi])
            tags[r, 1] = torch.arange(p0, p1, dtype=torch.int32)
    if stale:
        m = torch.rand(pool * BS, generator=g) < stale
        tags[m, 0] ^= 0x5A5A
    lat = latent_cache(torch.zeros(pool * BS, latent_width(), dtype=torch.bfloat16), tags, 7)
    x = torch.randn(pool * BS, 576, generator=g) * 0.3
    lat.data.copy_(lat.encode(x.to(torch.bfloat16)))
    tk = torch.full((B, S, K), -1, dtype=torch.long)
    for i in range(len(lens)):
        for s in range(S):
            p = int(pos[i, s])
            if p <= K - 1:
                tk[i, s, : p + 1] = torch.arange(p + 1)
            else:
                tk[i, s] = torch.randperm(p + 1, generator=g)[:K].sort()[0]
            if invalid:
                tk[i, s, torch.rand(K, generator=g) < invalid] = -1
    ctx = BatchCtx(
        positions=pos, block_table=bt, req_uid=uid, block_size=BS, block_epoch=ep, memo={}, host_max_pos=int(pos.max())
    )
    return lat, tk, ctx


def test_rows_bitwise():
    n = 0
    for seed, (lens, S, pad, stale, inv) in enumerate(
        [
            ([1500], 1, False, 0, 0),
            ([4096], 4, False, 0, 0.05),
            ([9000, 3000], 8, True, 0.02, 0.05),
            ([20000], 2, False, 0.05, 0),
            ([700, 2600, 5000], 1, True, 0, 0.1),
        ]
    ):
        lat, tk, ctx = world(lens, S, seed, pad, stale, inv)
        ref, rok = lat.read(ctx, tk.clamp(min=0), tk >= 0)
        idx, prow, pown = k2attn._addr(ctx, tk)
        got, ok = emu_k2_g2(lat.tags, lat.data, idx, prow, pown)
        R = tk.shape[0] * S
        assert torch.equal(ref.float().reshape(R, K, 576).view(torch.int32), got.view(torch.int32)), seed
        assert torch.equal(rok.reshape(R, K), ok), seed
        n += 1
    return n


def test_math_bitwise():
    cfg = SimpleNamespace(hidden_size=6144, num_attention_heads=64)

    def lin(name, n_in, n_out, kind, dtype):
        if kind == "column":
            n_out //= 8
        elif kind == "row":
            n_in //= 8
        return torch.nn.Linear(n_in, n_out, bias=False, dtype=torch.bfloat16)

    n = 0
    for rope in (True, False):
        m = A.ML4Attention(cfg, A.LayerRole(5, rope, 5, False, 0, True), lin, 8)
        with torch.no_grad():
            m.wkv_b.copy_(torch.randn(m.wkv_b.shape) * 0.05)
        lat, tk, ctx = world([5000, 1800], 4, 11 + rope, invalid=0.03)
        q = (torch.randn(2, 4, 8, 128) * 2).to(torch.bfloat16)
        W_UK, W_UV = m.w_uk_uv()
        ref = m.absorbed_mm(q, tk, ctx, lat, W_UK, W_UV, A.softmax_scale())
        orig = k2attn.rows
        k2attn.rows = lambda c, s, t: emu_k2_g2(s.tags, s.data, *k2attn._addr(c, t))
        try:
            got = k2attn.absorbed(m, q, tk, ctx, lat)
        finally:
            k2attn.rows = orig
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16)), rope
        n += 1
    return n


def test_eligible():
    lat, tk, ctx = world([4096], 1, 3)
    q = torch.zeros(1, 1, 8, 128, dtype=torch.bfloat16)
    assert not k2attn.ENABLED and not k1ix.ATTN  # default off
    assert not k2attn.eligible(None, q, tk, lat)  # off / CPU tensors
    k1ix.ATTN, k1ix._ops_loaded = True, True
    try:
        assert not k2attn.eligible(None, q, tk, lat)  # CPU q: never
        qh = SimpleNamespace(device=SimpleNamespace(type="hpu"))
        assert k2attn.eligible(None, qh, tk, lat)
        assert not k2attn.eligible(None, qh, torch.zeros(1, 9, K, dtype=torch.long), lat)  # > DECODE_S rows
        assert not k2attn.eligible(None, qh, tk[..., :1000], lat)  # K % 64
        lat.dcp = (8, 0)
        assert not k2attn.eligible(None, qh, tk, lat)
        lat.dcp = None
        lat.fake = True
        assert not k2attn.eligible(None, qh, tk, lat)
    finally:
        k1ix.ATTN, k1ix._ops_loaded = False, False
    return 1


if __name__ == "__main__":
    print("rows bitwise cases:", test_rows_bitwise())
    print("math bitwise cases:", test_math_bitwise())
    print("eligible:", test_eligible())
    print("OK")
