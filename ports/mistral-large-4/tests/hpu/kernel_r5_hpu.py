"""r5 kernels (ML4_R5_*) vs the native plugin functions on one HPU, bitwise. Eager lazy and inside wrap_in_hpu_graph.
Run with the package on PYTHONPATH, PT_HPU_LAZY_MODE=1, ML4_INDEX_KERNEL=1 (library registration), ML4_NUMERICS_RW=1:
  python3 tests/hpu/kernel_r5_hpu.py          -> prints one line per check, exit 1 on any mismatch."""

import os
import sys

for k, v in {
    "PT_HPU_LAZY_MODE": "1",
    "ML4_INDEX_KERNEL": "1",
    "ML4_NUMERICS_RW": "1",
    "ML4_KV_FP8": "1",
    "ML4_KV_CODEC": "i8-t128-f32",
    "ML4_INDEX_U8": "1",
    "ML4_TAG_CHECK": "mask",
    "ML4_MXK": "0",
}.items():
    os.environ.setdefault(k, v)
from mistral_large4_hpu import mxk  # noqa: E402

mxk.setup_env()
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import habana_frameworks.torch as ht  # noqa: E402
from mistral_large4_hpu import k1ix, r5k, numerics as NU, rope as RO, paged as PG, moe as MO  # noqa: E402

assert k1ix.load_ops() and r5k.ready(), "r5 kernels not in the loaded library"
dev = torch.device("hpu")
BF = torch.bfloat16
NU.rw_init(dev)
r5k.init(dev)
bad = []


def sync():
    ht.core.mark_step()
    torch.hpu.synchronize()


class _M(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *a):
        return self.f(*a)


def check(name, native, kernel, *args):
    """native / kernel: callables -> tuple of tensors; compared bitwise eager and under an HPU graph."""
    with torch.no_grad():
        ref = [t.cpu() for t in native(*args)]
        sync()
        out = [t.cpu() for t in kernel(*args)]
        sync()
        g = ht.hpu.wrap_in_hpu_graph(_M(lambda *a: tuple(kernel(*a))), disable_tensor_cache=True)
        g(*args)
        sync()
        gout = [t.cpu() for t in g(*args)]
        sync()

    def eq(a, b):  # bitwise (packed cache rows hold NaN bit patterns)
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.is_floating_point():
            a, b = (
                a.view(torch.int16 if a.element_size() == 2 else torch.int32),
                b.view(torch.int16 if b.element_size() == 2 else torch.int32),
            )
        return torch.equal(a, b)

    ok = all(eq(a, b) for a, b in zip(out, ref)) and all(eq(a, b) for a, b in zip(gout, ref))

    def nd(a, b):
        if a.dtype in (torch.bfloat16, torch.float16):
            a, b = a.view(torch.int16), b.view(torch.int16)
        elif a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        return int((a != b).sum()) if a.shape == b.shape else f"shape {tuple(a.shape)} vs {tuple(b.shape)}"

    det = "" if ok else f"  eager {[nd(a, b) for a, b in zip(out, ref)]} graph {[nd(a, b) for a, b in zip(gout, ref)]}"
    print(("PASS " if ok else "FAIL ") + name + det, flush=True)
    if not ok:
        bad.append(name)


gen = torch.Generator().manual_seed(0)
rc = RO.RopeCache(1 << 18, (torch.rand(32, generator=gen) * 1.2 - 0.35).to(BF))
rc.materialize(dev)
for B, S in ((1, 1), (1, 4), (4, 1), (4, 4)):
    N = B * S
    pos = torch.randint(0, 200000, (B, S), generator=gen).to(dev)
    cos, sin, lp = rc.at(pos)
    x = (torch.randn(B, S, 6144, generator=gen) * 2).to(BF).to(dev)
    w = (0.5 + torch.rand(6144, generator=gen)).to(BF).to(dev)
    check(f"rms 6144 B{B} S{S}", lambda x, w: (NU.rms_norm(x, w, 1e-5),), lambda x, w: (r5k.rms(x, w, 1e-5),), x, w)
    kv = (torch.randn(B, S, 576, generator=gen) * 2).to(BF).to(dev)
    w5 = (0.5 + torch.rand(512, generator=gen)).to(BF).to(dev)
    check(
        f"rms 512-of-576 B{B} S{S}",
        lambda kv, w: (NU.rms_norm(kv[..., :512], w, 1e-5),),
        lambda kv, w: (r5k.rms(kv, w, 1e-5, 512),),
        kv,
        w5,
    )
    for H in (8, 32):
        q = (torch.randn(B, S, H, 128, generator=gen) * 2).to(BF).to(dev)
        check(
            f"rope_head H{H} B{B} S{S}",
            lambda q: (RO.rope_head(q, cos.unsqueeze(-2), sin.unsqueeze(-2)),),
            lambda q: (r5k.rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2), 0),),
            q,
        )
        check(
            f"rope_tail H{H} B{B} S{S}",
            lambda q: (RO.rope_tail(q, cos.unsqueeze(-2), sin.unsqueeze(-2)),),
            lambda q: (r5k.rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2), 64),),
            q,
        )
        check(f"fake_fp8 H{H} B{B} S{S}", lambda q: (NU.fake_fp8_ue8m0(q),), lambda q: (r5k.ff8(q),), q)
        s = (torch.rand(H, generator=gen) * 0.3).to(BF).to(dev)
        b = (0.5 + torch.rand(H, generator=gen)).to(BF).to(dev)
        # SSL=f32 is defined by its consumer: q * lambda in fp32, then bf16 (numerics.ssmax_apply)
        r5k.SSL = "f32"
        check(
            f"ssmax f32 H{H} B{B} S{S}",
            lambda q: (NU.ssmax_apply(q, NU.ssmax_lambda(lp, s, b)),),
            lambda q: (NU.ssmax_apply(q, r5k.ssl(lp, s, b)),),
            q,
        )
        r5k.SSL = "0"
    k = (torch.randn(B, S, 128, generator=gen)).to(BF).to(dev)
    check(f"rope_head k B{B} S{S}", lambda k: (RO.rope_head(k, cos, sin),), lambda k: (r5k.rope(k, cos, sin, 0),), k)
    for rope in (1, 0):
        kvw = kv if rope else kv[..., :512].contiguous()
        check(
            f"latent row rope{rope} B{B} S{S}",
            lambda kv: (
                PG.fp8_latent_pack(
                    torch.cat(
                        [
                            NU.rms_norm(kv[..., :512], w5, 1e-5),
                            RO.rotate(kv[..., 512:], cos, sin) if rope else torch.zeros(B, S, 64, dtype=BF, device=dev),
                        ],
                        -1,
                    )
                ),
            ),
            lambda kv: (r5k.latent(kv, w5, 1e-5, cos, sin, bool(rope)),),
            kvw,
        )
    gw = (torch.randn(128, 6144, generator=gen) * 6144**-0.5).to(BF).to(dev)
    f = x.reshape(N, 6144)
    off = torch.tensor([0, 128], dtype=torch.long, device=dev)

    def nat_route(f):  # ids and the bf16 weights as moe._forward_decode hands them to mxk
        w_, ids = MO.route(f, gw)
        return ids, w_.unsqueeze(-1).expand(*w_.shape, 2).flatten(-2).to(BF)

    def ker_route(f):  # vid = [i0, i0 + 128, i1, ...]: even entries are the ids, odd = even + 128 (checked below)
        vid, vw = r5k.router(torch.softmax(F.linear(f.float(), gw.float()), dim=-1))
        return vid[:, 0::2].long(), vw

    check(f"router T{N}", nat_route, ker_route, f)
    with torch.no_grad():
        vid, _ = r5k.router(torch.softmax(F.linear(f.float(), gw.float()), dim=-1))
        vid = vid.cpu()
    if not torch.equal(vid[:, 1::2], vid[:, 0::2] + 128):
        print("FAIL router vid halves", flush=True)
        bad.append("router vid halves")
    # k2_g3 vs the native addressing + k2_g2 on a paged latent cache with B requests
    if B * S <= 16:
        BS, NB = 128, 40
        nrows = (B * NB + 8) * BS
        data = (torch.randint(-128, 128, (nrows, 656), generator=gen, dtype=torch.int8)).view(BF).to(dev)
        data.view(torch.uint8)[:, 512:528] = (
            torch.rand(nrows, 4, generator=gen).to(dev).view(torch.uint8).view(nrows, 16)
        )
        perm = torch.randperm(B * NB + 7, generator=gen)[: B * NB] + 1
        bt = perm.view(B, NB).long()
        bt[0, -3:] = -1  # unallocated blocks
        ep = torch.randint(1, 1 << 20, (B, NB), generator=gen, dtype=torch.int32)
        tags = torch.full((nrows, 2), -1, dtype=torch.int32)
        for bb in range(B):
            for i in range(NB):
                if bt[bb, i] >= 0:
                    r0 = int(bt[bb, i]) * BS
                    tags[r0 : r0 + BS, 0] = int(ep[bb, i]) if (i + bb) % 7 else 3  # some stale owners
                    tags[r0 : r0 + BS, 1] = torch.arange(i * BS, i * BS + BS, dtype=torch.int32)
        K = 2048
        tk = torch.stack([torch.randperm(NB * BS, generator=gen)[:K].sort()[0] for _ in range(N)]).view(B, S, K)
        tk[:, :, -5:] = -1
        tk32 = tk.to(torch.int32).to(dev)
        bt_d, ep_d, tags_d = bt.to(dev), ep.to(dev), tags.to(dev)
        src = PG.Fp8LatentCache(data, tags_d, "latent", 0)
        ctx = PG.BatchCtx(
            positions=torch.zeros(B, S, dtype=torch.long, device=dev),
            block_table=bt_d,
            req_uid=torch.ones(B, dtype=torch.int32, device=dev),
            block_size=BS,
            block_epoch=ep_d,
            memo={},
        )
        from mistral_large4_hpu import k2attn

        def nat(t32):
            ctx.memo = {}
            r, v = k2attn.rows(ctx, src, t32.long())
            return r.view(torch.int32), v

        def ker(t32):
            ctx.memo = {}
            t64 = t32.long()
            ctx.memo[("k1_t32", id(t64))] = t32
            r5k.K2FOLD = True
            try:
                r, v = k2attn.rows(ctx, src, t64)
            finally:
                r5k.K2FOLD = False
            return r.view(torch.int32), v

        check(f"k2_g3 B{B} S{S}", nat, ker, tk32)
print("FAILED: " + ", ".join(bad) if bad else "ALL PASS", flush=True)
sys.exit(1 if bad else 0)
