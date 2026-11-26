"""r6 kernels (ML4_R6_*) vs the native plugin functions on one HPU, bitwise. Eager lazy and inside wrap_in_hpu_graph.
Run with the package on PYTHONPATH, PT_HPU_LAZY_MODE=1, ML4_INDEX_KERNEL=1 (library registration), ML4_NUMERICS_RW=1:
  python3 tests/hpu/kernel_r6_hpu.py     -> one line per check, exit 1 on any mismatch. env MODES=0,1,2,4 (r6_ffk forms)."""

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
import habana_frameworks.torch as ht  # noqa: E402
from mistral_large4_hpu import k1ix, r5k, r6k, numerics as NU, paged as PG  # noqa: E402

assert k1ix.load_ops() and r5k.ready() and r6k.ready(), "r6 kernels not in the loaded library"
dev = torch.device("hpu")
BF = torch.bfloat16
NU.rw_init(dev)
r5k.init(dev)
r6k.init(dev)
bad = []
MODES = [int(m) for m in os.environ.get("MODES", "0").split(",")]
NROWS = int(os.environ.get("NROWS", "65536"))


def sync():
    ht.core.mark_step()
    torch.hpu.synchronize()


class _M(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *a):
        return self.f(*a)


def bits(t):
    if t.is_floating_point():
        return t.view(torch.int16 if t.element_size() == 2 else torch.int32)
    return t


def check(name, native, kernel, *args, native_graph=False):
    """native / kernel -> tuple of tensors; kernel eager and graph vs native (eager, or graph when native_graph: the
    compiler's own evaluation, e.g. a dropped cast)."""
    with torch.no_grad():
        if native_graph:
            gn = ht.hpu.wrap_in_hpu_graph(_M(lambda *a: tuple(native(*a))), disable_tensor_cache=True)
            gn(*args)
            sync()
            ref = [bits(t).cpu() for t in gn(*args)]
            sync()
        else:
            ref = [bits(t).cpu() for t in native(*args)]
            sync()
        out = [bits(t).cpu() for t in kernel(*args)]
        sync()
        g = ht.hpu.wrap_in_hpu_graph(_M(lambda *a: tuple(kernel(*a))), disable_tensor_cache=True)
        g(*args)
        sync()
        gout = [bits(t).cpu() for t in g(*args)]
        sync()

    def nd(a, b):
        return (
            int((a != b).sum())
            if a.shape == b.shape and a.dtype == b.dtype
            else f"{a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
        )

    de = [nd(a, b) for a, b in zip(out, ref)]
    dg = [nd(a, b) for a, b in zip(gout, ref)]
    ok = all(d == 0 for d in de + dg)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  eager {de} graph {dg}"), flush=True)
    if not ok:
        bad.append(name)
    return ok


gen = torch.Generator().manual_seed(0)


def krows(M, dtype):
    """index-K-like rows: heavy-tailed magnitudes (exp of a uniform over 30 binades), zeros, -0, all-zero rows,
    rows whose amax lands on 448 * 2^k after fake_fp8."""
    mag = torch.exp2(torch.rand(M, 128, generator=gen) * 30 - 24)
    x = mag * torch.randn(M, 128, generator=gen).sign() * (0.5 + torch.rand(M, 128, generator=gen))
    x[torch.rand(M, 128, generator=gen) < 0.02] = 0.0
    x[torch.rand(M, 128, generator=gen) < 0.01] = -0.0
    x[: M // 64] = 0.0
    sc = torch.exp2(torch.randint(-20, 10, (M // 8, 1), generator=gen).float())
    x[M // 64 : M // 64 + M // 8] = torch.randn(M // 8, 128, generator=gen) * sc
    x[M // 64 : M // 64 + M // 8, 0] = 448.0 * sc.flatten()
    return x.to(dtype).to(dev)


def nat_pack_bf16(k):
    return (PG.index_u8_pack(NU.fake_fp8_ue8m0(k)).view(torch.int16),)


for mode in MODES:
    r6k.FFK_MODE = mode
    for M in (1, 4, 32, NROWS):
        k = krows(M, BF)
        check(
            f"r6_ffk bf16 M{M} mode{mode}",
            nat_pack_bf16,
            lambda k: (r6k.ffk(k.view(1, M, 128)).view(M, 66).view(torch.int16),),
            k,
        )
        kf = krows(M, torch.float32)
        # fp32 k (NoPE): native in a graph (the compiler drops kf.to(bf16).float() ahead of fake_fp8) and the r5 path
        check(
            f"r6_ffkf f32 M{M} mode{mode} vs native graph",
            lambda kf: nat_pack_bf16(kf.to(BF)),
            lambda kf: (r6k.ffk(kf.view(1, M, 128)).view(M, 66).view(torch.int16),),
            kf,
            native_graph=True,
        )
        check(
            f"r6_ffkf f32 M{M} mode{mode} vs r5_ff8f + pack",
            lambda kf: (PG.index_u8_pack(r5k.ff8_f32(kf)).view(torch.int16),),
            lambda kf: (r6k.ffk(kf.view(1, M, 128)).view(M, 66).view(torch.int16),),
            kf,
        )
r6k.FFK_MODE = 0

for B, S in ((1, 1), (1, 4), (4, 1), (4, 4), (8, 4)):
    x = (torch.randn(B, S, 6144, generator=gen) * 2).to(BF).to(dev)
    w = (0.5 + torch.rand(6144, generator=gen)).to(BF).to(dev)
    r6k.DUAL = True

    def ker(x, w):
        y = r6k.rms_dual(x, w, 1e-5)
        return y, r6k.take(y)

    def nat(x, w):
        xf = x.float()
        t = NU.round_bf16(xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5))
        p = t * w.float()
        return NU.rms_norm(x, w, 1e-5), p

    check(f"r6_rmsd B{B} S{S}", nat, ker, x, w)
    # the consumer: the indexer's layer_norm on the dropped-cast value (native graph) vs on the kernel's fp32 product
    lw = (1 + 0.1 * torch.randn(6144, generator=gen)).to(BF).to(dev)
    lb = (0.02 * torch.randn(6144, generator=gen)).to(BF).to(dev)
    check(
        f"r6_rmsd -> layer_norm B{B} S{S} vs native graph",
        lambda x, w: (NU.layer_norm(NU.rms_norm(x, w, 1e-5), lw, lb, 1e-6),),
        lambda x, w: (torch.nn.functional.layer_norm(ker(x, w)[1], (6144,), lw.float(), lb.float(), 1e-6).to(BF),),
        x,
        w,
        native_graph=True,
    )
    r6k.DUAL = False
# ML4_R6_ROPEFMA: r5_rope / r5_lat with the fma form vs a CPU emulation (fl(x0 c) / fl(x0 s), then one fused rounding),
# and the separate form vs the CPU separate form (proves the TPC mac is fused and the form bit is honoured)
from mistral_large4_hpu import rope as RO  # noqa: E402

rc = RO.RopeCache(1 << 18, (torch.rand(32, generator=gen) * 1.2 - 0.35).to(BF))
rc.materialize(dev)


def cpu_rot(x, c, s, form):
    x0, x1 = x[..., 0::2].double(), x[..., 1::2].double()
    c, s = c.double(), s.double()
    if form:
        y0 = ((x0 * c).float().double() - x1 * s).float()
        y1 = ((x0 * s).float().double() + x1 * c).float()
    else:
        y0 = (x0 * c).float() - (x1 * s).float()
        y1 = (x1 * c).float() + (x0 * s).float()
    return torch.stack([y0, y1], -1).flatten(-2).to(BF)


for form in (0, 1):
    r6k.ROPEFMA, r6k.ROPEFMA_LO, r6k.ROPEFMA_HI = True, (1 << 30) if form else 0, (1 << 30) if form else 0
    for B, S, H in ((1, 1, 8), (4, 4, 8), (1, 1, 32), (8, 4, 32)):
        pos = torch.randint(0, 200000, (B, S), generator=gen).to(dev)
        cos, sin, _ = rc.at(pos)
        x = (torch.randn(B, S, H, 128, generator=gen) * 2).to(BF).to(dev)
        with torch.no_grad():
            y = r5k.rope(x, cos.unsqueeze(-2), sin.unsqueeze(-2), 64).cpu()
            sync()
        ref = cpu_rot(x.cpu()[..., 64:].float(), cos.cpu().unsqueeze(-2), sin.cpu().unsqueeze(-2), form)
        ok = torch.equal(y[..., 64:].view(torch.int16), ref.view(torch.int16)) and torch.equal(
            y[..., :64], x.cpu()[..., :64]
        )
        print(("PASS " if ok else "FAIL ") + f"r5_rope form{form} B{B} S{S} H{H} vs cpu", flush=True)
        if not ok:
            bad.append(f"rope form{form} {B} {S} {H}")
        kv = (torch.randn(B, S, 576, generator=gen) * 2).to(BF).to(dev)
        w5 = (0.5 + torch.rand(512, generator=gen)).to(BF).to(dev)
        with torch.no_grad():
            row = r5k.latent(kv, w5, 1e-5, cos, sin, True).cpu()
            sync()
        kref = cpu_rot(kv.cpu()[..., 512:].float(), cos.cpu(), sin.cpu(), form)
        ok = torch.equal(row.view(torch.int16)[..., 264:], kref.view(torch.int16))
        print(("PASS " if ok else "FAIL ") + f"r5_lat k_pe form{form} B{B} S{S} vs cpu", flush=True)
        if not ok:
            bad.append(f"lat form{form} {B} {S}")
r6k.ROPEFMA = False
print("FAILED: " + ", ".join(bad) if bad else "ALL PASS", flush=True)
sys.exit(1 if bad else 0)
