"""One-card kernel gate for ML4_INDEX_KERNEL: the env hook (mxk.setup_env, with the
image's mxk MoE library unless ML4_MXK=0) chains libk1_tpc.so, the op library loads, and the plugin's
ML4Indexer.forward with ML4_INDEX_KERNEL=1 returns the same indices as with the flag off (eager and inside
wrap_in_hpu_graph), on 4K / 64K tables, rows 1 / 4 / 8, plus a short-table batch that must take the native path.
Run: python3 tests/hpu/kernel_k1_hpu.py (installed port, one free card)"""

import json
import os
import sys

for k, v in {
    "ML4_INDEX_DECODE_FAST": "1",
    "ML4_INDEX_U8": "1",
    "ML4_KV_CODEC": "i8-t128-f32",
    "ML4_KV_FP8": "1",
    "ML4_TAG_CHECK": "mask",
    "PT_HPU_LAZY_MODE": "1",
    "ML4_INDEX_KERNEL": "1",
}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ml4_hpu"))
from mistral_large4_hpu import mxk, k1ix  # noqa: E402

mxk.setup_env()  # what patches/hpu_mxk_env_v30.py runs before the device opens
env = {
    k: os.environ.get(k) for k in ("GC_KERNEL_PATH", "MXK_TPC_FWD_LIB", "K1_TPC_FWD_LIB", "PT_HPU_LAZY_ACC_PAR_MODE")
}
print("ENV", json.dumps(env), flush=True)
import torch  # noqa: E402
import habana_frameworks.torch as ht  # noqa: E402

assert k1ix.load_ops(), "k1 ops not loaded"
if mxk.ENABLED:
    mxk.load_ops()
print("K1", k1ix.lib_version(), k1ix.LIB_DIR, "mxk", mxk.ENABLED and mxk.lib_version(), flush=True)
from types import SimpleNamespace  # noqa: E402
from mistral_large4_hpu import indexer as IX  # noqa: E402
from mistral_large4_hpu.numerics import fake_fp8_ue8m0  # noqa: E402
from mistral_large4_hpu.paged import BatchCtx, index_cache, index_u8_pack  # noqa: E402
from mistral_large4_hpu.rope import RopeCache  # noqa: E402

dev = torch.device("hpu")
BS, D = 128, 512


def sync():
    ht.core.mark_step()
    torch.hpu.synchronize()


torch.manual_seed(0)
ix = IX.ML4Indexer(
    SimpleNamespace(hidden_size=D),
    0,
    True,
    lambda name, i, o, kind, dt: torch.nn.Linear(i, o, bias=False).to(torch.bfloat16),
).to(dev)
with torch.no_grad():
    for p in ix.parameters():
        if p.dim() == 2:
            p.copy_((torch.randn(p.shape) * 0.05).to(p.dtype))
rope = RopeCache(262144, alpha=None)
rope.materialize(dev)
calls = {"k1": 0}
_sel = k1ix.select


def counted(*a, **kw):
    calls["k1"] += 1
    return _sel(*a, **kw)


k1ix.select = counted
res = []
for Lk, S, short in [
    (4096, 1, False),
    (4096, 4, False),
    (4096, 8, False),
    (65536, 1, False),
    (65536, 4, False),
    (4096, 1, True),
]:
    nb = Lk // BS
    kk = fake_fp8_ue8m0(torch.randn(Lk, 128, device=dev).to(torch.bfloat16))
    data = torch.zeros((nb + 1) * BS, 66, dtype=torch.bfloat16, device=dev)
    data[BS:] = index_u8_pack(kk)
    tags = torch.full(((nb + 1) * BS, 2), -1, dtype=torch.int32, device=dev)
    tags[BS:, 0] = 1
    tags[BS:, 1] = torch.arange(Lk, dtype=torch.int32, device=dev)
    L = 1000 if short else Lk
    P = torch.arange(L - S, L, device=dev).view(1, S)
    BT = torch.arange(1, nb + 1, device=dev).view(1, nb)
    A = (torch.randn(1, S, D, device=dev) * 0.5).to(torch.bfloat16)

    def run(flag, cache_copy):
        k1ix.ENABLED = flag
        ctx = BatchCtx(
            positions=P,
            block_table=BT,
            req_uid=torch.ones(1, dtype=torch.int32, device=dev),
            block_size=BS,
            block_epoch=torch.ones(1, nb, dtype=torch.int32, device=dev),
            memo={},
            host_max_pos=L - 1,
        )
        return ix(A, ctx, {"index_k": cache_copy}, rope)

    with torch.no_grad():
        c0 = calls["k1"]
        ref = run(False, index_cache(data.clone(), tags.clone(), 0)).cpu()
        got = run(True, index_cache(data.clone(), tags.clone(), 0))
        sync()
        got = got.cpu()
        used_eager = calls["k1"] - c0

        class M(torch.nn.Module):
            def forward(self, d_, t_):
                return run(True, index_cache(d_, t_, 0))

        g = ht.hpu.wrap_in_hpu_graph(M(), disable_tensor_cache=True)
        d2, t2 = data.clone(), tags.clone()
        g(d2, t2)
        sync()
        gg = g(d2, t2)
        sync()
        gg = gg.cpu()
    r = dict(
        Lk=Lk,
        S=S,
        short=short,
        eager_mismatch=int((ref != got).sum()),
        graph_mismatch=int((ref != gg).sum()),
        kernel_used=used_eager > 0,
        n=ref.numel(),
    )
    print(json.dumps(r), flush=True)
    res.append(r)
k1ix.ENABLED = True
ok = all(r["eager_mismatch"] == 0 and r["graph_mismatch"] == 0 and r["kernel_used"] == (not r["short"]) for r in res)
print("SMOKE", "PASS" if ok else "FAIL", flush=True)
out = os.environ.get("OUT")
if out:
    json.dump(
        {"env": env, "k1": k1ix.lib_version(), "mxk": mxk.ENABLED and mxk.lib_version(), "cases": res, "pass": ok},
        open(out, "w"),
        indent=1,
    )
sys.exit(0 if ok else 1)
