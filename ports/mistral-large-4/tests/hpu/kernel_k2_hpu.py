"""One-card kernel gate for ML4_ATTN_KERNEL: the env hook
(mxk.setup_env) chains libk1_tpc.so, the op library loads, and ML4Attention.forward with the flag on gives
the same bits as with it off (eager and inside wrap_in_hpu_graph) on RoPE and NoPE layers, 4K / 64K tables, rows 1 / 4 /
8, two requests + padding; a 16-row chunk must take the native path.
Run: python3 tests/hpu/kernel_k2_hpu.py (installed port, one free card)"""

import json
import os
import sys

for k, v in {
    "ML4_KV_CODEC": "i8-t128-f32",
    "ML4_KV_FP8": "1",
    "ML4_TAG_CHECK": "mask",
    "PT_HPU_LAZY_MODE": "1",
    "ML4_ATTN_KERNEL": "1",
}.items():
    os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(1, HERE)
from mistral_large4_hpu import mxk, k1ix, k2attn  # noqa: E402

mxk.setup_env()
env = {
    k: os.environ.get(k) for k in ("GC_KERNEL_PATH", "MXK_TPC_FWD_LIB", "K1_TPC_FWD_LIB", "PT_HPU_LAZY_ACC_PAR_MODE")
}
print("ENV", json.dumps(env), flush=True)
import torch  # noqa: E402
import habana_frameworks.torch as ht  # noqa: E402

assert k1ix.load_ops(), "k1/k2 ops not loaded"
if mxk.ENABLED:
    mxk.load_ops()
print("LIB", k1ix.lib_version(), k1ix.LIB_DIR, "attn", k2attn.ENABLED, "indexer", k1ix.ENABLED, flush=True)
from types import SimpleNamespace  # noqa: E402
from mistral_large4_hpu import attention as A  # noqa: E402
from mistral_large4_hpu.rope import RopeCache  # noqa: E402
from test_k2_cpu import world  # noqa: E402

dev = torch.device("hpu")


def sync():
    ht.core.mark_step()
    torch.hpu.synchronize()


cfg = SimpleNamespace(hidden_size=6144, num_attention_heads=64)


def lin(name, n_in, n_out, kind, dtype):
    if kind == "column":
        n_out //= 8
    elif kind == "row":
        n_in //= 8
    return torch.nn.Linear(n_in, n_out, bias=False, dtype=torch.bfloat16)


rope = RopeCache(262144, alpha=None)
rope.materialize(dev)
calls = {"k2": 0}
_abs = k2attn.absorbed


def counted(*a, **kw):
    calls["k2"] += 1
    return _abs(*a, **kw)


k2attn.absorbed = counted
res, ok_all = [], True
torch.manual_seed(0)
for use_rope in (True, False):
    m = A.ML4Attention(cfg, A.LayerRole(3, use_rope, 3, False, 0, True), lin, 8)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.copy_((torch.randn(p.shape) * p.shape[1] ** -0.5).to(p.dtype))
    m = m.to(dev)
    for Lk in (4096, 65536):
        for S, lens, pad in ((1, [Lk], False), (4, [Lk], False), (8, [Lk, Lk // 3], True), (16, [Lk], False)):
            lat, tk, ctx = world(lens, S, Lk + S + use_rope, pad=pad, invalid=0.03)
            lat.data, lat.tags = lat.data.to(dev), lat.tags.to(dev)
            tk = tk.to(dev)
            for f in ("positions", "block_table", "req_uid", "block_epoch"):
                setattr(ctx, f, getattr(ctx, f).to(dev))
            a = torch.randn(len(ctx.req_uid), S, 6144, device=dev).to(torch.bfloat16)

            def fwd(a, t):
                c = A.Shared()
                c.topk[0] = t
                ctx.memo = {}
                return m(a, ctx, {3: {"latent": lat}}, c, rope)

            k2attn.ENABLED = False
            ref = fwd(a, tk)
            sync()
            k2attn.ENABLED = True
            n0 = calls["k2"]
            got = fwd(a, tk)
            sync()
            used = calls["k2"] - n0

            class P(torch.nn.Module):
                def forward(self, a, t):
                    return fwd(a, t)

            gr = None
            for _ in range(3):
                gp = ht.hpu.wrap_in_hpu_graph(P(), disable_tensor_cache=True)
                gr = gp(a, tk)
                sync()
                if torch.equal(gr.view(torch.int16), ref.view(torch.int16)):
                    break
            d = {
                "rope": use_rope,
                "Lk": Lk,
                "S": S,
                "B": len(ctx.req_uid),
                "k2_used": used,
                "eager_mis": int((got.view(torch.int16) != ref.view(torch.int16)).sum()),
                "graph_mis": int((gr.view(torch.int16) != ref.view(torch.int16)).sum()),
            }
            d["pass"] = d["eager_mis"] == 0 and d["graph_mis"] == 0 and (d["k2_used"] == (1 if S <= 8 else 0))
            ok_all &= d["pass"]
            res.append(d)
            print("SMOKE", json.dumps(d), flush=True)
out = os.environ.get("K2_SMOKE_OUT")
if out:
    json.dump({"env": env, "lib": k1ix.lib_version(), "cases": res, "pass": ok_all}, open(out, "w"), indent=1)
print("SMOKE", "PASS" if ok_all else "FAIL", flush=True)
sys.exit(0 if ok_all else 1)
