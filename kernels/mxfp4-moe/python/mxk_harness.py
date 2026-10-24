# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the parity test and the micro benchmark (not installed into vLLM).

* expert weights of one gpt-oss layer, from a local Hugging Face snapshot (openai/gpt-oss-120b or openai/gpt-oss-20b)
  or random MXFP4 weights of the same shapes;
* decode routing cases;
* a CPU fp32 reference of the gpt-oss MoE block;
* runners for the stock vllm-gaudi op and for the kernels.
"""

import glob
import json
import os
import random

ALPHA, LIMIT = 1.702, 7.0  # gpt-oss SwiGLU-OAI
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# (tokens, routing): "rand" = top-k of all experts per token, "hot" = every route on a local expert,
# "real" = the layer's own router on the activation rows (needs a checkpoint; random router otherwise)
CASES = [(1, "rand"), (2, "rand"), (4, "rand"), (8, "rand"), (16, "rand"), (32, "rand"), (1, "hot"), (8, "hot")]
REAL_CASES = [(4, "real"), (32, "real")]
X_RMS = [1.0, 4.0]  # 4 drives the SwiGLU clamps


def snapshot_dir(path):
    """A snapshot directory (with model.safetensors.index.json) from a snapshot or a hub cache model directory."""
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return path
    snaps = sorted(glob.glob(os.path.join(path, "snapshots", "*", "")))
    for s in snaps:
        if os.path.isfile(os.path.join(s, "model.safetensors.index.json")):
            return s
    raise FileNotFoundError(f"no model.safetensors.index.json under {path}")


def load_layer(checkpoint, layer, e_local, e_min):
    """Experts e_min .. e_min + e_local - 1 of one gpt-oss layer, in the layout vLLM keeps on the device."""
    from safetensors import safe_open

    ck = snapshot_dir(checkpoint)
    with open(os.path.join(ck, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    with open(os.path.join(ck, "config.json")) as f:
        cfg = json.load(f)
    k, inter = cfg["hidden_size"], cfg["intermediate_size"]

    def get(name, sl):
        with safe_open(os.path.join(ck, idx[name]), "pt") as f:
            return f.get_slice(name)[sl]

    pre = f"model.layers.{layer}.mlp."
    sl = slice(e_min, e_min + e_local)
    d = {
        "w13": get(pre + "experts.gate_up_proj_blocks", sl).reshape(e_local, 2 * inter, k // 2).contiguous(),
        "s13": get(pre + "experts.gate_up_proj_scales", sl).contiguous(),
        "b13": get(pre + "experts.gate_up_proj_bias", sl).contiguous(),
        "w2": get(pre + "experts.down_proj_blocks", sl).reshape(e_local, k, inter // 2).contiguous(),
        "s2": get(pre + "experts.down_proj_scales", sl).contiguous(),
        "b2": get(pre + "experts.down_proj_bias", sl).contiguous(),
        "rw": get(pre + "router.weight", slice(None)).float(),
        "rb": get(pre + "router.bias", slice(None)).float(),
    }
    topk = cfg.get("num_experts_per_tok", cfg.get("experts_per_token"))
    d.update(k=k, inter=inter, e_global=cfg["num_local_experts"], topk=topk)
    check_layer(d, e_local)
    return d


def random_layer(e_local, k=2880, inter=2880, e_global=128, topk=4, seed=0):
    """Random MXFP4 experts with gpt-oss shapes. Scales near 2^-6 keep the activations in the clamp range."""
    import torch

    g = torch.Generator().manual_seed(seed)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.int32).to(torch.uint8)

    d = {
        "w13": u8(e_local, 2 * inter, k // 2),
        "s13": u8(e_local, 2 * inter, k // 32, lo=118, hi=123),
        "b13": (torch.randn(e_local, 2 * inter, generator=g) * 0.5).to(torch.bfloat16),
        "w2": u8(e_local, k, inter // 2),
        "s2": u8(e_local, k, inter // 32, lo=118, hi=123),
        "b2": (torch.randn(e_local, k, generator=g) * 0.5).to(torch.bfloat16),
        "rw": torch.randn(e_global, k, generator=g) * 0.02,
        "rb": torch.zeros(e_global),
    }
    d.update(k=k, inter=inter, e_global=e_global, topk=topk)
    check_layer(d, e_local)
    return d


def check_layer(d, e_local):
    k, inter = d["k"], d["inter"]
    assert d["w13"].shape == (e_local, 2 * inter, k // 2), d["w13"].shape
    assert d["s13"].shape == (e_local, 2 * inter, k // 32), d["s13"].shape
    assert d["b13"].shape == (e_local, 2 * inter), d["b13"].shape
    assert d["w2"].shape == (e_local, k, inter // 2), d["w2"].shape
    assert d["s2"].shape == (e_local, k, inter // 32), d["s2"].shape
    assert d["b2"].shape == (e_local, k), d["b2"].shape


def make_case(d, tokens, mode, x_rms, e_min, e_local):
    """(x bf16 [T, K], ids i32 [T, topk], tw f32 [T, topk]) for one case, deterministic per (T, mode, x_rms)."""
    import torch

    k, topk, eg = d["k"], d["topk"], d["e_global"]
    rng = random.Random(7919 * tokens + 31 * int(x_rms * 10) + {"rand": 0, "hot": 1, "real": 2}[mode])
    gen = torch.Generator().manual_seed(rng.randrange(1 << 30))
    x = (torch.randn(tokens, k, generator=gen) * x_rms).to(torch.bfloat16)
    if mode == "real":  # gpt-oss routing: top-k of the router logits, softmax over the k
        logits = x.float() @ d["rw"].t() + d["rb"]
        tv, ti = torch.topk(logits, topk, dim=-1)
        return x, ti.to(torch.int32), torch.softmax(tv, -1)
    pool = range(e_min, e_min + e_local) if mode == "hot" else range(eg)
    ids = torch.tensor([rng.sample(pool, topk) for _ in range(tokens)], dtype=torch.int32)
    s = torch.rand(tokens, topk, generator=gen) + 0.05
    return x, ids, s / s.sum(-1, keepdim=True)


def dequant(w, s):
    """Packed E2M1 u8 [N, K/2] + E8M0 u8 [N, K/32] -> fp32 [N, K] (low nibble = even k)."""
    import torch

    lut = torch.tensor(E2M1)
    n, kh = w.shape
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1).reshape(n, kh * 2)
    return v * torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, 1)


def reference(d, x, ids, tw, e_min, e_local, alpha=ALPHA, limit=LIMIT):
    """CPU fp32 reference: dequantised weights, fp32 x . W + bias, fp32 SwiGLU-OAI, bf16 intermediate (what both
    device paths store), fp32 w2 + bias, bf16 routing weights. Returns (y fp32 [T, K], local route count)."""
    import torch

    tokens, k = x.shape
    twb = tw.to(torch.bfloat16).float()
    y = torch.zeros(tokens, k)
    cache, local = {}, 0
    for t in range(tokens):
        for j in range(ids.shape[1]):
            e = int(ids[t, j]) - e_min
            if not 0 <= e < e_local:
                continue
            local += 1
            if e not in cache:
                cache[e] = (dequant(d["w13"][e], d["s13"][e]), dequant(d["w2"][e], d["s2"][e]))
            a13, a2 = cache[e]
            h = x[t].float() @ a13.t() + d["b13"][e].float()
            g, u = h[0::2].clamp(max=limit), h[1::2].clamp(-limit, limit)
            a = (g * torch.sigmoid(alpha * g) * (u + 1.0)).to(torch.bfloat16).float()
            y[t] += twb[t, j] * (a @ a2.t() + d["b2"][e].float())
    return y, local


def rel_l2(a, b):
    """||a - b|| / ||b||"""
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def device_layer(d, impl, dev="hpu"):
    """Device tensors for one implementation: the served tensors plus, for the kernels, the re-laid biases."""
    import mxfp4_tpc_moe_rt as rt

    t = {n: d[n].to(dev) for n in ("w13", "s13", "b13", "w2", "s2", "b2")}
    if impl != "stock":
        t["b13k"] = rt.bias13(d["b13"], d["inter"]).to(dev)
        t["b2k"] = rt.bias2(d["b2"]).to(dev)
    return t


def stock_runner(t, d, e_local, e_min):
    """The stock op as vllm-gaudi serves gpt-oss (bias_mxfp4_fused_weights through VllmMixtureOfExpertsOpMXFP4)."""
    from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4

    op = VllmMixtureOfExpertsOpMXFP4(d["e_global"], e_local, e_min, e_min + e_local - 1, block_size=32, has_bias=True)
    for e in range(e_local):
        op.w13_list[e].set_weight(t["w13"][e])
        op.w13_list[e].set_scale(t["s13"][e])
        op.w13_list[e].set_bias(t["b13"][e])
        op.w2_list[e].set_weight(t["w2"][e])
        op.w2_list[e].set_scale(t["s2"][e])
        op.w2_list[e].set_bias(t["b2"][e])
    op._cache_weight_lists()

    def run(x, ids, tw):
        return op(x, ids, tw, permuted_weights=True, activation="silu")

    return run


def kernel_runner(t, d, e_min):
    import mxfp4_tpc_moe_rt as rt

    topk = d["topk"]

    def run(x, ids, tw):
        return rt.moe_decode(
            x, ids, tw, t["w13"], t["s13"], t["b13k"], t["w2"], t["s2"], t["b2k"], topk, e_min, ALPHA, LIMIT
        )

    return run


def hpu_env_for_kernels(lib_dir):
    """Process env for the kernels; call before torch / habana_frameworks are imported."""
    os.environ["GC_KERNEL_PATH"] = os.path.join(lib_dir, "libmxk_tpc.so")
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def time_hpu_graph(fn, warm=3, reps=30):
    """Median wall time in us of one replay of an HPU graph that captures fn()."""
    import statistics
    import time

    import habana_frameworks.torch as ht

    for _ in range(2):
        fn()
        ht.hpu.synchronize()
    g = ht.hpu.HPUGraph()
    s = ht.hpu.Stream()
    with ht.hpu.stream(s):
        g.capture_begin()
        out = fn()  # keep the output: a graph without outputs is optimised away
        g.capture_end()
    for _ in range(warm):
        g.replay()
    ht.hpu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g.replay()
        ht.hpu.synchronize()
        ts.append(time.perf_counter() - t0)
    del out
    return statistics.median(ts) * 1e6
