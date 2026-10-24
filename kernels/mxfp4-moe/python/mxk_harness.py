# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the parity test and the micro benchmark (not installed into vLLM).

Per build variant (``VARIANTS``): the per-rank shapes the model serves, the expert weights of one layer (from a local
checkpoint, or random MXFP4 weights of the same shapes), decode routing cases, a CPU fp32 reference of the MoE
block, and runners for the stock vllm-gaudi op and for the kernels.
"""

import glob
import json
import os
import random

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
NO_LIMIT = 3.0e38
ALPHA, LIMIT = 1.702, 7.0  # gpt-oss SwiGLU-OAI


def _spec(**kw):
    s = dict(
        layout="cat",  # w13 rows: "cat" = [gate (I); up (I)], "il" = interleaved [g0, u0, g1, u1, ...]
        bias=False,  # gpt-oss expert biases
        act=(1.0, NO_LIMIT, 0.0),  # (alpha, limit, beta) of min(g, L) * sigmoid(alpha min(g, L)) * (clamp(u) + beta)
        w2="sum",  # "sum" = routing-weighted sum per token, "route" = one row per route
        scale_shift=0,  # E8M0 shift the model serves on w13 (DeepSeek: +7)
        route_scale=1.0,  # routing weights are normalised to sum to this
        prefill=(),  # token counts of the prefill unpack path
        chunk=256,  # chunk_size of the stock op (the production manifests' VLLM_MOE_CHUNK); 0 = none
        x_rms=(1.0,),
    )
    s.update(kw)
    return s


# Per-rank shapes as served: K hidden, I intermediate per rank, e_global experts, topk, e_local experts per rank.
VARIANTS = {
    # gpt-oss-120b TP2 + EP: 64 of 128 experts per card; prefill and decode above 32 tokens keep the stock op
    "gptoss": _spec(
        k=2880,
        inter=2880,
        e_global=128,
        topk=4,
        e_local=64,
        layout="il",
        bias=True,
        act=(ALPHA, LIMIT, 1.0),
        chunk=0,
        max_tokens=32,
        layer=18,
        x_rms=(1.0, 4.0),
    ),
    # GLM-5.3 / GLM-5.2, EP4 x TP2: 64 experts x half the intermediate dim (2048 / 2) per card
    "glm": _spec(
        k=6144, inter=1024, e_global=256, topk=8, e_local=64, max_tokens=16, layer=40, prefill=(256, 1024, 4096)
    ),
    # DeepSeek-V4.1-Flash EP8: 48 of 384 experts; SwiGLU clamped at 10, routed scale 1.5, w13 scales served +7
    "dsv41": _spec(
        k=5120,
        inter=2304,
        e_global=384,
        topk=6,
        e_local=48,
        layout="il",
        act=(1.0, 10.0, 0.0),
        scale_shift=7,
        route_scale=1.5,
        max_tokens=16,
        layer=3,
        x_rms=(1.0, 4.0),
    ),
    # Muse Spark Open EP2 x TP4: 128 of 256 experts x a quarter of the intermediate dim; per-route rows
    "muse": _spec(k=4096, inter=1024, e_global=256, topk=8, e_local=128, w2="route", max_tokens=4, layer=30),
    # Kimi-K2.7 EP8: 48 of 384 experts, K 7168 (longer than the w13 VLM stage)
    "kimi": _spec(k=7168, inter=2048, e_global=384, topk=8, e_local=48, max_tokens=16, layer=3),
}

# Tensor names of checkpoints already in the served layout (the ports' converted checkpoints)
SERVED_PREFIX = {
    "glm": "model.layers.{layer}.mlp.experts.routed_experts.",
    "dsv41": "layers.{layer}.ffn.experts.routed_experts.",
    "kimi": "language_model.model.layers.{layer}.mlp.experts.routed_experts.",
}


def decode_cases(spec):
    """(tokens, routing) cases up to the variant's token cap: "rand" = top-k of all experts per token (the local
    route count is what matters), "hot" = every route on a local expert (worst case)."""
    ts = [t for t in (1, 2, 4, 8, 16, 32) if t <= spec["max_tokens"]]
    out = [(t, "rand") for t in ts]
    out += [(1, "hot"), (min(8, spec["max_tokens"]), "hot")]
    return [(t, m) for t, m in dict.fromkeys(out) if t * spec["topk"] <= 128]


# ---------------------------------------------------------------------------------------------- weights
def snapshot_dir(path):
    """A snapshot directory (with model.safetensors.index.json) from a snapshot or a hub cache model directory."""
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return path
    for s in sorted(glob.glob(os.path.join(path, "snapshots", "*", ""))):
        if os.path.isfile(os.path.join(s, "model.safetensors.index.json")):
            return s
    raise FileNotFoundError(f"no model.safetensors.index.json under {path}")


def _reader(checkpoint):
    from safetensors import safe_open

    ck = snapshot_dir(checkpoint)
    with open(os.path.join(ck, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]

    def get(name, sl=None):
        with safe_open(os.path.join(ck, idx[name]), "pt") as f:
            t = f.get_slice(name)
            return t[sl] if sl is not None else t[:]

    return ck, get


def load_layer(variant, checkpoint, layer, e_local, e_min, tp_rank=0):
    """Experts e_min .. e_min + e_local - 1 of one layer, in the layout vLLM keeps on the device.

    gptoss: the Hugging Face openai/gpt-oss checkpoint. muse: the amd Quark MXFP4 checkpoint as published (per-expert
    gate/up/down tensors, TP slice ``tp_rank`` of 4). glm, dsv41, kimi: a checkpoint in the served layout
    (``SERVED_PREFIX``: fused w13_weight / w13_weight_scale / w2_weight / w2_weight_scale); glm takes TP slice
    ``tp_rank`` of 2 of the intermediate dim.
    """
    import torch

    spec = VARIANTS[variant]
    ck, get = _reader(checkpoint)
    sl = slice(e_min, e_min + e_local)
    d = {}
    if variant == "gptoss":
        with open(os.path.join(ck, "config.json")) as f:
            cfg = json.load(f)
        k, inter = cfg["hidden_size"], cfg["intermediate_size"]
        pre = f"model.layers.{layer}.mlp."
        d = {
            "w13": get(pre + "experts.gate_up_proj_blocks", sl).reshape(e_local, 2 * inter, k // 2).contiguous(),
            "s13": get(pre + "experts.gate_up_proj_scales", sl).contiguous(),
            "b13": get(pre + "experts.gate_up_proj_bias", sl).contiguous(),
            "w2": get(pre + "experts.down_proj_blocks", sl).reshape(e_local, k, inter // 2).contiguous(),
            "s2": get(pre + "experts.down_proj_scales", sl).contiguous(),
            "b2": get(pre + "experts.down_proj_bias", sl).contiguous(),
            "rw": get(pre + "router.weight").float(),
            "rb": get(pre + "router.bias").float(),
        }
    elif variant == "muse":
        inter = spec["inter"]
        lo, hi = tp_rank * inter, (tp_rank + 1) * inter
        pre = f"model.language_model.layers.{layer}.mlp.experts."

        def g(e, w, p, s):
            return get(f"{pre}{e}.{w}.{p}", s)

        rows, cols, scols = (
            (slice(lo, hi),),
            (slice(None), slice(lo // 2, hi // 2)),
            (slice(None), slice(lo // 32, hi // 32)),
        )
        experts = range(sl.start, sl.stop)

        def cat(p):  # [gate rows; up rows] of this TP slice
            return torch.stack([torch.cat([g(e, "gate_proj", p, rows), g(e, "up_proj", p, rows)]) for e in experts])

        d["w13"], d["s13"] = cat("weight").contiguous(), cat("weight_scale").contiguous()
        d["w2"] = torch.stack([g(e, "down_proj", "weight", cols).contiguous() for e in experts]).contiguous()
        d["s2"] = torch.stack([g(e, "down_proj", "weight_scale", scols).contiguous() for e in experts]).contiguous()
    else:
        pre = SERVED_PREFIX[variant].format(layer=layer)
        w13, s13 = get(pre + "w13_weight", sl), get(pre + "w13_weight_scale", sl)
        w2, s2 = get(pre + "w2_weight", sl), get(pre + "w2_weight_scale", sl)
        if variant == "glm" and w13.shape[1] == 4 * spec["inter"]:  # full checkpoint: take this rank's TP slice
            i_full = w13.shape[1] // 2
            lo, hi = tp_rank * i_full // 2, (tp_rank + 1) * i_full // 2
            w13 = torch.cat([w13[:, lo:hi], w13[:, i_full + lo : i_full + hi]], 1)
            s13 = torch.cat([s13[:, lo:hi], s13[:, i_full + lo : i_full + hi]], 1)
            w2, s2 = w2[..., lo // 2 : hi // 2], s2[..., lo // 32 : hi // 32]
        d = {n: t.contiguous() for n, t in (("w13", w13), ("s13", s13), ("w2", w2), ("s2", s2))}
    k, inter = d["w13"].shape[2] * 2, d["w13"].shape[1] // 2
    d.update(k=k, inter=inter, e_global=spec["e_global"], topk=spec["topk"])
    check_layer(d, e_local)
    return d


def random_layer(variant, e_local=None, seed=0, **shape):
    """Random MXFP4 experts with the variant's shapes (overridable). Scales of 2^-9 .. 2^-5 keep the activations in
    the range the models see; x RMS 4 drives the clamps of the clamped activations."""
    import torch

    spec = dict(VARIANTS[variant])
    spec.update(shape)
    e_local = e_local or spec["e_local"]
    k, inter = spec["k"], spec["inter"]
    g = torch.Generator().manual_seed(seed)

    def u8(*s, lo=0, hi=256):
        return torch.randint(lo, hi, s, generator=g, dtype=torch.int32).to(torch.uint8)

    d = {
        "w13": u8(e_local, 2 * inter, k // 2),
        "s13": u8(e_local, 2 * inter, k // 32, lo=118, hi=123),
        "w2": u8(e_local, k, inter // 2),
        "s2": u8(e_local, k, inter // 32, lo=118, hi=123),
    }
    if spec["bias"]:
        d["b13"] = (torch.randn(e_local, 2 * inter, generator=g) * 0.5).to(torch.bfloat16)
        d["b2"] = (torch.randn(e_local, k, generator=g) * 0.5).to(torch.bfloat16)
        d["rw"] = torch.randn(spec["e_global"], k, generator=g) * 0.02
        d["rb"] = torch.zeros(spec["e_global"])
    d.update(k=k, inter=inter, e_global=spec["e_global"], topk=spec["topk"])
    check_layer(d, e_local)
    return d


def check_layer(d, e_local):
    k, inter = d["k"], d["inter"]
    assert d["w13"].shape == (e_local, 2 * inter, k // 2), d["w13"].shape
    assert d["s13"].shape == (e_local, 2 * inter, k // 32), d["s13"].shape
    assert d["w2"].shape == (e_local, k, inter // 2), d["w2"].shape
    assert d["s2"].shape == (e_local, k, inter // 32), d["s2"].shape
    if "b13" in d:
        assert d["b13"].shape == (e_local, 2 * inter), d["b13"].shape
        assert d["b2"].shape == (e_local, k), d["b2"].shape


# ---------------------------------------------------------------------------------------------- cases
def make_case(spec, d, tokens, mode, x_rms, e_min, e_local):
    """(x bf16 [T, K], ids i32 [T, topk], tw f32 [T, topk]) for one case, deterministic per (T, mode, x_rms).
    mode "real" (gpt-oss with a checkpoint): the layer's own router."""
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
    return x, ids, s / s.sum(-1, keepdim=True) * spec["route_scale"]


# ---------------------------------------------------------------------------------------------- reference
def dequant(w, s):
    """Packed E2M1 u8 [N, K/2] + E8M0 u8 [N, K/32] -> fp32 [N, K] (low nibble = even k)."""
    import torch

    lut = torch.tensor(E2M1)
    n, kh = w.shape
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1).reshape(n, kh * 2)
    return v * torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, 1)


def activation(spec, h, inter):
    """Gated activation of w13 rows h [.., 2I] in the variant's row layout, fp32."""
    import torch

    alpha, limit, beta = spec["act"]
    g, u = (h[..., 0::2], h[..., 1::2]) if spec["layout"] == "il" else (h[..., :inter], h[..., inter:])
    g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    return g * torch.sigmoid(alpha * g) * (u + beta)


class Reference:
    """CPU fp32 reference: dequantised weights, fp32 x . W (+ bias), fp32 activation, bf16 intermediate (what both
    device paths store), fp32 w2 (+ bias), bf16 routing weights. Works per expert, so prefill sizes stay cheap."""

    def __init__(self, spec, d, e_min, e_local):
        self.spec, self.d, self.e_min, self.e_local = spec, d, e_min, e_local
        self._dq = {}

    def expert(self, e):
        if e not in self._dq:
            d = self.d
            self._dq[e] = (dequant(d["w13"][e], d["s13"][e]), dequant(d["w2"][e], d["s2"][e]))
        return self._dq[e]

    def __call__(self, x, ids, tw):
        """Returns (y fp32, local route count): y [T, K] for "sum" variants, [T * topk, K] per route for "route"."""
        import torch

        d, spec = self.d, self.spec
        tokens, k = x.shape
        topk = ids.shape[1]
        rows = torch.zeros(tokens, topk, k)
        loc = ids - self.e_min
        local = 0
        for e in sorted(set(int(v) for v in loc.flatten().tolist() if 0 <= v < self.e_local)):
            t, j = torch.nonzero(loc == e, as_tuple=True)
            local += t.numel()
            a13, a2 = self.expert(e)
            h = x[t].float() @ a13.t()
            if "b13" in d:
                h = h + d["b13"][e].float()
            a = activation(spec, h, d["inter"]).to(torch.bfloat16).float()
            o = a @ a2.t()
            if "b2" in d:
                o = o + d["b2"][e].float()
            rows[t, j] = o
        if spec["w2"] == "route":
            return rows.view(tokens * topk, k), local
        twb = tw.to(torch.bfloat16).float()
        return (rows * twb.unsqueeze(-1)).sum(1), local


def rel_l2(a, b):
    """||a - b|| / ||b||"""
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------------------------- device
def device_layer(spec, d, impl, dev="hpu"):
    """Device tensors for one implementation: the served tensors (w13 scales shifted as the model serves them) plus,
    for the gpt-oss kernels, the re-laid biases."""
    t = {n: d[n].to(dev) for n in ("w13", "s13", "w2", "s2", "b13", "b2") if n in d}
    if spec["scale_shift"]:
        t["s13"] = t["s13"] + spec["scale_shift"]
    if spec["bias"] and impl != "stock":
        import mxfp4_tpc_moe_rt as rt

        t["b13k"] = rt.bias13(d["b13"], d["inter"]).to(dev)
        t["b2k"] = rt.bias2(d["b2"]).to(dev)
    return t


def _lists(t, e_local):
    return ([t["w13"][e] for e in range(e_local)], [t["w2"][e] for e in range(e_local)],
            [t["s13"][e] for e in range(e_local)], [t["s2"][e] for e in range(e_local)])  # fmt: skip


def stock_runner(spec, t, d, e_local, e_min):
    """The stock op as the model serves it. run(x bf16 [T, K], ids i32 [T, topk], tw bf16 [T, topk])."""
    import torch

    e_max = e_min + e_local - 1
    if spec["bias"]:  # gpt-oss: VllmMixtureOfExpertsOpMXFP4 (bias_mxfp4_fused_weights), as vllm-gaudi serves it
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4

        op = VllmMixtureOfExpertsOpMXFP4(d["e_global"], e_local, e_min, e_max, block_size=32, has_bias=True)
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

    l13, l2, l13s, l2s = _lists(t, e_local)
    ch = dict(chunk_size=spec["chunk"], total_experts=d["e_global"]) if spec["chunk"] else {}
    moe = torch.ops.hpu.mixture_of_experts
    if spec["scale_shift"]:
        # DeepSeek: bias_mxfp4_fused_weights on the +k scales: SwiGLU-OAI with alpha 2^-k, limit L 2^k, an up bias
        # of -1 to cancel its +1, and the routing weights times 2^-2k
        sh = spec["scale_shift"]
        b13 = torch.zeros(2 * d["inter"], dtype=torch.bfloat16)
        b13[1::2] = -1.0
        b13l = [b13.to(t["w13"].device) for _ in range(e_local)]
        b2l = [torch.zeros(d["k"], dtype=torch.bfloat16, device=t["w13"].device) for _ in range(e_local)]

        def run(x, ids, tw):
            return moe.bias_mxfp4_fused_weights(
                x, ids, (tw.float() * 2.0 ** (-2 * sh)).to(torch.bfloat16), l13, l2, b13l, b2l, l13s, l2s,
                block_size=32, permuted_weights=True, experts_min=e_min, experts_max=e_max, is_fp4=True,
                alpha=2.0**-sh, limit=spec["act"][1] * 2.0**sh, **ch,
            )  # fmt: skip

        return run

    if spec["w2"] == "route":
        # Muse: one single-route row per (token, route), weight 1; rows of non-local routes are masked to zero
        def run(x, ids, tw):
            tokens, k = x.shape
            topk = ids.shape[1]
            xr = x.unsqueeze(1).expand(tokens, topk, k).reshape(tokens * topk, k)
            ir = ids.reshape(tokens * topk, 1)
            ones = torch.ones(tokens * topk, 1, dtype=x.dtype, device=x.device)
            y = moe.mxfp4_fused_weights(
                xr, ir, ones, l13, l2, l13s, l2s, block_size=32, permuted_weights=True, activation="silu",
                experts_min=e_min, experts_max=e_max, is_fp4=True, **ch,
            )  # fmt: skip
            loc = (ir >= e_min) & (ir <= e_max)
            return torch.where(loc, y, torch.zeros_like(y))

        return run

    def run(x, ids, tw):
        return moe.mxfp4_fused_weights(
            x, ids, tw, l13, l2, l13s, l2s, block_size=32, permuted_weights=True, activation="silu",
            experts_min=e_min, experts_max=e_max, is_fp4=True, **ch,
        )  # fmt: skip

    return run


def permute_x(x):
    """bf16 [T, K] (K a multiple of 512) -> the per-512 chunk permutation the w13 kernels read."""
    tokens, k = x.shape
    return x.reshape(tokens, k // 512, 128, 4).transpose(-1, -2).reshape(tokens, k)


def kernel_act_params(spec):
    """(alpha, limit, beta, ascale) of mxk_w13x for the variant. With w13 scales served shifted by +k (DeepSeek), the
    sums come out 2^k too large: alpha 2^-k, limit L 2^k and ascale 2^-2k give exactly the unshifted result."""
    alpha, limit, beta = spec["act"]
    sh = spec["scale_shift"]
    if not sh:
        return alpha, limit, beta, 1.0
    assert beta == 0.0, "an additive beta does not scale with the shifted sums"
    return alpha * 2.0**-sh, limit * 2.0**sh, beta, 2.0 ** (-2 * sh)


def kernel_runner(spec, t, d, e_min):
    """The kernels, called as the model's integration calls them. Same run() signature as stock_runner."""
    import torch

    co = torch.ops.custom_op
    topk = d["topk"]
    if spec["bias"]:
        import mxfp4_tpc_moe_rt as rt

        def run(x, ids, tw):
            return rt.moe_decode(
                x, ids, tw, t["w13"], t["s13"], t["b13k"], t["w2"], t["s2"], t["b2k"], topk, e_min, ALPHA, LIMIT
            )

        return run
    alpha, limit, beta, ascale = kernel_act_params(spec)
    il = spec["layout"] == "il"

    def run(x, ids, tw):
        idf = ids.reshape(-1)
        ap = co.mxk_w13x(permute_x(x), idf, t["w13"], t["s13"], topk, e_min, alpha, limit, beta, ascale, il)
        if spec["w2"] == "route":
            return co.mxk_w2xr(ap, idf, t["w2"], t["s2"], 1, e_min)
        return co.mxk_w2x(ap, idf, tw.reshape(-1), t["w2"], t["s2"], topk, e_min)

    return run


def prefill_runner(spec, t, d, e_local, e_min, impl):
    """GLM prefill: impl "stock" = mxfp4_fused_weights; "ours" = mxk_deq per expert, a graph break, then the stock
    bf16 MoE op with the same chunk arguments."""
    import habana_frameworks.torch as ht
    import torch

    stock = stock_runner(spec, t, d, e_local, e_min)
    if impl == "stock":
        return stock
    co = torch.ops.custom_op
    ch = dict(chunk_size=spec["chunk"], total_experts=d["e_global"]) if spec["chunk"] else {}

    def run(x, ids, tw):
        l13 = [co.mxk_deq(t["w13"][e], t["s13"][e], 0) for e in range(e_local)]
        l2 = [co.mxk_deq(t["w2"][e], t["s2"][e], 0) for e in range(e_local)]
        ht.core.mark_step()
        return torch.ops.hpu.mixture_of_experts(
            hidden_states=x, expert_routing_table=ids, router_weights=tw, w12=l13, w3=l2, permuted_weights=True,
            activation="silu", experts_min=e_min, experts_max=e_min + e_local - 1, **ch,
        )  # fmt: skip

    return run


def hpu_env_for_kernels(lib_dir):
    """Process env for the kernels; call before torch / habana_frameworks are imported."""
    os.environ["GC_KERNEL_PATH"] = os.path.join(lib_dir, "libmxk_tpc.so")
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)


def load_ops(lib_dir, variant):
    """Load the op library and check it is the given variant's (its MANIFEST names the variant)."""
    import torch

    with open(os.path.join(lib_dir, "MANIFEST")) as f:
        man = f.read()
    if f"variant: {variant}\n" not in man:
        raise SystemExit(f"{lib_dir} is not a '{variant}' build (MANIFEST: {man.splitlines()[:2]})")
    torch.ops.load_library(os.path.join(lib_dir, "libmxk_op_lazy.so"))


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


def time_lazy(fn, warm=2, reps=10):
    """Median wall time in us of fn() run as ordinary lazy graphs (for paths with a graph break, which an HPU graph
    cannot capture), including the host launch."""
    import statistics
    import time

    import habana_frameworks.torch as ht

    for _ in range(warm):
        fn()
        ht.core.mark_step()
        ht.hpu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ht.core.mark_step()
        ht.hpu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6
