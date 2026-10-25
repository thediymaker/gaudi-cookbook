#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card check of the MXFP4 MoE kernels exactly as MiMo-V2.6-Pro serves them: the glm build with its
torch.compile op library, called through mimo_v2_hpu.mxk in compile mode, on real checkpoint layers.

    python3 mxk_ops.py CHECKPOINT_DIR [--lib /opt/gaudi-cookbook-mimo/mxk] [--layers 1,69] [--ranks 0,7]

CHECKPOINT_DIR is the public checkpoint or the staged copy. For each (layer, TP8 rank): that rank's 48 experts of
the layer (K 6144, I 2048, top-8 of 384) and the decode cases of kernels/mxfp4-moe (1 to 16 tokens with random
routing, 1 and 8 tokens with every route local; mxk_harness.decode_cases). Each case runs

  ours     mxk.moe, as MiMoV2HpuMoE.forward calls it, compiled with torch.compile after the one eager call per
           process that the PT2 ops need (mxk.register_eager);
  eager    the same call without torch.compile;
  stock    VllmMixtureOfExpertsOpMXFP4, as the model calls it with MIMO_MXK=0;

and compares them with the CPU fp32 reference of kernels/mxfp4-moe (mxk_harness.Reference).

PASS mxk-pt2-parity: for every case the kernels' relative L2 error against the reference is at most 3e-3 and not
above the stock op's (the parity criterion of kernels/mxfp4-moe/tests/parity_hpu.py), and the compiled result equals
the eager one bit for bit. Also prints per-call times (INFO) of the compiled kernels and the stock op.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(PORT, "..", ".."))
sys.path.insert(0, os.path.join(PORT, "mimo_v2_hpu"))
sys.path.insert(0, os.path.join(REPO, "kernels", "mxfp4-moe", "python"))

E_GLOBAL, TOPK, K, INTER, TP = 384, 8, 6144, 2048, 8
E_LOCAL = E_GLOBAL // TP
MAX_REL = 3e-3


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--lib", default="/opt/gaudi-cookbook-mimo/mxk")
    ap.add_argument("--layers", default="1,69", help="MoE layers (1..69; layer 0 is dense)")
    ap.add_argument("--ranks", default="0,7", help="TP8 ranks whose experts to load")
    return ap.parse_args()


def load_layer(ckpt: str, layer: int, e_min: int) -> dict:
    """The rank's experts of one layer in the served layout: w13 [E, 2I, K/2] ([gate; up] rows), s13, w2, s2."""
    import torch
    from safetensors import safe_open

    with open(os.path.join(ckpt, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    want: dict[str, list[str]] = {}
    for e in range(e_min, e_min + E_LOCAL):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for kind in ("weight", "weight_scale"):
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.{kind}"
                want.setdefault(wm[name], []).append(name)
    got = {}
    for f, names in want.items():
        with safe_open(os.path.join(ckpt, f), framework="pt") as fh:
            for n in names:
                got[n] = fh.get_tensor(n)

    def stack(proj, kind):
        return [got[f"model.layers.{layer}.mlp.experts.{e}.{proj}.{kind}"] for e in range(e_min, e_min + E_LOCAL)]

    d = {"k": K, "inter": INTER, "topk": TOPK, "e_global": E_GLOBAL}
    for kind, key in (("weight", "w"), ("weight_scale", "s")):
        g, u, dn = stack("gate_proj", kind), stack("up_proj", kind), stack("down_proj", kind)
        d[key + "13"] = torch.stack([torch.cat((a, b)) for a, b in zip(g, u, strict=True)])
        d[key + "2"] = torch.stack(dn)
    assert d["w13"].shape == (E_LOCAL, 2 * INTER, K // 2) and d["w13"].dtype == torch.uint8, d["w13"].shape
    assert d["s13"].shape == (E_LOCAL, 2 * INTER, K // 32), d["s13"].shape
    assert d["w2"].shape == (E_LOCAL, K, INTER // 2) and d["s2"].shape == (E_LOCAL, K, INTER // 32), d["w2"].shape
    return d


def per_call_us(fn, args, reps=50):
    import torch

    for _ in range(3):
        fn(*args)
    torch.hpu.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    torch.hpu.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


def main() -> int:
    a = parse()
    # Before torch and habana_frameworks: compile mode, the kernel switch, and GC_KERNEL_PATH (mxk.setup_env).
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["MIMO_MXK"] = "1"
    os.environ["MIMO_MXK_LIB"] = os.path.abspath(a.lib)
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)
    os.environ.setdefault("VLLM_MOE_CHUNK", "256")  # as the recipe serves the stock op
    os.environ.setdefault("VLLM_MOE_TOKEN_BOUNDARY", "2048")
    from mimo_v2_hpu import mxk

    mxk.setup_env()
    if not mxk.ENABLED:
        print(f"FAIL mxk-pt2-parity: no usable glm build with libmxk_op_pt2.so in {a.lib}")
        return 1
    import habana_frameworks.torch  # noqa: F401  (registers the hpu device)
    import mxk_harness as H
    import torch
    from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4

    if not mxk.load():
        print("FAIL mxk-pt2-parity: the op library did not load (see the warning above)")
        return 1
    print(f"INFO {mxk.lib_version()}, GC_KERNEL_PATH={os.environ['GC_KERNEL_PATH']}", flush=True)
    # One graph per (layer, rank, token count): keep dynamo from falling back to eager after its default 8.
    for knob in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, knob):
            setattr(torch._dynamo.config, knob, 256)
    torch._dynamo.config.error_on_recompile = False
    spec = H._spec(k=K, inter=INTER, e_global=E_GLOBAL, topk=TOPK, e_local=E_LOCAL, max_tokens=16)
    bad, n, worst = 0, 0, 0.0
    for layer in (int(x) for x in a.layers.split(",")):
        for rank in (int(x) for x in a.ranks.split(",")):
            e_min = rank * E_LOCAL
            d = load_layer(a.checkpoint, layer, e_min)
            ref = H.Reference(spec, d, e_min, E_LOCAL)
            t = {k: d[k].to("hpu") for k in ("w13", "s13", "w2", "s2")}
            w13, s13, w2, s2 = t["w13"], t["s13"], t["w2"], t["s2"]
            mxk.register_eager(w13, s13, w2, s2, TOPK, e_min, K, E_LOCAL, INTER)  # no-op after the first layer

            def ours(x, ids, tw, w13=w13, s13=s13, w2=w2, s2=s2, e_min=e_min):
                return mxk.moe(x, ids, tw, w13, s13, w2, s2, TOPK, e_min)

            compiled = torch.compile(ours, backend="hpu_backend", dynamic=False)
            op = VllmMixtureOfExpertsOpMXFP4(E_GLOBAL, E_LOCAL, e_min, e_min + E_LOCAL - 1, block_size=32)
            for e in range(E_LOCAL):
                op.w13_list[e].set_weight(w13[e])
                op.w13_list[e].set_scale(s13[e])
                op.w2_list[e].set_weight(w2[e])
                op.w2_list[e].set_scale(s2[e])
            op._cache_weight_lists()

            def stock(x, ids, tw, op=op):
                return op(x, ids, tw, permuted_weights=True, activation="silu")

            print(f"layer {layer} rank {rank} (experts {e_min}..{e_min + E_LOCAL - 1})", flush=True)
            print(f"  {'case':10s}{'local':>6s}{'ours rel':>11s}{'stock rel':>11s}{'compiled=eager':>16s}  result")
            for tokens, mode in H.decode_cases(spec):
                x, ids, tw = H.make_case(spec, d, tokens, mode, 1.0, e_min, E_LOCAL)
                y_ref, local = ref(x, ids, tw)
                args = (x.to("hpu"), ids.to("hpu"), tw.to(torch.bfloat16).to("hpu"))
                y_c = compiled(*args).cpu()
                y_e = ours(*args).cpu()
                y_s = stock(*args).cpu()
                ro, rs = H.rel_l2(y_c, y_ref), H.rel_l2(y_s, y_ref)
                same = torch.equal(y_c, y_e)
                ok = ro <= MAX_REL and ro <= rs and same
                bad += not ok
                n += 1
                worst = max(worst, ro)
                case = f"T{tokens}/{mode}"
                res = "ok" if ok else "FAIL"
                print(f"  {case:10s}{local:6d}{ro:11.2e}{rs:11.2e}{str(same):>16s}  {res}", flush=True)
                if mode == "rand" and tokens in (1, 8) and rank == int(a.ranks.split(",")[0]):
                    tc, ts = per_call_us(compiled, args), per_call_us(stock, args)
                    print(f"INFO layer {layer} T={tokens}: kernels {tc:.0f} us, stock op {ts:.0f} us per call")
            del t, w13, s13, w2, s2, op, compiled, ref
    ok = bad == 0
    print(
        f"{'PASS' if ok else 'FAIL'} mxk-pt2-parity: {n - bad}/{n} cases, worst kernel rel_l2 {worst:.2e} "
        f"(bound {MAX_REL:g} and not above the stock op), compiled = eager",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
