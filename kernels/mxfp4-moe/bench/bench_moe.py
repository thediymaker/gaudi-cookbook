#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark: microseconds per MoE call on one Gaudi 2 card, the stock op vs one MXFP4 kernel variant.

One implementation per process (lazy-mode legacy custom ops cache recipes by input signature):

    bench_moe.py --variant glm --impl stock --json stock.json [--checkpoint <dir>] [--e-local N]
    bench_moe.py --variant glm --impl ours  --json ours.json  [--lib <build of the variant>]
    bench_moe.py --compare stock.json ours.json

Each call is timed inside an HPU graph that chains --chain calls (default 8; each call's output feeds the next
call's input, so identical calls are not merged), median of --reps replays, divided by the chain length. Shapes are
the variant's per-rank serving shapes (mxk_harness.VARIANTS). Cases: x RMS 1, T = 1 .. the variant's token cap with
random top-k routing over all experts ("rand": the local route count is what matters) and every route local ("hot",
worst case); for glm also the prefill unpack path at 256 .. 4096 tokens, timed as ordinary lazy graphs (one call
per synchronize, host launch included) because its graph break cannot be captured. Weights: one layer of a local
checkpoint, or random MXFP4 weights of the same shapes (timing does not depend on the values).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import mxk_harness as H  # noqa: E402


def compare(pa, pb):
    with open(pa) as f:
        a = json.load(f)
    with open(pb) as f:
        b = json.load(f)
    print(
        f"[{a['variant']}] {a['impl']} vs {b['impl']}: {a['e_local']} local experts, K {a['k']} I {a['inter']}, "
        f"{a['weights']} weights"
    )
    print(f"{'case':16s}{'local':>6s}{a['impl'] + ' us':>12s}{b['impl'] + ' us':>12s}{'ratio':>8s}")
    for key, ra in a["cases"].items():
        rb = b["cases"].get(key)
        if rb:
            print(f"{key:16s}{ra['local']:6d}{ra['us']:12.1f}{rb['us']:12.1f}{ra['us'] / rb['us']:7.2f}x")


def main():
    ap = argparse.ArgumentParser(description="MXFP4 MoE kernel micro benchmark (one Gaudi 2 card)")
    ap.add_argument("--variant", default="gptoss", choices=sorted(H.VARIANTS))
    ap.add_argument("--impl", choices=["stock", "ours"])
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--e-local", type=int, default=None)
    ap.add_argument("--e-min", type=int, default=0)
    ap.add_argument("--inter", type=int, default=None, help="intermediate size per rank (random weights only)")
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--lib", default=os.environ.get("MXFP4_TPC_MOE_LIB", os.path.join(HERE, "..", "dist")))
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if not a.impl:
        ap.error("--impl or --compare is required")
    spec = dict(H.VARIANTS[a.variant])
    if a.inter:
        spec["inter"] = a.inter
    e_local = a.e_local or spec["e_local"]
    layer = spec["layer"] if a.layer is None else a.layer
    if a.impl == "ours":
        H.hpu_env_for_kernels(os.path.abspath(a.lib))  # before torch / habana_frameworks are imported
    import torch

    import habana_frameworks.torch  # noqa: F401  registers the hpu device

    if a.checkpoint:
        d = H.load_layer(a.variant, a.checkpoint, layer, e_local, a.e_min, a.tp_rank)
    else:
        d = H.random_layer(a.variant, e_local, inter=spec["inter"])
    t_dev = H.device_layer(spec, d, a.impl)
    if a.impl == "stock":
        run = H.stock_runner(spec, t_dev, d, e_local, a.e_min)
    else:
        H.load_ops(os.path.abspath(a.lib), a.variant)
        run = H.kernel_runner(spec, t_dev, d, a.e_min)

    out = {
        "variant": a.variant,
        "impl": a.impl,
        "layer": layer,
        "e_local": e_local,
        "e_min": a.e_min,
        "k": d["k"],
        "inter": d["inter"],
        "weights": "checkpoint" if a.checkpoint else "random",
        "chain": a.chain,
        "cases": {},
    }
    lo, hi = a.e_min, a.e_min + e_local
    route = spec["w2"] == "route"

    def bench(key, fn, tokens, mode, chain, reps, graph=True):
        x, ids, tw = H.make_case(spec, d, tokens, mode, 1.0, a.e_min, e_local)
        xh, idh, twh = x.to("hpu"), ids.to("hpu"), tw.to(torch.bfloat16).to("hpu")

        def chained(xh=xh):
            v = xh
            for _ in range(chain):
                o = fn(v, idh, twh)
                v = o.view(tokens, d["topk"], d["k"]).sum(1) if route else o  # per-route rows: feed the token sum
            return v

        us = (H.time_hpu_graph(chained, reps=reps) if graph else H.time_lazy(chained, reps=reps)) / chain
        local = int(((ids >= lo) & (ids < hi)).sum())
        out["cases"][key] = {"us": us, "local": local}
        print(f"[{a.variant}] {a.impl} {key}: {local} local routes, {us:.1f} us per call", flush=True)

    for t, mode in H.decode_cases(spec):
        bench(f"T{t}/{mode}", run, t, mode, a.chain, a.reps)
    if spec["prefill"]:
        prun = H.prefill_runner(spec, t_dev, d, e_local, a.e_min, a.impl)
        for t in spec["prefill"]:  # the unpack path has a graph break: timed as lazy graphs, both implementations
            bench(f"prefill T{t}", prun, t, "rand", 1, max(5, a.reps // 3), graph=False)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
