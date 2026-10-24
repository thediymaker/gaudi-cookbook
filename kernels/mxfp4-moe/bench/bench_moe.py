#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark: microseconds per gpt-oss MoE call on one Gaudi 2 card, stock op vs the MXFP4 TPC kernels.

One implementation per process (lazy-mode legacy custom ops cache recipes by input signature):

    bench_moe.py --impl stock --json stock.json [--checkpoint <snapshot>] [--e-local 32]
    bench_moe.py --impl ours  --json ours.json  [--lib <dir>]
    bench_moe.py --compare stock.json ours.json

Each call is timed inside an HPU graph that chains --chain calls (default 8; each call's output is the next call's
input, so identical calls are not merged), median of --reps replays, divided by the chain length. Cases: x RMS 1,
T = 1..32 tokens, random top-4 routing over all experts ("rand", the local route count is what matters) and every
route local ("hot", worst case). Weights: one layer of a local checkpoint, or random MXFP4 weights of the same shapes
(timing does not depend on the values).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import mxk_harness as H  # noqa: E402

BENCH_CASES = [(1, "rand"), (2, "rand"), (4, "rand"), (8, "rand"), (16, "rand"), (32, "rand"), (1, "hot"), (8, "hot")]


def compare(pa, pb):
    with open(pa) as f:
        a = json.load(f)
    with open(pb) as f:
        b = json.load(f)
    print(f"{a['impl']} vs {b['impl']}: layer {a['layer']}, {a['e_local']} local experts, {a['weights']} weights")
    print(f"{'case':12s}{'local':>6s}{a['impl'] + ' us':>12s}{b['impl'] + ' us':>12s}{'ratio':>8s}")
    for key, ra in a["cases"].items():
        rb = b["cases"].get(key)
        if rb:
            print(f"{key:12s}{ra['local']:6d}{ra['us']:12.1f}{rb['us']:12.1f}{ra['us'] / rb['us']:7.2f}x")


def main():
    ap = argparse.ArgumentParser(description="MXFP4 MoE kernel micro benchmark (one Gaudi 2 card)")
    ap.add_argument("--impl", choices=["stock", "ours"])
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--e-local", type=int, default=32)
    ap.add_argument("--e-min", type=int, default=0)
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
    if a.impl == "ours":
        H.hpu_env_for_kernels(os.path.abspath(a.lib))  # before torch / habana_frameworks are imported
    import torch

    import habana_frameworks.torch  # noqa: F401  registers the hpu device

    d = H.load_layer(a.checkpoint, a.layer, a.e_local, a.e_min) if a.checkpoint else H.random_layer(a.e_local)
    t_dev = H.device_layer(d, a.impl)
    if a.impl == "stock":
        run = H.stock_runner(t_dev, d, a.e_local, a.e_min)
    else:
        import mxfp4_tpc_moe_rt as rt

        if not rt.load_ops(os.path.abspath(a.lib)):
            sys.exit(f"no mxk_w13xib / mxk_w2xb in {a.lib}/libmxk_op_lazy.so")
        run = H.kernel_runner(t_dev, d, a.e_min)

    out = {
        "impl": a.impl,
        "layer": a.layer,
        "e_local": a.e_local,
        "e_min": a.e_min,
        "weights": "checkpoint" if a.checkpoint else "random",
        "chain": a.chain,
        "cases": {},
    }
    lo, hi = a.e_min, a.e_min + a.e_local
    for t, mode in BENCH_CASES:
        if t * d["topk"] > 128:
            continue
        x, ids, tw = H.make_case(d, t, mode, 1.0, a.e_min, a.e_local)
        xh, idh, twh = x.to("hpu"), ids.to("hpu"), tw.to(torch.bfloat16).to("hpu")

        def chain(xh=xh, idh=idh, twh=twh):
            v = xh
            for _ in range(a.chain):
                v = run(v, idh, twh)
            return v

        us = H.time_hpu_graph(chain, reps=a.reps) / a.chain
        local = int(((ids >= lo) & (ids < hi)).sum())
        out["cases"][f"T{t}/{mode}"] = {"us": us, "local": local}
        print(f"{a.impl} T{t}/{mode}: {local} local routes, {us:.1f} us per call", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
