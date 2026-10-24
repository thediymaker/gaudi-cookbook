#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card parity test of the MXFP4 MoE decode kernels against the stock vllm-gaudi op and a CPU fp32 reference.

Run one implementation per process (lazy-mode legacy custom ops cache recipes by input signature, so two
implementations with identical inputs must not share a process). ``run_parity.sh`` runs the whole sequence:

    parity_hpu.py --impl ref    CPU fp32 reference (no device needed)
    parity_hpu.py --impl stock  VllmMixtureOfExpertsOpMXFP4(has_bias=True), as vllm-gaudi serves gpt-oss
    parity_hpu.py --impl ours   the kernels (--lib DIR with libmxk_tpc.so and libmxk_op_lazy.so)
    parity_hpu.py --impl sig    two kernel calls with identical input shapes in ONE lazy graph, per T = 1..32
    parity_hpu.py --impl cmp    compare the saved outputs; exit 1 on a failed check

Weights: one layer of a local openai/gpt-oss-120b or openai/gpt-oss-20b snapshot (--checkpoint), or random MXFP4
weights of the same shapes (default). --e-local 32 matches one rank of 4-way expert parallelism on gpt-oss-120b;
--e-local 128 is one card holding all experts.

Pass criteria (cmp): for every case the kernel's rel_l2 against the fp32 reference is below --max-rel (default 3e-3)
and not above the stock op's; sig: bit-identical results inside one graph and alone.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import mxk_harness as H  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description="MXFP4 MoE kernel parity (one Gaudi 2 card)")
    ap.add_argument("--impl", required=True, choices=["ref", "stock", "ours", "sig", "cmp"])
    ap.add_argument("--checkpoint", default="", help="HF snapshot of openai/gpt-oss-120b or -20b (default: random)")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--e-local", type=int, default=32, help="local experts on this card")
    ap.add_argument("--e-min", type=int, default=0, help="first local expert")
    ap.add_argument("--lib", default=os.environ.get("MXFP4_TPC_MOE_LIB", os.path.join(HERE, "..", "dist")))
    ap.add_argument("--out", default="parity_out")
    ap.add_argument("--max-rel", type=float, default=3e-3)
    return ap.parse_args()


def cases(d):
    out = list(H.CASES) + list(H.REAL_CASES)
    return [(xs, t, m) for xs in H.X_RMS for (t, m) in out if t * d["topk"] <= 128]


def tag(a):
    return f"L{a.layer}_E{a.e_local}_m{a.e_min}" + ("_ck" if a.checkpoint else "_rand")


def compare(a):
    import torch

    files = {os.path.basename(p).split("_", 1)[0]: p for p in glob.glob(os.path.join(a.out, f"*_{tag(a)}.pt"))}
    if "ref" not in files or "ours" not in files:
        sys.exit(f"need ref and ours results in {a.out} (have {sorted(files)})")
    ref = torch.load(files["ref"], weights_only=False)["cases"]
    ours = torch.load(files["ours"], weights_only=False)["cases"]
    stock = torch.load(files["stock"], weights_only=False)["cases"] if "stock" in files else {}
    print(f"{'case':18s}{'local':>6s}{'ours rel':>11s}{'stock rel':>11s}{'ours vs stock':>15s}  result")
    bad = 0
    for key, r in ref.items():
        if key not in ours:
            continue
        ro = H.rel_l2(ours[key]["y"], r["y"])
        rs = H.rel_l2(stock[key]["y"], r["y"]) if key in stock else float("nan")
        os_ = H.rel_l2(ours[key]["y"], stock[key]["y"]) if key in stock else float("nan")
        ok = ro <= a.max_rel and (key not in stock or ro <= rs)
        bad += 0 if ok else 1
        print(f"{key:18s}{r['local']:6d}{ro:11.2e}{rs:11.2e}{os_:15.2e}  {'ok' if ok else 'FAIL'}")
    print("PARITY", "PASS" if bad == 0 else f"FAIL ({bad} cases)")
    sys.exit(0 if bad == 0 else 1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    if a.impl == "cmp":
        compare(a)
    if a.impl in ("ours", "sig"):
        H.hpu_env_for_kernels(os.path.abspath(a.lib))  # before torch / habana_frameworks are imported
    import torch

    d = H.load_layer(a.checkpoint, a.layer, a.e_local, a.e_min) if a.checkpoint else H.random_layer(a.e_local)
    res = {"impl": a.impl, "cases": {}}

    if a.impl == "ref":
        for xs, t, m in cases(d):
            x, ids, tw = H.make_case(d, t, m, xs, a.e_min, a.e_local)
            y, local = H.reference(d, x, ids, tw, a.e_min, a.e_local)
            res["cases"][f"x{xs}/T{t}/{m}"] = {"y": y, "local": local}
            print(f"ref x{xs}/T{t}/{m}: {local} local routes", flush=True)
        torch.save(res, os.path.join(a.out, f"ref_{tag(a)}.pt"))
        return

    import habana_frameworks.torch as ht

    t_dev = H.device_layer(d, a.impl)
    if a.impl == "stock":
        run = H.stock_runner(t_dev, d, a.e_local, a.e_min)
    else:
        import mxfp4_tpc_moe_rt as rt

        if not rt.load_ops(os.path.abspath(a.lib)):
            sys.exit(f"no mxk_w13xib / mxk_w2xb in {a.lib}/libmxk_op_lazy.so")
        run = H.kernel_runner(t_dev, d, a.e_min)

    if a.impl == "sig":
        bad = 0
        lo, hi = a.e_min, a.e_min + a.e_local
        for t in range(1, 33):
            if t * d["topk"] > 128:
                break
            xa, ida, twa = H.make_case(d, t, "rand", 1.0, a.e_min, a.e_local)
            xb, idb, twb = H.make_case(d, t, "hot", 4.0, a.e_min, a.e_local)
            A = [v.to("hpu") for v in (xa, ida, twa.to(torch.bfloat16))]
            B = [v.to("hpu") for v in (xb, idb, twb.to(torch.bfloat16))]
            ya = run(*A)
            ht.core.mark_step()
            ya = ya.cpu()
            yb = run(*B)
            ht.core.mark_step()
            yb = yb.cpu()
            r1, r2 = run(*A), run(*B)  # same lazy graph, identical input shapes
            ht.core.mark_step()
            r1, r2 = r1.cpu(), r2.cpu()
            la = int(((ida >= lo) & (ida < hi)).sum())
            ok = torch.equal(ya, r1) and torch.equal(yb, r2) and not torch.equal(ya, yb)
            bad += 0 if ok else 1
            print(f"T={t}: {'ok' if ok else 'MISMATCH'} (local routes {la})", flush=True)
        print("SIG_DEVICE", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(0 if bad == 0 else 1)

    for xs, t, m in cases(d):
        x, ids, tw = H.make_case(d, t, m, xs, a.e_min, a.e_local)
        y = run(x.to("hpu"), ids.to("hpu"), tw.to(torch.bfloat16).to("hpu")).cpu().float()
        res["cases"][f"x{xs}/T{t}/{m}"] = {"y": y}
        print(f"{a.impl} x{xs}/T{t}/{m}: done", flush=True)
    torch.save(res, os.path.join(a.out, f"{a.impl}_{tag(a)}.pt"))


if __name__ == "__main__":
    main()
