#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card parity test of one MXFP4 MoE kernel variant against the stock vllm-gaudi op and a CPU fp32 reference.

Run one implementation per process (lazy-mode legacy custom ops cache recipes by input signature, so two
implementations with identical inputs must not share a process). ``run_parity.sh --variant V`` runs the sequence:

    parity_hpu.py --variant V --impl ref    CPU fp32 reference (no device needed)
    parity_hpu.py --variant V --impl stock  the stock op as the model serves it
    parity_hpu.py --variant V --impl ours   the kernels (--lib DIR: a build of variant V)
    parity_hpu.py --variant V --impl sig    every op of the variant twice with identical input shapes and different
                                            data in ONE lazy graph, per T = 1 .. the variant's token cap
    parity_hpu.py --variant V --impl cmp    compare the saved outputs; exit 1 on a failed check

Shapes are the per-rank shapes the model serves (mxk_harness.VARIANTS). Weights: random MXFP4 weights of those
shapes (default), or one layer of a local checkpoint (--checkpoint; see mxk_harness.load_layer for the formats).
The glm variant also runs its prefill unpack path at 256, 1024 and 4096 tokens.

Pass criteria (cmp):
  decode   for every case the kernel's rel_l2 against the fp32 reference is below --max-rel (default 3e-3) and not
           above the stock op's                                                            -> PARITY PASS
  prefill  the unpack is bit-exact against the CPU unpack, and the unpack path's rel_l2 against the reference is
           within 2% of the stock op's, with rel_l2 to the stock op below 1e-3              -> PREFILL PASS
  sig      results inside one graph are bit-identical to the same calls alone, and the two calls differ
                                                                                            -> SIG_DEVICE PASS
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
    ap.add_argument("--variant", default="gptoss", choices=sorted(H.VARIANTS))
    ap.add_argument("--impl", required=True, choices=["ref", "stock", "ours", "sig", "cmp"])
    ap.add_argument("--checkpoint", default="", help="local checkpoint (default: random weights)")
    ap.add_argument("--layer", type=int, default=None, help="checkpoint layer (default: the variant's)")
    ap.add_argument("--e-local", type=int, default=None, help="local experts on this card (default: as served)")
    ap.add_argument("--e-min", type=int, default=0, help="first local expert")
    ap.add_argument("--inter", type=int, default=None, help="intermediate size per rank (random weights only)")
    ap.add_argument("--tp-rank", type=int, default=0, help="TP slice of the experts' intermediate dim (glm, muse)")
    ap.add_argument("--lib", default=os.environ.get("MXFP4_TPC_MOE_LIB", os.path.join(HERE, "..", "dist")))
    ap.add_argument("--out", default="parity_out")
    ap.add_argument("--max-rel", type=float, default=3e-3)
    a = ap.parse_args()
    a.spec = dict(H.VARIANTS[a.variant])
    if a.inter:
        a.spec["inter"] = a.inter
    a.e_local = a.e_local or a.spec["e_local"]
    a.layer = a.spec["layer"] if a.layer is None else a.layer
    return a


def layer(a):
    if a.checkpoint:
        return H.load_layer(a.variant, a.checkpoint, a.layer, a.e_local, a.e_min, a.tp_rank)
    return H.random_layer(a.variant, a.e_local, inter=a.spec["inter"])


def decode_cases(a, d):
    out = [(xs, t, m) for xs in a.spec["x_rms"] for (t, m) in H.decode_cases(a.spec)]
    if a.variant == "gptoss" and a.checkpoint:
        out += [(xs, t, "real") for xs in a.spec["x_rms"] for t in (4, 32)]
    return out


def tag(a):
    w = f"ck_L{a.layer}" if a.checkpoint else "rand"
    return f"{a.variant}_E{a.e_local}_I{a.spec['inter']}_m{a.e_min}_{w}"


def compare(a):
    import torch

    files = {os.path.basename(p).split("_", 1)[0]: p for p in glob.glob(os.path.join(a.out, f"*_{tag(a)}.pt"))}
    if "ref" not in files or "ours" not in files:
        sys.exit(f"need ref and ours results in {a.out} (have {sorted(files)})")
    ref = torch.load(files["ref"], weights_only=False)
    ours = torch.load(files["ours"], weights_only=False)
    stock = torch.load(files["stock"], weights_only=False) if "stock" in files else {"cases": {}, "prefill": {}}
    print(f"[{a.variant}] decode: K {ref['k']} I {ref['inter']} E_local {a.e_local} top-{ref['topk']}")
    print(f"{'case':18s}{'local':>6s}{'ours rel':>11s}{'stock rel':>11s}{'ours vs stock':>15s}  result")
    bad = 0
    for key, r in ref["cases"].items():
        o, s = ours["cases"].get(key), stock["cases"].get(key)
        if o is None:
            print(f"{key:18s} missing in ours  FAIL")
            bad += 1
            continue
        ro = H.rel_l2(o["y"], r["y"])
        rs = H.rel_l2(s["y"], r["y"]) if s else float("nan")
        os_ = H.rel_l2(o["y"], s["y"]) if s else float("nan")
        ok = ro <= a.max_rel and (s is None or ro <= rs)
        bad += 0 if ok else 1
        print(f"{key:18s}{r['local']:6d}{ro:11.2e}{rs:11.2e}{os_:15.2e}  {'ok' if ok else 'FAIL'}")
    print(f"[{a.variant}] PARITY", "PASS" if bad == 0 else f"FAIL ({bad} cases)")
    pbad = 0
    if ref.get("prefill"):
        print(f"{'prefill':18s}{'local':>6s}{'ours rel':>11s}{'stock rel':>11s}{'ours vs stock':>15s}  result")
        for key, r in ref["prefill"].items():
            o, s = ours["prefill"].get(key), stock["prefill"].get(key)
            if o is None or s is None:
                print(f"{key:18s} missing  FAIL")
                pbad += 1
                continue
            ro, rs, os_ = H.rel_l2(o["y"], r["y"]), H.rel_l2(s["y"], r["y"]), H.rel_l2(o["y"], s["y"])
            mad = (o["y"] - s["y"]).abs().max().item()
            ok = ro <= rs * 1.02 and os_ < 1e-3
            pbad += 0 if ok else 1
            res = "ok" if ok else "FAIL"
            print(f"{key:18s}{r['local']:6d}{ro:11.2e}{rs:11.2e}{os_:15.2e}  {res} (max abs {mad:.3g})")
        exact = ours.get("deq_exact", {})
        for key, v in exact.items():
            print(f"unpack {key}: {'exact' if v else 'MISMATCH'}")
            pbad += 0 if v else 1
        if not exact:
            pbad += 1
        print(f"[{a.variant}] PREFILL", "PASS" if pbad == 0 else f"FAIL ({pbad})")
    sys.exit(0 if bad == 0 and pbad == 0 else 1)


def sig(a, d, run, t_dev):
    """Two calls of the variant's ops with identical input shapes in one lazy graph vs the same calls alone."""
    import habana_frameworks.torch as ht
    import torch

    co = torch.ops.custom_op
    lo, hi = a.e_min, a.e_min + a.e_local
    bad = 0
    for t in range(1, a.spec["max_tokens"] + 1):
        if t * d["topk"] > 128:
            break
        xa, ida, twa = H.make_case(a.spec, d, t, "rand", 1.0, a.e_min, a.e_local)
        xb, idb, twb = H.make_case(a.spec, d, t, "hot", 2.0, a.e_min, a.e_local)
        A = [v.to("hpu") for v in (xa, ida, twa.to(torch.bfloat16))]
        B = [v.to("hpu") for v in (xb, idb, twb.to(torch.bfloat16))]
        alone = []
        for args in (A, B):
            y = run(*args)
            ht.core.mark_step()
            alone.append(y.cpu())
        deq = a.variant == "glm"
        e1, e2 = (7 * t) % a.e_local, (7 * t + 13) % a.e_local
        # The expert views are taken inside each graph, as the prefill path does: a legacy custom op reads a view
        # that was materialised in an earlier lazy graph wrongly (README, "Lazy-mode pitfalls").
        unpack = [("13", e1), ("13", e2), ("2", e1), ("2", e2)] if deq else []

        def unpacked(w, e):
            return co.mxk_deq(t_dev["w" + w][e], t_dev["s" + w][e], 0)

        for w, e in unpack:
            y = unpacked(w, e)
            ht.core.mark_step()
            alone.append(y.cpu())
        together = [run(*A), run(*B)]  # same lazy graph, identical input shapes
        together += [unpacked(w, e) for w, e in unpack]
        ht.core.mark_step()
        together = [y.cpu() for y in together]
        names = ["moe a", "moe b"] + (["deq w13 e1", "deq w13 e2", "deq w2 e1", "deq w2 e2"] if deq else [])
        diff = [n for n, x, y in zip(names, alone, together, strict=True) if not torch.equal(x, y)]
        same = [f"{names[i]} = {names[i + 1]}" for i in range(0, len(names), 2) if torch.equal(alone[i], alone[i + 1])]
        ok = not diff and not same
        bad += 0 if ok else 1
        la = int(((ida >= lo) & (ida < hi)).sum())
        why = "" if ok else f": in one graph != alone: {diff}, equal pairs: {same}"
        extra = ", + 4 unpacks" if deq else ""
        print(f"[{a.variant}] T={t}: {'ok' if ok else 'MISMATCH'} (local routes {la}{extra}){why}")
    print(f"[{a.variant}] SIG_DEVICE", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(0 if bad == 0 else 1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    if a.impl == "cmp":
        compare(a)
    if a.impl in ("ours", "sig"):
        H.hpu_env_for_kernels(os.path.abspath(a.lib))  # before torch / habana_frameworks are imported
    import torch

    d = layer(a)
    res = {"impl": a.impl, "k": d["k"], "inter": d["inter"], "topk": d["topk"], "cases": {}, "prefill": {}}
    prefill = [(t, "rand") for t in a.spec["prefill"]]

    if a.impl == "ref":
        ref = H.Reference(a.spec, d, a.e_min, a.e_local)
        for xs, t, m in decode_cases(a, d):
            x, ids, tw = H.make_case(a.spec, d, t, m, xs, a.e_min, a.e_local)
            y, local = ref(x, ids, tw)
            res["cases"][f"x{xs}/T{t}/{m}"] = {"y": y, "local": local}
            print(f"ref x{xs}/T{t}/{m}: {local} local routes", flush=True)
        for t, m in prefill:
            x, ids, tw = H.make_case(a.spec, d, t, m, 1.0, a.e_min, a.e_local)
            y, local = ref(x, ids, tw)
            res["prefill"][f"T{t}/{m}"] = {"y": y, "local": local}
            print(f"ref prefill T{t}/{m}: {local} local routes", flush=True)
        torch.save(res, os.path.join(a.out, f"ref_{tag(a)}.pt"))
        return

    import habana_frameworks.torch as ht

    t_dev = H.device_layer(a.spec, d, a.impl)
    if a.impl == "stock":
        run = H.stock_runner(a.spec, t_dev, d, a.e_local, a.e_min)
    else:
        H.load_ops(os.path.abspath(a.lib), a.variant)
        run = H.kernel_runner(a.spec, t_dev, d, a.e_min)
    if a.impl == "sig":
        sig(a, d, run, t_dev)

    def call(fn, x, ids, tw):
        y = fn(x.to("hpu"), ids.to("hpu"), tw.to(torch.bfloat16).to("hpu"))
        ht.core.mark_step()
        return y.cpu().float()

    for xs, t, m in decode_cases(a, d):
        x, ids, tw = H.make_case(a.spec, d, t, m, xs, a.e_min, a.e_local)
        res["cases"][f"x{xs}/T{t}/{m}"] = {"y": call(run, x, ids, tw)}
        print(f"{a.impl} x{xs}/T{t}/{m}: done", flush=True)
    if prefill:
        prun = H.prefill_runner(a.spec, t_dev, d, a.e_local, a.e_min, a.impl)
        for t, m in prefill:
            x, ids, tw = H.make_case(a.spec, d, t, m, 1.0, a.e_min, a.e_local)
            res["prefill"][f"T{t}/{m}"] = {"y": call(prun, x, ids, tw)}
            print(f"{a.impl} prefill T{t}/{m}: done", flush=True)
        if a.impl == "ours":  # the unpack itself, bit for bit against the CPU unpack
            res["deq_exact"] = {}
            for e in sorted({0, a.e_local // 2, a.e_local - 1}):
                for w in ("13", "2"):
                    y = torch.ops.custom_op.mxk_deq(t_dev["w" + w][e], t_dev["s" + w][e], 0).cpu()
                    want = H.dequant(d["w" + w][e], d["s" + w][e]).to(torch.bfloat16)
                    res["deq_exact"][f"w{w}[{e}]"] = bool(torch.equal(y, want))
    torch.save(res, os.path.join(a.out, f"{a.impl}_{tag(a)}.pt"))


if __name__ == "__main__":
    main()
