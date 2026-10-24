#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card device checks of the port's math, before a full server run.

    python3 device_ops.py [CHECKPOINT_DIR]

- sliding-window attention with sinks (ops.swa_band_attention) on the device in bf16 against the fp32 reference,
  at the head layouts of TP8 (8 query heads, 1 KV head per rank) and of a wider split;
- the MXFP4 to FP8 expert re-encode on the device: exact, and byte-identical to the CPU result, on a sample of real
  expert tensors (needs CHECKPOINT_DIR, the public checkpoint or the staged copy);
- for information, the device's fp32 cos at a large argument, which is why RoPE tables are built on the CPU.
"""

import json
import math
import os
import sys

import habana_frameworks.torch  # noqa: F401  (registers the hpu device)
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mimo_v2_hpu"))
from mimo_v2_hpu import ops  # noqa: E402

DEV = "hpu"
failed = False


def verdict(ok: bool, name: str, detail: str) -> None:
    global failed
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def check_swa() -> None:
    torch.manual_seed(0)
    W, d, bs = 128, 192, 128
    worst = 0.0
    for nh, kvh in ((8, 1), (16, 2)):
        for S, ctx in ((1, 0), (128, 0), (129, 0), (300, 0), (37, 50), (200, 128), (256, 200), (64, 1000)):
            q = torch.randn(1, S, nh, d) * 0.5
            k = torch.randn(1, S, kvh, d) * 0.5
            v = torch.randn(1, S, kvh, d)
            sinks = torch.linspace(-1, 4, nh)
            kc = vc = ct = fr = None
            if ctx:
                wc = 2 * bs
                fr = torch.tensor([max(-(-ctx // bs) * bs - wc, 0)])
                kc, vc, ct = torch.randn(1, wc, kvh, d) * 0.5, torch.randn(1, wc, kvh, d), torch.tensor([ctx])
            ref = ops.swa_reference(q, k, v, sinks, d**-0.5, W, kc, vc, ct, fr)

            def to(t):
                if t is None:
                    return None
                return t.to(DEV) if not t.is_floating_point() else t.to(torch.bfloat16).to(DEV)

            out = ops.swa_band_attention(to(q), to(k), to(v), to(sinks), d**-0.5, W, to(kc), to(vc), to(ct), to(fr))
            worst = max(worst, ((out.float().cpu() - ref).norm() / ref.norm()).item())
    verdict(worst < 2e-2, "swa-band-device", f"worst relative error {worst:.4f} in bf16 (bound 0.02)")


def check_experts(ckpt: str) -> None:
    from safetensors import safe_open

    with open(os.path.join(ckpt, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    n = bad = 0
    for layer, e in ((1, 0), (5, 77), (24, 200), (47, 255)):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
            with safe_open(os.path.join(ckpt, wm[name]), framework="pt") as f:
                w, s = f.get_tensor(name), f.get_tensor(name + "_scale")
            q_cpu, sc_cpu = ops.mxfp4_to_fp8(w, s)
            q_dev, sc_dev = ops.mxfp4_to_fp8(w.to(DEV), s.to(DEV))
            same = torch.equal(q_dev.cpu().view(torch.uint8), q_cpu.view(torch.uint8)) and torch.equal(
                sc_dev.cpu(), sc_cpu
            )
            exact = torch.equal(q_cpu.double() * sc_cpu.double()[:, None], ops.mxfp4_reference(w, s))
            n += 1
            bad += not (same and exact)
    verdict(bad == 0, "expert-fp8-device", f"{n - bad}/{n} expert tensors exact and identical to the CPU encode")


def show_cos() -> None:
    x = 200000.0
    dev = torch.cos(torch.tensor([x], dtype=torch.float32, device=DEV)).item()
    print(f"INFO fp32 cos({x:.0f}) on the device {dev:+.4f}, exact {math.cos(x):+.4f}", flush=True)


if __name__ == "__main__":
    check_swa()
    if len(sys.argv) > 1:
        check_experts(sys.argv[1])
    else:
        print("SKIP expert-fp8-device: pass CHECKPOINT_DIR")
    show_cos()
    sys.exit(1 if failed else 0)
