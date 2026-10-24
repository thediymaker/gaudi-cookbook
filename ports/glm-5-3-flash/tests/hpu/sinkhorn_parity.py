#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gate: the mHC Sinkhorn kernel is bit-identical to vLLM's stock `mhc_pre_torch` on Gaudi 2.

Runs on one free card in the port image, without a server (stop the server first, or run it before serving):

    python3 ports/glm-5-3-flash/tests/hpu/sinkhorn_parity.py

Random bf16 residuals at the model's shape (4 streams x 4096, 20 iterations) for 1 to 1,024 rows: every output of
the wrapped op must equal the stock op's exactly (torch.equal). Prints one PASS or FAIL line.
"""

from __future__ import annotations

import os
import sys

os.environ["GLM5_NEXT_MHC_SINKHORN"] = "kernel"
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

from glm5_next_hpu import mhc  # noqa: E402

mhc.setup_env()  # before the device opens

import habana_frameworks.torch.core as htcore  # noqa: E402
import torch  # noqa: E402
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch as stock  # noqa: E402


def main() -> int:
    if not mhc._kernel_ok():
        print("FAIL sinkhorn kernel library not loaded (see the warning above)")
        return 1
    wrapped = mhc.make_pre(stock)
    gen = torch.Generator().manual_seed(0)
    hc, hidden = 4, 4096
    worst = []
    for rows in (1, 2, 7, 32, 33, 256, 1024):
        residual = (torch.randn(rows, hc, hidden, generator=gen) * 4).to(torch.bfloat16).to("hpu")
        fn = (torch.randn((2 + hc) * hc, hc * hidden, generator=gen) * 0.02).to("hpu")
        scale = (torch.rand(3, generator=gen) * 2).to("hpu")
        base = (torch.randn((2 + hc) * hc, generator=gen) * 0.5).to("hpu")
        args = (residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
        a = stock(*args)
        htcore.mark_step()
        b = wrapped(*args)
        htcore.mark_step()
        for name, x, y in zip(("post", "comb", "layer_input"), a, b, strict=False):
            x, y = x.cpu(), y.cpu()
            if not torch.equal(x, y):
                worst.append(f"{name} at {rows} rows differs by {(x.float() - y.float()).abs().max().item():.3g}")
    if worst:
        print("FAIL sinkhorn " + "; ".join(worst))
        return 1
    print("PASS sinkhorn kernel equals the stock loop bit for bit at 1 to 1,024 rows (post, comb, layer_input)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
