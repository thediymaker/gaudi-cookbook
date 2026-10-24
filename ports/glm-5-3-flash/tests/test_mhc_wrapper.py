# SPDX-License-Identifier: Apache-2.0
"""The mHC wrapper repeats vLLM's stock `mhc_pre_torch` exactly, except the Sinkhorn loop (needs torch and vLLM).

On CPU the kernel call is replaced by the kernel's arithmetic written in torch (column sums sequential, row sums
pairwise), which on CPU is not the device's reduction order, so the comparison allows float rounding. The bit-identity
of the real kernel against the stock loop on Gaudi 2 is a device gate (tests/hpu/run_gates.sh sinkhorn).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
mhc_kernels = pytest.importorskip("vllm.model_executor.kernels.mhc")

_MHC = Path(__file__).resolve().parents[1] / "glm5_next_hpu" / "glm5_next_hpu" / "mhc.py"
_spec = importlib.util.spec_from_file_location("glm5_next_mhc", _MHC)
mhc = importlib.util.module_from_spec(_spec)
sys.modules["glm5_next_mhc"] = mhc
_spec.loader.exec_module(mhc)


def kernel_in_torch(comb, iters, eps):
    c = comb.clone()

    def cols(x):
        return ((x[..., 0:1, :] + x[..., 1:2, :]) + x[..., 2:3, :]) + x[..., 3:4, :]

    def rows(x):
        return (x[..., 0:1] + x[..., 1:2]) + (x[..., 2:3] + x[..., 3:4])

    c = c / (cols(c) + eps)
    for _ in range(iters - 1):
        c = c / (rows(c) + eps)
        c = c / (cols(c) + eps)
    return c


@pytest.mark.parametrize("tokens", [1, 7, 32])
def test_wrapper_matches_stock(monkeypatch, tokens):
    stock = mhc_kernels.mhc_pre_torch
    monkeypatch.setattr(mhc, "_kernel_ok", lambda: True)
    monkeypatch.setattr(mhc, "_sinkhorn_kernel", kernel_in_torch)
    wrapped = mhc.make_pre(stock)

    gen = torch.Generator().manual_seed(tokens)
    hidden, hc = 64, 4
    residual = torch.randn(tokens, hc, hidden, generator=gen).to(torch.bfloat16)
    fn = torch.randn((2 + hc) * hc, hc * hidden, generator=gen) * 0.02
    scale = torch.rand(3, generator=gen)
    base = torch.randn((2 + hc) * hc, generator=gen) * 0.1
    args = (residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    for a, b in zip(stock(*args), wrapped(*args), strict=False):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert (a.float() - b.float()).abs().max() < 1e-5


def test_large_calls_take_the_stock_path(monkeypatch):
    calls = []
    monkeypatch.setattr(mhc, "_kernel_ok", lambda: True)
    wrapped = mhc.make_pre(lambda *a: calls.append(a) or "stock")
    residual = torch.zeros(mhc.MAX_ROWS + 1, 4, 8, dtype=torch.bfloat16)
    assert wrapped(residual, None, None, None, 0, 0, 0, 0, 20) == "stock"
    assert len(calls) == 1
