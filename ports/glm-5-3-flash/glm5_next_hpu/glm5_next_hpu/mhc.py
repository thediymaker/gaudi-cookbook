# SPDX-License-Identifier: Apache-2.0
"""mHC Sinkhorn as one TPC kernel (switch GLM5_NEXT_MHC_SINKHORN=kernel; default torch = vLLM's op unchanged).

GLM-5.3-Flash runs vLLM's pure-torch mHC ops at two sites per layer, 90 `mhc_pre` calls per decode step. Each runs a
20-iteration Sinkhorn normalisation of a 4 x 4 matrix per token as about 78 small dependent device ops. The kernel
(ports/glm-5-3-flash/kernels/glm_mhc_sinkhorn) runs the whole loop in one launch, with the device's own reduction
order and divide, so the result is bit-identical to the stock loop on Gaudi 2.

Two pieces:

* `setup_env()` registers the kernel library with the graph compiler. The compiler reads GC_KERNEL_PATH when a
  process opens its device, so the mhc-kernel-env patch calls this from vllm_gaudi's package import, which every vLLM
  process runs first. It also sets PT_HPU_LAZY_ACC_PAR_MODE=0: with the lazy accumulation thread on, a custom op fed
  by a view made in the same lazy graph fails intermittently.
* `install()` (from the plugin's register) rebinds `vllm.model_executor.kernels.mhc.mhc_pre_torch`, which vLLM's
  `MHCPreOp` looks up at call time. The wrapper repeats the stock ops in the stock order and swaps only the loop.
  Calls with more than 1,024 tokens (prompt chunks) and non-bf16 inputs take the stock function.

The libraries come from ports/glm-5-3-flash/kernels/build.sh: libmxk_tpc.so (this kernel plus every stock kernel,
forwarded, so GC_KERNEL_PATH stays one file) and libmxk_op_lazy.so (`torch.ops.custom_op.glm_mhc_sinkhorn`).
GLM5_NEXT_MHC_LIB names their directory (default: the build output inside the port). If they are missing the
Sinkhorn stays on the stock loop and one warning says why.
"""

from __future__ import annotations

import os
from pathlib import Path

SINKHORN = os.environ.get("GLM5_NEXT_MHC_SINKHORN", "torch")
MAX_ROWS = 1024
STOCK_LIB = "/usr/lib/habanalabs/libtpc_kernels.so"
_DEFAULT_LIB = Path(__file__).resolve().parents[2] / "kernels" / "dist"
LIB_DIR = Path(os.environ.get("GLM5_NEXT_MHC_LIB", str(_DEFAULT_LIB)))
_state: dict = {"kernel": None}


def _tpc_lib() -> Path:
    return LIB_DIR / "libmxk_tpc.so"


def _op_lib() -> Path:
    return LIB_DIR / "libmxk_op_lazy.so"


def _check_switch() -> None:
    if SINKHORN not in ("torch", "kernel"):
        raise ValueError(f"GLM5_NEXT_MHC_SINKHORN={SINKHORN!r}: expected 'torch' or 'kernel'")


def setup_env() -> None:
    """Point GC_KERNEL_PATH at the kernel library. Must run before the process opens its device."""
    _check_switch()
    if SINKHORN != "kernel":
        return
    tpc = str(_tpc_lib())
    if not (_tpc_lib().is_file() and _op_lib().is_file()):
        print(f"[glm5_next_hpu] GLM5_NEXT_MHC_SINKHORN=kernel but no kernel libraries in {LIB_DIR}; stock loop runs")
        return
    cur = os.environ.get("GC_KERNEL_PATH", STOCK_LIB)
    if cur != tpc:
        rest = [p for p in cur.split(":") if p and p != tpc]
        # The library forwards every GUID it does not own to this one.
        os.environ["MXK_TPC_FWD_LIB"] = rest[0] if len(rest) == 1 else STOCK_LIB
        os.environ["GC_KERNEL_PATH"] = tpc
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def _log():
    from vllm.logger import init_logger

    return init_logger("vllm.glm5_next_hpu.mhc")  # under vllm.* so the record reaches a handler


def _kernel_ok() -> bool:
    if _state["kernel"] is None:
        ok, why = False, ""
        if os.environ.get("GC_KERNEL_PATH") != str(_tpc_lib()):
            why = "the kernel library was not registered before the device opened (GC_KERNEL_PATH)"
        else:
            try:
                import torch

                torch.ops.load_library(str(_op_lib()))
                ok = hasattr(torch.ops.custom_op, "glm_mhc_sinkhorn")
                why = "" if ok else "the op library has no glm_mhc_sinkhorn"
            except Exception as e:  # noqa: BLE001
                why = f"{type(e).__name__}: {e}"
        _state["kernel"] = ok
        if ok:
            _log().info("glm5_next_hpu: mHC Sinkhorn runs as one TPC kernel")
        else:
            _log().warning("glm5_next_hpu: mHC Sinkhorn kernel unavailable, stock loop runs: %s", why)
    return _state["kernel"]


def _sinkhorn_kernel(comb, iters: int, eps: float):
    """comb [N, 4, 4] fp32 (softmax + eps applied) -> the normalised matrices. The kernel takes one matrix per lane."""
    import torch

    n = comb.shape[0]
    ct = comb.reshape(n, 16).t().contiguous()
    return torch.ops.custom_op.glm_mhc_sinkhorn(ct, iters, eps).t().reshape(n, 4, 4)


def make_pre(stock):
    """`mhc_pre_torch` with the Sinkhorn loop on the kernel. Every other op is the stock one, in the stock order."""
    import torch

    def mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
    ):
        hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
        num_tokens = residual.numel() // (hc_mult * hidden_size)
        if (
            hc_mult != 4
            or num_tokens == 0
            or num_tokens > MAX_ROWS
            or residual.dtype != torch.bfloat16
            or not _kernel_ok()
        ):
            return stock(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
            )
        outer_shape = residual.shape[:-2]
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(x, fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
        pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
        pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
        post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
        post_mix = torch.sigmoid(post_logits) * hc_post_mult_value
        comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[2] + hc_base[
            2 * hc_mult :
        ].view(1, hc_mult, hc_mult)
        comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
        comb_mix = _sinkhorn_kernel(comb_mix, sinkhorn_repeat, hc_sinkhorn_eps)
        layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1).to(torch.bfloat16)
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    mhc_pre_torch.__wrapped__ = stock
    return mhc_pre_torch


def install() -> None:
    """Rebind vLLM's torch mHC pre op when the switch is on. Idempotent."""
    _check_switch()
    if SINKHORN != "kernel":
        return
    import vllm.model_executor.kernels.mhc as mhc_kernels

    if getattr(mhc_kernels.mhc_pre_torch, "__wrapped__", None) is not None:
        return
    mhc_kernels.mhc_pre_torch = make_pre(mhc_kernels.mhc_pre_torch)
