# SPDX-License-Identifier: Apache-2.0
"""DSV41_MXK=1: decode MoE experts on the cookbook's MXFP4 TPC kernels (kernels/mxfp4-moe, variant dsv41).

Off by default; with the flag off nothing here touches the environment and DSV41MoEHPU runs the stock op.

With the flag on, a routed-expert call of at most MAX_TOKENS tokens (16: a decode batch of up to 16 requests, and
short prefill chunks) becomes two kernels that read the served packed E2M1 + E8M0 weights directly:
  x [T, D] -> permute -> mxk_w13xi (w13 GEMV of every local route, interleaved gate/up rows, clamped SwiGLU in fp32
                                    -> bf16 [T*k, Ip])
                      -> mxk_w2x   (w2 GEMV + routing-weighted sum over the token's local routes -> bf16 [T, D])
in place of torch.ops.hpu.mixture_of_experts.bias_mxfp4_fused_weights, which unpacks every touched expert to bf16
in memory first. The kernels run on the scales finalize() serves (E8M0 +7): alpha 2^-7, limit 10 * 2^7, beta 0,
ascale 2^-14 is the exact V4.1 clamp (min(g, 10) * sigmoid(min(g, 10)) * clamp(u, +-10)). The routing weights go in
unscaled (the stock op takes them x 2^-14). The shared expert and the all-reduce are unchanged.

The same kernel library carries the port's mHC Sinkhorn kernel (kernels/dsv41_sinkhorn, DSV41_HC_SINKHORN=kernel), so
DSV41_MXK=0 turns both off.

Libraries (DSV41_MXK_LIB, default /opt/mxfp4-moe/lib, built by install.sh):
  libmxk_tpc.so      one kernel library: the dsv41 kernels + forwarding of every other GUID to the stock
                     libtpc_kernels.so, so GC_KERNEL_PATH stays a single file (the recipe cache needs that);
  libmxk_op_lazy.so  torch.ops.custom_op.mxk_w13x (interleaved=True runs mxk_w13xi) / mxk_w2x / dsv41_sinkhorn.

setup_env() must run before the first HPU op of every process that opens a device: the graph compiler reads
GC_KERNEL_PATH when the device opens. patches/dsv41-kernel-env puts a call at the top of vllm_gaudi/__init__.py
(the platform plugin, imported by every vLLM process before the device opens). It sets
  GC_KERNEL_PATH=<libmxk_tpc.so>, MXK_TPC_FWD_LIB=<the previous single GC_KERNEL_PATH>;
  PT_HPU_RECIPE_CACHE_CONFIG -> a sibling dir <dir>_mxk-<version>/ (cache keys do not cover the kernel library);
  PT_HPU_LAZY_ACC_PAR_MODE=0 (with the lazy accumulation thread on, a custom op fed by a view of a tensor made in the
  same lazy graph fails intermittently).
If the libraries are missing, the flag turns itself off (stock path) and says so.
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("DSV41_MXK", "0") == "1"
MAX_TOKENS = 16
LIB_DIR = os.environ.get("DSV41_MXK_LIB", "/opt/mxfp4-moe/lib")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
MAX_ROUTES = 128  # kernel limit on T * top_k per call
MAX_K = 6144  # kernel limit on the w13 GEMV input width (x row staged in vector local memory)
_ops_loaded = False


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_op_lazy.so")


def lib_version(path: str | None = None) -> str | None:
    """'mxk_tpc <version>' string of a single-file kernel library, None if absent."""
    try:
        with open(path or tpc_lib(), "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = data.find(b"mxk_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def setup_env() -> None:
    """Before the first HPU op: register the kernel library with the graph compiler. Idempotent (a spawned
    worker inherits the parent's settings and leaves them as they are)."""
    global ENABLED
    if not ENABLED:
        return
    tpc = tpc_lib()
    ver = lib_version(tpc)
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = False
        print(
            f"[dsv41-mxk] DSV41_MXK=1 but no libmxk_tpc.so + libmxk_op_lazy.so in {LIB_DIR}: staying on the stock "
            "MoE path",
            flush=True,
        )
        return
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"[dsv41-mxk] GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only", flush=True)
        os.environ["MXK_TPC_FWD_LIB"] = STOCK
    os.environ["GC_KERNEL_PATH"] = tpc
    rc = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if rc:
        path, _, rest_cfg = rc.partition(",")
        tag = "mxk-" + ver.split()[-1]
        if path and tag not in path:
            path = path.rstrip("/") + "_" + tag + "/"
            try:
                os.makedirs(path, exist_ok=True)
                os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
            except OSError as e:
                print(f"[dsv41-mxk] cannot create recipe cache dir {path}: {e}; keeping {rc}", flush=True)
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def env_ok() -> bool:
    """True if this process's graph compiler sees our kernel library (setup_env ran before the device opened)."""
    return os.environ.get("GC_KERNEL_PATH") == tpc_lib() and os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE") == "0"


def load_ops() -> bool:
    global _ops_loaded
    if not _ops_loaded:
        import torch

        torch.ops.load_library(op_lib())
        _ops_loaded = True
    return True


def permute(x):
    """x [R, K] bf16 -> the kernels' per-512-k-chunk layout: xp[kc + 128 p + i] = x[kc + 4 i + p]. K % 512 == 0."""
    R, K = x.shape
    return x.reshape(R, K // 512, 128, 4).transpose(-1, -2).reshape(R, K)


def supported(n: int, K: int, inter: int, k: int) -> str | None:
    """None if the kernels cover a layer with n local experts, hidden size K, expert intermediate size `inter`
    (interleaved w13) and top-k k at MAX_TOKENS tokens; otherwise the reason."""
    if n > 128:
        return f"{n} local experts > 128"
    if K % 512 or K > MAX_K:
        return f"hidden {K} not a multiple of 512 or > {MAX_K}"
    if inter % 256:
        return f"intermediate {inter} not a multiple of 256"
    if MAX_TOKENS * k > MAX_ROUTES:
        return f"{MAX_TOKENS} tokens x top-{k} > {MAX_ROUTES} routes"
    return None
