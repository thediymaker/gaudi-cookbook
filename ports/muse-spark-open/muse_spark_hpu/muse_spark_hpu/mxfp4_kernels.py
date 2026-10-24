# SPDX-License-Identifier: Apache-2.0
"""MUSE_SPARK_MXFP4_KERNELS=1: decode experts on the cookbook's MXFP4 TPC kernels (kernels/mxfp4-moe, variant muse).

Off by default. With the switch off nothing here touches the environment and moe.py runs the stock op.

The stock op (torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights) unpacks every touched expert slice to bf16
in memory before its GEMMs, which was about 85% of a decode step. With the switch on, a pure decode step of at
most MAX_TOKENS tokens runs the expert call on two kernels that read the packed E2M1 + E8M0 weights directly:

    p [T, Hm] -> permute -> mxk_w13x (w13 GEMV of every local route, silu(gate) * up in fp32 -> bf16 [T*k, Is])
                         -> mxk_w2xr (w2 GEMV per route -> bf16 [T*k, Hm]; zero rows for routes on the other EP group)

Prefill and mixed steps, the bf16 expert layers (0 and 61) and any layout the kernels do not cover keep the
stock op. Routing, the all-reduce, the per-route norm and the weighted sum are unchanged.

setup_env() must run in every process before its device opens, because the graph compiler reads
GC_KERNEL_PATH then; patches/kernel_env_hook.py calls it from vllm_gaudi/__init__.py. It sets:
  GC_KERNEL_PATH            the kernel library (it forwards every stock GUID, so it is the only file and the
                            recipe cache stays on); the previous single library goes to MXK_TPC_FWD_LIB
  PT_HPU_RECIPE_CACHE_CONFIG a sibling directory <dir>_mxk-<version>/ (cache keys do not cover the library)
  PT_HPU_LAZY_ACC_PAR_MODE  0 (with the lazy accumulation thread on, a custom op fed by a view of a tensor made
                            in the same lazy graph fails intermittently)
If the libraries are missing, the switch turns itself off and says so.
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("MUSE_SPARK_MXFP4_KERNELS", "0") == "1"
LIB_DIR = "/opt/muse-spark-open/mxfp4-kernels"  # install.sh builds the variant here
MAX_TOKENS = 4  # the recipe's --max-num-seqs; parity and timing were measured at T <= 4
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
NO_LIMIT = 3.0e38  # silu(gate) * up without a clamp
MAX_ROUTES = 128  # kernel limit on T * top_k per call
MAX_K = 6144  # kernel limit on the GEMV input width
_ops_loaded = False


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_op_lazy.so")


def lib_version(path: str | None = None) -> str | None:
    """The 'mxk_tpc <version>' string embedded in the kernel library, None if absent."""
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
    """Register the kernel library with the graph compiler. Idempotent: a spawned worker inherits the settings."""
    global ENABLED
    if not ENABLED:
        return
    tpc = tpc_lib()
    ver = lib_version(tpc)
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = False
        print(f"[muse-spark-open] MUSE_SPARK_MXFP4_KERNELS=1 but no kernel build in {LIB_DIR}: stock MoE op")
        return
    rest = [p for p in os.environ.get("GC_KERNEL_PATH", STOCK).split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"[muse-spark-open] GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only")
        os.environ["MXK_TPC_FWD_LIB"] = STOCK
    os.environ["GC_KERNEL_PATH"] = tpc
    rc = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if rc:
        path, _, rest_cfg = rc.partition(",")
        tag = "mxk-" + ver.split()[-1]
        if path and tag not in path:
            path = path.rstrip("/") + "_" + tag + "/"
            os.makedirs(path, exist_ok=True)
            os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def env_ok() -> bool:
    """True if this process registered the library before its device opened."""
    return os.environ.get("GC_KERNEL_PATH") == tpc_lib() and os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE") == "0"


def load_ops() -> None:
    global _ops_loaded
    if not _ops_loaded:
        import torch

        torch.ops.load_library(op_lib())
        _ops_loaded = True


def permute(x):
    """x [R, K] bf16 -> the kernels' per-512-column layout: xp[kc + 128 p + i] = x[kc + 4 i + p]. K % 512 == 0."""
    R, K = x.shape
    return x.reshape(R, K // 512, 128, 4).transpose(-1, -2).reshape(R, K)


def unsupported(n: int, Hm: int, Is: int, k: int) -> str | None:
    """Why the kernels cannot serve a layer with n local experts, widths Hm / Is and top-k k (None if they can)."""
    if n > 128:
        return f"{n} local experts > 128"
    if Hm % 512 or Is % 512:
        return f"Hm {Hm} / Is {Is} not multiples of 512"
    if Hm > MAX_K or Is > MAX_K:
        return f"Hm {Hm} / Is {Is} > {MAX_K}"
    if MAX_TOKENS * k > MAX_ROUTES:
        return f"{MAX_TOKENS} tokens x top-{k} > {MAX_ROUTES} routes"
    return None
