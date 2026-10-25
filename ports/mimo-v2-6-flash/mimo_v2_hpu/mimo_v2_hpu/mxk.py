# SPDX-License-Identifier: Apache-2.0
"""MIMO_MXK=1: decode-sized MoE calls of MiMo-V2.6-Pro on the MXFP4 TPC decode kernels (torch.compile mode).

Off by default. With the switch off, setup_env() returns at once and the MXFP4 experts run the stock vllm-gaudi
op (VllmMixtureOfExpertsOpMXFP4, torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights). The switch only applies to
MXFP4 experts (MIMO_EXPERT_DTYPE=mxfp4); the FP8 experts of the Flash recipe never use it.

With the switch on, a routed-expert call with at most MIMO_MXK_MAX_TOKENS tokens (and at most 128 routes) runs two
kernels that read the packed E2M1 weights and E8M0 scales the model keeps on the device:

    x [T, 6144] -> permute per 512-k chunk
                -> mxk_w13x   w13 GEMV of every local route, rows [gate; up], silu(gate) * up in fp32, bf16 out
                -> mxk_w2x    w2 GEMV and the routing-weighted sum over the token's local routes, bf16 [T, 6144]

in place of the stock op, which unpacks every expert slot it runs to bf16 before an MME GEMM. Prefill and larger
calls keep the stock op. The router and the all-reduce are unchanged.

The kernels are the `glm` variant of kernels/mxfp4-moe, built with --pt2 (install.sh builds it into MIMO_MXK_LIB).
MiMo-V2.6-Pro's per-rank expert shapes at TP8 (K 6144, I 2048, 48 local experts, top-8) are inside that variant's
caps (K a multiple of 512 up to 6144, I a multiple of 512 up to 3072, at most 128 local experts, 16 tokens and 128
routes per call):
  libmxk_tpc.so      the kernel library: the variant's GUIDs plus forwarding of every other GUID to the stock
                     library, so GC_KERNEL_PATH stays a single file and the recipe cache keeps working;
  libmxk_op_pt2.so   the torch.ops.custom_op.* ops on the PT2 API, which torch.compile mode needs (the lazy-mode
                     library does not trace).

setup_env() must run before the first HPU op of every process that opens a device, because the graph compiler
reads GC_KERNEL_PATH when the device opens. The plugin's register() calls it; vLLM loads general plugins in the
API server process while it parses the command line, before it starts the engine and the workers, which inherit
the environment. With the switch on it sets:
  GC_KERNEL_PATH=<lib>/libmxk_tpc.so and MXK_TPC_FWD_LIB=<the previous single library>;
  PT_HPU_RECIPE_CACHE_CONFIG to a sibling directory named after the kernel build (cache keys do not cover the
  kernel library).
The PT2 ops register their kernel with the bridge on their first eager call, and a compiled graph does not make
that call, so the model calls the ops once eagerly per process after loading (MiMoV2HpuMoE.build_op).
If the libraries are missing, the mode is not compile mode, or a layer does not match the kernels, that layer keeps
the stock op and the log says so once.

Environment:
  MIMO_MXK=1               turn the kernels on (default 0)
  MIMO_MXK_MAX_TOKENS=N    per-call token cap, 1..16 (default 16: covers decode at --max-num-seqs 8 and 16)
  MIMO_MXK_LIB=DIR         directory with the two libraries (default /opt/gaudi-cookbook-mimo/mxk)
"""

from __future__ import annotations

import logging
import os

ENABLED = os.environ.get("MIMO_MXK", "0") == "1"
MAX_TOKENS = int(os.environ.get("MIMO_MXK_MAX_TOKENS", "16"))
LIB_DIR = os.environ.get("MIMO_MXK_LIB", "/opt/gaudi-cookbook-mimo/mxk")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
VARIANT = "glm"
MAX_ROUTES = 128  # kernel limit on T * top_k per call
MAX_LOCAL_EXPERTS = 128
MAX_K = 6144  # mxk_w13x stages the activation row in vector local memory
MAX_I = 3072
INF = 3.0e38
TAG = "[mimo-mxk]"
logger = logging.getLogger("vllm.mimo_v2_hpu.mxk")  # under vLLM's logger, like init_logger

_state = None  # None: not tried; False: stock op; "loaded"; "registered" (eager call made)
_logged: set[str] = set()


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_op_pt2.so")


def lib_version(path: str | None = None) -> str | None:
    """The 'mxk_tpc <version>' string embedded in a kernel library, None if absent."""
    try:
        with open(path or tpc_lib(), "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = data.find(b"mxk_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def _once(key: str, level: int, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def warn_once(key: str, msg: str, *args) -> None:
    _once(key, logging.WARNING, msg, *args)


def setup_env() -> None:
    """Register the kernel library with the graph compiler. Idempotent; spawned processes inherit the result."""
    global ENABLED
    if not ENABLED:
        return
    tpc = tpc_lib()
    ver = lib_version(tpc)
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = False
        print(f"{TAG} MIMO_MXK=1 but {LIB_DIR} has no libmxk_tpc.so + libmxk_op_pt2.so: stock MoE op", flush=True)
        return
    if not ver.split()[-1].startswith(VARIANT + "-"):
        ENABLED = False
        print(f"{TAG} {tpc} is {ver}, not a '{VARIANT}' build: stock MoE op", flush=True)
        return
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"{TAG} GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only", flush=True)
        os.environ["MXK_TPC_FWD_LIB"] = STOCK
    os.environ["GC_KERNEL_PATH"] = tpc
    cache_cfg = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if cache_cfg:
        path, _, rest_cfg = cache_cfg.partition(",")
        tag = "mxk-" + ver.split()[-1]
        if path and tag not in path:
            path = path.rstrip("/") + "_" + tag + "/"
            try:
                os.makedirs(path, exist_ok=True)
                os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
            except OSError as e:
                print(f"{TAG} cannot create recipe cache dir {path}: {e}; keeping {cache_cfg}", flush=True)


def env_ok() -> bool:
    """True if this process's graph compiler sees the kernel library (setup_env ran before the device opened)."""
    return os.environ.get("GC_KERNEL_PATH") == tpc_lib()


def layer_reason(n_local: int, hidden: int, inter: int, expert_dtype: str) -> str | None:
    """None if the kernels cover a MoE layer of this shape on this rank; otherwise the reason it keeps the stock op."""
    if expert_dtype != "mxfp4":
        return f"experts are {expert_dtype}, the kernels read MXFP4"
    if n_local > MAX_LOCAL_EXPERTS:
        return f"{n_local} local experts > {MAX_LOCAL_EXPERTS}"
    if hidden % 512 or hidden > MAX_K:
        return f"hidden {hidden} is not a multiple of 512 up to {MAX_K}"
    if inter % 512 or inter > MAX_I:
        return f"intermediate {inter} is not a multiple of 512 up to {MAX_I}"
    return None


def covers(tokens: int, topk: int) -> bool:
    return 1 <= tokens <= MAX_TOKENS and tokens * topk <= MAX_ROUTES


def load() -> bool:
    """Load the op library once per process. False (stock op) if the switch is off or anything is missing."""
    global _state
    if _state is None:
        _state = False
        if not ENABLED:
            return False
        if not 1 <= MAX_TOKENS <= 16:
            warn_once("cap", "%s MIMO_MXK_MAX_TOKENS=%d is outside 1..16: stock MoE op", TAG, MAX_TOKENS)
        elif os.environ.get("PT_HPU_LAZY_MODE", "1") != "0":
            warn_once("mode", "%s needs torch.compile mode (PT_HPU_LAZY_MODE=0): stock MoE op", TAG)
        elif not env_ok():
            warn_once("env", "%s GC_KERNEL_PATH is not %s (setup_env did not run): stock MoE op", TAG, tpc_lib())
        else:
            import torch

            torch.ops.load_library(op_lib())
            if hasattr(torch.ops.custom_op, "mxk_w13x") and hasattr(torch.ops.custom_op, "mxk_w2x"):
                _state = "loaded"
            else:
                warn_once("ops", "%s %s has no mxk_w13x / mxk_w2x: stock MoE op", TAG, op_lib())
    return _state in ("loaded", "registered")


def permute(x):
    """x [T, K] -> the kernels' per-512-k-chunk layout: xp[kc + 128 p + i] = x[kc + 4 i + p]. K % 512 == 0."""
    t, k = x.shape
    return x.reshape(t, k // 512, 128, 4).transpose(-1, -2).reshape(t, k)


def moe(x, topk_idx, topk_w, w13, s13, w2, s2, top_k: int, e_min: int):
    """Routed experts of this rank for x [T, K] bf16: routes to other ranks' experts add nothing."""
    import torch

    ids = topk_idx.reshape(-1).to(torch.int32)
    co = torch.ops.custom_op
    # plain silu(gate) * up: alpha 1, no clamp, beta 0, activation scale 1; rows [gate; up]
    act = co.mxk_w13x(permute(x), ids, w13, s13, top_k, e_min, 1.0, INF, 0.0, 1.0, False)
    return co.mxk_w2x(act, ids, topk_w.reshape(-1).to(torch.bfloat16), w2, s2, top_k, e_min)


def register_eager(w13, s13, w2, s2, top_k: int, e_min: int, hidden: int, n_local: int, inter: int) -> None:
    """One eager call per process, so the PT2 ops are in the bridge's kernel registry before a compiled graph."""
    global _state
    if _state != "loaded":
        return
    import torch

    dev = w13.device
    ids = torch.full((1, top_k), e_min, dtype=torch.int32, device=dev)
    tw = torch.full((1, top_k), 1.0 / top_k, dtype=torch.bfloat16, device=dev)
    moe(torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev), ids, tw, w13, s13, w2, s2, top_k, e_min)
    torch.hpu.synchronize()
    _state = "registered"
    logger.info(
        "%s enabled: %s, E_local=%d K=%d I=%d, <= %d tokens, experts %d..%d",
        TAG,
        lib_version(),
        n_local,
        hidden,
        inter,
        MAX_TOKENS,
        e_min,
        e_min + n_local - 1,
    )
