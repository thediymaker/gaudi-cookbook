# SPDX-License-Identifier: Apache-2.0
"""KIMI_MXK=1: decode-sized MoE calls of Kimi-K2.7 on the MXFP4 TPC decode kernels.

Off by default. With the switch off, setup_env() and install() return at once and the engine runs the stock
vllm-gaudi MXFP4 MoE op (torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights).

With the switch on, a routed-expert call with at most KIMI_MXK_MAX_TOKENS tokens (and at most 128 routes) runs two
kernels that read the packed E2M1 weights and E8M0 scales vLLM already keeps on the device:

    x [T, 7168] -> permute per 512-k chunk
                -> mxk_w13x   w13 GEMV of every local route, rows [gate; up], silu(gate) * up in fp32, bf16 out
                -> mxk_w2x    w2 GEMV and the routing-weighted sum over the token's local routes, bf16 [T, 7168]

in place of the stock op, which unpacks every expert slot it runs to bf16 before an MME GEMM. Prefill and larger
calls keep the stock op. The router, the shared expert, routed_scaling_factor and the all-reduce are unchanged.

The kernels come from the `kimi` variant of kernels/mxfp4-moe (install.sh builds it into KIMI_MXK_LIB):
  libmxk_tpc.so      the kernel library: the variant's GUIDs plus forwarding of every other GUID to the stock
                     library, so GC_KERNEL_PATH stays a single file and the recipe cache keeps working;
  libmxk_op_lazy.so  the torch.ops.custom_op.* ops for lazy mode.

setup_env() must run before the first HPU op of every process that opens a device, because the graph compiler
reads GC_KERNEL_PATH when the device opens. patches/mxk_hooks.py calls it from the top of vllm_gaudi/__init__.py
(the platform plugin, imported by every vLLM process first) and calls install() at the end of
vllm_gaudi/ops/hpu_mxfp4.py. With the switch on it sets:
  GC_KERNEL_PATH=<lib>/libmxk_tpc.so and MXK_TPC_FWD_LIB=<the previous single library>;
  PT_HPU_RECIPE_CACHE_CONFIG to a sibling directory named after the kernel build (cache keys do not cover the
  kernel library);
  PT_HPU_LAZY_ACC_PAR_MODE=0 (with the lazy accumulation thread on, a custom op fed by a view of a tensor made in
  the same lazy graph fails intermittently).
If the libraries are missing or a layer does not match the kernels, that layer keeps the stock op and the log says
so once.

Environment:
  KIMI_MXK=1               turn the kernels on (default 0)
  KIMI_MXK_MAX_TOKENS=N    per-call token cap, 1..16 (default 16: covers decode at --max-num-seqs 8)
  KIMI_MXK_LIB=DIR         directory with the two libraries (default /opt/gaudi-cookbook-kimi/mxk)
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("KIMI_MXK", "0") == "1"
MAX_TOKENS = int(os.environ.get("KIMI_MXK_MAX_TOKENS", "16"))
LIB_DIR = os.environ.get("KIMI_MXK_LIB", "/opt/gaudi-cookbook-kimi/mxk")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
MAX_ROUTES = 128  # kernel limit on T * top_k per call
W13_OP = "mxk_w13x"  # the kimi build registers its K = 7168 kernel (mxk_w13xt) under this op name
W2_OP = "mxk_w2x"
TAG = "[kimi-mxk]"
INF = 3.0e38
_installed = False


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_op_lazy.so")


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


def setup_env() -> None:
    """Register the kernel library with the graph compiler. Idempotent; a spawned worker inherits the result."""
    global ENABLED
    if not ENABLED:
        return
    tpc = tpc_lib()
    ver = lib_version(tpc)
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = False
        print(f"{TAG} KIMI_MXK=1 but {LIB_DIR} has no libmxk_tpc.so + libmxk_op_lazy.so: stock MoE op", flush=True)
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
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def env_ok() -> bool:
    """True if this process's graph compiler sees the kernel library (setup_env ran before the device opened)."""
    return os.environ.get("GC_KERNEL_PATH") == tpc_lib() and os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE") == "0"


def permute(x):
    """x [T, K] -> the kernels' per-512-k-chunk layout: xp[kc + 128 p + i] = x[kc + 4 i + p]. K % 512 == 0."""
    t, k = x.shape
    return x.reshape(t, k // 512, 128, 4).transpose(-1, -2).reshape(t, k)


def layer_reason(w13_shape, s13_shape, w2_shape, s2_shape, dtypes_ok: bool, has_bias: bool) -> str | None:
    """None if the kernels cover a layer with these packed shapes ([E, 2I, K/2], [E, 2I, K/32], [E, K, I/2],
    [E, K, I/32]); otherwise the reason it keeps the stock op."""
    if has_bias:
        return "layer has expert biases"
    if not dtypes_ok:
        return "weights or scales are not uint8"
    e, n13, kh = w13_shape
    k, inter = kh * 2, n13 // 2
    if e > 128:
        return f"{e} local experts > 128"
    if k % 512 or inter % 512:
        return f"hidden {k} or intermediate {inter} not a multiple of 512"
    want = ((e, n13, k // 32), (e, k, inter // 2), (e, k, inter // 32))
    if (tuple(s13_shape), tuple(w2_shape), tuple(s2_shape)) != want:
        return f"unexpected shapes {tuple(s13_shape)} {tuple(w2_shape)} {tuple(s2_shape)}"
    return None


def covers(tokens: int, topk: int) -> bool:
    return 1 <= tokens <= MAX_TOKENS and tokens * topk <= MAX_ROUTES


def install() -> None:
    """Wrap the HPU MXFP4 MoE method and op (called at the end of vllm_gaudi/ops/hpu_mxfp4.py)."""
    global _installed
    if not ENABLED or _installed:
        return
    _installed = True
    import torch
    from vllm.logger import init_logger
    from vllm_gaudi.extension import ops as gops
    from vllm_gaudi.ops import hpu_mxfp4

    log = init_logger("vllm.kimi_mxk")
    if not 1 <= MAX_TOKENS <= 16:
        log.warning("%s KIMI_MXK_MAX_TOKENS=%d is outside 1..16: stock MoE op", TAG, MAX_TOKENS)
        return
    if not env_ok():
        log.warning("%s GC_KERNEL_PATH is not the kernel library (setup_env did not run first): stock MoE op", TAG)
        return
    torch.ops.load_library(op_lib())
    if not (hasattr(torch.ops.custom_op, W13_OP) and hasattr(torch.ops.custom_op, W2_OP)):
        log.warning("%s %s does not register %s and %s: stock MoE op", TAG, op_lib(), W13_OP, W2_OP)
        return
    w13_op = getattr(torch.ops.custom_op, W13_OP)
    w2_op = getattr(torch.ops.custom_op, W2_OP)

    method = hpu_mxfp4.HPUGptOssMxfp4MoEMethod
    orig_pwal = method.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        op = layer.moe_op
        w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
        w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
        dtypes_ok = all(t.dtype == torch.uint8 for t in (w13, s13, w2, s2))
        why = layer_reason(w13.shape, s13.shape, w2.shape, s2.shape, dtypes_ok, op.has_bias)
        op._kimi_mxk = (w13, s13, w2, s2) if why is None else None
        if not getattr(method, "_kimi_mxk_logged", False):
            method._kimi_mxk_logged = True
            e, n13, kh = w13.shape
            if why is None:
                log.info(
                    "%s enabled: %s, E_local=%d K=%d I=%d, <= %d tokens, experts %d..%d",
                    TAG,
                    lib_version(),
                    e,
                    kh * 2,
                    n13 // 2,
                    MAX_TOKENS,
                    op.experts_min,
                    op.experts_max,
                )
            else:
                log.warning("%s not used for this layer (%s): stock MoE op", TAG, why)

    method.process_weights_after_loading = process_weights_after_loading

    op_cls = gops.VllmMixtureOfExpertsOpMXFP4
    orig_forward = op_cls.forward

    def forward(self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, activation="silu"):
        mx = getattr(self, "_kimi_mxk", None)
        t = hidden_states.shape[0]
        topk = expert_routing_table.shape[-1]
        if (
            mx is not None
            and covers(t, topk)
            and gops._as_activation_str(activation) == "silu"
            and hidden_states.dtype == torch.bfloat16
        ):
            w13, s13, w2, s2 = mx
            ids = expert_routing_table.reshape(-1)
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            tw = router_weights.reshape(-1)
            if tw.dtype != torch.bfloat16:
                tw = tw.to(torch.bfloat16)
            # plain silu(gate) * up: alpha 1, no clamp, beta 0, activation scale 1; rows [gate; up]
            act = w13_op(permute(hidden_states), ids, w13, s13, topk, self.experts_min, 1.0, INF, 0.0, 1.0, False)
            return w2_op(act, ids, tw, w2, s2, topk, self.experts_min)
        return orig_forward(self, hidden_states, expert_routing_table, router_weights, permuted_weights, activation)

    op_cls.forward = forward
    log.info(
        "%s installed: GC_KERNEL_PATH=%s MXK_TPC_FWD_LIB=%s PT_HPU_RECIPE_CACHE_CONFIG=%s",
        TAG,
        os.environ.get("GC_KERNEL_PATH"),
        os.environ.get("MXK_TPC_FWD_LIB"),
        os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG"),
    )
