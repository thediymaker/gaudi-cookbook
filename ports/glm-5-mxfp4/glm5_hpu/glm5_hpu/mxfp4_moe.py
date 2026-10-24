# SPDX-License-Identifier: Apache-2.0
"""MXFP4 MoE kernels for GLM-5.x on vllm-gaudi (lazy mode, HPU_MXFP4_NATIVE=1).

``patches/mxk_hooks.py`` wires two calls into vllm-gaudi:

* :func:`setup_env` at the top of ``vllm_gaudi/__init__.py``. Every vLLM process imports that platform plugin before
  it opens a device, and the graph compiler reads ``GC_KERNEL_PATH`` when the device opens.
* :func:`install` at the end of ``vllm_gaudi/ops/hpu_mxfp4.py``. It wraps ``HPUGptOssMxfp4MoEMethod`` (to keep a
  reference to each layer's packed weights) and ``VllmMixtureOfExpertsOpMXFP4.forward``.

Switches (both off by default; off means the stock ``mxfp4_fused_weights`` op runs every call):

``GLM_MXK=1``
    MoE calls of at most 16 tokens (decode, MTP verify rows included) run two TPC kernels that read the packed
    E2M1 + E8M0 expert weights in place and unpack them in registers::

        x -> permute per 512-k chunk -> mxk_w13x (w13 GEMV + SiLU-and-mul, fp32 accumulation)
          -> mxk_w2x (w2 GEMV + routing-weight sum per token)

    The stock op instead unpacks every touched expert to bf16 in memory before tiny GEMMs. The kernels accumulate
    in fp32, so greedy outputs are close to the stock op but not bit-identical.

``GLM_MXK_PREFILL=1`` (needs ``GLM_MXK=1``)
    Calls of 256 to 4096 tokens unpack each local expert to bf16 with the ``mxk_deq`` kernel, break the lazy graph
    and run the stock bf16 ``mixture_of_experts`` op. That is the unpack the stock MXFP4 op does internally,
    without its slow unpack kernel: the output is bit-identical to the stock op for these sizes. Costs about
    2.4 GB of transient bf16 per running layer at 32 local experts. 128-token calls are not bit-identical
    (hence the 256 minimum); 8192-token calls hit a graph-compiler error on one card and keep the stock op.

The kernels come from ``kernels/mxfp4-moe/build.sh --variant glm``, which ``install.sh`` runs into :data:`LIB_DIR`.
Anything unexpected keeps the stock op and is logged once: libraries missing, an op library without the GLM ops,
a layer with biases or another shape.
"""

import os

ENABLED = os.environ.get("GLM_MXK", "0") == "1"
PREFILL = os.environ.get("GLM_MXK_PREFILL", "0") == "1"
LIB_DIR = "/opt/gaudi-cookbook-glm5/mxk"
TPC_LIB = "libmxk_tpc.so"
OP_LIB = "libmxk_op_lazy.so"
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
# Decode-sized calls: decode and MTP verify rows at up to 4 sequences x 2 tokens, with headroom. The kernels take
# at most 16 tokens and 128 routes per call.
MAX_TOKENS = 16
PREFILL_MIN_TOKENS = 256
PREFILL_MAX_TOKENS = 4096
TAG = "[glm-mxk]"
_installed = False


def lib_version(path=None):
    """The 'mxk_tpc <version>' string build.sh embeds in the kernel library, or None."""
    try:
        with open(path or os.path.join(LIB_DIR, TPC_LIB), "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = data.find(b"mxk_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def setup_env():
    """Register the kernel library with the graph compiler. Runs before the first HPU op in every process."""
    global ENABLED
    if not ENABLED:
        return
    tpc = os.path.join(LIB_DIR, TPC_LIB)
    ver = lib_version(tpc)
    if ver is None or not os.path.isfile(os.path.join(LIB_DIR, OP_LIB)):
        ENABLED = False
        print(f"{TAG} GLM_MXK=1 but no kernel library in {LIB_DIR}: keeping the stock MoE op")
        return
    # One file: the library serves its own GUIDs and forwards every other GUID to the stock library, so
    # GC_KERNEL_PATH stays a single file and the on-disk recipe cache keeps working.
    rest = [p for p in os.environ.get("GC_KERNEL_PATH", STOCK).split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"{TAG} GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only")
    os.environ["GC_KERNEL_PATH"] = tpc
    # Recipe cache keys do not cover the kernel library: give each kernel build its own cache directory.
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
                print(f"{TAG} cannot create recipe cache dir {path}: {e}; keeping {cache_cfg}")
    # With the lazy accumulation thread on, a custom op whose input is a view of a tensor produced in the same lazy
    # graph fails intermittently ("Empty tensor optional ... Strided Params Has Value: 0").
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def install():
    """Wrap the HPU MXFP4 MoE method and op (called once, at the end of vllm_gaudi/ops/hpu_mxfp4.py)."""
    global _installed
    if not ENABLED or _installed:
        return
    _installed = True
    import habana_frameworks.torch as htorch
    import torch
    from vllm.logger import init_logger
    from vllm_gaudi.extension import ops as gops
    from vllm_gaudi.ops import hpu_mxfp4

    log = init_logger("vllm.glm5_hpu.mxk")
    torch.ops.load_library(os.path.join(LIB_DIR, OP_LIB))
    co = torch.ops.custom_op
    if not (hasattr(co, "mxk_w13x") and hasattr(co, "mxk_w2x")):
        log.warning("%s op library %s has no mxk_w13x / mxk_w2x: keeping the stock MoE op", TAG, OP_LIB)
        return
    prefill = PREFILL and hasattr(co, "mxk_deq")
    if PREFILL and not prefill:
        log.warning("%s GLM_MXK_PREFILL=1 but the op library has no mxk_deq: prefill keeps the stock op", TAG)
    inf = 3.0e38

    method = hpu_mxfp4.HPUGptOssMxfp4MoEMethod
    orig_pwal = method.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        op = layer.moe_op
        w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
        w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
        e, n13, kh = w13.shape
        k, inter = kh * 2, n13 // 2
        ok = (
            not op.has_bias
            and w13.dtype == torch.uint8
            and w2.dtype == torch.uint8
            and e <= 128
            and k % 512 == 0
            and inter % 512 == 0
            and k <= 6144  # mxk_w13x stages the x row in 12 KB of vector local memory
            and tuple(w2.shape) == (e, k, inter // 2)
            and tuple(s13.shape) == (e, n13, k // 32)
            and tuple(s2.shape) == (e, k, inter // 32)
        )
        op._glm_mxk = (w13, s13, w2, s2) if ok else None
        if not getattr(method, "_glm_mxk_logged", False):
            method._glm_mxk_logged = True
            log.info(
                "%s %s (E_local=%d K=%d I=%d, <= %d tokens, experts %d..%d; prefill %s)",
                TAG,
                "enabled" if ok else "not supported for this layer, stock op kept",
                e,
                k,
                inter,
                MAX_TOKENS,
                op.experts_min,
                op.experts_max,
                f"mxk_deq for {PREFILL_MIN_TOKENS}..{PREFILL_MAX_TOKENS} tokens" if prefill and ok else "stock",
            )

    method.process_weights_after_loading = process_weights_after_loading

    op_cls = gops.VllmMixtureOfExpertsOpMXFP4
    orig_fwd = op_cls.forward

    def forward(self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, activation="silu"):
        mx = getattr(self, "_glm_mxk", None)
        t, k = hidden_states.shape
        topk = expert_routing_table.shape[-1]
        silu_bf16 = gops._as_activation_str(activation) == "silu" and hidden_states.dtype == torch.bfloat16
        if mx is not None and silu_bf16 and t <= MAX_TOKENS and t * topk <= 128:
            w13, s13, w2, s2 = mx
            ids = expert_routing_table.reshape(-1)
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            tw = router_weights.reshape(-1)
            if tw.dtype != torch.bfloat16:
                tw = tw.to(torch.bfloat16)
            # lane i of sub-vector p of 512-chunk b holds k = 512 b + 4 i + p (the kernel's nibble-plane order)
            xp = hidden_states.reshape(t, k // 512, 128, 4).transpose(-1, -2).reshape(t, k)
            # plain silu(gate) * up: alpha 1, no clamp, beta 0, activation scale 1; rows are [gate; up]
            ap = torch.ops.custom_op.mxk_w13x(xp, ids, w13, s13, topk, self.experts_min, 1.0, inf, 0.0, 1.0, False)
            return torch.ops.custom_op.mxk_w2x(ap, ids, tw, w2, s2, topk, self.experts_min)
        if prefill and mx is not None and silu_bf16 and PREFILL_MIN_TOKENS <= t <= PREFILL_MAX_TOKENS:
            # One bf16 tensor per expert: the bf16 op rejects views of a stacked unpack. The graph break is needed
            # because the bf16 MoE op does not compile on weights produced inside the same lazy graph.
            w13, s13, w2, s2 = mx
            l13 = [torch.ops.custom_op.mxk_deq(w13[i], s13[i], 0) for i in range(w13.shape[0])]
            l2 = [torch.ops.custom_op.mxk_deq(w2[i], s2[i], 0) for i in range(w2.shape[0])]
            htorch.core.mark_step()
            ids = expert_routing_table
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            return torch.ops.hpu.mixture_of_experts(
                hidden_states=hidden_states,
                expert_routing_table=ids,
                router_weights=router_weights,
                w12=l13,
                w3=l2,
                permuted_weights=True,
                activation="silu",
                experts_min=self.experts_min,
                experts_max=self.experts_max,
                **self._get_extra_kwargs(t),
            )
        return orig_fwd(self, hidden_states, expert_routing_table, router_weights, permuted_weights, activation)

    op_cls.forward = forward
    log.info(
        "%s installed: %s, prefill %s, GC_KERNEL_PATH=%s MXK_TPC_FWD_LIB=%s PT_HPU_RECIPE_CACHE_CONFIG=%s",
        TAG,
        lib_version(),
        "on" if prefill else "stock",
        os.environ.get("GC_KERNEL_PATH"),
        os.environ.get("MXK_TPC_FWD_LIB"),
        os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG"),
    )
