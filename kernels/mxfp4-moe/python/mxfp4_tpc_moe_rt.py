# SPDX-License-Identifier: Apache-2.0
"""Runtime of the MXFP4 MoE decode TPC kernels for vllm-gaudi (gpt-oss, lazy mode, HPU_MXFP4_NATIVE=1).

``install_mxfp4_tpc_moe.py`` copies this file to ``vllm_gaudi/ops/mxfp4_tpc_moe_rt.py`` and adds two hooks:

* top of ``vllm_gaudi/__init__.py``: :func:`setup_env`, which runs in every vLLM process before the device opens;
* end of ``vllm_gaudi/ops/hpu_mxfp4.py``: :func:`install`, which wraps ``HPUGptOssMxfp4MoEMethod`` and
  ``VllmMixtureOfExpertsOpMXFP4``.

Off unless ``MXFP4_TPC_MOE=1``. With the flag off both hooks return at once and nothing changes. With the flag on, a
``VllmMixtureOfExpertsOpMXFP4`` call of a bias / SwiGLU-OAI layer with at most ``MXFP4_TPC_MOE_MAX_TOKENS`` tokens
(default and maximum 32) runs two custom TPC kernels that read the packed E2M1 + E8M0 expert weights in place:

    x -> zero-pad K to a multiple of 512, permute per 512-k chunk
      -> mxk_w13xib (w13 GEMV + bias + SwiGLU-OAI, fp32 accumulation, bf16 out)
      -> mxk_w2xb   (w2 GEMV + bias, routing-weight sum per token, bf16 out)

Prefill and any call with more tokens keep the stock ``bias_mxfp4_fused_weights`` op. The expert weights are not
copied; only the biases are re-laid out once at load time (f32, about 0.8 MB per layer and rank for gpt-oss-120b at
32 local experts).

Anything unexpected keeps the stock op: libraries missing, a layer without biases, other dtypes or shapes, an
exception while preparing a layer. Each of these is logged once.

Environment:
    MXFP4_TPC_MOE=1                 turn the kernels on (default 0)
    MXFP4_TPC_MOE_MAX_TOKENS=N      per-call token cap, at most 32 (default 32)
    MXFP4_TPC_MOE_LIB=DIR           directory with libmxk_tpc.so and libmxk_op_lazy.so (default /opt/mxfp4-moe/lib)
"""

import os

ENABLED = os.environ.get("MXFP4_TPC_MOE", "0") == "1"
MAX_TOKENS = max(1, min(32, int(os.environ.get("MXFP4_TPC_MOE_MAX_TOKENS", "32"))))
LIB_DIR = os.environ.get("MXFP4_TPC_MOE_LIB", "/opt/mxfp4-moe/lib")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
TPC_LIB = "libmxk_tpc.so"
OP_LIB = "libmxk_op_lazy.so"
LOG_TAG = "[mxfp4-tpc-moe]"
_installed = False


# ---------------------------------------------------------------------------------------------- layout helpers
def round512(n):
    return (n + 511) // 512 * 512


def perm_index(n):
    """Index map of the per-512 chunk permutation: position c of a permuted row holds natural index perm(c).

    Lane i of sub-vector p of chunk b holds k = 512 b + 4 i + p, so that the four nibble planes of 128 packed u16
    lanes line up with four activation vectors. ``n`` must be a multiple of 512.
    """
    import torch

    c = torch.arange(n)
    return (c // 512) * 512 + 4 * (c % 128) + (c % 512) // 128


def permute_x(x, kp):
    """bf16 [T, K] -> [T, Kp]: zero-pad K to ``kp`` (a multiple of 512) and apply the per-512 chunk permutation."""
    import torch

    t, k = x.shape
    if kp != k:
        x = torch.nn.functional.pad(x, (0, kp - k))
    return x.reshape(t, kp // 512, 128, 4).transpose(-1, -2).reshape(t, kp)


def bias13(b13, inter):
    """w13 bias [E, 2I] (interleaved gate/up rows) -> f32 [E, 4, Ip/2] in the kernel's column order.

    Column c of the permuted activation is intermediate index perm(c). Planes: gate bias of the even / odd columns,
    up bias of the even / odd columns; zeros past I.
    """
    import torch

    ip = round512(inter)
    i = perm_index(ip)
    valid = (i < inter).float()
    i = i.clamp(max=inter - 1)
    b = b13.detach().to("cpu", torch.float32)
    g = b[:, 0::2][:, i] * valid
    u = b[:, 1::2][:, i] * valid
    return torch.stack([g[:, 0::2], g[:, 1::2], u[:, 0::2], u[:, 1::2]], 1).contiguous()


def bias2(b2):
    """w2 bias [E, K] -> f32 [E, 2, K/2]: even rows, odd rows (the A / B lane order of the kernel's packed sums)."""
    import torch

    b = b2.detach().to("cpu", torch.float32)
    return torch.stack([b[:, 0::2], b[:, 1::2]], 1).contiguous()


def moe_decode(x, ids, tw, w13, s13, b13k, w2, s2, b2k, topk, e_min, alpha, limit):
    """One MoE call on the kernels. x bf16 [T, K], ids i32 [T, topk] (global expert ids), tw bf16 [T, topk].

    w13 u8 [E, 2I, K/2], s13 u8 [E, 2I, K/32], w2 u8 [E, K, I/2], s2 u8 [E, K, I/32] are the served tensors of the
    local experts e_min .. e_min + E - 1; b13k / b2k come from :func:`bias13` / :func:`bias2`. Returns bf16 [T, K].
    Routes to experts on other ranks contribute zero, like the stock op.
    """
    import torch

    k = x.shape[1]
    xp = permute_x(x, round512(k))
    idf = ids.reshape(-1)
    ap = torch.ops.custom_op.mxk_w13xib(xp, idf, w13, s13, b13k, topk, e_min, float(alpha), float(limit), 1.0, 1.0)
    return torch.ops.custom_op.mxk_w2xb(ap, idf, tw.reshape(-1), w2, s2, b2k, topk, e_min)


def supported(e, k, inter, topk, tokens):
    """Shape limits of the compiled kernels (mirrors the checks in mxk_tpc_lib.cpp)."""
    return (
        0 < e <= 128
        and k % 32 == 0
        and inter % 32 == 0
        and 0 < k <= 3072
        and 0 < inter <= 3072
        and 0 < tokens <= 32
        and tokens * topk <= 128
    )


# ---------------------------------------------------------------------------------------------- libraries
def lib_version(path):
    """'mxk_tpc <version>' string embedded in a kernel library, None if absent."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = data.find(b"mxk_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def libs_present(lib_dir=None):
    d = lib_dir or LIB_DIR
    return all(os.path.isfile(os.path.join(d, n)) for n in (TPC_LIB, OP_LIB))


def load_ops(lib_dir=None):
    """Load the torch op library; True if both ops are registered."""
    import torch

    torch.ops.load_library(os.path.join(lib_dir or LIB_DIR, OP_LIB))
    co = torch.ops.custom_op
    return hasattr(co, "mxk_w13xib") and hasattr(co, "mxk_w2xb")


def setup_env():
    """Register the kernel library with the graph compiler. Must run before the first HPU op of the process."""
    global ENABLED
    if not ENABLED:
        return
    tpc = os.path.join(LIB_DIR, TPC_LIB)
    ver = lib_version(tpc)
    if not libs_present() or not ver:
        ENABLED = False
        print(f"{LOG_TAG} MXFP4_TPC_MOE=1 but no {TPC_LIB} / {OP_LIB} in {LIB_DIR}: staying on the stock MoE op")
        return
    # One file in GC_KERNEL_PATH (required with the recipe cache on); the library forwards stock GUIDs itself.
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"{LOG_TAG} GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only")
    os.environ["GC_KERNEL_PATH"] = tpc
    # Recipe cache keys do not cover the kernel library: keep one cache directory per kernel build.
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
                print(f"{LOG_TAG} cannot create recipe cache dir {path}: {e}; keeping {rc}")
    # Required: with the lazy accumulation thread on, a custom op whose input is a view of a tensor produced in the
    # same lazy graph fails intermittently. With the kernels off this setting gave bit-identical output.
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


# ---------------------------------------------------------------------------------------------- vLLM hooks
def _prepare_layer(torch, layer, op):
    """Kernel tensors of one layer, or (None, reason)."""
    w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
    w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
    b13 = getattr(layer, "w13_bias", None)
    b2 = getattr(layer, "w2_bias", None)
    e, n13, kh = w13.shape
    k, inter = kh * 2, n13 // 2
    if not op.has_bias or b13 is None or b2 is None:
        return None, "no expert bias (not a gpt-oss style layer)"
    b13, b2 = b13.data, b2.data
    ok = (
        w13.dtype == torch.uint8
        and w2.dtype == torch.uint8
        and b13.dtype == torch.bfloat16
        and b2.dtype == torch.bfloat16
        and supported(e, k, inter, 1, 1)
        and tuple(w2.shape) == (e, k, inter // 2)
        and tuple(s13.shape) == (e, n13, k // 32)
        and tuple(s2.shape) == (e, k, inter // 32)
        and tuple(b13.shape) == (e, n13)
        and tuple(b2.shape) == (e, k)
        and all(t.is_contiguous() for t in (w13, s13, w2, s2, b13, b2))
    )
    if not ok:
        return None, f"unsupported shapes or dtypes E={e} K={k} I={inter}"
    b13k = bias13(b13, inter).to(w13.device)
    b2k = bias2(b2).to(w13.device)
    return (w13, s13, b13k, w2, s2, b2k, e, k, inter), ""


def install():
    """Wrap the stock gpt-oss MXFP4 MoE method and op. Called from the hook at the end of ops/hpu_mxfp4.py."""
    global _installed
    if not ENABLED or _installed or not libs_present():
        return
    _installed = True
    import torch
    from vllm.logger import init_logger

    from vllm_gaudi.extension import ops as gops
    from vllm_gaudi.ops import hpu_mxfp4

    log = init_logger("vllm.mxfp4_tpc_moe")  # under vllm.* so INFO lines are kept
    if not load_ops():
        log.warning("MXFP4 TPC MoE NOT installed (stock MoE op kept): %s has no mxk_w13xib / mxk_w2xb", OP_LIB)
        return
    ver = lib_version(os.path.join(LIB_DIR, TPC_LIB))

    method_cls = hpu_mxfp4.HPUGptOssMxfp4MoEMethod
    orig_pwal = method_cls.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        op = layer.moe_op
        try:
            mx, why = _prepare_layer(torch, layer, op)
        except Exception as e:  # any surprise keeps the stock op for this layer
            mx, why = None, f"{type(e).__name__}: {e}"
        op._mxk = mx
        if not getattr(method_cls, "_mxk_logged", False):
            method_cls._mxk_logged = True
            if mx is not None:
                log.info(
                    "MXFP4 TPC MoE enabled (E_local=%d K=%d I=%d, <= %d tokens, experts %d..%d, alpha %s limit %s)",
                    mx[6],
                    mx[7],
                    mx[8],
                    MAX_TOKENS,
                    op.experts_min,
                    op.experts_max,
                    op.alpha,
                    op.limit,
                )
            else:
                log.info("MXFP4 TPC MoE NOT used for this layer, stock op kept: %s", why)

    method_cls.process_weights_after_loading = process_weights_after_loading

    op_cls = gops.VllmMixtureOfExpertsOpMXFP4
    orig_fwd = op_cls.forward

    def forward(self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, activation="silu"):
        mx = getattr(self, "_mxk", None)
        if mx is not None and hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2:
            tokens = hidden_states.shape[0]
            topk = expert_routing_table.shape[-1]
            if 0 < tokens <= MAX_TOKENS and tokens * topk <= 128:
                w13, s13, b13k, w2, s2, b2k = mx[:6]
                ids = expert_routing_table
                if ids.dtype != torch.int32:
                    ids = ids.to(torch.int32)
                tw = router_weights
                if tw.dtype != torch.bfloat16:
                    tw = tw.to(torch.bfloat16)
                return moe_decode(
                    hidden_states, ids, tw, w13, s13, b13k, w2, s2, b2k, topk, self.experts_min, self.alpha, self.limit
                )
        return orig_fwd(self, hidden_states, expert_routing_table, router_weights, permuted_weights, activation)

    op_cls.forward = forward
    log.info(
        "MXFP4 TPC MoE installed: %s, <= %d tokens, GC_KERNEL_PATH=%s MXK_TPC_FWD_LIB=%s "
        "PT_HPU_LAZY_ACC_PAR_MODE=%s PT_HPU_RECIPE_CACHE_CONFIG=%s",
        ver,
        MAX_TOKENS,
        os.environ.get("GC_KERNEL_PATH"),
        os.environ.get("MXK_TPC_FWD_LIB"),
        os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE"),
        os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG"),
    )
