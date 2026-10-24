#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU-OAI routed experts on compressed-tensors FP8 weights (MiniMax-M3, vllm-gaudi v0.26.0).

The FP8 checkpoint from convert/repack.py loads through the stock HPUCompressedTensorsW8A8Fp8MoEMethod. Its
apply_monolithic hands layer.activation to torch.ops.hpu.mixture_of_experts, whose activation string covers
silu and gelu only. MiniMax-M3's experts use swigluoai_uninterleave (gpt-oss's activation: clamp to +-7,
alpha-sigmoid with alpha 1.702, (up + 1)), so unpatched the experts would fail or run the wrong activation.
This patch routes SwiGLU-OAI layers to one of two paths; every other activation still calls the stock op.

VLLM_M3_FP8_MOE=native (the recipe)
    torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights, the one Habana FP8 MoE overload with gpt_swiglu.
    Its alpha and limit defaults are M3's values. The FP8 weights go to the matrix engine directly:
    - the kernel reads gate and up rows interleaved (g0, u0, g1, u1, ...); the rows and their scales are
      interleaved once at load, on the uint8 view so no FP8 arithmetic is involved. Concatenated rows give
      fluent garbage, not an error.
    - activations are quantized to FP8 per token on the fly.
    - the down projection's input uses a fixed scale of 56/240: SwiGLU-OAI's clamp bounds it by
      limit * (limit + 1) = 56, so it never saturates.
    - the bias overload needs biases, so zero biases are registered.
    - VLLM_MOE_CHUNK and VLLM_MOE_TOKEN_BOUNDARY pass the chunk-size hints the stock FP8 ops pass.
VLLM_M3_FP8_MOE=dequant (reference, slow)
    The routed experts are decoded FP8 x 2^k to bf16 (exact: the scales are powers of two) and run through
    SwiGLU-OAI in PyTorch with bf16 activations, in 32-expert chunks for prefill and only the routed experts
    for small decode batches (vllm-gaudi's VLLM_MINIMAX_M3_MOE_* gather settings). Numerically this is the
    MXFP4 checkpoint run in bf16, so it is the reference the native path is compared against.

The startup log prints "[minimax-m3:fp8-moe] SwiGLU-OAI FP8 experts: mode=..." once per worker.

Usage: swigluoai_fp8_moe.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[minimax-m3:fp8-moe]"
TARGET = ("vllm_gaudi", "ops/hpu_compressed_tensors.py")

CALL_OLD = """        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=_normalize_moe_activation(layer.activation),
        )
"""

CALL_NEW = f"""        # {MARKER} SwiGLU-OAI experts bypass the stock op, whose activation string has no SwiGLU-OAI.
        output = _m3_fp8_moe_call(self, layer, x, topk_ids, topk_weights,
                                  _normalize_moe_activation(layer.activation))
"""

# Names the appended block uses from the module it is appended to.
REQUIRED = (
    "class HPUCompressedTensorsW8A8Fp8MoEMethod(",
    "import vllm_gaudi.extension.ops as hpu_ops\n",
    "from vllm_gaudi import envs\n",
    "from compressed_tensors.quantization import (QuantizationArgs, QuantizationStrategy)\n",
    "logger = init_logger(",
)

TAIL = f'''

# {MARKER} ---------------------------------------------------------------------------------------------
import os as _m3_os  # noqa: E402

_M3_SWIGLU_OAI = "swigluoai_uninterleave"
_M3_MODE = _m3_os.environ.get("VLLM_M3_FP8_MOE", "native").strip().lower()
if _M3_MODE not in ("native", "dequant"):
    raise ValueError(f"VLLM_M3_FP8_MOE={{_M3_MODE!r}}: expected native or dequant")
_M3_EXPERT_CHUNK = 32
_m3_logged = set()


def _m3_once(key, msg):
    if key not in _m3_logged:
        _m3_logged.add(key)
        # INFO lines from this module do not always reach the server log; print as well.
        logger.info(msg)
        print(msg, flush=True)


def _m3_dq(w, s, dtype):
    """FP8 [..., R, C] x f32 [..., R, 1] to dtype. Exact: power-of-two scales on MXFP4 values."""
    return (w.to(torch.float32) * s).to(dtype)


def _m3_gate_weights(x, topk_ids, topk_weights, e_local, experts_min):
    """Dense [tokens, local experts] routing weights; experts on other EP ranks get 0."""
    local_ids = topk_ids - experts_min
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(x.shape[0], e_local, dtype=torch.float32)
    gate_w.scatter_add_(1, safe_ids, torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).float())
    return gate_w


def _m3_swiglu(h, d, alpha, beta, limit, out_dtype):
    g, u = h[..., :d].float(), h[..., d:].float()
    if limit is not None:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    return (g * torch.sigmoid(alpha * g) * (u + beta)).to(out_dtype)


def _m3_dense(layer, x, gate_w, swiglu):
    """Every token through every local expert, one dequantized 32-expert chunk at a time. The expert loop is
    outer, token tiles inner, and partial results are concatenated and summed functionally: in-place slice
    updates in this loop make lazy-mode graph building take minutes."""
    w13, w2, s13, s2 = layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale
    e_local, d, hidden, tokens = w13.shape[0], w13.shape[1] // 2, x.shape[-1], x.shape[0]
    tile = envs.VLLM_MINIMAX_M3_MOE_TOKEN_TILE
    tile = tokens if tile <= 0 or tile >= tokens else tile
    acc = None
    for start in range(0, e_local, _M3_EXPERT_CHUNK):
        end = min(start + _M3_EXPERT_CHUNK, e_local)
        w13c = _m3_dq(w13[start:end], s13[start:end], x.dtype).transpose(1, 2)
        w2c = _m3_dq(w2[start:end], s2[start:end], x.dtype).transpose(1, 2)
        tiles = []
        for t0 in range(0, tokens, tile):
            xt = x[t0 : t0 + tile]
            xe = xt.unsqueeze(0).expand(end - start, xt.shape[0], hidden)
            y = torch.bmm(_m3_swiglu(torch.bmm(xe, w13c), d, *swiglu, x.dtype), w2c)
            tiles.append((y.float() * gate_w[t0 : t0 + tile, start:end].t().unsqueeze(-1)).sum(0).to(x.dtype))
        contrib = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        acc = contrib if acc is None else acc + contrib
    return acc


def _m3_gather(layer, x, gate_w, k, swiglu):
    """Only the G = min(E_local, tokens * top_k) routed experts are gathered and decoded; same accumulation
    order as the dense pass (fp32 within a 32-expert chunk, bf16 across chunks)."""
    w13, w2, s13, s2 = layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale
    e_local, d, hidden, tokens = w13.shape[0], w13.shape[1] // 2, x.shape[-1], x.shape[0]
    g = min(e_local, tokens * k)
    hit = (gate_w > 0).float().sum(0)
    ids, _ = torch.sort(torch.topk(hit, g, sorted=False).indices)
    w13g = _m3_dq(w13.index_select(0, ids), s13.index_select(0, ids), x.dtype)
    w2g = _m3_dq(w2.index_select(0, ids), s2.index_select(0, ids), x.dtype)
    xe = x.unsqueeze(0).expand(g, tokens, hidden)
    y = torch.bmm(_m3_swiglu(torch.bmm(xe, w13g.transpose(1, 2)), d, *swiglu, x.dtype), w2g.transpose(1, 2))
    cont = y.float() * gate_w.index_select(1, ids).t().unsqueeze(-1)
    chunk_of = torch.div(ids, _M3_EXPERT_CHUNK, rounding_mode="floor")
    acc = x.new_zeros(tokens, hidden)
    for c in range((e_local + _M3_EXPERT_CHUNK - 1) // _M3_EXPERT_CHUNK):
        acc = acc + (cont * (chunk_of == c).to(cont.dtype)[:, None, None]).sum(0).to(x.dtype)
    return acc


def _m3_dequant_moe(layer, x, topk_ids, topk_weights, swiglu):
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local, tokens, k = layer.w13_weight.shape[0], x.shape[0], topk_ids.shape[-1]
    gate_w = _m3_gate_weights(x, topk_ids, topk_weights, e_local, experts_min)
    if envs.VLLM_MINIMAX_M3_MOE_DECODE_GATHER and tokens <= envs.VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS \\
            and tokens * k < e_local:
        return _m3_gather(layer, x, gate_w, k, swiglu)
    return _m3_dense(layer, x, gate_w, swiglu)


def _m3_native_moe(layer, x, topk_ids, topk_weights, alpha, limit):
    n = layer._m3_native
    x_fp8, x_scale = hpu_ops.dynamic_quant(x, single_scale=False)  # per-token scales
    extra = layer.moe_op._get_extra_kwargs(x.shape[0]) if getattr(layer, "moe_op", None) is not None else {{}}
    return torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights(
        hidden_states=x_fp8,
        expert_routing_table=topk_ids,
        router_weights=topk_weights,
        w12=n["w12"],
        w3=n["w3"],
        w12_bias=n["b12"],
        w3_bias=n["b3"],
        d_scale_hidden_states=x_scale,
        d_scale_intermediate_hidden_states=n["s_mid"],
        d_scale_w12=n["s12"],
        d_scale_w3=n["s3"],
        permuted_weights=True,
        experts_min=n["min"],
        experts_max=n["max"],
        alpha=alpha,
        limit=limit,
        **extra,
    )


def _m3_fp8_moe_call(method, layer, x, topk_ids, topk_weights, activation):
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(x.dtype)
    if activation != _M3_SWIGLU_OAI:
        return layer.moe_op(x, topk_ids, topk_weights, permuted_weights=True, activation=activation)
    alpha, beta, limit = method._m3_swiglu
    if getattr(layer, "_m3_native", None) is not None:
        return _m3_native_moe(layer, x, topk_ids, topk_weights, alpha, limit)
    return _m3_dequant_moe(layer, x, topk_ids, topk_weights, method._m3_swiglu)


_m3_orig_init = HPUCompressedTensorsW8A8Fp8MoEMethod.__init__


def _m3_init(self, *args, **kwargs):
    _m3_orig_init(self, *args, **kwargs)
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    tc = None
    if cfg is not None and cfg.model_config is not None:
        tc = getattr(cfg.model_config, "hf_text_config", None) or cfg.model_config.hf_config
    limit = getattr(tc, "swiglu_limit", 7.0)
    self._m3_swiglu = (float(getattr(tc, "swiglu_alpha", 1.702)), float(getattr(tc, "swiglu_beta", 1.0)),
                       None if limit is None else float(limit))


_m3_orig_pwal = HPUCompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading


def _m3_build_native(method, layer):
    alpha, beta, limit = method._m3_swiglu
    if limit is None or beta != 1.0:
        raise ValueError(f"VLLM_M3_FP8_MOE=native needs swiglu_limit set and beta 1.0, got {{limit}}, {{beta}}")
    w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
    E, two_i, H = w13.shape
    # rows [g0..gI-1, u0..uI-1] -> [g0, u0, g1, u1, ...], scales alike
    w13 = w13.view(torch.uint8).view(E, 2, two_i // 2, H).transpose(1, 2).reshape(E, two_i, H)
    layer.w13_weight.data = w13.contiguous().view(torch.float8_e4m3fn)
    layer.w13_weight_scale.data = s13.view(E, 2, two_i // 2, 1).transpose(1, 2).reshape(E, two_i, 1).contiguous()
    if getattr(layer, "moe_op", None) is not None:
        # Re-point the stock op's per-expert views so the old storage is freed.
        hpu_ops.fp8_channel_moe_prepare_weights(layer)
        layer.moe_op._cache_weight_lists()
    w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
    w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
    dev = w13.device
    shift = int(layer.moe_config.ep_rank * layer.local_num_experts)
    mid = torch.tensor(limit * (limit + beta) / hpu_ops.FP8_MAX, dtype=torch.float32, device=dev)
    layer._m3_native = {{
        "w12": tuple(w13[e] for e in range(E)),
        "w3": tuple(w2[e] for e in range(E)),
        "b12": tuple(torch.zeros(two_i, dtype=torch.bfloat16, device=dev) for _ in range(E)),
        "b3": tuple(torch.zeros(H, dtype=torch.bfloat16, device=dev) for _ in range(E)),
        "s12": tuple(s13[e].reshape(-1) for e in range(E)),
        "s3": tuple(s2[e].reshape(-1) for e in range(E)),
        "s_mid": [mid.clone() for _ in range(E)],
        "min": shift,
        "max": shift + E - 1,
    }}


def _m3_pwal(self, layer):
    _m3_orig_pwal(self, layer)
    if _normalize_moe_activation(layer.activation) != _M3_SWIGLU_OAI:
        return
    if self.block_quant or self.static_input_scales or self.weight_quant.strategy != QuantizationStrategy.CHANNEL:
        raise ValueError("{MARKER} SwiGLU-OAI FP8 experts need per-channel weights with dynamic activations")
    if _M3_MODE == "native":
        _m3_build_native(self, layer)
    _m3_once("mode", f"{MARKER} SwiGLU-OAI FP8 experts: mode={{_M3_MODE}} (alpha, beta, limit {{self._m3_swiglu}}); "
             f"w13 {{tuple(layer.w13_weight.shape)}} w2 {{tuple(layer.w2_weight.shape)}}")


HPUCompressedTensorsW8A8Fp8MoEMethod.__init__ = _m3_init
HPUCompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading = _m3_pwal
'''


class Drift(Exception):
    pass


def patch(src: str) -> str:
    n = src.count(CALL_OLD)
    if n != 1:
        raise Drift(f"anchor 'W8A8 FP8 MoE moe_op call' found {n} times, expected 1")
    for needle in REQUIRED:
        if src.count(needle) != 1:
            raise Drift(f"anchor {needle.strip()!r} found {src.count(needle)} times, expected 1")
    return src.replace(CALL_OLD, CALL_NEW, 1) + TAIL


def locate(root: str | None) -> Path | None:
    pkg, rel = TARGET
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        if (Path(loc) / rel).is_file():
            return Path(loc) / rel
    return None


def run(check: bool, root: str | None) -> int:
    path = locate(root)
    if path is None:
        print(f"{MARKER} {'/'.join(TARGET)} not found", file=sys.stderr)
        return 2
    src = path.read_text()
    if MARKER in src:
        print(f"{MARKER} applied")
        return 0
    try:
        new = patch(src)
        compile(new, str(path), "exec", dont_inherit=True)
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    tmp = path.with_name(path.name + ".port.tmp")
    tmp.write_text(new)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not)")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
