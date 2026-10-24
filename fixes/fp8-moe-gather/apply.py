#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fp8-moe-gather: gathered-expert FP8 MoE combine at low token counts (backport of vllm-gaudi #1731).

At low token counts a fused MoE decode reads every local expert's weights, although each token
routes to only top_k of them. vllm-gaudi #1731 (in v0.30.0) gathers just the
g = min(local_experts, tokens * top_k) routed experts and runs a pure-PyTorch silu / FP8 combine
over them instead of the Habana mixture_of_experts op, which speeds up single-stream decode on
fine-grained MoE models. This backports the merged version to v0.26.0: the combine kernel and the
verify helpers are appended to vllm_gaudi/ops/hpu_fp8.py, and HPUFp8MoEMethod.apply_monolithic
routes its expert call through the same gate as v0.30.0.

Opt-in and off at runtime unless VLLM_HPU_MOE_GATHER=1. The environment variables match v0.30.0:

  VLLM_HPU_MOE_GATHER          1: use the gather path when it wins (default 0, the stock op)
  VLLM_HPU_MOE_GATHER_RATIO    use it while tokens * top_k <= local_experts * ratio (default 0.4)
  VLLM_HPU_MOE_GATHER_VERIFY   1: run both paths and log FP8-ULP differences (slow; checks only)

Only dynamic-activation, block-quantized FP8 checkpoints with VLLM_HPU_FORCE_CHANNEL_FP8 (the
default) and silu experts take the gather path; everything else keeps the stock op. On v0.30.0
the upstream code is present and the script reports "not needed".

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "fp8-moe-gather"
MARKER = f"[gaudi-fix:{FIX_ID}]"

CALL_OLD = """        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=_normalize_moe_activation(layer.activation),
        )
        return output.view(*(output.size(0), *input_shape[1:]))
"""
CALL_NEW = f"""        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        # {MARKER} gathered-expert combine at low token counts, else the stock op.
        output = _gaudi_fix_moe_gather_apply(self, layer, x, topk_ids, topk_weights,
                                             _normalize_moe_activation(layer.activation))
        return output.view(*(output.size(0), *input_shape[1:]))
"""

# Appended at the end of hpu_fp8.py. The combine kernel and the verify helpers are the
# vllm-gaudi v0.30.0 code (ops/hpu_moe_combine.py and ops/hpu_fp8.py, Apache-2.0).
BLOCK = '''

# ---- {MARKER} ------------------------------------------------------------------------------
# Backport of the gathered-expert FP8 MoE combine from vllm-gaudi v0.30.0 (vllm-gaudi #1731).
import os as _gaudi_fix_os  # noqa: E402

from vllm.distributed import get_ep_group  # noqa: E402
from vllm.logger import init_logger as _gaudi_fix_init_logger  # noqa: E402

logger = _gaudi_fix_init_logger(__name__)

_HPU_MOE_GATHER = _gaudi_fix_os.environ.get("VLLM_HPU_MOE_GATHER", "0").lower() in ("1", "true")
_HPU_MOE_GATHER_VERIFY = (_gaudi_fix_os.environ.get("VLLM_HPU_MOE_GATHER_VERIFY", "0").lower() in ("1", "true")
                          and _HPU_MOE_GATHER)
_HPU_MOE_GATHER_VERIFY_MAX_ULP = 2
_HPU_MOE_GATHER_VERIFY_REL_ERR = 0.05
_HPU_MOE_GATHER_RATIO = float(_gaudi_fix_os.environ.get("VLLM_HPU_MOE_GATHER_RATIO", "0.4"))
if _HPU_MOE_GATHER:
    logger.info("MoE gather combine enabled (ratio = %s, verify = %s)", _HPU_MOE_GATHER_RATIO,
                _HPU_MOE_GATHER_VERIFY)


def _dynamic_quant(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # import lazily to avoid pulling heavy deps at module import
    from vllm_gaudi.extension.ops import dynamic_quant
    return dynamic_quant(data)


def gather_silu_fp8_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the rank-local FP8 silu MoE partial for routed experts only.

    topk_ids [T,K] int64 (global expert ids), topk_weights [T,K] bf16.
    Mirrors the Habana op's data flow: fp8-quantize x, per-token weighted sum
    over the routed experts, each expert computed as silu(w13 x) w2.
    """
    tokens, hidden_size = x.shape
    num_topk = topk_ids.shape[-1]
    w13 = layer.w13_weight  # [E, 2I, H] fp8
    w2 = layer.w2_weight  # [E, H, I] fp8
    scale13 = layer.w13_weight_scale_inv  # [E, 2I]
    scale2 = layer.w2_weight_scale_inv  # [E, H]
    local_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2

    # ---- FP8-quantize x per token (match the op's input quantization) ----
    # x_scale is kept as [T,1] (NOT squeezed): fp8_gemm_v2 requires the row
    # scale in 2-D form; squeezing only works at T=1 and silently mis-scales at
    # larger T.
    x_fp8, x_scale = _dynamic_quant(x)  # x_fp8 [T,H], x_scale [T,1] f32

    # ---- per-token expert combine weights (scatter over E) ----
    # topk_ids are GLOBAL expert ids; remap to this rank's local ids and mask
    # experts owned by other EP ranks (they contribute +0.0 to this rank's
    # partial, which is then reduce-scattered by the runner). At TP=1
    # ep_rank=0 -> experts_min=0, identical to the unremapped path.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    local_ids = topk_ids - experts_min  # [T,K]
    in_range = (local_ids >= 0) & (local_ids < local_experts)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    safe_weights = torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).float()
    gate_weights = x.new_zeros(tokens, local_experts, dtype=torch.float32)
    gate_weights.scatter_add_(1, safe_ids, safe_weights)  # [T,E] f32

    # ---- static gathered-expert count (>= # distinct hit experts) ----
    # The number of distinct routed experts is provably <= tokens * K, so with
    # g = min(E, tokens*K) every real hit is included and the extra (zero-weight)
    # padding experts contribute exactly +0.0 to the weighted sum. Keeping `g`
    # static (no torch.nonzero, no `if G == 0`) lets this capture cleanly into a
    # compiled HPU graph (mirrors _gather_swigluoai_moe).
    gathered_experts = min(local_experts, tokens * num_topk)
    hit_counts = (gate_weights != 0).float().sum(0)  # [E]
    gather_ids = torch.topk(hit_counts, gathered_experts, sorted=False).indices  # [G]
    gather_ids, _ = torch.sort(gather_ids)  # ascending ids

    # ---- gather only the active experts' weights + per-channel scales ----
    w13_gathered = w13.index_select(0, gather_ids)  # [G, 2I, H] fp8
    w2_gathered = w2.index_select(0, gather_ids)  # [G, H, I] fp8
    scale13_gathered = scale13.index_select(0, gather_ids)  # [G, 2I]
    scale2_gathered = scale2.index_select(0, gather_ids)  # [G, H]

    # ---- per-expert MLP, all in FP8 (no dequant to fp32) ----
    #
    # UP/GATE projection: x is shared across all routed experts, so the G
    # per-expert GEMMs `x @ w13[g]` collapse into ONE wide fp8 GEMM over the
    # concatenated expert weights [T,H] x [H, G*2I] -> [T, G*2I]. Expert g's
    # block lands on columns [g*2I:(g+1)*2I] via the permute(2,0,1) ordering
    # (NOT permute(1,0,2), which interleaves experts and is wrong). The
    # per-channel B_scale_inv is concatenated to match, with B_scale_shape
    # declaring per-channel (not per-block) scaling. fp8_gemm_v2 accumulates in
    # fp32 internally and returns out_dtype; A_scale_inv MUST be passed as [T,1].
    w13_columns = w13_gathered.permute(2, 0, 1).reshape(hidden_size, -1)  # [H, G*2I] fp8
    scale13_columns = scale13_gathered.reshape(-1)  # [G*2I]
    projected = torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w13_columns,
        trans_B=False,
        D=None,
        out_dtype=torch.float32,
        A_scale_inv=x_scale,
        B_scale_inv=scale13_columns,
        B_scale_shape=[gathered_experts * 2 * intermediate_size],
        bias=None,
        accumulate=False,
    )  # [T, G*2I] f32
    projected = projected.reshape(tokens, gathered_experts, 2 * intermediate_size).permute(1, 0, 2)  # [G, T, 2I] f32
    gate, up = projected[..., :intermediate_size], projected[..., intermediate_size:]
    activations = gate * torch.sigmoid(gate) * up  # silu(gate)*up, [G,T,I] f32

    # DOWN projection: act has a wide dynamic range (silu output), so it CANNOT
    # be fp8-quantized without losing ~2-6% precision (fp8 has only 3 mantissa
    # bits), which would blow the 2-ULP bar. The stock op keeps act high-precision
    # here, and HPU's fp8 GEMM ops (fp8_gemm_v2 / fp8_gemm) only accept fp8
    # activations. So the down projection dequantizes w2 to fp32 and runs a plain
    # fp32 bmm (same as the baseline); only the UP/GATE projection runs in native
    # fp8.
    w2_float = w2_gathered.float() * scale2_gathered.unsqueeze(-1)  # [G, H, I] f32
    expert_outputs = torch.bmm(activations, w2_float.transpose(1, 2))  # [G, T, H] f32

    # ---- weighted sum over experts -> [T, H] ----
    gathered_weights = gate_weights.index_select(1, gather_ids).t()  # [G, T]
    out = (expert_outputs * gathered_weights.unsqueeze(-1)).sum(0)  # [T, H] f32
    return out.to(x.dtype)


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign-aware FP8-ULP distance + out-of-range mask.

    Same-sign pairs use real E4M3 ulps: |uint8(quantize(a)) - uint8(quantize(b))|.
    In E4M3FN adjacent same-sign representable values differ by exactly +/-1 in
    uint8, so this IS the representable-step count. Cross-sign pairs use
    universal subnormal units |a - b| / 2^-9. Returns elementwise Float32 ulps
    (set to 0.0 wherever either input is out of range) plus an ``out_of_range``
    bool mask flagging those positions.

    Elements above the E4M3 maximum representable value (448) saturate to
    ``0x7F`` / ``0xFF`` (NaN/inf codes) during the uint8 cast and would falsely
    compare as 0 ULP, so the caller must not fold them into the finite-ULP bar;
    they are handled separately via ``out_of_range``.
    """
    a_fp32 = a.float()
    b_fp32 = b.float()
    e4m3_max = 448.0
    out_of_range = (a_fp32.abs() > e4m3_max) | (b_fp32.abs() > e4m3_max)
    a_bits = a_fp32.to(torch.float8_e4m3fn).view(torch.uint8).int()
    b_bits = b_fp32.to(torch.float8_e4m3fn).view(torch.uint8).int()
    same_sign = (a_bits >= 128) == (b_bits >= 128)
    same_sign_ulp = (a_bits - b_bits).abs().float()
    cross_ulp = (a_fp32 - b_fp32).abs() / (2.0**-9)
    ulp = torch.where(same_sign, same_sign_ulp, cross_ulp)
    ulp = torch.where(out_of_range, torch.zeros_like(ulp), ulp)
    return ulp, out_of_range


def _verify_moe_combine(stock: torch.Tensor, custom: torch.Tensor) -> None:
    ulp, out_of_range = _fp8_ulp(stock, custom)
    max_in_range_ulp = ulp.amax()
    # Out-of-range positions cannot be compared in ULP (the fp8 cast saturates
    # them), so measure their divergence as a relative error instead. Both paths
    # agreeing out of range (rel_err 0) stays quiet.
    a_fp32 = stock.float()
    b_fp32 = custom.float()
    a_finite, b_finite = torch.isfinite(a_fp32), torch.isfinite(b_fp32)
    denom = a_fp32.abs() + b_fp32.abs()
    rel_err = ((a_fp32 - b_fp32).abs() / denom.clamp(min=1e-9)).where(out_of_range & a_finite & b_finite,
                                                                      torch.zeros_like(a_fp32))
    # one side finite and the other inf/NaN is a divergence whatever the magnitude
    rel_err = torch.where(a_finite != b_finite, torch.ones_like(rel_err), rel_err)
    max_out_of_range_rel = rel_err.amax()
    reduced = torch.stack([max_in_range_ulp, max_out_of_range_rel])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ep_group = get_ep_group()
        if ep_group.world_size > 1:
            torch.distributed.all_reduce(
                reduced,
                op=torch.distributed.ReduceOp.MAX,
                group=ep_group.device_group,
            )
    max_in_range_ulp, max_out_of_range_rel = reduced[0], reduced[1]
    if max_out_of_range_rel.item() > _HPU_MOE_GATHER_VERIFY_REL_ERR:
        logger.warning(
            "MoE gather combine: out-of-range element(s) (value > 448, "
            "outside E4M3 finite range) diverge by relative error %s", max_out_of_range_rel.item())
    if max_in_range_ulp.item() > _HPU_MOE_GATHER_VERIFY_MAX_ULP:
        logger.warning("MoE gather combine mismatch: max FP8-ULP = %s exceeds %d", max_in_range_ulp.item(),
                       _HPU_MOE_GATHER_VERIFY_MAX_ULP)


def _gaudi_fix_moe_gather_apply(method, layer, x, topk_ids, topk_weights, activation):
    """The v0.30.0 gate: gather only while it wins, else the stock fused op."""
    gather_ok = getattr(method, "block_quant", False) and envs.VLLM_HPU_FORCE_CHANNEL_FP8
    use_gather = (_HPU_MOE_GATHER and activation == "silu" and method.quant_config.activation_scheme != "static"
                  and gather_ok
                  and x.shape[0] * topk_ids.shape[-1] <= layer.local_num_experts * _HPU_MOE_GATHER_RATIO)
    if not use_gather:
        return layer.moe_op(x, topk_ids, topk_weights, permuted_weights=True, activation=activation)
    if _HPU_MOE_GATHER_VERIFY:
        stock = layer.moe_op(x, topk_ids, topk_weights, permuted_weights=True, activation=activation)
        custom = gather_silu_fp8_moe(layer, x, topk_ids, topk_weights)
        _verify_moe_combine(stock, custom)
        del stock
        return custom
    return gather_silu_fp8_moe(layer, x, topk_ids, topk_weights)
'''.replace("{MARKER}", MARKER)

# Names the appended block defines; none may exist in the stock file.
NEW_NAMES = (
    "def _dynamic_quant(",
    "def gather_silu_fp8_moe(",
    "def _fp8_ulp(",
    "def _verify_moe_combine(",
    "\nlogger = ",
    "_HPU_MOE_GATHER",
)
REQUIRED = (
    "class HPUFp8MoEMethod(",
    "from vllm_gaudi import envs\n",
    "fp8.Fp8MoEMethod = HPUFp8MoEMethod\n",
)


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_fp8(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "hpu_fp8.py")
    for name in NEW_NAMES:
        if name in src:
            raise Drift(f"hpu_fp8.py already defines {name.strip()!r}")
    require_once(ctx["ops"], "def dynamic_quant(data", "extension/ops.py")
    src = replace_once(src, CALL_OLD, CALL_NEW, "expert call in HPUFp8MoEMethod.apply_monolithic")
    return src.rstrip("\n") + "\n" + BLOCK


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("fp8", "from vllm_gaudi.ops.hpu_moe_combine import gather_silu_fp8_moe"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "extension/ops.py", None, "ops"),
    ("vllm_gaudi", "ops/hpu_fp8.py", patch_fp8, "fp8"),
)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
