#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""laguna-moe-sigmoid-routing: honour scoring_func, correction bias and renormalize in MoE top-k routing.

The monolithic MoE apply paths (unquantized, FP8, compressed-tensors W8A8 FP8 and WNA16, MXFP4)
send every layer that is not grouped top-k or custom routing through an inline softmax top-k that
always renormalizes. That branch ignores the layer's scoring_func, e_score_correction_bias and
renormalize, so sigmoid-scored MoEs with an aux-loss-free selection bias (for example Laguna) and
softmax MoEs that do not renormalize pick or weight experts differently from the reference.

The inline fast path is kept only for plain renormalized softmax layers (is_plain_softmax_topk).
Everything else goes through select_experts_from_routed, which now handles non-grouped top-k
itself like vLLM's fused_topk_bias: softmax or sigmoid scores in fp32, the correction bias used
for expert selection only, weights gathered from the unbiased scores, then optional
renormalization and the layer's routed_scaling_factor. On v0.26.0 the MXFP4 path also switches
from the removed layer.router to select_experts_from_routed, as v0.30.0 already does.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "laguna-moe-sigmoid-routing"
MARKER = f"[gaudi-fix:{FIX_ID}]"

STOCK_COND = 'if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:\n'
NEW_COND = f"if not is_plain_softmax_topk(layer):  # {MARKER}\n"

HELPERS = f'''def is_plain_softmax_topk(layer) -> bool:
    """{MARKER} Whether the layer routes with plain renormalized softmax top-k.

    The monolithic apply paths compute this routing inline. Any other routing (grouped top-k,
    custom routing, non-softmax scoring, a score correction bias or no renormalization) must go
    through ``select_experts_from_routed``.
    """
    return (not layer.use_grouped_topk and getattr(layer, "custom_routing_function", None) is None
            and getattr(layer, "scoring_func", "softmax") == "softmax"
            and getattr(layer, "e_score_correction_bias", None) is None and getattr(layer, "renormalize", True))


def _topk_routing(layer, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """{MARKER} Non-grouped top-k routing with an optional score correction bias.

    Matches vLLM's ``fused_topk_bias``: the bias only affects which experts are selected, the
    routing weights are the unbiased scores.
    """
    scoring_func = getattr(layer, "scoring_func", "softmax")
    logits = router_logits.float()
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function for non-grouped top-k routing: {{scoring_func}}")

    bias = getattr(layer, "e_score_correction_bias", None)
    if bias is None:
        topk_weights, topk_ids = torch.topk(scores, layer.top_k, dim=-1)
    else:
        topk_ids = torch.topk(scores + bias.float().unsqueeze(0), layer.top_k, dim=-1)[1]
        topk_weights = scores.gather(1, topk_ids)

    if layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if layer.routed_scaling_factor != 1.0:
        topk_weights = topk_weights * layer.routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


'''
SELECT_DEF = "def select_experts_from_routed(layer, hidden_states: torch.Tensor,\n"
SELECT_RETURN = "    return select_experts(\n        hidden_states=hidden_states,\n"
SELECT_RETURN_NEW = (
    f"    # {MARKER} the select_experts helper only handles plain softmax for non-grouped top-k.\n"
    '    if not layer.use_grouped_topk and getattr(layer, "custom_routing_function", None) is None:\n'
    "        return _topk_routing(layer, router_logits)\n\n" + SELECT_RETURN
)

INLINE = "            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)\n"
FUSED_SITES = (
    # apply_monolithic and forward_oot of HPUUnquantizedFusedMoEMethod, told apart by the next branch
    (
        "        " + STOCK_COND + INLINE + "        else:\n            import torch.nn.functional as F\n\n"
        '            if self.model_type == "gpt_oss":\n',
        "unquantized apply_monolithic",
    ),
    (
        "        " + STOCK_COND + INLINE + "        else:\n            import torch.nn.functional as F\n\n"
        '            if self.model_type is not None and self.model_type in ["gpt_oss"]:\n',
        "unquantized forward_oot",
    ),
)
CT_SITES = (
    ("        x = x.view(-1, x.shape[-1])\n        " + STOCK_COND + INLINE, "W8A8 FP8 MoE"),
    ("        x = x.view(-1, x.shape[-1])\n\n        " + STOCK_COND + INLINE, "WNA16 MoE"),
)
FP8_SITE = ("        x = x.view(-1, x.shape[-1])\n        " + STOCK_COND + INLINE, "FP8 MoE")
MXFP4_SITES = (
    # v0.30.0 layout
    "            " + STOCK_COND + "                # `RoutedExperts` no longer owns a `.router`",
    # v0.26.0 layout: still calls the removed layer.router
    "            "
    + STOCK_COND
    + "                topk_weights, topk_ids = layer.router.select_experts("
    + "hidden_states=x, router_logits=router_logits)\n",
)

FP8_IMPORT = (
    "from vllm_gaudi.ops.hpu_fused_moe import (_normalize_moe_activation, model_has_quant_config, "
    "select_experts_from_routed)\n"
)
CT_IMPORTS = (
    "from vllm_gaudi.ops.hpu_fused_moe import (\n    _normalize_moe_activation,\n    select_experts_from_routed,\n)\n",
    "from vllm_gaudi.ops.hpu_fused_moe import _normalize_moe_activation, select_experts_from_routed\n",
)
MXFP4_IMPORTS = (
    "from vllm_gaudi.ops.hpu_fused_moe import (\n    _normalize_moe_activation,\n    select_experts_from_routed,\n)\n",
    "from vllm_gaudi.ops.hpu_fused_moe import _normalize_moe_activation\n",
)
NEW_IMPORT = (
    "from vllm_gaudi.ops.hpu_fused_moe import (  # {marker}\n"
    "    {names},\n"
    "    is_plain_softmax_topk,\n"
    "    select_experts_from_routed,\n"
    ")\n"
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


def _site(src: str, old: str, label: str) -> str:
    return replace_once(src, old, old.replace(STOCK_COND, NEW_COND, 1), label)


def _import(src: str, olds: tuple[str, ...], names: str, label: str) -> str:
    hits = [old for old in olds if src.count(old) == 1]
    if len(hits) != 1:
        raise Drift(f"{label}: the hpu_fused_moe import has an unknown layout")
    return src.replace(hits[0], NEW_IMPORT.format(marker=MARKER, names=names), 1)


def patch_fused_moe(src: str, ctx: dict[str, str]) -> str:
    if "def is_plain_softmax_topk(" in src:
        raise Drift("is_plain_softmax_topk already defined")
    src = replace_once(src, SELECT_DEF, HELPERS + SELECT_DEF, "select_experts_from_routed definition")
    src = replace_once(src, SELECT_RETURN, SELECT_RETURN_NEW, "select_experts call")
    for old, label in FUSED_SITES:
        src = _site(src, old, label)
    return src


def patch_fp8(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(
        src,
        FP8_IMPORT,
        NEW_IMPORT.format(marker=MARKER, names="_normalize_moe_activation,\n    model_has_quant_config"),
        "hpu_fused_moe import",
    )
    return _site(src, *FP8_SITE)


def patch_ct(src: str, ctx: dict[str, str]) -> str:
    src = _import(src, CT_IMPORTS, "_normalize_moe_activation", "hpu_compressed_tensors.py")
    for old, label in CT_SITES:
        src = _site(src, old, label)
    return src


def patch_mxfp4(src: str, ctx: dict[str, str]) -> str:
    src = _import(src, MXFP4_IMPORTS, "_normalize_moe_activation", "hpu_mxfp4.py")
    hits = [old for old in MXFP4_SITES if src.count(old) == 1]
    if len(hits) != 1:
        raise Drift("hpu_mxfp4.py: the routing branch has an unknown layout")
    new = hits[0].replace(STOCK_COND, NEW_COND, 1)
    new = new.replace(
        "layer.router.select_experts(hidden_states=x, router_logits=router_logits)",
        "select_experts_from_routed(layer, x, router_logits)",
    )
    return src.replace(hits[0], new, 1)


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = (("fused_moe", "def is_plain_softmax_topk(layer)"),)

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "ops/hpu_fused_moe.py", patch_fused_moe, "fused_moe"),
    ("vllm_gaudi", "ops/hpu_fp8.py", patch_fp8, "fp8"),
    ("vllm_gaudi", "ops/hpu_compressed_tensors.py", patch_ct, "ct"),
    ("vllm_gaudi", "ops/hpu_mxfp4.py", patch_mxfp4, "mxfp4"),
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
