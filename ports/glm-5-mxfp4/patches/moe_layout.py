#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe-layout: EP 4 x TP 2 routed experts and per-layer expert placement (glm5_hpu.moe_layout).

GLM-5.x only. Needs mxfp4-fused-load first.

  1. vLLM's FusedMoEParallelConfig.make: with GLM_MOE_TP=t the TP ranks form groups of t, experts are split
     over the groups and each expert's intermediate dim over the ranks of a group (unset: vLLM's EP over all ranks).
  2. vLLM's fused MXFP4 loader (gpt_oss_mxfp4 branch): glm5_hpu.moe_layout.load_fused() copies this rank's slice
     (TP slice and / or the experts GLM_MOE_PLACEMENT puts in this group); otherwise the stock branch runs.
  3. vllm-gaudi's MXFP4 MoE method: router top-k ids are mapped to placement slots with one gather, and the
     per-layer slot tables are built after the weights are loaded (no-op without GLM_MOE_PLACEMENT).
Needs the glm5_hpu package (install.sh installs it).

Usage: moe_layout.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:moe-layout"
MARKER = f"[{FIX_ID}]"

OLD_CFG = """\
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
"""

NEW_CFG = """\
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        from glm5_hpu import moe_layout as _glm_moe  # [glm-5-mxfp4:moe-layout] GLM_MOE_TP

        _glm = _glm_moe.group_layout(tp_size, tp_rank, dp_size, pcp_size)
        if _glm is not None:
            return FusedMoEParallelConfig(
                tp_size=_glm[0],
                tp_rank=_glm[1],
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=_glm[2],
                ep_rank=_glm[3],
                sp_size=sp_size_,
                use_ep=True,
                all2all_backend=vllm_parallel_config.all2all_backend,
                enable_eplb=vllm_parallel_config.enable_eplb,
            )
"""

OLD_LOAD = """\
            else:
                n_local = param.data.shape[0]
"""

NEW_LOAD = """\
            else:
                from glm5_hpu import moe_layout as _glm_moe  # [glm-5-mxfp4:moe-layout] EP x TP slice, placement

                if _glm_moe.load_fused(self, param, loaded_weight):
                    return True if return_success else None
                n_local = param.data.shape[0]
"""

OLD_ROUTE = """\
                topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
"""

NEW_ROUTE = """\
                topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
                topk_ids = _glm_moe.map_ids(layer, topk_ids)  # [glm-5-mxfp4:moe-layout] expert id -> placement slot
"""

MXFP4_TAIL = """\


# ---- [glm-5-mxfp4:moe-layout] GLM_MOE_PLACEMENT: per-layer slot tables (glm5_hpu.moe_layout; no-op when off) ----
from glm5_hpu import moe_layout as _glm_moe  # noqa: E402

_glm_moe.install(HPUGptOssMxfp4MoEMethod)
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_config(src: str, ctx: dict[str, str]) -> str:
    for need in ("        dp_size = dp_size_\n", "        pcp_rank = get_pcp_group().rank_in_group"):
        require_once(src, need, "vllm/model_executor/layers/fused_moe/config.py")
    return replace_once(src, OLD_CFG, NEW_CFG, "EP parallel config")


def patch_experts(src: str, ctx: dict[str, str]) -> str:
    if "[glm-5-mxfp4:mxfp4-fused-load]" not in src:
        raise Drift("mxfp4-fused-load must be applied first")
    return replace_once(src, OLD_LOAD, NEW_LOAD, "fused mxfp4 loader")


def patch_mxfp4(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\nclass HPUGptOssMxfp4MoEMethod(", "vllm_gaudi/ops/hpu_mxfp4.py")
    return replace_once(src, OLD_ROUTE, NEW_ROUTE, "router") + MXFP4_TAIL


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "model_executor/layers/fused_moe/config.py", patch_config, "config"),
    ("vllm", "model_executor/layers/fused_moe/routed_experts.py", patch_experts, "experts"),
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
