#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mxfp4-fused-load: load fused MXFP4 expert tensors of a DeepSeek-V3-style checkpoint under expert parallelism.

The converted Kimi checkpoint (convert/repack_quark_mxfp4.py) stores each layer's routed experts fused, as
vLLM's `gpt_oss_mxfp4` format expects: `...mlp.experts.routed_experts.w13_weight` [384, 4096, 3584] and so on.
RoutedExperts.weight_loader has a fast path for that format which assumes the model's own loader already cut
the tensor to this rank, as gpt-oss's does. DeepSeek-V2/V3 (the Kimi text model) loads these names directly,
so two things fail:

  1. it calls weight_loader(param, loaded_weight) with two arguments, but weight_name, shard_id and expert_id
     have no defaults: TypeError;
  2. the fast path copies the global [384, ...] tensor into the rank's [48, ...] parameter: shape error.

The patch defaults the three arguments and, when the tensor has more experts than the parameter, copies this
rank's contiguous block of experts (ep_rank * n_local, the linear placement vLLM and the HPU MoE op use).
Splitting inside an expert (tensor parallelism without expert parallelism) is not implemented and raises: serve
with --enable-expert-parallel. gpt-oss is unaffected: its tensors arrive per rank, so both narrows are no-ops.

Usage: mxfp4_fused_load.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "kimi-k2-7:mxfp4-fused-load"
MARKER = f"[{FIX_ID}]"

OLD_SIG = """    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
"""
NEW_SIG = f"""    def weight_loader(  # {MARKER} the DeepSeek loader passes only (param, loaded_weight)
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = None,
        expert_id: int = None,
        return_success: bool = False,
    ) -> bool | None:
"""

OLD_BRANCH = """            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None
"""
NEW_BRANCH = f"""            else:
                # {MARKER} a fused checkpoint carries every expert: take this rank's block
                n_local = param.data.shape[0]
                if loaded_weight.shape[0] > n_local:
                    loaded_weight = loaded_weight.narrow(0, self.moe_config.ep_rank * n_local, n_local)
                if loaded_weight.shape[1] > param.data.shape[1] or loaded_weight.shape[2] > param.data.shape[2]:
                    raise ValueError(
                        f"fused MXFP4 expert tensor {{tuple(loaded_weight.shape)}} does not fit the parameter "
                        f"{{tuple(param.data.shape)}}: splitting inside an expert is not supported, "
                        "serve with --enable-expert-parallel"
                    )
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[: loaded_weight.shape[0], :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None
"""
FAST_PATH = 'if quant_config_name == "gpt_oss_mxfp4":\n'


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_loader(src: str, ctx: dict[str, str]) -> str:
    for needle in (OLD_SIG, OLD_BRANCH, FAST_PATH):
        require_once(src, needle, "routed_experts.py")
    if not src.index(OLD_SIG) < src.index(FAST_PATH) < src.index(OLD_BRANCH):
        raise Drift("routed_experts.py: the gpt_oss_mxfp4 fast path is not where expected")
    return src.replace(OLD_SIG, NEW_SIG, 1).replace(OLD_BRANCH, NEW_BRANCH, 1)


UPSTREAM_FIXED = ()

TARGETS = (("vllm", "model_executor/layers/fused_moe/routed_experts.py", patch_loader, "loader"),)


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
