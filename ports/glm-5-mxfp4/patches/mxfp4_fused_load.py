#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mxfp4-fused-load: load fused MXFP4 experts of models other than gpt-oss under expert parallelism.

Generic (any DeepSeek-style model on vllm-gaudi's native MXFP4 MoE path); a candidate for fixes/.

vLLM's RoutedExperts.weight_loader has a gpt_oss_mxfp4 branch that assumes gpt-oss's own loader already sliced the
fused expert tensors per rank. A DeepSeek-style model reaches it through deepseek_v2's generic loader, which
  1. calls weight_loader(param, loaded_weight) with two arguments where the signature requires five: TypeError
     before anything is copied;
  2. hands over the global [256, ...] expert tensor, which the branch copies into the per-rank [32, ...] parameter:
     shape error.
The patch defaults the extra arguments and takes this rank's contiguous expert block when the checkpoint tensor is
larger than the parameter. TP slicing inside an expert is refused, so the model must run with
--enable-expert-parallel. gpt-oss is unaffected: its tensors arrive pre-sliced and the narrow is a no-op.

Usage: mxfp4_fused_load.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:mxfp4-fused-load"
MARKER = f"[{FIX_ID}]"

OLD_SIG = """\
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
"""

NEW_SIG = """\
    def weight_loader(  # [glm-5-mxfp4:mxfp4-fused-load] deepseek_v2 passes two args
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = None,
        expert_id: int = None,
        return_success: bool = False,
    ) -> bool | None:
"""

OLD_BRANCH = """\
        quant_config_name = self.quant_config and self.quant_config.get_name()
        if quant_config_name == "gpt_oss_mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None
"""

NEW_BRANCH = """\
        quant_config_name = self.quant_config and self.quant_config.get_name()
        if quant_config_name == "gpt_oss_mxfp4":
            # [glm-5-mxfp4:mxfp4-fused-load] gpt-oss arrives sliced per rank (the narrows below are
            # no-ops). Other fused checkpoints arrive with the global expert dim through
            # deepseek_v2's direct-name path: take this rank's contiguous EP block. No TP
            # slicing inside an expert: serve with --enable-expert-parallel.
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                n_local = param.data.shape[0]
                if loaded_weight.shape[0] > n_local:
                    ep_rank = self.moe_config.ep_rank
                    loaded_weight = loaded_weight.narrow(
                        0, ep_rank * n_local, n_local)
                if (loaded_weight.shape[1] > param.data.shape[1]
                        or loaded_weight.shape[2] > param.data.shape[2]):
                    raise ValueError(
                        "[glm-5-mxfp4:mxfp4-fused-load] fused mxfp4 tensor "
                        f"{tuple(loaded_weight.shape)} exceeds param "
                        f"{tuple(param.data.shape)}: intra-expert TP slicing "
                        "unsupported; use --enable-expert-parallel")
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:loaded_weight.shape[0], :dim1, :dim2].copy_(
                    loaded_weight)
            return True if return_success else None
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


def patch_experts(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(src, OLD_SIG, NEW_SIG, "weight_loader signature")
    return replace_once(src, OLD_BRANCH, NEW_BRANCH, "gpt_oss_mxfp4 branch")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "model_executor/layers/fused_moe/routed_experts.py", patch_experts, "experts"),)


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
