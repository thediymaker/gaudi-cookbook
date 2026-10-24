#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""decode-lean: wire the decode-step levers of glm5_hpu.decode_lean into vLLM and vllm-gaudi.

GLM-5.x with MTP. Needs spec-draft-prefill, spec-sampling and spec-penalties first. Each lever has its own switch
and is off by default (see glm5_hpu/decode_lean.py); greedy outputs are bit-identical on and off.
  1. GLM_ROUTER_FAST: vLLM's torch grouped top-k skips the group stage when every group is selected.
  2. GLM_EARLY_DRAFT: in a decode-only step the runner enqueues the MTP draft right after the rejection sampler
     and propose_eagle_decode returns it.
  3. GLM_SPEC_DEVICE_REJECT: the rejection walk runs on the device (falls back to the host walk where it does not
     apply).
Needs the glm5_hpu package (install.sh installs it).

Usage: decode_lean.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:decode-lean"
MARKER = f"[{FIX_ID}]"

ROUTER_DEF = """\

def _grouped_topk(
"""

ROUTER_IMPORT = """\

from glm5_hpu import decode_lean as _glm_lean  # noqa: E402  [glm-5-mxfp4:decode-lean] GLM_ROUTER_FAST

"""

OLD_ROUTER = """\
    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
"""

NEW_ROUTER = """\
    num_token = scores.shape[0]
    # [glm-5-mxfp4:decode-lean] every group selected: the group mask is all ones, skip it
    if _glm_lean.ROUTER_FAST and 0 < num_expert_group <= topk_group:
        return _glm_lean.grouped_topk_all_groups(
            scores, topk, renormalize, routed_scaling_factor, e_score_correction_bias
        )
    if e_score_correction_bias is not None:
"""

OLD_RESET = """\
        sampling_metadata = None  # [glm-5-mxfp4:spec-draft-prefill] partial-chunk-only steps sample nothing
"""

NEW_RESET = """\
        sampling_metadata = None  # [glm-5-mxfp4:spec-draft-prefill] partial-chunk-only steps sample nothing
        self._glm_early_draft = None  # [glm-5-mxfp4:decode-lean] early draft of this step
"""

OLD_EARLY = """\
                        sampled_token_ids = sampler_output.sampled_token_ids
                        decode_sampled_token_ids = \\
"""

NEW_EARLY = """\
                        sampled_token_ids = sampler_output.sampled_token_ids
                        if _glm_lean.EARLY_DRAFT:  # [glm-5-mxfp4:decode-lean]
                            _glm_lean.early_draft(self, sampled_token_ids, spec_decode_metadata,
                                                  non_flattened_hidden_states, num_decodes, num_prefills,
                                                  decode_data, warmup_mode)
                        decode_sampled_token_ids = \\
"""

OLD_TAKE = """\
        if decode_data.spec_decode_metadata is None:
            # No sequence scheduled any spec decode tokens
"""

NEW_TAKE = """\
        _glm_draft = _glm_lean.take_early_draft(self, num_decodes)  # [glm-5-mxfp4:decode-lean]
        if _glm_draft is not None:
            return _glm_draft
        if decode_data.spec_decode_metadata is None:
            # No sequence scheduled any spec decode tokens
"""

RUNNER_TAIL = """\


# ---- [glm-5-mxfp4:decode-lean] GLM_EARLY_DRAFT (glm5_hpu.decode_lean) ----
from glm5_hpu import decode_lean as _glm_lean  # noqa: E402
"""

SAMPLER_TAIL = """\


# ---- [glm-5-mxfp4:decode-lean] GLM_SPEC_DEVICE_REJECT: the rejection walk on the device (glm5_hpu.decode_lean) ----
from glm5_hpu import decode_lean as _glm_lean  # noqa: E402

if _glm_lean.DEVICE_REJECT:
    rejection_sampler.rejection_sample = _glm_lean.make_device_reject(
        rejection_sampler.rejection_sample, PLACEHOLDER_TOKEN_ID, _HPU_SPEC_EPS
    )
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


def patch_router(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(src, OLD_ROUTER, NEW_ROUTER, "group stage")
    return replace_once(src, ROUTER_DEF, ROUTER_IMPORT + ROUTER_DEF[1:], "_grouped_topk")


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(src, OLD_RESET, NEW_RESET, "sample_tokens start")
    src = replace_once(src, OLD_EARLY, NEW_EARLY, "after the rejection sampler")
    src = replace_once(src, OLD_TAKE, NEW_TAKE, "propose_eagle_decode")
    return src + RUNNER_TAIL


def patch_sampler(src: str, ctx: dict[str, str]) -> str:
    for need in ("\nPLACEHOLDER_TOKEN_ID = ", "\n_HPU_SPEC_EPS = "):
        require_once(src, need, "vllm_gaudi/v1/sample/hpu_rejection_sampler.py (spec-sampling)")
    if "[glm-5-mxfp4:spec-penalties]" not in src:
        raise Drift("spec-penalties must be applied first")
    return src + SAMPLER_TAIL


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "model_executor/layers/fused_moe/router/cpu_router.py", patch_router, "router"),
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
    ("vllm_gaudi", "v1/sample/hpu_rejection_sampler.py", patch_sampler, "sampler"),
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
