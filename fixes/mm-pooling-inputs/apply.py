#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mm-pooling-inputs: the pooling runner runs the vision encoder and uses M-RoPE positions.

vllm-gaudi's pooling branch of HPUModelRunner.execute_model calls self.model.forward with the
prompt token ids and 1D positions only. For a multimodal embedding model (Qwen3-VL-Embedding)
that means two things: the vision encoder never runs, so image placeholder tokens are embedded
as text and every image lands on nearly the same vector, and M-RoPE models get 1D positions.
Before it gets that far, _update_states asks self.model.model for get_mrope_input_positions,
but pooling wrappers such as Qwen3VLForEmbedding expose that method on self.model itself, so
the first request stops the engine with AttributeError.

The fix calls get_mrope_input_positions on whichever of self.model and self.model.model has it,
and in the pooling branch uses the request's M-RoPE positions when the model uses them and builds
inputs_embeds through _get_model_mm_inputs (the generation path's encoder and merge) for
requests with media. Text-only requests on models without M-RoPE are unchanged.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "mm-pooling-inputs"
MARKER = f"[gaudi-fix:{FIX_ID}]"

MROPE_OLD = "                    self.model.model.get_mrope_input_positions(\n"
# A comment line cannot go here (the call continues a backslash line), so the marker trails the first line.
MROPE_NEW = f"""                    (self.model if hasattr(self.model, "get_mrope_input_positions")  # {MARKER}
                     else self.model.model).get_mrope_input_positions(
"""

POOL_OLD = """                htorch.core.mark_step()
                with set_forward_context(attn_metadata, self.vllm_config):
                    hidden_states = self.model.forward(
                        input_ids=input_ids,
                        positions=position_ids,
                        **model_kwargs,
                    )
"""
POOL_NEW = f"""                htorch.core.mark_step()
                # {MARKER} M-RoPE positions and the vision encoder, as on the generation path
                req_state = self.requests[req_id]
                if self.uses_mrope and req_state.mrope_positions is not None:
                    position_ids = self._align_and_pad_mrope_positions(
                        [req_id], [int(req_state.num_computed_tokens)], [int(num_scheduled_tokens)],
                        (1, int(input_ids.shape[-1])), -1).to(device=self.device, dtype=torch.long)
                if self.supports_mm_inputs and req_state.mm_features:
                    inputs_embeds, _ = self._get_model_mm_inputs(input_ids, int(input_ids.shape[-1]),
                                                                 scheduler_output, [req_id])
                    model_kwargs["inputs_embeds"] = inputs_embeds
                with set_forward_context(attn_metadata, self.vllm_config):
                    hidden_states = self.model.forward(
                        input_ids=input_ids,
                        positions=position_ids,
                        **model_kwargs,
                    )
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for needle in ("def _get_model_mm_inputs(", "def _align_and_pad_mrope_positions("):
        if src.count(needle) != 1:
            raise Drift(f"{needle} not found once")
    src = replace_once(src, MROPE_OLD, MROPE_NEW, "get_mrope_input_positions call")
    return replace_once(src, POOL_OLD, POOL_NEW, "pooling forward")


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),)


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
