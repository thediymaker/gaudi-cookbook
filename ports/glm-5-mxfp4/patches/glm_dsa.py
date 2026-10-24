#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm-dsa: serve GLM-5.x DSA layers as dense MLA and wire in glm5_hpu.dsa (GLM_DSA_SPARSE).

GLM-5.x only.

vllm-gaudi's MLA wrapper passes the model's DSA flag (is_sparse) to its attention layer, and the native sparse
path does not run under this configuration (it fails at the first forward). The patch turns it off, so every layer
runs dense MLA (exact up to index_topk keys; with the accurate FusedSDPA softmax it passes the needle gates to
128K), and hooks in glm5_hpu.dsa, which does the real top-k selection when GLM_DSA_SPARSE=1:
  1. end of vllm_gaudi/attention/oot_mla.py: glm5_hpu.dsa.install() swaps in the DSA forward methods (no-op with
     the switch off);
  2. after the runner binds the KV cache: glm5_hpu.dsa.alloc_buffers() allocates the indexer-key buffers;
  3. at the top of the prompt attention-bias builder: glm5_hpu.dsa.skip_prompt_bias() skips the dense
     [bs, 1, Q, ctx + Q] bias for long single-sequence prefills, which build their own masks.
Needs the glm5_hpu package (install.sh installs it).

Usage: glm_dsa.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:glm-dsa"
MARKER = f"[{FIX_ID}]"

OLD_SPARSE = """\
        self.is_sparse = mla_modules.is_sparse
"""

NEW_SPARSE = """\
        # [glm-5-mxfp4:glm-dsa] the native DSA path does not run here: dense MLA, top-k via glm5_hpu.dsa
        self.is_sparse = False
"""

MLA_TAIL = """\


# ---- [glm-5-mxfp4:glm-dsa] GLM_DSA_SPARSE: top-k sparse attention (glm5_hpu.dsa; no-op when off) ----
from glm5_hpu import dsa as _glm_dsa  # noqa: E402

_glm_dsa.install(HPUMultiHeadLatentAttentionWrapper, HPUMLAAttention)
"""

OLD_BIND = """\
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, self.kv_caches)
"""

NEW_BIND = """\
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, self.kv_caches)
        from glm5_hpu import dsa as _glm_dsa  # [glm-5-mxfp4:glm-dsa] indexer-key buffers
        _glm_dsa.alloc_buffers(self.model, kv_caches)
"""

OLD_BIAS = """\
        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        assert seq_lens_t is not None, "seq_lens_tensor is required to build attn_bias"
"""

NEW_BIAS = """\
        from glm5_hpu import dsa as _glm_dsa  # [glm-5-mxfp4:glm-dsa] sparse prefill builds its own masks
        if _glm_dsa.skip_prompt_bias(attn_metadata, batch_size, seq_len):
            return attn_metadata
        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        assert seq_lens_t is not None, "seq_lens_tensor is required to build attn_bias"
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


def patch_mla(src: str, ctx: dict[str, str]) -> str:
    for need in ("\nclass HPUMultiHeadLatentAttentionWrapper(", "\nclass HPUMLAAttention("):
        require_once(src, need, "vllm_gaudi/attention/oot_mla.py")
    return replace_once(src, OLD_SPARSE, NEW_SPARSE, "is_sparse") + MLA_TAIL


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    src = replace_once(src, OLD_BIND, NEW_BIND, "bind_kv_cache")
    return replace_once(src, OLD_BIAS, NEW_BIAS, "prompt attention bias")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "attention/oot_mla.py", patch_mla, "mla"),
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
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
