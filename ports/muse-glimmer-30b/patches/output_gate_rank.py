#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""output-gate-rank: Muse Glimmer attention keeps its 3D shape on HPU, so batched decode works.

`MuseGlimmerAttention.forward` in vLLM multiplies a sigmoid gate computed from the layer input by the
attention output. On CUDA both are 2D `[tokens, hidden]`. The Gaudi runner drives the model in 3D
(`[batch, query_bucket, hidden]`) while the HPU attention op returns `[batch * query_bucket, hidden]`.
Broadcasting from the right makes batch 1 work by accident; any batch above 1 fails with "The size of
tensor a (128) must match the size of tensor b (256) at non-singleton dimension 1" and stops the
engine. Single-stream tests pass, the first two concurrent requests kill the server.

The patch restores the input's rank once, right after the attention call. The element count and the
token order are unchanged, so it is a pure re-view, and a no-op where the ranks already agree (CUDA).
That also repairs the `residual + hidden_states` adds in the decoder layer, which see the same output.

Usage: output_gate_rank.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PATCH_ID = "muse-glimmer-output-gate-rank"
MARKER = f"[gaudi-port:{PATCH_ID}]"
TARGET = ("vllm", "model_executor/models/muse_glimmer.py")

OLD = """        attn_output = self.attn(q, k, v)

        if self.use_output_gate:
"""
NEW = f"""        attn_output = self.attn(q, k, v)
        # {MARKER} the HPU attention op returns [batch * query, hidden] while the Gaudi
        # runner passes hidden_states as [batch, query, hidden]; restore the rank before the
        # gate multiply and the residual adds (batch 1 broadcasts by accident, batch 2 fails).
        if attn_output.dim() != hidden_states.dim():
            attn_output = attn_output.view(*hidden_states.shape[:-1], -1)

        if self.use_output_gate:
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def patch(src: str) -> str:
    n = src.count(OLD)
    if n != 1:
        raise Drift(f"attention output block found {n} times, expected 1")
    return src.replace(OLD, NEW, 1)


def locate(root: str | None) -> Path | None:
    pkg, rel = TARGET
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
    tmp = path.with_name(path.name + ".gaudi-port.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


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
    write_atomic(path, new)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
