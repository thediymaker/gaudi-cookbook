#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the vendored files of vLLM PR #62052 (Beam parsers) into the installed vLLM tree.

The pull request's parsers sit on a response-template framework (vllm/parser/chat_parsing/,
vllm/parser/response_template.py) and a response grammar (vllm/parser/response_template_grammar.py)
that vLLM v0.30.0 does not have. Those files import each other by absolute `vllm.*` paths, so they
go in-tree at their upstream paths. None of them exists in v0.30.0, so nothing stock is replaced.
The Beam parser classes themselves are beam_hpu/parser.py (a backport, not a vendored file).

One edit is made on the way in: response_template.py imports make_tool_call_id from
vllm.renderers.chat_utils, which is vllm.entrypoints.chat_utils in v0.30.0 (exactly one hit).

Every source file must match the sha256 in vendor/MANIFEST before anything is written.

Usage: vllm_pr_62052.py [--check] [--root DIR_HOLDING_vllm]
Exit codes: 0 installed or already installed, 1 (--check) not installed yet, 2 a source or the
target differs from what this script expects (nothing written).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

MARKER = "[beam-1-0:vllm-pr-62052]"
VENDOR = Path(__file__).resolve().parent.parent / "vendor"
MANIFEST = VENDOR / "MANIFEST"

TEMPLATE = "vllm/parser/response_template.py"
IMPORT_OLD = "from vllm.renderers.chat_utils import make_tool_call_id\n"
IMPORT_NEW = "from vllm.entrypoints.chat_utils import make_tool_call_id\n"


class Drift(Exception):
    pass


def manifest() -> list[tuple[str, str]]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, rel = line.split()
        out.append((digest, rel))
    if not out:
        raise Drift(f"no entries in {MANIFEST}")
    return out


def planned() -> list[tuple[str, bytes]]:
    """(path relative to the vLLM root, content) for every file this script owns."""
    files = []
    for digest, rel in manifest():
        src = (VENDOR / rel).read_bytes()
        if hashlib.sha256(src).hexdigest() != digest:
            raise Drift(f"vendor/{rel} does not match vendor/MANIFEST")
        if rel == TEMPLATE:
            text = src.decode()
            if text.count(IMPORT_OLD) != 1:
                raise Drift(f"{rel}: expected one {IMPORT_OLD.strip()!r}")
            src = text.replace(IMPORT_OLD, IMPORT_NEW).encode()
        files.append((rel, src))
    return files


def vllm_root(root: str | None) -> Path | None:
    if root:
        return Path(root) if (Path(root) / "vllm" / "__init__.py").is_file() else None
    spec = importlib.util.find_spec("vllm")
    locs = list(spec.submodule_search_locations or []) if spec else []
    return Path(locs[0]).parent if locs else None


def run(check: bool, root: str | None) -> int:
    base = vllm_root(root)
    if base is None:
        print(f"{MARKER} vllm package not found", file=sys.stderr)
        return 2
    if not (base / "vllm" / "parser" / "parser_manager.py").is_file():
        print(f"{MARKER} {base}/vllm/parser/parser_manager.py missing: not a vLLM v0.30 tree", file=sys.stderr)
        return 2
    try:
        files = planned()
    except Drift as e:
        print(f"{MARKER} {e}; nothing written", file=sys.stderr)
        return 2
    todo = []
    for rel, data in files:
        dst = base / rel
        if dst.is_file() and dst.read_bytes() == data:
            continue
        if dst.exists():
            # a different file already sits at an upstream path: a newer vLLM, do not overwrite
            print(f"{MARKER} {dst} exists with other content; nothing written", file=sys.stderr)
            return 2
        todo.append((dst, data))
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied ({len(todo)} of {len(files)} files to install)")
        return 1
    for dst, data in todo:
        compile(data, str(dst), "exec", dont_inherit=True)
    for dst, data in todo:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    print(f"{MARKER} installed {len(todo)} files under {base}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
