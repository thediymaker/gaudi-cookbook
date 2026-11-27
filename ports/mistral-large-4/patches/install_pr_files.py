#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the vendored vLLM PR #64534 frontend files into a vLLM v0.30.0 source tree.

    python3 install_pr_files.py <vendor dir> <vllm repo root, the parent of the vllm package>
    python3 install_pr_files.py --check <vendor dir> <vllm repo root>   verify only, write nothing

The files are the PR's params.json adapter (the `dsa` branch: model_type mistral_large_4,
per-layer NoPE / KV-sharing / index-sharing tables), the tokenizer, renderer and
/tokenize changes that render `response_format` into the prompt, and the parser that
builds its grammar from mistral-common's request. Every one of them REPLACES a stock
v0.30.0 file, so the stock bytes are pinned too: vendor/MANIFEST records
both hashes per file (`<vendored sha256>  <path>  replaces <stock sha256>`).

Exit codes follow the fix contract (fixes/README.md): 0 installed (or already installed), 1 not installed
(--check), 2 a file or target does not match (nothing is written).

Steps, each fail-closed (exit 2, nothing half-written is left as "done"):
  1. every file listed in vendor/MANIFEST exists under <vendor dir> with the recorded sha256;
  2. every target in the tree is either the recorded stock v0.30.0 file (it is replaced)
     or already the vendored file (left alone); anything else (another vLLM, a local
     edit, a different PR revision) stops the install before any write;
  3. the files are copied to <root>/<path>;
  4. every installed file compiles, and its content is re-read and checked.

Idempotent: a second run finds identical content and rewrites nothing.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

MARK = "[mistral-large-4 install_pr_files]"
HERE = pathlib.Path(__file__).resolve().parent
VENDORED_AT = HERE.parent / "vendor" / "MANIFEST"


def fail(msg: str) -> int:
    print(f"{MARK} FAILED: {msg}", file=sys.stderr)
    return 2


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def read_manifest(path: pathlib.Path) -> list[tuple[str, str, str]]:
    """(vendored sha256, path, stock sha256) per file line."""
    entries = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and _is_sha(parts[0]) and parts[2] == "replaces" and _is_sha(parts[3]):
            entries.append((parts[0], parts[1], parts[3]))
    return entries


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    args = [a for a in argv[1:] if a != "--check"]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    vendor = pathlib.Path(args[0]).resolve()
    root = pathlib.Path(args[1]).resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        return fail(f"{root} is not a vLLM source root (no vllm/__init__.py)")
    if not VENDORED_AT.is_file():
        return fail(f"{VENDORED_AT} missing")
    manifest = read_manifest(VENDORED_AT)
    if not manifest:
        return fail(f"no file entries in {VENDORED_AT}")

    # 1./2. check every source against the recorded hash and every target against the
    # recorded stock (or vendored) hash before touching the tree
    planned: list[tuple[str, bytes, bool]] = []
    for digest, rel, stock in manifest:
        src_path = vendor / rel
        if not src_path.is_file():
            return fail(f"{src_path} missing")
        src = src_path.read_bytes()
        if sha256(src) != digest:
            return fail(f"{rel}: sha256 {sha256(src)} != vendor/MANIFEST {digest}")
        dst = root / rel
        if not dst.is_file():
            return fail(f"{dst} missing: not the v0.30.0 tree this merge was made against")
        have = sha256(dst.read_bytes())
        if have not in (stock, digest):
            return fail(
                f"{dst}: sha256 {have} is neither the stock v0.30.0 file ({stock[:12]}) "
                f"nor the vendored one ({digest[:12]})"
            )
        planned.append((rel, src, have == digest))

    if check_only:
        stale = [rel for rel, _, done in planned if not done]
        if stale:
            print(f"{MARK} not installed: {', '.join(stale)}")
            return 1
        print(f"{MARK} check ok: {len(planned)} files installed in {root}")
        return 0

    # 3. install
    written = same = 0
    for rel, data, done in planned:
        if done:
            same += 1
            continue
        (root / rel).write_bytes(data)
        written += 1

    # 4. verify what is on disk now
    for rel, data, _ in planned:
        dst = root / rel
        on_disk = dst.read_bytes()
        if on_disk != data:
            return fail(f"{dst} content differs after install")
        try:
            compile(on_disk, str(dst), "exec")
        except SyntaxError as e:
            return fail(f"{dst} does not compile: {e}")

    print(
        f"{MARK} ok: {len(planned)} files into {root} ({written} written, {same} unchanged); "
        f"source {VENDORED_AT.read_text().split('source: ', 1)[-1].splitlines()[0]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
