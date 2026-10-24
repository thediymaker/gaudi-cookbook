# SPDX-License-Identifier: Apache-2.0
"""Driver shared by the port's source patches; the same contract as fixes/ (see fixes/README.md, "Contract").

* Targets are found with importlib.util.find_spec (vLLM is not imported) or under --root, a directory holding vllm/
  and vllm_gaudi/.
* Every anchor must match exactly once; any mismatch writes nothing and exits 2.
* The patched source is compiled before it is written; files are replaced atomically, __pycache__ entries removed.
* A marker makes a patch idempotent; a patch over several files refuses a partly applied set.

Exit codes: 0 applied now or already applied; 1 --check only, not applied and the anchors match; 2 drift, a missing
target or a compile failure (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path


class Drift(Exception):
    """The installed source does not match what the patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
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


Target = tuple[str, str, Callable[[str], str]]  # (package, path inside it, patch function)


def run(marker: str, targets: tuple[Target, ...], check: bool, root: str | None) -> int:
    done, todo = [], []
    try:
        for pkg, rel, fn in targets:
            path = locate(pkg, rel, root)
            if path is None:
                print(f"{marker} {pkg}/{rel} not found", file=sys.stderr)
                return 2
            src = path.read_text()
            if marker in src:
                done.append(path)
                continue
            new = fn(src)
            if marker not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(path), "exec", dont_inherit=True)
            todo.append((path, new))
    except (Drift, SyntaxError) as e:
        print(f"{marker} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{marker} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{marker} applied")
        return 0
    if check:
        print(f"{marker} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{marker} patched {path}")
    return 0


def main(doc: str, marker: str, targets: tuple[Target, ...]) -> int:
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    a = ap.parse_args()
    return run(marker, targets, a.check, a.root)
