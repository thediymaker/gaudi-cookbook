#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""rank-exception-fatal: a worker that fails a call on a non-output rank stops the engine.

In vLLM's MultiprocExecutor each worker runs RPCs from a broadcast queue. When a call raises, the
worker logs "WorkerProc hit an exception." and sends the exception back only if it is the call's
output rank (rank 0 for execute_model and sample_tokens). On every other rank the exception is
dropped. The engine keeps serving from rank 0 while the failed rank skipped the rest of that call,
so from then on the ranks are out of step and the output is undefined (for example all-NaN logits
producing one repeated token), while /health still reports OK.

With the fix such a worker logs the error and exits. The executor's worker monitor sees the dead
process and fails the engine, so the server stops instead of serving garbage and an orchestrator
can restart it. Calls whose result every rank returns, and failures on the output rank, behave as
before.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "rank-exception-fatal"
MARKER = f"[gaudi-fix:{FIX_ID}]"

_ELSE = """else:
{ind}    # {marker} nobody reads this rank's reply, so the engine would carry on
{ind}    # with this rank out of step with the others: stop the worker instead.
{ind}    _gaudi_fix_rank_exception_fatal(self.rank, output_rank)
"""

# vLLM 0.26 line: the except block sits in worker_busy_loop.
OLD_V26 = """                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
"""
# vLLM 0.30 line: the same block moved into WorkerProc._execute_worker_rpc.
OLD_V30 = """            logger.exception("WorkerProc hit an exception.")
            # enqueue_output converts the exception to a FAILURE response
            # containing its string representation before transport.
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)
"""
VARIANTS = (("v0.26 worker_busy_loop", OLD_V26, " " * 16), ("v0.30 _execute_worker_rpc", OLD_V30, " " * 12))

HELPER = f"""

def _gaudi_fix_rank_exception_fatal(rank, output_rank):  # {MARKER}
    logger.error(
        "Worker rank %s failed a call whose result is taken from rank %s. The ranks are now out of "
        "step, so this worker exits and the engine stops instead of serving undefined output.",
        rank,
        output_rank,
    )
    parent = getattr(logger, "parent", None)
    for handler in list(getattr(logger, "handlers", [])) + list(getattr(parent, "handlers", [])):
        try:
            handler.flush()
        except Exception:
            pass
    os._exit(1)
"""

REQUIRED = ("\nimport os\n", "\nlogger = init_logger(__name__)\n")


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_executor(src: str, ctx: dict[str, str]) -> str:
    for needle in REQUIRED:
        require_once(src, needle, "multiproc_executor.py")
    counts = {name: src.count(old) for name, old, _ in VARIANTS}
    hits = [(old, ind) for name, old, ind in VARIANTS if counts[name] == 1]
    if len(hits) != 1 or sum(counts.values()) != 1:
        raise Drift(f"worker exception handler: anchor counts {counts}, expected exactly one variant once")
    old, ind = hits[0]
    new = old + ind + _ELSE.format(ind=ind, marker=MARKER)
    return src.replace(old, new, 1).rstrip("\n") + "\n" + HELPER


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm", "v1/executor/multiproc_executor.py", patch_executor, "executor"),)


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
