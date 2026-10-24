#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Register the Muse Spark MXFP4 decode kernel library before any process opens a device.

The graph compiler reads GC_KERNEL_PATH when the device opens, so the kernel library has to be in
the environment before the first HPU op of every vLLM process (API server, engine core, workers).
vllm_gaudi/__init__.py is the platform plugin that vLLM imports in each of them before that point.
This script puts a call to muse_spark_hpu.mxfp4_kernels.setup_env() at its top. setup_env() returns
at once unless MUSE_SPARK_MXFP4_KERNELS=1, so with the switch off nothing changes.

Usage: kernel_env_hook.py [--check] [--root DIR_HOLDING_vllm_gaudi]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 target missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[muse-spark-open:kernel-env-hook]"
HOOK = (
    f"# {MARKER} MUSE_SPARK_MXFP4_KERNELS=1: register the kernel library before the first HPU op\n"
    "try:\n"
    "    from muse_spark_hpu.mxfp4_kernels import setup_env as _muse_spark_setup_env\n"
    "\n"
    "    _muse_spark_setup_env()\n"
    "except Exception as _muse_spark_e:  # the platform plugin must import; the model logs the stock fallback\n"
    "    print(f'[muse-spark-open] kernel env setup failed: {_muse_spark_e}')\n"
)


def locate(root: str | None) -> Path | None:
    if root:
        p = Path(root) / "vllm_gaudi" / "__init__.py"
        return p if p.is_file() else None
    spec = importlib.util.find_spec("vllm_gaudi")
    locs = list(spec.submodule_search_locations or []) if spec else []
    return Path(locs[0]) / "__init__.py" if locs else None


def run(check: bool, root: str | None) -> int:
    path = locate(root)
    if path is None:
        print(f"{MARKER} vllm_gaudi/__init__.py not found", file=sys.stderr)
        return 2
    src = path.read_text()
    if src.startswith(HOOK):
        print(f"{MARKER} applied")
        return 0
    if MARKER in src:
        print(f"{MARKER} a different hook is present; restore the file and re-run", file=sys.stderr)
        return 2
    if check:
        print(f"{MARKER} not applied")
        return 1
    new = HOOK + src
    compile(new, str(path), "exec", dont_inherit=True)
    tmp = path.with_name(path.name + ".muse.tmp")
    tmp.write_text(new)
    os.replace(tmp, path)
    for pyc in (path.parent / "__pycache__").glob("__init__.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm_gaudi/ (default: the installed package)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
