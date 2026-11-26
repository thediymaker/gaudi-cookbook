# SPDX-License-Identifier: Apache-2.0
"""Build-time patch (vllm-gaudi v0.30.0): register the port's kernel libraries before the first HPU op.

Puts a hook at the top of vllm_gaudi/__init__.py, the platform plugin that vLLM's plugin loader imports in every
process (API server, engine core, workers) before the device opens. The hook calls mistral_large4_hpu.mxk.setup_env()
(mxk.py has what it sets and why; ML4_MXK=0 makes it a no-op), which also chains the decode-step library (k1ix.py).
The hook does nothing when the plugin is not installed, and never breaks the platform plugin import.

    python3 kernel_env_hook.py [--root DIR] [--check]

--root: the directory holding vllm_gaudi/ (default: the installed package). Exit codes follow the fix contract
(fixes/README.md): 0 applied (or already applied), 1 not applied (--check), 2 vllm_gaudi/__init__.py not found.
Idempotent (keeps one hook); purges the stale __pycache__ entry.
"""

import argparse
import importlib.util
import os
import sys

MARK = "[ml4-mxk]"
HOOK = (
    "# [ml4-mxk] register the ML4 MXFP4 kernel libraries (mistral_large4_hpu.mxk, ML4_MXK) before the first HPU op\n"
    "try:\n"
    "    import importlib.util as _ml4_mxk_u\n"
    "    if _ml4_mxk_u.find_spec('mistral_large4_hpu.mxk') is not None:\n"
    "        from mistral_large4_hpu.mxk import setup_env as _ml4_mxk_setup_env\n"
    "        _ml4_mxk_setup_env()\n"
    "except Exception as _ml4_mxk_e:  # never break the platform plugin import\n"
    "    print(f'[ml4-mxk] setup_env failed: {_ml4_mxk_e}')\n"
)


def init_path(root):
    if root:
        return os.path.join(root, "vllm_gaudi", "__init__.py")
    spec = importlib.util.find_spec("vllm_gaudi")
    if not spec or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "__init__.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    init_py = init_path(a.root)
    if not init_py or not os.path.isfile(init_py):
        print(f"{MARK} vllm_gaudi/__init__.py NOT FOUND", file=sys.stderr)
        return 2
    with open(init_py) as f:
        src = f.read()
    if a.check:
        ok = src.startswith(HOOK)
        print(f"{MARK} hook {'present' if ok else 'MISSING'} in {init_py}")
        return 0 if ok else 1
    if MARK in src:
        print(f"{MARK} hook already in {init_py}")
        return 0
    with open(init_py, "w") as f:
        f.write(HOOK + src)
    pyc = os.path.join(os.path.dirname(init_py), "__pycache__")
    try:
        for fn in os.listdir(pyc):
            if fn.startswith("__init__."):
                os.remove(os.path.join(pyc, fn))
    except OSError:
        pass
    print(f"{MARK} hook added to {init_py}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
