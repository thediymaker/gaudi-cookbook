#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the MXFP4 MoE decode kernel hooks into an installed vllm-gaudi (run once at image build or container start).

    python3 install_mxfp4_tpc_moe.py           check every anchor, then install (idempotent)
    python3 install_mxfp4_tpc_moe.py --check   check the anchors only, write nothing

What it does:
  1. copies ``mxfp4_tpc_moe_rt.py`` (next to this script) to ``vllm_gaudi/ops/mxfp4_tpc_moe_rt.py``;
  2. prepends a hook to ``vllm_gaudi/__init__.py`` that calls ``setup_env()`` (every vLLM process imports the platform
     plugin before the device opens, which is when the graph compiler reads GC_KERNEL_PATH);
  3. appends a hook to ``vllm_gaudi/ops/hpu_mxfp4.py`` that calls ``install()``.

Both hooks are inert unless ``MXFP4_TPC_MOE=1`` at runtime. The script fails closed: if vllm_gaudi or any anchor it
relies on is missing, it exits 1 before writing anything. Checked against vllm-gaudi v0.26.0 and releases/v0.30.0.
"""

import argparse
import importlib.util
import os
import sys

MARK = "[mxfp4-tpc-moe]"
RT_NAME = "mxfp4_tpc_moe_rt.py"

INIT_HOOK = (
    f"# {MARK} register the MXFP4 MoE kernel library before the first HPU op (inert unless MXFP4_TPC_MOE=1)\n"
    "try:\n"
    "    from vllm_gaudi.ops.mxfp4_tpc_moe_rt import setup_env as _mxk_setup_env\n"
    "    _mxk_setup_env()\n"
    "except Exception as _mxk_e:  # never break the plugin import\n"
    f"    print(f'{MARK} setup_env failed: {{_mxk_e}}')\n"
)
OPS_HOOK = (
    f"\n# {MARK} decode-sized gpt-oss MoE calls -> custom TPC kernels (inert unless MXFP4_TPC_MOE=1)\n"
    "from vllm_gaudi.ops import mxfp4_tpc_moe_rt as _mxk_rt\n"
    "_mxk_rt.install()\n"
)

# (file relative to the vllm_gaudi package, text that must be present)
ANCHORS = [
    ("ops/hpu_mxfp4.py", "class HPUGptOssMxfp4MoEMethod"),
    ("ops/hpu_mxfp4.py", "def process_weights_after_loading"),
    ("ops/hpu_mxfp4.py", "layer.moe_op = VllmMixtureOfExpertsOpMXFP4("),
    ("ops/hpu_mxfp4.py", "mxfp4_module.GptOssMxfp4MoEMethod = HPUGptOssMxfp4MoEMethod"),
    ("extension/ops.py", "class VllmMixtureOfExpertsOpMXFP4"),
    (
        "extension/ops.py",
        "def forward(self, hidden_states, expert_routing_table, router_weights, "
        'permuted_weights=True, activation="silu"):',
    ),
    ("extension/ops.py", "self.has_bias = has_bias"),
    ("extension/ops.py", "self.alpha = alpha"),
    ("extension/ops.py", "self.limit = limit"),
    ("extension/ops.py", "self.experts_min"),
    ("__init__.py", "def register"),
]


def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    return list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None


def purge_pyc(path):
    pyc = os.path.join(os.path.dirname(path), "__pycache__")
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        for fn in os.listdir(pyc):
            if fn.startswith(base + "."):
                os.remove(os.path.join(pyc, fn))
    except OSError:
        pass


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    purge_pyc(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="check the anchors only, write nothing")
    args = ap.parse_args()

    root = pkg_dir("vllm_gaudi")
    if not root:
        sys.exit(f"{MARK} vllm_gaudi not found; refusing")
    rt_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RT_NAME)
    if not os.path.isfile(rt_src_path):
        sys.exit(f"{MARK} {rt_src_path} not found; refusing")
    rt_src = read(rt_src_path)
    compile(rt_src, rt_src_path, "exec", dont_inherit=True)

    missing = []
    for rel, need in ANCHORS:
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or need not in read(p):
            missing.append(f"{rel}: {need!r}")
    if missing:
        print(f"{MARK} vllm_gaudi at {root} does not match; refusing to patch:", file=sys.stderr)
        for m in missing:
            print(f"  missing {m}", file=sys.stderr)
        sys.exit(1)
    if args.check:
        print(f"{MARK} all {len(ANCHORS)} anchors found in {root}")
        return

    ops_py = os.path.join(root, "ops", "hpu_mxfp4.py")
    init_py = os.path.join(root, "__init__.py")
    write(os.path.join(root, "ops", RT_NAME), rt_src)
    src = read(ops_py)
    if MARK not in src:
        write(ops_py, src.rstrip("\n") + "\n" + OPS_HOOK)
    src = read(init_py)
    if MARK not in src:
        write(init_py, INIT_HOOK + src)
    # verify what was written
    for p in (ops_py, init_py):
        if MARK not in read(p):
            sys.exit(f"{MARK} hook missing in {p} after install")
    print(
        f"{MARK} installed in {root} (runtime ops/{RT_NAME}; hooks in ops/hpu_mxfp4.py and __init__.py); "
        f"MXFP4_TPC_MOE={os.environ.get('MXFP4_TPC_MOE', '0')}"
    )


if __name__ == "__main__":
    main()
