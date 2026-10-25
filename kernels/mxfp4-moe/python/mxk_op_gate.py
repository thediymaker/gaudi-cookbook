#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Op-library gate for one kernel build (no Gaudi device needed); build.sh runs it last.

    mxk_op_gate.py --lib DIR --guids mxk_w13x,mxk_w2x,mxk_deq [--op-lib libmxk_op_pt2.so]

Checks:

* the version string in ``libmxk_tpc.so`` is the one in ``MANIFEST``;
* the op library (``libmxk_op_lazy.so``, or ``libmxk_op_pt2.so`` of a ``--pt2`` build, loaded with
  ``PT_HPU_LAZY_MODE=0``) loads into torch with the habana plugin (in a subprocess);
* it registers the op of every GUID of the build (``mxk_w13xi`` and ``mxk_w13xt`` are reached
  through ``mxk_w13x``), and no MXFP4 MoE op
  of another variant. Lazy-mode legacy custom ops are cached by input signature, not by GUID, so a process must
  never hold two ops that could see the same input shapes. One variant per op library makes that structural.

Exit 0 = pass.
"""

import argparse
import os
import subprocess
import sys

# every MoE op any variant registers
KNOWN_OPS = {"mxk_w13x", "mxk_w13xi", "mxk_w13xt", "mxk_w13xib", "mxk_w2x", "mxk_w2xr", "mxk_w2xb", "mxk_deq"}


def expected_ops(guids):
    ops = set(guids)
    if ops & {"mxk_w13xi", "mxk_w13xt"}:  # reached through the mxk_w13x op
        ops.add("mxk_w13x")
    return ops


def lib_version(path):
    with open(path, "rb") as f:
        data = f.read()
    i = data.find(b"mxk_tpc ")
    return data[i : data.index(b"\0", i)].decode() if i >= 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--guids", required=True, help="comma-separated GUIDs of the build")
    ap.add_argument("--op-lib", default="libmxk_op_lazy.so", choices=["libmxk_op_lazy.so", "libmxk_op_pt2.so"])
    a = ap.parse_args()
    fails = []

    def check(name, ok, detail=""):
        print(f"[mxk-op-gate] {'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)
        if not ok:
            fails.append(name)

    ver = lib_version(os.path.join(a.lib, "libmxk_tpc.so"))
    with open(os.path.join(a.lib, "MANIFEST")) as f:
        manifest = f.read()
    check("version", bool(ver) and f"version: {ver.split()[-1]}\n" in manifest, str(ver))

    want = expected_ops([g for g in a.guids.split(",") if g])
    probe = sorted(want | KNOWN_OPS)
    code = (
        "import sys, torch, habana_frameworks.torch.core; torch.ops.load_library(sys.argv[1]); "
        "co = torch.ops.custom_op; print('HAS ' + ' '.join(n for n in sys.argv[2:] if hasattr(co, n)))"
    )
    env = dict(os.environ)
    if a.op_lib == "libmxk_op_pt2.so":
        env["PT_HPU_LAZY_MODE"] = "0"
    r = subprocess.run(
        [sys.executable, "-c", code, os.path.join(a.lib, a.op_lib), *probe],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    has = next((set(ln.split()[1:]) for ln in r.stdout.splitlines() if ln.startswith("HAS")), None)
    check(f"{a.op_lib} loads", r.returncode == 0 and has is not None, (r.stderr or "")[-400:] if r.returncode else "")
    if has is not None:
        check("ops registered", want <= has, " ".join(sorted(want)))
        other = (has & KNOWN_OPS) - want
        check("no op of another variant", not other, " ".join(sorted(other)))
    print(f"[mxk-op-gate] {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
