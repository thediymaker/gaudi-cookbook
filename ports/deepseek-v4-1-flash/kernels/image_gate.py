#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Image gate for the port's kernels (DSV41_MXK, dsv41_hpu/mxk.py). Runs at the end of install.sh; no device needed.

1. The op library loads into torch with the Habana plugin and registers custom_op::mxk_w13x, mxk_w2x and
   dsv41_sinkhorn (the mHC Sinkhorn kernel, DSV41_HC_SINKHORN=kernel).
2. The kernel library is a dsv41 build (its version string) and covers the served layer: 48 local experts at EP 8,
   hidden size 5120, expert width 2304, top-6, 16 tokens.
3. The vllm_gaudi/__init__.py hook (patches/dsv41-kernel-env), each case in a fresh process:
   - flag off: GC_KERNEL_PATH, PT_HPU_RECIPE_CACHE_CONFIG and PT_HPU_LAZY_ACC_PAR_MODE are left exactly as given;
   - flag on: GC_KERNEL_PATH = the library, MXK_TPC_FWD_LIB = the previous single path, PT_HPU_LAZY_ACC_PAR_MODE=0,
     and the recipe cache moved to a sibling directory tagged with the kernel version (created);
   - flag on with the libraries missing: everything left as given (the stock path).

Prints one PASS or FAIL line per check; exit 1 on any FAIL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

LIB = os.environ.get("DSV41_MXK_LIB", "/opt/mxfp4-moe/lib")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
HOOK_ENV = ("GC_KERNEL_PATH", "MXK_TPC_FWD_LIB", "PT_HPU_LAZY_ACC_PAR_MODE", "PT_HPU_RECIPE_CACHE_CONFIG")
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[dsv41 kernel gate] {'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)
    if not ok:
        fails.append(name)


PROBE = f"import os, json; import vllm_gaudi; print(json.dumps({{k: os.environ.get(k) for k in {HOOK_ENV!r}}}))"


def probe(env: dict[str, str]) -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("DSV41_MXK", *HOOK_ENV[1:])}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", PROBE], env=e, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode or not lines:
        return {"error": (out.stderr or out.stdout)[-800:]}
    return json.loads(lines[-1])


def main() -> int:
    # 1. op library
    code = (
        "import sys, torch, habana_frameworks.torch; torch.ops.load_library(sys.argv[1]); "
        "[print(getattr(torch.ops.custom_op, n).default._schema) for n in ('mxk_w13x', 'mxk_w2x', 'dsv41_sinkhorn')]"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, os.path.join(LIB, "libmxk_op_lazy.so")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    ok = r.returncode == 0 and all(f"{n}(" in r.stdout for n in ("mxk_w13x", "mxk_w2x", "dsv41_sinkhorn"))
    check(
        "op library loads and registers mxk_w13x, mxk_w2x, dsv41_sinkhorn",
        ok,
        r.stdout.strip().replace("\n", " | ") if r.returncode == 0 else r.stderr[-600:],
    )
    # 2. kernel library version and coverage
    os.environ["DSV41_MXK_LIB"] = LIB
    from dsv41_hpu import mxk

    ver = mxk.lib_version(os.path.join(LIB, "libmxk_tpc.so"))
    check("kernel library is a dsv41 build", bool(ver and ver.startswith("mxk_tpc dsv41-")), str(ver))
    why = mxk.supported(48, 5120, 2304, 6)
    check("served layer covered (48 experts, D 5120, I 2304, top-6, 16 tokens)", why is None, str(why or ""))
    # 3. environment hook
    with tempfile.TemporaryDirectory() as td:
        rc = f"{td}/rc/,False,1024"
        base = {"GC_KERNEL_PATH": STOCK, "PT_HPU_RECIPE_CACHE_CONFIG": rc}
        off = probe(base)
        want_off = {"GC_KERNEL_PATH": STOCK, "MXK_TPC_FWD_LIB": None, "PT_HPU_LAZY_ACC_PAR_MODE": None}
        check("flag off: environment untouched", off == {**want_off, "PT_HPU_RECIPE_CACHE_CONFIG": rc}, json.dumps(off))
        on = probe(dict(base, DSV41_MXK="1", DSV41_MXK_LIB=LIB))
        tag = "mxk-" + (ver or "?").split()[-1]
        want_on = {
            "GC_KERNEL_PATH": os.path.join(LIB, "libmxk_tpc.so"),
            "MXK_TPC_FWD_LIB": STOCK,
            "PT_HPU_LAZY_ACC_PAR_MODE": "0",
            "PT_HPU_RECIPE_CACHE_CONFIG": f"{td}/rc_{tag}/,False,1024",
        }
        check(
            "flag on: library registered, ACC_PAR_MODE=0, recipe cache tagged",
            on == want_on and os.path.isdir(f"{td}/rc_{tag}"),
            json.dumps(on),
        )
        miss = probe(dict(base, DSV41_MXK="1", DSV41_MXK_LIB=f"{td}/nolib"))
        check("flag on, libraries missing: environment untouched", miss == off, json.dumps(miss))
    print(f"[dsv41 kernel gate] {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
