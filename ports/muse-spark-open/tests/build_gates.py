#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install-time gates for the Muse Spark Open port: run by install.sh after every step, no device or checkpoint.

Each gate prints one PASS or FAIL line; any failure exits 1.

  patches       every port patch reports "applied" (--check exits 0)
  entry_point   muse_spark_hpu:register is installed in the vllm.general_plugins group
  register      load_general_plugins() registered the architecture (by module path), the four config classes,
                parsers muse / musespark1x and the quant method muse_spark_hpu_mxfp4 (also for the HPU platform)
  imports       every module of the package imports; none imports the PR's NVIDIA or AMD model code
  parsers       both parser names construct
  quant         vLLM's _verify_quantization turns the checkpoint's Quark MXFP4 block into muse_spark_hpu_mxfp4
                for model type muse_spark; a non-MXFP4 Quark config is not claimed
  model_class   the architecture resolves to muse_spark_hpu.model's class
  runner_window the sliding-window masks of the patched runner (tests/test_runner_window.py)
  kernels       with --kernels: the op library registers both ops, the kernel library carries a version and
                covers the served layer, and the vllm_gaudi hook sets the environment only when the switch is on
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import importlib.metadata
import json
import os
import pathlib
import pkgutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PORT = HERE.parent
ARCH = "MuseSparkForConditionalGeneration"
MODEL_MODULE, MODEL_CLASS = "muse_spark_hpu.model", "MuseSparkHPUForConditionalGeneration"
QUANT = "muse_spark_hpu_mxfp4"
FORBIDDEN = ("vllm.models.muse_spark.nvidia", "vllm.models.muse_spark.amd")
PATCHES = ("vllm_pr_61879.py", "runner_window.py", "swa_kv_hooks.py", "kernel_env_hook.py")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"

# The quantization block of amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 config.json, without its exclude list.
QUARK_MXFP4 = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"},
        "input_tensors": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"},
    },
    "export": {"weight_format": "real_quantized"},
}

results: list[tuple[str, bool, str]] = []


def gate(name, fn, *args):
    try:
        results.append((name, True, fn(*args) or "ok"))
    except Exception as e:  # every failure is reported, then exit 1
        results.append((name, False, f"{type(e).__name__}: {e}"))


def g_patches():
    for p in PATCHES:
        r = subprocess.run([sys.executable, str(PORT / "patches" / p), "--check"], capture_output=True, text=True)
        assert r.returncode == 0, f"{p}: {(r.stdout + r.stderr).strip()}"
    return f"{len(PATCHES)} applied"


def g_entry_point():
    eps = [e.value for e in importlib.metadata.entry_points(group="vllm.general_plugins")]
    assert "muse_spark_hpu:register" in eps, eps


def g_register():
    import vllm.plugins
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.platforms import current_platform
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    vllm.plugins.load_general_plugins()
    m = ModelRegistry.models[ARCH]
    assert (m.module_name, m.class_name) == (MODEL_MODULE, MODEL_CLASS), m
    for k in ("muse_spark", "muse_spark_text", "muse_spark_vision", "muse_spark_audio"):
        assert k in _CONFIG_REGISTRY, k
    for name in ("muse", "musespark1x"):
        assert ToolParserManager.lazy_parsers[name] == ("muse_spark_hpu.tool_parser", "MuseSparkHPUToolParser")
        assert ReasoningParserManager.lazy_parsers[name][0] == "vllm.reasoning.muse_reasoning_parser"
    assert QUANT in QUANTIZATION_METHODS and QUANT in current_platform.supported_quantization
    return "architecture, 4 configs, 2 parser names, quant method"


def g_imports():
    import muse_spark_hpu

    pkg = pathlib.Path(muse_spark_hpu.__file__).parent
    names = sorted(m.name for m in pkgutil.walk_packages([str(pkg)], prefix="muse_spark_hpu."))
    assert MODEL_MODULE in names, names
    for name in names:
        importlib.import_module(name)
    bad = []
    for path in pkg.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            bad += [f"{path.name}: {m}" for m in mods if m.startswith(FORBIDDEN)]
    assert not bad and not [m for m in sys.modules if m.startswith(FORBIDDEN)], bad
    return f"{len(names)} modules"


def g_parsers():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    for name in ("muse", "musespark1x"):
        tp = ToolParserManager.get_tool_parser(name)(object())
        rp = ReasoningParserManager.get_reasoning_parser(name)(object())
        assert type(tp).__name__ == "MuseSparkHPUToolParser" and type(rp).__name__ == "MuseReasoningParser"


def g_quant():
    from types import SimpleNamespace

    from vllm.config.model import ModelConfig

    hf = SimpleNamespace(model_type="muse_spark")

    def verify(qcfg):
        stub = SimpleNamespace(
            quantization=None,
            hf_config=hf,
            allow_deprecated_quantization=False,
            model_arch_config=SimpleNamespace(quantization_config=qcfg),
        )
        ModelConfig._verify_quantization(stub)
        return stub.quantization

    assert verify(copy.deepcopy(QUARK_MXFP4)) == QUANT
    other = copy.deepcopy(QUARK_MXFP4)
    other["global_quant_config"]["weight"]["dtype"] = "fp8_e4m3"
    try:
        claimed = verify(other)
    except ValueError:
        claimed = None
    assert claimed != QUANT, "a non-MXFP4 Quark config was claimed"
    return f"quark MXFP4 -> {QUANT}; other Quark exports not claimed"


def g_model_class():
    from vllm import ModelRegistry

    cls = ModelRegistry._try_load_model_cls(ARCH)
    assert cls is not None and (cls.__module__, cls.__name__) == (MODEL_MODULE, MODEL_CLASS), cls


def g_runner_window():
    r = subprocess.run([sys.executable, str(HERE / "test_runner_window.py")], capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    return tail[0]


PROBE = (
    "import os, json, vllm_gaudi; print(json.dumps({k: os.environ.get(k) for k in "
    "('GC_KERNEL_PATH', 'MXK_TPC_FWD_LIB', 'PT_HPU_LAZY_ACC_PAR_MODE', 'PT_HPU_RECIPE_CACHE_CONFIG')}))"
)


def _probe(env, code=PROBE):
    keep = ("MUSE_SPARK_MXFP4_KERNELS", "MXK_TPC_FWD_LIB", "PT_HPU_LAZY_ACC_PAR_MODE", "PT_HPU_RECIPE_CACHE_CONFIG")
    e = {k: v for k, v in os.environ.items() if k not in keep}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    assert out.returncode == 0 and lines, (out.stderr or out.stdout)[-800:]
    return json.loads(lines[-1])


def g_kernels():
    from muse_spark_hpu import mxfp4_kernels as mxk

    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch, habana_frameworks.torch; torch.ops.load_library(sys.argv[1]); "
            "torch.ops.custom_op.mxk_w13x.default; torch.ops.custom_op.mxk_w2xr.default",
            mxk.op_lib(),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"op library: {r.stderr[-600:]}"
    ver = mxk.lib_version()
    assert ver, f"no version string in {mxk.tpc_lib()}"
    why = mxk.unsupported(128, 4096, 1024, 8)  # EP2 x TP4: 128 local experts, Hm 4096, Is 1024, top-8
    assert why is None, why
    with tempfile.TemporaryDirectory() as td:
        rc = f"{td}/rc/,False,1024"
        base = {"GC_KERNEL_PATH": STOCK, "PT_HPU_RECIPE_CACHE_CONFIG": rc}
        off = _probe(base)
        want_off = {
            "GC_KERNEL_PATH": STOCK,
            "MXK_TPC_FWD_LIB": None,
            "PT_HPU_LAZY_ACC_PAR_MODE": None,
            "PT_HPU_RECIPE_CACHE_CONFIG": rc,
        }
        assert off == want_off, f"switch off changed the environment: {off}"
        on = _probe(dict(base, MUSE_SPARK_MXFP4_KERNELS="1"))
        tag = "mxk-" + ver.split()[-1]
        want_on = {
            "GC_KERNEL_PATH": mxk.tpc_lib(),
            "MXK_TPC_FWD_LIB": STOCK,
            "PT_HPU_LAZY_ACC_PAR_MODE": "0",
            "PT_HPU_RECIPE_CACHE_CONFIG": f"{td}/rc_{tag}/,False,1024",
        }
        assert on == want_on and os.path.isdir(f"{td}/rc_{tag}"), f"switch on: {on}"
        missing = _probe(
            dict(base, MUSE_SPARK_MXFP4_KERNELS="1"),
            "import os, json; from muse_spark_hpu import mxfp4_kernels as m; m.LIB_DIR = '/nonexistent'; "
            "m.setup_env(); print(json.dumps({k: os.environ.get(k) for k in ('GC_KERNEL_PATH', 'MXK_TPC_FWD_LIB', "
            "'PT_HPU_LAZY_ACC_PAR_MODE', 'PT_HPU_RECIPE_CACHE_CONFIG')}))",
        )
        assert missing == want_off, f"switch on without libraries changed the environment: {missing}"
    return ver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernels", action="store_true", help="also check the MXFP4 kernel build and its env hook")
    args = ap.parse_args()
    gate("patches", g_patches)
    gate("entry_point", g_entry_point)
    gate("register", g_register)
    gate("imports", g_imports)
    gate("parsers", g_parsers)
    gate("quant", g_quant)
    gate("model_class", g_model_class)
    gate("runner_window", g_runner_window)
    if args.kernels:
        gate("kernels", g_kernels)
    for name, ok, detail in results:
        print(f"[muse-spark-open build gate] {'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"[muse-spark-open build gate] {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
