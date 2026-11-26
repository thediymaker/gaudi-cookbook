#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build gates of the mistral-large-4 port (install.sh runs them last).

    python3 tests/build_gates.py [--kernels]   --kernels also checks the two kernel libraries and the env hook

Every gate prints one line; any failure exits 1. No HPU, network or checkpoint is needed: the AMD checkpoint's
params.json (gzip) and its config.json without the quantization block are fixtures next to this file
(tests/fixtures/; sha256 of the originals in fixtures/README).

Gates:
  entry_point    mistral_large4_hpu:register is in importlib.metadata group vllm.general_plugins
  mistral_common 1.13.0, ReasoningEffort has low / medium / high, and the grammar API the vendored parser
                 calls (GrammarFactory.get_grammar_for_chat_request)
  register       load_general_plugins() registered the architecture (lazy path), quant mistral_large4_hpu_mxfp4
                 (also in the HPU platform list) and the text-only wrapper on the params.json adapter
  imports        every module of the package imports (the model module is required unless --frontend-only);
                 none imports the PR's nvidia/amd model code
  pr_files       the five vendored PR files are installed byte-exact over v0.30.0 (install_pr_files.py --check)
                 and carry the PR's additions (dsa adapter branch, response_format rendering, the
                 mistral-common request grammar)
  config         params.json -> text-only MistralLarge4ForCausalLM: 63 KV owners (9 readers of layer 35), 44/28
                 indexers, 18 NoPE layers; control: without the text-only filter the config.json keys leak
                 text_config / vision_config onto the text config
  quant          vLLM's _verify_quantization turns the checkpoint's `quark` into mistral_large4_hpu_mxfp4 (404
                 FP8-block overrides); controls: an override on a routed expert, or an FP8 global format, is not
                 claimed
  model_class    the architecture resolves to mistral_large4_hpu.vllm_model's class
  kernels        (--kernels) the ml4 MXFP4 MoE library and the decode-step library are where the plugin reads them,
                 with their version strings and op libraries, and vllm_gaudi/__init__.py carries the env hook
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import importlib
import importlib.metadata
import importlib.util
import json
import pathlib
import pkgutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
ENTRY_POINT = "mistral_large4_hpu:register"
ARCH = "MistralLarge4ForCausalLM"
MODEL_MODULE, MODEL_CLASS = "mistral_large4_hpu.vllm_model", "ML4ForCausalLM"
QUANT = "mistral_large4_hpu_mxfp4"
FORBIDDEN = ("vllm.models.mistral_large4", "vllm.models.mistral.nvidia", "vllm.models.mistral.amd")

results: list[tuple[str, bool, str]] = []


def gate(name):
    def wrap(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k) or "ok"
                results.append((name, True, detail))
            except Exception as e:  # noqa: BLE001  every failure is reported, then exit 1
                results.append((name, False, f"{type(e).__name__}: {e}"))

        return run

    return wrap


def _params() -> dict:
    return json.loads(gzip.decompress((FIXTURES / "amd_params.json.gz").read_bytes()))


def _defaults() -> dict:
    return json.loads((FIXTURES / "amd_config_noquant.json").read_text())


@gate("entry_point")
def g_entry_point():
    eps = [e.value for e in importlib.metadata.entry_points(group="vllm.general_plugins")]
    assert ENTRY_POINT in eps, eps
    return f"{ENTRY_POINT} installed"


@gate("mistral_common")
def g_mistral_common():
    import mistral_common
    from mistral_common.guidance.grammar_factory import GrammarFactory
    from mistral_common.protocol.instruct.request import ReasoningEffort

    values = [e.value for e in ReasoningEffort]
    assert mistral_common.__version__ == "1.13.0", mistral_common.__version__
    assert {"low", "medium", "high"} <= set(values), values
    assert hasattr(GrammarFactory, "get_grammar_for_chat_request")
    return f"{mistral_common.__version__}, ReasoningEffort {values}"


@gate("register")
def g_register():
    import vllm.plugins

    vllm.plugins.load_general_plugins()
    import vllm.transformers_utils.configs.mistral as M
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.platforms import current_platform

    m = ModelRegistry.models[ARCH]
    assert (getattr(m, "module_name", None), getattr(m, "class_name", None)) == (MODEL_MODULE, MODEL_CLASS), m
    assert QUANT in QUANTIZATION_METHODS and QUANT in current_platform.supported_quantization
    assert getattr(M.adapt_config_dict, "_mistral_large4_hpu_text_only", False), "text-only wrapper not installed"
    return f"arch -> {MODEL_MODULE}:{MODEL_CLASS} (lazy), quant, text-only adapter"


@gate("imports")
def g_imports(frontend_only: bool):
    import mistral_large4_hpu

    pkg_dir = pathlib.Path(mistral_large4_hpu.__file__).parent
    names = sorted(m.name for m in pkgutil.walk_packages([str(pkg_dir)], prefix="mistral_large4_hpu."))
    if not frontend_only:
        assert MODEL_MODULE in names, f"{MODEL_MODULE} missing from the package"
    for name in names:
        importlib.import_module(name)
    bad_src = []
    for path in pkg_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            bad_src += [f"{path.name}: {m}" for m in mods if m.startswith(FORBIDDEN)]
    assert not bad_src, bad_src
    loaded = [m for m in sys.modules if m.startswith(FORBIDDEN)]
    assert not loaded, loaded
    return f"{len(names)} modules import; no nvidia/amd model imports"


@gate("pr_files")
def g_pr_files():
    import vllm

    root = pathlib.Path(vllm.__file__).resolve().parent.parent
    r = subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "patches" / "install_pr_files.py"),
            "--check",
            str(HERE.parent / "vendor"),
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout + r.stderr).strip()
    from vllm.parser import mistral as P
    from vllm.renderers import mistral as R
    from vllm.tokenizers import mistral as T
    from vllm.transformers_utils.configs import mistral as C

    assert hasattr(C, "_remap_mistral_large_4_args") and hasattr(C, "get_mistral_large_4_dspark_config")
    assert hasattr(T, "to_mistral_response_format") and hasattr(R, "_get_chat_template_kwargs")
    assert hasattr(P, "_mistral_response_format") and not hasattr(P, "_DEFAULT_JSON_SCHEMA")
    return r.stdout.strip().splitlines()[-1]


@gate("config")
def g_config():
    import vllm.transformers_utils.configs.mistral as M

    from mistral_large4_hpu import config as C

    params, defaults = _params(), _defaults()
    hf = M.adapt_config_dict(copy.deepcopy(params), defaults=copy.deepcopy(defaults))
    assert hf.architectures == [ARCH] and hf.model_type == "mistral_large_4", (hf.architectures, hf.model_type)
    leaked = [k for k in ("text_config", "vision_config", "image_token_index") if getattr(hf, k, None) is not None]
    assert not leaked, f"multimodal keys on the text config: {leaked}"
    s = C.layer_tables(hf).summary()
    want = {
        "layers": 72,
        "kv_owners": 63,
        "kv_shared": 9,
        "kv_shared_sources": [35],
        "index_full": 44,
        "index_shared": 28,
        "nope": 18,
        "indexer_nope": 18,
        "max_readers": 10,
    }
    assert s == want, s
    # control: the stock adapter on the same params without vision_encoder but with every config.json default
    p = {k: v for k, v in copy.deepcopy(params).items() if k != "vision_encoder"}
    raw = M.adapt_config_dict.__wrapped__(p, defaults=copy.deepcopy(defaults))
    assert getattr(raw, "text_config", None) is not None, "control: config.json keys did not leak without the filter"
    return "63 KV owners (9 readers of 35), 44/28 indexers, 18 NoPE; control: unfiltered defaults leak text_config"


@gate("quant")
def g_quant():
    from types import SimpleNamespace

    import vllm.transformers_utils.configs.mistral as M
    from vllm.config.model import ModelConfig

    hf = M.adapt_config_dict(_params(), defaults=_defaults())
    q = hf.quantization_config

    def verify(qcfg):
        stub = SimpleNamespace(
            quantization=None,
            hf_config=hf,
            allow_deprecated_quantization=False,
            model_arch_config=SimpleNamespace(quantization_config=qcfg),
        )
        ModelConfig._verify_quantization(stub)
        return stub.quantization

    assert len(q["layer_quant_config"]) == 404, len(q["layer_quant_config"])
    assert verify(copy.deepcopy(q)) == QUANT
    controls = []
    for what, mutate in (
        (
            "expert override",
            lambda c: c["layer_quant_config"].__setitem__(
                "layers.5.feed_forward.experts.3.w1", copy.deepcopy(c["layer_quant_config"]["layers.5.attention.wq"])
            ),
        ),
        ("fp8 global", lambda c: c["global_quant_config"]["weight"].__setitem__("dtype", "fp8_e4m3")),
    ):
        bad = copy.deepcopy(q)
        mutate(bad)
        try:
            claimed = verify(bad)
        except ValueError:
            claimed = None
        assert claimed != QUANT, f"control: {what} was claimed"
        controls.append(what)
    return f"quark -> {QUANT} (404 FP8-block overrides); not claimed: {', '.join(controls)}"


@gate("model_class")
def g_model_class():
    from vllm import ModelRegistry

    cls = ModelRegistry._try_load_model_cls(ARCH)
    assert cls is not None and cls.__name__ == MODEL_CLASS and cls.__module__ == MODEL_MODULE, cls
    return f"{cls.__module__}.{cls.__name__}"


@gate("kernels")
def g_kernels():
    import os

    from mistral_large4_hpu import k1ix, mxk

    ver = mxk.lib_version()
    assert ver and ver.startswith("mxk_tpc ml4-v8-"), f"no ml4 kernel library in {mxk.LIB_DIR} ({ver})"
    assert os.path.isfile(mxk.op_lib()), mxk.op_lib()
    with open(os.path.join(mxk.LIB_DIR, "MANIFEST")) as f:
        assert "variant: ml4\n" in f.read(), "MANIFEST does not name variant ml4"
    kv = k1ix.lib_version()
    assert kv and kv.startswith("k1_tpc k1-"), f"no decode-step kernel library in {k1ix.LIB_DIR} ({kv})"
    assert os.path.isfile(k1ix.op_lib()), k1ix.op_lib()
    r = subprocess.run(
        [sys.executable, str(HERE.parent / "patches" / "kernel_env_hook.py"), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, (r.stdout + r.stderr).strip()
    return f"{ver.split()[-1]} in {mxk.LIB_DIR}, {kv.split()[-1]} in {k1ix.LIB_DIR}, env hook present"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernels", action="store_true", help="also check the kernel libraries and the env hook")
    args = ap.parse_args()

    g_entry_point()
    g_mistral_common()
    g_register()
    g_imports(False)
    g_pr_files()
    g_config()
    g_quant()
    g_model_class()
    if args.kernels:
        g_kernels()

    for name, ok, detail in results:
        print(f"[mistral_large4_hpu build_gates] {'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(
        f"[mistral_large4_hpu build_gates] {len(results) - len(failed)}/{len(results)} passed"
        + (f"; FAILED: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
