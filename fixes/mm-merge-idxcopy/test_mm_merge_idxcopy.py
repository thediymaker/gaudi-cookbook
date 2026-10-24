# SPDX-License-Identifier: Apache-2.0
"""CPU test for mm-merge-idxcopy: the patched merge is bit-identical to masked_scatter_.

Needs torch (CPU build is enough) and a vllm-gaudi source tree: the installed vllm_gaudi, or the
directory in GAUDI_FIXES_ROOT (a site-packages-like directory holding vllm_gaudi/).
Only _merge_multimodal_embeddings is executed, with vLLM's helpers and the Habana mark_step
stubbed. Run with: pytest fixes/mm-merge-idxcopy
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _apply_module():
    spec = importlib.util.spec_from_file_location("mm_merge_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sources():
    roots = [os.environ["GAUDI_FIXES_ROOT"]] if os.environ.get("GAUDI_FIXES_ROOT") else []
    paths = [Path(r) / "vllm_gaudi" / "models" / "utils.py" for r in roots]
    if not paths:
        spec = importlib.util.find_spec("vllm_gaudi")
        paths = [Path(loc) / "models" / "utils.py" for loc in (spec.submodule_search_locations if spec else [])]
    return [p for p in paths if p.is_file()]


SOURCES = _sources()


def _function(src, mod):
    start, end = src.index(mod.DEF_ANCHOR), src.index(mod.REBIND_ANCHOR)
    ns = {
        "torch": torch,
        "NestedTensors": object,
        "_flatten_embeddings": lambda e: torch.cat(list(e)) if isinstance(e, list | tuple) else e,
        "_embedding_count_expression": lambda e: "mm",
    }
    exec(src[start:end], ns)
    return ns["_merge_multimodal_embeddings"]


@pytest.fixture(autouse=True)
def _fake_habana(monkeypatch):
    core = types.ModuleType("habana_frameworks.torch.core")
    core.mark_step = lambda: None
    pkg = types.ModuleType("habana_frameworks")
    sub = types.ModuleType("habana_frameworks.torch")
    pkg.torch, sub.core = sub, core
    for name, m in (
        ("habana_frameworks", pkg),
        ("habana_frameworks.torch", sub),
        ("habana_frameworks.torch.core", core),
    ):
        monkeypatch.setitem(sys.modules, name, m)


def _cases():
    g = torch.Generator().manual_seed(0)
    for rows, hidden, picks in ((16, 8, [1, 2, 3, 9]), (64, 32, list(range(10, 50))), (7, 4, [])):
        mask = torch.zeros(rows, dtype=torch.bool)
        mask[picks] = True
        mm = [torch.randn(len(picks), hidden, generator=g, dtype=torch.bfloat16)] if picks else []
        yield torch.randn(rows, hidden, generator=g, dtype=torch.bfloat16), mm, mask
    mask = torch.zeros(24, dtype=torch.bool)  # 3-D inputs_embeds, flat placeholder mask
    mask[3:7] = True
    mask[12:14] = True
    yield torch.randn(2, 12, 16, generator=g), [torch.randn(6, 16, generator=g)], mask


@pytest.mark.skipif(not SOURCES, reason="no vllm_gaudi source tree found")
@pytest.mark.parametrize("path", SOURCES, ids=[str(p) for p in SOURCES])
def test_index_copy_merge_matches_masked_scatter(path):
    mod = _apply_module()
    src = path.read_text()
    if mod.MARKER in src:
        pytest.skip("source already patched")
    stock, fixed = _function(src, mod), _function(mod.patch_utils(src, {}), mod)
    for embeds, mm, mask in _cases():
        want = stock(embeds.clone(), mm, mask)
        got = fixed(embeds.clone(), mm, mask)
        assert torch.equal(got, want)


@pytest.mark.skipif(not SOURCES, reason="no vllm_gaudi source tree found")
@pytest.mark.parametrize("path", SOURCES, ids=[str(p) for p in SOURCES])
def test_count_mismatch_raises_value_error(path):
    mod = _apply_module()
    src = path.read_text()
    if mod.MARKER in src:
        pytest.skip("source already patched")
    fixed = _function(mod.patch_utils(src, {}), mod)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:3] = True
    with pytest.raises(ValueError):
        fixed(torch.zeros(8, 4), [torch.ones(2, 4)], mask)
