# SPDX-License-Identifier: Apache-2.0
"""CPU tests for moe-router-weight-on-input.

The helper test needs torch (CPU build). The source test needs GAUDI_FIXES_ROOT, a directory
holding vllm_gaudi/, or an installed vllm_gaudi. Run with: pytest fixes/moe-router-weight-on-input
"""

import importlib.util
import os
import shutil
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("moe_rwi_apply", HERE / "apply.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


def _helper(torch):
    ns = {"torch": torch}
    exec(fix.HELPER, ns)
    return ns["_gaudi_fix_router_weight_on_input"]


def _expert(torch, x, w13, w2):
    a, b = (x @ w13).chunk(2, dim=-1)
    return (torch.nn.functional.silu(a) * b) @ w2


def test_weight_on_input_matches_reference():
    torch = pytest.importorskip("torch")
    helper = _helper(torch)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 16, generator=g)
    w = torch.rand(5, 1, generator=g)
    w13, w2 = torch.randn(16, 64, generator=g), torch.randn(32, 16, generator=g)
    reference = _expert(torch, x * w, w13, w2)

    def op(x, weights):  # the HPU op applies the router weight to the expert output
        return weights * _expert(torch, x, w13, w2)

    layer = types.SimpleNamespace(apply_router_weight_on_input=True)
    x2, w2_ = helper(layer, x, w)
    assert torch.allclose(op(x2, w2_), reference, atol=1e-5)
    assert not torch.allclose(op(x, w), reference, atol=1e-3)


def test_other_layers_unchanged_and_top_k_checked():
    torch = pytest.importorskip("torch")
    helper = _helper(torch)
    x, w = torch.randn(3, 4), torch.rand(3, 2)
    out_x, out_w = helper(types.SimpleNamespace(), x, w)
    assert out_x is x and out_w is w
    with pytest.raises(ValueError):
        helper(types.SimpleNamespace(apply_router_weight_on_input=True), x, w)


def _source_root():
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if root:
        return Path(root)
    s = importlib.util.find_spec("vllm_gaudi")
    locs = list(s.submodule_search_locations or []) if s else []
    return Path(locs[0]).parent if locs else None


def test_patches_every_site(tmp_path):
    root = _source_root()
    if root is None or not (root / "vllm_gaudi" / "ops" / "hpu_fused_moe.py").is_file():
        pytest.skip("no vllm_gaudi source (set GAUDI_FIXES_ROOT)")
    for name in ("hpu_fused_moe.py", "hpu_compressed_tensors.py"):
        dst = tmp_path / "vllm_gaudi" / "ops" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "vllm_gaudi" / "ops" / name, dst)
    if fix.MARKER in (tmp_path / "vllm_gaudi" / "ops" / "hpu_fused_moe.py").read_text():
        pytest.skip("source is already patched")
    assert fix.run(True, str(tmp_path)) == 1
    assert fix.run(False, str(tmp_path)) == 0
    hook = "x, topk_weights = _gaudi_fix_router_weight_on_input(layer, x, topk_weights)"
    for name in ("hpu_fused_moe.py", "hpu_compressed_tensors.py"):
        assert (tmp_path / "vllm_gaudi" / "ops" / name).read_text().count(hook) == 2
    assert fix.run(True, str(tmp_path)) == 0
