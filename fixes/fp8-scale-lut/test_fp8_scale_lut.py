# SPDX-License-Identifier: Apache-2.0
"""Tests for fp8-scale-lut.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift.
2. The table helper (needs torch on CPU): bit-identical to the stock halving for all 256 byte values and on a
   random non-contiguous tensor, and the result keeps the input's shape.

Run with: pytest fixes/fp8-scale-lut
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = "extension/ops.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "fp8_scale_lut_apply")


@pytest.fixture
def site(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/ and vllm_gaudi/ of a release")
    for pkg, rel, _, _ in fix_mod.TARGETS:
        dst = tmp_path / pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(root) / pkg / rel, dst)
    return tmp_path


def _run(root, *extra):
    cmd = [sys.executable, str(HERE / "apply.py"), "--root", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode


def test_source_patch(site):
    path = site / "vllm_gaudi" / REL
    assert _run(site, "--check") == 1
    assert _run(site) == 0
    text = path.read_text()
    assert fix_mod.MARKER in text and "(loaded_weight.float() * 0.5)" not in text
    compile(text, REL, "exec")
    assert _run(site) == 0 and path.read_text() == text
    assert _run(site, "--check") == 0


def test_source_patch_refuses_drift(site):
    path = site / "vllm_gaudi" / REL
    path.write_text(path.read_text().replace("* 0.5).to(torch.float8_e4m3fn)", "* 0.25).to(torch.float8_e4m3fn)", 1))
    before = path.read_text()
    assert _run(site) == 2 and path.read_text() == before


def test_bit_identical():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch has no float8_e4m3fn")
    ns = {"torch": torch}
    exec(fix_mod.HELPER, ns)
    half = ns["_fp8_half_lut"]

    def stock(w):
        return (w.float() * 0.5).to(torch.float8_e4m3fn)

    every = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    assert torch.equal(half(every).view(torch.uint8), stock(every).view(torch.uint8))
    g = torch.Generator().manual_seed(0)
    w = torch.randint(0, 256, (64, 96), dtype=torch.int32, generator=g).to(torch.uint8).view(torch.float8_e4m3fn).t()
    out = half(w)
    assert out.shape == w.shape and torch.equal(out.view(torch.uint8), stock(w).view(torch.uint8))
