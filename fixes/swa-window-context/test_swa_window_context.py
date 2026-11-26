# SPDX-License-Identifier: Apache-2.0
"""Tests for swa-window-context.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift, leaves no
   gemma4 condition on the two window sites, and also applies after the muse-glimmer-30b port widened them.
2. The patch function on a minimal text holding both sites (stdlib only).

Run with: pytest fixes/swa-window-context
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = "v1/worker/hpu_model_runner.py"
MUSE_PATCH = HERE.parent.parent / "ports" / "muse-glimmer-30b" / "patches" / "swa_window_blocks.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "swa_window_context_apply")


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


def _run(script, root, *extra):
    cmd = [sys.executable, str(script), "--root", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode


def test_source_patch(site):
    path = site / "vllm_gaudi" / REL
    assert _run(HERE / "apply.py", site, "--check") == 1
    assert _run(HERE / "apply.py", site) == 0
    text = path.read_text()
    assert text.count(fix_mod.MARKER) == 2
    for gate in fix_mod.GATES:
        assert gate + ":" not in text and gate + " and target_blocks > 0:" not in text
    compile(text, REL, "exec")
    assert _run(HERE / "apply.py", site) == 0 and path.read_text() == text
    assert _run(HERE / "apply.py", site, "--check") == 0


def test_source_patch_refuses_drift(site):
    path = site / "vllm_gaudi" / REL
    path.write_text(path.read_text().replace("            # Keep one extra block", "            # changed", 1))
    before = path.read_text()
    assert _run(HERE / "apply.py", site) == 2 and path.read_text() == before


@pytest.mark.parametrize("fix_first", [False, True])
def test_with_muse_glimmer_port(site, fix_first):
    path = site / "vllm_gaudi" / REL
    order = [HERE / "apply.py", MUSE_PATCH] if fix_first else [MUSE_PATCH, HERE / "apply.py"]
    for script in order:
        assert _run(script, site) == 0
    text = path.read_text()
    assert text.count(fix_mod.MARKER) == 2 and 'muse_glimmer")' not in text.split("# [gaudi-port", 1)[0]
    compile(text, REL, "exec")


def test_patch_text():
    src = (
        "x\n        if " + fix_mod.GATES[0] + ":\n            # Keep one extra block: y\n"
        "        if " + fix_mod.GATES[0] + " and target_blocks > 0:\n"
        "            window_context_blocks = align_and_pad(z)\n"
    )
    out = fix_mod.patch_runner(src, {})
    assert "gemma4" not in out and out.count(fix_mod.MARKER) == 2
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_runner(src + src, {})
