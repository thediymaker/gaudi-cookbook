# SPDX-License-Identifier: Apache-2.0
"""The port's runner patches on a copy of vllm-gaudi v0.26.0 sources (stdlib only).

Needs GAUDI_FIXES_ROOT: a directory holding the release's vllm/ and vllm_gaudi/ (for example a copy of its
site-packages), as for fixes/test_fixes.py. Skips without it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parents[1]
COOKBOOK = PORT.parents[1]
ROOT = os.environ.get("GAUDI_FIXES_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="GAUDI_FIXES_ROOT not set")


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


@pytest.fixture
def tree(tmp_path):
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(ROOT) / pkg, tmp_path / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def apply_profile(root):
    r = run(COOKBOOK / "fixes" / "apply_all.py", "--root", root, "--profile", "v0.26.0", "vllm-long-prefill-share")
    assert r.returncode == 0, r.stdout + r.stderr


def test_order_and_idempotence(tree):
    window, kv = PORT / "patches" / "runner_window.py", PORT / "patches" / "runner_kv.py"
    # before the fix profile: refuse, write nothing
    assert run(window, "--root", tree).returncode == 2
    apply_profile(tree)
    runner = tree / "vllm_gaudi" / "v1" / "worker" / "hpu_model_runner.py"
    before = runner.read_text()
    assert run(kv, "--root", tree).returncode == 2  # runner_window.py first
    assert runner.read_text() == before
    assert run(window, "--check", "--root", tree).returncode == 1
    assert run(window, "--root", tree).returncode == 0
    assert run(kv, "--check", "--root", tree).returncode == 1
    assert run(kv, "--root", tree).returncode == 0
    patched = runner.read_text()
    for script in (window, kv):  # second runs report "applied" and write nothing
        assert run(script, "--root", tree).returncode == 0
    apply_profile(tree)  # the launch command's fix step still passes on the port image
    assert runner.read_text() == patched
    assert 'in ("gemma4", "mimo_v2")' in patched and 'in ["gpt_oss", "mimo_v2"]' in patched
    assert patched.count("[mimo-v2-6-flash:runner-kv]") == 13
    worker = (tree / "vllm_gaudi" / "v1" / "worker" / "hpu_worker.py").read_text()
    assert "SlidingWindowSpec" in worker
    compile(patched, str(runner), "exec")


def test_drift_writes_nothing(tree):
    apply_profile(tree)
    runner = tree / "vllm_gaudi" / "v1" / "worker" / "hpu_model_runner.py"
    assert run(PORT / "patches" / "runner_window.py", "--root", tree).returncode == 0
    worker = tree / "vllm_gaudi" / "v1" / "worker" / "hpu_worker.py"
    worker.write_text(worker.read_text().replace("if isinstance(layer_spec, FullAttentionSpec):", "if False:"))
    before = runner.read_text()
    assert run(PORT / "patches" / "runner_kv.py", "--root", tree).returncode == 2
    assert runner.read_text() == before  # the runner was not written either
