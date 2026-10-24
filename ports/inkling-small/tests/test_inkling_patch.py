# SPDX-License-Identifier: Apache-2.0
"""The runner patch on a copy of release sources, before and after the v0.30.0 fix profile (both orders must
give the same file). Needs GAUDI_FIXES_ROOT (a directory holding vllm/ and vllm_gaudi/ of releases/v0.30.0);
skips without it. Stdlib only."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
REPO = PORT.parent.parent
PATCH = PORT / "patches" / "runner_decode_window.py"
ROOT = os.environ.get("GAUDI_FIXES_ROOT")
RUNNER = Path("vllm_gaudi/v1/worker/hpu_model_runner.py")

pytestmark = pytest.mark.skipif(not ROOT, reason="GAUDI_FIXES_ROOT not set")


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def _copy(tmp_path, name):
    dst = tmp_path / name
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(ROOT) / pkg, dst / pkg)
    return dst


def _fixes(root):
    r = _run(REPO / "fixes" / "apply_all.py", "--root", root, "--profile", "v0.30.0", "vllm-long-prefill-share")
    assert r.returncode == 0, r.stdout + r.stderr


def test_patch_contract_and_fix_order(tmp_path):
    a = _copy(tmp_path, "port_first")
    assert _run(PATCH, "--check", "--root", a).returncode == 1  # not applied, anchors match
    assert _run(PATCH, "--root", a).returncode == 0
    assert _run(PATCH, "--root", a).returncode == 0  # idempotent
    assert _run(PATCH, "--check", "--root", a).returncode == 0
    _fixes(a)

    b = _copy(tmp_path, "fixes_first")
    _fixes(b)
    assert _run(PATCH, "--root", b).returncode == 0
    assert (a / RUNNER).read_text() == (b / RUNNER).read_text()
    assert (a / RUNNER).read_text().count("[inkling-small:decode-window]") == 2


def test_patch_refuses_drifted_source(tmp_path):
    c = _copy(tmp_path, "drift")
    runner = c / RUNNER
    runner.write_text(runner.read_text().replace("# Adjust sliding block size for specific model types", "# moved"))
    before = runner.read_text()
    assert _run(PATCH, "--root", c).returncode == 2
    assert runner.read_text() == before
