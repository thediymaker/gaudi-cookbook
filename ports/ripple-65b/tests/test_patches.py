# SPDX-License-Identifier: Apache-2.0
"""The port's patch scripts follow the fix contract (fixes/README.md): exit 0 applied, 1 not applied (--check),
2 anchors missing or unexpected code, and no partial writes.

Stdlib and pytest only. The chain test runs when GAUDI_FIXES_ROOT points at a directory holding the vllm/ and
vllm_gaudi/ packages of vllm-gaudi releases/v0.30.0 (as for fixes/test_fixes.py); it works on a temporary copy:
fix profile v0.30.0 plus swa-window-context, then every port patch twice (the second run reports
"applied"), then --check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
COOKBOOK = PORT.parent.parent
PATCHES = ["parser_streams_all.py", "runner_window.py", "swa_kv_hooks.py"]


def run(script, root, *args):
    return subprocess.run(
        [sys.executable, str(PORT / "patches" / script), "--root", str(root), *args], capture_output=True, text=True
    )


def fake_tree(tmp_path):
    w = tmp_path / "vllm_gaudi" / "v1" / "worker"
    w.mkdir(parents=True)
    (tmp_path / "vllm_gaudi" / "__init__.py").write_text("")
    (w / "hpu_model_runner.py").write_text("class HPUModelRunner:\n    pass\n")
    (w / "hpu_worker.py").write_text("class HPUWorker:\n    pass\n")
    (tmp_path / "vllm").mkdir()
    return tmp_path


@pytest.mark.parametrize("script", ["runner_window.py", "swa_kv_hooks.py"])
def test_fail_closed_without_anchors(tmp_path, script):
    root = fake_tree(tmp_path)
    before = {p: p.read_text() for p in root.rglob("*.py")}
    for args in ([], ["--check"]):
        r = run(script, root, *args)
        assert r.returncode == 2, r.stdout + r.stderr
    assert {p: p.read_text() for p in root.rglob("*.py")} == before


def test_missing_package(tmp_path):
    for script in PATCHES:
        r = run(script, tmp_path, "--check")
        assert r.returncode == 2, (script, r.stdout + r.stderr)


def test_parser_patch_fails_closed(tmp_path):
    p = tmp_path / "vllm" / "parser"
    p.mkdir(parents=True)
    (p / "abstract_parser.py").write_text("class DelegatingParser:\n    pass\n")
    for args in ([], ["--check"]):
        r = run("parser_streams_all.py", tmp_path, *args)
        assert r.returncode == 2, r.stdout + r.stderr
    assert (p / "abstract_parser.py").read_text() == "class DelegatingParser:\n    pass\n"


def test_runner_window_needs_both_window_fixes(tmp_path):
    root = fake_tree(tmp_path)
    runner = root / "vllm_gaudi" / "v1" / "worker" / "hpu_model_runner.py"
    runner.write_text("# [gaudi-fix:swa-decode-window]\nclass HPUModelRunner:\n    pass\n")
    r = run("runner_window.py", root, "--check")
    assert r.returncode == 2 and "swa-window-context" in r.stderr, r.stdout + r.stderr


def test_chain_on_release(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to run the patch chain on a release copy")
    site = tmp_path / "site"
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(root) / pkg, site / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run(
        [
            sys.executable,
            str(COOKBOOK / "fixes" / "apply_all.py"),
            "--profile",
            "v0.30.0",
            "--root",
            str(site),
            "swa-window-context",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    for script in PATCHES:  # in install.sh's order: each one requires the previous ones
        assert run(script, site, "--check").returncode == 1, script
        r = run(script, site)
        assert r.returncode == 0, (script, r.stdout + r.stderr)
    for script in PATCHES:
        r = run(script, site)
        assert r.returncode == 0 and "applied" in r.stdout, (script, r.stdout + r.stderr)
    for script in PATCHES:
        assert run(script, site, "--check").returncode == 0, script
