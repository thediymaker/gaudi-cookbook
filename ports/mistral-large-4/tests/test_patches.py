# SPDX-License-Identifier: Apache-2.0
"""The port's patch scripts follow the fix contract (fixes/README.md): exit 0 applied, 1 not applied (--check),
2 a missing or unexpected file, and no partial writes.

Stdlib and pytest only. The chain test runs when GAUDI_FIXES_ROOT points at a directory holding the vllm/ and
vllm_gaudi/ packages of vllm-gaudi releases/v0.30.0 (as for fixes/test_fixes.py); it works on a temporary copy:
fix profile v0.30.0 plus the recipe's opt-in fix, then both port patches twice (the second run reports nothing new
to do), then --check.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
COOKBOOK = PORT.parent.parent
VENDOR = PORT / "vendor"


def manifest():
    rows = []
    for ln in (VENDOR / "MANIFEST").read_text().splitlines():
        p = ln.split()
        if len(p) == 4 and p[2] == "replaces":
            rows.append((p[0], p[1], p[3]))
    return rows


def pr_files(root, *args):
    cmd = [sys.executable, str(PORT / "patches" / "install_pr_files.py"), *args, str(VENDOR), str(root)]
    return subprocess.run(cmd, capture_output=True, text=True)


def hook(root, *args):
    cmd = [sys.executable, str(PORT / "patches" / "kernel_env_hook.py"), "--root", str(root), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_vendor_manifest_matches_files():
    rows = manifest()
    assert len(rows) == 5
    for digest, rel, _stock in rows:
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == digest, rel


def test_pr_files_fail_closed(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "__init__.py").write_text("")
    for _d, rel, _s in manifest():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("# not the stock v0.30.0 file\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.py")}
    for args in ([], ["--check"]):
        r = pr_files(tmp_path, *args)
        assert r.returncode == 2, r.stdout + r.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.py")} == before


def test_kernel_env_hook_idempotent(tmp_path):
    (tmp_path / "vllm_gaudi").mkdir()
    init = tmp_path / "vllm_gaudi" / "__init__.py"
    init.write_text("def register():\n    return 'vllm_gaudi.platform.HpuPlatform'\n")
    assert hook(tmp_path, "--check").returncode == 1
    assert hook(tmp_path).returncode == 0
    once = init.read_text()
    compile(once, str(init), "exec")
    assert hook(tmp_path).returncode == 0
    assert init.read_text() == once
    assert hook(tmp_path, "--check").returncode == 0
    assert hook(tmp_path / "missing", "--check").returncode == 2


def test_hook_never_breaks_import(tmp_path):
    (tmp_path / "vllm_gaudi").mkdir()
    (tmp_path / "vllm_gaudi" / "__init__.py").write_text("def register():\n    return 'ok'\n")
    assert hook(tmp_path).returncode == 0
    pkg = tmp_path / "mistral_large4_hpu"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mxk.py").write_text("def setup_env():\n    raise OSError('no library')\n")
    out = subprocess.run(
        [sys.executable, "-c", "import vllm_gaudi; print(vllm_gaudi.register())"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert out.returncode == 0, out.stderr
    assert "setup_env failed" in out.stdout and out.stdout.strip().endswith("ok")


def test_chain_on_release(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to run the patch chain on a release copy")
    if os.environ.get("GAUDI_FIXES_PROFILE", "v0.30.0") != "v0.30.0":
        pytest.skip("the port targets v0.30.0")
    work = tmp_path / "rel"
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(root) / pkg, work / pkg, symlinks=True)
    fix = [sys.executable, str(COOKBOOK / "fixes" / "apply_all.py"), "--root", str(work), "--profile", "v0.30.0"]
    r = subprocess.run([*fix, "penalty-shapes"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert pr_files(work, "--check").returncode == 1
    for _ in range(2):
        r = pr_files(work)
        assert r.returncode == 0, r.stdout + r.stderr
        r = hook(work)
        assert r.returncode == 0, r.stdout + r.stderr
    assert pr_files(work, "--check").returncode == 0
    assert hook(work, "--check").returncode == 0
    for _d, rel, _s in manifest():
        compile((work / rel).read_bytes(), str(work / rel), "exec")
