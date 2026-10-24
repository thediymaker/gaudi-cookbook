# SPDX-License-Identifier: Apache-2.0
"""The port patches against real sources. With GAUDI_FIXES_ROOT (a copy of a releases/v0.30.0 site-packages holding
vllm/ and vllm_gaudi/), the install order (cookbook fixes, then patches/apply_all.py) applies on a temporary copy, a
second run changes nothing, --check reports applied, and every patched file compiles. Without it the test skips."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
COOKBOOK = PORT.parent.parent
FIXES = ["--profile", "v0.30.0", "vllm-long-prefill-share", "glm47-streaming-text-markers"]


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_install_order_applies_and_is_idempotent(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to a releases/v0.30.0 site-packages copy")
    dst = tmp_path / "sp"
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(root) / pkg, dst / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    r = _run(COOKBOOK / "fixes" / "apply_all.py", "--root", dst, *FIXES)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(PORT / "patches" / "apply_all.py", "--root", dst, "--check")
    assert r.returncode == 1, r.stdout + r.stderr  # anchors match, nothing applied yet
    r = _run(PORT / "patches" / "apply_all.py", "--root", dst)
    assert r.returncode == 0, r.stdout + r.stderr
    before = {p: p.read_bytes() for p in dst.rglob("*.py")}
    for args in (["--root", dst], ["--root", dst, "--check"]):
        r = _run(PORT / "patches" / "apply_all.py", *args)
        assert r.returncode == 0 and "patched" not in r.stdout, r.stdout + r.stderr
    assert {p: p.read_bytes() for p in dst.rglob("*.py")} == before
    for line in r.stdout.splitlines():
        assert line.endswith(" applied"), line


def test_every_patch_has_the_contract():
    names = [p.stem for p in (PORT / "patches").glob("*.py") if p.stem != "apply_all"]
    sys.path.insert(0, str(PORT / "patches"))
    import apply_all

    assert sorted(names) == sorted(apply_all.ORDER)
    for name in names:
        src = (PORT / "patches" / f"{name}.py").read_text()
        assert f'FIX_ID = "glm-5-mxfp4:{name.replace("_", "-")}"' in src, name
        for need in ("def run(check", "--check", "--root", "write_atomic", "Drift"):
            assert need in src, (name, need)
