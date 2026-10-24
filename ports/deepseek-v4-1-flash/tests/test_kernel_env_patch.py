# SPDX-License-Identifier: Apache-2.0
"""patches/dsv41-kernel-env on a fake vllm_gaudi tree (standard library only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APPLY = Path(__file__).resolve().parents[1] / "patches" / "dsv41-kernel-env" / "apply.py"
HEAD = "import os\nimport json\nimport sys\n\n\ndef register():\n    return 'vllm_gaudi.platform.HpuPlatform'\n"


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(APPLY), "--root", str(root), *args], capture_output=True, text=True)


def tree(tmp_path: Path, text: str) -> Path:
    (tmp_path / "vllm_gaudi").mkdir()
    (tmp_path / "vllm_gaudi" / "__init__.py").write_text(text)
    return tmp_path / "vllm_gaudi" / "__init__.py"


def test_apply_check_idempotent(tmp_path):
    init = tree(tmp_path, HEAD)
    assert run(tmp_path, "--check").returncode == 1  # not applied, anchors match
    assert run(tmp_path).returncode == 0
    patched = init.read_text()
    assert patched.startswith("# [gaudi-fix:dsv41-kernel-env]")
    assert patched.endswith(HEAD)
    compile(patched, str(init), "exec")
    assert run(tmp_path).returncode == 0  # second run: no change
    assert init.read_text() == patched
    assert run(tmp_path, "--check").returncode == 0


def test_hook_never_breaks_import(tmp_path):
    init = tree(tmp_path, HEAD)
    assert run(tmp_path).returncode == 0
    # a dsv41_hpu whose setup_env raises: the hook must swallow the error and the module must still load
    (tmp_path / "dsv41_hpu").mkdir()
    (tmp_path / "dsv41_hpu" / "__init__.py").write_text("")
    (tmp_path / "dsv41_hpu" / "mxk.py").write_text("def setup_env():\n    raise OSError('no library')\n")
    out = subprocess.run(
        [sys.executable, "-c", "import vllm_gaudi; print(vllm_gaudi.register())"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert out.returncode == 0, out.stderr
    assert "setup_env failed" in out.stdout
    assert out.stdout.strip().endswith("vllm_gaudi.platform.HpuPlatform")
    assert init.exists()


def test_drift_writes_nothing(tmp_path):
    init = tree(tmp_path, "import os\nimport sys\n")
    r = run(tmp_path)
    assert r.returncode == 2
    assert init.read_text() == "import os\nimport sys\n"
