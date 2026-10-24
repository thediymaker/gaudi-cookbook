# SPDX-License-Identifier: Apache-2.0
"""CPU test for mm-pooling-inputs: the fix applies once to a copy of a release's
sources, the result compiles, a second run reports "applied" and --check then exits 0.
Needs GAUDI_FIXES_ROOT (a directory holding vllm/ and vllm_gaudi/).
Run with: GAUDI_FIXES_ROOT=<site-packages> pytest fixes/mm-pooling-inputs
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("GAUDI_FIXES_ROOT")
TARGETS = {
    "mm-encoder-budget": "vllm_gaudi/v1/worker/hpu_model_runner.py",
    "mm-pooling-inputs": "vllm_gaudi/v1/worker/hpu_model_runner.py",
    "glm4-moe-3d-input": "vllm/model_executor/models/glm4_moe.py",
    "glm45v-box-tokens": "vllm/parser/glm47_moe.py",
}
REL = TARGETS[HERE.name]

pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / REL).is_file(), reason="needs GAUDI_FIXES_ROOT with the release sources"
)


def run(root, *args):
    return subprocess.run(
        [sys.executable, str(HERE / "apply.py"), "--root", str(root), *args], capture_output=True, text=True
    )


def test_applies_once_and_is_idempotent(tmp_path):
    dst = tmp_path / REL
    dst.parent.mkdir(parents=True)
    shutil.copy(Path(ROOT) / REL, dst)
    assert run(tmp_path, "--check").returncode == 1
    first = run(tmp_path)
    assert first.returncode == 0, first.stderr
    text = dst.read_text()
    assert "[gaudi-fix:mm-pooling-inputs]" in text
    compile(text, str(dst), "exec")
    assert "applied" in run(tmp_path).stdout
    assert run(tmp_path, "--check").returncode == 0
    assert dst.read_text() == text


def test_drift_writes_nothing(tmp_path):
    dst = tmp_path / REL
    dst.parent.mkdir(parents=True)
    dst.write_text("# not the release source\n")
    assert run(tmp_path).returncode == 2
    assert dst.read_text() == "# not the release source\n"
