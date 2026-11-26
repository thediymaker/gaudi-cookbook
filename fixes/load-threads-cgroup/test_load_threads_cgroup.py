# SPDX-License-Identifier: Apache-2.0
"""Tests for load-threads-cgroup.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift.
2. The inserted lines (stdlib only), with cpu.max contents faked: a quota caps the count (rounded up),
   "max" or a missing or malformed file keeps it.

Run with: pytest fixes/load-threads-cgroup
"""

import importlib.util
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = "v1/worker/hpu_model_runner.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "load_threads_cgroup_apply")


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
    assert fix_mod.MARKER in text
    compile(text, REL, "exec")
    assert _run(site) == 0 and path.read_text() == text
    assert _run(site, "--check") == 0


def test_source_patch_refuses_drift(site):
    path = site / "vllm_gaudi" / REL
    path.write_text(path.read_text().replace(fix_mod.OLD, "            num_cores = 8\n", 1))
    before = path.read_text()
    assert _run(site) == 2 and path.read_text() == before


def _cores(cpu_max, host=160):
    def fake_open(path, *a, **kw):
        assert path == "/sys/fs/cgroup/cpu.max"
        if cpu_max is None:
            raise FileNotFoundError(path)
        return io.StringIO(cpu_max)

    body = "".join(line[12:] for line in fix_mod.CAP.splitlines(keepends=True))
    ns = {"num_cores": host, "open": fake_open}
    exec(body, ns)
    return ns["num_cores"]


@pytest.mark.parametrize(
    ("cpu_max", "want"),
    [
        ("1600000 100000\n", 16),
        ("1650000 100000\n", 17),
        ("50000 100000\n", 1),
        ("max 100000\n", 160),
        (None, 160),
        ("garbage\n", 160),
        ("32000000 100000\n", 160),
    ],
)
def test_quota_cap(cpu_max, want):
    assert _cores(cpu_max) == want
