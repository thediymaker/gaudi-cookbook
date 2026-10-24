# SPDX-License-Identifier: Apache-2.0
"""Contract tests for every fix, plus an optional end-to-end run of a release profile.

Stdlib and pytest only. The end-to-end test runs when GAUDI_FIXES_ROOT points at a directory
holding vllm/ and vllm_gaudi/ of a release and GAUDI_FIXES_PROFILE names its profile; it works on
a temporary copy and never writes to GAUDI_FIXES_ROOT.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = "# ---- common driver (identical in every fix)"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


apply_all = _load(HERE / "apply_all.py", "fixes_apply_all")
FIX_DIRS = sorted(p.parent.name for p in HERE.glob("*/apply.py"))
# Fixes that are not source patches (a startup step, a parser plugin) declare KIND and skip the driver.
KINDS = {name: _load(HERE / name / "apply.py", f"kind_{name.replace('-', '_')}") for name in FIX_DIRS}
KINDS = {name: getattr(mod, "KIND", "patch") for name, mod in KINDS.items()}
PATCH_DIRS = [name for name in FIX_DIRS if KINDS[name] == "patch"]


def test_order_lists_every_fix_once():
    assert sorted(list(apply_all.ORDER) + list(apply_all.STANDALONE)) == FIX_DIRS
    assert len(set(apply_all.ORDER)) == len(apply_all.ORDER)
    assert all(KINDS[name] == "startup" for name in apply_all.STANDALONE)
    for members in apply_all.PROFILES.values():
        assert set(members) <= set(apply_all.ORDER)


def test_driver_is_identical():
    drivers = {name: (HERE / name / "apply.py").read_text().split(DRIVER, 1) for name in PATCH_DIRS}
    assert all(len(parts) == 2 for parts in drivers.values()), "driver marker missing"
    assert len({parts[1] for parts in drivers.values()}) == 1, "driver block differs between fixes"


@pytest.mark.parametrize("name", FIX_DIRS)
def test_fix_contract(name):
    mod = _load(HERE / name / "apply.py", f"fix_{name.replace('-', '_')}")
    assert name == mod.FIX_ID
    assert f"[gaudi-fix:{name}]" == mod.MARKER
    assert KINDS[name] in ("patch", "plugin", "startup")
    if KINDS[name] == "patch":
        assert mod.TARGETS and any(fn is not None for _, _, fn, _ in mod.TARGETS)
        for pkg, rel, _, _ in mod.TARGETS:
            assert pkg in ("vllm", "vllm_gaudi") and not rel.startswith("/")
    readme = (HERE / name / "README.md").read_text()
    assert readme.startswith(f"# {name}\n")
    for row in ("| Affects |", "| Releases |", "| Upstream |", "| Test |"):
        assert row in readme, f"{name}: README lacks {row}"


@pytest.mark.parametrize("name", [n for n in FIX_DIRS if KINDS[n] != "startup"])
def test_missing_root_is_an_error(name, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(HERE / name / "apply.py"), "--check", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_profile_end_to_end(tmp_path):
    root, profile = os.environ.get("GAUDI_FIXES_ROOT"), os.environ.get("GAUDI_FIXES_PROFILE")
    if not root or not profile:
        pytest.skip("set GAUDI_FIXES_ROOT and GAUDI_FIXES_PROFILE to run a profile end to end")
    site = tmp_path / "site"
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(root) / pkg, site / pkg, ignore=shutil.ignore_patterns("__pycache__"))

    def run(*extra):
        cmd = [sys.executable, str(HERE / "apply_all.py"), "--profile", profile, "--root", str(site), *extra]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    first = run("--check")
    assert first.returncode in (0, 1), first.stdout + first.stderr
    for _ in range(2):
        proc = run()
        assert proc.returncode == 0, proc.stdout + proc.stderr
    final = run("--check")
    assert final.returncode == 0, final.stdout + final.stderr
    patched = [p for p in site.rglob("*.py") if "[gaudi-fix:" in p.read_text()]
    assert patched
    for path in patched:
        compile(path.read_text(), str(path), "exec")
