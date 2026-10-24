# SPDX-License-Identifier: Apache-2.0
"""CPU tests for pipeline-parallel.

* The split arithmetic of the transfer (which slice of a tensor each tensor-parallel rank sends) is taken from the
  appended runner code and checked to cover every element exactly once. Stdlib only.
* With GAUDI_FIXES_ROOT (a directory holding vllm/ and vllm_gaudi/ of a release), the fix is applied to a copy
  together with the release profile, twice, and the patched files must compile and keep one marker block each.

Run with: pytest fixes/pipeline-parallel
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
APPLY = HERE / "apply.py"
APPLY_ALL = HERE.parent / "apply_all.py"


def _runner_code() -> str:
    tree = ast.parse(APPLY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "RUNNER_CODE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("RUNNER_CODE not found")


def _link_methods():
    """_bounds and _padded of _GfppLink, compiled on their own (no torch needed)."""
    tree = ast.parse(_runner_code())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_GfppLink")
    keep = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_bounds", "_padded")]
    ns = {"_GFPP_ALIGN": 16, "staticmethod": staticmethod}
    stub = ast.ClassDef(name="Link", bases=[], keywords=[], body=keep, decorator_list=[], type_params=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), "link", "exec"), ns)
    return ns["Link"]


class _TP:
    def __init__(self, world, rank):
        self.world_size, self.rank_in_group = world, rank


@pytest.mark.parametrize("n", [1, 7, 8, 4096, 7168 * 3 + 5])
@pytest.mark.parametrize("world", [1, 2, 8])
def test_split_covers_every_element_once(n, world):
    link_cls = _link_methods()
    covered = []
    width = None
    for rank in range(world):
        link = link_cls()
        link.tp = _TP(world, rank)
        lo, hi, c = link._bounds(n, True)
        assert 0 <= lo <= hi <= n and hi - lo <= c
        width = c if width is None else width
        assert c == width  # every rank pads its piece to the same size, so the all-gather lines up
        covered.extend(range(lo, hi))
    assert covered == list(range(n))
    link = link_cls()
    link.tp = _TP(world, 0)
    assert link._bounds(n, False) == (0, n, n)


def test_padding_is_aligned():
    link_cls = _link_methods()
    for c in (1, 3, 1000):
        for es in (1, 2, 4):
            p = link_cls._padded(c, es)
            assert p % 16 == 0 and c * es <= p < c * es + 16


def _release_root():
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root or not (Path(root) / "vllm_gaudi").is_dir() or not (Path(root) / "vllm").is_dir():
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/ and vllm_gaudi/ of a release")
    return Path(root)


def test_applies_with_profile_twice(tmp_path):
    root = _release_root()
    profile = os.environ.get("GAUDI_FIXES_PROFILE", "v0.30.0")
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(root / pkg, tmp_path / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    cmd = [sys.executable, str(APPLY_ALL), "--profile", profile, "pipeline-parallel", "--root", str(tmp_path)]
    for _ in range(2):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    check = subprocess.run([sys.executable, str(APPLY), "--check", "--root", str(tmp_path)], capture_output=True)
    assert check.returncode == 0
    files = (
        "vllm_gaudi/v1/worker/hpu_model_runner.py",
        "vllm_gaudi/v1/worker/hpu_worker.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/model_executor/model_loader/weight_utils.py",
    )
    for rel in files:
        text = (tmp_path / rel).read_text()
        compile(text, rel, "exec")
        assert text.count("--- [gaudi-fix:pipeline-parallel]") + text.count("# [gaudi-fix:pipeline-parallel]") == 1
    runner = (tmp_path / files[0]).read_text()
    # the profile includes prompt-order, so the non-last stages must mirror its sort
    assert "def sort_prompts_by_context" in runner and '"sort_prompts_by_context" in globals()' in runner
