# SPDX-License-Identifier: Apache-2.0
"""CPU tests for so-row-owners: the appended helpers, and the source patch on a runner copy.

The helper tests need torch and numpy. The source tests need a runner source: set
GAUDI_FIXES_ROOT to a directory holding vllm_gaudi/, or run where vllm_gaudi is installed.

Run with: pytest fixes/so-row-owners
"""

import importlib.util
import os
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUNNER_REL = "v1/worker/hpu_model_runner.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "so_row_owners_apply")
rows_mod = _load(HERE.parent / "structured-output-rows" / "apply.py", "structured_output_rows_apply")


@pytest.fixture
def helpers():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    calls, warnings = [], []

    def apply_cpu(logits, bitmask, indices=None):
        calls.append((bitmask.clone(), None if indices is None else indices.tolist()))

    ns = {
        "np": np,
        "torch": torch,
        "xgr_cpu": types.SimpleNamespace(apply_token_bitmask_inplace_cpu=apply_cpu),
        "logger": types.SimpleNamespace(warning_once=lambda *a: warnings.append(a)),
    }
    exec(fix_mod.HELPERS, ns)
    return ns, calls, warnings, torch, np


def test_row_owners_match_or_fall_back(helpers):
    ns, _, warnings, _, _ = helpers
    owners = ns["_gaudi_fix_so_row_owners"]
    assert owners(["d0", "d1"], ["p1"], 3) == ["d0", "d1", "p1"]
    assert owners(["d0", "d1"], ["p1"], 4) is None
    assert owners(["d0", "d0"], [], 2) is None
    assert len(warnings) == 2


def test_bitmask_follows_row_owners(helpers):
    ns, calls, _, torch, np = helpers
    # rows: two decodes and one finishing prompt; "partial" is a grammar request with no row
    rows = ["d0", "d1", "p_full"]
    grammar = types.SimpleNamespace(
        structured_output_request_ids=["partial", "p_full", "d0"],
        grammar_bitmask=np.array([[1], [2], [3]], dtype=np.int32),
    )
    sched = types.SimpleNamespace(scheduled_spec_decode_tokens={})
    logits = torch.zeros(3, 8)
    ns["_gaudi_fix_so_apply_bitmask_by_rows"](sched, grammar, logits, rows)
    (bitmask, indices) = calls[0]
    assert bitmask[:, 0].tolist() == [3, -1, 2]  # d0 gets its own mask, p_full its own, d1 none
    assert indices == [2, 0]


def test_no_grammar_rows_is_noop(helpers):
    ns, calls, _, torch, np = helpers
    grammar = types.SimpleNamespace(
        structured_output_request_ids=["partial"], grammar_bitmask=np.array([[1]], dtype=np.int32)
    )
    sched = types.SimpleNamespace(scheduled_spec_decode_tokens={})
    ns["_gaudi_fix_so_apply_bitmask_by_rows"](sched, grammar, torch.zeros(2, 8), ["d0", "d1"])
    assert calls == []


@pytest.fixture
def runner_root(tmp_path):
    path = fix_mod.locate("vllm_gaudi", RUNNER_REL, os.environ.get("GAUDI_FIXES_ROOT"))
    if path is None:
        pytest.skip("no vllm_gaudi runner source (set GAUDI_FIXES_ROOT)")
    src = path.read_text()
    if "[gaudi-fix:" in src:
        pytest.skip("runner source is already patched")
    target = tmp_path / "vllm_gaudi" / RUNNER_REL
    target.parent.mkdir(parents=True)
    target.write_text(src)
    return tmp_path, target


def test_applies_alone_with_decode_first_fallback(runner_root):
    root, target = runner_root
    assert fix_mod.run(True, str(root)) == 1
    assert fix_mod.run(False, str(root)) == 0
    patched = target.read_text()
    assert "pd_info.prompt_req_ids + pd_info.decode_req_ids" not in patched
    assert fix_mod.run(True, str(root)) == 0
    compile(patched, str(target), "exec")


def test_applies_after_structured_output_rows(runner_root):
    root, target = runner_root
    assert rows_mod.run(False, str(root)) == 0
    assert fix_mod.run(True, str(root)) == 1
    assert fix_mod.run(False, str(root)) == 0
    patched = target.read_text()
    assert fix_mod.run(False, str(root)) == 0 and target.read_text() == patched
    assert rows_mod.run(True, str(root)) == 0 and fix_mod.run(True, str(root)) == 0
    compile(patched, str(target), "exec")
