# SPDX-License-Identifier: Apache-2.0
"""CPU test for gemma4-buffered-tool-args: the escape-repair rules and the edit itself.

Stdlib only. The source check runs when GAUDI_FIXES_ROOT points at a directory holding vllm/.
Run with: pytest fixes/gemma4-buffered-tool-args
"""

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("gemma4_buffered_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _repair():
    ns: dict = {}
    exec(mod.REPAIR, ns)
    return ns["_g4_repair"]


def test_all_quotes_escaped_is_repaired():
    repair = _repair()
    value = '\\"\\"\\"Alpha helpers.\\"\\"\\"\\nimport os\\n'
    assert repair({"content": value}) == {"content": '"""Alpha helpers."""\nimport os\n'}


def test_legitimate_escapes_are_kept():
    repair = _repair()
    code = 'print("a \\"quoted\\" word")\nx = 1\n'
    assert repair({"content": code}) == {"content": code}
    assert repair({"n": 3, "items": ["plain", None]}) == {"n": 3, "items": ["plain", None]}


def test_nested_values_are_repaired():
    repair = _repair()
    out = repair({"edits": [{"old": '\\"a\\"', "new": "b"}]})
    assert out == {"edits": [{"old": '"a"', "new": "b"}]}


def test_patch_on_release_source(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root or not (Path(root) / "vllm" / "parser" / "gemma4.py").is_file():
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/")
    for rel in ("parser/gemma4.py", "parser/engine/parser_engine.py"):
        dst = tmp_path / "vllm" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(root) / "vllm" / rel, dst)
    assert mod.run(True, str(tmp_path)) == 1
    assert mod.run(False, str(tmp_path)) == 0
    src = (tmp_path / "vllm" / "parser" / "gemma4.py").read_text()
    assert src.count(mod.MARKER) >= 3 and "_g4_repair(_parse_gemma4_args(" in src
    assert mod.run(True, str(tmp_path)) == 0
