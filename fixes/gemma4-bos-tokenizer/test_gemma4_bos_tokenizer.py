# SPDX-License-Identifier: Apache-2.0
"""CPU test for gemma4-bos-tokenizer: the directory build step on a synthetic snapshot.

Stdlib only (the transformers check in apply.py is not exercised here).
Run with: pytest fixes/gemma4-bos-tokenizer
"""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("gemma4_bos_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _snapshot(tmp_path, post_processor):
    src = tmp_path / "snap"
    src.mkdir()
    tok = {
        "added_tokens": [{"id": 2, "content": "<bos>"}],
        "model": {"vocab": {"<bos>": 2, "a": 3}},
        "post_processor": post_processor,
    }
    (src / "tokenizer.json").write_text(json.dumps(tok))
    (src / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>"}))
    (src / "chat_template.jinja").write_text("{{ bos_token }}")
    (src / "model-00001.safetensors").write_text("weights")
    return src


def test_build_adds_bos_and_links_the_rest(tmp_path):
    src = _snapshot(tmp_path, {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}]})
    out = tmp_path / "out"
    assert mod.build(src, out) == ("<bos>", 2)
    tok = json.loads((out / "tokenizer.json").read_text())
    assert tok["post_processor"]["single"][0] == {"SpecialToken": {"id": "<bos>", "type_id": 0}}
    assert json.loads((out / "tokenizer_config.json").read_text())["add_bos_token"] is True
    assert (out / "chat_template.jinja").is_symlink()
    assert not (out / "model-00001.safetensors").exists()


def test_build_keeps_a_tokenizer_that_already_adds_bos(tmp_path):
    pp = mod.bos_post_processor("<bos>", 2)
    src = _snapshot(tmp_path, pp)
    out = tmp_path / "out"
    mod.build(src, out)
    assert (out / "tokenizer.json").is_symlink()


def test_unexpected_post_processor_fails(tmp_path):
    src = _snapshot(tmp_path, {"type": "ByteLevel"})
    with pytest.raises(mod.Failure):
        mod.build(src, tmp_path / "out")


def test_usage_error_exits_2():
    assert mod.main([]) == 2
