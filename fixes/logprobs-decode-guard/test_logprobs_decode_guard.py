# SPDX-License-Identifier: Apache-2.0
"""CPU test for logprobs-decode-guard: the inserted wrapper, run against a stand-in tokenizer.

Stdlib only. Run with: pytest fixes/logprobs-decode-guard
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_apply():
    spec = importlib.util.spec_from_file_location("decode_guard_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    vocab = ["<s>", "a", "b", "c"]

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)  # IndexError past the end


def _orig_convert_ids_list_to_tokens(tokenizer, token_ids):
    out = []
    for t in token_ids:
        if t >= 2**31:
            raise OverflowError("out of range integral type conversion attempted")
        out.append(tokenizer.decode([t]))
    return out


def _wrapper():
    mod = _load_apply()
    ns = {"convert_ids_list_to_tokens": _orig_convert_ids_list_to_tokens, "__name__": "detok_test"}
    exec(mod.DETOK_NEW[: -len(mod.DETOK_ANCHOR)], ns)
    return ns["convert_ids_list_to_tokens"]


def test_good_ids_pass_through():
    conv = _wrapper()
    assert conv(FakeTokenizer(), [1, 2, 3]) == ["a", "b", "c"]


def test_bad_ids_become_placeholders_and_neighbours_survive():
    conv = _wrapper()
    assert conv(FakeTokenizer(), [1, 2**40, 3, 7]) == ["a", "<bad_id>", "c", "<bad_id>"]


def test_logprobs_edit_returns_empty_string():
    mod = _load_apply()
    body = (
        "def f(self, context_token_ids, token_id):\n"
        "    for num_ctx in (1,):\n"
        "        if True:\n" + mod.LP_NEW + "            return full_decoded\n"
    )
    ns: dict = {}
    exec(body, ns)
    self = type("P", (), {"tokenizer": FakeTokenizer()})()
    assert ns["f"](self, [1], 2) == "ab"
    assert ns["f"](self, [1], 99) == ""
