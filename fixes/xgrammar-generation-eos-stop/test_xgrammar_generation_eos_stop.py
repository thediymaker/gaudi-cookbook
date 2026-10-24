# SPDX-License-Identifier: Apache-2.0
"""CPU test for xgrammar-generation-eos-stop: the stop-token helper, with stand-in objects.

Stdlib only. Run with: pytest fixes/xgrammar-generation-eos-stop
"""

import importlib.util
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _helper():
    spec = importlib.util.spec_from_file_location("xgr_eos_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logged = []
    ns = {"logger": types.SimpleNamespace(info=lambda *a: logged.append(a))}
    exec(mod.HELPER, ns)
    return ns["_gaudi_fix_generation_stop_ids"], logged


def _backend(eos, gen_eos, vocab_size=100, raises=False):
    def gen_config():
        if raises:
            raise OSError("no generation_config.json")
        return {} if gen_eos is None else {"eos_token_id": gen_eos}

    model_config = types.SimpleNamespace(try_get_generation_config=gen_config)
    return types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(eos_token_id=eos),
        vllm_config=types.SimpleNamespace(model_config=model_config),
        vocab_size=vocab_size,
    )


def test_generation_config_eos_ids_are_added():
    fn, logged = _helper()
    assert fn(_backend(2, [7, 9])) == [2, 7, 9]
    assert logged


def test_single_int_and_duplicates():
    fn, _ = _helper()
    assert fn(_backend(2, 2)) == [2]
    assert fn(_backend(2, 5)) == [2, 5]


def test_out_of_vocab_ids_are_dropped_and_missing_config_is_tolerated():
    fn, _ = _helper()
    assert fn(_backend(2, [7, 500], vocab_size=100)) == [2, 7]
    assert fn(_backend(2, None, raises=True)) == [2]


def test_none_when_nothing_is_known():
    fn, _ = _helper()
    assert fn(_backend(None, None)) is None
