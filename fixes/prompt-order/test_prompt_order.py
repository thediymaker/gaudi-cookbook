# SPDX-License-Identifier: Apache-2.0
"""CPU test for prompt-order: the inserted sort, run against a stand-in input batch.

Stdlib only. Run with: pytest fixes/prompt-order
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_apply():
    spec = importlib.util.spec_from_file_location("prompt_order_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sort_fn():
    mod = _load_apply()
    ns: dict = {}
    exec(mod.FUNC_NEW.split("\n\ndef get_target_layer_suffix_list")[0], ns)
    return ns["sort_prompts_by_context"]


class FakeBatch:
    """Just the parts of InputBatch the sort touches; swap_states swaps every row field."""

    def __init__(self, rows):
        self.req_ids = [r[0] for r in rows]
        self.num_computed_tokens_cpu = [r[1] for r in rows]
        self.payload = [f"state-of-{r[0]}" for r in rows]
        self.req_id_to_index = {r: i for i, r in enumerate(self.req_ids)}
        self.swaps = 0

    @property
    def num_reqs(self):
        return len(self.req_ids)

    def swap_states(self, i, j):
        self.swaps += 1
        for field in (self.req_ids, self.num_computed_tokens_cpu, self.payload):
            field[i], field[j] = field[j], field[i]
        self.req_id_to_index[self.req_ids[i]] = i
        self.req_id_to_index[self.req_ids[j]] = j


def test_reader_moves_after_writer_and_state_follows():
    sort = _sort_fn()
    b = FakeBatch([("d0", 4), ("reader", 32), ("writer", 0)])
    sort(b, 1)
    assert b.req_ids == ["d0", "writer", "reader"]
    assert b.num_computed_tokens_cpu == [4, 0, 32]
    assert b.payload == ["state-of-d0", "state-of-writer", "state-of-reader"]
    assert all(b.req_id_to_index[r] == i for i, r in enumerate(b.req_ids))


def test_sort_is_ascending_and_stable():
    sort = _sort_fn()
    b = FakeBatch([("p48", 48), ("p0a", 0), ("p16", 16), ("p0b", 0)])
    sort(b, 0)
    assert b.req_ids == ["p0a", "p0b", "p16", "p48"]


def test_decode_rows_untouched_and_noop_cases():
    sort = _sort_fn()
    b = FakeBatch([("d0", 9), ("d1", 5), ("p0", 2)])
    sort(b, 2)
    assert b.req_ids == ["d0", "d1", "p0"] and b.swaps == 0
    b = FakeBatch([("d0", 9), ("d1", 5)])
    sort(b, b.num_reqs)
    assert b.req_ids == ["d0", "d1"] and b.swaps == 0


def test_anchors_are_unique_in_patch_text():
    mod = _load_apply()
    assert mod.MARKER in mod.FUNC_NEW and mod.MARKER in mod.CALL_NEW
    assert mod.FUNC_NEW.endswith(mod.FUNC_ANCHOR)
