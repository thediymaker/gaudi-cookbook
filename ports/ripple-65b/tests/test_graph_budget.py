# SPDX-License-Identifier: Apache-2.0
"""ripple_hpu.graph_budget on CPU torch: a fake runner whose graph cache (the dict its model's clear_cache()
closes over, as in habana_frameworks' wrap_in_hpu_graph) gains one entry the first time each decode shape runs.
Skips without torch.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from ripple_hpu import graph_budget as gb  # noqa: E402


class _Md:
    def __init__(self, blocks, win, is_prompt=False):
        self.is_prompt = is_prompt
        self.block_list = torch.zeros(blocks, dtype=torch.int32) if blocks else None
        self.window_block_list = torch.zeros(win, dtype=torch.int32) if win else None


def _runner_cls(second_graph_for=()):
    """A runner class like vllm-gaudi's: _execute_model_generic captures a graph (a new cache key) the first time a
    decode shape runs with graphs on; shapes in `second_graph_for` capture one more graph on their next run."""

    class Runner:
        def __init__(self):
            cache = {}

            def clear_cache():
                cache.clear()

            self.model = types.SimpleNamespace(clear_cache=clear_cache)
            self.cache, self.graphed, self.eager, self.again = cache, [], [], set(second_graph_for)
            self._n = 0

        def _use_graphs(self, attn_metadata, batch_size):
            return True

        def _execute_model_generic(self, token_ids, position_ids, attn_metadata, *a, **k):
            md = attn_metadata
            key = (
                token_ids.shape[0],
                0 if md.block_list is None else md.block_list.shape[0],
                0 if md.window_block_list is None else md.window_block_list.shape[0],
            )
            if not self._use_graphs(md, key[0]):
                self.eager.append(key)
                return "out"
            self.graphed.append(key)
            first = not any(h[0] == key for h in self.cache)
            if first or key in self.again:
                if not first:
                    self.again.discard(key)
                self._n += 1
                self.cache[(key, self._n)] = object()
            return "out"

    return Runner


@pytest.fixture
def env(monkeypatch):
    fake_hpu = types.SimpleNamespace(synchronize=lambda: None, memory_stats=lambda: {"InUse": 0})
    monkeypatch.setattr(torch, "hpu", fake_hpu, raising=False)
    graphs = types.ModuleType("habana_frameworks.torch.hpu.graphs")
    graphs.CachedParams = types.SimpleNamespace(cache_hits={}, skip_replay_cnt={})
    for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.hpu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.hpu.graphs", graphs)
    monkeypatch.setattr(gb, "BUDGET", "1.0")
    fresh = {
        "installed": False,
        "budget": 1.0,
        "scale": 4,
        "graphs": {},
        "step": 0,
        "used": 0.0,
        "repeat_logged": 0,
        "evicted": 0,
        "bypass": False,
        "rank": 1,
    }
    monkeypatch.setattr(gb, "_st", fresh)
    return fresh


def _run(r, rows, blocks, win):
    return r._execute_model_generic(torch.zeros(rows, 1), None, _Md(blocks, win))


def test_graph_gib_matches_measured_tp4_graphs():
    # (rows, blocks, window blocks) -> GiB kept, measured at TP 4 (block FP8 weights)
    for shape, gib in [
        ((32, 1024, 384), 0.789),
        ((64, 1024, 640), 1.108),
        ((64, 256, 128), 0.248),
        ((1, 128, 128), 0.196),
        ((32, 384, 384), 0.594),
    ]:
        assert gb.graph_gib(*shape) == pytest.approx(gib, abs=0.005)
    assert gb.graph_gib(64, 256, 128, tp=2) == pytest.approx(2 * gb.graph_gib(64, 256, 128))


def test_least_recently_used_graphs_are_dropped_first(env):
    R = _runner_cls()
    assert gb.install(R)
    r = R()
    # three graphs of 0.31-0.33 GiB fill most of a 1 GiB budget (0.97)
    a, b, c, d = (8, 1024, 0), (8, 1056, 0), (8, 1088, 0), (8, 1152, 0)
    for s in (a, b, c):
        _run(r, *s)
    assert len(r.cache) == 3 and env["used"] == pytest.approx(sum(gb.graph_gib(*s) for s in (a, b, c)))
    _run(r, *a)  # a is now the most recent; b is the oldest
    _run(r, *d)  # 0.35 GiB more needs room: b goes, c stays
    kept = {h[0] for h in r.cache}
    assert kept == {a, c, d}
    assert env["used"] <= 1.0 + 1e-9 and env["evicted"] >= 1
    assert {k[:3] for k in env["graphs"]} == kept


def test_a_known_shape_capturing_again_is_brought_back_under_the_budget(env):
    a, b = (8, 1024, 0), (8, 1400, 0)  # 0.31 and 0.43 GiB
    R = _runner_cls(second_graph_for={b})
    gb.install(R)
    r = R()
    _run(r, *a)
    _run(r, *b)  # first graph of b
    _run(r, *b)  # b captures a second graph: 0.31 + 2 x 0.43 > 1 GiB
    assert env["used"] <= 1.0 + 1e-9
    assert a not in {h[0] for h in r.cache}  # the older shape made room


def test_a_shape_larger_than_the_budget_runs_without_a_graph(env):
    R = _runner_cls()
    gb.install(R)
    r = R()
    _run(r, 32, 4096, 384)  # 1.7 GiB > 1 GiB
    assert r.eager == [(32, 4096, 384)] and not r.cache and env["used"] == 0.0
    _run(r, 8, 128, 128)  # graphs are back on for the next shape
    assert r.graphed == [(8, 128, 128)]


def test_prompts_and_warmup_are_not_managed(env):
    R = _runner_cls()
    gb.install(R)
    r = R()
    r._execute_model_generic(torch.zeros(4096, 1), None, _Md(0, 0, is_prompt=True))
    r._execute_model_generic(torch.zeros(8, 1), None, _Md(128, 128), warmup_mode=True)
    assert env["graphs"] == {} and env["step"] == 0
