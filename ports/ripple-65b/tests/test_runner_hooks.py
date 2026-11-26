# SPDX-License-Identifier: Apache-2.0
"""The pure decisions of ripple_hpu.runner_hooks (stdlib only): when a FusedSDPA call switches to fast softmax,
and how many query tiles a biased prompt call gets."""

import importlib

import pytest


@pytest.fixture
def hooks(monkeypatch):
    monkeypatch.setenv("RIPPLE_FAST_SOFTMAX_MIN_KV", "16384")
    monkeypatch.setenv("RIPPLE_FSDPA_WS_GIB", "2")
    monkeypatch.setenv("RIPPLE_FSDPA_Q_TILE", "256")
    import ripple_hpu.runner_hooks as rh

    return importlib.reload(rh)


def test_fast_softmax_only_full_layers_with_long_context(hooks):
    bias = object()
    assert hooks.fast_softmax_wanted("fp32", True, None, bias, 4096, 4096 + 16384)
    assert not hooks.fast_softmax_wanted("fp32", False, None, bias, 4096, 4096 + 16384)  # sliding layer
    assert not hooks.fast_softmax_wanted("fp32", True, None, bias, 4096, 4096 + 8192)  # short context
    assert not hooks.fast_softmax_wanted("fp32", True, None, None, 4096, 65536)  # no bias
    assert not hooks.fast_softmax_wanted("fp32", True, 1024, bias, 4096, 65536)  # a window
    assert not hooks.fast_softmax_wanted("fp32", True, None, bias, 32768, 32768)  # no cached context
    assert not hooks.fast_softmax_wanted("fast", True, None, bias, 4096, 65536)  # already fast


def test_fast_softmax_off_by_default(monkeypatch):
    monkeypatch.delenv("RIPPLE_FAST_SOFTMAX_MIN_KV", raising=False)
    import ripple_hpu.runner_hooks as rh

    rh = importlib.reload(rh)
    assert not rh.fast_softmax_wanted("fp32", True, None, object(), 4096, 262144)


def test_query_tiles_above_the_workspace(hooks):
    assert hooks.num_q_tiles(1, 4096, 4096) == 1  # 1 GiB of workspace: whole
    assert hooks.num_q_tiles(1, 4096, 65536) == 16  # 16 GiB: 256-row tiles
    assert hooks.num_q_tiles(32, 4096, 65536) == 32  # vllm-gaudi's own count wins when larger
    assert hooks.num_q_tiles(1, 200, 262144) == 1  # fewer rows than a tile
