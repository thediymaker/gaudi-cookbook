# SPDX-License-Identifier: Apache-2.0
"""Tests for hpu-graph-lru.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift.
2. The cache policy (stdlib only), driven the way Habana's wrapped_hpugraph_forward drives it: get(), then
   the `len(cache) == max_graphs` check, then capture through __setitem__:
   - a cap full of early shapes, then a new decode shape: Habana's plain dict never captures it, the LRU
     captures it on its second miss and replays it from then on;
   - prompt graphs are evicted before decode graphs, and a prompt shape never evicts a decode graph;
   - one-off shapes never evict with admit-after 2, and the cache never exceeds the cap;
   - two caches fed the same shape sequence (two tensor-parallel ranks) make the same decisions;
   - every evicted entry is released exactly once.
3. The wrapper on a stub habana_frameworks module: unset cap keeps the stock wrapper, a set cap routes every
   call through the cache, and an unexpected wrapped_hpugraph_forward signature stops start-up.

Run with: pytest fixes/hpu-graph-lru
"""

import importlib.util
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = "v1/worker/hpu_model_runner.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "hpu_graph_lru_apply")


@pytest.fixture
def site(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/ and vllm_gaudi/ of a release")
    for pkg, rel, _, _ in fix_mod.TARGETS:
        dst = tmp_path / pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(root) / pkg / rel, dst)
    return tmp_path


def _run(root, *extra):
    cmd = [sys.executable, str(HERE / "apply.py"), "--root", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode


def test_source_patch(site):
    path = site / "vllm_gaudi" / REL
    assert _run(site, "--check") == 1
    assert _run(site) == 0
    text = path.read_text()
    assert fix_mod.MARKER in text and text.count("def _maybe_wrap_in_hpu_graph(") == 1
    compile(text, REL, "exec")
    assert _run(site) == 0 and path.read_text() == text
    assert _run(site, "--check") == 0


def test_source_patch_refuses_drift(site):
    path = site / "vllm_gaudi" / REL
    path.write_text(
        path.read_text().replace("disable_tensor_cache=True) if htorch", "disable_tensor_cache=False) if htorch", 1)
    )
    before = path.read_text()
    assert _run(site) == 2 and path.read_text() == before


@pytest.fixture
def lru():
    ns = {}
    exec(compile(fix_mod.LRU_CODE, "lru", "exec"), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


def forward(cache, key, phase, max_graphs):
    """Habana's decision for one call: 'replay', 'capture' or 'uncaptured'."""
    if hasattr(cache, "current_phase"):
        cache.current_phase = phase
    if cache.get(key) is not None:
        return "replay"
    if len(cache) == max_graphs:
        return "uncaptured"
    cache[key] = types.SimpleNamespace(key=key)
    return "capture"


def run(cache, seq, max_graphs):
    return [forward(cache, k, p, max_graphs) for k, p in seq]


def test_new_decode_shape_is_captured(lru):
    cap = 8
    early = [(("p", i), "prompt") for i in range(5)] + [(("d", i), "decode") for i in range(3)]
    live = [(("d", "live"), "decode")] * 6
    assert run({}, early + live, cap)[len(early) :] == ["uncaptured"] * 6
    released = []
    c = lru.HpuGraphLRU(cap, admit_after=2, destroy=released.append)
    got = run(c, early + live, cap)[len(early) :]
    assert got == ["uncaptured", "capture", "replay", "replay", "replay", "replay"]
    assert len(c) == cap
    assert [d.key for d in released] == [("p", 0)]
    assert c.stats["uncaptured", "decode"] == 1 and c.stats["evicted", "prompt"] == 1


def test_prompt_never_evicts_decode(lru):
    c = lru.HpuGraphLRU(3, admit_after=1)
    run(c, [(("d", i), "decode") for i in range(3)], 3)
    assert run(c, [(("p", "x"), "prompt")] * 4, 3) == ["uncaptured"] * 4
    assert c.held() == {"prompt": 0, "decode": 3}


def test_decode_evicts_least_recently_used(lru):
    released = []
    c = lru.HpuGraphLRU(3, admit_after=1, destroy=released.append)
    run(c, [(("d", 0), "decode"), (("d", 1), "decode"), (("d", 2), "decode"), (("d", 0), "decode")], 3)
    run(c, [(("d", 9), "decode")], 3)
    assert [d.key for d in released] == [("d", 1)]


def test_one_off_shapes_do_not_churn(lru):
    c = lru.HpuGraphLRU(4, admit_after=2)
    run(c, [(("d", i), "decode") for i in range(4)], 4)
    outs = run(c, [(("once", i), "prompt" if i % 2 else "decode") for i in range(200)], 4)
    assert set(outs) == {"uncaptured"} and len(c) == 4
    assert sum(c.stats["evicted", p] for p in ("prompt", "decode")) == 0


def test_ranks_stay_in_step(lru):
    rng = random.Random(7)
    seq = [((rng.choice("pd"), rng.randrange(20)), rng.choice(("prompt", "decode"))) for _ in range(3000)]
    a, b = lru.HpuGraphLRU(16, 2), lru.HpuGraphLRU(16, 2)
    assert run(a, seq, 16) == run(b, seq, 16) and list(a) == list(b)
    assert max(len(a), len(b)) <= 16


def test_clear_releases_every_graph_once(lru):
    released = []
    c = lru.HpuGraphLRU(4, 1, destroy=released.append)
    run(c, [(("d", i), "decode") for i in range(4)], 4)
    c.clear()
    assert len(c) == 0 and len(released) == 4


def _stub_habana(monkeypatch, wrapped):
    calls = []
    hpu = types.SimpleNamespace(
        wrap_in_hpu_graph=lambda m, **kw: calls.append(("stock", kw)) or m,
        default_stream=lambda: "stream",
    )
    torch_mod = types.ModuleType("habana_frameworks.torch")
    torch_mod.hpu = hpu
    graphs = types.ModuleType("habana_frameworks.torch.hpu.graphs")
    graphs.wrapped_hpugraph_forward = wrapped
    hpu_mod = types.ModuleType("habana_frameworks.torch.hpu")
    hpu_mod.graphs = graphs
    pkg = types.ModuleType("habana_frameworks")
    pkg.torch = torch_mod
    for name, mod in (
        ("habana_frameworks", pkg),
        ("habana_frameworks.torch", torch_mod),
        ("habana_frameworks.torch.hpu", hpu_mod),
        ("habana_frameworks.torch.hpu.graphs", graphs),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _wrapped(cache, stream, orig_fwd, args, kwargs, dtc, asynchronous, dry_run, max_graphs, a, b, c):
    """The decision of Habana's wrapped_hpugraph_forward, with a call counter as the graph."""
    key = (tuple(args), kwargs["attn_metadata"].is_prompt)
    entry = cache.get(key)
    if entry is None:
        if max_graphs is not None and len(cache) == max_graphs:
            return ("uncaptured", orig_fwd(*args, **kwargs))
        cache[key] = entry = types.SimpleNamespace(graph=types.SimpleNamespace(reset=lambda: RESETS.append(key)))
        return ("capture", orig_fwd(*args, **kwargs))
    return ("replay", None)


RESETS = []


class _Module:
    def forward(self, *args, **kwargs):
        return sum(args)


def _meta(prompt):
    return types.SimpleNamespace(is_prompt=prompt)


def test_wrapper_unset_cap_is_stock(lru, monkeypatch):
    calls = _stub_habana(monkeypatch, _wrapped)
    monkeypatch.delenv("VLLM_HPU_MAX_GRAPHS", raising=False)
    m = _Module()
    assert lru._hgl_wrap(m) is m and calls == [("stock", {"disable_tensor_cache": True})]
    assert not hasattr(m, "graph_cache")


def test_wrapper_routes_through_cache(lru, monkeypatch):
    _stub_habana(monkeypatch, _wrapped)
    syncs = []
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(hpu=types.SimpleNamespace(synchronize=lambda: syncs.append(1)))
    )
    monkeypatch.setenv("VLLM_HPU_MAX_GRAPHS", "2")
    monkeypatch.setenv("VLLM_HPU_GRAPH_STATS_INTERVAL", "0")
    RESETS.clear()
    m = lru._hgl_wrap(_Module())
    seq = [(1,), (2,), (3,), (3,), (3,)]
    got = [m.forward(*a, attn_metadata=_meta(False))[0] for a in seq]
    assert got == ["capture", "capture", "uncaptured", "capture", "replay"]
    assert len(m.graph_cache) == 2 and m.graph_cache.stats["evicted", "decode"] == 1
    assert RESETS == [((1,), False)] and syncs == [1]


def test_wrapper_refuses_unknown_signature(lru, monkeypatch):
    _stub_habana(monkeypatch, lambda cache, stream, orig_fwd, args, kwargs: None)
    monkeypatch.setenv("VLLM_HPU_MAX_GRAPHS", "2")
    with pytest.raises(RuntimeError, match="signature"):
        lru._hgl_wrap(_Module())
