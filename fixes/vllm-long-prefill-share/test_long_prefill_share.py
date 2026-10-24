# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm-long-prefill-share.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift.
2. The scheduler helpers on stub requests (stdlib only, the clock is faked):
   - off unless VLLM_LONG_PREFILL_SHARE is between 0 and 1;
   - after a step with a long chunk that took T seconds, long chunks wait T * (1 - s) / s seconds,
     but only while another request decodes or a short request waits;
   - with VLLM_LONG_PREFILL_SHARE_ASYNC=1 a long chunk still in flight also defers the others.

Run with: pytest fixes/vllm-long-prefill-share
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = "v1/core/sched/scheduler.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "long_prefill_share_apply")


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
    path = site / "vllm" / REL
    assert _run(site, "--check") == 1
    assert _run(site) == 0
    text = path.read_text()
    assert fix_mod.MARKER in text
    compile(text, REL, "exec")
    assert _run(site) == 0 and path.read_text() == text
    assert _run(site, "--check") == 0


@pytest.mark.parametrize("idx", range(1, len(fix_mod.EDITS)))
def test_source_patch_refuses_drift(site, idx):
    path = site / "vllm" / REL
    path.write_text(path.read_text().replace(fix_mod.EDITS[idx][1], "", 1))
    before = path.read_text()
    assert _run(site) == 2 and path.read_text() == before


class Clock:
    def __init__(self):
        self.t = 100.0

    def monotonic(self):
        return self.t


def _helpers(monkeypatch, **env):
    for k in ("VLLM_LONG_PREFILL_SHARE", "VLLM_LONG_PREFILL_MIN_TOKENS", "VLLM_LONG_PREFILL_SHARE_ASYNC"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns = {}
    exec(compile(fix_mod.HELPERS, "helpers", "exec"), ns)
    clock = Clock()
    ns["_lps_time"] = clock
    return ns, clock


def req(prompt, computed, chunk=True, in_flight=0):
    return types.SimpleNamespace(
        num_prompt_tokens=prompt, num_computed_tokens=computed, is_prefill_chunk=chunk, num_in_flight_tokens=in_flight
    )


def sched(running, waiting=(), requests=None):
    return types.SimpleNamespace(running=list(running), waiting=list(waiting), requests=requests or {})


def step(ns, clock, s, took, scheduled):
    clock.t += took
    ns["_lps_step_done"](s, types.SimpleNamespace(num_scheduled_tokens=scheduled))


def test_off_by_default(monkeypatch):
    ns, clock = _helpers(monkeypatch)
    s = sched([req(200000, 1000), req(100, 120, chunk=False)])
    assert not ns["_LPS_ON"] and not ns["_lps_defer_long"](s)


def test_share_window(monkeypatch):
    ns, clock = _helpers(monkeypatch, VLLM_LONG_PREFILL_SHARE="0.25", VLLM_LONG_PREFILL_MIN_TOKENS="1000")
    long_r, chat = req(200000, 4096), req(100, 150, chunk=False)
    s = sched([long_r, chat], requests={"long": long_r, "chat": chat})
    step(ns, clock, s, 0.0, {})  # first completion only starts the clock
    step(ns, clock, s, 2.0, {"long": 4096, "chat": 1})  # a 2 s step with a long chunk
    assert ns["_lps_defer_long"](s)  # decode-only for 2 * 0.75 / 0.25 = 6 s
    clock.t += 5.9
    assert ns["_lps_defer_long"](s)
    clock.t += 0.2
    assert not ns["_lps_defer_long"](s)


def test_nobody_else_no_deferral(monkeypatch):
    ns, clock = _helpers(monkeypatch, VLLM_LONG_PREFILL_SHARE="0.5", VLLM_LONG_PREFILL_MIN_TOKENS="1000")
    long_r = req(200000, 4096)
    s = sched([long_r], requests={"long": long_r})
    step(ns, clock, s, 0.0, {})
    step(ns, clock, s, 2.0, {"long": 4096})
    assert not ns["_lps_defer_long"](s)  # alone: full speed
    s.waiting.append(req(50, 0))
    assert ns["_lps_defer_long"](s)  # a short request waiting to start gets in
    s.waiting[0] = req(5000, 0)
    assert not ns["_lps_defer_long"](s)  # another long one does not


def test_short_chunks_do_not_open_a_window(monkeypatch):
    ns, clock = _helpers(monkeypatch, VLLM_LONG_PREFILL_SHARE="0.5", VLLM_LONG_PREFILL_MIN_TOKENS="1000")
    short, chat = req(800, 0), req(100, 150, chunk=False)
    s = sched([short, chat], requests={"short": short, "chat": chat})
    step(ns, clock, s, 0.0, {})
    step(ns, clock, s, 3.0, {"short": 800, "chat": 1})
    assert not ns["_lps_defer_long"](s)


def test_async_in_flight(monkeypatch):
    long_r, chat = req(200000, 8192, in_flight=4096), req(100, 150, chunk=False)
    s = sched([long_r, chat])
    ns, clock = _helpers(monkeypatch, VLLM_LONG_PREFILL_SHARE="0.5", VLLM_LONG_PREFILL_MIN_TOKENS="1000")
    assert not ns["_lps_defer_long"](s)  # default: only the window counts
    ns, clock = _helpers(
        monkeypatch,
        VLLM_LONG_PREFILL_SHARE="0.5",
        VLLM_LONG_PREFILL_MIN_TOKENS="1000",
        VLLM_LONG_PREFILL_SHARE_ASYNC="1",
    )
    assert ns["_lps_defer_long"](s)
    long_r.num_computed_tokens = 200000 + 4096  # in-flight tokens are decode tokens now
    assert not ns["_lps_defer_long"](s)
