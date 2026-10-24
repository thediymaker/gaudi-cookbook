# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the health latch, the watchdog, the log filter and the SIGPIPE guard (`pytest servers/common`)."""

import logging
import os
import signal
import time

import hpu_service as hs
import sigpipe_guard


def test_sigpipe_blocked_and_epipe_is_an_exception():
    assert sigpipe_guard.is_blocked()
    r, w = os.pipe()
    os.close(r)
    try:
        os.write(w, b"x")
        raise AssertionError("write to a closed pipe succeeded")
    except BrokenPipeError:
        pass  # the process is still here: the signal was blocked, the write failed with EPIPE
    finally:
        os.close(w)
    assert signal.SIGPIPE in signal.pthread_sigmask(signal.SIG_BLOCK, set())


def test_epipe_selfcheck_runs_the_device_op():
    calls = []
    hs.epipe_selfcheck(lambda: calls.append(1) or True)
    assert calls == [1]


def test_only_fatal_markers_latch():
    seen = []
    h = hs.Health(on_fatal=seen.append)
    h.ready()
    assert not h.note_fatal(RuntimeError("out of device memory"))
    assert h.status()[0] == 200
    assert h.note_fatal(RuntimeError("synStatus 31 [Synapse is being terminated due to fatality]"))
    assert h.note_fatal(RuntimeError("Fatal error :: module 3"))  # fatal too, but the first fault is kept
    assert h.status()[0] == 503 and "Synapse" in h.fatal and len(seen) == 1
    assert h.unavailable().startswith("replica faulted")


def test_forced_latch_and_startup_states():
    h = hs.Health()
    assert h.status() == (503, {"status": "loading"})
    h.failed_startup(ValueError("no weights"))
    assert h.status()[1]["status"] == "error"
    h2 = hs.Health()
    h2.ready()
    assert h2.note_fatal(KeyError("loop died"), force=True) and h2.status()[0] == 503


def test_watchdog_latches_a_step_that_never_returns():
    h = hs.Health(deadline_s=0.2, poll_s=0.05)
    h.ready()
    h.start_watchdog()
    with h.device_step():
        time.sleep(0.5)
    assert h.fatal and "wedged" in h.fatal


def test_watchdog_ignores_idle_and_short_steps():
    h = hs.Health(deadline_s=0.3, poll_s=0.05)
    h.ready()
    h.start_watchdog()
    for _ in range(5):
        with h.device_step():
            time.sleep(0.05)
    time.sleep(0.4)  # idle longer than the deadline
    assert h.fatal is None


def test_probe_filter():
    f = hs.ProbeNoiseFilter()

    def rec(path, status):
        return logging.LogRecord(
            "uvicorn.access", logging.INFO, "", 0, "%s", ("1.2.3.4", "GET", path, "1.1", status), None
        )

    assert not f.filter(rec("/health", 200))
    assert not f.filter(rec("/metrics?x=1", 200))
    assert f.filter(rec("/health", 503))
    assert f.filter(rec("/healthz", 200))
    assert f.filter(rec("/v1/audio/speech", 200))
