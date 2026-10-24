# SPDX-License-Identifier: Apache-2.0
"""Health handling for a service that drives a Gaudi card directly (not through vLLM).

A Synapse fatality is terminal for the process, not the request: every later HPU op raises the same error. It
surfaces as an exception inside a request, where a broad `except` would turn it into a 500 and leave the health
check green. A wedged card raises nothing at all; the call never returns. So:

* `note_fatal(exc)` latches the first fatality (narrow markers, so a bad request or a transient device OOM never
  takes a replica offline); `/health` then fails and the orchestrator restarts the process.
* `device_step()` stamps the start of each unit of device work; a watchdog thread latches a fault when one step
  runs past `deadline_s`. The deadline separates "slow" from "never returning", so set it far above any real step.
"""

import contextlib
import logging
import socket
import threading
import time

FATAL_MARKERS = ("synapse is being terminated", "fatal error :: module")

log = logging.getLogger("hpu-service")


class Health:
    def __init__(self, deadline_s: float = 300.0, poll_s: float = 30.0, on_fatal=None):
        self.deadline_s = deadline_s
        self.poll_s = poll_s
        self.on_fatal = on_fatal  # called once with the reason (sets a metric, for example)
        self.fatal: str | None = None
        self.loading = True
        self.startup_error: str | None = None
        self._step_started: float | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- latching
    def _latch(self, reason: str) -> None:
        if self.fatal is None:
            self.fatal = reason[:500]
            log.error("[fatal] failing health checks so the process is restarted: %s", self.fatal)
            if self.on_fatal is not None:
                self.on_fatal(self.fatal)

    def note_fatal(self, exc, force: bool = False) -> bool:
        """Latch if `exc` means this process can never use the HPU again. `force` is for a dead device loop:
        every request goes through it, so the replica serves nothing whatever the cause."""
        if not force and not any(m in str(exc).lower() for m in FATAL_MARKERS):
            return False
        self._latch(f"{type(exc).__name__}: {exc}" if force else str(exc))
        return True

    def failed_startup(self, exc) -> None:
        self.startup_error = f"{type(exc).__name__}: {exc}"
        self.note_fatal(exc)

    def ready(self) -> None:
        self.loading = False

    def status(self) -> tuple[int, dict]:
        """(HTTP status, body) for /health."""
        if self.fatal is not None:
            return 503, {"status": "faulted", "detail": self.fatal}
        if self.startup_error is not None:
            return 503, {"status": "error", "detail": self.startup_error}
        if self.loading:
            return 503, {"status": "loading"}
        return 200, {"status": "ok"}

    def unavailable(self) -> str | None:
        """Why requests should get a 503 now, or None."""
        if self.fatal is not None:
            return f"replica faulted and is restarting: {self.fatal}"
        if self.startup_error is not None:
            return self.startup_error
        if self.loading:
            return "The model is still loading."
        return None

    # ---------------------------------------------------------------- watchdog
    @contextlib.contextmanager
    def device_step(self):
        with self._lock:
            self._step_started = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._step_started = None

    def start_watchdog(self) -> None:
        threading.Thread(target=self._watch, name="hpu-watchdog", daemon=True).start()

    def _watch(self) -> None:
        while True:
            time.sleep(self.poll_s)
            with self._lock:
                started = self._step_started
            if self.fatal is None and started is not None:
                age = time.monotonic() - started
                if age > self.deadline_s:
                    self._latch(f"device wedged: one step has run {age:.0f}s (limit {self.deadline_s:.0f}s)")


class ProbeNoiseFilter(logging.Filter):
    """Drop successful probe and scrape lines from uvicorn's access log, so they do not rotate real events out
    of the container log. Failed probes still log."""

    QUIET_PATHS = ("/health", "/metrics")

    def filter(self, record) -> bool:
        args = getattr(record, "args", None)
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        try:
            status = int(args[4])
        except (TypeError, ValueError):
            return True
        return status >= 400 or str(args[2]).split("?", 1)[0] not in self.QUIET_PATHS


def epipe_selfcheck(device_op) -> None:
    """Provoke an EPIPE in a fresh thread, then run `device_op()` (a tiny HPU computation that returns True when
    its result is right). Raises if the HPU context did not survive, which means SIGPIPE was not blocked."""

    def provoke():
        a, b = socket.socketpair()
        b.close()
        try:
            for _ in range(4):
                a.send(b"\0" * 65536)
        except OSError:
            pass
        finally:
            a.close()

    t = threading.Thread(target=provoke)
    t.start()
    t.join()
    if not device_op():
        raise RuntimeError("HPU check after EPIPE returned a wrong value")
    log.info("EPIPE self-check passed: SIGPIPE is blocked and the HPU context survived")
