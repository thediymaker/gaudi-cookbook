#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stall-aware readiness probe for a vLLM pod (exec probe, standard library only).

vLLM's /health only says that the API server process is up. When the engine hangs (for
example a multi-card collective that never completes), /health keeps returning 200, the pod
stays Ready and every request routed to it waits forever. This probe also fails readiness
when requests are in flight but the engine has not finished a single step for a while:

  fail  if /health fails
  fail  if STALL_PROBE_READY_FILE is set and that file does not exist yet
  fail  if running + waiting > 0 and the step counter has not moved for STALL_PROBE_SECONDS
  pass  otherwise (also when /metrics cannot be read: /health decides then)

The step counter is the sample count of the vllm:iteration_tokens_total histogram, which
grows by one per engine step. An idle engine resets the clock, so only time spent busy
without progress counts. State lives in STALL_PROBE_STATE (under /tmp, so a restarted
container starts clean).

Use it as a readiness probe only. The pod leaves the Service and the router while it is
stalled and comes back by itself once steps resume; nothing is restarted. Leave liveness on
/health so a dead server is still restarted.

Choosing STALL_PROBE_SECONDS: it must be longer than the longest legitimate busy period with
no finished step on your model (the first compile of a new graph shape on a cold recipe
cache is the usual one, often tens of seconds) and shorter than the hangs you want to cut
short. Scan your own metrics for busy-without-progress gaps before picking a value.

STALL_PROBE_READY_FILE (optional) lets a start-up script warm the pod before it takes
traffic: the script sends its warm-up requests to localhost, then touches the file.

Environment:
  STALL_PROBE_PORT         vLLM port (default 8000)
  STALL_PROBE_SECONDS      busy-without-progress limit in seconds (default 180)
  STALL_PROBE_STATE        state file (default /tmp/vllm-stall-probe.json)
  STALL_PROBE_STEP_METRIC  step counter (default vllm:iteration_tokens_total_count)
  STALL_PROBE_READY_FILE   if set, readiness also waits for this file to exist

Usage (container spec; mount this file from a ConfigMap):
  readinessProbe:
    exec: {command: [python3, -I, /opt/stall-probe/vllm_stall_probe.py]}
    periodSeconds: 10
    timeoutSeconds: 10
    failureThreshold: 3
"""

import json
import os
import re
import sys
import time
import urllib.request

PORT = int(os.environ.get("STALL_PROBE_PORT", "8000"))
LIMIT = float(os.environ.get("STALL_PROBE_SECONDS", "180"))
STATE = os.environ.get("STALL_PROBE_STATE", "/tmp/vllm-stall-probe.json")
STEP_METRIC = os.environ.get("STALL_PROBE_STEP_METRIC", "vllm:iteration_tokens_total_count")
READY_FILE = os.environ.get("STALL_PROBE_READY_FILE", "")


def get(path: str, timeout: float) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as resp:
        return resp.read().decode()


def total(text: str, name: str) -> float:
    """Sum every sample of one metric across its label sets."""
    pattern = rf"^{re.escape(name)}(?:\{{[^}}]*\}})? (\S+)$"
    return sum(float(v) for v in re.findall(pattern, text, re.M))


def save(steps: float, since: float) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"steps": steps, "since": since}, fh)
    os.replace(tmp, STATE)


def main() -> int:
    try:
        get("/health", 5)
    except Exception as exc:
        print(f"health: {exc}")
        return 1
    if READY_FILE and not os.path.exists(READY_FILE):
        print(f"waiting for {READY_FILE}")
        return 1
    try:
        metrics = get("/metrics", 5)
    except Exception as exc:
        print(f"metrics unreadable ({exc}); /health ok")
        return 0
    steps = total(metrics, STEP_METRIC)
    busy = total(metrics, "vllm:num_requests_running") + total(metrics, "vllm:num_requests_waiting")
    now = time.time()
    try:
        with open(STATE) as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    if busy == 0 or state.get("steps") != steps:
        save(steps, now)
        return 0
    stalled = now - state.get("since", now)
    if stalled >= LIMIT:
        print(f"stalled: {busy:.0f} request(s) in flight, no engine step for {stalled:.0f} s")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
