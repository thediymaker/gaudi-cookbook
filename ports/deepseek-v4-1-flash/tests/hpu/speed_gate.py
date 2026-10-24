#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode speed at fixed concurrency: N streams, each a different short prompt, greedy, ignore_eos, fixed length.
Reports per-stream decode tok/s (tokens after the first over the time after the first token, so prefill does not
count) and the aggregate.

PASS: every stream completes; with --min-tps, the 1-stream rate is at least that. The numbers depend on the node and
on what else the server is doing: run it on an otherwise idle server.

  python3 speed_gate.py [--conc 1,4] [--tokens 300] [--min-tps 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import NO_THINK, URL, model, verdict  # noqa: E402

TOPICS = [
    "volcanoes",
    "the history of chess",
    "how compilers work",
    "coral reefs",
    "the printing press",
    "jazz improvisation",
    "plate tectonics",
    "the immune system",
]


def stream_one(i: int, tokens: int, out: dict) -> None:
    body = {
        "model": model(),
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        **NO_THINK,
        "messages": [{"role": "user", "content": f"Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}."}],
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0, t1, n = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if d.get("choices") and t1 is None and d["choices"][0].get("delta", {}).get("content"):
                t1 = time.time()
    t2 = time.time()
    out[i] = {"tokens": n, "ttft": t1 - t0, "decode_tps": (n - 1) / (t2 - t1), "end": t2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", default="1,4")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--min-tps", type=float, default=0.0)
    args = ap.parse_args()
    stream_one(0, 32, {})  # touch the decode graphs once
    ok, one = True, None
    for c in [int(x) for x in args.conc.split(",")]:
        out: dict = {}
        threads = [threading.Thread(target=stream_one, args=(i, args.tokens, out)) for i in range(c)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = ok and len(out) == c and all(o["tokens"] == args.tokens for o in out.values())
        per = [round(o["decode_tps"], 2) for o in out.values()]
        wall = max(o["end"] for o in out.values()) - t0 if out else 0
        if c == 1 and per:
            one = per[0]
        print(
            json.dumps(
                {
                    "streams": c,
                    "per_stream_decode_tps": per,
                    "mean_per_stream": round(sum(per) / max(1, len(per)), 2),
                    "aggregate_decode_tps": round(sum(per), 1),
                    "aggregate_wall_tps": round(sum(o["tokens"] for o in out.values()) / max(wall, 1e-9), 1),
                    "ttft_s": [round(o["ttft"], 2) for o in out.values()],
                }
            ),
            flush=True,
        )
    if one is not None and one < args.min_tps:
        ok = False
    return 0 if verdict("SPEED", ok, f"1 stream {one} tok/s" if one is not None else "") else 1


if __name__ == "__main__":
    sys.exit(main())
