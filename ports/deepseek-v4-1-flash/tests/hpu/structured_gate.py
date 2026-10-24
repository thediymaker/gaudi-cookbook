#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structured output and tool calls under concurrent load, through the chat API.

With the deepseek_v41 reasoning parser the grammar applies only after the end of reasoning, so /v1/completions is not
constrained; this drives /v1/chat/completions. Each round sends, concurrently, groups of:

  json        response_format json_schema, thinking off: the reply must parse and match the schema
  json_think  the same with thinking on
  tool        tool_choice "required": a tool call with parseable arguments must come back
  tool_auto   tool_choice "auto", thinking off, a prompt that needs the tool: the deepseek_v41 tool parser must
              turn the model's own tool-call text into a tool call
  long        a ~20K-token plain prompt ending "Reply with exactly: DONE" (greedy), sent while grammar requests are
              in flight. A reply that opens with filler words (long_stray) carries tokens sampled at mid-prompt
              positions: structured-output row bookkeeping went wrong (fixes structured-output-rows, so-row-owners)

An HTTP 500 is a grammar rejection; a refused connection is a dead engine.

PASS: every row of every kind is ok.

  python3 structured_gate.py [--rounds 2] [--per-round 6] [--conc 12]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import post_status, verdict  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "maxLength": 24},
        "population": {"type": "integer"},
        "coastal": {"type": "boolean"},
    },
    "required": ["city", "population", "coastal"],
    "additionalProperties": False,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
CITIES = ["Paris", "Lima", "Osaka", "Cairo", "Oslo", "Perth"]
WORDS = ["kernel", "river", "signal", "matrix", "harbor", "cable"]
A = argparse.Namespace()


def think(on: bool) -> dict:
    return {"chat_template_kwargs": {"thinking": on}}


def ok_json(t: str) -> bool:
    try:
        d = json.loads(t)
        return (
            set(d) == {"city", "population", "coastal"}
            and isinstance(d["population"], int)
            and isinstance(d["coastal"], bool)
        )
    except Exception:
        return False


def js(tag: str, on: bool) -> tuple:
    kind = "json_think" if on else "json"
    st, r = post_status(
        "/v1/chat/completions",
        {
            "max_tokens": A.json_max_tokens,
            "temperature": 0.7,
            **think(on),
            "messages": [{"role": "user", "content": f"Describe a coastal city as JSON ({tag})."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA}},
        },
    )
    if st != 200:
        return (kind, st, False, str(r)[:120])
    c = r["choices"][0]["message"].get("content") or ""
    return (kind, st, ok_json(c), c[:120])


def tool(tag: str, choice: str) -> tuple:
    kind = "tool" if choice == "required" else "tool_auto"
    city = CITIES[sum(map(ord, tag)) % len(CITIES)]
    body = {
        "max_tokens": A.json_max_tokens,
        "tools": TOOLS,
        "tool_choice": choice,
        "messages": [{"role": "user", "content": f"What is the weather in {city} right now? ({tag})"}],
    }
    if choice == "auto":
        body.update(think(False), temperature=0)
    st, r = post_status("/v1/chat/completions", body)
    if st != 200:
        return (kind, st, False, str(r)[:120])
    tc = r["choices"][0]["message"].get("tool_calls") or []
    try:
        ok = len(tc) >= 1 and tc[0]["function"]["name"] == "get_weather"
        ok = ok and "city" in json.loads(tc[0]["function"]["arguments"])
    except Exception:
        ok = False
    detail = tc[0]["function"]["arguments"][:80] if tc else (r["choices"][0]["message"].get("content") or "")[:80]
    return (kind, st, ok, detail)


def long_words(seed: int, n: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n))


def longplain(seed: int) -> tuple:
    st, r = post_status(
        "/v1/chat/completions",
        {
            "max_tokens": 64,
            "temperature": 0,
            **think(False),
            "messages": [{"role": "user", "content": long_words(seed, A.long_words) + "\nReply with exactly: DONE"}],
        },
    )
    c = (r["choices"][0]["message"].get("content") or "") if st == 200 else str(r)[:120]
    first = (c.split() or [""])[0].strip(".,")
    kind = "long_stray" if st == 200 and first in WORDS else "long"
    return (kind, st, st == 200 and c.strip().startswith("DONE"), c[:80])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--per-round", type=int, default=6, help="groups of (json, json_think, tool, tool_auto, long)")
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--stagger", type=float, default=0.5, help="seconds between groups")
    ap.add_argument("--long-words", type=int, default=20000)
    ap.add_argument("--json-max-tokens", type=int, default=2000)
    ap.parse_args(namespace=A)
    total: Counter = Counter()
    bad = []
    for rd in range(A.rounds):
        base = random.randrange(1 << 30)
        with cf.ThreadPoolExecutor(A.conc) as ex:
            jobs = []
            for i in range(A.per_round):
                tag = f"{base}-{i}"
                jobs += [
                    ex.submit(js, tag, False),
                    ex.submit(js, tag, True),
                    ex.submit(tool, tag, "required"),
                    ex.submit(tool, tag, "auto"),
                    ex.submit(longplain, base + i),
                ]
                time.sleep(A.stagger)
            res = [j.result() for j in jobs]
        c = Counter((k, st, ok) for k, st, ok, _ in res)
        print(
            f"round {rd}:", {f"{k}/{st}/{'ok' if ok else 'BAD'}": v for (k, st, ok), v in sorted(c.items())}, flush=True
        )
        total.update(c)
        bad += [r for r in res if not r[2]]
    for r in bad:
        print("BAD", r)
    n = sum(total.values())
    return 0 if verdict("STRUCTURED", not bad, f"{n - len(bad)}/{n} rows ok") else 1


if __name__ == "__main__":
    sys.exit(main())
