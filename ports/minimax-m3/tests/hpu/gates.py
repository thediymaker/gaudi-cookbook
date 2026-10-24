#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving gates for the MiniMax-M3 port, run against a server started with the recipe's command.

Prints one PASS or FAIL line per gate and exits non-zero if any gate fails. Synthetic prompts only; stdlib only.

  golden     5 greedy prompts: 17 * 23, an is_prime function, the capital of Australia, a train time, and a
             get_weather tool call
  loops      10 short greedy prompts with a 2048-token budget: none may end with finish_reason "length"
  needles    3 codes at 10/50/90% depth in synthetic text of 16K, 32K, 64K, 100K and 125K tokens: all found
  logprobs   prompt_logprobs aligned with the prompt over a 2-chunk prompt; top_logprobs ids inside the
             vocabulary for 16 concurrent requests; the server stays healthy
  reasoning  streamed and non-streamed chat: the reasoning arrives in `reasoning`, no <mm:think> marker in
             `content`; a streamed tool call carries its name and arguments
  speed      warm decode tok/s per stream at 1 and 8 concurrent streams (2K prompt, 256 tokens)

  gates.py --base http://127.0.0.1:8000 --model amd/MiniMax-M3-MXFP4 [--skip needles,speed]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

VOCAB = 200064
RESULTS: list[tuple[str, bool, str]] = []


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, model, messages, max_tokens=1024, **extra):
    d = post(
        base,
        "/v1/chat/completions",
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra},
    )
    c = d["choices"][0]
    m = c["message"]
    return {
        "finish": c["finish_reason"],
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
        "tools": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
        "ptok": d["usage"]["prompt_tokens"],
        "ctok": d["usage"]["completion_tokens"],
        "raw": d,
    }


def stream_chat(base, model, messages, max_tokens=1024, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, **extra}
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    content, reasoning, names, args = [], [], [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                d = ch.get("delta") or {}
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
                for t in d.get("tool_calls") or []:
                    f = t.get("function") or {}
                    names.append(f.get("name") or "")
                    args.append(f.get("arguments") or "")
    return "".join(content), "".join(reasoning), "".join(names), "".join(args)


def gate(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def g_golden(base, model):
    cases = [
        ("What is 17 * 23? Answer with just the number.", lambda r: "391" in r["content"]),
        (
            "Write a Python function is_prime(n) that returns True if n is prime. Code only.",
            lambda r: "def is_prime" in r["content"],
        ),
        ("Name the capital of Australia and one fact about it, in one sentence.", lambda r: "Canberra" in r["content"]),
        (
            "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the arithmetic briefly.",
            lambda r: any(
                s in r["content"].replace(" ", " ")
                for s in ("3 hours and 25", "3 hours 25", "3h 25", "205 minutes", "3 h 25")
            ),
        ),
    ]
    ok, notes = 0, []
    for prompt, check in cases:
        r = chat(base, model, [{"role": "user", "content": prompt}], 4096)
        good = check(r) and r["finish"] == "stop"
        ok += good
        notes.append(f"{'ok' if good else 'BAD'}:{r['content'][:40]!r}")
    r = chat(
        base, model, [{"role": "user", "content": "What's the weather in Phoenix right now?"}], 4096, tools=WEATHER
    )
    good = any(n == "get_weather" and "Phoenix" in a for n, a in r["tools"])
    ok += good
    notes.append(f"{'ok' if good else 'BAD'}:tool {r['tools']}")
    gate("golden", ok == 5, f"{ok}/5 " + " | ".join(notes))


LOOP = [
    "Name the capital of Australia and one fact about it, in one sentence.",
    "What is the boiling point of water at sea level in Fahrenheit? One sentence.",
    "Give three synonyms for 'quick'.",
    "Translate 'good morning, how are you?' into Spanish.",
    "What is 144 divided by 12? Just the number.",
    "Write a haiku about autumn leaves.",
    "Who wrote 'Pride and Prejudice'? One sentence.",
    "Explain in two sentences what a hash table is.",
    "List the first five prime numbers.",
    "What does HTTP stand for?",
]


def g_loops(base, model):
    with ThreadPoolExecutor(2) as ex:
        rr = list(ex.map(lambda p: chat(base, model, [{"role": "user", "content": p}], 2048), LOOP))
    n = sum(r["finish"] == "length" for r in rr)
    gate("loops", n == 0, f"finish=length {n}/10, completion tokens {[r['ctok'] for r in rr]}")


TOPICS = [
    "The survey team logged tidal measurements along the northern inlet",
    "Maintenance records for the pumping station were filed out of order",
    "A revised schedule moved the inspection window into the following quarter",
    "The soil samples were catalogued by depth rather than by collection date",
    "Correspondence about the boundary fence continued for several months",
]
TAILS = [
    "and the results were entered into the register without further comment",
    "which delayed the report but did not change any of its conclusions",
    "before the file was passed to the regional office for archiving",
]
CODES = [("northern", "PELICAN-4471"), ("central", "OSPREY-8802"), ("southern", "KESTREL-1359")]


def haystack(n, nonce):
    parts = [f"{TOPICS[i % len(TOPICS)]} in section {i}, {TAILS[(i * 7) % len(TAILS)]}" for i in range(n)]
    for depth, (w, c) in zip((0.1, 0.5, 0.9), CODES, strict=True):
        parts.insert(
            int(len(parts) * depth), f"IMPORTANT FACT: the survey reference number for the {w} district is {c}"
        )
    q = (
        "\n\nQuestion: what are the survey reference numbers for the northern, central and southern districts? "
        "Reply with the three codes only, one per line, in that order."
    )
    return [{"role": "user", "content": f"Batch {nonce}.\n\n" + ". ".join(parts) + "." + q}]


def g_needles(base, model, targets, max_len=131072, answer=2048):
    # Section numbers grow with the haystack, so calibrate tokens per sentence near the sizes tested.
    a = chat(base, model, haystack(1000, "cal1"), 1)["ptok"]
    b = chat(base, model, haystack(3000, "cal2"), 1)["ptok"]
    per = (b - a) / 2000
    found, notes = 0, []
    for t in targets:
        n = int(min(t * 1000, max_len - answer - 256) / per) - int(a / per) + 1000
        try:
            r = chat(base, model, haystack(n, f"{random.randrange(16**8):08x}"), answer)
        except urllib.error.HTTPError as e:
            notes.append(f"~{t:g}K HTTP {e.code}")
            continue
        hit = sum(c in r["content"] + r["reasoning"] for _, c in CODES)
        found += hit == 3
        notes.append(f"{r['ptok']} tok {hit}/3")
    gate("needles", found == len(targets), f"{found}/{len(targets)} " + ", ".join(notes))


def g_logprobs(base, model):
    text = "".join(
        f"The survey team logged tidal measurements in section {i}, and the results were entered. " for i in range(700)
    )
    ids = post(base, "/tokenize", {"model": model, "prompt": text})["tokens"]
    d = post(
        base,
        "/v1/completions",
        {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 2},
    )
    plp = d["choices"][0]["prompt_logprobs"]
    bad = 0
    for i, (t, e) in enumerate(zip(ids, plp, strict=True)):
        if i == 0:
            continue
        if e is None or str(t) not in e:
            bad += 1
            continue
        for k, v in e.items():
            lp = v["logprob"]
            bad += not (0 <= int(k) < VOCAB) or lp is None or not math.isfinite(lp) or lp > 1e-4
    errors = []

    def one(i):
        try:
            r = chat(
                base,
                model,
                [{"role": "user", "content": f"Say a random word, request {i}."}],
                32,
                logprobs=True,
                top_logprobs=5,
                chat_template_kwargs={"thinking_mode": "disabled"},
            )
            for tok in r["raw"]["choices"][0]["logprobs"]["content"]:
                for alt in tok["top_logprobs"]:
                    if "<bad_id>" in alt["token"] or not math.isfinite(alt["logprob"]):
                        errors.append(alt)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, range(16)))
    with urllib.request.urlopen(base + "/health", timeout=30) as h:
        healthy = h.status == 200
    gate(
        "logprobs",
        bad == 0 and not errors and healthy,
        f"prompt_logprobs {len(ids)} tokens, {bad} bad entries; top_logprobs 16 requests, {len(errors)} errors; "
        f"health {'ok' if healthy else 'BAD'}",
    )


def g_reasoning(base, model):
    msgs = [{"role": "user", "content": "Is 391 a prime number? Think it through, then answer yes or no."}]
    notes, ok = [], True
    c, r, _, _ = stream_chat(base, model, msgs, 4096)
    good = bool(r) and "<mm:think>" not in c and "</mm:think>" not in c and bool(c.strip())
    ok &= good
    notes.append(f"streamed reasoning={len(r)} content={len(c)} {'ok' if good else 'BAD'}")
    n = chat(base, model, msgs, 4096)
    good = bool(n["reasoning"]) and "<mm:think>" not in n["content"]
    ok &= good
    notes.append(f"non-streamed reasoning={len(n['reasoning'])} {'ok' if good else 'BAD'}")
    c, r, name, args = stream_chat(
        base, model, [{"role": "user", "content": "What's the weather in Phoenix?"}], 4096, tools=WEATHER
    )
    good = name == "get_weather" and "Phoenix" in args and "<mm:think>" not in c
    ok &= good
    notes.append(f"streamed tool call {name}({args}) {'ok' if good else 'BAD'}")
    gate("reasoning", ok, "; ".join(notes))


def g_speed(base, model, floor):
    filler = " ".join(f"Entry {i} records the weather at the ridge station." for i in range(220))

    def one(_):
        body = {
            "model": model,
            "prompt": f"{random.random()} {filler}\nContinue the log:",
            "max_tokens": 256,
            "temperature": 0,
            "ignore_eos": True,
        }
        t0 = time.time()
        d = post(base, "/v1/completions", body)
        return d["usage"]["completion_tokens"] / (time.time() - t0)

    def run(conc, rounds=2):
        rate = 0.0
        for _ in range(rounds):  # the first round compiles new shapes; report the last
            with ThreadPoolExecutor(conc) as ex:
                rates = list(ex.map(one, range(conc)))
            rate = sum(rates) / len(rates)
        return rate

    r1, r8 = run(1), run(8)
    gate(
        "speed",
        r1 >= floor and r8 >= floor,
        f"per-stream tok/s incl. prefill: 1 stream {r1:.1f}, 8 streams {r8:.1f} (floor {floor})",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="amd/MiniMax-M3-MXFP4")
    ap.add_argument("--needles", default="16,32,64,100,125", help="needle prompt sizes in thousands of tokens")
    ap.add_argument("--speed-floor", type=float, default=15.0, help="minimum tok/s per stream")
    ap.add_argument("--skip", default="", help="comma list of gates to skip")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    gates = {
        "golden": lambda: g_golden(a.base, a.model),
        "loops": lambda: g_loops(a.base, a.model),
        "needles": lambda: g_needles(a.base, a.model, [float(x) for x in a.needles.split(",")]),
        "logprobs": lambda: g_logprobs(a.base, a.model),
        "reasoning": lambda: g_reasoning(a.base, a.model),
        "speed": lambda: g_speed(a.base, a.model, a.speed_floor),
    }
    for name, fn in gates.items():
        if name in skip:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            gate(name, False, f"error {e!r}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
