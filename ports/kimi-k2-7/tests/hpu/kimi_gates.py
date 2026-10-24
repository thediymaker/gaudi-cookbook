#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving gates and measurements for a Kimi-K2.7-Code endpoint. Standard library only.

  golden OUT.json          greedy answers to fixed prompts with known answers; PASS if every answer is right
  tf OUT.json              teacher-forced scoring: prompt_logprobs over fixed code texts (top-1 ids, logprobs)
  compare A.json B.json    agreement of two golden or two tf files (same endpoint, two settings)
  tools                    tool call, non-streaming and streaming (kimi_k2 parser)
  needle T1,T2,...         needle recall at the given prompt lengths (3 depths each)
  logprobs N               N requests with top_logprobs=20 and CJK output: every position complete and finite
  sweep CTX C1,C2,...      decode speed on a CTX-token cached prefix: per-stream and aggregate tok/s per level
  ttft CTX                 time to first token of an uncached CTX-token prompt (3 runs)

Environment: BASE (default http://localhost:8000), MODEL (default: the first model the server lists),
GEN (sweep output tokens, default 256). Each gate prints one line that starts with PASS or FAIL.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import statistics as st
import sys
import time
import urllib.request
import uuid

BASE = os.environ.get("BASE", "http://localhost:8000")
GEN = int(os.environ.get("GEN", "256"))
_MODEL = os.environ.get("MODEL")


def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)


def post(path, body, timeout=3600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model():
    global _MODEL
    if not _MODEL:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
            _MODEL = json.loads(r.read())["data"][0]["id"]
    return _MODEL


def chat(messages, max_tokens=4096, **kw):
    body = {"model": model(), "messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw}
    return post("/v1/chat/completions", body)


def ntok(text):
    return len(post("/tokenize", {"model": model(), "prompt": text})["tokens"])


# ------------------------------------------------------------------------------------------------ golden
GOLDEN = [
    ("What is 17 * 23? Reply with just the number.", ["391"]),
    ("What is the capital of Australia? One word.", ["Canberra"]),
    ("How many sides does a hexagon have? Reply with just the number.", ["6"]),
    ("What is the chemical symbol for gold? Reply with just the symbol.", ["Au"]),
    ("Reverse the string 'gaudi' and reply with just the result.", ["iduag"]),
    ("What is 2 to the power of 10? Reply with just the number.", ["1024"]),
    ("Which planet is known as the Red Planet? One word.", ["Mars"]),
    ("Write a Python expression that returns the length of a list named xs. Reply with just the code.", ["len(xs)"]),
    ("What is the square root of 144? Reply with just the number.", ["12"]),
    ("In which year did the first human land on the Moon? Reply with just the year.", ["1969"]),
    ("What is the next prime number after 31? Reply with just the number.", ["37"]),
    ("Translate 'thank you' into French. Reply with just the translation.", ["merci", "Merci"]),
]


def golden(out):
    rows, ok = [], 0
    for prompt, answers in GOLDEN:
        m = chat([{"role": "user", "content": prompt}], max_tokens=4096)["choices"][0]["message"]
        text = m.get("content") or ""
        hit = any(a in text for a in answers)
        ok += hit
        rows.append({"prompt": prompt, "content": text, "reasoning": m.get("reasoning_content") or "", "ok": hit})
    save(out, {"kind": "golden", "rows": rows})
    print(f"{'PASS' if ok == len(GOLDEN) else 'FAIL'} golden {ok}/{len(GOLDEN)} answers correct ({out})")
    return ok == len(GOLDEN)


# ------------------------------------------------------------------------------------------------ teacher-forced
def tf_texts():
    """Real code text for scoring: this repository's own sources (deterministic, in every port image)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    picks = ["fixes/apply_all.py", "kernels/mxfp4-moe/build.sh", "ports/kimi-k2-7/kimi_hpu/kimi_hpu/mxk.py"]
    texts = []
    for p in picks:
        path = os.path.join(root, p)
        if os.path.isfile(path):
            with open(path) as f:
                texts.append(f.read()[:12000])
    return texts


def tf(out):
    rows = []
    for text in tf_texts():
        r = post(
            "/v1/completions",
            {"model": model(), "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1},
        )
        plp = r["choices"][0]["prompt_logprobs"]
        ids = post("/tokenize", {"model": model(), "prompt": text})["tokens"]
        if len(ids) != len(plp):
            sys.exit(f"tf: {len(ids)} prompt tokens but {len(plp)} prompt_logprobs positions")
        lps, top1 = [], []
        # position i maps token id -> {logprob, rank}: the prompt token itself and the rank-1 token
        for tok, pos in zip(ids[1:], plp[1:], strict=True):
            lps.append(float(pos[str(tok)]["logprob"]))
            top1.append(int(min(pos.items(), key=lambda kv: kv[1]["rank"])[0]))
        rows.append({"n": len(lps), "logprob": lps, "top1": top1})
    save(out, {"kind": "tf", "rows": rows})
    n = sum(r["n"] for r in rows)
    nll = -sum(sum(r["logprob"]) for r in rows) / max(1, n)
    print(f"PASS tf {n} positions scored, NLL {nll:.4f} ({out})")
    return True


def compare(a, b):
    A, B = load(a), load(b)
    if A["kind"] != B["kind"]:
        sys.exit("different file kinds")
    if A["kind"] == "golden":
        same = sum(x["content"].strip() == y["content"].strip() for x, y in zip(A["rows"], B["rows"], strict=True))
        oka = sum(x["ok"] for x in A["rows"])
        okb = sum(y["ok"] for y in B["rows"])
        print(f"golden: identical answers {same}/{len(A['rows'])}; correct {oka} vs {okb}")
        print(f"{'PASS' if oka == okb == len(A['rows']) else 'FAIL'} golden agreement (both sides all correct)")
        return oka == okb == len(A["rows"])
    agree = tot = 0
    nll_a = nll_b = 0.0
    for x, y in zip(A["rows"], B["rows"], strict=True):
        for i in range(min(x["n"], y["n"])):
            agree += x["top1"][i] == y["top1"][i]
            tot += 1
            nll_a -= x["logprob"][i]
            nll_b -= y["logprob"][i]
    rate = agree / max(1, tot)
    print(f"teacher-forced: top-1 agreement {agree}/{tot} = {rate:.4f}; NLL {nll_a / tot:.4f} vs {nll_b / tot:.4f}")
    ok = rate >= 0.95 and abs(nll_a - nll_b) / tot <= 0.02
    print(f"{'PASS' if ok else 'FAIL'} teacher-forced agreement (top-1 >= 0.95, NLL within 0.02)")
    return ok


# ------------------------------------------------------------------------------------------------ tools
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def tools():
    msgs = [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]
    m = chat(msgs, tools=[WEATHER], tool_choice="auto")["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    ok1 = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "city" in args
    body = {
        "model": model(),
        "messages": msgs,
        "tools": [WEATHER],
        "tool_choice": "auto",
        "max_tokens": 4096,
        "temperature": 0,
        "stream": True,
    }
    name, argbuf = None, ""
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = name or fn.get("name")
                    argbuf += fn.get("arguments") or ""
    try:
        sargs = json.loads(argbuf)
    except ValueError:
        sargs = None
    ok2 = name == "get_weather" and isinstance(sargs, dict) and "city" in sargs
    print(f"{'PASS' if ok1 and ok2 else 'FAIL'} tools non-streaming {args} streaming {sargs}")
    return ok1 and ok2


# ------------------------------------------------------------------------------------------------ needle
PLACES = ["north", "south", "east", "west", "harbour", "hill", "river", "airport", "market", "station"]
ITEMS = ["valves", "gaskets", "bearings", "relays", "sensors", "pumps", "filters", "cables", "brackets", "fuses"]
VERBS = ["shipped", "received", "returned", "inspected", "rejected", "stocked", "repaired", "ordered"]


def needle_prompt(target, depth, seed):
    # varied records, not one repeated sentence: repeated filler makes models degenerate for the wrong reason
    rng = random.Random(seed)
    code = str(rng.randint(100000, 999999))
    recs = [
        f"Record {rng.randint(10000, 99999)}: the {rng.choice(PLACES)} depot {rng.choice(VERBS)} "
        f"{rng.randint(2, 900)} {rng.choice(ITEMS)} on day {rng.randint(1, 365)}, batch "
        f"{rng.choice('ABCDEFGHJK')}{rng.randint(10, 99)}."
        for _ in range(max(50, target // 20))
    ]
    per = ntok(" ".join(recs[:400])) / 400.0
    recs = recs[: max(10, int((target - 100) / per))]
    recs.insert(int(len(recs) * depth), f"IMPORTANT: the access code for the north vault is {code}.")
    q = "\n\nUsing only the records above: what is the access code for the north vault? Reply with just the number."
    return " ".join(recs) + q, code


def needle(targets):
    ok = total = 0
    for t in targets:
        for j, d in enumerate((0.1, 0.5, 0.9)):
            text, code = needle_prompt(t, d, seed=t * 10 + j)
            t0 = time.time()
            r = chat([{"role": "user", "content": text}], max_tokens=4096)
            m = r["choices"][0]["message"]
            hit = code in (m.get("content") or "")
            ok += hit
            total += 1
            print(
                f"  needle target {t} prompt {r['usage']['prompt_tokens']} depth {d}: {'found' if hit else 'MISS'} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    print(f"{'PASS' if ok == total else 'FAIL'} needles {ok}/{total} at {targets}")
    return ok == total


# ------------------------------------------------------------------------------------------------ logprobs soak
def logprobs(n, conc=4):
    def one(i):
        body = {
            "model": model(),
            "max_tokens": 64,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [{"role": "user", "content": f"({i}) Answer in Chinese with two emoji: what is a GPU?"}],
        }
        content = post("/v1/chat/completions", body)["choices"][0]["logprobs"]["content"]
        short = sum(len(c.get("top_logprobs") or []) < 20 for c in content)
        dead = sum(c["logprob"] <= -9999.0 for c in content)
        return len(content), short, dead

    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(n)))
    pos = sum(r[0] for r in res)
    bad = sum(r[1] + r[2] for r in res)
    print(f"{'PASS' if bad == 0 else 'FAIL'} logprobs {n} requests, {pos} positions, {bad} incomplete or dead")
    return bad == 0


# ------------------------------------------------------------------------------------------------ speed
FILLER = (
    "The museum's east wing holds a collection of nineteenth-century maritime instruments, including a brass "
    "sextant recovered from a whaling vessel that ran aground near the northern shoals. "
)


def stream(prompt, max_tokens):
    """(ttft s, total s, completion tokens) of one streamed completion."""
    t0, ttft, n = time.time(), None, 0
    body = {
        "model": model(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if any(c.get("text") for c in d.get("choices", [])):
                ttft = ttft or time.time() - t0
    return ttft or 0.0, time.time() - t0, n


def filler(n_tokens):
    unit = ntok(FILLER)
    return FILLER * max(1, n_tokens // unit)


def sweep(ctx, levels):
    prefix = f"[{uuid.uuid4()}] Reference material follows.\n" + filler(ctx)
    stream(prefix + "\nSummarize.\n", 1)  # prime the prefix cache
    print(f"sweep: ~{ctx}-token cached prefix, {GEN} tokens per request, ignore_eos", flush=True)
    for c in levels:

        def one(i):
            return stream(prefix + f"\n[{uuid.uuid4()}] Write a long technical essay, part {i}.\n", GEN)

        t0 = time.time()
        with cf.ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, range(2 * c)))
        wall = time.time() - t0
        tpot = [(r[1] - r[0]) / max(1, r[2] - 1) for r in res]
        print(
            f"C={c} per-stream {1 / st.median(tpot):6.2f} tok/s  aggregate {sum(r[2] for r in res) / wall:7.1f} "
            f"tok/s  TPOT p50 {st.median(tpot) * 1000:5.0f} ms  TTFT p50 {st.median([r[0] for r in res]):5.2f} s",
            flush=True,
        )
    return True


def ttft(ctx):
    res = []
    for _ in range(3):
        p = f"[{uuid.uuid4()}] " + filler(ctx) + "\nSummarize in one line.\n"
        res.append(stream(p, 8)[0])
    print(f"ttft: uncached ~{ctx} tokens: {', '.join(f'{x:.2f}' for x in res)} s (median {st.median(res):.2f} s)")
    return True


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "golden":
        ok = golden(a[1])
    elif cmd == "tf":
        ok = tf(a[1])
    elif cmd == "compare":
        ok = compare(a[1], a[2])
    elif cmd == "tools":
        ok = tools()
    elif cmd == "needle":
        ok = needle([int(x) for x in a[1].split(",")])
    elif cmd == "logprobs":
        ok = logprobs(int(a[1]))
    elif cmd == "sweep":
        ok = sweep(int(a[1]), [int(x) for x in a[2].split(",")])
    elif cmd == "ttft":
        ok = ttft(int(a[1]))
    else:
        sys.exit(__doc__)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
