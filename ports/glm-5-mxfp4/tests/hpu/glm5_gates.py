#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving gates and measurements for a GLM-5.2 or GLM-5.3 MXFP4 endpoint. Standard library only.

  needles T1,T2,...        3 needles (10 / 50 / 90% depth) in two synthetic fillers at each prompt length (tokens),
                           reasoning low, greedy: PASS if every answer has all three codes
  gsm8k                    first 50 GSM8K test questions, reasoning low, greedy, 4 concurrent: PASS at >= 47 / 50
  tools                    tool_choice required / named / auto / none and a JSON schema, streaming and not, greedy
                           and at the server's default sampling (glm47 parser): PASS if every case is right
  mtp                      natural chat prompts at the server's default sampling, 1 stream: MTP acceptance from
                           /metrics; PASS at >= 0.70
  logprobs N               N sampled chat requests with top_logprobs=5 under MTP: every token has finite logprobs
  speed                    decode tok/s, greedy, ignore_eos, 256 tokens on a ~2K-token prompt at 1 and 4 streams
                           (per stream and aggregate; recorded, PASS when every stream finished)
  ttft                     time to first token of an uncached ~2K and ~8K-token prompt, median of 3 (recorded)
  stress                   long-prefill stress: 4 decoding streams while 64K- and 100K-token needle prompts
                           arrive: PASS if everything finishes, both needle answers are right, and the decode
                           streams keep moving (largest inter-token gap recorded)
  greedy OUT.json          16 fixed raw-text prompts, 192 greedy tokens each (decode path); recorded
  compare A.json B.json    PASS if two greedy files have identical tokens (same server build, two settings)

Environment: BASE (default http://127.0.0.1:8000), MODEL (default: the first model the server lists).
Each gate prints one line that starts with PASS or FAIL.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import random
import re
import statistics as st
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.environ.get("MODEL")


def model():
    global _MODEL
    if not _MODEL:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
            _MODEL = json.loads(r.read())["data"][0]["id"]
    return _MODEL


def _req(path, body):
    return urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})


def post(path, body, timeout=3600):
    with urllib.request.urlopen(_req(path, body), timeout=timeout) as r:
        return json.loads(r.read())


def stream(path, body, timeout=3600):
    """Yield the JSON events of a streamed response."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    with urllib.request.urlopen(_req(path, body), timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data: {"):
                yield json.loads(line[6:])


def chat(messages, max_tokens, effort="low", **kw):
    body = {"model": model(), "messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    d = post("/v1/chat/completions", body)
    return d["choices"][0], d["usage"]


def ntok(text):
    return post("/tokenize", {"model": model(), "prompt": text})["count"]


def result(ok, name, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------------- needles
TOPICS = [
    "The survey team logged tidal measurements along the northern inlet",
    "Maintenance records for the pumping station were filed out of order",
    "A revised schedule moved the inspection window into the following quarter",
    "The soil samples were catalogued by depth rather than by collection date",
    "Correspondence about the boundary fence continued for several months",
    "The equipment inventory was reconciled against the purchase ledger",
    "Weather observations were taken twice daily from the ridge station",
    "The access road required grading after the spring thaw",
]
TAILS = [
    "and the results were entered into the register without further comment",
    "which delayed the report but did not change any of its conclusions",
    "though nobody involved considered the matter especially urgent at the time",
    "before the file was passed to the regional office for archiving",
]
SUBJ = ["The harbour master", "A visiting botanist", "The night clerk", "The village council", "A retired engineer"]
SUBJ += ["The ferry crew", "The orchard keeper", "The archivist", "The quarry foreman", "The mill owner"]
VERB = ["repainted", "measured", "catalogued", "repaired", "photographed", "inspected", "relocated", "weighed"]
OBJ = ["the copper weathervane", "three crates of slate", "the eastern footbridge", "an old brass sextant"]
OBJ += ["the stone granary", "a row of pear trees", "the chapel bell", "the cider press", "the iron gate"]
WHEN = ["before dawn", "after the autumn fair", "during the long drought", "on a windy Tuesday", "at low tide"]
WHY = ["because the previous record was illegible", "to settle a long dispute", "for an upcoming exhibition"]
WHY += ["to prepare for the inspectors", "as part of a wager", "because the weather finally allowed it"]
NEEDLE = "IMPORTANT FACT: the survey reference number for the {w} district is {c}"
CODES = [("northern", "PELICAN-4471"), ("central", "OSPREY-8802"), ("southern", "KESTREL-1359")]
QUESTION = (
    "\n\nQuestion: what are the survey reference numbers for the northern, central and southern districts? "
    "Reply with the three codes only, one per line, in that order."
)


def filler(kind, n, seed=0):
    if kind == "survey":
        return [f"{TOPICS[i % len(TOPICS)]} in section {i}, {TAILS[(i * 7) % len(TAILS)]}" for i in range(n)]
    rng = random.Random(seed)
    return [
        f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(WHEN)} {rng.choice(WHY)}, and entry "
        f"{rng.randint(100, 99999)} recorded {rng.randint(2, 900)} {rng.choice(['units', 'hours', 'metres'])}"
        for _ in range(n)
    ]


def needle_prompt(kind, n):
    parts = filler(kind, n)
    for d, (w, c) in zip((0.10, 0.50, 0.90), CODES, strict=True):
        parts.insert(int(len(parts) * d), NEEDLE.format(w=w, c=c))
    # a random first line keeps prefix caching from serving an earlier run
    return f"Batch {random.randrange(16**8):08x}.\n\n" + ". ".join(parts) + "." + QUESTION


def sentences_for(kind, target):
    a, b = ntok(needle_prompt(kind, 200)), ntok(needle_prompt(kind, 400))
    per = (b - a) / 200
    return max(8, int(200 + (target - a) / per))


def needle_once(kind, target):
    n = sentences_for(kind, target)
    ch, usage = chat([{"role": "user", "content": needle_prompt(kind, n)}], 2500)
    text = ch["message"].get("content") or ""
    hits = [c in text for _, c in CODES]
    return all(hits), usage["prompt_tokens"], hits, text[-120:]


def gate_needles(targets):
    fails, rows = 0, []
    for kind in ("survey", "varied"):
        for t in (int(x) for x in targets.split(",")):
            ok, ptok, hits, tail = needle_once(kind, t)
            rows.append(f"{kind}@{ptok}:{sum(hits)}/3")
            print(f"  needle {kind} {ptok} tokens: {hits} {tail!r}", flush=True)
            fails += not ok
    return result(fails == 0, "needles", " ".join(rows))


# ------------------------------------------------------------------------------------------------------------- gsm8k
def gate_gsm8k():
    with open(os.path.join(HERE, "gsm8k50.json")) as f:
        items = json.load(f)["items"]

    def one(it):
        q = it["q"] + "\nGive the final answer as a number on the last line, prefixed by '####'."
        ch, _ = chat([{"role": "user", "content": q}], 4096)
        ans = ch["message"].get("content") or ""
        m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", ans) or re.findall(r"(-?[\d,]*\.?\d+)", ans)
        pred = m[-1].replace(",", "").rstrip(".") if m else ""
        try:
            return abs(float(pred) - float(it["a"])) < 1e-6
        except ValueError:
            return False

    with cf.ThreadPoolExecutor(4) as ex:
        oks = list(ex.map(one, items))
    missed = [i for i, ok in enumerate(oks) if not ok]
    return result(sum(oks) >= 47, "gsm8k", f"{sum(oks)}/{len(oks)} (missed {missed})")


# ------------------------------------------------------------------------------------------------------------- tools
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}
ASK_WEATHER = [{"role": "user", "content": "What is the weather in Paris right now?"}]
MULTI = [
    {"role": "user", "content": "What's the weather in Chicago?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Chicago"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_f": 61, "conditions": "windy"}'},
    {"role": "user", "content": "And Paris?"},
]
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def tool_cases():
    named = {"type": "function", "function": {"name": "get_weather"}}
    history = [{"role": "user", "content": "Tell me about the history of Rome."}]
    return [
        ("required-weather", {"messages": ASK_WEATHER, "tools": [WEATHER, SEARCH], "tool_choice": "required"}),
        ("required-history", {"messages": history, "tools": [WEATHER, SEARCH], "tool_choice": "required"}),
        ("required-multiturn", {"messages": MULTI, "tools": [WEATHER, SEARCH], "tool_choice": "required"}),
        ("named", {"messages": ASK_WEATHER, "tools": [WEATHER, SEARCH], "tool_choice": named}),
        ("auto-weather", {"messages": ASK_WEATHER, "tools": [WEATHER, SEARCH], "tool_choice": "auto"}),
        ("auto-math", {"messages": [{"role": "user", "content": "What is 2+2? Answer with the number."}],
                       "tools": [WEATHER, SEARCH], "tool_choice": "auto"}),
        ("none", {"messages": ASK_WEATHER, "tools": [WEATHER], "tool_choice": "none"}),
        ("json-schema", {"messages": [{"role": "user", "content": "Invent a person; give their name and age as JSON."}],
                         "response_format": {"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}}}),
    ]  # fmt: skip


def run_case(body, streamed):
    body = dict(body, model=model(), max_tokens=4000)
    if not streamed:
        d = post("/v1/chat/completions", body)
        ch = d["choices"][0]
        m = ch["message"]
        calls = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
        return ch.get("finish_reason"), calls, m.get("content") or ""
    calls, content, finish = {}, [], None
    for ev in stream("/v1/chat/completions", body):
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            content.append(d.get("content") or "")
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), ["", ""])
                c[0] += (tc.get("function") or {}).get("name") or ""
                c[1] += (tc.get("function") or {}).get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return finish, [tuple(c) for _, c in sorted(calls.items())], "".join(content)


def judge(case, finish, calls, content):
    def args_ok(a):
        try:
            return isinstance(json.loads(a), dict)
        except ValueError:
            return False

    if case.startswith("required"):
        return len(calls) == 1 and finish == "tool_calls" and args_ok(calls[0][1])
    if case == "named":
        return [c[0] for c in calls] == ["get_weather"] and args_ok(calls[0][1])
    if case == "auto-weather":
        return len(calls) >= 1 and finish == "tool_calls" and all(args_ok(c[1]) for c in calls)
    if case == "auto-math":
        return not calls and "4" in content
    if case == "none":
        return not calls and "<tool_call>" not in content
    try:
        o = json.loads(content)
        return finish == "stop" and set(o) == {"name", "age"} and isinstance(o["age"], int)
    except (ValueError, TypeError):
        return False


def gate_tools():
    jobs = [
        (case, body, s, samp) for case, body in tool_cases() for s in (False, True) for samp in ({"temperature": 0}, {})
    ]

    def one(job):
        case, body, s, samp = job
        try:
            ok = judge(case, *run_case(dict(body, **samp), s))
        except Exception as e:  # an HTTP 500 is a failure of the case
            print(f"  {case}: {e!r}", flush=True)
            ok = False
        return f"{case}/{'stream' if s else 'plain'}/{'greedy' if samp else 'default'}", ok

    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    bad = [n for n, ok in res if not ok]
    return result(not bad, "tools", f"{len(res) - len(bad)}/{len(res)} cases" + (f", failed {bad}" if bad else ""))


# --------------------------------------------------------------------------------------------------------------- mtp
CHAT_PROMPTS = [
    "Write a Python function that parses an ISO 8601 duration like P3DT4H5M into seconds, with tests.",
    "Explain how a B-tree differs from a binary search tree and why databases use B-trees.",
    "A train leaves at 9:40 and travels 270 km at 90 km/h, then 120 km at 60 km/h. When does it arrive?",
    "Summarize the main arguments for and against a four-day work week.",
    "Write a bash script that finds the 10 largest files under a directory and prints their sizes in MB.",
    "Implement an LRU cache in Go with Get and Put in O(1), and explain the design.",
]


def spec_counts():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
        txt = r.read().decode()
    out = {}
    for k in ("accepted_tokens", "draft_tokens"):
        pat = r"^vllm:spec_decode_num_" + k + r"_total(?:\{[^}]*\})? ([0-9.e+]+)$"
        out[k] = sum(float(m.group(1)) for m in re.finditer(pat, txt, re.M))
    return out


def gate_mtp():
    m0 = spec_counts()
    for p in CHAT_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        post("/v1/chat/completions", {"model": model(), "messages": msgs, "max_tokens": 512})
    m1 = spec_counts()
    drafts = m1["draft_tokens"] - m0["draft_tokens"]
    if drafts <= 0:
        return result(False, "mtp", "no draft tokens counted: is MTP on?")
    rate = (m1["accepted_tokens"] - m0["accepted_tokens"]) / drafts
    return result(rate >= 0.70, "mtp", f"acceptance {rate:.3f} over {int(drafts)} drafts (1 stream, default sampling)")


# ---------------------------------------------------------------------------------------------------------- logprobs
def gate_logprobs(n):
    def one(i):
        body = {"model": model(), "messages": [{"role": "user", "content": CHAT_PROMPTS[i % len(CHAT_PROMPTS)]}],
                "max_tokens": 200, "logprobs": True, "top_logprobs": 5, "temperature": 1.0}  # fmt: skip
        d = post("/v1/chat/completions", body)
        lp = (d["choices"][0].get("logprobs") or {}).get("content") or []
        ok = len(lp) == d["usage"]["completion_tokens"] and all(
            math.isfinite(t["logprob"]) and len(t["top_logprobs"]) == 5 for t in lp
        )
        return ok

    with cf.ThreadPoolExecutor(4) as ex:
        oks = list(ex.map(one, range(n)))
    return result(all(oks), "logprobs", f"{sum(oks)}/{n} requests with complete, finite logprobs")


# ------------------------------------------------------------------------------------------------------------- speed
SPEED_TEXT = (
    "The history of numerical computing is a story of trade-offs between precision, speed and memory. Early machines "
    "used fixed-point arithmetic; floating point arrived with dedicated hardware, and every generation since has "
    "revisited how many bits a number really needs. "
)


def speed_prompt(tokens, rng):
    per = ntok(SPEED_TEXT)
    head = f"[session {rng.randrange(1 << 30)}] Read the following notes, then continue them in the same style.\n\n"
    return head + SPEED_TEXT * max(1, (tokens - 40) // per)


def timed_stream(prompt, out, max_tokens=256):
    body = {"model": model(), "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    t0, stamps, usage = time.time(), [], None
    for ev in stream("/v1/completions", body):
        usage = ev.get("usage") or usage
        if ev.get("choices") and ev["choices"][0].get("text"):
            stamps.append(time.time())
    n = usage["completion_tokens"]
    dec = stamps[-1] - stamps[0]
    gap = max((b - a for a, b in zip(stamps, stamps[1:], strict=False)), default=0.0)
    tok_s = (n - 1) / dec if dec > 0 else 0.0
    out.append({"n": n, "ttft": stamps[0] - t0, "tok_s": tok_s, "t0": t0, "t_end": stamps[-1], "gap": gap})


def run_streams(prompts, max_tokens=256):
    out = []
    th = [threading.Thread(target=timed_stream, args=(p, out, max_tokens)) for p in prompts]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def gate_speed():
    rng = random.Random(1234)
    rows = []
    for c in (1, 4):
        run_streams([speed_prompt(2048, rng) for _ in range(c)])  # warm-up: compile the shapes
        reps = [run_streams([speed_prompt(2048, rng) for _ in range(c)]) for _ in range(3)]
        if any(len(r) != c or any(o["n"] != 256 for o in r) for r in reps):
            return result(False, "speed", f"a stream at C={c} did not finish 256 tokens")
        per = st.median(st.median(o["tok_s"] for o in r) for r in reps)
        agg = st.median(sum(o["n"] for o in r) / (max(o["t_end"] for o in r) - min(o["t0"] + o["ttft"] for o in r))
                        for r in reps)  # fmt: skip
        ttft = st.median(st.median(o["ttft"] for o in r) for r in reps)
        rows.append(f"C={c}: {per:.2f} tok/s per stream, {agg:.2f} aggregate, TTFT {ttft:.2f} s")
    return result(True, "speed", "; ".join(rows) + " (~2K-token prompt, greedy, 256 tokens, median of 3)")


def gate_ttft():
    """Time to first token of one uncached prompt at ~2K and ~8K tokens, 1 stream: one warm-up, median of 3."""
    rng = random.Random(4321)
    rows = []
    for plen in (2048, 8192):
        run_streams([speed_prompt(plen, rng)], max_tokens=8)  # warm-up: compile the shapes
        runs = [run_streams([speed_prompt(plen, rng)], max_tokens=8)[0]["ttft"] for _ in range(3)]
        rows.append(f"{plen}: {st.median(runs):.3f} s ({', '.join(f'{x:.3f}' for x in runs)})")
    return result(True, "ttft", "; ".join(rows) + " (unique prefix per prompt, median of 3)")


# ------------------------------------------------------------------------------------------------------------ stress
def gate_stress():
    rng = random.Random(7)
    decodes, needles = [], {}

    def needle(t):
        needles[t] = needle_once("varied", t)

    th = [threading.Thread(target=timed_stream, args=(speed_prompt(1024, rng), decodes, 1024)) for _ in range(4)]
    th += [threading.Thread(target=needle, args=(t,)) for t in (65536, 102400)]
    for t in th:
        t.start()
        time.sleep(2)
    for t in th:
        t.join()
    ok = len(decodes) == 4 and all(o["n"] == 1024 for o in decodes) and len(needles) == 2
    ok = ok and all(v[0] for v in needles.values())
    gap = max((o["gap"] for o in decodes), default=float("nan"))
    detail = f"{len(decodes)}/4 decodes finished, needles {[(v[1], sum(v[2])) for v in needles.values()]}"
    return result(ok, "stress", detail + f", largest decode inter-token gap {gap:.1f} s")


# ------------------------------------------------------------------------------------------------- greedy, compare
GREEDY_PROMPTS = [
    'def merge_intervals(intervals):\n    """Merge overlapping intervals."""\n',
    "The three main causes of the French Revolution were",
    "Q: A farmer has 17 sheep and buys 23 more, then sells 9. How many sheep are left?\nA: Let's think step by step.",
    "# Rust: a thread-safe LRU cache\nuse std::collections::HashMap;\n",
    "Photosynthesis converts light energy into chemical energy. In detail,",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\n",
    "The proof that there are infinitely many primes goes as follows.",
    "Translate to French: 'The weather is nice today and we are going to the beach.'\nFrench:",
    "import numpy as np\n\ndef softmax(x, axis=-1):\n",
    "Once upon a time in a small village by the sea, there lived an old clockmaker who",
    "Explain the difference between TCP and UDP in networking.",
    "<html>\n<head>\n<title>Dashboard</title>\n",
    "The derivative of x^3 * sin(x) with respect to x is",
    "List ten countries in Africa and their capitals:\n1.",
    "Kubernetes Deployments manage ReplicaSets. When you update a Deployment,",
    "In 1969, the Apollo 11 mission",
]


def gate_greedy(path):
    out = []
    for p in GREEDY_PROMPTS:
        d = post("/v1/completions", {"model": model(), "prompt": p, "max_tokens": 192, "temperature": 0, "logprobs": 1})
        lp = d["choices"][0]["logprobs"]
        out.append({"prompt": p, "tokens": lp["tokens"], "logprobs": lp["token_logprobs"]})
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return result(True, "greedy", f"{len(out)} prompts x 192 tokens recorded in {path}")


def gate_compare(a, b):
    with open(a) as f:
        ra = json.load(f)
    with open(b) as f:
        rb = json.load(f)
    same = sum(x["tokens"] == y["tokens"] for x, y in zip(ra, rb, strict=True))
    first = [
        next((i for i, (s, t) in enumerate(zip(x["tokens"], y["tokens"], strict=False)) if s != t), None)
        for x, y in zip(ra, rb, strict=True)
    ]
    diverged = [(k, i) for k, i in enumerate(first) if i is not None]
    return result(same == len(ra), "compare", f"{same}/{len(ra)} prompts token-identical; diverged at {diverged}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {
        "needles": gate_needles,
        "gsm8k": gate_gsm8k,
        "tools": gate_tools,
        "mtp": gate_mtp,
        "logprobs": lambda n: gate_logprobs(int(n)),
        "speed": gate_speed,
        "ttft": gate_ttft,
        "stress": gate_stress,
        "greedy": gate_greedy,
        "compare": gate_compare,
    }[cmd]
    sys.exit(fn(*args))


if __name__ == "__main__":
    main()
