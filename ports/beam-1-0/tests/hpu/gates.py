#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gates for Beam-1.0 on Gaudi 2, run against a server started with the recipe's command.

Stdlib only; talks to BEAM_URL (default http://127.0.0.1:8000) and prints one PASS or FAIL line per
gate, plus INFO lines for speed. Details of every gate go to the JSON file named by --out.

  smoke     20 fixed chat prompts at temperature 0: expected answers, finish "stop", reasoning
            present, no template markup in the answer
  ppl       prompt logprobs on fixed passages against two CPU references of the same checkpoint
            (reference/beam_ppl_reference.json.gz): perplexity within 1.5% of the fp32 reference and
            of its per-token FP8 activation variant ("w8a8"), top-1 agreement with w8a8 at least the
            fp32-vs-w8a8 floor minus 2 points, mean top-20 KL to w8a8 at most 0.02
  dvp       8 greedy decodes of 512 tokens from prompts past the 512-token window, re-scored by
            prefill: mean |logprob difference| at most 0.05
  needle    a short code in neutral filler at 16K / 64K / 126K / 250K tokens x depths 0.1 / 0.5 / 0.9,
            answered in the content; a control without the needle must not produce it
  copy      a 1,521-token passage early in a 130K and a 255K prompt, restarted at the end: the greedy
            continuation reproduces its next 64 tokens exactly (only the 9 full-attention layers can
            carry it across the context)
  tools     tool_choice auto / required / named / none, non-streaming and streaming (streamed
            fragments join to the same calls), parallel calls, response_format json_schema,
            reasoning_effort "none" accepted
  isolation 8 different requests at once, each with its own code: every answer has its own code and
            no other request's code or topic
  stress    15 streaming decodes while a 250K-token prompt prefills: all complete, server healthy
  speed     INFO only: decode tok/s at 1 and 16 streams (completions, 256 tokens)

  python3 gates.py all --out gates.json          # or a list of gate names
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import gzip
import importlib.util
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("BEAM_URL", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference", "beam_ppl_reference.json.gz")
MARKUP = re.compile(r"<\|[^|>]{1,40}\|>|rfl:parameter")
MODEL = None


# ------------------------------------------------------------------------------------------ http
def post(path, body, timeout=7200):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:1500].decode(errors="replace")}


def ok_post(path, body, timeout=7200):
    code, r = post(path, body, timeout)
    if code != 200:
        raise RuntimeError(f"{path}: HTTP {code} {r}")
    return r


def get(path, timeout=30):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return r.read().decode()


def sse(path, body, timeout=7200):
    """Server-sent events of a streaming request, with the arrival time of each."""
    body = {**body, "stream": True}
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0, events = time.time(), []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append((time.time() - t0, json.loads(line[6:])))
    return events


def tokenize(text):
    return ok_post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def detok(ids):
    return ok_post("/detokenize", {"model": MODEL, "tokens": list(ids)})["prompt"]


def chat(messages, max_tokens=2048, temperature=0.0, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
    t = time.time()
    r = ok_post("/v1/chat/completions", body)
    c = r["choices"][0]
    m = c["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
        "tool_calls": m.get("tool_calls") or [],
        "finish": c.get("finish_reason"),
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "completion_tokens": r["usage"]["completion_tokens"],
        "seconds": round(time.time() - t, 1),
    }


# ---------------------------------------------------------------------------------------- filler
_FILLER: list[int] = []


def filler_ids(n):
    """>= n token ids of neutral filler: vLLM's own documentation (Markdown) when the source tree has it,
    else its Python sources, in sorted order. `<|` is escaped as a user would send it."""
    global _FILLER
    if len(_FILLER) >= n:
        return _FILLER
    spec = importlib.util.find_spec("vllm")
    pkg = list(spec.submodule_search_locations)[0]
    files = sorted(glob.glob(os.path.join(os.path.dirname(pkg), "docs", "**", "*.md"), recursive=True))
    files = files or sorted(glob.glob(os.path.join(pkg, "**", "*.py"), recursive=True))
    ids: list[int] = []
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as f:
            t = f.read().replace("<|", "< |")
        for k in range(0, len(t), 200000):
            ids += tokenize(t[k : k + 200000] + "\n\n")
        if len(ids) >= n:
            break
    if len(ids) < n:
        raise RuntimeError(f"filler has only {len(ids)} tokens, need {n}")
    _FILLER = ids
    return ids


# ----------------------------------------------------------------------------------------- gates
SMOKE = [
    ("What is the capital of France? Answer in one word.", r"\bparis\b"),
    ("What is 17 * 23? Give just the number.", r"\b391\b"),
    (
        "Write a Python function fizzbuzz(n) that prints the numbers from 1 to n, but prints Fizz for multiples of 3, "
        "Buzz for multiples of 5 and FizzBuzz for multiples of both.",
        r"def fizzbuzz",
    ),
    ("Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle.", None),
    ("Who wrote the novel Pride and Prejudice?", r"austen"),
    ("What is the chemical symbol for gold?", r"\bAu\b"),
    ("How many days are there in a leap year?", r"\b366\b"),
    ("What is the square root of 144?", r"\b12\b"),
    ("Translate into Spanish: Good morning, how are you?", r"buenos d[ií]as"),
    ("What is the largest planet in our solar system?", r"jupiter"),
    ("What is 1234 + 5678?", r"6,?912"),
    ("Explain in two sentences what photosynthesis is.", r"light"),
    (
        "Write a SQL query that selects every row of the table users whose age is greater than 30.",
        r"select\s+\*\s+from\s+users\s+where\s+age\s*>\s*30",
    ),
    ("What is the boiling point of water at sea level in degrees Celsius?", r"\b100\b"),
    ("A train travels at 60 miles per hour for 3 hours. How far does it travel?", r"\b180\b"),
    ("Write a haiku about autumn leaves.", None),
    ("In which year did World War II end?", r"\b1945\b"),
    ("Write a JavaScript function that reverses a string.", r"function|=>"),
    ("What is the derivative of x^3 + 2x with respect to x?", r"3x\^?2|3x²|3 ?x\^\{?2"),
    ("List the first five prime numbers.", r"2.*3.*5.*7.*11"),
]


def gate_smoke():
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda q: chat([{"role": "user", "content": q[0]}], reasoning_effort="low"), SMOKE))
    items = []
    for (q, rx), o in zip(SMOKE, outs, strict=True):
        good = bool(re.search(rx, o["content"], re.I | re.S)) if rx else len(o["content"].split()) >= 5
        items.append(
            {
                "prompt": q,
                "correct": good,
                "finish": o["finish"],
                "reasoning": bool(o["reasoning"]),
                "markup": MARKUP.findall(o["content"])[:3],
                "content": o["content"][:300],
            }
        )
    n = len(items)
    correct = sum(i["correct"] for i in items)
    stop = sum(i["finish"] == "stop" for i in items)
    reasoning = sum(i["reasoning"] for i in items)
    markup = sum(bool(i["markup"]) for i in items)
    ok = correct == n and stop == n and reasoning == n and markup == 0
    return ok, f"correct {correct}/{n}, stop {stop}/{n}, reasoning {reasoning}/{n}, markup {markup}", items


def score_ids(ids, k=20):
    """prompt_logprobs for a token-id prompt: per position j >= 1, (logprob of ids[j], {id: logprob} of the
    top k, top-1 id); entry j - 1 predicts ids[j]."""
    r = ok_post(
        "/v1/completions",
        {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k},
    )
    pl = r["choices"][0]["prompt_logprobs"]
    rows = []
    for j in range(1, len(ids)):
        d = {int(t): v for t, v in pl[j].items()}
        top = min(d, key=lambda t: d[t]["rank"])
        rows.append((d[ids[j]]["logprob"], {t: v["logprob"] for t, v in d.items() if v["rank"] <= k}, top))
    return rows


def kl_top20(ref_ids, ref_lp, cand, cand_min):
    """KL(reference || candidate) over the reference's top 20 plus one 'rest' bucket (a lower bound of the
    full KL); candidate logprobs missing from its top 20 are bounded by its smallest returned logprob."""
    pr = [math.exp(x) for x in ref_lp]
    pc = [math.exp(cand.get(i, cand_min)) for i in ref_ids]
    kl = sum(p * (lr - math.log(q)) for p, lr, q in zip(pr, ref_lp, pc, strict=True))
    rr, rc = max(1.0 - sum(pr), 1e-12), max(1.0 - sum(pc), 1e-12)
    return kl + rr * (math.log(rr) - math.log(rc))


def compare(rows, ref):
    n = len(rows)
    kls, agree, nll = [], 0, 0.0
    for j, (lp, cand, top) in enumerate(rows):
        kls.append(kl_top20(ref["topk_ids"][j], ref["topk_lp"][j], cand, min(cand.values())))
        agree += int(top == ref["topk_ids"][j][0])
        nll -= lp
    return {
        "ppl_ratio_minus_1": math.exp(nll / n - sum(ref["nll"][:n]) / n) - 1,
        "top1_agree": agree / n,
        "mean_kl_top20": sum(kls) / n,
    }


def ref_rows(ref):
    return [
        (-ref["nll"][j], dict(zip(ref["topk_ids"][j], ref["topk_lp"][j], strict=True)), ref["topk_ids"][j][0])
        for j in range(len(ref["nll"]))
    ]


def load_reference():
    with gzip.open(REFERENCE, "rt") as f:
        return json.load(f)["passages"]


def gate_ppl():
    res, ok, parts = {}, True, []
    for name, p in load_reference().items():
        rows = score_ids(p["ids"])
        r = {
            "tokens": len(p["ids"]),
            "vs_oracle": compare(rows, p["oracle"]),
            "vs_w8a8": compare(rows, p["w8a8"]),
            "floor": compare(ref_rows(p["w8a8"]), p["oracle"]),
        }
        good = (
            abs(r["vs_w8a8"]["ppl_ratio_minus_1"]) <= 0.015
            and abs(r["vs_oracle"]["ppl_ratio_minus_1"]) <= 0.015
            and r["vs_w8a8"]["top1_agree"] >= r["floor"]["top1_agree"] - 0.02
            and r["vs_w8a8"]["mean_kl_top20"] <= 0.02
        )
        ok &= good
        res[name] = r
        parts.append(
            f"{name} ppl {r['vs_w8a8']['ppl_ratio_minus_1']:+.2%} vs w8a8, top-1 {r['vs_w8a8']['top1_agree']:.2%} "
            f"(floor {r['floor']['top1_agree']:.2%}), KL {r['vs_w8a8']['mean_kl_top20']:.4f}"
        )
    return ok, "; ".join(parts), res


def gate_dvp():
    ref = load_reference()
    texts = [ref["vllm_arch_overview"]["ids"], ref["vllm_kv_cache_utils_py"]["ids"]]
    prompts = []
    for i in range(8):
        ids = texts[i % 2]
        start = 600 + 97 * i  # prompts of 600 to 1279 tokens: decode runs past the 512 window throughout
        off = 0 if i < 2 else min(200, max(0, len(ids) - start - 1))
        prompts.append(ids[off : off + start])

    def gen(p):
        r = ok_post(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": p,
                "max_tokens": 512,
                "temperature": 0,
                "logprobs": 1,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        c = r["choices"][0]
        return c["token_ids"], c["logprobs"]["token_logprobs"]

    with cf.ThreadPoolExecutor(4) as ex:
        gens = list(ex.map(gen, prompts))
    diffs, items = [], []
    for p, (g, glp) in zip(prompts, gens, strict=True):
        rows = score_ids(p + g, k=1)
        pre = [rows[len(p) - 1 + j][0] for j in range(len(g))]
        d = [abs(x - y) for x, y in zip(glp, pre, strict=True)]
        diffs += d
        items.append({"prompt_tokens": len(p), "mean_abs": sum(d) / len(d), "max_abs": max(d)})
    mean = sum(diffs) / len(diffs)
    return mean <= 0.05, f"mean |dlogprob| {mean:.4f} (gate 0.05), max {max(diffs):.2f}", items


NEEDLE = " The catalogue number of the blue notebook is {}. "
NEEDLE_Q = (
    "\n\nQuestion: What is the catalogue number of the blue notebook mentioned above? Reply with the number only."
)
CODES = ["KESTREL-4719-VANTA", "OSPREY-2286-LUMEN", "HERON-9053-QUILL"]


def needle_prompt(length, depth, code, offset, needle=True):
    nd = tokenize(NEEDLE.format(code)) if needle else []
    n = length - len(nd) - 150  # template and question take about 110 tokens
    base = filler_ids(offset + n + 10)[offset : offset + n]
    k = int(n * depth)
    return detok(base[:k] + nd + base[k:]) + NEEDLE_Q


def gate_needle():
    items = []
    for li, length in enumerate((16384, 65536, 129000, 255000)):
        for depth in (0.1, 0.5, 0.9):
            code = CODES[(li + int(depth * 10)) % 3]
            o = chat([{"role": "user", "content": needle_prompt(length, depth, code, 0)}], 1024, reasoning_effort="low")
            items.append(
                {"length": length, "depth": depth, "prompt_tokens": o["prompt_tokens"], "hit": code in o["content"]}
                | {"seconds": o["seconds"], "answer": o["content"][:120]}
            )
    o = chat([{"role": "user", "content": needle_prompt(16384, 0.5, CODES[0], 0, needle=False)}], 1024)
    control = not any(c in o["content"] for c in CODES)
    hits = sum(i["hit"] for i in items)
    detail = f"{hits}/{len(items)} hit at 16K to 255K x 3 depths; control without needle clean: {control}"
    return hits == len(items) and control, detail, items


def gate_copy():
    passage = load_reference()["vllm_kv_cache_utils_py"]["ids"]
    restart, gen = 400, 64
    items = []
    for total in (130000, 255000):
        n_fill = total - len(passage) - restart - gen
        fill = filler_ids(n_fill + 20000)[20000 : 20000 + n_fill]
        k = int(n_fill * 0.1)
        ids = fill[:k] + passage + fill[k:] + [198, 198] + passage[:restart]
        expect = passage[restart : restart + gen]
        t = time.time()
        r = ok_post(
            "/v1/completions",
            {"model": MODEL, "prompt": ids, "max_tokens": gen, "temperature": 0, "return_token_ids": True},
        )
        g = r["choices"][0]["token_ids"]
        match = next((j for j, (x, y) in enumerate(zip(g, expect, strict=False)) if x != y), min(len(g), len(expect)))
        items.append({"prompt_tokens": len(ids), "exact_prefix": match, "seconds": round(time.time() - t, 1)})
    ok = all(i["exact_prefix"] == gen for i in items)
    detail = ", ".join(f"{i['prompt_tokens']} tokens {i['exact_prefix']}/{gen} in {i['seconds']} s" for i in items)
    return ok, detail, items


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
Q_WEATHER = [{"role": "user", "content": "What is the weather like in Paris right now? Use celsius."}]


def streamed_chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 2048, **kw}
    content, calls, finish = [], {}, None
    for _, e in sse("/v1/chat/completions", body):
        for ch in e.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return {"content": "".join(content), "tool_calls": [calls[k] for k in sorted(calls)], "finish": finish}


def _calls(o):
    """[(name, arguments dict or None)] of a response, streamed or not."""
    out = []
    for tc in o["tool_calls"]:
        f = tc.get("function", tc)
        try:
            args = json.loads(f["arguments"])
        except (TypeError, ValueError):
            args = None
        out.append((f["name"], args))
    return out


def gate_tools():
    checks = {}
    named = {"type": "function", "function": {"name": "get_weather"}}
    cases = {
        "auto": dict(tools=[WEATHER, TIME], tool_choice="auto"),
        "required": dict(tools=[WEATHER, TIME], tool_choice="required"),
        "named": dict(tools=[WEATHER, TIME], tool_choice=named),
    }
    for name, kw in cases.items():
        for mode, fn in (("non-streaming", lambda m, **k: chat(m, **k)), ("streaming", streamed_chat)):
            o = fn(Q_WEATHER, **kw)
            calls = _calls(o)
            good = (
                len(calls) == 1
                and calls[0][0] == "get_weather"
                and isinstance(calls[0][1], dict)
                and "paris" in str(calls[0][1].get("city", "")).lower()
            )
            checks[f"{name} {mode}"] = good
    for mode, fn in (("non-streaming", lambda m, **k: chat(m, **k)), ("streaming", streamed_chat)):
        o = fn(Q_WEATHER, tools=[WEATHER, TIME], tool_choice="none")
        checks[f"none {mode}"] = not o["tool_calls"] and bool(o["content"]) and not MARKUP.search(o["content"])
    o = chat(
        [{"role": "user", "content": "What is the weather in Tokyo and in Lima? Call the tool for both cities."}],
        tools=[WEATHER],
        tool_choice="auto",
    )
    cities = sorted(str((a or {}).get("city", "")).lower() for _, a in _calls(o))
    checks["parallel calls"] = len(cities) == 2 and "tokyo" in cities[0] + cities[1] and "lima" in cities[0] + cities[1]
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
        "required": ["name", "year"],
        "additionalProperties": False,
    }
    o = chat(
        [{"role": "user", "content": "Give the name and founding year of the company that makes the iPhone."}],
        response_format={"type": "json_schema", "json_schema": {"name": "company", "schema": schema, "strict": True}},
    )
    try:
        doc = json.loads(o["content"])
        checks["json_schema"] = set(doc) == {"name", "year"} and isinstance(doc["year"], int)
    except ValueError:
        checks["json_schema"] = False
    code, _ = post(
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64, "reasoning_effort": "none"},
    )
    checks["reasoning_effort none"] = code == 200
    bad = [k for k, v in checks.items() if not v]
    return not bad, f"{len(checks) - len(bad)}/{len(checks)}" + (f", failed: {bad}" if bad else ""), checks


TOPICS = ["vorsatil", "pelumbra", "quandrix", "mobrevel", "tasquine", "orvelline", "glimtor", "zephandra"]
ALPH = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def gate_isolation():
    rng = random.Random(7)
    base = filler_ids(200000)
    reqs = []
    for i, topic in enumerate(TOPICS):
        n = (300, 2000, 6000, 9000)[i % 4]
        code = "".join(rng.choice(ALPH) for _ in range(10))
        off, k = rng.randrange(0, 150000), rng.randrange(0, n)
        body = base[off : off + n]
        nd = tokenize(f" Reference card {topic}: its catalogue number is {code}. ")
        q = (
            f"\n\nWhat is the catalogue number of reference card {topic} in the text above? Reply with the catalogue "
            f"number first, then write two sentences about reference card {topic}."
        )
        reqs.append((code, topic, detok(body[:k] + nd + body[k:]) + q))
    with cf.ThreadPoolExecutor(len(reqs)) as ex:
        outs = list(ex.map(lambda r: chat([{"role": "user", "content": r[2]}], 1024, reasoning_effort="minimal"), reqs))
    own, foreign, items = 0, 0, []
    for (code, _, _), o in zip(reqs, outs, strict=True):
        text = o["content"] + "\n" + o["reasoning"]
        f = [c for c, t, _ in reqs if c != code and (c in text or t in text)]
        own += code in o["content"]
        foreign += len(f)
        items.append(
            {"own_code": code in o["content"], "foreign": f, "finish": o["finish"], "answer": o["content"][:120]}
        )
    return own == len(reqs) and foreign == 0, f"own code {own}/{len(reqs)}, foreign content {foreign}", items


SPEED_PROMPTS = [
    "The history of the printing press begins",
    "A short guide to growing tomatoes at home starts",
    "The theory of plate tectonics explains",
    "In the early days of aviation, pilots",
    "Modern compilers translate source code",
]


def stream_completion(ids, max_tokens):
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    return [t for t, e in sse("/v1/completions", body) if e.get("choices") and e["choices"][0].get("text") is not None]


def gate_stress():
    decodes, gen = 15, 600
    results, errors = [], []

    def dec(i):
        try:
            results.append(len(stream_completion(tokenize(SPEED_PROMPTS[i % 5]), gen)))
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:200])

    threads = [threading.Thread(target=dec, args=(i,)) for i in range(decodes)]
    for t in threads:
        t.start()
    time.sleep(5)
    t0 = time.time()
    long_ids = filler_ids(250000 + 30000)[30000 : 30000 + 250000]
    r = ok_post("/v1/completions", {"model": MODEL, "prompt": long_ids, "max_tokens": 16, "temperature": 0})
    long_s = round(time.time() - t0, 1)
    for t in threads:
        t.join()
    healthy = True
    try:
        get("/health", timeout=10)
    except Exception:  # noqa: BLE001
        healthy = False
    complete = sum(n >= gen - 1 for n in results)
    long_ok = r["usage"]["completion_tokens"] == 16
    ok = healthy and long_ok and complete == decodes and not errors
    detail = f"{complete}/{decodes} decodes of {gen} tokens complete, 250K prompt {long_s} s, healthy {healthy}"
    return ok, detail, {"decode_tokens": results, "errors": errors, "long_seconds": long_s}


def speed():
    out = {}
    for streams in (1, 16):
        prompts = [tokenize(SPEED_PROMPTS[i % 5]) for i in range(streams)]
        stream_completion(prompts[0], 32)  # warm the decode graphs for this size first
        with cf.ThreadPoolExecutor(streams) as ex:
            times = list(ex.map(lambda p: stream_completion(p, 256), prompts))
        lo, hi = max(t[0] for t in times), min(t[-1] for t in times)
        inside = sum(sum(1 for x in t if lo <= x <= hi) for t in times)
        out[streams] = {
            "aggregate": (inside - streams) / (hi - lo) if hi > lo else 0.0,
            "per_stream": sum((len(t) - 1) / (t[-1] - t[0]) for t in times) / streams,
        }
    return out


GATES = {
    "smoke": gate_smoke,
    "ppl": gate_ppl,
    "dvp": gate_dvp,
    "needle": gate_needle,
    "copy": gate_copy,
    "tools": gate_tools,
    "isolation": gate_isolation,
    "stress": gate_stress,
}


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gates", nargs="*", default=["all"], help=f"all, speed, or any of {', '.join(GATES)}")
    ap.add_argument("--out", default="beam_gates.json")
    a = ap.parse_args()
    MODEL = json.loads(get("/v1/models"))["data"][0]["id"]
    names = list(GATES) + ["speed"] if a.gates == ["all"] else a.gates
    report, failed = {}, 0
    for name in names:
        t = time.time()
        if name == "speed":
            s = speed()
            report["speed"] = s
            print(
                f"INFO speed: 1 stream {s[1]['per_stream']:.1f} tok/s; 16 streams {s[16]['aggregate']:.0f} tok/s "
                f"aggregate, {s[16]['per_stream']:.1f} per stream (completions, 256 tokens)",
                flush=True,
            )
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1, ensure_ascii=False)
            continue
        try:
            ok, detail, data = GATES[name]()
        except Exception as e:  # noqa: BLE001
            ok, detail, data = False, f"{type(e).__name__}: {e}", None
        failed += not ok
        report[name] = {"pass": ok, "detail": detail, "seconds": round(time.time() - t, 1), "data": data}
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail} ({time.time() - t:.0f} s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
