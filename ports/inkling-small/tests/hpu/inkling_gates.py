#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serving gates for Inkling-Small against a running server. Prints one PASS or FAIL line per gate.

    BASE=http://localhost:8000 MODEL=thinkingmachines/Inkling-Small python3 inkling_gates.py [gate ...]

Gates (default: all of the quick tier; `long` adds the long-context tier):
  smoke       chat arithmetic, tool call, JSON mode, tool_choice "required"
  greedy      fixed greedy completions are right and repeat byte for byte
  reasoning   reasoning and content split, no block markers in either
  streaming   streamed reasoning/content without markers; streamed tool call with an id
  chunks      needles at the start and end of 3K and 6K prompts (2 and 3 prefill chunks)
  needles     needles at 8K, 32K and 128K, depths 0.1 / 0.5 / 0.9
  logprobs    200 requests at concurrency 8 with logprobs 5 / top_logprobs 20: every sampled token is in its
              own top list with the same value, ids in the vocabulary
  cache       prefix-cache hit vs cold (cache_salt) greedy output: every hit answers right, and any difference
              starts at a near-tie of the cold run (top-2 gap under TIE_GAP); the identical count is reported
  concurrency 16 concurrent 4K needles with unique prefixes, all recalled
  overlong    a prompt over max_model_len is refused with HTTP 400 (not queued)
  long        200K and 250K needles, 3 concurrent 100K prompts resident at once, and 16 streams decoding
              through a 250K prefill (needs VLLM_LONG_PREFILL_SHARE on the server)
"""

import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import uuid

import requests

BASE = os.environ.get("BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "thinkingmachines/Inkling-Small")
MAX_LEN = int(os.environ.get("MAX_MODEL_LEN", "262144"))
S = requests.Session()
FILLER = (
    "The museum's east wing holds a collection of nineteenth-century maritime instruments, including a brass "
    "sextant recovered from a whaling vessel that ran aground near the northern shoals. "
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
MARK = "<|"
TIE_GAP = 0.5  # nats between the top two cold logprobs below which a hit may pick the other token
RESULTS = []
_UNIT = None


def gate(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""), flush=True)


def completion(prompt, max_tokens=32, salt=None, **extra):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, **extra}
    if salt:
        body["cache_salt"] = salt
    r = S.post(f"{BASE}/v1/completions", json=body, timeout=7200)
    r.raise_for_status()
    return r.json()


def chat(messages, max_tokens=1024, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, **extra}
    r = S.post(f"{BASE}/v1/chat/completions", json=body, timeout=7200)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def reasoning_of(msg):
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def ntokens(text):
    r = S.post(f"{BASE}/tokenize", json={"model": MODEL, "prompt": text}, timeout=600)
    r.raise_for_status()
    return r.json()["count"]


def filler(n_tokens):
    # Copies merge a token at each join, so measure the unit inside a run, not on its own.
    global _UNIT
    _UNIT = _UNIT or (ntokens(FILLER * 11) - ntokens(FILLER)) / 10
    return FILLER * max(1, int(n_tokens / _UNIT))


def needle_prompt(n_tokens, depth, code, unique=True):
    body = filler(n_tokens - 120)
    k = int(len(body) * depth)
    head = f"Document {uuid.uuid4()}.\n" if unique else ""
    return (
        head + body[:k] + f"\nIMPORTANT: The secret code for vault 7 is {code}.\n" + body[k:]
        + "\n\nQuestion: What is the secret code for vault 7? Answer with just the number.\nAnswer:"
    )  # fmt: skip


def needle_ok(n_tokens, depth):
    code = random.randint(100000, 999999)
    t0 = time.time()
    d = completion(needle_prompt(n_tokens, depth, code), 24)
    return str(code) in d["choices"][0]["text"], d["usage"]["prompt_tokens"], time.time() - t0


def stream(body):
    content, reasoning, calls = [], [], []
    body = {"model": MODEL, "temperature": 0.0, "stream": True, **body}
    with S.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: {"):
                continue
            for c in json.loads(line[6:]).get("choices", []):
                d = c.get("delta") or {}
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
                calls += d.get("tool_calls") or []
    return "".join(content), "".join(reasoning), calls


def g_smoke():
    m = chat([{"role": "user", "content": "What is 17*23? Answer briefly."}], reasoning_effort="low")
    gate("smoke/chat", "391" in (m.get("content") or ""), repr((m.get("content") or "")[:60]))
    m = chat([{"role": "user", "content": "What is the weather in Paris?"}], tools=TOOLS)
    tc = m.get("tool_calls") or []
    gate(
        "smoke/tool-call",
        bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"],
        json.dumps(tc[:1])[:120],
    )
    m = chat(
        [{"role": "user", "content": "Give a JSON object with keys a and b, both integers."}],
        response_format={"type": "json_object"},
    )
    try:
        obj = json.loads(m.get("content") or "")
        ok = isinstance(obj, dict) and {"a", "b"} <= set(obj)
    except ValueError:
        ok = False
    gate("smoke/json-mode", ok, repr((m.get("content") or "")[:80]))
    m = chat([{"role": "user", "content": "Tell me about Paris."}], tools=TOOLS, tool_choice="required")
    tc = m.get("tool_calls") or []
    gate("smoke/tool-choice-required", bool(tc) and tc[0]["function"]["name"] == "get_weather", json.dumps(tc)[:120])


def g_greedy():
    cases = [("The capital of France is", 20, "Paris"), ("Q: What is 7 times 8?\nA:", 8, "56")]
    for prompt, n, want in cases:
        a = completion(prompt, n)["choices"][0]["text"]
        b = completion(prompt, n)["choices"][0]["text"]
        gate(f"greedy/{want}", want in a and a == b, repr(a[:50]))


def g_reasoning():
    m = chat([{"role": "user", "content": "Is 91 prime? Think it through, then answer yes or no."}], 2048)
    content, reasoning = m.get("content") or "", reasoning_of(m)
    ok = "no" in content.lower() and len(reasoning) > 20 and MARK not in content and MARK not in reasoning
    gate("reasoning/split", ok, f"reasoning {len(reasoning)} ch, content {content[:60]!r}")


def g_streaming():
    content, reasoning, _ = stream(
        {"max_tokens": 2048, "messages": [{"role": "user", "content": "Is 91 prime? Think, then answer."}]}
    )
    ok = "no" in content.lower() and len(reasoning) > 20 and MARK not in content and MARK not in reasoning
    gate("streaming/reasoning-content", ok, f"reasoning {len(reasoning)} ch, content {content[:60]!r}")
    _, _, calls = stream(
        {
            "max_tokens": 1024,
            "tools": TOOLS,
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}],
        }
    )
    names = [c["function"]["name"] for c in calls if (c.get("function") or {}).get("name")]
    args = "".join((c.get("function") or {}).get("arguments") or "" for c in calls)
    ids = [c.get("id") for c in calls if (c.get("function") or {}).get("name")]
    ok = names == ["get_weather"] and "Paris" in args and all(isinstance(i, str) and i for i in ids)
    gate("streaming/tool-call", ok, f"names={names} ids={ids} args={args[:60]}")


def g_chunks():
    for n in (3000, 6000):
        for depth in (0.02, 0.98):
            ok, pt, dt = needle_ok(n, depth)
            gate(f"chunks/{n}@{depth}", ok, f"{pt} prompt tokens, {dt:.0f} s")


def g_needles(lengths=(8000, 32000, 131072)):
    for n in lengths:
        res = [needle_ok(n, depth) for depth in (0.1, 0.5, 0.9)]
        gate(f"needles/{n // 1000}K", all(r[0] for r in res), f"{[r[0] for r in res]} ttft~{res[0][2]:.0f} s")


def g_logprobs(nreq=200):
    vocab = int(os.environ.get("VOCAB", "201088"))

    def one(i):
        u = uuid.uuid4().hex
        try:
            if i % 2:
                d = completion(
                    f"[{u}] Item {i}: describe a {u[:4]} in one line.", 8, logprobs=5, return_tokens_as_token_ids=True
                )
                lp = d["choices"][0]["logprobs"]
                rows, k = zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"], strict=True), 5
            else:
                r = S.post(
                    f"{BASE}/v1/chat/completions",
                    timeout=900,
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": f"[{u}] Say a word about {i}."}],
                        "max_tokens": 8,
                        "temperature": 0.0,
                        "logprobs": True,
                        "top_logprobs": 20,
                        "return_tokens_as_token_ids": True,
                    },
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["logprobs"]["content"]
                rows = (
                    (t["token"], t["logprob"], {x["token"]: x["logprob"] for x in t["top_logprobs"]}) for t in content
                )
                k = 20
        except Exception as e:  # noqa: BLE001
            return f"ERROR {type(e).__name__}"
        for tok, lpv, top in rows:
            if not top or len(top) < k:
                return "TRUNCATED"
            if any(not 0 <= int(t.split(":")[1]) < vocab for t in top):
                return "OOV"
            if tok not in top or abs(top[tok] - lpv) > 1e-3:
                return "MISMATCH"
        return "CLEAN"

    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(nreq)))
    counts = {r: res.count(r) for r in set(res)}
    gate("logprobs/soak", counts.get("CLEAN", 0) == nreq, str(counts))


def g_cache():
    # A hit prefills only the uncached tail, so its chunk shapes differ from the cold run and BF16 rounding can
    # flip a near-tie later in the output. A wrong conv state or block table shows up as a wrong answer or as a
    # divergence where the cold run was confident.
    salt = uuid.uuid4().hex[:8]
    same = {"repeat": [], "extend": [], "shared": []}
    bad = []

    def text(p, n=48, s=None):
        c = completion(p, n, salt=s, logprobs=2)["choices"][0]
        return c["text"], c["logprobs"]["tokens"], c["logprobs"]["top_logprobs"]

    def check(kind, cold, hit, code=None):
        same[kind].append(cold[0] == hit[0])
        if code is not None and str(code) not in hit[0]:
            bad.append(f"{kind}: no {code} in {hit[0][:40]!r}")
        if cold[0] != hit[0]:
            k = next(
                (k for k, (a, b) in enumerate(zip(cold[1], hit[1], strict=False)) if a != b),
                min(len(cold[1]), len(hit[1])),
            )
            top = sorted(cold[2][k].values(), reverse=True) if k < len(cold[2]) else []
            gap = top[0] - top[1] if len(top) > 1 else 0.0
            if gap >= TIE_GAP:
                bad.append(f"{kind}: diverged at token {k}, top-2 gap {gap:.2f}")

    for i in range(3):
        p = f"[{salt}-{i}] " + needle_prompt(4000, 0.5, 55000 + i, unique=False)
        cold = text(p, s=str(uuid.uuid4()))
        text(p)  # fills the cache under the default salt
        check("repeat", cold, text(p), 55000 + i)
    for i in range(3):
        base = f"[{salt}-x{i}] " + needle_prompt(4000, 0.5, 66000 + i, unique=False)
        text(base)
        ext = base + f" {66000 + i}\n\nNow repeat the code twice, separated by a comma.\nAnswer:"
        check("extend", text(ext, s=str(uuid.uuid4())), text(ext))
    shared = f"[{salt}-shared] " + FILLER * 80
    text(shared, 1)
    prompts = [
        shared + f"\nIMPORTANT: The secret code for vault {i} is {77700 + i}.\n" + FILLER * 20
        + f"\n\nQuestion: What is the secret code for vault {i}? Answer with just the number.\nAnswer:"
        for i in range(8)
    ]  # fmt: skip
    with cf.ThreadPoolExecutor(8) as ex:
        colds = list(ex.map(lambda p: text(p, s=str(uuid.uuid4())), prompts))
    with cf.ThreadPoolExecutor(8) as ex:
        hits = list(ex.map(text, prompts))
    for i, (cold, hit) in enumerate(zip(colds, hits, strict=True)):
        check("shared", cold, hit, 77700 + i)
    total = sum(sum(v) for v in same.values())
    # The identical count is reported, not gated: two cold runs of the 8 concurrent shared-prefix prompts differ
    # just as often (batch composition changes the kernel shapes).
    gate("cache/hit-vs-cold", not bad, f"{total}/14 identical {same} {bad}")


def g_concurrency(n=16, tokens=4000):
    def one(i):
        ok, _, dt = needle_ok(tokens, 0.5)
        return ok

    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    gate(f"concurrency/{n}x{tokens // 1000}K", all(res), f"{sum(res)}/{n} in {time.time() - t0:.0f} s")


def g_overlong():
    p = filler(MAX_LEN + 2000)
    while (n := ntokens(p)) <= MAX_LEN:
        p += FILLER * 100
    t0 = time.time()
    r = S.post(f"{BASE}/v1/completions", json={"model": MODEL, "prompt": p, "max_tokens": 8}, timeout=600)
    gate("overlong/400", r.status_code == 400, f"{n} prompt tokens: HTTP {r.status_code} in {time.time() - t0:.1f} s")


def _metrics():
    t = S.get(f"{BASE}/metrics", timeout=30).text
    out = {}
    for k in ("num_requests_running", "num_preemptions_total"):
        m = re.search(rf"^vllm:{k}\{{[^}}]*\}} ([0-9.e+-]+)", t, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def g_long():
    g_needles((200000, 250000))
    # Three 100K prompts resident at once, 93% of the recipe's 327,680-token pool (the conv state does not take
    # KV blocks of its own).
    prompts = [f"[{uuid.uuid4()}] " + filler(100000) + "\nSummarize in one line." for _ in range(3)]
    peak, stop = [0], threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], int(_metrics()["num_requests_running"]))
            time.sleep(1)

    pre0 = _metrics()["num_preemptions_total"]
    th = threading.Thread(target=poll)
    th.start()
    with cf.ThreadPoolExecutor(3) as ex:
        codes = list(ex.map(lambda p: completion(p, 2000, ignore_eos=True)["usage"]["completion_tokens"], prompts))
    stop.set()
    th.join()
    pre = _metrics()["num_preemptions_total"] - pre0
    gate(
        "long/3x100K-resident",
        peak[0] >= 3 and pre == 0 and codes == [2000] * 3,
        f"peak running {peak[0]}, preemptions {pre:.0f}",
    )

    # 16 streams decoding, then a 250K needle: the streams keep decoding during the long prefill.
    stamps = [[] for _ in range(16)]

    def short(i):
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] Write a long essay about harbors."}],
            "max_tokens": 1800,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "reasoning_effort": "none",
        }
        with S.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=7200) as r:
            for line in r.iter_lines():
                if line.startswith(b"data: {"):
                    stamps[i].append(time.time())

    ex = cf.ThreadPoolExecutor(20)
    futs = [ex.submit(short, i) for i in range(16)]
    time.sleep(40)
    t_long = time.time()
    lf = ex.submit(needle_ok, 250000, 0.5)
    ok, _, ttft = lf.result()
    for f in futs:
        f.result()
    ex.shutdown()

    def rate(a, b):
        r = [(len(w) - 1) / (w[-1] - w[0]) for s in stamps if len(w := [t for t in s if a <= t <= b]) > 2]
        return statistics.median(r) if r else 0.0

    before, during = rate(t_long - 25, t_long), rate(t_long + 5, t_long + ttft)
    gate(
        "long/250K-prefill-share",
        ok and during >= 0.5 * before,
        f"needle {'ok' if ok else 'missed'}, 250K request {ttft:.0f} s, streams {before:.1f} tok/s before, "
        f"{during:.1f} during",
    )


GATES = {
    "smoke": g_smoke,
    "greedy": g_greedy,
    "reasoning": g_reasoning,
    "streaming": g_streaming,
    "chunks": g_chunks,
    "needles": g_needles,
    "logprobs": g_logprobs,
    "cache": g_cache,
    "concurrency": g_concurrency,
    "overlong": g_overlong,
    "long": g_long,
}
QUICK = [g for g in GATES if g != "long"]

if __name__ == "__main__":
    random.seed(int(time.time()))
    for name in sys.argv[1:] or QUICK:
        try:
            GATES[name]()
        except Exception as e:  # noqa: BLE001
            gate(f"{name}/error", False, f"{type(e).__name__}: {e}"[:200])
    sys.exit(0 if RESULTS and all(RESULTS) else 1)
