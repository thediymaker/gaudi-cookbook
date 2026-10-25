#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gates for the MiMo-V2.6 port (Flash and Pro), run against a live server (stdlib only).

    gates.py needles --tokens 131072 [--depths 0.1,0.5,0.9] [--seed 11]
    gates.py concurrent --tokens 500000 [--seeds 21,22,23,24]
    gates.py tools
    gates.py decode [--streams 1,4,16] [--min-tok-s 20]
    gates.py kv-record OUT.json        # greedy outputs of the KV A/B cases (warm)
    gates.py kv-diff A.json B.json     # token-identical or not

Every gate prints one line that starts with PASS or FAIL (and details above it). Set URL (default
http://localhost:8000) and MODEL (default: the first model the server lists).

Needles are plain facts ("the access code for the blue storage room is 4821-BRAVO"), not imperative notes: this
model treats inserted instructions as prompt injection and may decline to repeat them, which is not a retrieval
failure. The haystack is real prose, the Markdown files of the vLLM and vllm-gaudi checkouts in the image and of
this repository (override with HAYSTACK_DIRS, colon separated), so both the windowed and the global layers read
natural text.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import sys
import threading
import time
import urllib.request

URL = os.environ.get("URL", "http://localhost:8000").rstrip("/")


def post(path: str, body: dict, timeout: float = 7200) -> dict:
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model_name() -> str:
    if os.environ.get("MODEL"):
        return os.environ["MODEL"]
    with urllib.request.urlopen(URL + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]


def chat(messages, **kw) -> dict:
    return post("/v1/chat/completions", {"model": model_name(), "messages": messages, **kw})


def verdict(ok: bool, name: str, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return ok


# --------------------------------------------------------------------------------------------------------------------
# Needles
# --------------------------------------------------------------------------------------------------------------------
def haystack_dirs() -> list[str]:
    if os.environ.get("HAYSTACK_DIRS"):
        return os.environ["HAYSTACK_DIRS"].split(":")
    dirs = []
    for pkg in ("vllm", "vllm_gaudi"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            dirs.append(os.path.dirname(list(spec.submodule_search_locations)[0]))
    dirs.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    return dirs


_HAY: dict = {}


def haystack_ids(n_tokens: int) -> list[int]:
    """The first n_tokens of the haystack, tokenized file by file (each request stays far below the context)."""
    if len(_HAY.setdefault("ids", [])) < n_tokens:
        files = []
        for d in haystack_dirs():
            files += sorted(glob.glob(os.path.join(d, "**", "*.md"), recursive=True))
        ids: list[int] = []
        for f in files:
            with open(f, errors="ignore") as fh:
                text = fh.read()[:200_000] + "\n\n"
            ids += post("/tokenize", {"model": model_name(), "prompt": text, "add_special_tokens": False})["tokens"]
            if len(ids) >= n_tokens:
                break
        if len(ids) < n_tokens:
            sys.exit(f"haystack holds {len(ids)} tokens, need {n_tokens}: set HAYSTACK_DIRS")
        _HAY["ids"] = ids
    return _HAY["ids"][:n_tokens]


def detokenize(ids: list[int]) -> str:
    step = 65536
    return "".join(
        post("/detokenize", {"model": model_name(), "tokens": ids[i : i + step]})["prompt"]
        for i in range(0, len(ids), step)
    )


def needle_prompt(n_tokens: int, depth: float, rng: random.Random) -> tuple[str, str]:
    code = f"{rng.randint(1000, 9999)}-{rng.choice(['ALPHA', 'BRAVO', 'DELTA', 'OSCAR'])}"
    ids = haystack_ids(n_tokens)
    cut = int(len(ids) * depth)
    needle = f"\n\nFor the record, the access code for the blue storage room is {code}.\n\n"
    hay = detokenize(ids[:cut]) + needle + detokenize(ids[cut:])
    msg = (
        "<document>\n" + hay + "\n</document>\n\nThe document above mentions the access code for the blue storage "
        "room. What is it? Reply with the code only."
    )
    return msg, code


def ask_needle(msg: str, code: str, label: str) -> bool:
    t = time.time()
    r = chat([{"role": "user", "content": msg}], temperature=0, max_tokens=2048)
    m = r["choices"][0]["message"]
    content = m.get("content") or ""
    ok = code in content
    print(
        f"  {label} ctx {r['usage']['prompt_tokens']} {'hit ' if ok else 'MISS'} want {code} "
        f"({time.time() - t:.0f} s) -> {content[:80]!r}",
        flush=True,
    )
    return ok


def cmd_needles(a) -> bool:
    rng = random.Random(a.seed)
    depths = [float(x) for x in a.depths.split(",")]
    hits = sum(ask_needle(*needle_prompt(a.tokens, d, rng), f"depth {d:.2f}") for d in depths)
    return verdict(hits == len(depths), f"needles-{a.tokens // 1000}k", f"{hits}/{len(depths)} (seed {a.seed})")


def cmd_concurrent(a) -> bool:
    seeds = [int(s) for s in a.seeds.split(",")]
    prompts = [needle_prompt(a.tokens, 0.5, random.Random(s)) for s in seeds]
    res = [False] * len(seeds)

    def go(i):
        try:
            res[i] = ask_needle(*prompts[i], f"seed {seeds[i]}")
        except Exception as e:  # a dead engine shows up here
            print(f"  seed {seeds[i]} error {e}", flush=True)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(seeds))]
    t = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    n = sum(res)
    return verdict(
        n == len(seeds),
        f"concurrent-{len(seeds)}x{a.tokens // 1000}k",
        f"{n}/{len(seeds)} hit at once, {time.time() - t:.0f} s",
    )


# --------------------------------------------------------------------------------------------------------------------
# Tool calls
# --------------------------------------------------------------------------------------------------------------------
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}
STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the latest price of a stock by its ticker symbol",
        "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
    },
}
LOOKUP = {
    "type": "function",
    "function": {
        "name": "lookup_part",
        "description": "Look up a part in the inventory by its part number",
        "parameters": {
            "type": "object",
            "properties": {"part_number": {"type": "string"}},
            "required": ["part_number"],
        },
    },
}


def tool_calls(r) -> list[tuple[str, dict]]:
    out = []
    for c in r["choices"][0]["message"].get("tool_calls") or []:
        out.append((c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")))
    return out


def cmd_tools(a) -> bool:
    results = []

    def case(name, ok, detail):
        results.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)

    q = [{"role": "user", "content": "What's the weather in Lisbon right now? Use celsius."}]
    r = chat(q, tools=[WEATHER], temperature=0, max_tokens=2048)
    calls = tool_calls(r)
    case(
        "single call with arguments",
        len(calls) == 1 and calls[0][0] == "get_weather" and "lisbon" in calls[0][1].get("city", "").lower(),
        calls,
    )

    r = chat(
        [{"role": "user", "content": "What is NVDA trading at?"}],
        tools=[WEATHER, STOCK],
        temperature=0,
        max_tokens=2048,
    )
    calls = tool_calls(r)
    case(
        "picks the right tool",
        len(calls) == 1 and calls[0][0] == "get_stock_price" and calls[0][1].get("ticker", "").upper() == "NVDA",
        calls,
    )

    r = chat(
        [{"role": "user", "content": "What is 2 + 2? Answer directly."}],
        tools=[WEATHER],
        temperature=0,
        max_tokens=2048,
    )
    case("no call when none is needed", not tool_calls(r) and "4" in (r["choices"][0]["message"]["content"] or ""), "")

    r = chat(q, tools=[WEATHER], temperature=0, max_tokens=2048)
    call = r["choices"][0]["message"]["tool_calls"][0] if tool_calls(r) else None
    if call:
        follow = q + [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"temp_c": 23, "sky": "clear"})},
        ]
        r2 = chat(follow, tools=[WEATHER], temperature=0, max_tokens=2048)
        text = r2["choices"][0]["message"]["content"] or ""
        case("uses the tool result", "23" in text and not tool_calls(r2), repr(text[:80]))
    else:
        case("uses the tool result", False, "no first call")

    r = chat(
        [{"role": "user", "content": "Tell me something about Paris."}],
        tools=[WEATHER],
        tool_choice="required",
        temperature=0,
        max_tokens=2048,
    )
    calls = tool_calls(r)
    case("tool_choice required", len(calls) >= 1 and calls[0][0] == "get_weather", calls)

    rng = random.Random(5)
    part = f"PN-{rng.randint(10000, 99999)}"
    ids = haystack_ids(32768)
    det = detokenize(ids)
    cut = len(det) // 2
    doc = det[:cut] + f"\n\nThe replacement valve for the pump is part number {part}.\n\n" + det[cut:]
    msg = "<document>\n" + doc + "\n</document>\n\nLook up the replacement valve for the pump in the inventory."
    r = chat([{"role": "user", "content": msg}], tools=[LOOKUP], temperature=0, max_tokens=2048)
    calls = tool_calls(r)
    case(
        f"call after a {r['usage']['prompt_tokens']}-token document",
        len(calls) == 1 and calls[0][1].get("part_number") == part,
        calls,
    )
    return verdict(all(results), "tools", f"{sum(results)}/{len(results)}")


# --------------------------------------------------------------------------------------------------------------------
# Decode speed
# --------------------------------------------------------------------------------------------------------------------
def one_stream(res: list, i: int, n: int) -> None:
    body = {
        "model": model_name(),
        "prompt": f"Write a detailed essay about the history of the Roman Empire (variant {i}).",
        "max_tokens": n,
        "min_tokens": n,
        "ignore_eos": True,
        "temperature": 0.7,
        "stream": True,
    }
    req = urllib.request.Request(
        URL + "/v1/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    t0, first, count = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                count += 1
                first = first or time.time()
    end = time.time()
    res.append((first - t0, (count - 1) / (end - first) if count > 1 else 0.0))


def measure(c: int, n: int) -> tuple[float, float]:
    res: list = []
    threads = [threading.Thread(target=one_stream, args=(res, i, n)) for i in range(c)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return sum(r[1] for r in res) / c, sum(r[0] for r in res) / c


def cmd_decode(a) -> bool:
    ok, parts = True, []
    for c in (int(x) for x in a.streams.split(",")):
        measure(c, a.tokens)  # first run compiles new shapes; time the second
        tps, ttft = measure(c, a.tokens)
        print(f"  {c} streams: {tps:.1f} tok/s per stream, {tps * c:.0f} aggregate, TTFT {ttft:.2f} s", flush=True)
        parts.append(f"{tps:.1f}")
        ok &= tps >= a.min_tok_s
    return verdict(ok, "decode", f"{' / '.join(parts)} tok/s per stream at {a.streams} streams (floor {a.min_tok_s})")


# --------------------------------------------------------------------------------------------------------------------
# Window-sized KV vs the stock layout
# --------------------------------------------------------------------------------------------------------------------
WORDS = [
    "river",
    "lamp",
    "copper",
    "orbit",
    "maple",
    "signal",
    "harbor",
    "violet",
    "engine",
    "quartz",
    "meadow",
    "falcon",
    "ledger",
    "canyon",
    "ember",
    "glacier",
    "tundra",
    "beacon",
    "cobalt",
    "saddle",
]


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def kv_cases() -> list[tuple[str, str]]:
    """Short chats, prompts longer than one 8192-token chunk (sliding slots across chunk boundaries), and a repeat of
    a long prompt (a prefix-cache hit)."""
    out = [
        ("chat-1", "Explain in three sentences why the sky is blue."),
        ("chat-2", "Write a haiku about a lighthouse, then explain its imagery in one sentence."),
    ]
    for name, words, seed in (("long-12k", 9000, 1), ("long-30k", 22000, 2)):
        text = filler(words, seed)
        cut = len(text) // 3
        body = text[:cut] + " The access code is 7391-QX. " + text[cut:]
        out.append((name, body + "\n\nWhat is the access code mentioned above? Then summarise the text in one line."))
    out.append(("long-12k-repeat", out[2][1]))
    return out


def kv_greedy(prompt: str) -> list[str]:
    r = chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=128, logprobs=True, top_logprobs=1)
    return [t["token"] for t in (r["choices"][0].get("logprobs") or {}).get("content") or []]


def kv_batch(salt: str) -> dict:
    """Four prompts in one /v1/completions call, so both arms admit the same batch in the same order."""
    rng = random.Random("B")
    prompts = [
        f"[B-{i}] Read the list and then continue it with a short poem.\n"
        + " ".join(rng.choice(WORDS) for _ in range(n))
        + "\nPoem:"
        for i, n in enumerate((900, 4200, 7300, 11800))
    ]
    body = {"model": model_name(), "prompt": prompts, "max_tokens": 96, "temperature": 0.0, "logprobs": 1}
    if salt:
        body["cache_salt"] = salt  # same prompts, no prefix hits
    d = post("/v1/completions", body)
    return {f"batch-{c['index']}": c["logprobs"]["tokens"] for c in d["choices"]}


def cmd_kv_record(a) -> bool:
    # Compile mode computes a new batch shape differently the first time it runs: record the second run of each case.
    res = {}
    for name, prompt in kv_cases():
        kv_greedy(prompt)
        res[name] = kv_greedy(prompt)
    kv_batch("warm-1")
    res.update(kv_batch("warm-2"))
    with open(a.out, "w") as fh:
        json.dump(res, fh)
    print(f"recorded {len(res)} cases to {a.out}")
    return True


def cmd_kv_diff(a) -> bool:
    with open(a.a) as fa, open(a.b) as fb:
        x, y = json.load(fa), json.load(fb)
    bad = []
    for k in x:
        same = x[k] == y.get(k)
        if not same:
            n = next((i for i, (p, q) in enumerate(zip(x[k], y.get(k) or [], strict=False)) if p != q), None)
            bad.append(f"{k} (first difference at token {n})")
        print(f"  {'same' if same else 'DIFF'} {k}: {len(x[k])} tokens", flush=True)
    return verdict(not bad, "kv-identical", "token-identical" if not bad else "; ".join(bad))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("needles")
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--depths", default="0.1,0.5,0.9")
    p.add_argument("--seed", type=int, default=11)
    p = sp.add_parser("concurrent")
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--seeds", default="21,22,23,24")
    sp.add_parser("tools")
    p = sp.add_parser("decode")
    p.add_argument("--streams", default="1,4,16")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--min-tok-s", type=float, default=20.0)
    p = sp.add_parser("kv-record")
    p.add_argument("out")
    p = sp.add_parser("kv-diff")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    fn = {
        "needles": cmd_needles,
        "concurrent": cmd_concurrent,
        "tools": cmd_tools,
        "decode": cmd_decode,
        "kv-record": cmd_kv_record,
        "kv-diff": cmd_kv_diff,
    }[a.cmd]
    return 0 if fn(a) else 1


if __name__ == "__main__":
    sys.exit(main())
