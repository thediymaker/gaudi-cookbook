#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gates for the Muse-Glimmer-30B port, run against a server started with the recipe command.

Each gate prints one line, `PASS <gate> <detail>` or `FAIL <gate> <detail>`, and the exit code is the
number of failed gates. Stdlib only, plus Pillow for the vision gates (skipped with FAIL if missing).

    python3 gates.py --base-url http://127.0.0.1:8000 [--model meta-models/Muse-Glimmer-30B] [--quick]

--quick skips the 120K needle and the sampled forced-tool runs.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request

ARGS = argparse.Namespace()
FAILED = []

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
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


def post(path, body, timeout=1800):
    req = urllib.request.Request(ARGS.base_url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(messages, **kw):
    body = {"model": ARGS.model, "messages": messages, "max_tokens": 2048, "temperature": 0}
    body.update(kw)
    return post("/v1/chat/completions", body)


def content(resp):
    return resp["choices"][0]["message"].get("content") or ""


def report(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def gate(fn):
    def run():
        try:
            fn()
        except Exception as e:  # a gate that raises is a failed gate, the others still run
            report(fn.__name__.removeprefix("gate_"), False, f"{type(e).__name__}: {e}")

    run.__name__ = fn.__name__
    return run


def healthy():
    try:
        with urllib.request.urlopen(ARGS.base_url + "/health", timeout=30) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


@gate
def gate_smoke():
    a = content(chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}]))
    b = content(chat([{"role": "user", "content": "Compute 157*23+89. Reply with the number only."}]))
    report("smoke", "Paris" in a and "3700" in b.replace(",", ""), f"france={a.strip()[:40]!r} math={b.strip()[:40]!r}")


@gate
def gate_concurrency():
    # Batch > 1 is what the output-gate patch fixes: without it the second concurrent request stops the engine.
    n = 32

    def one(i):
        a, b = 100 + i, 7 * i + 3
        out = content(chat([{"role": "user", "content": f"What is {a}+{b}? Reply with the number only."}]))
        return str(a + b) in out

    with cf.ThreadPoolExecutor(n) as ex:
        ok = sum(ex.map(one, range(n)))
    report("concurrency", ok == n and healthy(), f"{ok}/{n} correct at {n} concurrent")


def needle_prompt(tokens, depth, key, value):
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    # 24 tokens per filler group on this tokenizer.
    reps = max(1, tokens // 24)
    pos = int(reps * depth)
    parts = [filler] * reps
    parts.insert(pos, f" The secret code for {key} is {value}. ")
    return "".join(parts) + f"\n\nWhat is the secret code for {key}? Reply with the code only."


@gate
def gate_needles():
    sizes = [(8000, 0.5), (32000, 0.05), (32000, 0.95)]
    if not ARGS.quick:
        sizes += [(120000, 0.5), (127000, 0.95)]
    rng = random.Random(7)
    hits, details = 0, []
    for tokens, depth in sizes:
        key, value = f"vault {rng.randint(100, 999)}", str(rng.randint(100000, 999999))
        r = chat([{"role": "user", "content": needle_prompt(tokens, depth, key, value)}], max_tokens=1024)
        ok = value in content(r)
        hits += ok
        details.append(f"{r['usage']['prompt_tokens']}@{depth}:{'ok' if ok else 'miss'}")
    report("needles", hits == len(sizes), " ".join(details))


def png_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def shapes_image(w=896, h=640):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    s = min(w, h) // 5
    d.ellipse([w * 0.08, h * 0.1, w * 0.08 + s, h * 0.1 + s], fill="red")
    d.rectangle([w * 0.4, h * 0.1, w * 0.4 + s, h * 0.1 + s], fill="blue")
    d.polygon([(w * 0.75, h * 0.1 + s), (w * 0.75 + s, h * 0.1 + s), (w * 0.75 + s / 2, h * 0.1)], fill="green")
    try:
        font = ImageFont.load_default(size=max(24, min(w, h) // 9))  # the text must fit the narrowest image
    except TypeError:
        font = ImageFont.load_default()
    d.text((w * 0.08, h * 0.6), "GAUDI 4729", fill="black", font=font)
    return img


def image_message(url, text):
    return [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}
    ]


@gate
def gate_vision():
    url = png_data_url(shapes_image())
    count = content(chat(image_message(url, "How many colored shapes are in this image? Reply with a number only.")))
    ocr = content(chat(image_message(url, "What text is written in the image? Reply with the text only.")))
    ok = "3" in count or "three" in count.lower()
    report("vision", ok and "4729" in ocr, f"shapes={count.strip()[:20]!r} text={ocr.strip()[:30]!r}")


@gate
def gate_vision_shapes_then_text():
    # Distinct image sizes (each a new vision shape), then text-only requests must not see image state.
    sizes = [(448, 448), (672, 448), (896, 672), (1120, 784), (560, 1008), (1344, 896), (784, 784), (1568, 1008)]
    ok, missed = 0, []
    for w, h in sizes:
        out = content(chat(image_message(png_data_url(shapes_image(w, h)), "What number is written in the image?")))
        ok += "4729" in out
        if "4729" not in out:
            missed.append(f"{w}x{h}:{out.strip()[:40]!r}")
    after = content(chat([{"role": "user", "content": "What is the capital of Japan? Answer in one word."}]))
    report(
        "vision_shapes_then_text",
        ok == len(sizes) and "Tokyo" in after,
        f"{ok}/{len(sizes)} image sizes read {missed}; text after images={after.strip()[:30]!r}",
    )


def tool_calls(resp):
    return resp["choices"][0]["message"].get("tool_calls") or []


@gate
def gate_forced_tools():
    msg = [{"role": "user", "content": "Tell me about the history of Rome."}]
    tools = [SEARCH, WEATHER]

    def required(seed):
        kw = {"temperature": 0} if seed is None else {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": seed}
        calls = tool_calls(chat(msg, tools=tools, tool_choice="required", **kw))
        return bool(calls) and all(json.loads(c["function"]["arguments"]) is not None for c in calls)

    runs = [None] * 8 + ([] if ARGS.quick else list(range(16)))
    with cf.ThreadPoolExecutor(4) as ex:
        ok_req = sum(ex.map(required, runs))
    named = tool_calls(
        chat(
            [{"role": "user", "content": "Rome."}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
    )
    named_ok = len(named) >= 1 and all(c["function"]["name"] == "get_weather" for c in named)
    args = json.loads(named[0]["function"]["arguments"]) if named else {}
    named_ok = named_ok and "city" in args and args.get("unit") in (None, "celsius", "fahrenheit")
    auto = chat([{"role": "user", "content": "What is the capital of France?"}], tools=tools, tool_choice="auto")
    auto_ok = not tool_calls(auto) and "Paris" in content(auto)
    report(
        "forced_tools",
        ok_req == len(runs) and named_ok and auto_ok,
        f"required {ok_req}/{len(runs)}; named args={args}; auto prose={auto_ok}",
    )


@gate
def gate_streaming_tools():
    body = {
        "model": ARGS.model,
        "messages": [{"role": "user", "content": "What is the weather in Rome?"}],
        "tools": [WEATHER],
        "tool_choice": "required",
        "max_tokens": 2048,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        ARGS.base_url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    calls, bad = {}, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                bad.append(str(chunk["error"])[:200])
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"]
            for tc in delta.get("tool_calls") or []:
                if "index" not in tc:
                    bad.append("index missing")
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                if tc.get("id"):
                    c["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") == "":
                    bad.append("empty name")
                c["name"] += fn.get("name") or ""
                c["args"] += fn.get("arguments") or ""
    ok = len(calls) >= 1 and not bad
    for c in calls.values():
        ok = ok and isinstance(c["id"], str) and c["name"] == "get_weather" and "city" in json.loads(c["args"])
    report("streaming_tools", ok, f"calls={list(calls.values())} problems={bad}")


@gate
def gate_logprobs():
    ok = 0
    for i in range(3):
        r = post(
            "/v1/completions",
            {
                "model": ARGS.model,
                "prompt": f"Count from {i} to {i + 20}:",
                "max_tokens": 120,
                "logprobs": 5,
                "temperature": 0,
            },
        )
        lp = r["choices"][0]["logprobs"]
        tops = lp["top_logprobs"]
        clean = all("<bad_id>" not in t for t in lp["tokens"]) and len(tops) == len(lp["tokens"])
        top1 = sum(1 for tok, top in zip(lp["tokens"], tops, strict=True) if top and max(top, key=top.get) == tok)
        ok += clean and top1 == len(tops)
    c = chat([{"role": "user", "content": "Say hello."}], max_tokens=200, logprobs=True, top_logprobs=5)
    chat_ok = bool(c["choices"][0]["logprobs"]["content"])
    report(
        "logprobs",
        ok == 3 and chat_ok and healthy(),
        f"completions {ok}/3 clean with greedy top-1; chat top_logprobs={chat_ok}",
    )


@gate
def gate_seeded():
    # Seeded rows used to segfault a tensor-parallel worker (seeded-sampling fix); repeats must match.
    def one(i):
        r = chat(
            [{"role": "user", "content": f"Write one sentence about the number {i}."}],
            max_tokens=200,
            temperature=1.0,
            seed=i,
        )
        return r["choices"][0]["message"].get("content") is not None or bool(
            r["choices"][0]["message"].get("reasoning")
        )

    with cf.ThreadPoolExecutor(64) as ex:
        done = sum(ex.map(one, range(128)))
    rep = [
        content(chat([{"role": "user", "content": "Name a random fruit."}], max_tokens=300, temperature=1.0, seed=1234))
        for _ in range(3)
    ]
    report(
        "seeded",
        done == 128 and len(set(rep)) == 1 and healthy(),
        f"{done}/128 seeded at 64 concurrent; repeats identical={len(set(rep)) == 1}",
    )


@gate
def gate_reasoning_split():
    r = chat([{"role": "user", "content": "Is 221 a prime number? Explain briefly."}], max_tokens=4096)
    m = r["choices"][0]["message"]
    text = m.get("content") or ""
    leaked = any(t in text for t in ("<|message|>", "<|eom|>", "to=self", "<|start|>"))
    report(
        "reasoning_split",
        bool(text) and not leaked and ("13" in text or "17" in text),
        f"content={text.strip()[:60]!r}",
    )


GATES = [
    gate_smoke,
    gate_concurrency,
    gate_needles,
    gate_vision,
    gate_vision_shapes_then_text,
    gate_forced_tools,
    gate_streaming_tools,
    gate_logprobs,
    gate_seeded,
    gate_reasoning_split,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these gates")
    ap.parse_args(namespace=ARGS)
    start = time.time()
    for g in GATES:
        if ARGS.only and g.__name__.removeprefix("gate_") not in ARGS.only:
            continue
        g()
    report("health_after", healthy(), f"{time.time() - start:.0f} s total")
    return len(FAILED)


if __name__ == "__main__":
    sys.exit(main())
