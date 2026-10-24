#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gates for GLM-5.3-Flash against a running server (Python stdlib; the image gate also needs Pillow).

    gates.py --base http://localhost:8000 GATE [GATE ...]

Each gate prints one line, `PASS <gate> <detail>` or `FAIL <gate> <detail>`, and the exit code is the number of
failed gates. The gates are the model's acceptance card:

  golden    12 greedy single-stream answers token-identical to golden_reference.json
  needles   three codes at 10/50/90 % depth, all retrieved, at 16K, 64K, 128K and near the served window (240K)
  tools     tool calls streamed and not (single, parallel, multi-turn, no call) and one after a 32K-token document
  images    solid colours, quadrants, counting, text before and after images; 5 images per prompt rejected (400)
  decode    single-stream decode at a 2K and a 128K prompt, warm: at least 15 and 10 tokens/s
  prefill   warm time to first token at 16K, 64K and 128K; 128K within 120 s

Prompt sizes are measured with the server's /tokenize, so they hold for this tokenizer.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "http://localhost:8000"
MODEL = ""
TIMEOUT = 3600


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def chat(messages, **kw) -> dict:
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 2048}
    body.update(kw)
    return post("/v1/chat/completions", body)


def chat_stream(messages, **kw):
    """Stream a chat completion. Returns (content, reasoning, tool_calls, t_first, t_last, completion_tokens)."""
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 2048, "stream": True}
    body["stream_options"] = {"include_usage": True}
    body.update(kw)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    content, reasoning, calls = [], [], {}
    t_first = t_last = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens", ntok)
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                got = False
                if delta.get("content"):
                    content.append(delta["content"])
                    got = True
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    reasoning.append(delta.get("reasoning") or delta.get("reasoning_content"))
                    got = True
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                    got = True
                if got:
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now
    return "".join(content), "".join(reasoning), [calls[i] for i in sorted(calls)], t_first, t_last, ntok


def ntokens(messages) -> int:
    return post("/tokenize", {"model": MODEL, "messages": messages, "add_generation_prompt": True})["count"]


# ---------------------------------------------------------------- long prompts
SUBJ = [
    "The harbour master",
    "A visiting botanist",
    "The night clerk",
    "The cartographer",
    "A retired engineer",
    "The station keeper",
    "A junior surveyor",
    "The ferry captain",
    "The archivist",
    "A field geologist",
]
VERB = ["recorded", "noted", "catalogued", "reported", "checked", "measured", "observed", "logged", "reviewed"]
OBJ = [
    "the tide levels",
    "the wind direction",
    "the soil moisture",
    "the lamp hours",
    "the cargo weights",
    "the rainfall totals",
    "the bridge clearances",
    "the well depths",
    "the radio schedule",
    "the fuel stock",
]
WHEN = [
    "before dawn",
    "after the evening meal",
    "during the storm",
    "at the turn of the month",
    "on the late shift",
    "while the road was closed",
    "ahead of the inspection",
    "after the thaw",
]


def filler(n_sentences: int, seed: int) -> list[str]:
    """Varied, numbered prose: a repeated block measures a different failure (this model degenerates on repetition)."""
    rnd = random.Random(seed)
    return [
        f"Entry {i}: {rnd.choice(SUBJ)} {rnd.choice(VERB)} {rnd.choice(OBJ)} {rnd.choice(WHEN)}, "
        f"and the value {rnd.randint(10, 9999)} went into ledger {rnd.randint(1, 400)}."
        for i in range(n_sentences)
    ]


def sized_document(target_tokens: int, seed: int, inserts=(), question="") -> list[dict]:
    """A user message of about target_tokens (measured), with `inserts` [(fraction, sentence)] placed at depths."""
    probe = filler(400, seed)
    per = max(1.0, (ntokens([{"role": "user", "content": " ".join(probe)}]) - 20) / len(probe))
    n = int(target_tokens / per)
    for _ in range(4):
        sents = filler(n, seed)
        for frac, text in sorted(inserts, reverse=True):
            sents.insert(int(frac * len(sents)), text)
        msg = [{"role": "user", "content": f"Document {seed}:\n" + " ".join(sents) + "\n\n" + question}]
        got = ntokens(msg)
        if abs(got - target_tokens) <= max(256, target_tokens // 200):
            return msg
        n = int(n * target_tokens / got)
    return msg


# ---------------------------------------------------------------- gates
def gate_golden():
    ref = json.loads((HERE / "golden_reference.json").read_text())
    same, notes = 0, []
    for i, case in enumerate(ref["cases"]):
        d = chat([{"role": "user", "content": case["prompt"]}], **ref["request"])
        ids = d["choices"][0].get("token_ids") or []
        want = case["token_ids"]
        if ids == want:
            same += 1
            continue
        k = next((j for j in range(min(len(ids), len(want))) if ids[j] != want[j]), min(len(ids), len(want)))
        gap = case["top2_gap"][k] if k < len(case["top2_gap"]) else None
        notes.append(f"#{i} differs at token {k} (reference top-2 gap {gap})")
    detail = f"{same}/{len(ref['cases'])} identical" + (f"; {'; '.join(notes)}" if notes else "")
    return same == len(ref["cases"]), detail


def gate_needles():
    rows, ok = [], True
    for target in (16_000, 64_000, 128_000, 240_000):
        rnd = random.Random(target)
        codes = {name: str(rnd.randint(10000, 99999)) for name in ("Amber", "Birch", "Cobalt")}
        inserts = [
            (f, f"The access code for vault {n} is {c}.")
            for f, (n, c) in zip((0.1, 0.5, 0.9), codes.items(), strict=False)
        ]
        q = "What are the access codes for vaults Amber, Birch and Cobalt? Answer as Amber=..., Birch=..., Cobalt=..."
        msgs = sized_document(target, target, inserts, q)
        n = ntokens(msgs)
        d = chat(msgs, max_tokens=1024, chat_template_kwargs={"reasoning_effort": "low"})
        text = d["choices"][0]["message"].get("content") or ""
        hits = sum(c in text for c in codes.values())
        ok &= hits == 3
        rows.append(f"{n // 1000}K {hits}/3")
    return ok, ", ".join(rows)


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _calls(message):
    return [
        {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
        for t in (message.get("tool_calls") or [])
    ]


def _cities(calls):
    out = []
    for c in calls:
        try:
            out.append(str(json.loads(c["arguments"]).get("city", "")).lower())
        except (ValueError, AttributeError):
            out.append("")
    return out


def gate_tools():
    kw = {"tools": [WEATHER], "chat_template_kwargs": {"reasoning_effort": "low"}}
    results = {}
    single = [{"role": "user", "content": "What's the weather in Phoenix right now?"}]
    parallel = [{"role": "user", "content": "What's the weather in Paris and in Tokyo right now? Check both."}]
    nocall = [{"role": "user", "content": "What is 12 + 30? Answer with the number only."}]
    for stream in (False, True):
        tag = "stream" if stream else "plain"

        def run(msgs, stream=stream):
            if stream:
                content, _, calls, *_ = chat_stream(msgs, **kw)
                return content, calls
            m = chat(msgs, **kw)["choices"][0]["message"]
            return m.get("content") or "", _calls(m)

        _, calls = run(single)
        ok = len(calls) == 1 and calls[0]["name"] == "get_weather"
        results[f"single-{tag}"] = ok and "phoenix" in _cities(calls)[0]
        _, calls = run(parallel)
        cities = " ".join(_cities(calls))
        results[f"parallel-{tag}"] = len(calls) >= 2 and "paris" in cities and "tokyo" in cities
        content, calls = run(nocall)
        results[f"nocall-{tag}"] = not calls and "42" in content
    # multi-turn: the tool result goes back and the answer uses it
    m = chat(single, **kw)["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    if calls:
        follow = single + [
            {"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp_f": 107, "sky": "clear"})},
        ]
        answer = chat(follow, **kw)["choices"][0]["message"].get("content") or ""
        results["multi-turn"] = "107" in answer
    else:
        results["multi-turn"] = False
    # a tool call after a long document
    long_msgs = sized_document(32_000, 7, (), "After reading the log above: what's the weather in Phoenix right now?")
    calls = _calls(chat(long_msgs, **kw)["choices"][0]["message"])
    results["after-32K"] = len(calls) == 1 and "phoenix" in _cities(calls)[0]
    passed = sum(results.values())
    bad = [k for k, v in results.items() if not v]
    return passed == len(results), f"{passed}/{len(results)}" + (f" failed: {', '.join(bad)}" if bad else "")


def _png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gate_images():
    from PIL import Image, ImageDraw

    cases = []
    for name, rgb in (("red", (220, 20, 20)), ("green", (20, 180, 20)), ("blue", (20, 20, 220))):
        img = Image.new("RGB", (448, 448), rgb)
        cases.append((f"solid-{name}", img, "What colour is this image? One word.", [name]))
    q = Image.new("RGB", (448, 448), (255, 255, 255))
    d = ImageDraw.Draw(q)
    d.rectangle([0, 0, 223, 223], fill=(220, 20, 20))
    d.rectangle([224, 0, 447, 223], fill=(20, 20, 220))
    d.rectangle([0, 224, 223, 447], fill=(20, 180, 20))
    d.rectangle([224, 224, 447, 447], fill=(240, 220, 20))
    cases.append(
        ("quadrants", q, "This image has four coloured quadrants. What colour is the TOP-LEFT one? One word.", ["red"])
    )
    c = Image.new("RGB", (448, 448), (255, 255, 255))
    d = ImageDraw.Draw(c)
    for cx, cy in ((110, 110), (330, 110), (220, 330)):
        d.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(0, 0, 0))
    cases.append(("count", c, "How many black circles are in this image? A single digit.", ["3", "three"]))

    kw = {"max_tokens": 1024, "chat_template_kwargs": {"reasoning_effort": "low"}}
    text_q = [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]
    results = {"text-before": "391" in (chat(text_q, **kw)["choices"][0]["message"].get("content") or "")}
    for name, img, question, accept in cases:
        content = [{"type": "image_url", "image_url": {"url": _png(img)}}, {"type": "text", "text": question}]
        answer = chat([{"role": "user", "content": content}], **kw)["choices"][0]["message"].get("content") or ""
        results[name] = any(a in answer.lower() for a in accept)
    results["text-after"] = "391" in (chat(text_q, **kw)["choices"][0]["message"].get("content") or "")
    too_many = [{"type": "image_url", "image_url": {"url": _png(cases[0][1])}}] * 5
    try:
        chat([{"role": "user", "content": too_many + [{"type": "text", "text": "Describe."}]}], max_tokens=8)
        results["5-images-rejected"] = False
    except urllib.error.HTTPError as e:
        results["5-images-rejected"] = e.code == 400
    passed = sum(results.values())
    bad = [k for k, v in results.items() if not v]
    return passed == len(results), f"{passed}/{len(results)}" + (f" failed: {', '.join(bad)}" if bad else "")


def _decode_rate(msgs, tokens=512):
    _, _, _, t0, t1, n = chat_stream(msgs, max_tokens=tokens, ignore_eos=True)
    return (n - 1) / (t1 - t0) if t0 and t1 and t1 > t0 and n > 1 else 0.0


def gate_decode():
    rows, ok = [], True
    for target, floor in ((2_000, 15.0), (128_000, 10.0)):
        rate = 0.0
        for rep in range(2):  # the second run is warm
            rate = _decode_rate(sized_document(target, 1000 + target + rep, (), "Summarize the log above."))
        ok &= rate >= floor
        rows.append(f"{target // 1000}K {rate:.1f} tok/s (floor {floor:g})")
    return ok, ", ".join(rows)


def gate_prefill():
    rows, warm128 = [], None
    for target in (16_000, 64_000, 128_000):
        ttft = None
        for rep in range(2):
            msgs = sized_document(target, 2000 + target + rep, (), "Reply with OK.")
            t0 = time.monotonic()
            chat(msgs, max_tokens=1)
            ttft = time.monotonic() - t0
        rows.append(f"{target // 1000}K {ttft:.1f} s")
        if target == 128_000:
            warm128 = ttft
    return warm128 is not None and warm128 < 120.0, "warm TTFT " + ", ".join(rows) + " (128K limit 120 s)"


GATES = {
    "golden": gate_golden,
    "needles": gate_needles,
    "tools": gate_tools,
    "images": gate_images,
    "decode": gate_decode,
    "prefill": gate_prefill,
}


def main() -> int:
    global BASE, MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--model", default="", help="served model name (default: the first one /v1/models lists)")
    ap.add_argument("gates", nargs="+", choices=sorted(GATES))
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        MODEL = a.model or json.loads(r.read())["data"][0]["id"]
    failed = 0
    for name in a.gates:
        t0 = time.monotonic()
        try:
            ok, detail = GATES[name]()
        except Exception as e:  # noqa: BLE001 - a crashed gate is a failed gate
            ok, detail = False, f"error {type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail} ({time.monotonic() - t0:.0f} s)", flush=True)
        failed += not ok
    return failed


if __name__ == "__main__":
    sys.exit(main())
