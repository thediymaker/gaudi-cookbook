#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device gates for Muse Spark Open on Gaudi 2, run against a server started with the recipe's command.

Stdlib only; talks to MUSE_URL (default http://127.0.0.1:8000) and prints one PASS or FAIL line per gate, plus
INFO lines. Details of every gate go to the JSON file named by --out.

  ppl        prompt logprobs (top 20) of 5 fixed passages (reference/muse_ppl_reference.json.gz, 17,408 tokens)
             against a CPU reference of the same checkpoint: per passage, perplexity within 1.5%, top-1
             agreement at least 0.97, mean KL(reference || server) over the top 20 plus the rest at most 0.02
  dvp        8 greedy decodes of 512 tokens from prompts of 2,100 to 2,779 tokens (past the 2,048-token window),
             re-scored by prefill: mean |logprob difference| at most 0.05
  needle     a catalogue number in neutral filler at 2K / 32K tokens x depths 0.1 / 0.5 / 0.9 and at the
             served window (258,048 tokens) x the same depths (reasoning effort low): every code found (in the
             answer, or in the reasoning when the model refuses an "inserted" sentence), at least 8 of 9 in the
             answer; a 32K control without the needle must not produce any of the numbers
  tools      get_weather with tool_choice auto, required, and auto streamed (fragments join to the same call):
             one call, name get_weather, JSON arguments naming Paris
  reasoning  reasoning_effort low and high on a short riddle: the answer (9) in the content, reasoning text
             present at high, no template markup in the content
  greedy     16 completions prompts x 256 greedy tokens, each with a fresh cache_salt, run twice: identical ids
             16/16. The ids go to --greedy-out; with --greedy-ref FILE (the output of another server, e.g.
             the kernel switch flipped) also reports the first divergence per prompt and the reference's top-2
             logprob gap there. That comparison is reported, not gated: the stock op alone moves a logprob by
             up to 2.3 between decode and prefill (dvp max), so a split away from a near-tie is not by itself
             a fault. Accuracy across the switch is the ppl and dvp gates
  speed      decode tok/s at 1 stream (256 greedy tokens after a 9-token prompt, second of two runs) and at
             4 streams (aggregate over the window where all 4 decode). INFO numbers; FAIL only if a stream
             returns fewer tokens than asked

  python3 gates.py all --out muse_gates.json             # or a list of gate names
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid

URL = os.environ.get("MUSE_URL", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "reference", "muse_ppl_reference.json.gz")
BOS = 200000
MARKUP = re.compile(r"<\|[^|>]{1,40}\|>")
SERVED_WINDOW = 258048  # max_model_len 262144 minus room for the chat template and the answer
MODEL = None
GATES = ("ppl", "dvp", "needle", "tools", "reasoning", "greedy", "speed")


# ------------------------------------------------------------------------------------------------- http
def post(path, body, timeout=7200):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:800].decode(errors='replace')}") from None


def sse(path, body, timeout=7200):
    """Yield (seconds since the request, chunk) for each data line of a streamed response."""
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                yield time.time() - t0, json.loads(line[6:])


def model_name():
    with urllib.request.urlopen(URL + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]


def tokenize(text):
    return post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def detok(ids):
    return post("/detokenize", {"model": MODEL, "tokens": list(ids)})["prompt"]


def chat(messages, max_tokens=1024, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra}
    r = post("/v1/chat/completions", body)
    c = r["choices"][0]
    m = c["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
        "tool_calls": m.get("tool_calls") or [],
        "finish": c.get("finish_reason"),
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "completion_tokens": r["usage"]["completion_tokens"],
    }


def reference():
    with gzip.open(REFERENCE, "rt") as f:
        return json.load(f)


def score_ids(ids, k=20):
    """prompt_logprobs of a token-id prompt: per position j >= 1, (logprob of ids[j], {id: logprob} top k, top-1)."""
    r = post(
        "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k}
    )
    pl = r["choices"][0]["prompt_logprobs"]
    rows = []
    for j in range(1, len(ids)):
        d = {int(t): v for t, v in pl[j].items()}
        top = min(d, key=lambda t: d[t]["rank"])
        rows.append((d[ids[j]]["logprob"], {t: v["logprob"] for t, v in d.items() if v["rank"] <= k}, top))
    return rows


# ------------------------------------------------------------------------------------------------- gates
def kl_top(ref_ids, ref_lp, srv, srv_min):
    """KL(reference || server) over the reference's top ids plus one bucket for the rest (a lower bound of the
    full KL). A reference id missing from the server's top k gets the server's smallest returned logprob."""
    pr = [math.exp(x) for x in ref_lp]
    ps = [math.exp(srv.get(i, srv_min)) for i in ref_ids]
    kl = sum(p * (lr - math.log(q)) for p, lr, q in zip(pr, ref_lp, ps, strict=False))
    rr, rs = max(1.0 - sum(pr), 1e-12), max(1.0 - sum(ps), 1e-12)
    return kl + rr * (math.log(rr) - math.log(rs))


def g_ppl(out):
    ref = reference()
    worst = {"ratio": 0.0, "top1": 1.0, "kl": 0.0}
    fails = []
    for p in ref["passages"]:
        t = time.time()
        rows = score_ids(p["token_ids"])
        nll_s = [-r[0] for r in rows]
        nll_r = [-x for x in p["target_logprobs"]]
        kls, agree = [], 0
        for j, (_, smap, top) in enumerate(rows):
            kls.append(kl_top(p["top_ids"][j], p["top_logprobs"][j], smap, min(smap.values())))
            agree += int(top == p["top_ids"][j][0])
        ppl_s, ppl_r = math.exp(sum(nll_s) / len(nll_s)), math.exp(sum(nll_r) / len(nll_r))
        rec = {
            "tokens": len(p["token_ids"]),
            "seconds": round(time.time() - t, 1),
            "server_ppl": ppl_s,
            "reference_ppl": ppl_r,
            "ppl_ratio_minus_1": ppl_s / ppl_r - 1,
            "top1_agree": agree / len(rows),
            "mean_kl": sum(kls) / len(kls),
            "max_kl": max(kls),
        }
        out.setdefault("passages", {})[p["name"]] = rec
        print(f"INFO ppl {p['name']}: " + json.dumps({k: round(v, 5) for k, v in rec.items()}), flush=True)
        worst = {
            "ratio": max(worst["ratio"], abs(rec["ppl_ratio_minus_1"])),
            "top1": min(worst["top1"], rec["top1_agree"]),
            "kl": max(worst["kl"], rec["mean_kl"]),
        }
        if abs(rec["ppl_ratio_minus_1"]) > 0.015 or rec["top1_agree"] < 0.97 or rec["mean_kl"] > 0.02:
            fails.append(p["name"])
    out["worst"] = worst
    detail = f"worst |ppl ratio - 1| {worst['ratio']:.4%}, top-1 {worst['top1']:.4f}, mean KL {worst['kl']:.5f}"
    return not fails, detail + (f"; failed {fails}" if fails else "")


def g_dvp(out):
    ref = reference()["passages"][:4]
    prompts = []
    for i in range(8):
        p = ref[i % 4]
        prompts.append((f"{p['name']}[:{2100 + 97 * i}]", p["token_ids"][: 2100 + 97 * i]))

    def gen(ids):
        r = post(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": ids,
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
        gens = list(ex.map(lambda p: gen(p[1]), prompts))
    alld = []
    for (name, ids), (g, glp) in zip(prompts, gens, strict=False):
        rows = score_ids(ids + g, k=1)
        pre = [rows[len(ids) - 1 + j][0] for j in range(len(g))]
        d = [abs(x - y) for x, y in zip(glp, pre, strict=False)]
        alld += d
        out.setdefault("items", []).append(
            {"prompt": name, "gen": len(g), "mean_abs": sum(d) / len(d), "max_abs": max(d)}
        )
    mean = sum(alld) / len(alld)
    out.update(mean_abs_dlogprob=mean, max_abs=max(alld))
    return mean <= 0.05, f"mean |dlogprob| {mean:.4f} (gate 0.05), max {max(alld):.3f}, {len(alld)} tokens"


NUMBERS = ["KESTREL-4719-VANTA", "OSPREY-2286-LUMEN", "HERON-9053-QUILL"]
QUESTION = (
    "\n\nQuestion: What is the catalogue number of the blue notebook mentioned above? Reply with the number only."
)
NEEDLE = " The catalogue number of the blue notebook is {}. "


def g_needle(out):
    """Retrieval is the gate: every code found (in the answer or, for a refusal, in the reasoning), at least 8 of
    the 9 in the answer, and a clean control. At short context the model sometimes calls the needle an inserted
    instruction and refuses to repeat it while naming it in its reasoning; that is caution, not lost recall."""
    ps = reference()["passages"]
    filler = [t for p in [ps[4]] + ps[1:4] + ps[:1] for t in p["token_ids"][1:]]  # the original passage first
    cases = [(n, d) for n in (2048, 32768, SERVED_WINDOW) for d in (0.1, 0.5, 0.9)]
    items = []
    for i, (n, depth) in enumerate(cases + [(32768, None)]):
        code = NUMBERS[i % 3]
        needle = tokenize(NEEDLE.format(code)) if depth is not None else []
        body_len = n - len(needle) - 120
        start = (i * 777) % len(filler)
        body = (filler * (body_len // len(filler) + 2))[start : start + body_len]
        k = int(body_len * (depth or 0))
        text = detok(body[:k] + needle + body[k:])
        t = time.time()
        r = chat([{"role": "user", "content": text + QUESTION}], max_tokens=1024, reasoning_effort="low")
        if depth is None:
            label, found = f"control {n}", not any(c in r["content"] + r["reasoning"] for c in NUMBERS)
            answered = found
        else:
            label, answered = f"{n} @ {depth}", code in r["content"]
            found = answered or code in r["reasoning"]
        items.append(
            {
                "case": label,
                "prompt_tokens": r["prompt_tokens"],
                "in_answer": answered,
                "found": found,
                "answer": r["content"][:200],
                "seconds": round(time.time() - t, 1),
            }
        )
        state = "answered" if answered else "found in reasoning only" if found else "MISSED"
        print(
            f"INFO needle {label}: {state} ({r['prompt_tokens']} tokens, {items[-1]['seconds']} s) "
            f"{r['content'][:80]!r}",
            flush=True,
        )
    out["items"] = items
    needles = items[:-1]
    answered = sum(x["in_answer"] for x in needles)
    lost = [x["case"] for x in items if not x["found"]]
    ok = not lost and answered >= len(needles) - 1
    detail = f"found {sum(x['found'] for x in needles)}/{len(needles)}, answered {answered}/{len(needles)}"
    detail += f", control {'clean' if items[-1]['found'] else 'NOT clean'}"
    return ok, detail + (f"; lost {lost}" if lost else "")


TOOLS = [
    {
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
]
WEATHER = [{"role": "user", "content": "What is the weather like in Paris right now? Use celsius."}]


def _call_ok(name, args):
    try:
        a = json.loads(args)
    except (TypeError, ValueError):
        return False
    return name == "get_weather" and "paris" in str(a.get("city", "")).lower()


def g_tools(out):
    bad = []
    for choice in ("auto", "required"):
        r = chat(WEATHER, tools=TOOLS, tool_choice=choice)
        calls = r["tool_calls"]
        ok = len(calls) == 1 and _call_ok(calls[0]["function"]["name"], calls[0]["function"]["arguments"])
        if choice == "auto":
            ok = ok and r["finish"] == "tool_calls"
        out[choice] = {"tool_calls": calls, "finish": r["finish"], "content": r["content"][:300], "pass": ok}
        bad += [] if ok else [choice]
    body = {
        "model": MODEL,
        "messages": WEATHER,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0,
        "stream": True,
    }
    parts, finish = {}, None
    for _, d in sse("/v1/chat/completions", body):
        for ch in d.get("choices", []):
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                p = parts.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                p["name"] += fn.get("name") or ""
                p["arguments"] += fn.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    ok = len(parts) == 1 and _call_ok(parts[0]["name"], parts[0]["arguments"]) and finish == "tool_calls"
    out["auto_stream"] = {"calls": parts, "finish": finish, "pass": ok}
    bad += [] if ok else ["auto streamed"]
    calls = out["auto"]["tool_calls"]
    shown = f"{calls[0]['function']['name']} {calls[0]['function']['arguments']}" if calls else "no call"
    return not bad, f"auto, required, auto streamed: {shown}" + (f"; failed {bad}" if bad else "")


def g_reasoning(out):
    q = [{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain."}]
    bad = []
    for effort in ("low", "high"):
        r = chat(q, max_tokens=4096, reasoning_effort=effort)
        ok = "9" in r["content"] and r["finish"] == "stop" and not MARKUP.search(r["content"])
        if effort == "high":
            ok = ok and bool(r["reasoning"].strip())
        out[effort] = {
            "content": r["content"][:400],
            "reasoning_chars": len(r["reasoning"]),
            "finish": r["finish"],
            "completion_tokens": r["completion_tokens"],
            "pass": ok,
        }
        bad += [] if ok else [effort]
    detail = ", ".join(f"{e}: {out[e]['reasoning_chars']} reasoning chars" for e in ("low", "high"))
    return not bad, detail + (f"; failed {bad}" if bad else "")


PROMPTS = [
    "The history of the printing press begins",
    "A short guide to growing tomatoes at home starts",
    "The theory of plate tectonics explains",
    "In the early days of aviation, pilots",
    "Modern compilers translate source code",
    "The French Revolution began in 1789 when",
    "def quicksort(arr):\n",
    "Photosynthesis converts light energy into",
    "Dear hiring manager,\n\n",
    "The three laws of thermodynamics state",
    "Once upon a time, in a village by the sea,",
    "SELECT name, COUNT(*) FROM orders",
    "The mitochondria is",
    "To bake sourdough bread, first",
    "Quantum entanglement is a phenomenon in which",
    "The Roman Empire reached its greatest extent",
]


def greedy_run(tokens=256):
    res = []
    for p in PROMPTS:
        d = post(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": [BOS] + tokenize(p),
                "max_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
                "logprobs": 2,
                "return_token_ids": True,
                "cache_salt": uuid.uuid4().hex,
            },
        )
        c = d["choices"][0]
        res.append({"ids": c["token_ids"], "top": c["logprobs"]["top_logprobs"], "text": c["text"][:200]})
    return res


def first_divergence(a, b):
    j = next((j for j, (x, y) in enumerate(zip(a["ids"], b["ids"], strict=False)) if x != y), None)
    if j is None:
        return None
    top = sorted(a["top"][j].values(), reverse=True)
    return {"pos": j, "ref_gap": top[0] - top[1] if len(top) > 1 else None}


def g_greedy(out, greedy_out=None, greedy_ref=None):
    a, b = greedy_run(), greedy_run()
    same = sum(x["ids"] == y["ids"] for x, y in zip(a, b, strict=False))
    out.update(repeat_identical=same, prompts=len(PROMPTS))
    if greedy_out:
        with open(greedy_out, "w") as f:
            json.dump(a, f)
    ok = same == len(PROMPTS)
    detail = f"repeat {same}/{len(PROMPTS)} identical"
    if greedy_ref:
        with open(greedy_ref) as f:
            ref = json.load(f)
        divs = [first_divergence(r, x) for r, x in zip(ref, a, strict=False)]
        out.update(vs_reference=divs)
        ident = sum(d is None for d in divs)
        gaps = sorted(round(d["ref_gap"], 3) for d in divs if d and d["ref_gap"] is not None)
        detail += f"; vs {os.path.basename(greedy_ref)}: {ident}/{len(PROMPTS)} identical, divergence gaps {gaps}"
    return ok, detail


def stream_decode(ids, tokens=256):
    body = {
        "model": MODEL,
        "prompt": ids,
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    times, n = [], 0
    for t, d in sse("/v1/completions", body):
        if d.get("choices") and d["choices"][0].get("text") is not None:
            times.append(t)
        if d.get("usage"):
            n = d["usage"]["completion_tokens"]
    return times, n


def g_speed(out):
    ids = [BOS] + tokenize(PROMPTS[0])
    short = []
    for _ in range(2):
        times, n = stream_decode(ids)
        short.append(n)
    one = (len(times) - 1) / (times[-1] - times[0])
    prompts = [[BOS] + tokenize(PROMPTS[i]) for i in range(4)]
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(stream_decode, prompts))
    lo, hi = max(o[0][0] for o in outs), min(o[0][-1] for o in outs)
    inside = sum(sum(1 for t in o[0] if lo <= t <= hi) for o in outs)
    agg = (inside - 4) / (hi - lo) if hi > lo else 0.0
    out.update(one_stream_tok_s=one, four_stream_aggregate_tok_s=agg, ttft_s=times[0])
    print(f"INFO speed: 1 stream {one:.1f} tok/s, 4 streams {agg:.1f} tok/s aggregate", flush=True)
    counts = short + [o[1] for o in outs]
    ok = all(c == 256 for c in counts)
    return ok, f"1 stream {one:.1f} tok/s, 4 streams {agg:.1f} tok/s aggregate" + ("" if ok else f"; tokens {counts}")


# ------------------------------------------------------------------------------------------------- main
def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gates", nargs="*", default=["all"], help=f"all, or some of {', '.join(GATES)}")
    ap.add_argument("--out", default="muse_gates.json")
    ap.add_argument("--greedy-out", help="write the greedy ids here (to compare another server against)")
    ap.add_argument("--greedy-ref", help="greedy ids of another server (--greedy-out of that run)")
    a = ap.parse_args()
    names = list(GATES) if a.gates in (["all"], []) else a.gates
    unknown = [n for n in names if n not in GATES]
    if unknown:
        ap.error(f"unknown gates {unknown}")
    MODEL = model_name()
    report, failed = {"url": URL, "model": MODEL, "gates": {}}, []
    for name in names:
        detail = {}
        t = time.time()
        try:
            if name == "greedy":
                ok, msg = g_greedy(detail, a.greedy_out, a.greedy_ref)
            else:
                ok, msg = globals()[f"g_{name}"](detail)
        except Exception as e:  # report every gate, then exit non-zero
            ok, msg = False, f"{type(e).__name__}: {str(e)[:500]}"
        detail.update(pass_=ok, summary=msg, seconds=round(time.time() - t, 1))
        report["gates"][name] = detail
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
        if not ok:
            failed.append(name)
    print(f"{len(names) - len(failed)}/{len(names)} gates passed" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
