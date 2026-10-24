#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Request isolation: no request may ever see another request's content.

Floods the server with concurrent requests. Each carries a unique secret code and an invented topic word at a random
depth in filler of random length, and asks for its own code. With --shared-prefix-words every prompt opens with the
same filler, so all requests share those prefix-cached blocks and each request's secret sits after them; --repeat
sends every request a second time (served mostly from the prefix cache). A fraction of requests first open a
streaming request and drop it after a few chunks (aborts free blocks mid-flight).

PASS: 0 foreign codes or topics in any reply, no errors, own-code retrieval >= 98%, and three temperature-0 probes
answer identically before and after the flood.

For the cache-tag count, serve with DSV41_TAG_CHECK=count: the server log's "tag check" line must then report 0
violating rows (run_gates.sh checks it when given the log).

  python3 isolation_gate.py --n 24 --workers 8 --min-words 600 --max-words 3000 --shared-prefix-words 4000 --repeat
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import string
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import NO_THINK, URL, chat, filler, metric, model, post, verdict  # noqa: E402

PROBES = [
    "List the first five prime numbers.",
    "What is 17 * 23? Answer with the number only.",
    "Name three primary colors, comma separated.",
]


def pseudo_word(rng: random.Random) -> str:
    """An invented word: real topic words collide semantically, so a hit on one would not prove a leak."""
    c, v = "bdfgklmnprstvz", "aeiou"
    return "".join(rng.choice(c) + rng.choice(v) for _ in range(3)) + rng.choice(c)


def make_request(i: int, rng: random.Random, args) -> dict:
    code = "".join(rng.choice(string.ascii_uppercase + "234567") for _ in range(12))
    topic = pseudo_word(rng)
    n = rng.randint(args.min_words, args.max_words)
    w = filler(n, 1000 + i + args.seed * 100003)
    k = rng.randint(0, n)
    text = " ".join(w[:k]) + f" Note for the {topic} file: the access code is {code}. " + " ".join(w[k:])
    if args.shared_prefix_words:
        text = " ".join(filler(args.shared_prefix_words, 7 + args.seed)) + "\n\n" + text
    q = (
        f"{text}\n\nWhat is the access code in the note for the {topic} file? "
        f"Reply with the code first, then say in one sentence that the {topic} file is now unlocked."
    )
    return {"i": i, "code": code, "topic": topic, "prompt": q}


def aborted_stream(req: dict) -> None:
    """Open a streaming request, read a few chunks, and drop the connection."""
    u = urllib.parse.urlparse(URL)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    body = {
        "model": model(),
        "messages": [{"role": "user", "content": req["prompt"]}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": True,
        **NO_THINK,
    }
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = c.getresponse()
    for _ in range(3):
        if not resp.readline():
            break
    c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-words", type=int, default=600)
    ap.add_argument("--max-words", type=int, default=3000)
    ap.add_argument("--shared-prefix-words", type=int, default=4000)
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--abort-frac", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    reqs = [make_request(i, rng, args) for i in range(args.n)]
    allreqs = reqs + ([dict(r, i=r["i"] + len(reqs)) for r in reqs] if args.repeat else [])
    before = [chat(p, 32) for p in PROBES]
    pre0 = metric("vllm:num_preemptions_total")
    results: dict[int, str] = {}
    lock = threading.Lock()
    queue = list(allreqs)
    rng.shuffle(queue)

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                req = queue.pop()
                abort = rng.random() < args.abort_frac
            try:
                if abort:
                    aborted_stream(req)
                r = post(
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": req["prompt"]}],
                        "max_tokens": args.max_tokens,
                        "temperature": 0,
                        **NO_THINK,
                    },
                )
                out = r["choices"][0]["message"].get("content") or ""
            except Exception as e:
                out = f"<error {e}>"
            with lock:
                results[req["i"]] = out

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = time.time() - t0
    preemptions = metric("vllm:num_preemptions_total") - pre0
    after = [chat(p, 32) for p in PROBES]

    codes = {r["i"]: r["code"] for r in allreqs}
    topics = {r["i"]: r["topic"] for r in allreqs}
    own = errors = 0
    foreign = []
    for i, out in results.items():
        if out.startswith("<error"):
            errors += 1
            continue
        own += codes[i] in out
        low = out.lower()
        for j in codes:
            if j != i and codes[j] != codes[i] and codes[j] in out:  # a repeat shares its original's code
                foreign.append((i, "code", j))
            if j != i and topics[j] != topics[i] and topics[j] in low:
                foreign.append((i, "topic", j))
    done = len(results) - errors
    rate = own / max(1, done)
    print(
        json.dumps(
            {
                "requests": len(allreqs),
                "completed": done,
                "errors": errors,
                "seconds": round(seconds),
                "own_code_hits": own,
                "own_rate": round(rate, 4),
                "foreign_hits": foreign[:10],
                "preemptions": preemptions,
                "probes_identical": before == after,
            },
            indent=1,
        )
    )
    misses = [(i, results[i][:80]) for i in results if codes[i] not in results[i]][:8]
    if misses:
        print("misses:", json.dumps(misses))
    ok = not foreign and errors == 0 and rate >= 0.98 and before == after
    detail = f"own {own}/{done}, {len(foreign)} foreign hits, probes identical {before == after}"
    return 0 if verdict("ISOLATION", ok, detail) else 1


if __name__ == "__main__":
    sys.exit(main())
