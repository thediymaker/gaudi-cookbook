#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle in a haystack: a secret code at a given depth in plain filler, asked for at the end (thinking off).

PASS: every length finds its code. The default lengths (12,000 and 40,500 words) are about 13K and 45K tokens.

  python3 needle_gate.py [--words 12000,40500] [--depth 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import NO_THINK, filler, post, verdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="12000,40500")
    ap.add_argument("--depth", type=float, default=0.5)
    args = ap.parse_args()
    rng = random.Random(time.time_ns())  # fresh secrets and filler each run, so no cached prefix answers
    hits = []
    for n in [int(x) for x in args.words.split(",")]:
        secret = "-".join("".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(4)) for _ in "abc")
        words = filler(n, rng.randrange(1 << 30))
        k = int(len(words) * args.depth)
        text = " ".join(words[:k]) + f" The secret access code for the vault is {secret}. Remember it. "
        text += " ".join(words[k:]) + "\n\nQuestion: What is the secret access code for the vault? Reply with the code."
        t = time.time()
        r = post(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": text}], "max_tokens": 24, "temperature": 0, **NO_THINK},
        )
        ans = r["choices"][0]["message"].get("content") or ""
        hits.append(secret in ans)
        print(
            json.dumps(
                {
                    "words": n,
                    "prompt_tokens": r["usage"]["prompt_tokens"],
                    "depth": args.depth,
                    "hit": hits[-1],
                    "answer": ans[:80],
                    "seconds": round(time.time() - t, 1),
                }
            ),
            flush=True,
        )
    return 0 if verdict("NEEDLES", all(hits), f"{sum(hits)}/{len(hits)}") else 1


if __name__ == "__main__":
    sys.exit(main())
