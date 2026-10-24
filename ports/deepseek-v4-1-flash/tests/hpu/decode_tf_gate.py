#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode vs prefill, teacher-forced: greedy tokens from the decode path (the per-layer-group HPU graphs, the MXFP4
decode kernels when on) are re-scored through the prefill path with prompt_logprobs.

Neither path is ground truth: against the fp32 CPU oracle both are equally close, and every token where decode leaves
the oracle's argmax is a close call. So this bounds the noise instead of demanding argmax agreement.

PASS: mean |decode logprob - prefill logprob| < 0.05, at most 3% of tokens off the prefill argmax, each within 2 nats.

  python3 decode_tf_gate.py [--max-tokens 200] [--concurrent 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import NO_THINK, post, verdict  # noqa: E402

PROMPTS = [
    "Explain in about 150 words how a hash table handles collisions, with one concrete example.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive? Show your steps.",
    "Describe the water cycle to a ten-year-old in one paragraph.",
    "List four differences between TCP and UDP.",
]


def prompt_ids(text: str) -> list[int]:
    r = post("/tokenize", {"messages": [{"role": "user", "content": text}], "add_generation_prompt": True, **NO_THINK})
    return r["tokens"]


def generate(ids: list[int], max_tokens: int) -> tuple[list[int], list[float]]:
    r = post(
        "/v1/completions",
        {"prompt": ids, "max_tokens": max_tokens, "temperature": 0, "logprobs": 1, "return_tokens_as_token_ids": True},
    )
    lp = r["choices"][0]["logprobs"]
    return [int(t.split(":", 1)[1]) for t in lp["tokens"]], lp["token_logprobs"]


def rescore(ids: list[int], gen: list[int]) -> list[tuple[float, int, float]]:
    r = post("/v1/completions", {"prompt": ids + gen, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
    out = []
    for t, entry in zip(gen, r["choices"][0]["prompt_logprobs"][len(ids) :], strict=True):
        e = entry[str(t)]
        out.append((e["logprob"], e["rank"], max(v["logprob"] for v in entry.values()) - e["logprob"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrent", type=int, default=4)
    args = ap.parse_args()
    prompts = PROMPTS[: max(1, min(args.concurrent, len(PROMPTS)))]
    ids = [prompt_ids(p) for p in prompts]
    gens: list = [None] * len(ids)

    def run(i: int) -> None:
        gens[i] = generate(ids[i], args.max_tokens)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(ids))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = off_top = 0
    sum_d = max_gap = worst = 0.0
    for i, (g, dlp) in enumerate(gens):
        sc = rescore(ids[i], g)
        d = [abs(a - p[0]) for a, p in zip(dlp, sc, strict=True)]
        bad = [(k, p[1], round(p[2], 4)) for k, p in enumerate(sc) if p[1] != 1]
        total += len(g)
        off_top += len(bad)
        sum_d += sum(d)
        worst = max(worst, max(d))
        max_gap = max([max_gap] + [gap for _, _, gap in bad])
        print(
            json.dumps(
                {
                    "prompt": i,
                    "tokens": len(g),
                    "mean_abs_dlp": round(sum(d) / len(d), 5),
                    "not_prefill_argmax": bad[:6],
                }
            )
        )
    mean = sum_d / max(1, total)
    print(
        json.dumps(
            {
                "tokens": total,
                "mean_abs_dlp": round(mean, 4),
                "not_prefill_argmax": off_top,
                "max_gap": round(max_gap, 4),
                "worst_abs_dlp": round(worst, 4),
            }
        )
    )
    ok = mean < 0.05 and off_top <= 0.03 * total and max_gap < 2.0
    detail = f"mean |dlp| {mean:.4f}, {off_top}/{total} off argmax, max gap {max_gap:.3f}"
    return 0 if verdict("DECODE-TF", ok, detail) else 1


if __name__ == "__main__":
    sys.exit(main())
