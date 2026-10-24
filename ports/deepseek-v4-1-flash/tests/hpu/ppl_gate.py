#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Perplexity vs the fp32 CPU oracle: the oracle's exact token ids go through the server teacher-forced
(prompt_logprobs, which runs the full stack, decoder replay off) and the mean NLL is compared.

PASS: no missing token, and |server perplexity - oracle perplexity| <= --tol (default 0.015). Healthy servers land
within about 0.006 of the oracle; a wrong cache read, a broken kernel or wrong weights moves it by far more. The
oracle in data/ is for the -1017 weights; the original V4.1-Flash weights give a different perplexity.

  python3 ppl_gate.py [--oracle data/oracle_1017.json] [--tol 0.015]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gatelib import post, verdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=os.path.join(HERE, "data", "oracle_1017.json"))
    ap.add_argument("--tol", type=float, default=0.015)
    args = ap.parse_args()
    with open(args.oracle) as f:
        o = json.load(f)
    ids, ref = o["ids"], o["nll"]
    t = time.time()
    r = post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    pl = r["choices"][0]["prompt_logprobs"]
    nll, missing = [], 0
    for i, entry in enumerate(pl[1:], start=1):
        e = (entry or {}).get(str(ids[i]))
        if e is None:
            missing += 1
            continue
        nll.append(-e["logprob"])
    n = len(nll)
    ppl = math.exp(sum(nll) / max(1, n))
    ref_ppl = math.exp(sum(ref[: len(pl) - 1]) / max(1, len(pl) - 1))
    diff = sum(abs(a - b) for a, b in zip(nll, ref, strict=False)) / max(1, n) if not missing else float("nan")
    out = {
        "tokens": n,
        "missing": missing,
        "server_ppl": round(ppl, 4),
        "oracle_ppl": round(ref_ppl, 4),
        "mean_abs_token_nll_diff": round(diff, 4),
        "seconds": round(time.time() - t, 1),
    }
    print(json.dumps(out))
    ok = missing == 0 and abs(ppl - ref_ppl) <= args.tol
    return 0 if verdict("PPL", ok, f"{ppl:.4f} vs oracle {ref_ppl:.4f}") else 1


if __name__ == "__main__":
    sys.exit(main())
