#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Startup warm-up for a DeepSeek-V4.1-Flash server: compile every prefill and decode shape before the replica takes
traffic, so no user request pays a one-time graph compile.

Why a client-side warm-up. The recipe serves with --enforce-eager (prefill runs as lazy-mode graphs cut at every
layer; decode replays the port's per-layer-group HPU graphs, DSV41_LAYER_GRAPHS), and vllm-gaudi's own warm-up warms
nothing under --enforce-eager. So the first request that reaches a new shape compiles it inline, which takes from a
few seconds to minutes. This script sends that set of requests to the local server first.

The shapes (vllm-gaudi v0.30.0 exponential buckets, the port's runner hook, 4096-token chunks, block size 128):
  prefill  query bucket S (11 buckets, 128 .. 4096 tokens) x block-table width W (64, 128, ... blocks, doubling up to
           the context length: 5 widths at 128K): 55 cells
  decode   batch bucket (1 2 4 6 8 with --max-num-seqs 8) x prefix (none, or half of each width): 25 cells
The port's runner hook skips vllm-gaudi's prompt attention mask, so the context-block bucket does not change the
graphs and (S, W) is the whole set.

How: one unique prompt of nearly the full context first (its blocks become a cached prefix); every later cell is a
block-aligned prefix of it (a prefix-cache hit, free) plus a fresh random tail, so the server computes only that
cell's chunk. Token ids are random per run, so nothing matches user traffic. Idempotent: a re-run compiles nothing
new. Order: the base prompt, then prefill cells widest first and, within a width, small query buckets first, then
decode.

Readiness. DONE (DSV41_WARMUP_DONE, default /tmp/dsv41-warmup.done) is written when the ladder finishes, when
DSV41_WARMUP_MAX_S (counted from the server first answering /health) runs out, or on any error: a replica that serves
some shapes cold is better than one that never serves. Point the readiness probe at that file. DSV41_WARMUP=0 writes
it at once.

  python3 warmup.py              # the startup warm-up (the recipe's setup starts it in the background)
  python3 warmup.py --dry-run    # print the cells for a context length and batch size, send nothing

Env: DSV41_WARMUP_URL (default http://localhost:8000), DSV41_WARMUP_MAX_SEQS (the serve --max-num-seqs, default 8),
DSV41_WARMUP_MAX_S (default 2400), DSV41_WARMUP_SERVER_WAIT_S (default 7200).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request

URL = os.environ.get("DSV41_WARMUP_URL", "http://localhost:8000")
DONE = os.environ.get("DSV41_WARMUP_DONE", "/tmp/dsv41-warmup.done")
MAX_S = float(os.environ.get("DSV41_WARMUP_MAX_S", "2400"))
MAX_SEQS = int(os.environ.get("DSV41_WARMUP_MAX_SEQS", "8"))
SERVER_WAIT_S = float(os.environ.get("DSV41_WARMUP_SERVER_WAIT_S", "7200"))
CHUNK, BLOCK, BT_MIN = 4096, 128, 64  # --max-num-batched-tokens, --block-size, the runner's narrowest table
LO, HI = 1000, 100000  # random token ids are drawn from here
T0 = time.time()


def log(msg: str) -> None:
    print(f"[dsv41-warmup] +{time.time() - T0:6.0f}s {msg}", flush=True)


def exp_buckets(bmin: int, bstep: int, bmax: int, n: int) -> list[int]:
    """vllm-gaudi's exponential bucket range (extension/bucketing/exponential.py, warmup_range_with_limit)."""
    out = set()
    if bmin in (0, 1):
        out.add(bmin)
        bmin = bstep
    for i in range(n):
        p = bmin * math.pow(bmax / bmin, i / (n - 1)) if n > 1 else bmax
        out.add(math.ceil(p / bstep) * bstep)
    s = sorted(out)
    s[-1] = min(s[-1], bmax)
    return s


QB = exp_buckets(BLOCK, BLOCK, CHUNK, math.ceil(math.log2(CHUNK)))


def width(npos: int, max_len: int) -> int:
    """Block-table width the runner hook uses for a highest position npos: BT_MIN doubled until it covers npos."""
    need = -(-npos // BLOCK)
    w = BT_MIN
    while w < need:
        w *= 2
    return min(w, -(-max_len // BLOCK))


def prefill_cells(max_len: int) -> list[tuple[int, int, tuple[int, int]]]:
    """(S, prefix blocks k, (S, W)): the chunk is S tokens after k * BLOCK cached tokens."""
    cells: dict[tuple[int, int], int] = {}
    for s in QB:
        for k in range((max_len - 2 - s) // BLOCK + 1):
            cells.setdefault((s, width(k * BLOCK + s, max_len)), k)
    out = sorted(cells.items(), key=lambda e: (-e[0][1], e[0][0], e[1]))
    return [(key[0], k, key) for key, k in out]


def decode_cells(max_len: int, max_seqs: int) -> list[tuple[int, int]]:
    """(batch, prefix blocks per request): half of every block-table width, and an empty prefix."""
    batches = exp_buckets(1, 2, max_seqs, math.ceil(math.log2(max_seqs)) + 1)
    max_blocks = -(-max_len // BLOCK)
    ks = {0} | {(128 << i) // 2 for i in range(max(1, (max_blocks // 128).bit_length()))}
    return [(b, k) for b in batches for k in sorted(ks, reverse=True)]


def _get(path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def healthy() -> bool:
    try:
        with urllib.request.urlopen(URL + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def wait_up(limit_s: float) -> bool:
    t = time.time()
    while time.time() - t < limit_s:
        if healthy():
            return True
        time.sleep(10)
    return False


class Deadline(Exception):
    pass


class Server:
    def __init__(self) -> None:
        m = _get("/v1/models")["data"][0]
        self.model, self.max_len = m["id"], int(m.get("max_model_len") or 131072)

    def complete(self, ids: list[int], max_tokens: int = 1, sample: bool = False) -> float:
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 1.0 if sample else 0.0,
        }
        if sample:
            body["top_p"] = 0.95
        req = urllib.request.Request(
            URL + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        t = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()
        return time.time() - t


def run(srv: Server, deadline: float | None = None, decode_tokens: int = 8) -> dict:
    """Send the ladder. Raises Deadline once time.time() passes deadline (checked between cells)."""
    rng = random.Random(time.time_ns())
    pcells, dcells = prefill_cells(srv.max_len), decode_cells(srv.max_len, MAX_SEQS)
    log(json.dumps({"model": srv.model, "max_len": srv.max_len, "prefill_cells": len(pcells), "decode": len(dcells)}))
    t0, errors, rows = time.time(), 0, []

    def check() -> None:
        if deadline is not None and time.time() > deadline:
            raise Deadline()

    base_len = (srv.max_len - BLOCK) // BLOCK * BLOCK
    base = [rng.randrange(LO, HI) for _ in range(base_len)]
    rows.append(("base", srv.complete(base)))
    log(json.dumps({"cell": "base", "tokens": base_len, "s": round(rows[-1][1], 2)}))
    for s, k, key in pcells:
        check()
        try:
            dt = srv.complete(base[: k * BLOCK] + [rng.randrange(LO, HI) for _ in range(s)])
        except Exception as e:
            errors += 1
            log(json.dumps({"cell": ["prefill", *key], "error": str(e)}))
            continue
        rows.append((["prefill", *key], dt))
        log(json.dumps({"cell": ["prefill", *key], "prefix_tokens": k * BLOCK, "s": round(dt, 2)}))
    for b, k in dcells:
        check()
        errs: list[str] = []

        def one(i: int, k: int = k, errs: list[str] = errs) -> None:
            try:
                tail = [rng.randrange(LO, HI) for _ in range(64)]
                srv.complete(base[: k * BLOCK] + tail, decode_tokens, sample=bool(i % 2))
            except Exception as e:
                errs.append(str(e))

        threads = [threading.Thread(target=one, args=(i,)) for i in range(b)]
        t = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        errors += len(errs)
        rows.append((["decode", b, k * BLOCK], time.time() - t))
        log(json.dumps({"cell": ["decode", b, k * BLOCK], "s": round(rows[-1][1], 2), "errors": len(errs)}))
    slow = sorted(rows, key=lambda r: -r[1])[:5]
    return {
        "cells": len(rows),
        "errors": errors,
        "total_s": round(time.time() - t0, 1),
        "slowest": [[c, round(s, 2)] for c, s in slow],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the cells and exit")
    ap.add_argument("--max-len", type=int, default=131072, help="--dry-run only: the serve --max-model-len")
    args = ap.parse_args()
    if args.dry_run:
        pcells, dcells = prefill_cells(args.max_len), decode_cells(args.max_len, MAX_SEQS)
        print(json.dumps({"query_buckets": QB, "prefill_cells": len(pcells), "decode_cells": len(dcells)}))
        for _, k, key in pcells:
            print(json.dumps({"prefill": key, "prefix_tokens": k * BLOCK}))
        for b, k in dcells:
            print(json.dumps({"decode_batch": b, "prefix_tokens": k * BLOCK}))
        return 0

    result: dict = {"status": "skipped"}
    try:
        if os.environ.get("DSV41_WARMUP", "1") != "1":
            log("DSV41_WARMUP != 1: no warm-up")
            return 0
        if not wait_up(SERVER_WAIT_S):
            result = {"status": "server never healthy"}
            log("server never became healthy")
            return 1
        t_up = time.time()
        log(f"server healthy after {t_up - T0:.0f}s; warming (cap {MAX_S:.0f}s)")
        try:
            result = {"status": "done", **run(Server(), deadline=t_up + MAX_S)}
        except Deadline:
            result = {"status": "cap"}
            log(f"cap {MAX_S:.0f}s reached: ready with the remaining shapes cold")
        result["warmup_s"] = round(time.time() - t_up, 1)
        log(f"finished: {json.dumps(result)}")
        return 0
    except Exception as e:  # fail open
        result = {"status": f"error: {e}"}
        log(f"error, failing open: {e}")
        return 1
    finally:
        try:
            with open(DONE, "w") as f:
                json.dump({**result, "elapsed_s": round(time.time() - T0, 1)}, f)
        except OSError as e:
            log(f"cannot write {DONE}: {e}")


if __name__ == "__main__":
    sys.exit(main())
