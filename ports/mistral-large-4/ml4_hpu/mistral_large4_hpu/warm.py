"""Start-up warm-up: the shapes the first users would otherwise compile, sent as requests to the
server's own loopback port before the server reports ready.

Under --enforce-eager with VLLM_SKIP_WARMUP every shape compiles on first use and the recipe cache is local, so
on a fresh container the first prompt of each size pays: a 5K prompt's first token 36 s instead of 4, 20K 69 s instead of
17, 70K +67 s. Shapes are (chunk rows, block-table width) for prompt
chunks and (rows, width) for decode / verify graphs, widths the powers of two from 16 blocks (runner_patch._bt_width);
a prompt that reaches width w compiles every narrower width's 2048-row chunk on the way, so one ladder whose last
prompt reaches the widest table (65537+ tokens at 128-token blocks for 1024 blocks) compiles all the chunk shapes a
longer prompt uses (100K and 128K then cost nothing extra). Each ladder prompt is sent again with 300 tokens appended,
the next turn of a conversation: with prefix caching it hits and runs one short chunk at that width, a shape
the cold ladder never makes (a 110K conversation's turn 3 compiled one: 16 s instead of 5). Each request also decodes,
which captures the 1-stream decode and verify-rows graphs at that width. Concurrent short requests capture
every 1- to 8-stream shape, with and without drafts.

  python3 -m mistral_large4_hpu.warm &      # the manifest starts it in the background, before exec vllm serve

Readiness (as the DeepSeek-V4.1-Flash port's warm-up): a readiness probe passes only once ML4_WARMUP_DONE
(/tmp/ml4-warm.done) exists. It is written when the set finishes, when ML4_WARMUP_MAX_S (counted from the server
first answering /health) runs out, or on any error (fail open: a replica that serves cold beats one that never
serves; the log says which). ML4_WARMUP=0 writes it at once (rollback knob). Token-id prompts (exact lengths, no
tokenizer here), temperature 0, ignore_eos; prompts longer than MAX_MODEL_LEN - 64 are skipped.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import urllib.request

URL = os.environ.get("ML4_WARMUP_URL", "http://127.0.0.1:8000")
DONE = os.environ.get("ML4_WARMUP_DONE", "/tmp/ml4-warm.done")
MAX_S = float(os.environ.get("ML4_WARMUP_MAX_S", "2700"))
MAX_LEN = int(os.environ.get("MAX_MODEL_LEN", "131072"))
TOKENS = 48
# The prompt-graph buckets (graphs.PREFILL_BUCKETS: 64-768 rows), then one prompt per table width up to 1024 blocks.
SHORT = (30, 50, 64, 100, 128, 200, 300, 512, 700, 768)
LADDER = (2000, 5000, 9242, 20000, 34000, 70000)
# Each ladder prompt is then sent again with this many tokens appended: a follow-up turn's prefix-cache hit.
HIT_TAIL = 300
# (streams, prompt tokens, seeded): every batch size 1-8 on short prompts, twice: plain requests (DSpark verify
# rows: 4 per request up to 4 requests, 3 above) and seeded ones, which never draft (spec._plain), for the one-token
# decode shapes the acceptance tracker falls back to. Decode graphs key on the exact batch, so without 3 / 5 / 6 / 7 the
# first 8-user chat paid ~8.5 s per shape mid-answer (8.7 tok/s per user cold, 11.3-11.8 warm). Then 4 users
# on ~9K pastes.
CONCURRENT = tuple((k, 300, seeded) for k in range(1, 9) for seeded in (False, True)) + ((4, 9242, False),)
# Users at once at longer table widths (vLLM pads a decode batch to 1 / 2 / 4 / 6 / 8 requests): after the ladder prompt
# of n tokens, k requests of that prompt + their own HIT_TAIL tokens, so with prefix caching each computes only its
# tail. 2000 / 9242 / 34000: tables 32 / 128 / 512 (chat answers of ~2000 tokens, the pastes), drafting and not (a
# request that does not draft runs one-token rows). A day of real use on captured ~30 such shapes at ~12 s
# each, every one a pause mid-answer; a real-use run on the served config still met 8 x 1 and 6 x 1 at table 128.
# Skipped when the hit turn shows no cached tokens (prefix caching off: k cold prompts each).
MULTI_AT = {n: tuple((k, s) for k in (2, 4, 6, 8) for s in (False, True)) for n in (2000, 9242, 34000)}


def request(
    url: str, n: int, tokens: int, rng: random.Random, ids: list | None = None, seed: int | None = None
) -> float:
    ids = ids if ids is not None else [rng.randrange(1000, 100000) for _ in range(n)]
    body = {"model": request.model, "prompt": ids, "max_tokens": tokens, "ignore_eos": True, "temperature": 0}
    if seed is not None:
        body["seed"] = seed
    t0 = time.time()
    with urllib.request.urlopen(
        urllib.request.Request(
            url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        ),
        timeout=3600,
    ) as r:
        out = json.load(r)
    request.cached = (out["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    if out["usage"]["completion_tokens"] != tokens:
        raise RuntimeError(f"{len(ids)}-token prompt: {out['usage']['completion_tokens']} of {tokens} tokens")
    return time.time() - t0


class Deadline(Exception):
    pass


def log(msg: str) -> None:
    print(f"ml4 warm: {msg}", flush=True)


def wait_up(limit_s: float = 3 * 3600) -> bool:
    """The server answering /health (weights load first: 10-85 min)."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001  (not up yet)
            pass
        time.sleep(15)
    return False


def together(k: int, prompts, tokens: int, seeded: bool) -> None:
    """k requests at once (prompts(i) -> token ids); raises on the first failure."""
    errs = []

    def one(i):
        try:
            request(URL, 0, tokens, random.Random(i), prompts(i), seed=1000 * k + i if seeded else None)
        except Exception as e:  # noqa: BLE001  (raised below)
            errs.append(repr(e))

    th = [threading.Thread(target=one, args=(i,)) for i in range(k)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise RuntimeError(f"{k} at once: {errs[0]}")


def warm(deadline: float) -> None:
    with urllib.request.urlopen(URL + "/v1/models", timeout=60) as r:
        request.model = json.load(r)["data"][0]["id"]
    rng = random.Random(0)
    hits = {}
    for n in SHORT + LADDER:
        if n > MAX_LEN - 64 - HIT_TAIL:
            continue
        if time.time() > deadline:
            raise Deadline
        ids = [rng.randrange(1000, 100000) for _ in range(n)]
        log(f"{n} tokens x 1: {request(URL, n, TOKENS, rng, ids):.1f} s")
        if n in LADDER:
            # the next turn of a conversation: a prefix-cache hit on the prompt above, then a short chunk at this
            # table width (with prefix caching off it is one more cold prompt of the same width)
            log(f"{n} + {HIT_TAIL} tokens (hit) x 1: {request(URL, 0, TOKENS, rng, ids + ids[:HIT_TAIL]):.1f} s")
            if request.cached:
                hits[n] = ids
    for k, n, seeded in CONCURRENT:
        if time.time() > deadline:
            raise Deadline
        t0 = time.time()
        together(
            k, lambda i: [random.Random(1000 * k + i).randrange(1000, 100000) for _ in range(n)], TOKENS * 2, seeded
        )
        log(f"{n} tokens x {k}{' seeded' if seeded else ''}: {time.time() - t0:.1f} s")
    for n, groups in MULTI_AT.items():
        if n not in hits:
            continue
        for k, seeded in groups:
            if time.time() > deadline:
                raise Deadline
            t0 = time.time()
            tails = [[random.Random(7000 * k + i).randrange(1000, 100000) for _ in range(HIT_TAIL)] for i in range(k)]
            together(k, lambda i: hits[n] + tails[i], TOKENS, seeded)
            log(f"{n} + {HIT_TAIL} tokens (hit) x {k}{' seeded' if seeded else ''}: {time.time() - t0:.1f} s")


def main() -> int:
    status = "off"
    if os.environ.get("ML4_WARMUP", "1") != "0":
        if not wait_up():
            status = "server never answered /health"
        else:
            t0 = time.time()
            log(f"server healthy; warming (cap {MAX_S:.0f} s)")
            try:
                warm(t0 + MAX_S)
                status = f"done in {time.time() - t0:.0f} s"
            except Deadline:
                status = f"cap {MAX_S:.0f} s reached: Ready with the remaining shapes cold"
            except Exception as e:  # noqa: BLE001  (fail open)
                status = f"failed after {time.time() - t0:.0f} s ({e!r}): Ready with the remaining shapes cold"
    log(status)
    with open(DONE, "w") as f:
        f.write(status + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
