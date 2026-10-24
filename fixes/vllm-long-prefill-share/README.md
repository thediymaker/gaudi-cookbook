# vllm-long-prefill-share

Let long prompts take at most a set share of wall time while other requests decode, so one very
long prefill does not stall every other stream. Opt-in scheduler feature, off until configured.

| | |
| --- | --- |
| Affects | long-context deployments (hundreds of thousands of tokens) with chunked prefill and concurrent users |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/v1/core/sched/scheduler.py` |
| Upstream | not filed yet (scheduler feature proposal) |
| Test | `test_long_prefill_share.py` (stdlib; the source tests need a release's sources) |

## Problem

With chunked prefill, every scheduler step that carries a chunk of a long prompt runs that
chunk's attention over all of the prompt's context so far. Deep into a long prompt one chunk can
take seconds, and every other running request gets only one decode token per step.
`--long-prefill-token-threshold` makes the chunk smaller, which helps a new request get its first
token, but it cannot help decode, because a long chunk is still in every step.

## Symptom

While one prompt of several hundred thousand tokens prefills, other users' streams slow to well
under one token per second for minutes, and short new requests wait a long time for their first
token. The same requests are fast as soon as the long prefill finishes.

## Who is affected

Servers with a large `--max-model-len` that receive very long prompts next to interactive
traffic. Short-context deployments never reach the threshold and are not affected.

## The fix

vLLM already defers in-progress prefill chunks on some steps while decodes run (`defer_prefills`,
used for DP prefill balancing). The fix adds a second reason to take that branch: after a step that
carried a long chunk and took `T` seconds, the next `T * (1 - share) / share` seconds are
decode-only for long prompts, as long as another request is decoding or a short request is waiting
to start. New and short requests are never deferred, and with nobody else to serve nothing is
deferred. Step times are measured where steps complete, which is correct under async scheduling.

The feature is off unless `VLLM_LONG_PREFILL_SHARE` is set, so applying the fix alone changes
nothing:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_LONG_PREFILL_SHARE` | fraction of wall time long prefills may take while others decode, between 0 and 1 | unset (off) |
| `VLLM_LONG_PREFILL_MIN_TOKENS` | prompt length that counts as long | `65536` |
| `VLLM_LONG_PREFILL_SHARE_ASYNC` | `1`: also defer long chunks while one is still in flight | `0` |

With async scheduling the next step is scheduled before the previous one completes, so by default
long chunks run in back-to-back pairs and the decode share is `(1 - s) / (1 + s)` rather than
`1 - s`. Set `VLLM_LONG_PREFILL_SHARE_ASYNC=1` to make the share exact. Pair the feature with
`--long-prefill-token-threshold`, so that a single long step stays short and decode is smooth
rather than stalling for seconds.

## Measured trade-off

The time given to the other streams is taken from the long prompts; the total is fixed. Both
measurements below are single runs.

Qwen3.5-397B-A17B-FP8 on eight cards, `releases/v0.30.0`, the
[recipe](../../docs/recipes/qwen3-5-397b-a17b-fp8.md)'s settings (16 sequences, prefix caching on,
4,096-token chunks, `VLLM_LONG_PREFILL_SHARE=0.5`, `VLLM_LONG_PREFILL_SHARE_ASYNC=1`,
`VLLM_LONG_PREFILL_MIN_TOKENS=32768`), same pod, share on vs off:

| | Share off | Share 0.5 |
| --- | --- | --- |
| 14 streams decoding while one 250K prompt prefills | 1.0 tokens/s per stream | 9.7 tokens/s per stream |
| the 250K prompt's own time to first token | about 60 s | about 120 s |
| 16 synthetic agent sessions, wall time | 284 s | 269 s |
| their later-turn TTFT p50 / p90 | 3.6 / 7.2 s | 4.1 / 14.1 s |

GLM-5.3 (an MXFP4 MoE model port) on eight cards, on an older vllm-gaudi base with the port's
patches (the fix applied unchanged), 35 minutes of two workers sending short chats while two workers send 44K
and 91K prompts back to back, so a long prompt is always prefilling (the worst case for both
sides). Chained runs on one pod, `VLLM_LONG_PREFILL_SHARE_ASYNC=1` in every run with a share:

| Setting | Chat decode, median | Chat TTFT p50 / p95 | Long TTFT p50 / p95 | Long prompts done |
| --- | --- | --- | --- | --- |
| no share | 1.2 tokens/s | 105 s (p50) | 48 / 103 s | 71 |
| share 0.25, 2,048-token chunks, min 64K | 23.5 tokens/s | 1.2 / 4.0 s | 383 / 624 s | 12 |
| share 0.5, 4,096-token chunks, min 32K | 14.9 tokens/s | 0.6 / 14.3 s | 148 / 252 s | 33 |
| share 0.5, 8,192-token chunks, min 32K | 12.8 tokens/s | 0.8 / 7.3 s | 110 / 324 s | 33 |

0.25 protects the chats best, but the long prompts then queue for minutes. 0.5 with 4,096-token
chunks (`--long-prefill-token-threshold 4096` or `--max-num-batched-tokens 4096`) keeps chats at about
a third of their idle speed while long prompts finish at half the rate; 8,192-token chunks were
slower and burstier for the chats. All runs had 0 errors and 0 preemptions.

## Verify

```bash
python3 fixes/vllm-long-prefill-share/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end, with `VLLM_LONG_PREFILL_SHARE=0.5`: start a short streaming chat, then send a prompt
well above `VLLM_LONG_PREFILL_MIN_TOKENS`. The chat's tokens per second should drop by about the
configured share instead of collapsing, and the long prompt still completes, more slowly.
