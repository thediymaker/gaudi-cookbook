# Prefix caching on Gated DeltaNet hybrids

The Qwen3.5, Qwen3.6 and Qwen3.8 families mix Gated DeltaNet (GDN) linear-attention layers with full attention.
Agent traffic re-sends the whole conversation every turn, so prefix caching is the largest lever these models
have, but stock vllm-gaudi cannot cache them: it keeps GDN state in compact per-request slots and only logs
"Compact GDN mode does not support prefix caching". This page describes a layout that stores the GDN state
checkpoints inside the KV pages of the cache, so prefix caching works with the full KV pool.

!!! summary "At a glance"
    - **Checkpoints:** measured on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
      [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8), both Apache 2.0.
    - **Hardware:** 1 Gaudi 2 per replica for the 27B models, 256K context; vllm-gaudi v0.26.0, Gaudi software
      1.24.1. Ported to releases/v0.30.0 for a TP8 397B MoE.
    - **Status:** serving production traffic on several Qwen3.5/3.6/3.8 models (27B dense to 397B MoE) and
      Qwen3-Coder-Next.
    - **Headline:** agent replay wall 1214 to 281 s, later-turn TTFT 19.8 to 2.6 s (same-pod A/B, caching off vs
      on). Last turn of a 229K session 200 to 21.5 s (same-pod A/B, see the note on conditions). KV pool 288,446
      tokens vs 150,775 for per-block state tensors (same flags).

## Why hits were wrong

With `--enable-prefix-caching`, vLLM puts GDN layers in Mamba "align" mode. Every KV group (one attention group,
three GDN groups on the 27B models) uses one block size, chosen so an attention page can hold a whole GDN state:
3.06 MiB per layer (48 heads x 128 x 128 fp32 SSM state plus the conv window), which makes the block 896 tokens.
Prefill chunks end on block boundaries, and a full block must keep the recurrent state after its last token, so a
hit at k x 896 tokens can resume from block k-1.

vllm-gaudi v0.26.0 breaks that contract:

1. **Compact GDN is the default.** One state slot per request, never written into a KV block, so a hit resumes
   from whatever the slot's previous owner left.
2. **With compact off, nothing carries state forward.** The GDN layer loads and stores the block of the last
   computed token, so a multi-block first chunk writes its state into the null block.
3. **The block size flips to 128 after startup.** A config hook resets the hybrid block size after the KV cache
   and scheduler were built with 896, so chunks split on 128-token multiples.
4. **The prefill caching branch is Mamba2's** and asserts on a merged prefill batch.

Before the fix, greedy text on hits often still matched (GDN forgives a stale state surprisingly often), but
logprobs drifted and two hits of the same prompt disagreed. Cold against cold was bit-identical.

## The in-page layout

The obvious fix (compact off, one state tensor per KV block, as upstream now does) needs a second allocation next
to the attention KV. On Qwen3.8-27B at `--gpu-memory-utilization 0.60` that left 150,775 tokens, too few for
`--max-model-len 262144`. The in-page layout avoids the second allocation:

- **GDN layers keep the compact slots** the engine already allocates without caching (4 requests x 3 GDN groups + 2 =
  14 slots, 0.7 GiB).
- **Checkpoints live in attention pages.** A KV block belongs to one cache group at a time, so the attention rows
  of a block owned by a GDN group are unused, and the page is already padded to fit one GDN state. Upstream vLLM
  shares one buffer for this; vllm-gaudi cannot, because torch.compile rejects writes through views that change
  dtype. The patch moves the bytes with explicit copies outside the compiled graphs.
- **Save and restore.** After a step that ends on a block boundary, the slot is copied into that block. Before a
  step that starts from a hit, or from a request resumed after preemption, the block is copied back. Copies go
  through int32 views and are bit-exact, NaN encodings included.
- **Preemption hands the slot back**, so preemption cannot run the slots dry.
- **Fail closed.** A partial-block copy or a resume off a block boundary stops the engine rather than serving a
  wrong state. It never happened in testing.

Two vllm-gaudi defects had to be fixed along the way:

1. **New attention blocks are not zeroed.** vLLM lists them in `new_block_ids_to_zero` for Mamba-style models;
   vllm-gaudi ignores the field. A mask does not cancel a NaN, so a block that had held a checkpoint made the
   next request's first decode NaN, and the NaN then stuck in its slot (NaN times zero is NaN).
2. **Decode padding writes into a live block.** The pad block id counts KV blocks, but hybrid block tables count
   128-token kernel blocks, so the pad lands inside a real block and overwrites one token of K/V. This affects
   hybrids with caching off too: [hybrid-decode-pad](../fixes/hybrid-decode-pad.md).

## Correctness

Qwen3.8-27B, util 0.60, 4 sequences, 1024-token batches, 262,144 context:

| Gate | Result |
| --- | --- |
| Greedy parity, 32 prompts of 5K to 31K tokens (single, multi-turn agent, extension, diverging prefix) | hit equals cold 32/32, all 32 logprobs bit-identical |
| Logprob drift, unrelated request between fill and hit | 0.0 |
| Cross-request isolation, 24 rounds | 0 leaks |
| Needles at 65K, 130K, 217K and 260K | 12/12 cold, 12/12 hit, hit equals cold |
| Preemption mid-decode (229K + 49K) | resumed from cache; 0 argmax mismatches against cold re-runs at 23 checkpoints |
| CPU simulator: real vLLM KV manager, patched scheduler, 12 seeds | 0 failures in 10,891 state checks with 427 preemptions; mutants without saves, restores or slot hand-back all fail |

On Qwen3.5-27B-FP8 at 1792-token batches, 5 of 32 diverging-prefix hits were not bit-identical, with the same
values as the per-block layout; at 1024-token batches all 32 were.

## Measured results

Synthetic agent sessions: shared system prompt and tools, about 3K new tokens and 200 output tokens per turn.

| Model, test | Caching off / per-block | In-page | Measured as |
| --- | --- | --- | --- |
| Qwen3.8-27B, 4 sessions 8K to 64K: wall | 1214 s | 281 s | same-pod A/B, caching off |
| later-turn TTFT mean / p90 | 19.8 / 31.6 s | 2.6 / 4.2 s | same run |
| decode tok/s per stream | 7.5 | 18.0 | same run |
| Qwen3.8-27B, 1 session 8K to 229K (16K new per turn): last-turn TTFT | 200 s | 21.5 s | same-pod A/B, caching off; the off arm ran without the untimed warm-up session |
| Qwen3.8-27B, needle at 260K to 262K: TTFT | 247 to 249 s cold | 1.0 to 1.7 s hit | same pod, cold then hit |
| Qwen3.8-27B, KV pool at util 0.60 | 150,775 tokens (per-block) | 288,446 tokens | engine startup log, same flags (caching off: 291,369) |
| Qwen3.5-27B-FP8, 8 sessions 8K to 64K: wall | 1886 s (per-block) | 414 s | same-node A/B, same time |
| later-turn TTFT mean | 48.9 s (per-block) | 2.65 s | same run |

The Qwen3.5 row shows why the pool matters. Eight sessions to 64K need about 512K tokens of live context: the
in-page pool (623,359 tokens) holds it; the per-block pool (332,282) does not, so cached prefixes are evicted and
the hit rate collapses. With 4 sessions, where both fit, the two layouts tie (wall 254 s in-page, 251 s per-block). Cold TTFT and
single-stream decode are unchanged.

## Limits

- Prefill chunks end on block boundaries (896 of a 1024-token budget), so cold 260K TTFT is 247 s against 243 s
  on the caching-off configuration.
- Prompts shorter than one block never hit. On a MoE with 1152-token blocks and prompts averaging 427 tokens,
  the hit rate was zero.
- Live context per replica must fit the pool; beyond it, requests are preempted and resume from cache.
- Decode at batch 2 is not bitwise equal to batch 1 on long contexts, with or without caching, so whole-output
  comparisons need care.
- Caching only pays if the router sends a session back to the same replica. With 4 sequences per replica,
  prefix-affinity routing queued a burst behind busy replicas while others idled; check per-replica capacity first.

## Status upstream and in this cookbook

The in-page layout is in the fixes pack as the opt-in fix [gdn-prefix-cache](../fixes/gdn-prefix-cache.md),
for both v0.26.0 and `releases/v0.30.0`, and the cookbook's GDN recipes now run with prefix caching on:
[Qwen3.6-27B-FP8](../recipes/qwen3-6-27b-fp8.md), [Qwen3.6-35B-A3B-FP8](../recipes/qwen3-6-35b-a3b-fp8.md) and
[Qwen3.5-397B-A17B-FP8](../recipes/qwen3-5-397b-a17b-fp8.md). Each recipe page lists the gates it passed with
the fix. The two runner defects found along the way are fixes of their own or part of this one: decode padding
is [hybrid-decode-pad](../fixes/hybrid-decode-pad.md) (in both profiles, and required first), and new blocks and
slots are zeroed by gdn-prefix-cache.

Upstream vllm-gaudi took a different path for the same problem: prefix caching with compact GDN off, one state
tensor per KV block next to the attention pool. It shipped in the v0.30.0 tag
([vllm-gaudi#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827),
[#1842](https://github.com/vllm-project/vllm-gaudi/pull/1842)) and on main
([#1831](https://github.com/vllm-project/vllm-gaudi/pull/1831), [#1853](https://github.com/vllm-project/vllm-gaudi/pull/1853),
with [#1828](https://github.com/vllm-project/vllm-gaudi/pull/1828) for the async scheduler's chunk split,
[#1890](https://github.com/vllm-project/vllm-gaudi/pull/1890) for resumes off the chunk grid and
[#1895](https://github.com/vllm-project/vllm-gaudi/pull/1895) for the multiply-by-mask NaN in prefill). It is not
in `releases/v0.30.0` at `058f107`, the branch the cookbook's v0.30.0 recipes use. On an image that has it,
upstream's path needs no runtime patch at about half the KV pool on these models (the 150,775 against 288,446
tokens above); the fix's anchors are written against the two cookbook releases and refuse other sources. The
in-page layout itself is not upstream; it is a design discussion, since it keeps compact slots and roughly
doubles the usable pool on large hybrids.

## Availability

The code is the fix [gdn-prefix-cache](../fixes/gdn-prefix-cache.md): one stdlib-only `apply.py` that patches the
HPU model runner, vLLM's scheduler and vllm-gaudi's async scheduler, with a CPU test that drives the runner helpers
on a toy recurrence. Apply it with `python3 fixes/apply_all.py --profile <release> gdn-prefix-cache` and serve
with `--enable-prefix-caching` and `VLLM_COMPACT_GDN=1`. The measurements on this page were taken with
an earlier form of the same layout on v0.26.0; the recipe pages carry the runs made with the fix as published.
