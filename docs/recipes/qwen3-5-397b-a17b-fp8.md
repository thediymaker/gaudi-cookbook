# Qwen3.5-397B-A17B-FP8

Qwen's 397B mixture-of-experts hybrid model (17B active, 512 experts, top 10; Gated DeltaNet plus
attention), from the official FP8 checkpoint, text only, at its native 256K context on one full
8-card Gaudi 2 node, with the KV pool pinned so that each of 16 sequences can hold 128K at once,
and prefix caching on through the [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |
| Hardware | 8x Gaudi 2 (TP 8, `--enable-expert-parallel`) |
| Image | vllm-gaudi `releases/v0.30.0` at `058f107` (before the v0.30.0 tag), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (406 GB) |
| Context | 262,144 (native) per request; 16 x 131,072 resident at once with the pool pinned as below |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen3.5-397B-A17B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3.5-397B-A17B-FP8/recipe.yaml) |

## Prefix caching

On the stock release, prefix-cache hits on the Gated DeltaNet layers resume from the wrong state,
so prefix caching is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md), which
stores GDN state checkpoints in the attention pages of the KV blocks the GDN groups own and keeps
the compact GDN slots. See [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md#prefix-caching) and
[Prefix caching on Gated DeltaNet hybrids](../results/gdn-prefix-caching.md) for the design. With
expert parallelism on eight cards the hybrid KV block is 640 tokens, and prefill chunks end on
block boundaries.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export VLLM_SKIP_WARMUP=true
export VLLM_HPU_MOE_GATHER=1
export VLLM_HPU_MOE_GATHER_RATIO=0.2
export VLLM_HPU_MOE_GATHER_VERIFY=0
export VLLM_LONG_PREFILL_SHARE=0.5
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=32768
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 \
    gdn-prefix-cache vllm-long-prefill-share || exit 1

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --num-gpu-blocks-override 3392 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- `--language-model-only` skips the vision tower and its profiling.
- `--num-gpu-blocks-override 3392` pins the KV pool at 3,392 blocks of 640 tokens: 16 sequences of
  131,072 tokens resident at once. The full pool leaves too little HBM under 256K prompts (see
  "What not to do"). The value assumes this layout (prefix caching on with `gdn-prefix-cache`); with
  a different layout, size it again from the start-up line `Maximum concurrency for 262,144 tokens
  per request`.
- `--enable-prefix-caching` is correct here only with `gdn-prefix-cache`; `VLLM_COMPACT_GDN=1`
  keeps the GDN state in compact per-request slots, and the fix refuses to start without it.
- On v0.30 the gathered-expert FP8 MoE is upstream, and the gate is
  `tokens x top_k <= local experts x VLLM_HPU_MOE_GATHER_RATIO`. With 64 local experts per card and
  top 10, 0.2 keeps the gather to single-token decode steps.
- The RoPE-on-CPU fix is upstream in this image, so `rope-cpu-cache` is not needed.
- `VLLM_LONG_PREFILL_SHARE=0.5` with `vllm-long-prefill-share`: once a prompt of 32K or more is
  prefilling and someone else is decoding, long-prompt chunks get at most half the wall time.

### Buckets

No bucket file and no bucket variables: compile mode with the image's default bucketing.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.30.0`, which also applies the release's other generic fixes (see [Using
a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | v0.30 still never fills the logits-processor batch update |
| [`prompt-order`](../fixes/prompt-order.md) | with prefix caching, a request that hits a block another request writes in the same step read it before it was written |
| [`async-inputs`](../fixes/async-inputs.md) | under async scheduling a decode request that was not in the previous step reads a stale input id, so greedy output repeats a token or derails (`async-discard-counter` is not needed on v0.30) |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request; the start-up log says whether the pad was moved |
| [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) (opt-in) | prefix-cache hits on the GDN layers resume from a wrong state on the stock release; it needs `hybrid-decode-pad` first |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the int64 concat and cast of the logprob ids is still in v0.30 |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the flat-hidden-state indexing bug is unchanged in v0.30 |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`moe-explicit-overload`](../fixes/moe-explicit-overload.md) | the per-channel FP8 MoE op is called through its overload packet; failed overload matches format a tensor repr, which forces a blocking host sync per MoE layer on every eager prefill |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a 2 GiB allocation failure on one rank during a 260K prefill left the engine serving `!` for every later request; the fix stops the engine instead |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) (opt-in) | without it every other stream drops to about 1 token/s while a 250K prompt prefills |

## Verified

On the image above, eight Gaudi 2 cards, the launch above as written (prefix caching on with
`gdn-prefix-cache`, the 3,392-block pool of 640-token blocks):

- Greedy parity, 16 prompts of 5K to 31K tokens, the same request served from the cache and cold:
  12 requests hit the cache and all 12 matched the cold run token for token. Two cold runs: 16/16
  identical. The one other split was a request that got no cache hit (0 cached tokens), at a
  top-2 gap of 0.875: batch composition, not the cache.
- Cross-request isolation, 12 rounds with an unrelated request between fill and hit: 0 leaks.
- Needles at 128K, 200K and 250K, three depths each: 9/9 cold, 9/9 from the cache.
- 16 contexts of about 129K tokens resident at once (1,024 decoded each, each with its own
  `cache_salt`): 16/16 correct, 0 errors, 0 preemptions, KV peak 96.9%.
- GSM8K first 50 (thinking off, greedy): 50/50.
- Tools (auto with thinking on and off, required, named, streaming, multi-turn with a result),
  `json_schema` strict 9/9 (greedy, sampled at 0.7 and with thinking on), reasoning separation
  streamed and not: all pass.
- Request options: `top_logprobs` 20, `prompt_logprobs`, a seeded request (same seed, same text),
  `logit_bias` and penalties all take effect, and the engine stays healthy. `bad_words` blocks the
  listed token sequences: asked to reply "hello world" with `hello` banned, the model wrote `h` +
  `elloworld`, which `bad_words` does not cover by design.
- Minimum free HBM on any card across all of the above: 5.48 GiB, 0 allocation failures.

From the earlier form of the same layout (same flags), not repeated on this run: 16 cold 128K
prompts fired at once 16/16 with 0 preemptions, and greedy parity against the v0.26.0 build of
this model, 16 prompts x 192 tokens, 16/16 identical with identical top-5 logprobs.

## Performance

Single measurements on eight cards, the image above, the same harness for both columns, both with
prefix caching on with the earlier form of the layout; TTFTs are for requests that did not hit
the cache. The v0.26.0 column is the earlier build of this model (same layout, full KV pool, no
long-prefill share), for reference.

| Test | This recipe | v0.26.0 build |
| --- | --- | --- |
| Tokens/s per stream at 1 / 4 / 8 / 16 in flight | 27.4 / 18.8 / 18.4 / 17.8 | 28.1 / 15.9 / 15.3 / 15.0 |
| Tokens/s per replica at 16 in flight | 255 | 217 |
| Cold TTFT at 8K / 32K / 128K | 0.94 / 3.89 / 20.2 s | 0.94 / 3.86 / 19.8 s |
| 16 x 32K concurrent, wall time | 100 s | 121 to 137 s |
| Other streams while a 250K prompt prefills | 9.6 tokens/s | 1.1 tokens/s |

Re-run with the launch above as written: 8 streams decoding at 16.75 tokens/s each dropped to 9.46
while a 250K prompt prefilled (its own TTFT 118.5 s), and short requests sent meanwhile got their
first token in 0.29 to 1.33 s.

A single stream is 2.6% slower on this image (repeated three times on both); the cause is not
found. The 250K prompt's own TTFT doubles with the long-prefill share (about 60 s to 120 s).

### Alternative: 32 sequences

For throughput over a per-sequence guarantee, `--max-num-seqs 32` passed the same gates (prefix
caching on, a 3,200-block pool of 640-token blocks): 436 tokens/s per replica at 32 in flight, 16.2
per stream, 6.3 GiB HBM free at the minimum. The 32 sequences then share the pool (about 63K each
when all are full).

## Known issues

- This is the `releases/v0.30.0` branch before the tag. Upstream prefix caching for Qwen3.5 uses
  the non-compact layout (state blocks next to the KV pool) and is not in this commit; the recipe
  uses `gdn-prefix-cache` instead.
- A cold load takes about 40 minutes (406 GB).

## What not to do

- Do not turn on prefix caching on this release without `gdn-prefix-cache`: hits then resume the
  GDN layers from a wrong state.
- Do not run with the full KV pool at 256K. Under 250K prompts the cards then had 0.07 to 0.25 GiB
  free (prefix caching on), and on the v0.26.0 build one rank failed a 2 GiB attention allocation
  during a 200K prefill.
- Do not raise `--max-num-batched-tokens` to 8,192: a 4,096-token chunk against a 256K context
  already needs a 2 GiB attention buffer per card, and 8,192 doubles it.
- Do not enable MTP speculative decoding on this recipe. It needs fixes that are not in this
  cookbook, and in tests (with prefix caching on) GSM8K kept only 48/50 answers and greedy outputs
  forked at near-ties.
- Do not use lazy mode for this model.
