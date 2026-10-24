# Qwen3.5-397B-A17B-FP8

Qwen's 397B mixture-of-experts hybrid model (17B active, 512 experts, top 10; Gated DeltaNet plus
attention), from the official FP8 checkpoint, text only, at its native 256K context on one full
8-card Gaudi 2 node, with the KV pool pinned so that each of 16 sequences can hold 128K at once.
Prefix caching is off on this release (see [Prefix caching](#prefix-caching)).

**Status: not re-run as written.** Every check on this page ran on an earlier form of the recipe
with prefix caching on. Read [Prefix caching](#prefix-caching) before using it.

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

Prefix caching is off: on this release, prefix-cache hits on Gated DeltaNet layers resume from the
wrong state. Upstream vllm-gaudi main has merged prefix caching for the non-compact layout; use a
release that includes it before turning prefix caching on.

The checks and measurements below were taken on an earlier form of this recipe with prefix caching
on, a GDN prefix-cache patch that is not in this cookbook, and a KV pool pinned at 3,392 blocks of
640 tokens. Without that patch the KV layout differs, so:

- the old block count no longer means 16 x 128K, and the pool has to be sized again (below);
- the HBM headroom (4.95 GiB at the minimum) has to be measured again;
- cache-hit results (hit-against-cold parity, needles and prefix-hit TTFTs served from the cache)
  do not apply and are left out.

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
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 vllm-long-prefill-share || exit 1

vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --num-gpu-blocks-override <N> \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- `--language-model-only` skips the vision tower and its profiling.
- `--num-gpu-blocks-override <N>` pins the KV pool; the full pool leaves too little HBM under 256K
  prompts (see "What not to do"). Size it on your image: start once, read the start-up line
  `KV cache size: ... tokens, Maximum concurrency for 262,144 tokens per request: ...x`, and choose
  N so that it reports about 8.1x, which is 16 sequences of 128K plus a small margin (vLLM counts
  every KV-cache group in that figure). The earlier value, 3,392, was computed for 640-token blocks
  that also held GDN states with prefix caching on, and does not carry over.
- `--no-enable-prefix-caching`: see [Prefix caching](#prefix-caching). `VLLM_COMPACT_GDN=1` is the
  stock default without it (GDN state in compact per-request slots); the recipe repeats it.
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
| [`async-inputs`](../fixes/async-inputs.md) | under async scheduling a decode request that was not in the previous step reads a stale input id, so greedy output repeats a token or derails (`async-discard-counter` is not needed on v0.30) |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request; the start-up log says whether the pad was moved |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the int64 concat and cast of the logprob ids is still in v0.30 |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the flat-hidden-state indexing bug is unchanged in v0.30 |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`moe-explicit-overload`](../fixes/moe-explicit-overload.md) | the per-channel FP8 MoE op is called through its overload packet; failed overload matches format a tensor repr, which forces a blocking host sync per MoE layer on every eager prefill |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a 2 GiB allocation failure on one rank during a 260K prefill left the engine serving `!` for every later request; the fix stops the engine instead |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) (opt-in) | without it every other stream drops to about 1 token/s while a 250K prompt prefills |

## Verified

On the image above, eight Gaudi 2 cards, with prefix caching on and the 3,392-block pool described
under [Prefix caching](#prefix-caching); the other flags were the same:

- 16 contexts of about 130K tokens resident at once (prompts of 129,575 to 129,670 tokens, 1,024
  decoded each, each with its own `cache_salt`): 16/16 correct, 0 preemptions, KV peak 98.7%.
- 16 cold 128K prompts fired at once: 16/16, 0 errors, 0 preemptions.
- Needles at 128K, 200K and 250K, three depths each, cold requests: 9/9.
- Greedy parity against the v0.26.0 build, 16 prompts x 192 tokens: 16/16 identical, top-5
  logprobs identical. GSM8K first 50 (thinking off, greedy): 50/50, same answers.
- Tools (auto with thinking on and off, required, named, streaming, multi-turn with a result),
  `json_schema` strict 9/9, reasoning separation: all pass. Request options (`logit_bias`,
  `bad_words`, `allowed_token_ids`, `min_tokens`, `min_p`): all take effect; penalties are applied.
- Minimum free HBM on any card across all of the above: 4.95 GiB, 0 allocation failures.

## Performance

Single measurements on eight cards, the image above, the same harness for both columns, both with
prefix caching on; TTFTs are for requests that did not hit the cache. The v0.26.0 column is the
earlier build of this model (same layout, full KV pool, no long-prefill share), for reference.
Prefix-hit TTFTs from those runs do not apply to this recipe and are not listed.

| Test | This recipe | v0.26.0 build |
| --- | --- | --- |
| Tokens/s per stream at 1 / 4 / 8 / 16 in flight | 27.4 / 18.8 / 18.4 / 17.8 | 28.1 / 15.9 / 15.3 / 15.0 |
| Tokens/s per replica at 16 in flight | 255 | 217 |
| Cold TTFT at 8K / 32K / 128K | 0.94 / 3.89 / 20.2 s | 0.94 / 3.86 / 19.8 s |
| 16 x 32K concurrent, wall time | 100 s | 121 to 137 s |
| Other streams while a 250K prompt prefills | 9.6 tokens/s | 1.1 tokens/s |

A single stream is 2.6% slower on this image (repeated three times on both); the cause is not
found. The 250K prompt's own TTFT doubles with the long-prefill share (about 60 s to 120 s).
Without prefix caching every turn of a multi-turn session is prefilled from the start.

### Alternative: 32 sequences

For throughput over a per-sequence guarantee, `--max-num-seqs 32` passed the same gates (prefix
caching on, a 3,200-block pool of 640-token blocks): 436 tokens/s per replica at 32 in flight, 16.2
per stream, 6.3 GiB HBM free at the minimum. The 32 sequences then share the pool (about 63K each
when all are full). Size the pool again for this recipe as described under Launch.

## Known issues

- This is the `releases/v0.30.0` branch before the tag, without GDN prefix caching. Upstream
  prefix caching for Qwen3.5 uses the non-compact layout (state blocks next to the KV pool); see
  [Prefix caching](#prefix-caching).
- A cold load takes about 40 minutes (406 GB).

## What not to do

- Do not run with the full KV pool at 256K. Under 250K prompts the cards then had 0.07 to 0.25 GiB
  free (prefix caching on), and on the v0.26.0 build one rank failed a 2 GiB attention allocation
  during a 200K prefill.
- Do not raise `--max-num-batched-tokens` to 8,192: a 4,096-token chunk against a 256K context
  already needs a 2 GiB attention buffer per card, and 8,192 doubles it.
- Do not enable MTP speculative decoding on this recipe. It needs fixes that are not in this
  cookbook, and in tests (with prefix caching on) GSM8K kept only 48/50 answers and greedy outputs
  forked at near-ties.
- Do not use lazy mode for this model.
