# Qwen3.6-27B-FP8

Qwen's 27B dense hybrid model (48 Gated DeltaNet layers and 16 attention layers) with vision and
reasoning, from the official FP8 checkpoint, at its native 256K context on a single Gaudi 2 card.
Prefix caching is off on this release (see [Prefix caching](#prefix-caching)).

| | |
| --- | --- |
| Model | [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 32` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen3.6-27B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3.6-27B-FP8/recipe.yaml) |

[Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) has the same geometry and runs
this recipe unchanged, except that it does not need `qwen3-strict-think-end` (120/120 on tool
reliability without it). Some measurements below come from it and are marked.

## Prefix caching

Prefix caching is off: on this release, prefix-cache hits on Gated DeltaNet layers resume from the
wrong state. Upstream vllm-gaudi main has merged prefix caching for the non-compact layout; use a
release that includes it before turning prefix caching on.

An earlier form of this recipe ran with prefix caching on, through a GDN prefix-cache patch that is
not in this cookbook. Most of the checks below come from that configuration and are marked
"prefix caching on". Checks of cache hits (hit-against-cold parity, needles and images served from
the cache, agent replay hit rates and their TTFTs) do not apply and are left out. The cold-path
results should carry over, but this exact command line has not been re-run end to end.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export VLLM_QWEN3_STRICT_THINK_END=1
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=32
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=262144
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=16384
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 qwen3-strict-think-end || exit 1

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 1792 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --structured-outputs-config '{"disable_any_whitespace": true}' \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":16,"video":0}'
```

- Compile mode (`PT_HPU_LAZY_MODE=0`). The GDN layers lose their state cache at the first decode in
  lazy mode.
- `--no-enable-prefix-caching`: see [Prefix caching](#prefix-caching).
- `VLLM_COMPACT_GDN=1` is the stock default when prefix caching is off: the GDN layers keep their
  state in compact per-request slots instead of in KV blocks. The recipe sets it so a different
  image default cannot change it.
- `--structured-outputs-config '{"disable_any_whitespace": true}'` stops forced tool calls from
  emitting whitespace until `max_tokens`. With the default `auto` backend it is refused at start-up
  without `structured-outputs-auto-whitespace`.
- 16 images per prompt is the worst-case context clamp (16,384 tokens per maximum-size image). The
  encoder cache is sized from one item, so the limit costs no memory.

### Buckets

No bucket file. Prompt batch buckets 1 to 8, decode batch buckets 1 to 32, prompt length 128 to
262,144 in steps of 4,096 with exponential bucketing, decode blocks 128 to 16,384 in steps of
1,024.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0`, which also applies the release's other generic fixes (see
[Using a recipe](index.md#using-a-recipe)). The profile also runs `seeded-sampling` and
`prompt-order`; this recipe does not depend on them (the seed segfault is a lazy-mode bug, and
prefix caching is off).

| Fix | Why this model needs it |
| --- | --- |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request; the start-up log says whether the pad was moved |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the id corruption is lazy-mode only, but the fix is harmless here and keeps one fix set across recipes |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | multimodal model with one prompt in a prefill batch returns flat hidden states; the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |
| [`structured-outputs-auto-whitespace`](../fixes/structured-outputs-auto-whitespace.md) | lets `disable_any_whitespace` work with the `auto` backend, which keeps the fallback to guidance for schemas xgrammar cannot compile |
| [`qwen3-strict-think-end`](../fixes/qwen3-strict-think-end.md) (opt-in) | v0.26.0's Qwen3 parser ends reasoning at a `<tool_call>` the model quotes while thinking (the template's example) and returns it as a real call; with `VLLM_QWEN3_STRICT_THINK_END=1` only `</think>` ends reasoning |

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card. "Prefix caching on" marks the earlier configuration
described under [Prefix caching](#prefix-caching); the other flags were the same.

- Needles to 250K tokens on cold requests: 9/9 (prefix caching on).
- 16 images in one prompt: 16/16 colours named in order, a 17th image rejected with a clean 400
  (earlier image, prefix caching off).
- Forced tool calls (`tool_choice: required`): whitespace runaway 0/32 with
  `disable_any_whitespace` (prefix caching on), 7/32 without (earlier image).
- Tool reliability, 3 cases x 40 sampled at 0.6: 117/120 and 120/120 with strict think-end (prefix
  caching on); 110/120 without, with prefix caching on and off alike (calls to the template's
  `example_function_name` and duplicates).
- `prompt_logprobs`: 68/68 sizes x values 0, 1, 5 and 20; greedy parity 10/10 bit-identical (prefix
  caching on).
- Model test suite (health, smoke, tools, coding agent, vision): 29 passed (prefix caching on).

## Performance

Single measurements on one card, vllm-gaudi v0.26.0, Qwen3.5-27B-FP8 (same geometry and settings),
warm, with prefix caching on and requests that did not hit the cache. Cache-hit TTFTs from that
configuration do not apply to this recipe and are not listed.

| Test | Result |
| --- | --- |
| Single-stream decode at 2K / 8K / 32K context | 48.8 / 48.0 / 45.0 tokens/s |
| Cold TTFT at 2K / 8K / 32K | 0.53 / 2.10 / 9.96 s |
| Needle TTFT, cold, at 65K / 130K / 260K | 25 / 69 / 224 s |
| 32 concurrent 12K-token documents, 1,000 tokens each | 32/32 correct, all 32 running at once, KV peak 80% |

Without prefix caching every turn of a multi-turn session is prefilled from the start, so TTFT
grows with the conversation: plan for the cold TTFT of the whole prompt on every request.

## Known issues

- Temperature 0 still varies with batch composition on HPU; expect two or three trajectory families
  across repeated runs under load.
- Give test requests a real token budget: this model reasons at length, and a short `max_tokens`
  returns empty `content` with `finish_reason: length`.

## What not to do

- Do not use lazy mode for this model.
- Do not pass `{"backend": "xgrammar", "disable_any_whitespace": true}` to get past the start-up
  check: schemas xgrammar cannot compile (`multipleOf`, `uniqueItems`, `patternProperties`, ...)
  then fail with HTTP 400 instead of falling back to guidance.
- Do not raise `--max-num-batched-tokens` to 3,584 or more at 256K: a cold prefill past about 131K
  context then died with `PT_DEVMEM OOM: No enough memory for defragment` (seen with prefix caching
  on).
- Do not set `VLLM_QWEN3_STRICT_THINK_END=1` on models that sometimes call tools without closing
  `</think>`; it is for models that always close it.
