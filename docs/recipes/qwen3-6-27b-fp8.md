# Qwen3.6-27B-FP8

Qwen's 27B dense hybrid model (48 Gated DeltaNet layers and 16 attention layers) with vision and
reasoning, from the official FP8 checkpoint, at its native 256K context on a single Gaudi 2 card,
with prefix caching on through the [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

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

On the stock release, prefix-cache hits on the Gated DeltaNet layers resume from the wrong state,
so prefix caching is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md). The fix
keeps the GDN layers in their compact per-request slots (`VLLM_COMPACT_GDN=1`) and stores a state
checkpoint at the end of every full KV block in that block's own attention page, which a block
owned by a GDN group does not otherwise use. A hit restores the checkpoint, bit for bit, before the
next chunk. The full KV pool stays available: 623,359 tokens at the flags below.

With caching on, the hybrid KV block is 896 tokens (one page holds a whole GDN state) and prefill
chunks end on block boundaries. The start-up log confirms both:

```text
[gaudi-fix:gdn-prefix-cache] active: GDN groups [0, 1, 2], block 896 tokens, 711 KV blocks, 16 layer positions; ...
[gaudi-fix:gdn-prefix-cache] scheduler: GDN prefill chunks end on 896-token blocks
```

The design, the earlier measurements and the upstream status are in
[Prefix caching on Gated DeltaNet hybrids](../results/gdn-prefix-caching.md).

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
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 \
    gdn-prefix-cache qwen3-strict-think-end || exit 1

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 1792 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --structured-outputs-config '{"disable_any_whitespace": true}' \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":16,"video":0}'
```

- Compile mode (`PT_HPU_LAZY_MODE=0`). The GDN layers lose their state cache at the first decode in
  lazy mode.
- `--enable-prefix-caching` is correct here only with `gdn-prefix-cache`; see
  [Prefix caching](#prefix-caching).
- `VLLM_COMPACT_GDN=1`: the GDN layers keep their state in compact per-request slots, and
  `gdn-prefix-cache` refuses to start without it.
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
[Using a recipe](index.md#using-a-recipe)). The profile also runs `seeded-sampling`; this recipe
does not depend on it (the seed segfault is a lazy-mode bug).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | with prefix caching, a request that hits a block another request writes in the same step read it before it was written |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request; the start-up log says whether the pad was moved |
| [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) (opt-in) | prefix-cache hits on the GDN layers resume from a wrong state on the stock release; the fix stores and restores GDN checkpoints in the KV pages and zeroes new blocks and slots. It needs `hybrid-decode-pad` first |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the id corruption is lazy-mode only, but the fix is harmless here and keeps one fix set across recipes |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | multimodal model with one prompt in a prefill batch returns flat hidden states; the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |
| [`structured-outputs-auto-whitespace`](../fixes/structured-outputs-auto-whitespace.md) | lets `disable_any_whitespace` work with the `auto` backend, which keeps the fallback to guidance for schemas xgrammar cannot compile |
| [`qwen3-strict-think-end`](../fixes/qwen3-strict-think-end.md) (opt-in) | v0.26.0's Qwen3 parser ends reasoning at a `<tool_call>` the model quotes while thinking (the template's example) and returns it as a real call; with `VLLM_QWEN3_STRICT_THINK_END=1` only `</think>` ends reasoning |

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card, the launch above as written (prefix caching on with
`gdn-prefix-cache`):

- Greedy parity, 32 prompts of 5K to 31K tokens (single questions, multi-turn agent sessions with
  tools, extensions, diverging prefixes; thinking on and off), the same request served from the
  cache and cold: 31/32 identical, 27/32 with bit-identical logprobs. The other one forked at an exact
  tie (the two top tokens both at logprob -0.6932). Two cold runs: 32/32 identical.
- Cross-request isolation, 24 rounds with an unrelated request between fill and hit: 0 leaks.
- Requests sharing a prefix in the same step, 24: 24/24 correct; 22/24 word for word as when sent
  one at a time (the two differences are batch-composition wording, one of them on a request
  without a cache hit).
- Images served from the cache, 8 cases: hit equals cold 8/8.
- Needles at 65K, 130K and 250K tokens, three depths each: 9/9 cold, 9/9 from the cache, hit
  equals cold 9/9.
- 16 images in one prompt: 16/16 colours named in order (thinking off), a 17th image rejected with
  a clean 400.
- Forced tool calls (`tool_choice: required`): whitespace runaway 0/32 with
  `disable_any_whitespace`.
- Tool reliability, 3 cases x 40 sampled at 0.6: 119/120 with strict think-end (one wrong tool).
- `prompt_logprobs`: ids aligned with the prompt and in vocabulary at every size checked, values
  consistent with decode logprobs (median difference 0.001 to 0.007).
- Agent replay, 4 sessions growing from 8K to 64K tokens: 0 preemptions, 0 engine restarts across
  all of the above.

## Performance

Single runs on one card, vllm-gaudi v0.26.0, warm.

| Test | Prefix caching on (this recipe) | Prefix caching off | Measured as |
| --- | --- | --- | --- |
| Agent replay, 4 sessions 8K to 64K, 200 output tokens per turn: wall | 171 s | 513 s | same pod and card, the off arm launched with `--no-enable-prefix-caching` |
| later-turn TTFT mean / p50 / p90 | 11.8 / 6.7 / 22.2 s | 32.4 / 24.2 / 81.2 s | same run |
| first-turn TTFT mean | 14.4 s | 26.7 s | same run |
| server prefix-cache hit rate | 78% | | same run |
| Needle at 250K: TTFT cold / from the cache | 224 to 226 s / 1.1 to 1.7 s | | same pod, cold then hit |
| Needle at 130K / 65K: TTFT cold | 69 / 25 to 26 s | | same pod |
| KV pool | 623,359 tokens | | start-up log |

Earlier single measurements on Qwen3.5-27B-FP8 (same geometry and settings), requests that did not
hit the cache: single-stream decode 48.8 / 48.0 / 45.0 tokens/s at 2K / 8K / 32K context; cold
TTFT 0.53 / 2.10 / 9.96 s at the same lengths; 32 concurrent 12K-token documents with 1,000 output
tokens each, all 32 running at once, 32/32 correct, KV peak 80%.

## Known issues

- Prompts shorter than one 896-token block never hit the cache, and a hit resumes from the last
  full block before the end of the shared prefix.
- Cold outputs with prefix caching on and off are not bit-identical. On the 32 parity prompts,
  greedy, the two launches agreed on 27/32 outputs; two launches with caching off that differ only
  in `--max-num-batched-tokens` (1,792 against 896) agreed on 20/32. That is the noise floor of
  changing the prefill chunking on this FP8 model, and the caching difference stays inside it.
- Temperature 0 still varies with batch composition on HPU; expect two or three trajectory families
  across repeated runs under load.
- Give test requests a real token budget: this model reasons at length, and a short `max_tokens`
  returns empty `content` with `finish_reason: length`.

## What not to do

- Do not turn on prefix caching on this release without `gdn-prefix-cache`: hits then resume the
  GDN layers from a wrong state.
- Do not use lazy mode for this model.
- Do not pass `{"backend": "xgrammar", "disable_any_whitespace": true}` to get past the start-up
  check: schemas xgrammar cannot compile (`multipleOf`, `uniqueItems`, `patternProperties`, ...)
  then fail with HTTP 400 instead of falling back to guidance.
- Do not raise `--max-num-batched-tokens` to 3,584 or more at 256K: a cold prefill past about 131K
  context then died with `PT_DEVMEM OOM: No enough memory for defragment` (seen with prefix caching
  on).
- Do not set `VLLM_QWEN3_STRICT_THINK_END=1` on models that sometimes call tools without closing
  `</think>`; it is for models that always close it.
