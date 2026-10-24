# Qwen3.5-122B-A10B-FP8

Qwen's 122B hybrid MoE (Gated DeltaNet and attention layers, 256 experts, 10B active) reasoning
model from the official FP8 checkpoint, text only, at its native 256K context on four Gaudi 2 cards
with expert parallelism. Prefix caching is on through the
[`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) |
| Hardware | 4x Gaudi 2 (TP 4, expert parallel) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations, 127 GB) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen3.5-122B-A10B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3.5-122B-A10B-FP8/recipe.yaml) |

## Prefix caching

On the stock release, prefix-cache hits on the Gated DeltaNet layers resume from the wrong state,
so prefix caching is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md). The fix
keeps the GDN layers in their compact per-request slots (`VLLM_COMPACT_GDN=1`) and stores a state
checkpoint at the end of every full KV block in that block's own attention page. A hit restores the
checkpoint before the next chunk, and the full KV pool stays available. With caching on, the hybrid
KV block is 1,152 tokens and prefill chunks end on block boundaries; prompts shorter than one block
never hit. The design and the upstream status are in
[Prefix caching on Gated DeltaNet hybrids](../results/gdn-prefix-caching.md).

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_HPU_MOE_GATHER=1
export VLLM_HPU_MOE_GATHER_RATIO=1.0
export VLLM_HPU_MOE_GATHER_VERIFY=0
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=16
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=16
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=262144
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=16384
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 gdn-prefix-cache fp8-moe-gather || exit 1

vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.75 \
  --block-size 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only
```

- Compile mode (`PT_HPU_LAZY_MODE=0`), with the compact GDN state layout that `gdn-prefix-cache` needs.
- `--enable-expert-parallel` puts 64 of the 256 experts on each card.
- `fp8-moe-gather` with `VLLM_HPU_MOE_GATHER=1`: at low token counts only the routed experts are
  read. The ratio is per card, so with 64 local experts `1.0` keeps the gather path for up to 8
  decode tokens (8 x 8 routed experts); larger steps use the stock MoE op.
- `--language-model-only` serves the model as text only and skips the vision tower's memory
  profiling.
- `--max-num-seqs 16` and `--gpu-memory-utilization 0.75` are the values the deployment ran with;
  per-stream decode falls as concurrency rises, so 16 rather than 32.
- With prefix caching on, prefill chunks end on 1,152-token blocks: the 4,096-token budget runs
  as 3 blocks (3,456 tokens) per chunk.

### Buckets

No bucket file. Prompt and decode batch buckets 1 to 16, prompt length 128 to 262,144 in steps of
4,096 with exponential bucketing, decode blocks 128 to 16,384 in steps of 1,024.

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
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request |
| [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) (opt-in) | prefix-cache hits on the GDN layers resume from a wrong state on the stock release; the fix stores and restores GDN checkpoints in the KV pages and zeroes new blocks and slots. It needs `hybrid-decode-pad` first |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the id corruption is lazy-mode only, but the fix is harmless here and keeps one fix set across recipes |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | `qwen3_5_moe` returns flat hidden states (also with `--language-model-only`); the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`fp8-moe-gather`](../fixes/fp8-moe-gather.md) (opt-in) | gathered-expert FP8 MoE, a backport from v0.30.0: faster low-batch decode |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, with exactly the launch command above in a container
started from the stock image (the fixes applied at start-up). Requests went to the
OpenAI-compatible API.

| Check | Result |
| --- | --- |
| Known-answer prompts (arithmetic, facts, translation), temperature 0 | 8/8 |
| The same prompts, 16 in flight | 16/16 |
| Tool call, `tool_choice: auto`, sampled at 0.6 | 5/5 valid `get_weather` calls with the right argument |
| Streamed tool call | name and JSON arguments assembled correctly |
| Tool result fed back | the answer uses the returned values |
| `top_logprobs: 20` | 20 entries on every token |
| `prompt_logprobs: 5` | returned; server healthy afterwards |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect |
| Prefix-cache hit vs cold (`cache_salt`), greedy, 8 prompts of 6K to 126K tokens with a needle | identical text and bit-identical logprobs 8/8; cached tokens a multiple of 1,152 on every hit |
| Needle in a haystack, 3 depths, cold | 3/3 at 65K, 131K and 244K tokens |
| Server health after all checks | healthy, no restarts |

KV pool at start-up: 2,885,824 tokens.

## Performance

Single runs on the server above (measured as: per-call, one server; times are end to end and
include time to first token).

| Test | Result |
| --- | --- |
| One stream, short prompt, 256 tokens | 43.8 tokens/s |
| 16 streams, short prompts, 256 tokens each | 315.6 tokens/s aggregate, 19.7 per stream |
| Cold prompt of 65K tokens (needle) | 16 to 25 s to the answer |
| Cold prompt of 131K tokens (needle) | 24 to 33 s to the answer |
| Cold prompt of 244K tokens (needle) | 56 to 82 s to the answer |

## Known issues

- Loading is slow under a CPU quota: vllm-gaudi sizes its load thread pool from the host's core
  count, and the per-tensor FP8 rescale on Gaudi 2 is then CPU-throttled. Run without a CPU limit,
  or cap the pool at the quota.
- Give test requests a real token budget: this model reasons at length.

## What not to do

- Do not use lazy mode for this model.
- Do not turn on prefix caching without `gdn-prefix-cache`, and do not set `VLLM_COMPACT_GDN=0`
  with it.
- Do not copy `VLLM_HPU_MOE_GATHER_RATIO` to a layout without expert parallelism: the ratio is
  relative to the experts on one card.
