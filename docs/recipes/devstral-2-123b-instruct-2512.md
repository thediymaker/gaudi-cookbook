# Devstral-2-123B-Instruct-2512

Mistral AI's 123B dense agentic coding model from the official FP8 checkpoint, at 128K context with
tool calling and prefix caching, on four Gaudi 2 cards.

| | |
| --- | --- |
| Model | [mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) |
| Licence | Modified MIT: no rights for companies above USD 20 million monthly revenue without a commercial licence from Mistral AI |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (128 GB of weights) |
| Context | 131,072 (the checkpoint allows 262,144) |
| Concurrency | `--max-num-seqs 4` |
| Parsers | tools `mistral` |
| Recipe | [`recipes/mistralai/Devstral-2-123B-Instruct-2512/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/mistralai/Devstral-2-123B-Instruct-2512/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.40
export VLLM_GRAPH_PROMPT_RATIO=0.30
export VLLM_GRAPH_PROMPT_STRATEGY=max_bs
export VLLM_GRAPH_DECODE_STRATEGY=max_bs
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=32
export VLLM_PROMPT_CTX_BUCKET_MAX=992
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=4
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=4
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=2048
export VLLM_PROMPT_SEQ_BUCKET_MAX=4096
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=4096
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 || exit 1

vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.92 \
  --block-size 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --trust-remote-code
```

- Prefix caching is on. Agent and coding traffic re-sends the previous turn's prefix, so most
  prompt tokens come from the cache after the first turn.
- `--max-num-seqs 4`: at 1 the server serializes and a short request waits for the whole of a long
  one. 4 keeps the KV pool far from full at 128K.
- `--trust-remote-code` is kept from the verified command line; the checkpoint loads through vLLM's
  own Mistral 3 classes.

### Buckets

No bucket file. Linear bucketing (`VLLM_EXPONENTIAL_BUCKETING=False`): batch buckets 1 to 4,
prompt chunks 128 to 4,096 in steps of 2,048 (a chunk is never larger than
`--max-num-batched-tokens`), prompt context 0 to 992 blocks in steps of 32 blocks (one 4,096-token
chunk), decode blocks 128 to 4,096 in steps of 1,024. Exponential context buckets pad a chunk's
attention up to 2x and made cold long prefills slower (cold 32K TTFT 18.5 s vs 14.7 s linear,
130K needle 239 s vs 180 s; measured as: same-pod A/B). Prefix hits above 992 context blocks use
one fallback bucket and log a warning.

## Required fixes

All from [Fixes](../fixes/index.md), all in `--profile v0.26.0`, which also applies the release's
other generic fixes (see [Using a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on: a request must not read KV that another request in the same step has not written yet |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | the `mistral` tool parser gives every chat request a grammar, so every step that mixes a final prefill chunk with decodes sampled rows with another request's temperature, seed and `logit_bias` |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits repeats its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` token ids read back corrupted in lazy mode |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | `prompt_logprobs` requests crash or return wrong values |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | seeded requests can segfault a tensor-parallel worker in lazy mode |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, with exactly the launch command above in a container
started from the stock image (the fixes applied at start-up). Requests went to the
OpenAI-compatible API; needles used a different haystack per depth, so none was served from the
prefix cache.

| Check | Result |
| --- | --- |
| Known-answer prompts (arithmetic, facts, translation), temperature 0 | 8/8 |
| The same prompts, 4 in flight | 4/4 |
| Tool call, `tool_choice: auto`, sampled at 0.6 | 5/5 valid `get_weather` calls with the right argument |
| Streamed tool call | name and JSON arguments assembled correctly |
| Tool result fed back | the answer uses the returned values |
| `top_logprobs: 20` | 20 entries on every token |
| `prompt_logprobs: 5` | returned; server healthy afterwards |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect |
| Needle in a haystack, 3 depths, cold | 3/3 at 32K tokens, 3/3 at 120K tokens |
| Server health after all checks | healthy, no restarts |

KV pool at start-up: 425,216 tokens.

## Performance

Single runs on the server above (measured as: per-call, one server; times are end to end and
include time to first token).

| Test | Result |
| --- | --- |
| One stream, short prompt, 256 tokens | 22.1 tokens/s |
| 4 streams, short prompts, 256 tokens each | 85.7 tokens/s aggregate, 21.4 per stream |
| Cold prompt of 32K tokens (needle) | 15 to 18 s to the answer |
| Cold prompt of 120K tokens (needle) | 137 to 158 s to the answer |

## Known issues

- Chunked prefill does not interleave a short request with a long prefill: the 4,096-token budget
  goes to the long prompt's chunk on every step, so a short request admitted behind it waits until
  that prefill finishes. Raising `--max-num-seqs` does not change this.
- A cold 120K-token prompt took 137 to 158 s to answer (see [Performance](#performance)); size
  client and gateway timeouts for it. With prefix caching only the first turn of a conversation
  pays it.
- A named `tool_choice` returns `finish_reason: "stop"`, also when streaming.

## What not to do

- Do not switch to exponential bucketing for long-context traffic (see [Buckets](#buckets)).
- Do not go back to `--max-num-seqs 1`.
