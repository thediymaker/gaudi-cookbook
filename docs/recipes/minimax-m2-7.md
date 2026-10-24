# MiniMax-M2.7

MiniMax's 229B MoE reasoning and agent model (256 experts with sigmoid routing, about 10B active)
from the official FP8 checkpoint, at 128K context with tool calling and prefix caching, on four
Gaudi 2 cards.

| | |
| --- | --- |
| Model | [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |
| Licence | MiniMax M2.7 licence: non-commercial use; commercial use needs written authorization from MiniMax |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations, 230 GB) |
| Context | 131,072 (the checkpoint allows 204,800) |
| Concurrency | `--max-num-seqs 4` |
| Parsers | tools `minimax_m2`, reasoning `minimax_m2` |
| Recipe | [`recipes/MiniMaxAI/MiniMax-M2.7/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/MiniMaxAI/MiniMax-M2.7/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_WEIGHT_SHARING=0
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0.30
export VLLM_GRAPH_PROMPT_STRATEGY=max_bs
export VLLM_GRAPH_DECODE_STRATEGY=max_bs
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.30.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 || exit 1

vllm serve MiniMaxAI/MiniMax-M2.7 \
  --dtype auto \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.94 \
  --block-size 128 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2
```

- TP 4 without expert parallelism: each card holds a 384-wide slice of every expert's 1,536-wide
  MLP, a multiple of the 128-wide FP8 weight block.
- Prefix caching is on. Agent loops re-send a growing prefix, and a repeated long prompt is served
  from the cache.
- `VLLM_GRAPH_RESERVED_MEM=0.2` leaves graph memory for the batch-4 decode graphs.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`: with warm-up skipped every new shape compiles inside
  the model call, and the 300 s default can stop the engine mid-compile.
- The recipe runs on the v0.30 branch: the v0.26.0 release cannot load this checkpoint (its
  `minimax_m2` model calls `FusedMoE.make_expert_params_mapping`, which vLLM `568afb3a` no longer
  has).

### Buckets

No bucket file. Exponential bucketing with the default ranges for 4 sequences and 1,024-token
chunks.

## Required fixes

All from [Fixes](../fixes/index.md), all in `--profile v0.30.0`, which also applies the release's
other generic fixes (see [Using a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on: a request must not read KV that another request in the same step has not written yet |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits repeats its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` token ids read back corrupted in lazy mode |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | `prompt_logprobs` requests crash or return wrong values |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | seeded requests can segfault a tensor-parallel worker in lazy mode |
| [`laguna-moe-sigmoid-routing`](../fixes/laguna-moe-sigmoid-routing.md) | MiniMax-M2 routes with sigmoid scores and a correction bias; the stock HPU MoE routing applies softmax and ignores the bias |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |

## Verified

On vllm-gaudi `releases/v0.30.0` (`058f107`), four Gaudi 2 cards, with exactly the launch command
above in a container started from the stock image (the fixes applied at start-up). Requests went to
the OpenAI-compatible API; needles used a different haystack per depth, so none was served from the
prefix cache.

| Check | Result |
| --- | --- |
| Known-answer prompts (arithmetic, facts, translation), temperature 0 | 7/8; the eighth repeated itself inside its reasoning until `max_tokens` (see [Known issues](#known-issues)) |
| The same prompts, 4 in flight | 4/4 |
| 10 short prompts with the checkpoint's sampling defaults (temperature 1.0, top-p 0.95, top-k 40) | 10/10 finished with a correct answer |
| The same 10 prompts at temperature 0, 8,192-token budget | 7/10 finished; 3 looped in their reasoning |
| Tool call, `tool_choice: auto`, sampled at 0.6 | 5/5 valid `get_weather` calls with the right argument |
| Streamed tool call | name and JSON arguments assembled correctly |
| Tool result fed back | the answer uses the returned values |
| `top_logprobs: 20` | 20 entries on every token |
| `prompt_logprobs: 5` | returned; server healthy afterwards |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect |
| Needle in a haystack, 3 depths, cold | 3/3 at 32K tokens, 3/3 at 120K tokens |
| Server health after all checks | healthy, no restarts |

KV pool at start-up: 481,152 tokens (3.67 requests of 131,072 tokens).

## Performance

Single runs on the server above (measured as: per-call, one server; times are end to end and
include time to first token and the model's reasoning).

| Test | Result |
| --- | --- |
| One stream, short prompt, 256 tokens | 22.9 to 25.4 tokens/s (two runs) |
| 4 streams, short prompts, 256 tokens each | 87.6 to 95.1 tokens/s aggregate (two runs) |
| Cold prompt of 32K tokens (needle), answer after reasoning | 53 to 56 s (213 s for the first, which also compiled its graphs) |
| Cold prompt of 120K tokens (needle), answer after reasoning | 144 to 226 s |

## Known issues

- At temperature 0 the model sometimes repeats itself inside its reasoning until `max_tokens` (3 of
  10 short prompts here). Use the checkpoint's sampling defaults (temperature 1.0, top-p 0.95,
  top-k 40); vLLM applies them from `generation_config.json` when a request does not set them.
- Loading is CPU-bound: every FP8 tensor is rescaled on the CPU, and vllm-gaudi sizes its load
  thread pool from the host's core count. Run without a CPU limit, or cap the pool at the quota.
- A full warm-up of this MoE takes far too long; graphs compile on first use, so the first request
  at each new shape is slow.

## What not to do

- Do not set `VLLM_ROW_PARALLEL_CHUNKS` above 1: on v0.26.0 any such value stops vLLM at startup
  (`AttributeError` in `register_hpu_lora_layers`), and with 1,024-token chunks a 16,384-token
  chunking threshold would never be reached anyway.
- Do not use the v0.26.0 release for this model (see above).
