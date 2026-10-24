# Qwen-AgentWorld-35B-A3B

Qwen's 35B hybrid MoE agent model (30 Gated DeltaNet and 10 attention layers, 256 experts, 3B
active) in BF16, text only, at its native 256K context on two Gaudi 2 cards, 16 sequences at a
time. Prefix caching is off on this release (see [Prefix caching](#prefix-caching)).

| | |
| --- | --- |
| Model | [Qwen/Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (69 GB of weights) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen-AgentWorld-35B-A3B/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen-AgentWorld-35B-A3B/recipe.yaml) |

## Prefix caching

Prefix caching is off: on this release, prefix-cache hits on Gated DeltaNet layers resume from the
wrong state. Upstream vllm-gaudi main has merged prefix caching for the non-compact layout; use a
release that includes it before turning prefix caching on.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 || exit 1

vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- Compile mode (`PT_HPU_LAZY_MODE=0`), with the compact GDN state layout that is the stock default
  without prefix caching, as for the other Qwen3.5-family recipes. On an earlier vllm-gaudi release
  lazy mode failed the first decode at TP 2 in the GDN layers (`tensor_data is empty` in
  `ValidateSyncInputTensors`).
- No `VLLM_*_BUCKET_*` variables: compile mode, default buckets.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`: with warm-up skipped every new shape compiles inside
  the model call, and the 300 s default can stop the engine mid-compile.
- `--language-model-only` is required: `config.json` describes a vision tower, but the checkpoint
  ships only the language model, and without the flag vLLM tries to build the visual modules and
  fails.
- TP 2 leaves room for the large contiguous allocation of the GDN chunked prefill next to about
  35 GB of weights per card.

### Buckets

No bucket file and no bucket variables.

## Required fixes

All from [Fixes](../fixes/index.md), all in `--profile v0.26.0`, which also applies the release's
other generic fixes (see [Using a recipe](index.md#using-a-recipe)). The profile also runs
`seeded-sampling`, `prompt-order` and `resumed-prefill-logits`; this recipe does not depend on them
(the seed segfault is a lazy-mode bug, and prefix caching is off).

| Fix | Why this model needs it |
| --- | --- |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the id corruption is lazy-mode only, but the fix is harmless here and keeps one fix set across recipes |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | `qwen3_5_moe` returns flat hidden states; the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |

## Verified

On vllm-gaudi v0.26.0, two Gaudi 2 cards, with exactly the launch command above in a container
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
| Needle in a haystack, 3 depths, cold | 3/3 at 65K, 131K and 244K tokens |
| Server health after all checks | healthy, no restarts |

KV pool at start-up: 5,197,487 tokens.

## Performance

Single runs on the server above (measured as: per-call, one server; times are end to end and
include time to first token).

| Test | Result |
| --- | --- |
| One stream, short prompt, 256 tokens | 26.3 tokens/s |
| 16 streams, short prompts, 256 tokens each | 321.4 tokens/s aggregate, 20.1 per stream |
| Cold prompt of 65K tokens (needle) | 17 s (99 s for the first, which also compiled its graphs) to the answer |
| Cold prompt of 131K tokens (needle) | 33 to 46 s to the answer |
| Cold prompt of 244K tokens (needle) | 72 to 85 s to the answer |

## Known issues

- Offered a calculator tool, the model often works out easy arithmetic itself instead of calling
  it. The tools are rendered and parsed correctly; it calls the tool when asked to, or for harder
  sums. This is model behaviour.
- Give test requests a real token budget: the model reasons before answering.

## What not to do

- Do not drop `--language-model-only`.
- Do not use lazy mode.
- Do not turn on prefix caching on this release.
