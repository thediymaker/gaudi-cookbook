# Gemma 4 31B IT

Google's 31B dense multimodal instruct model (image and text in, text out), served in BF16 at 128K
context on four Gaudi 2 cards with up to 64 images per prompt.

| | |
| --- | --- |
| Model | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (62.5 GB of weights) |
| Context | 131,072 served (262,144 native) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `gemma4` (engine-based, with two fixes) |
| Recipe | [`recipes/google/gemma-4-31B-it/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/google/gemma-4-31B-it/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_MULTIMODAL_BUCKETS=None
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_GRAPH_PROMPT_RATIO=0.15
export VLLM_GRAPH_PROMPT_STRATEGY=max_bs
export VLLM_GRAPH_DECODE_STRATEGY=max_bs
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=16
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=131072
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=4096
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 \
    gemma4-required-tool-grammar gemma4-buffered-tool-args || exit 1
# Separate step, not run by apply_all.py: a tokenizer copy that adds <bos>
python3 "$COOKBOOK/fixes/gemma4-bos-tokenizer/apply.py" google/gemma-4-31B-it /tmp/gemma4-bos-tokenizer || exit 1

vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.78 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt '{"image":64,"video":0}' \
  --trust-remote-code \
  --tokenizer /tmp/gemma4-bos-tokenizer
```

Images cost a fixed 280 soft tokens each, so `{"image":64}` is at most about 18K tokens of context.
Thinking is off by default in this model's template; a request turns it on with
`chat_template_kwargs: {"enable_thinking": true}`.

### Buckets

No bucket file. Prompt batch buckets 1 to 8, decode batch buckets 1 to 16, prompt length 128 to
131,072 in steps of 4,096 with exponential bucketing, decode blocks 128 to 4,096 in steps of 1,024.
`VLLM_MULTIMODAL_BUCKETS=None` leaves vision shapes unbucketed.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0`, which also applies the release's other generic fixes (see [Using
a recipe](index.md#using-a-recipe)). `gemma4-bos-tokenizer` is not a source patch: it is a separate
startup step that writes a tokenizer directory for `--tokenizer`.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the `top_logprobs` id tensor is corrupted by an int64 concat in lazy mode: 26 to 50% of prefill replies were truncated, empty or out of vocabulary |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`mm-merge-idxcopy`](../fixes/mm-merge-idxcopy.md) | the multimodal embedding merge uses a broadcast `masked_scatter_` that fails to graph-compile (`synStatus 26`) when several images arrive in one step |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`gemma4-required-tool-grammar`](../fixes/gemma4-required-tool-grammar.md) (opt-in) | v0.26.0 skips the grammar for `tool_choice` required or named on gemma4, so forced calls came back as prose |
| [`gemma4-buffered-tool-args`](../fixes/gemma4-buffered-tool-args.md) (opt-in) | Gemma 4 escapes quotes inside its own string delimiters; the fix repairs the arguments and sends them in one delta at the end of the call |
| [`gemma4-bos-tokenizer`](../fixes/gemma4-bos-tokenizer.md) (separate step) | the instruct repos ship a tokenizer that never adds `<bos>`, so raw `/v1/completions` prompts produced loops such as `LLLL...` |

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, the flags above. The `top_logprobs` checks ran with one
more logprobs patch (an extra host sync) that is not part of the fixes pack.

- Tool reliability, 3 cases x 40 sampled: 120/120. Streaming with `tool_choice` required, named
  and auto: all correct. Coding-agent tier 5/5 on two runs.
- Streamed tool-argument replay through the parser (escaped quotes, regex backslashes, a quote
  arriving late): 10/10 with `gemma4-buffered-tool-args`, 5/10 stock.
- Multi-image requests with `top_logprobs=20`, 16 concurrent x 3 rounds: 96/96, 0 restarts.
- Teacher-forced top-1 agreement against a reference run, 24 prompts, 11,754 tokens: 0.9924, mean
  NLL 0.05.
- `top_logprobs` soak (unique single-token prompts at 4, 8 and 16 concurrent): 0/1000 corrupt with
  `gather-logprobs-int32`, 52/200 and 75/200 corrupt without.
- BOS tokenizer: raw completions equal explicit-`<bos>` prompts byte for byte; chat prompts tokenize
  to the same ids (5/5 text, an image prompt at 298 tokens); a model test suite (health, smoke, quality, determinism, tools, vision) passes 23/23.
- Seeded traffic at saturation with client aborts on three replicas: 8,650 requests, 0 errors, 0
  restarts; sequential same-seed repeats identical 12/12.

## Performance

Single measurements on four cards, vllm-gaudi v0.26.0, warm.

| Load | Tokens/s per stream | Tokens/s per replica |
| --- | --- | --- |
| 1 stream, short prompt | 42.9 | 42.9 |
| 4 streams, short prompt | 37.4 | about 150 |
| 16 streams, short prompt | 32.9 | about 526 |
| 16 streams, 256 output tokens, `ignore_eos` (unseeded / seeded) | | 626 / 590 to 608 |

## Known issues

- Rare `PT_DEVMEM` allocation failures during HPU graph replay under heavy load are not
  root-caused. vLLM shuts down "gracefully" after the worker error and the container exits 0, so it
  looks like a clean exit: read the previous container log.
- Penalties (including a `--override-generation-config` `repetition_penalty`) are not applied under
  async scheduling on v0.26.0 (see [Qwen3-30B-A3B-2507](qwen3-30b-a3b-2507.md#known-issues)).
- Concurrent repeats of a seeded request in one batch can differ at near-ties: logits move slightly
  with the batch bucket on HPU. Sequential repeats are identical.

## What not to do

- Do not load the older `gemma4_buffered` tool-parser plugin on v0.26.0: the module it imports was
  removed. Use the engine-based `gemma4` parser with the two gemma4 fixes.
- Do not prepend `<bos>` to raw prompts on the client once `gemma4-bos-tokenizer` is in place: the
  model then sees two and degrades the same way the base model does.
- Do not use this model's `top_logprobs` as a judge or guard signal without
  `gather-logprobs-int32`: without it the guard keeps the server alive but the numbers are wrong.
