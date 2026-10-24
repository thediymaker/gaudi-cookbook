# Gemma 4 E2B IT

Google's smallest Gemma 4 instruct model (2.3B effective parameters, 5.1B with the per-layer
embeddings) on one Gaudi 2 card, with KV-sharing fast prefill: the last 20 of its 35 layers run
only on the token each prompt samples, which roughly halves prefill time. Text only.

| | |
| --- | --- |
| Model | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) (Apache 2.0) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 (10.2 GB checkpoint) |
| Context | 32,768 served (131,072 native) |
| Concurrency | `--max-num-seqs 64` |
| Parsers | tools `gemma4` (engine-based, with two fixes) |
| Recipe | [`recipes/google/gemma-4-E2B-it/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/google/gemma-4-E2B-it/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
unset VLLM_SKIP_WARMUP   # warm-up must run; vllm-gaudi reads any non-empty value as true

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 \
    gemma4-required-tool-grammar gemma4-buffered-tool-args gemma4-fast-prefill || exit 1
# Separate step, not run by apply_all.py: a tokenizer copy that adds <bos>
python3 "$COOKBOOK/fixes/gemma4-bos-tokenizer/apply.py" google/gemma-4-E2B-it /tmp/gemma4-bos-tokenizer || exit 1

vllm serve google/gemma-4-E2B-it \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --kv-sharing-fast-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --tokenizer /tmp/gemma4-bos-tokenizer
```

- `--kv-sharing-fast-prefill` with [`gemma4-fast-prefill`](../fixes/gemma4-fast-prefill.md): on a
  prompt batch with no cached context, layers 0 to 14 run on every token and layers 15 to 34 (the
  KV-shared ones, with the double-wide MLPs) only on each prompt's last token. Without the fix the
  flag does not work on HPU. vLLM rejects `prompt_logprobs` requests while the flag is on.
  Measured cost and gain are under [Performance](#performance).
- Warm-up runs. Without it, decode shapes compile in the middle of traffic and the first-token gain
  disappears. `VLLM_SKIP_WARMUP=false` still skips it, so the variable must be unset.
- `--no-enable-prefix-caching`: a prompt with a cache hit takes the full 35-layer path (correct, no
  speed-up), and the recipe's checks ran without caching.
- Text only: the image and audio encoders load but are not used. The vision and audio paths have
  not been checked with fast prefill.
- The prompt batch cap (`VLLM_PROMPT_BS_BUCKET_MAX`) stays at its default of 1. See
  [What not to do](#what-not-to-do) before raising it.

### Buckets

No bucket file and no bucket variables: the release's default bucketing, with the warm-up
compiling every shape before the server takes requests.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.30.0`, which also applies the release's other generic fixes (see
[Using a recipe](index.md#using-a-recipe)). The reasons are the same as on
[Gemma 4 31B IT](gemma-4-31b-it.md#required-fixes) where the fix appears there.

| Fix | Why this model needs it |
| --- | --- |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`so-row-owners`](../fixes/so-row-owners.md) | grammar masks and parameters land on the wrong row next to a partial prompt chunk |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids are corrupted by an int64 concat in lazy mode |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault the worker in lazy mode |
| [`gemma4-required-tool-grammar`](../fixes/gemma4-required-tool-grammar.md) (opt-in) | `tool_choice` required or named on gemma4 skips the grammar, so forced calls come back as prose |
| [`gemma4-buffered-tool-args`](../fixes/gemma4-buffered-tool-args.md) (opt-in) | Gemma 4 escapes quotes inside its own string delimiters; the fix repairs the arguments and sends them in one delta |
| [`gemma4-fast-prefill`](../fixes/gemma4-fast-prefill.md) (opt-in) | makes `--kv-sharing-fast-prefill` work on HPU |
| [`gemma4-bos-tokenizer`](../fixes/gemma4-bos-tokenizer.md) (separate step) | the instruct repo's tokenizer never adds `<bos>`, so raw `/v1/completions` prompts degrade |

## Verified

On vllm-gaudi `releases/v0.30.0`, one Gaudi 2 card, the launch above (the offline checks use the same
image, fixes and lazy mode through `LLM.generate`).

- Fast prefill against stock, offline, 64 wikitext prompts of 1,024 tokens, greedy: first token
  64/64, 32-token continuations identical 64/64. Two stock runs were bit-identical to each other.
- Served, 64 concurrent code-generation prompts (about 600 tokens in, 640 out), two rounds per
  mode: every output starts the same in both modes (first 40 characters 64/64); whole outputs are
  58 to 61 of 64 identical between the modes and 60 of 64 between the two stock rounds, so the
  remaining differences are run-to-run decode variation at 64 concurrent streams, not fast prefill.
- Tool reliability, 3 cases x 40 sampled: 120/120.

## Performance

Single runs on one card, the same card and image for both columns.

Offline, the benchmark method of vllm#60445: `LLM.generate` with `bs` prompts of random tokens,
scored as `bs x (input + output) / batch latency`, three repetitions after two warm-ups, at the
harness's own settings (8,192 context, `--gpu-memory-utilization 0.85`). Prompt batch cap 1 (the
recipe's):

| Case | Stock | Fast prefill | Measured as |
| --- | --- | --- | --- |
| prefill-heavy (1,024 in, 1 out), bs 16 | 59.2k tok/s | 108.7k | same card, same harness |
| prefill-heavy, bs 64 | 61.0k | 106.9k | same run |
| balanced (512 in, 4 out), bs 64 | 48.4k | 71.2k | same run |
| decode-heavy (128 in, 8 out), bs 64 | 21.8k | 20.3k | same run |
| long context (4,096 in, 1 out), bs 16 | 33.0k | 78.5k | same run |

With `VLLM_PROMPT_BS_BUCKET_MAX=8` (not the recipe; see [What not to do](#what-not-to-do)), the
same harness gave 73.6k stock against 170.8k fast prefill on prefill-heavy at bs 64, with stock at
`--gpu-memory-utilization 0.6` because it does not fit warm-up at 0.9.

Served, the launch above, 64 concurrent code-generation requests of about 600 prompt tokens,
two warm-up rounds then two recorded rounds per mode:

| | Stock | Fast prefill | Measured as |
| --- | --- | --- | --- |
| First-token median | 1.01 / 1.00 s | 0.76 / 0.77 s | same card, two rounds each |
| Wall time for the round | 9.29 / 9.18 s | 9.09 / 8.96 s | same rounds |

Writing code is decode-bound, so wall time barely moves.

The [write-up](../results/gemma-4-e2b-prefill.md) has the full study, including higher prompt
batch caps and the comparison with published A100 figures.

## Known issues

- Decode is unchanged by fast prefill; on decode-bound traffic (long answers) wall time barely
  moves. Gemma 4 multi-token prediction has no HPU path yet.
- Prefix-cache hits and later chunks of a prompt longer than `--max-num-batched-tokens` take the
  full path.

## What not to do

- Do not set `VLLM_SKIP_WARMUP` to anything, including `false`.
- Do not raise `VLLM_PROMPT_BS_BUCKET_MAX` if you need outputs that do not depend on which prompts
  arrive together. Several prompts in one prefill forward change the bf16 rounding of the prompt
  K/V with or without fast prefill, and long greedy outputs then drift. It is faster (see
  [Performance](#performance)); stock at cap 8 also ran out of device memory during warm-up at
  `--gpu-memory-utilization 0.9`.
- Do not prepend `<bos>` to raw prompts on the client once `gemma4-bos-tokenizer` is in place.
- Do not expect `prompt_logprobs` from this server: vLLM rejects them with a validation error while
  `--kv-sharing-fast-prefill` is on. Serve without the flag if you need them.
