# gpt-oss-120b

OpenAI's 117B mixture-of-experts reasoning model (5.1B active, harmony format), served from its
native MXFP4 weights at 128K context on four Gaudi 2 cards with expert parallelism. This is the
stock path; the cookbook's MXFP4 MoE kernel makes it faster (see the end of the page).

| | |
| --- | --- |
| Model | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Hardware | 4x Gaudi 2 (TP 4, `--enable-expert-parallel`) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | MXFP4 MoE weights kept packed on the device (65 GB checkpoint) |
| Context | 131,072 (native) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `openai`, reasoning `openai_gptoss` |
| Recipe | [`recipes/openai/gpt-oss-120b/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/openai/gpt-oss-120b/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container. Point `TIKTOKEN_RS_CACHE_DIR`
at storage that survives restarts.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export HPU_MXFP4_NATIVE=1
export HPU_FUSED_MOE=1
export PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128
export TIKTOKEN_RS_CACHE_DIR=/data/tiktoken-rs-cache
export VLLM_SKIP_WARMUP=true
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=16
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=16
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
mkdir -p "$TIKTOKEN_RS_CACHE_DIR"

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 gptoss-harmony-parse-recovery || exit 1

vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --trust-remote-code
```

- `HPU_MXFP4_NATIVE=1` keeps the MXFP4 quantization config, so the MoE runs on the
  `bias_mxfp4_fused_weights` op, which dequantizes in the TPC and applies the clamped SwiGLU inside
  the op.
- `openai_harmony` fetches its vocabulary file at start-up, does not create its own cache
  directory, and fails before the server binds if the directory is missing or the download is
  truncated. Create the directory and keep it between restarts.
- Reasoning effort is per request: `reasoning_effort` `low`, `medium` or `high` (`none` is
  rejected).

### Buckets

No bucket file. Batch buckets 1 to 16 for prompt and decode with exponential bucketing; sequence
and block ranges are the image defaults.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0`, which also applies the release's other generic fixes (see [Using
a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | in lazy mode the `top_logprobs` ids come back corrupted |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`sink-once`](../fixes/sink-once.md) | attention sinks: the prefill kernel ignores the sink argument, and paged decode adds the sink once per KV block (about 80 copies at 10K context) |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | the 128-token sliding layers saw 129 to 256 tokens in decode |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`gptoss-harmony-parse-recovery`](../fixes/gptoss-harmony-parse-recovery.md) (opt-in) | one malformed tool-call header raised out of the harmony parser and turned the whole request into HTTP 500 (5/5 with a system prompt and 3 tools) |

`logits-empty-guard` is in the profile but not needed by this recipe: with async scheduling (the
default) a prompt chunk that does not finish its prompt keeps one logits row, so the
tensor-parallel gather is never empty. It becomes necessary if you add `--no-async-scheduling`.

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, TP 4 with expert parallelism:

- 16 in flight with about 4.1K-token prompts and 1.7K-token outputs: 96/96. Two prompts of about
  107K tokens: 2/2 (TTFT 78 to 99 s).
- Request options: structured output, named and required tool calls through harmony, `min_tokens`,
  `min_p` and `logit_bias` all take effect.
- With `sink-once` and `swa-decode-window`: long-context needles 9/9 (5/9 without), the model test
  suite 17/17 on two runs (14/17 and 15/17 without), coding-agent tier 5/5 twice (2/5 and 3/5
  without). The failures without them were malformed tool JSON and leaked headers.
- At `--max-num-seqs 32`: harmony tools 8/8 per case (one-shot, required, tool-result turn,
  second-call turn), `json_schema` 6/6, two needles at about 112K tokens found, GSM8K 49/50.

## Performance

Single measurements on four cards, vllm-gaudi v0.26.0, stock MoE op, about 3.2K-token prompts and
1,024 output tokens (run at `--max-num-seqs 32`).

| In flight | Tokens/s per stream | Tokens/s per replica | TTFT |
| --- | --- | --- | --- |
| 1 | 57.0 | 57 | 0.61 s |
| 4 | 20.1 | 78 | 1.94 s |
| 16 | 10.0 | 150 | 6.57 s |
| 32 | 8.9 | 253 | 13.20 s |

At `--max-num-seqs 16` with production-shaped requests (about 4.1K-token prompts, 1.7K-token outputs) a
replica gave 71.7 tokens/s at 4 in flight and 133.9 at 16. The KV pool holds 3.7M tokens, so
concurrency is limited by decode speed, not memory.

### Faster: the MXFP4 MoE kernel

The MoE is most of a decode step. The cookbook's [MXFP4 MoE decode kernel](../kernels/index.md)
replaces the stock op for decode steps of up to 32 tokens and reads the packed weights directly.
Measured on the same server (TP 4, expert parallel), per stream: 62.6 vs 57.0 at 1 in flight, 59.6 vs
20.1 at 4, 22.4 vs 10.0 at 16. With it, two cards (TP 2) give 52 tokens/s per stream at 4 in flight,
where this four-card stock recipe gives 20. Prefill is unchanged.

## Known issues

- Multi-turn agent sessions in harmony still have rough edges with some clients; keep both parsers
  on and the harmony recovery fix in place.
- Penalties are not applied under async scheduling on v0.26.0 (see
  [Qwen3-30B-A3B-2507](qwen3-30b-a3b-2507.md#known-issues)).

## What not to do

- Do not run the tool parser without `--reasoning-parser openai_gptoss`: the analysis channel is
  then never split off and its text lands inside the tool arguments. A short single call still
  looks fine, so simple tests miss it.
- Do not set `HPU_MXFP4_NATIVE=0` on Gaudi 2. The weights are then dequantized to BF16 and the MoE
  lands on a fused BF16 TPC kernel larger than the 64 KB instruction cache, which faults the device.
  The old workaround (a non-fused per-expert Python loop) costs about 3x throughput.
- Do not leave the harmony vocabulary cache in the container's `/tmp`: every start then downloads it
  again, and a truncated download is fatal before the server binds.
