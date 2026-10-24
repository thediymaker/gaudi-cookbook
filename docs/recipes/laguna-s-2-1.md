# Laguna-S-2.1

poolside's 118B mixture-of-experts coding and agent model (about 8B active; 256 experts, top 10,
sigmoid router with a selection bias; sliding-window and full-attention layers), served in BF16 at
128K context on four Gaudi 2 cards.

| | |
| --- | --- |
| Model | [poolside/Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `releases/v0.30.0` at `058f107`, vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 (235 GB of weights) |
| Context | 131,072 served (1,048,576 native) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools and reasoning `poolside_v1` (bundled with vLLM) |
| Recipe | [`recipes/poolside/Laguna-S-2.1/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/poolside/Laguna-S-2.1/recipe.yaml) |

`LagunaForCausalLM` is upstream in vLLM and runs on the generic HPU path: no model code to port.
The routing fix below is the only model-specific change.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=4
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=1024
export VLLM_PROMPT_SEQ_BUCKET_MAX=2048
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=16
export VLLM_PROMPT_CTX_BUCKET_MAX=1008
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=16
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=5120
export VLLM_GRAPH_RESERVED_MEM=0.2
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_RPC_TIMEOUT=7200000

# Fixes: the v0.30.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 || exit 1

vllm serve poolside/Laguna-S-2.1 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --no-async-scheduling \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

- `--default-chat-template-kwargs '{"enable_thinking": true}'`: the template defaults thinking on,
  but the reasoning parser defaults it off and then passes everything through as content. The flag
  makes the parser split reasoning; it does not change the rendered prompt.
- `--no-async-scheduling`: this recipe was verified with sync scheduling. With TP 4 that makes
  `logits-empty-guard` necessary; `async-inputs` (in the profile) then has nothing to do.
- Coarse linear buckets with `VLLM_SKIP_WARMUP=true`: the shape set is small enough to fill once,
  after which agent turns stop hitting fresh graph compiles.

### Buckets

No bucket file. Linear bucketing (`VLLM_EXPONENTIAL_BUCKETING=False`): prompt batch 1 to 4, prompt
query length 128 to 2,048 in steps of 1,024, prompt context 0 to 1,008 blocks in steps of 16
(2K tokens), decode batch 1 to 16 in steps of 4, decode blocks 128 to 5,120 in steps of 1,024.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.30.0`, which also applies the release's other generic fixes (see [Using
a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`so-row-owners`](../fixes/so-row-owners.md) | with a prompt chunk that does not finish its prompt in the step, grammar masks and sampling parameters land one row off (another request forced into JSON, or an `IndexError` that stops the engine); without async scheduling such a chunk also leaked a stray token |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`logits-empty-guard`](../fixes/logits-empty-guard.md) | sync scheduling with TP 4: a prompt chunk that does not finish its prompt has no logits rows, and the tensor-parallel all-gather over the empty tensor stops the engine (`ValidateSyncInputTensors`) |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the logprob id corruption is still in v0.30 in lazy mode |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`laguna-moe-sigmoid-routing`](../fixes/laguna-moe-sigmoid-routing.md) | vllm-gaudi's monolithic MoE path routes every non-grouped MoE with softmax and drops `e_score_correction_bias`, so it picks the wrong experts: prose stays fluent, counting, arithmetic and code collapse. The fix computes sigmoid scores, adds the bias for selection only and renormalizes, and fails closed |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |

`rope-cpu-cache` is not listed: the RoPE fix is upstream in v0.30.0, and the v0.30.0 profile leaves
it out.

On v0.30 the bundled `poolside_v1` tool parser already accepts the template's
`<tool_call>NAME<arg_key>` form without a newline, so no tool-parser fix is needed.

## Verified

On the image above, four Gaudi 2 cards, the flags above:

- Exact-tracking probes (17 x 20, 23 x 17, primes between 50 and 70, a sum of squares, a letter
  count), three runs each: all correct.
- Tool calls 4/4 non-streaming and 4/4 streaming; reasoning separated from content.
- Needle at 128K tokens: found (117 s cold).
- LiveBench coding at temperature 1.0 with a 32K budget: LCB generation 85.9; coding completion
  52.0, 56.0 and 50.0 over three runs. The previous image scored 56.0 the same day: run-to-run
  spread on this task is about 7 points.
- Sampler check at temperature 1.0 (400 first-token draws x 4 prompts): sampled frequency of the
  top token within 0.01 to 0.02 of its probability.

## Performance

Single measurements on four cards, the image above, warm shapes, unique prompts at temperature 0
with `ignore_eos`.

| Test | Result |
| --- | --- |
| 1 stream, 1K prompt | 53.2 tokens/s, TTFT 0.15 s |
| 2 / 4 / 8 / 16 streams, per stream | 56.2 / 44.7 / 36.7 / 23.2 tokens/s |
| 4 / 8 / 16 streams, per replica | 167 / 256 / 308 tokens/s |
| TTFT at 8K / 32K, cold prefix | 1.52 to 1.61 / 7.79 s |
| TTFT at 8K, repeated (7,936 tokens cached) | 0.52 s |
| Agent session (8 turns, about 1,700 new tokens each, after warm-up) | TTFT 0.2 to 0.6 s, 44 to 48 tokens/s, 5.9 to 6.4 s per turn |

## Known issues

- The first request at each new shape still compiles on a fresh replica (seconds to tens of
  seconds). With these coarse buckets the set fills quickly; with step-1 buckets it never does.
- Laguna loops in its reasoning at temperature 0: at a 16K budget about 37% of coding answers hit
  the cap while still thinking. Use the model's sampling defaults and a generous `max_tokens`.

## What not to do

- Do not serve without `laguna-moe-sigmoid-routing`. The failure is silent: output reads well and
  exact answers are wrong, deterministically and at any precision.
- Do not use TP 8. With one KV head per rank the output was corrupt in both lazy and compile mode;
  TP 4 (two KV heads per rank) is correct.
- Do not go past 128K or raise `--max-num-batched-tokens` above 2,048. The Synapse device watchdog
  ("No progress") tracks chunk size times context: 2,048 x 130K passes, 2,048 x 212K and
  4,096 x 96K kill the engine on all four cards.
- Do not use step-1 decode block buckets (or the default prompt-context step of 1) with
  `VLLM_SKIP_WARMUP=true`: every new context length compiles a new graph. With step-1 buckets an
  8-turn agent session ran at 29 tokens/s and 3.0 s TTFT on new context sizes, against 46 tokens/s
  and 1.2 s when repeated at sizes already compiled; real sessions keep hitting new sizes.
