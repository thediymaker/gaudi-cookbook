# Qwen3.6-35B-A3B-FP8

Qwen's 35B mixture-of-experts hybrid model (3B active, 256 experts, top 8; Gated DeltaNet plus
attention) with vision and reasoning, from the official FP8 checkpoint, at its native 256K context
on a single Gaudi 2 card, with prefix caching on through the
[`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen3.6-35B-A3B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3.6-35B-A3B-FP8/recipe.yaml) |

The checkpoint declares the same architecture class as Qwen3.5-35B-A3B-FP8
(`Qwen3_5MoeForConditionalGeneration`), which serves with the same recipe.

## Prefix caching

On the stock release, prefix-cache hits on the Gated DeltaNet layers resume from the wrong state,
so prefix caching is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md), which
stores GDN state checkpoints in the attention pages of the KV blocks the GDN groups own and keeps
the compact GDN slots. See [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md#prefix-caching) and
[Prefix caching on Gated DeltaNet hybrids](../results/gdn-prefix-caching.md) for the design.

On this model the hybrid KV block is 1,152 tokens; prefill chunks end on block boundaries, so each
1,280-token batch carries one block, and prompts shorter than 1,152 tokens never hit. The start-up
log:

```text
GPU KV cache size: 1,581,826 tokens
[gaudi-fix:gdn-prefix-cache] active: GDN groups [0, 1, 2], block 1152 tokens, 1412 KV blocks, 10 layer positions; ...
[gaudi-fix:gdn-prefix-cache] scheduler: GDN prefill chunks end on 1152-token blocks
```

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_HPU_MOE_GATHER=1
export VLLM_HPU_MOE_GATHER_RATIO=0.25
export VLLM_HPU_MOE_GATHER_VERIFY=0
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
export VLLM_DECODE_BS_BUCKET_MAX=8
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
    gdn-prefix-cache fp8-moe-gather qwen3-strict-think-end || exit 1

vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.75 \
  --block-size 128 \
  --max-num-batched-tokens 1280 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":16,"video":0}'
```

- Compile mode (`PT_HPU_LAZY_MODE=0`); the GDN layers lose their state cache at the first decode in
  lazy mode.
- `VLLM_HPU_MOE_GATHER=1` with `fp8-moe-gather`: a decode step gathers only the experts its tokens
  route to (8 of 256 per token) instead of running all 256. The gather is used while
  `tokens x top_k <= local experts x VLLM_HPU_MOE_GATHER_RATIO`; with 256 experts on one card, 0.25
  gives 64, which covers up to 8 decode tokens, that is `--max-num-seqs 8`.
- `--enable-prefix-caching` is correct here only with `gdn-prefix-cache`; `VLLM_COMPACT_GDN=1`
  keeps the GDN state in compact per-request slots, and the fix refuses to start without it.
- `--max-num-batched-tokens 1280`: with prefix caching on, a GDN prefill chunk ends on a 1,152-token
  block boundary, so a prefill step carries 1,152 prompt tokens of one request.
- `--gpu-memory-utilization 0.75` and `--max-num-seqs 8` come from the 3.5 sibling, where lowering
  the utilization from 0.85 left room for the graphs of more concurrent sequences. They were not
  swept again on this checkpoint.

### Buckets

No bucket file. Batch buckets 1 to 8 for prompt and decode, prompt length 128 to 262,144 in steps
of 4,096 with exponential bucketing, decode blocks 128 to 16,384 in steps of 1,024.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0` (see [Using a recipe](index.md#using-a-recipe)). The reasons
are the same as on [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md#required-fixes) (including `prompt-order`,
`resumed-prefill-logits`, `hybrid-decode-pad` and the opt-in `gdn-prefix-cache`), without
`structured-outputs-auto-whitespace`, plus:

| Fix | Why this model needs it |
| --- | --- |
| [`fp8-moe-gather`](../fixes/fp8-moe-gather.md) (opt-in) | gathered-expert FP8 MoE combine (vllm-gaudi PR #1731, merged for v0.30). On two one-card replicas of the 3.5 sibling: 3.85x per-stream decode alone, 1.41x at 8 in flight, 3.4x at 32K context, TTFT unchanged |
| [`qwen3-strict-think-end`](../fixes/qwen3-strict-think-end.md) (opt-in) | the model drafts a `<tool_call>` while thinking; v0.26.0's parser returned the draft as a call and then the real one (115/120 on tool reliability without the fix, 240/240 with it) |

This recipe does not set `disable_any_whitespace`: forced-tool whitespace runaway measured 0/32
here without it (prefix caching on).

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card, the launch above as written (prefix caching on with
`gdn-prefix-cache`):

- Greedy parity, 32 prompts of 5K to 31K tokens (single questions, multi-turn agent sessions with
  tools, extensions, diverging prefixes; thinking on and off), the same request served from the
  cache and cold: 32/32 identical, all with bit-identical logprobs. Two cold runs: 32/32 identical.
- Cross-request isolation, 24 rounds with an unrelated request between fill and hit: 0 leaks.
- Images served from the cache, 8 cases: hit equals cold 8/8.
- Needles at 65K, 130K and 250K tokens, three depths each: 9/9 cold, 9/9 from the cache, hit
  equals cold 9/9.
- 16 images in one prompt: 16/16 colours named in order (thinking off), a 17th image rejected with
  a clean 400.
- Requests sharing a prefix in the same step, 24: 24/24 found the right code. 11 of 24 were worded
  differently from the same request sent alone, including 2 that had no cache hit: on this MoE the
  batch a request decodes in changes its greedy wording. With prefix caching off the same test
  words 16 of 24 differently, so the cache adds nothing to it.
- Forced tool calls (`tool_choice` required): whitespace runaway 0/32.
- Tool reliability, 3 cases x 40 sampled requests: 120/120 with strict think-end.
- `prompt_logprobs`: ids aligned with the prompt and in vocabulary, values consistent with decode
  logprobs (plp check PASS).
- Agent replay, 4 sessions growing from 8K to 64K tokens: 0 preemptions, 0 engine restarts across
  all of the above.

From earlier runs of this recipe with prefix caching off (the decode path is the same):

- Functional set (arithmetic, factual, code, lateral, multi-turn, streaming, tool calls): 10/10.
- Gathered-expert MoE against the stock op: no output element beyond 2 FP8 ULPs.
- Tool reliability 240/240 over two runs with strict think-end, 115/120 without.

## Performance

Single measurements on one card, vllm-gaudi v0.26.0 with `fp8-moe-gather` (same gate as above),
prefix caching off, every batch size warmed first, 256 output tokens per stream, `ignore_eos`, two
repetitions.

| In flight | Tokens/s per stream | Tokens/s per card |
| --- | --- | --- |
| 1 | 74.5 | 71.9 |
| 2 | 48.8 / 50.2 | 93.4 / 96.0 |
| 4 | 42.5 | 162.0 |
| 6 | 30.8 / 32.4 | 177.9 / 186.9 |
| 8 | 26.5 | 204.8 |

The curve has not flattened at 8; see "What not to do" before raising the cap.

Needle requests on the launch above (prefix caching on), the same pod, each prompt sent cold and
then again from the cache:

| Prompt | Cold TTFT | TTFT from the cache | Measured as |
| --- | --- | --- | --- |
| 65K tokens | 35 to 53 s | 0.41 to 0.46 s | same pod, three depths |
| 130K tokens | 46 to 80 s | 0.55 to 0.62 s | same pod, three depths |
| 260K tokens | 134 to 167 s | 0.71 to 1.29 s | same pod, three depths |

The cold TTFTs vary by depth and run order (first-touch graph compiles in compile mode); treat them
as a range, not a curve.

Agent replay, 4 sessions growing from 8K to 64K tokens, 200 output tokens per turn, 4 in flight:

| Test | Prefix caching on (this recipe) | Prefix caching off | Measured as |
| --- | --- | --- | --- |
| Wall time, 36 requests | 130.5 s | 607.5 s | same pod and card, the off arm launched with `--no-enable-prefix-caching` |
| Later-turn TTFT mean / p50 / p90 | 4.1 / 3.7 / 4.8 s | 25.0 / 18.3 / 59.2 s | same run |
| First-turn TTFT mean | 4.6 s | 59.8 s | same run |
| Prompt tokens per second | 9,609 | 2,064 | same run |
| Server prefix-cache hit rate | 78% | | same run |

## Known issues

- Prompts shorter than one 1,152-token block never hit the cache.
- Cold load on the test pod (no CPU limit, weights on a network file system) took about 19 minutes
  from start to a ready KV cache.
- Cold start under a CPU quota: loading took 78 minutes because vllm-gaudi sizes its load thread
  pool from the host's core count while every FP8 expert tensor (about 61,000) is rescaled on the
  CPU, and the pool is throttled by the quota (about 73 ms per tensor instead of 0.2 ms). With the
  pool capped at the quota the load took 46 to 49 s and the server was ready in under 2 minutes. Run
  without a CPU limit, or cap the pool.
- Weights from Hugging Face over Xet can stall silently for these repos; `HF_HUB_DISABLE_XET=1`
  avoids it when pre-fetching.

## What not to do

- Do not turn on prefix caching on this release without `gdn-prefix-cache`: hits then resume the
  GDN layers from a wrong state.
- Do not raise `--max-num-seqs` above 8 without raising `VLLM_HPU_MOE_GATHER_RATIO` with it (16
  sequences: 0.5), and re-measure. A decode batch of 9 or more otherwise falls back to the stock
  fused op, and every stream in it drops toward about 19 tokens/s.
- Do not use lazy mode for this model.
