# Qwen3.6-35B-A3B-FP8

Qwen's 35B mixture-of-experts hybrid model (3B active, 256 experts, top 8; Gated DeltaNet plus
attention) with vision and reasoning, from the official FP8 checkpoint, at its native 256K context
on a single Gaudi 2 card. Prefix caching is off on this release (see
[Prefix caching](#prefix-caching)).

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

Prefix caching is off: on this release, prefix-cache hits on Gated DeltaNet layers resume from the
wrong state. Upstream vllm-gaudi main has merged prefix caching for the non-compact layout; use a
release that includes it before turning prefix caching on.

An earlier form of this recipe ran with prefix caching on, through a GDN prefix-cache patch that is
not in this cookbook. The checks marked "prefix caching on" below come from that configuration.
Checks of cache hits (hit-against-cold parity, needles and images served from the cache, agent
replay hit rates) do not apply and are left out. The decode measurements and the functional set
ran with prefix caching off; this exact command line has not been re-run end to end.

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
    fp8-moe-gather qwen3-strict-think-end || exit 1

vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.75 \
  --block-size 128 \
  --max-num-batched-tokens 1280 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
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
- `--no-enable-prefix-caching`: see [Prefix caching](#prefix-caching). `VLLM_COMPACT_GDN=1` is the
  stock default without it (GDN state in compact per-request slots); the recipe repeats it.
- `--max-num-batched-tokens 1280` is the value the checks ran with. Without prefix caching the
  chunks do not have to line up with the 1,152-token hybrid KV block; earlier deployments with
  prefix caching off used 1024.
- `--gpu-memory-utilization 0.75` and `--max-num-seqs 8` come from the 3.5 sibling, where lowering
  the utilization from 0.85 left room for the graphs of more concurrent sequences. They were not
  swept again on this checkpoint.

### Buckets

No bucket file. Batch buckets 1 to 8 for prompt and decode, prompt length 128 to 262,144 in steps
of 4,096 with exponential bucketing, decode blocks 128 to 16,384 in steps of 1,024.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0` (see [Using a recipe](index.md#using-a-recipe)). The reasons
are the same as on [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md#required-fixes), without
`structured-outputs-auto-whitespace`, plus:

| Fix | Why this model needs it |
| --- | --- |
| [`fp8-moe-gather`](../fixes/fp8-moe-gather.md) (opt-in) | gathered-expert FP8 MoE combine (vllm-gaudi PR #1731, merged for v0.30). On two one-card replicas of the 3.5 sibling: 3.85x per-stream decode alone, 1.41x at 8 in flight, 3.4x at 32K context, TTFT unchanged |
| [`qwen3-strict-think-end`](../fixes/qwen3-strict-think-end.md) (opt-in) | the model drafts a `<tool_call>` while thinking; v0.26.0's parser returned the draft as a call and then the real one (115/120 on tool reliability without the fix, 240/240 with it) |

This recipe does not set `disable_any_whitespace`: forced-tool whitespace runaway measured 0/32
here without it (prefix caching on).

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card. "Prefix caching on" marks the earlier configuration
described under [Prefix caching](#prefix-caching); the other flags were the same.

- Functional set (arithmetic, factual, code, lateral, multi-turn, streaming, tool calls): 10/10
  (prefix caching off).
- Gathered-expert MoE against the stock op: no output element beyond 2 FP8 ULPs.
- Needles to 250K tokens on cold requests: 9/9 (prefix caching on).
- Preemption and resume gate: 18/18 checks; the preempted request resumed and produced exactly its
  10,000 tokens (prefix caching on).
- Request options (`logit_bias`, `min_p`, `min_tokens`, structured output) all take effect;
  forced-tool whitespace runaway 0/32 (prefix caching on).
- Tool reliability 240/240 over two runs with strict think-end, 115/120 without; model test suite
  29 passed.

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

## Known issues

- Cold start under a CPU quota: loading took 78 minutes because vllm-gaudi sizes its load thread
  pool from the host's core count while every FP8 expert tensor (about 61,000) is rescaled on the
  CPU, and the pool is throttled by the quota (about 73 ms per tensor instead of 0.2 ms). With the
  pool capped at the quota the load took 46 to 49 s and the server was ready in under 2 minutes. Run
  without a CPU limit, or cap the pool.
- Weights from Hugging Face over Xet can stall silently for these repos; `HF_HUB_DISABLE_XET=1`
  avoids it when pre-fetching.

## What not to do

- Do not raise `--max-num-seqs` above 8 without raising `VLLM_HPU_MOE_GATHER_RATIO` with it (16
  sequences: 0.5), and re-measure. A decode batch of 9 or more otherwise falls back to the stock
  fused op, and every stream in it drops toward about 19 tokens/s.
- Do not use lazy mode for this model.
