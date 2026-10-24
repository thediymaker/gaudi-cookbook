# Qwen3-Coder-30B-A3B-Instruct

Qwen's 30B mixture-of-experts coding model (3.3B active) with XML-style tool calls, served in BF16
at 128K context on a single Gaudi 2 card.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (61 GB of weights) |
| Context | 131,072 served (262,144 native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `qwen3_coder` |
| Recipe | [`recipes/Qwen/Qwen3-Coder-30B-A3B-Instruct/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-Coder-30B-A3B-Instruct/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0.5
export VLLM_GRAPH_PROMPT_STRATEGY=max_bs
export VLLM_GRAPH_DECODE_STRATEGY=max_bs
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=8
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=131072
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=32768
export OMP_NUM_THREADS=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 || exit 1

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --max-cudagraph-capture-size 2048 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

`--max-cudagraph-capture-size 2048` makes prefill steps above 2,048 tokens run without capturing an
HPU graph, which keeps graph memory bounded with `VLLM_SKIP_WARMUP=true`.

### Buckets

No bucket file. Batch buckets 1 to 8, prompt length 128 to 131,072 in steps of 4,096 with
exponential bucketing, decode blocks 128 to 32,768 in steps of 1,024.

## Required fixes

The same set as [Qwen3-30B-A3B-2507](qwen3-30b-a3b-2507.md#required-fixes) (same architecture and
image), all from [Fixes](../fixes/index.md). No tool-parser fix is needed on v0.26.0: the
engine-based `qwen3_coder` parser already recovers a call whose opening `<tool_call>` the model
never wrote, which needed a plugin on older images.

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card, the flags above (all prompts synthetic):

- Long-prompt ladder from 38,656 to 130,900 tokens, greedy and default sampling with tools: every
  needle found.
- 38-minute soak with 8 coding and tool workers plus a 2K to 131K ladder worker: 90 requests,
  needles 37/37, tool calls 22/22 with valid arguments, 0 markup leaks, 0 garbage outputs.
- Streamed tool calls 15/15 clean; replaying bare `<function=...>` outputs token by token gives the
  right calls with no leaked markup.
- Against the previous image at temperature 0: 6/6 coding answers and 6/6 tool calls
  byte-identical; `top_logprobs=20` under load 20/20 well-formed; repetition-loop rate 0/10.
- 75 minutes and 303 requests: 0 restarts, 0 errors, 0 preemptions; HBM held at 88 to 91 GiB of 96.

## Performance

Single measurements on one card, vllm-gaudi v0.26.0.

| Load | Result |
| --- | --- |
| 1 stream, short prompt | TTFT 0.15 s |
| 16 clients (8 running), 5.4K-token prompts, 545-token outputs | 200.1 tokens/s per card, TTFT 20.9 s (queueing included), ITL 35.4 ms |
| Long prefill sharing the card with decodes | about 2,100 prompt tokens/s at 38K context, about 780 at 131K |

Decode speed per stream matches the [Qwen3-30B-A3B](qwen3-30b-a3b-2507.md#performance) table
(same architecture).

## Known issues

- Long prefills share the single card with running decodes (head-of-line blocking): a 131K prompt
  slows every other stream while it prefills. That is capacity, not stability; scale out with
  replicas.
- At a 55K-token prompt one verbatim-copy task came back as an unrelated reasoning block. Not
  reproduced or explained; treat very long verbatim copies with care.
- Penalties are not applied under async scheduling on v0.26.0 (see
  [Qwen3-30B-A3B-2507](qwen3-30b-a3b-2507.md#known-issues)).

## What not to do

- Do not raise `--max-num-batched-tokens` to 8,192. Exponential bucketing pads a chunk to 8,192
  query tokens against a 481-block context bucket (8,192 x 69,760 elements), and that tile corrupts
  the forward in lazy mode. A 38,656-token prompt is fine and a 38,657-token prompt returns `!`
  (token 0) with greedy decoding, or kills the engine with sampling. 2,048 x 131,072 = 2^28 stays
  at half the smallest failing size.
- Do not try 4,096 without walking the long-prompt ladder again: 4,096 x 131,072 is exactly 2^29.
  A quick test for any setting is a needle prompt whose final chunk lands just above a query-bucket
  boundary at a large context bucket.
