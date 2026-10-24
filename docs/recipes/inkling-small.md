# Inkling-Small

Thinking Machines' 276B-parameter mixture-of-experts model (12B active), served text only in BF16 at 256K context on
one 8-card Gaudi 2 node. vllm-gaudi has no path for it, so it runs from the [Inkling-Small port](../ports/inkling-small.md),
built on top of the stock `releases/v0.30.0` image.

| | |
| --- | --- |
| Model | [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) (Apache-2.0) |
| Hardware | 8x Gaudi 2 (TP 8), one node |
| Image | port image built from vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 (532 GB of weights) |
| Context | 262,144 served (KV pool 327,680 tokens) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools and reasoning `inkling_py` (from the port) |
| Recipe | [`recipes/thinkingmachines/Inkling-Small/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/thinkingmachines/Inkling-Small/recipe.yaml) |

## Build the image

From a checkout of this repository, on top of a stock `releases/v0.30.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/inkling-small/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-inkling-small-image> .
```

`install.sh` checks the vllm-gaudi and vLLM commits, installs the `inkling_hpu` plugin and patches the runner's decode
window. It stops on any other base. The image carries this repository at `/opt/gaudi-cookbook`.

## Launch

Inside the port image, with `COOKBOOK=/opt/gaudi-cookbook`:

```bash
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=true
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_CONTIGUOUS_PA=false
export VLLM_GRAPH_RESERVED_MEM=0.4
export INKLING_PREFILL_BIAS_BUDGET=805306368
export INKLING_DECODE_WINDOW_PLUS1=1
export INKLING_MOE_SINGLE_REDUCE=1
export INKLING_LOCAL_WINDOW_PREFILL=1
export INKLING_STEP_HOIST=1
export VLLM_LONG_PREFILL_SHARE=0.5
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=32768
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=5400

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 vllm-long-prefill-share || exit 1

vllm serve thinkingmachines/Inkling-Small \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --num-gpu-blocks-override 2560 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --block-size 128 \
  --max-cudagraph-capture-size 2048 \
  --tokenizer-mode hf \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser inkling_py \
  --reasoning-parser inkling_py
```

- `VLLM_USE_V2_MODEL_RUNNER=0`: vLLM picks its V2 model runner for Inkling by default; the HPU runner is V1.
- `VLLM_CONTIGUOUS_PA=false`: the conv-state slots in decode need each request's block list in logical order. The
  model refuses to start with contiguous PA.
- `--num-gpu-blocks-override 2560` pins the KV pool at 327,680 tokens (1.25 full-length requests). The full budget
  gives 3,135 blocks, but 256K prefills then leave only about 2.7 GiB of HBM per card; at 2,560 blocks it is
  4.4 to 5.6 GiB, at the same speed.
- `--gpu-memory-utilization 0.75`, `VLLM_GRAPH_RESERVED_MEM=0.4` and `INKLING_PREFILL_BIAS_BUDGET` (768 MiB per prefill
  bias sub-block) are one memory guard: loosening any one of them brought back device out-of-memory errors during
  graph replay at around 100K tokens.
- `--tokenizer-mode hf` makes vLLM use the checkpoint's `chat_template.jinja`.
- `--block-size 128` is required: the conv-state snapshots are per 128-token block.
- The four `INKLING_*` switches are explained in the [port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/inkling-small/README.md#switches).
  The long-prefill share keeps other streams decoding while a prompt of 32K tokens or more prefills.

## Required fixes

The v0.30.0 profile covers the generic fixes this configuration depends on: prompt-order, sampling-rows-refresh,
structured-output-rows, so-row-owners, bad-words-index, logitsprocs-rows, async-inputs, resumed-prefill-logits,
gather-logprobs-int32, prompt-logprobs, logprobs-decode-guard, seeded-sampling and rank-exception-fatal. The opt-in
`vllm-long-prefill-share` adds the scheduler's long-prefill time share. The port's own runner patch is applied at
image build, not here.

## Verified

Run on one 8-card Gaudi 2 node: a stock `releases/v0.30.0` image (`058f107`, vLLM `ced6857a`), the port's
`install.sh` run as the Dockerfile runs it (`install OK`), then the Launch block above unchanged. The fix step
reported every fix of the v0.30.0 profile and `vllm-long-prefill-share` as patched. At startup the port logged one
KV group (`FullAttentionSpecx42`) and `KV pool: 2560 blocks = 327680 tokens = 1.25 x max_model_len 262144`.

`ports/inkling-small/tests/hpu/run_gates.sh quick`:

| Gate | Result |
| --- | --- |
| Preflight: plugin, parsers, KV layout and the CPU tests inside the image | 21 passed, 0 skipped |
| Smoke: arithmetic, tool call, JSON mode, `tool_choice` required | pass |
| Greedy repeats, reasoning split, streamed reasoning and tool call without block markers | pass |
| Needles at both ends of 3K and 6K prompts (2 and 3 prefill chunks) | 4/4 |
| Needles at 8K, 32K and 131K, depths 0.1 / 0.5 / 0.9 | 9/9 |
| 200 logprobs requests at concurrency 8 | 200/200 clean |
| Prefix-cache hit vs cold output, 14 prompts (gate: every hit recalls its code, and any split from the cold run sits at a near-tie) | every hit recalled its code; 9 to 11 of 14 byte-identical over three runs, every split at a near-tie. The gate first required byte-identical output and failed on the first run; it was changed after that run, because repeats with no cache hit also split at near-ties on this model |
| 16 concurrent 4K needles | 16/16 |
| A 264,111-token prompt | HTTP 400 in 0.4 s |
| Server log: one KV group, the pool line, decode window of 5 blocks | pass |

A 256,796-token prompt (under the limit) was served, in 154 s for 8 output tokens.

`ports/inkling-small/tests/hpu/run_gates.sh long`, the same server:

| Gate | Result |
| --- | --- |
| Needles at 200K and 250K, depths 0.1 / 0.5 / 0.9 | 6/6 (TTFT about 107 s and 145 s) |
| Three 100K-token prompts resident at once | all three ran together, 0 preemptions |
| A 250K prompt while 16 streams decode (long-prefill share 0.5) | needle found, 168 s for the 250K request; the streams ran at 28.3 tok/s before and 34.7 during |

## Performance

From the run above, one 8-card node, BF16. Single runs; do not carry them to other hardware or flags.

| What | Result | Measured as |
| --- | --- | --- |
| Weight load from network storage | 4,079 s (server ready 73 min after launch) | single cold start |
| Request time, needle prompt + 24 output tokens, at 8K / 32K / 131K | 3 / 12 / 50 s | first request at each length, graphs already compiled by an earlier pass |
| Same, first use of each length | 23 / 30 / 53 s (the last prompt was 127.6K tokens) | first pass after startup, includes graph compiles |
| 16 concurrent 4K needle requests | 24 s for all 16 | second pass, warm |
| 256,796-token prompt, 8 output tokens | 154 s | single request, first use of that length |

More numbers from the port's development (decode per stream, switch A/Bs, long-prefill share) are on the
[port page](../ports/inkling-small.md#performance), each with how it was measured.

## Known issues

- Cold start is dominated by reading 532 GB of weights; from network storage it took 35 to 75 minutes.
- New prompt lengths compile graphs on first use (skip-warmup): the first 32K request took 23.5 s against 9.7 s warm.
- Text only. Image and audio inputs are not supported; the towers are not built.

## What not to do

- Do not turn on contiguous PA, merged prefill, an FP8 KV cache, speculative decoding or a block size other than
  128: the model refuses to start, because the conv-state snapshots cannot serve them.
- Do not raise `--gpu-memory-utilization` or lower `VLLM_GRAPH_RESERVED_MEM` to grow the KV pool; see the memory
  guard above.
- Do not use vLLM's built-in `inkling` parsers on Gaudi: they need a Rust extension that the Gaudi build does not
  have, and fail on the first chat request.
