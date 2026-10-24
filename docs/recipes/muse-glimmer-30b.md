# Muse Glimmer 30B

Meta's 29.6B dense vision-language agent model (52 layers, 3:1 sliding-window and global attention, a
50-layer vision encoder), served in BF16 at 128K context on two Gaudi 2 cards with up to eight images per
prompt. vLLM `ced6857a` already has the model and both parsers; a small port adds three model-only patches
that the stock `releases/v0.30.0` image needs on Gaudi.

| | |
| --- | --- |
| Model | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache 2.0, with Meta's usage policy) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | port image built from vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 (59.5 GB of weights, vision tower included) |
| Context | 131,072; KV pool 1,571,584 tokens |
| Concurrency | `--max-num-seqs 192`, `--api-server-count 4` |
| Parsers | tools and reasoning `muse_glimmer` (vLLM, with the port's grammar for forced calls) |
| Port | [`ports/muse-glimmer-30b/`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/muse-glimmer-30b/README.md) |
| Recipe | [`recipes/meta-models/Muse-Glimmer-30B/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/meta-models/Muse-Glimmer-30B/recipe.yaml) |

## Build the image

From a checkout of this repository, on top of a stock `releases/v0.30.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/muse-glimmer-30b/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-muse-glimmer-30b-image> .
```

`install.sh` checks the vllm-gaudi and vLLM commits and applies three patches, each fail-closed with
`--check`:

- **Attention output rank.** The model multiplies its sigmoid output gate, computed from the 3D
  `[batch, query, hidden]` layer input, by the HPU attention output, which comes back flattened as
  `[batch * query, hidden]`. Batch 1 broadcasts by accident, so single requests work and the first two
  concurrent requests stop the engine (`The size of tensor a (128) must match the size of tensor b (256)`).
  The patch restores the rank once, after the attention call.
- **Sliding-window blocks in chunked prefill.** vllm-gaudi builds the window block list for sliding layers
  only when `model_type` is `gemma4`; the patch adds `muse_glimmer`. Without it the 39 sliding layers read
  the whole context in every 8K prefill chunk. On the same image without the patch, a 120K-token prompt
  returned one word repeated to `max_tokens`, every later request on that server returned garbage, and the
  engine then failed with `PT_DEVMEM Allocation failed` (reproduced twice; not root-caused further).
- **Forced tool calls.** The `muse_glimmer` tool parser skips vLLM's JSON-schema path for `tool_choice`
  "required" and named functions (the model writes ATEM XML) and installs nothing instead, so those choices
  decode like "auto". The patch installs an xgrammar structural tag in the model's own channel format,
  applied from the first token.

The image carries this repository at `/opt/gaudi-cookbook`. Details: the
[port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/muse-glimmer-30b/README.md).

## Launch

Inside the port image, with `COOKBOOK=/opt/gaudi-cookbook`:

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
export VLLM_PROMPT_BS_BUCKET_MAX=16
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=192
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=131072
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=6144
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 topk-subset-sampler || exit 1

vllm serve meta-models/Muse-Glimmer-30B \
  --api-server-count 4 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 192 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --limit-mm-per-prompt '{"image":8,"video":0}'
```

- `--api-server-count 4`: on a vLLM 0.26 build of this model, one API server capped a pod at about 180
  streams.
- `--max-num-seqs 192` over-subscribes the KV pool for long prompts on purpose: the scheduler admits a
  sequence only when blocks are free and queues the rest.
- The two parsers work as a pair: the reasoning parser leaves a tool channel to the tool parser, and both
  need the channel markers that the tool parser keeps (`skip_special_tokens` false). Enable both or neither.
- The weights load in about 4 minutes from a cold page cache (9 s warm) at 0.85 memory utilization.

### Buckets

No bucket file. The ranges above bound the graph space: prompt batch 1 to 16, prompt length 128 to
131,072 in steps of 4,096, decode batch 1 to 192 (kept equal to `--max-num-seqs`), decode blocks 128 to
6,144, exponential bucketing. Warmup is skipped, so the first request of each new shape compiles.

## Required fixes

The `v0.30.0` profile plus one opt-in fix. The ones this configuration depends on:

| Fix | Why this model needs it |
| --- | --- |
| [topk-subset-sampler](../fixes/topk-subset-sampler.md) (opt-in) | `generation_config.json` sets `top_k` 64 and `top_p` 0.95, so every request that does not override them sorted the 202,048-entry vocabulary on every decode step |
| [seeded-sampling](../fixes/seeded-sampling.md) | requests with `seed` can segfault a tensor-parallel worker |
| [gather-logprobs-int32](../fixes/gather-logprobs-int32.md), [logprobs-decode-guard](../fixes/logprobs-decode-guard.md) | one `logprobs` request could stop the API server |
| [prompt-logprobs](../fixes/prompt-logprobs.md) | multimodal models return flat prefill states, which the stock runner indexes as 3D |
| [mm-merge-idxcopy](../fixes/mm-merge-idxcopy.md) | multi-image steps fail to graph-compile |
| [structured-output-rows](../fixes/structured-output-rows.md), [so-row-owners](../fixes/so-row-owners.md) | forced tool calls are grammar-constrained, so their rows must keep their own parameters |
| [prompt-order](../fixes/prompt-order.md), [sampling-rows-refresh](../fixes/sampling-rows-refresh.md), [bad-words-index](../fixes/bad-words-index.md), [logitsprocs-rows](../fixes/logitsprocs-rows.md), [async-inputs](../fixes/async-inputs.md), [rank-exception-fatal](../fixes/rank-exception-fatal.md) | generic request and engine fixes from the profile |

## Client usage

- The model card's sampling is temperature 1.0, top_p 0.95, top_k 64; these are the server defaults.
- The model reasons in a separate channel before answering, even for one-word answers. Give it room: a
  reply cut off in the reasoning channel has `content: null`.
- `tool_choice` "required" and named functions are enforced; the model may still reason first, which
  costs about 150 to 300 tokens per forced call.

## Verified

On the port image path exactly as written (stock `releases/v0.30.0` image, `install.sh`, the fix step,
the serve command), two Gaudi 2 cards, `ports/muse-glimmer-30b/tests/hpu/run_gates.sh`: 11/11 PASS.

| Gate | Result |
| --- | --- |
| Smoke | `Paris`; `157*23+89` = 3700 |
| Concurrency | 32 concurrent requests, 32/32 correct (stock v0.30.0: the engine stops on the first two) |
| Needles | 8,078 tokens; 32,078 at 5% and 95% depth; 120,086; 127,070 at 95% depth: 5/5 |
| Vision | three shapes counted, "GAUDI 4729" read; 8 image sizes from 448x448 to 1568x1008 read 8/8 |
| Text after images | `Tokyo` (no image state carried into a text-only request) |
| Forced tools | "required" 24/24 (8 greedy, 4 at a time; 16 sampled at the defaults), named call with `{"city": "Rome", "unit": "celsius"}`, "auto" answers in prose. Same image without the grammar patch: 0/8 greedy |
| Streaming tools | "required" streamed: one call with a string `id`, name and parseable arguments |
| Logprobs | `logprobs: 5` on completions 3/3 with no bad ids and the sampled token top-1 at every position; chat `top_logprobs: 5` |
| Seeded traffic | 128 seeded requests at 64 concurrent, 0 errors; same-seed repeats identical |
| Reasoning split | content holds the answer, no channel markers |
| Health afterwards | 200 |

## Performance

One server, two cards, the Launch command above, synthetic chat prompts of about 285 tokens, 512 output
tokens per request (`ignore_eos`), sampling at the generation-config defaults, measured after a warm-up
round. Per stream is output tokens over each request's wall time, first token included. Single runs.

| Streams | Aggregate output | Per stream |
| --- | --- | --- |
| 1 | 50 tok/s | 50.1 tok/s |
| 64 | 1,955 tok/s | 30.6 tok/s |
| 192 | 3,362 tok/s | 17.6 tok/s |

## Known issues

- `response_format` with a JSON schema is not enforced: the grammar waits for the reasoning parser's end
  signal, which a plain answer never sends. Asked for `{name, year: integer}`, the model returned a fenced
  block with other keys and `"753 BC"` as a string.
- `tool_choice: "none"` keeps the tool schemas in the prompt and runs no parser, so a model that calls a
  tool anyway returns its raw ATEM markup in `content`.
- DFlash speculative decoding (the model's draft head in vLLM) was not tried on Gaudi.

## What not to do

- Do not drop the port and serve the stock image: one concurrent request pair stops the engine.
- Do not lower `--api-server-count` to 1 or drop `topk-subset-sampler` at high concurrency. On a vLLM 0.26
  build of this model each one capped a pod's aggregate throughput (the single frontend process, and the
  full-vocabulary sort at the default `top_k`); neither was re-measured on this image.
- Do not enable only one of the two `muse_glimmer` parsers.
