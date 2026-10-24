# Qwen3-VL-32B-Instruct-FP8

Qwen's 32B dense vision-language model (images and video in, text out) in FP8, served at 128K
context on two Gaudi 2 cards with up to 16 images and 2 videos per prompt.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | FP8, 128x128 weight blocks (35.5 GB of weights; vision tower and `lm_head` in BF16) |
| Context | 131,072 served (262,144 native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `hermes` |
| Recipe | [`recipes/Qwen/Qwen3-VL-32B-Instruct-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-VL-32B-Instruct-FP8/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_SKIP_WARMUP=true
export VLLM_MULTIMODAL_BUCKETS=None
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0.5
export VLLM_HPU_MM_ENCODER_MAX_PATCHES=65536
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 mm-encoder-budget || exit 1

vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.55 \
  --max-num-batched-tokens 1024 \
  --max-cudagraph-capture-size 128 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"image":16,"video":2}'
```

- `--max-cudagraph-capture-size 128` is the main memory setting. vllm-gaudi captures an HPU graph
  for every prefill chunk whose query plus context fits under this size (the default is
  `--max-num-batched-tokens`), and with the graph tensor cache on, each captured prefill graph keeps
  all of its activations, never released (about 12 MiB per query token, measured on an older
  vllm-gaudi release: 3 GiB for a 256-token chunk, 9 GiB for 768). Varied prompt lengths walk the
  card into `PT_DEVMEM Allocation failed`; on this image ten novel prompt lengths took it from 44.6
  to 90.6 GB at the default. At 128 only the smallest prefill shape is captured; decode stays
  graphed.
- `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false` keeps the graph tensor cache on, as vllm-gaudi's
  configuration guide asks for Qwen vision models. With it off, image requests crashed on a freed
  graph input (`ValidateSyncInputTensors ... tensor_data is empty`) on older releases.
- `--gpu-memory-utilization 0.55` leaves room outside vLLM's KV budget for the vision tower's
  transient memory and long prefills, which vLLM's memory profiling does not account for. It still
  fits one full 128K sequence (KV pool 208,768 tokens).
- `VLLM_HPU_MM_ENCODER_MAX_PATCHES=65536` is read by the `mm-encoder-budget` fix: an encoder call
  holds at most one maximum-size image (16,384 tokens) instead of every pending image of every
  scheduled request.
- A maximum-size image costs 16,384 tokens, so 8 such images fill 131K; `{"image":16}` relies on
  real images being smaller, and vLLM rejects over-length prompts cleanly.

### Buckets

No bucket file. Default bucketing, `VLLM_MULTIMODAL_BUCKETS=None` (no fixed vision shapes),
`VLLM_GRAPH_RESERVED_MEM=0.4` with `VLLM_GRAPH_PROMPT_RATIO=0.5` for the long-prefill graph pool.

## Required fixes

All from [Fixes](../fixes/index.md). `--profile v0.30.0` applies the release's generic fixes;
the opt-in fix is named in the launch command.

| Fix | Why this model needs it |
| --- | --- |
| [`mm-merge-idxcopy`](../fixes/mm-merge-idxcopy.md) | the multimodal embedding merge uses a broadcast `masked_scatter_` that fails to compile when several images arrive in one step |
| [`mm-encoder-budget`](../fixes/mm-encoder-budget.md) (opt-in) | vllm-gaudi sends every pending image of every scheduled request through the vision tower in one call; a 16-image request asks for tens of GiB at once |

## Verified

On vllm-gaudi `releases/v0.30.0`, two Gaudi 2 cards, the launch above run as written (KV pool 208,768
tokens, 1.59x concurrency at 131,072). Every gate passed on two separate pods.

| Check | Result |
| --- | --- |
| Text: "capital of France", 157 x 23 + 89 worked out | Paris; 3700 |
| Vision: read an arbitrary number from a rendered image; three coloured shapes | "4729"; red circle, blue square, green triangle |
| Text-only request right after an image request, 3 rounds | 3/3 correct (no image state leaks into the next request) |
| 16 images of 2048x2048 in one request | answered "16" |
| 7 images of 4096x4096 (114,719 prompt tokens) in one request | answered "7"; peak device memory 87,580 and 91,345 MiB with `mm-encoder-budget` (two pods), 95,055 MiB without (one run, same pod as the first) |
| 16 mixed text and image requests at 8 concurrent | 16/16 correct |
| Tool call (`hermes`) | `get_weather` with `{"city": "Rome"}` |
| `top_logprobs` and `prompt_logprobs` | both returned, no bad ids |
| Needle at about 120K tokens, middle of the prompt | found |
| Ten novel prompt lengths (150 to 990 words) | device memory flat (57,572 to 56,887 MiB; 58,811 to 56,733 MiB on a second pod). With the default capture size on the same pod as the first: 44,619 to 90,637 MiB |

## Performance

Single measurements on two cards, `releases/v0.30.0`, warm; short chat prompt, 256 output tokens
with `ignore_eos`, temperature 0.7. Per-stream speed is decode only (after the first token).

| Streams | Tokens/s per stream | Tokens/s aggregate | Mean time to first token |
| --- | --- | --- | --- |
| 1 | 30.6 to 33.3 | 31 to 33 | 0.04 s |
| 4 | 27.4 to 29.2 | 107 to 114 | 0.15 to 0.16 s |
| 8 | 26.7 to 28.1 | 206 to 217 | 0.29 to 0.30 s |

Ranges are two runs on two pods (two nodes) with the same launch.

A 120K-token prompt took 194 s end to end (prefill in 1,024-token chunks).

## Known issues

- Video is enabled (`"video":2`) but was not part of the checks below.

## What not to do

- Do not set `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1`: with the graph tensor cache off, an image request
  stops the engine within a few requests (seen on older releases; not re-run on this image).
- Do not raise `--max-cudagraph-capture-size` back to the default for speed: each captured prefill
  shape keeps several GiB for the life of the server.
- Do not raise `--gpu-memory-utilization` much above 0.55: the encoder and long-prefill peaks sit
  outside the KV budget.
