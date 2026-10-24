# GLM-5.2 (MXFP4)

Z.ai's GLM-5.2 (744B MoE, about 41B active, DeepSeek sparse attention, one MTP layer) from AMD's MXFP4 release, on
one 8-card Gaudi 2 node at 128K context with MTP speculative decoding. vllm-gaudi cannot load AMD's Quark layout
(one tensor per expert), so the [glm-5-mxfp4 port](../ports/glm-5-mxfp4.md) repacks it byte for byte and adds what
vllm-gaudi lacks (fused MXFP4 loading for models other than gpt-oss, top-k sparse attention, MTP on the HPU runner,
MXFP4 MoE kernels). Same architecture and port as [GLM-5.3](glm-5-3-mxfp4.md); the command differs in the default
sampling and reasoning settings and has no expert placement file.

| | |
| --- | --- |
| Model | [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) (AMD Quark MXFP4 of [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)), repacked by the port |
| Hardware | 8x Gaudi 2 (TP 8, `--enable-expert-parallel`, experts laid out EP 4 x TP 2) |
| Image | the glm-5-mxfp4 port image on vllm-gaudi `releases/v0.30.0` at `058f107`, vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | routed experts MXFP4 (E2M1 with an E8M0 scale per 32; AMD's bytes, and the MTP layer's experts encoded by the port), everything else BF16: 399 GiB |
| Context | 131,072 served (1,048,576 native) |
| Concurrency | `--max-num-seqs 4` |
| Parsers | reasoning and tools `glm47` |
| Licence | MIT |
| Recipe | [`recipes/amd/GLM-5.2-MXFP4/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/amd/GLM-5.2-MXFP4/recipe.yaml) |

## Convert the checkpoint

Download `amd/GLM-5.2-MXFP4` and repack it once, on any machine with Python, torch and safetensors (CPU only). The
commands are in
[`ports/glm-5-mxfp4/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/glm-5-mxfp4/README.md#1-convert-the-checkpoint).
The routed experts are copied byte for byte into the fused layout (`[gate; up]` concatenated, experts stacked), the
shared experts become BF16, and the MTP layer's experts, which AMD left in BF16, are encoded to MXFP4. The result
(399 GiB) goes in `$CHECKPOINT_DIR`.

## Build the image

From a checkout of this repository, on top of your stock `releases/v0.30.0` image (built as in
[Getting started](../getting-started.md)):

```bash
docker build -f ports/glm-5-mxfp4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-glm-5-mxfp4-image> .
```

The build copies the repository to `/opt/gaudi-cookbook` (so `$COOKBOOK` is set there), checks the image commits,
applies the fixes of the Launch block, installs the `glm5_hpu` package, applies the port's 14 patches and builds the
MXFP4 MoE kernels (`kernels/mxfp4-moe/build.sh --variant glm`). Details:
[`ports/glm-5-mxfp4/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/glm-5-mxfp4/README.md).

## Launch

Inside the port image (`$COOKBOOK` is `/opt/gaudi-cookbook` there).

```bash
export CHECKPOINT_DIR=/models/GLM-5.2-MXFP4-fused
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=true
export VLLM_BUCKETING_FROM_FILE=/opt/gaudi-cookbook/ports/glm-5-mxfp4/serving/tp8-seqs4-128k.txt
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=2048
export PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_HPU_FSDPA_SOFTMAX_MODE=None
export VLLM_LONG_PREFILL_SHARE=0.5
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=32768
export GLM_DSA_SPARSE=1
export GLM_MXK=1
export GLM_MXK_PREFILL=1
export GLM_MOE_TP=2
export GLM_ROUTER_FAST=1
export GLM_EARLY_DRAFT=1
export GLM_SPEC_DEVICE_REJECT=1

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 \
    vllm-long-prefill-share glm47-streaming-text-markers || exit 1

vllm serve $CHECKPOINT_DIR \
  --served-model-name amd/GLM-5.2-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.86 \
  --max-cudagraph-capture-size 2048 \
  --trust-remote-code \
  --enable-prompt-tokens-details \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort": "high"}' \
  --override-generation-config '{"temperature": 0.8}'
```

- `--enable-expert-parallel` is required: the fused MXFP4 experts are loaded by expert group.
- `GLM_MOE_TP=2` lays the experts out as 4 groups of 2 cards (every card holds 64 experts x half the intermediate
  dimension), which evens out the busiest card of every decode MoE layer. GLM-5.2 runs with contiguous expert
  groups; the placement file of the GLM-5.3 recipe is fitted to GLM-5.3 routing.
- `GLM_MXK=1` runs decode-sized MoE calls on the MXFP4 kernels; `GLM_MXK_PREFILL=1` unpacks the experts of 256- to
  4096-token calls with a faster kernel (output bit-identical to the stock op).
- `GLM_DSA_SPARSE=1` attends to each query's top 2,048 keys (the model's DSA) once a sequence has more than 32K
  keys. Below that, dense attention with the accurate softmax (`VLLM_HPU_FSDPA_SOFTMAX_MODE=None`) is exact.
- `GLM_ROUTER_FAST`, `GLM_EARLY_DRAFT` and `GLM_SPEC_DEVICE_REJECT` take host work off the decode step; greedy
  output is the same with them off.
- `VLLM_LONG_PREFILL_SHARE=0.5` with `--long-prefill-token-threshold 4096` lets prompts above 32K tokens take at most
  half of the time while other requests decode ([vllm-long-prefill-share](../fixes/vllm-long-prefill-share.md)).
- `--gpu-memory-utilization 0.86`: the DSA indexer-key buffers (one per full indexer layer, sized like the MLA cache)
  are allocated after vLLM's memory profiling.
- `--max-cudagraph-capture-size 2048` keeps prefill steps above 2048 tokens out of HPU graphs.
- `PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128` is kept from the configuration this recipe was taken from; it was not measured
  on its own.

### Buckets

`VLLM_BUCKETING_FROM_FILE` loads `ports/glm-5-mxfp4/serving/tp8-seqs4-128k.txt`: the default exponential buckets for
this geometry plus a decode row at batch size 3, so three running requests do not pay for a batch of four.

## Required fixes

The `v0.30.0` profile plus two opt-in fixes, applied by the image build and again (as a check) at launch.

| Fix | Why this model needs it |
| --- | --- |
| [`structured-output-rows`](../fixes/structured-output-rows.md), [`so-row-owners`](../fixes/so-row-owners.md) | tool_choice `required`, named tools and JSON schemas mask the right rows |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids are read back corrupted otherwise |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | one undecodable logprob id would stop the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with a `seed` would segfault a tensor-parallel worker |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a failed rank stops the engine instead of serving garbage |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) | opt-in: `VLLM_LONG_PREFILL_SHARE` |
| [`glm47-streaming-text-markers`](../fixes/glm47-streaming-text-markers.md) | opt-in: streamed `glm47` tool calls and reasoning |

The port adds 14 patches of its own (in the image). Ten are generic and listed in the port README as candidates for
`fixes/`: fused MXFP4 loading for models other than gpt-oss, the FusedSDPA softmax mode, and the speculative-decoding
chain (MTP output tuple, drafting with chunked prefill, verify-row block lists, the MTP residual shape, sampling at
temperature above 0, logprobs, penalties and structured output under speculation).

## Verified

On the image built by `ports/glm-5-mxfp4/install.sh` from the stock `releases/v0.30.0` image (fix profile, 14 port
patches and the `glm` kernel build all applied), eight Gaudi 2 cards, served with the Launch block above (plus `--host`,
`--port`, `--served-model-name`), checked with `ports/glm-5-mxfp4/tests/hpu/run_gates.sh`:

| Gate | Result |
| --- | --- |
| Needles: 3 codes at 10 / 50 / 90% depth, two fillers, 16K to 126K tokens | 30/30 (10 prompts x 3 codes) |
| GSM8K first 50, reasoning low, greedy (gate 47) | 48/50 |
| Tools: `required`, named, `auto`, `none` and a JSON schema, streaming and not, greedy and sampled | 32/32 |
| MTP acceptance at the server's default sampling (gate 0.70) | 0.858 over 1,651 drafts |
| Logprobs: 40 sampled requests with `top_logprobs` 5 under MTP | 40/40 complete and finite |
| Stress: 4 decodes while 64K and 100K-token prompts prefill | 4/4 finished, needles 3/3 in both, largest decode gap 0.9 s |
| Greedy: 16 prompts x 192 tokens | recorded |

The KV pool is 312,960 tokens (2.39 requests at the full 131,072-token window). The checkpoint load took 25 minutes
from shared network storage.

## Performance

One server, the Launch block above, `run_gates.sh` speed and TTFT gates: greedy, 256 output tokens, median of 3.

| Test | Result | Measured as |
| --- | --- | --- |
| Decode at 1 stream, ~2K-token prompt | 67.3 tokens/s | one server |
| Decode at 4 streams, ~2K-token prompt | 33.7 tokens/s per stream, 112.3 aggregate | one server |
| TTFT, uncached 2K / 8K-token prompt | 0.49 / 1.65 s | one server |

What each switch is worth was measured on [GLM-5.3](glm-5-3-mxfp4.md#performance) (same port, same layout, same
pod, each switch flipped with a server restart): the three decode levers add 11% at 1 stream and 13% aggregate at 4,
and the prefill kernel cuts uncached TTFT by 19% at 2K tokens and 10% at 8K.

## Known issues

- A prompt made of one word repeated tens of thousands of times makes the model loop under any accurate attention.
- Speculative decoding turns async scheduling off.
- A sampled request's `seed` is not honoured while MTP is on (the speculative rejection sampler draws without it).

## What not to do

- Do not serve without `--enable-expert-parallel`: the stock loader does not split a fused MXFP4 expert, and the load
  fails.
- Do not drop `VLLM_HPU_FSDPA_SOFTMAX_MODE=None`: with the stock `fast` softmax, needles fail from 32K tokens.
- Do not raise `--gpu-memory-utilization` above 0.86: the DSA buffers are allocated after profiling.
- Do not set `GLM_MOE_PLACEMENT` to the GLM-5.3 file: it is fitted to GLM-5.3 routing (the loader accepts any
  permutation, so a wrong file is slower, not wrong).
- Do not serve AMD's checkpoint directly: vllm-gaudi's MXFP4 path expects the fused expert names, and unknown names
  are skipped without an error (the experts stay zero).
