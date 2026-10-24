# DeepSeek-V4.1-Flash-1017

DeepSeek's 40-layer mixture-of-experts model (384 routed experts, top-6) with compressed sparse attention,
hyper-connections and two hashed Engram tables, served text only at 128K context on one 8-card Gaudi 2 node.
vllm-gaudi has no path for it, so it runs from the [DeepSeek-V4.1-Flash port](../ports/deepseek-v4-1-flash.md),
built on top of the stock `releases/v0.30.0` image, from a lossless re-layout of the public checkpoint.

| | |
| --- | --- |
| Model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (public, MIT), converted by the port |
| Hardware | 8x Gaudi 2 (attention TP 8, experts EP 8), one node |
| Image | port image built from vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | routed experts MXFP4 (DeepSeek's own bytes), everything else BF16 (FP8 linears dequantized exactly); 517.68 GB, 61 GiB per card |
| Context | 131,072 served (1,048,576 native); KV pool 294,912 tokens |
| Concurrency | `--max-num-seqs 8` |
| Parsers | reasoning, tools and tokenizer `deepseek_v41` (vLLM v0.30.0) |
| Recipe | [`recipes/deepseek-ai/DeepSeek-V4.1-Flash-1017/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/deepseek-ai/DeepSeek-V4.1-Flash-1017/recipe.yaml) |

## Convert the checkpoint

Download `deepseek-ai/DeepSeek-V4.1-Flash-1017` (510 GB) and convert it once, on any machine with Python and torch
(CPU is enough, about 520 GB of free disk):

```bash
python3 ports/deepseek-v4-1-flash/convert/convert_to_hpu_mxfp4.py --src <download> --dst <converted> --part all
python3 ports/deepseek-v4-1-flash/convert/validate_conversion.py --src <download> --dst <converted> --part all
```

Nothing is requantized: expert bytes are copied with gate and up rows interleaved, FP8 linears become bf16 exactly,
and the Engram tables are split into hash buckets byte for byte. The validator decodes both sides with its own code
and must print `VALIDATE_OK`. The original `deepseek-ai/DeepSeek-V4.1-Flash` (also MIT) converts with the same
commands. Put the result in `$DSV41_CHECKPOINT`; the launch refuses any other layout.

## Build the image

From a checkout of this repository, on top of a stock `releases/v0.30.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/deepseek-v4-1-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-deepseek-v4-1-flash-image> .
```

`install.sh` checks the vllm-gaudi and vLLM commits, installs the `dsv41_hpu` plugin, patches
`vllm_gaudi/__init__.py` to register the kernel library in every process, builds the MXFP4 decode kernels with the
port's Sinkhorn kernel, and runs the image gate. It stops on any other base. The image carries this repository at
`/opt/gaudi-cookbook`.

## Launch

Inside the port image, with `COOKBOOK=/opt/gaudi-cookbook`:

```bash
export DSV41_CHECKPOINT=/models/DeepSeek-V4.1-Flash-1017-hpu
export HF_HUB_OFFLINE=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_RECIPE_CACHE_CONFIG=/data/dsv41-recipe-cache,False,32768
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=65536
export DSV41_LAYER_GRAPHS=1
export DSV41_LAYER_GROUP=10
export DSV41_DET_TOPK=1
export DSV41_TOPK_SELECT=thresh
export DSV41_PREFILL_ATTN=1
export DSV41_INDEX_BF16_DOT=1
export DSV41_REPLAY_KEEP=140
export DSV41_PREFILL_HEADROOM_GIB=6
export DSV41_MHC_SLICED=1
export DSV41_HC_SINKHORN=kernel
export DSV41_MHC_TPC_DOT=1
export DSV41_FQ_FASTBITS=1
export DSV41_LEAN_OPS=1
export DSV41_DECODE_HOIST=1
export DSV41_ROPE_FUSED=1
export DSV41_INDEX_DECODE=1
export DSV41_WOA_LINEAR=1
export DSV41_MXK=1
export DSV41_WARMUP=1
export DSV41_WARMUP_MAX_SEQS=8
export DSV41_WARMUP_MAX_S=2400

python3 "$COOKBOOK/ports/deepseek-v4-1-flash/serving/check_checkpoint.py" "$DSV41_CHECKPOINT" || exit 1
mkdir -p "${PT_HPU_RECIPE_CACHE_CONFIG%%,*}"
python3 "$COOKBOOK/ports/deepseek-v4-1-flash/serving/warmup.py" > /tmp/dsv41-warmup.log 2>&1 &

# Fixes: the v0.30.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 || exit 1

vllm serve "$DSV41_CHECKPOINT" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --num-gpu-blocks-override 2304 \
  --gpu-memory-utilization 0.90 \
  --block-size 128 \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enforce-eager \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --exclude-tools-when-tool-choice-none \
  --language-model-only
```

The setup lines check the checkpoint (fail closed), create the recipe-cache directory and start
`serving/warmup.py` in the background. The warm-up waits for the server, compiles every prefill shape (11 query
buckets x 5 block-table widths) and decode shape (5 batch sizes x 5 context depths), and writes
`/tmp/dsv41-warmup.done` when it finishes or after `DSV41_WARMUP_MAX_S`. Point the readiness probe at that file:
under `--enforce-eager` vllm-gaudi's own warm-up compiles nothing, so a request on a cold shape compiles it inline.

The `DSV41_*` switches are documented in [`ports/deepseek-v4-1-flash/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/deepseek-v4-1-flash/README.md#switches).
Each is off by default in the plugin; the recipe turns on what the gates below were run with. `DSV41_MXK=0` turns
off both custom kernels (the MXFP4 decode kernels and the Sinkhorn kernel) and runs the stock ops.

### Buckets

No bucket file: prefill shapes follow vllm-gaudi's default exponential query buckets, and the plugin sizes block
tables by the batch's real context (64 blocks, doubling). The warm-up covers both.

## Required fixes

The v0.30.0 profile, applied at container start. The plugin depends on four of them and refuses to start without
them: `prompt-order` (a prefix-cache reader can run before its writer in the same step), `resumed-prefill-logits`
(a request resumed after preemption repeats its output), `gather-logprobs-int32` and `prompt-logprobs` (prompt
logits in row chunks; a 4096-token chunk runs out of memory otherwise). The row fixes (`sampling-rows-refresh`,
`structured-output-rows`, `so-row-owners`, `bad-words-index`, `logitsprocs-rows`) matter for structured output and
tool calls under load: long prompts chunked next to grammar requests otherwise get stray tokens or grammar
rejections. `logprobs-decode-guard` and `rank-exception-fatal` keep a bad request or a failed rank from leaving a
broken server up.

The port adds one patch of its own (in the image): the kernel-library hook in `vllm_gaudi/__init__.py`.

## Verified

Run exactly as written: the stock `releases/v0.30.0` image (`058f107`, vLLM `ced6857a`) on one 8-card Gaudi 2 node,
this repository copied to `/opt/gaudi-cookbook`, `install.sh` run as the Dockerfile runs it, then the Launch block
above with the converted `-1017` checkpoint. The gates are `tests/hpu/run_gates.sh`, after the warm-up, on an
otherwise idle server.

| Check | Result |
| --- | --- |
| `install.sh` | commits match; plugin installed; kernel-env patch applied; kernel build `dsv41-v8` (library gate and op gate PASS); image gate 6/6 PASS |
| Sinkhorn kernel | machine code (code sha256 of the TPC ELF) identical to the build the write-up's numbers come from |
| Conversion | re-run as published on one MoE layer and all dense tensors: sha256 equal to the served checkpoint for 1,266 of 1,266 tensors; validator 0 mismatches |
| Startup | checkpoint check OK (95 shards, 1,530 tensors); fixes applied; weights loaded in 45 min from network storage; warm-up 81 shapes, 0 errors, 910 s |
| Kernels | MXFP4 decode kernels ON and the Sinkhorn TPC kernel on all 8 ranks (`KERNEL` gate) |
| Perplexity vs the fp32 CPU oracle, 1,536 tokens | 1.6180 vs 1.6239, no missing token (`PPL` PASS) |
| Decode vs prefill, teacher-forced, 4 concurrent prompts | mean \|Δlogprob\| 0.016, 10 of 756 tokens off the prefill argmax, largest gap 0.69 nats (`DECODE-TF` PASS) |
| Needles at half depth | 13.4K and 45.3K tokens: 2/2 |
| Isolation: 24 requests x 2 over a shared 4,000-word cached prefix, 20% aborted streams | own code 48/48, 0 foreign codes or topics, probes identical before and after |
| Structured output and tools, 2 rounds x 6 groups, 12 concurrent | json, json with thinking, `tool_choice` required and auto, long plain prompts: 60/60 |

The same gates with `DSV41_MXK=0` (both custom kernels off, stock ops; same pod, server restarted, everything else
as above) also all pass:

| Check, kernels off | Result |
| --- | --- |
| Kernels | no rank reports the MXFP4 decode kernels or the Sinkhorn kernel (`KERNEL` gate, expected off) |
| Perplexity | 1.6171 vs 1.6239 (kernels on: 1.6180) |
| Decode vs prefill | mean \|Δlogprob\| 0.025, 13 of 738 tokens off the prefill argmax, largest gap 1.59 nats |
| Needles | 13.5K and 45.3K tokens: 2/2 |
| Isolation | own code 48/48, 0 foreign hits, probes identical |
| Structured output and tools | 60/60 |

The kernels on their own, one card, synthetic weights (`kernels/mxfp4-moe/tests/run_parity.sh --variant dsv41` on
this build): `PARITY PASS` and `SIG_DEVICE PASS`; relative error against fp32 1.6e-3 to 1.7e-3 for the kernels,
3.2e-3 to 2.5e-2 for the stock op, at 1 to 16 tokens.

## Performance

Decode, greedy, 300 tokens per stream, short chat prompts, every shape warmed, one idle server (measured as: one
test pod, kernels on):

| Streams | Per stream | Aggregate | First token |
| --- | --- | --- | --- |
| 1 | 37.6 tok/s | 37.6 tok/s | 0.68 s |
| 4 | 27.0 tok/s | 108.1 tok/s | 0.64 to 2.55 s |

Kernels off vs on (measured as: same-pod A/B, the server restarted with `DSV41_MXK=0`, same warm-up and gates):

| Streams | Kernels off | Kernels on | Gain |
| --- | --- | --- | --- |
| 1, per stream | 20.7 tok/s | 37.6 tok/s | 1.81x |
| 4, aggregate | 52.0 tok/s | 108.1 tok/s | 2.08x |

Startup in the same pod (measured as: one run each, network storage): weights loaded in 45 minutes from a cold page
cache and in 19 minutes on the restart; the warm-up then took 910 s (kernels on) and 1,045 s (kernels off) for 81
shapes.

Prefill, decoder replay and the decode-step changes are measured one by one in the
[port write-up](../ports/deepseek-v4-1-flash.md#performance).

## Known issues

- Cold start is long. Loading 61 GiB per card from network storage took 45 minutes in the verification run (the
  Engram tables are most of it), then the warm-up compiles the prefill and decode shapes. Fast local storage and a
  persistent `PT_HPU_RECIPE_CACHE_CONFIG` directory shorten both.
- A prompt whose shape the warm-up has not compiled yet (for example while `/tmp/dsv41-warmup.done` is missing) waits
  for the compile, from seconds to minutes.
- Long cold prompts hold up the replica: a 120K-token prompt takes about 44 s to read in (cold TTFT 4.9 s at 16K,
  21.6 s at 64K, 43.5 s at 120K; different conditions, see the [port write-up](../ports/deepseek-v4-1-flash.md#performance)),
  and running decodes slow down while it does.
- `prompt_logprobs` requests run the full stack (decoder replay off), so they are slower than a plain prefill of the
  same prompt.
- Text only. The checkpoint's vision configuration is ignored (`--language-model-only`).

## What not to do

- Do not turn async scheduling on: Engram hashing needs every sampled token on the host before the next step.
- Do not serve the original FP8 checkpoint, or any directory `check_checkpoint.py` rejects. The model name matches,
  and vLLM would read FP8 bytes through the wrong path.
- Do not drop `--enforce-eager`. Decode already runs in the port's own HPU graphs (one per 10 layers); a whole-model
  graph never finishes compiling.
- Do not skip the fix step. The plugin refuses to start without its four required fixes, and without the row fixes
  structured output and tool calls break under load.
- Do not raise `--num-gpu-blocks-override` past 2304 without re-running the 128K needle and isolation gates: device
  memory is tight at 128K prompts.
- Do not set `DSV41_HC_SINKHORN=kernel` with `DSV41_MXK=0` and expect the kernel: it needs the kernel library that
  `DSV41_MXK=1` registers, and falls back to the torch loop otherwise.
