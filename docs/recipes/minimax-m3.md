# MiniMax-M3 (FP8 experts)

MiniMax's 427B-parameter MoE (23B active), served text-only on one 8-card Gaudi 2 node at 128K context. The
checkpoint is AMD's public MXFP4 release with the routed experts re-encoded, exactly, to per-channel FP8 by the
[MiniMax-M3 port](../ports/minimax-m3.md), which also runs M3's SwiGLU-OAI experts on Habana's fused FP8 kernel.

| | |
| --- | --- |
| Model | [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) (quantized from [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3)), converted to FP8 experts |
| Hardware | 8x Gaudi 2 (TP 8 with expert parallelism) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1, plus [`ports/minimax-m3`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/minimax-m3) |
| Precision | routed experts FP8 per channel (the MXFP4 values, exactly), FP8 activations per token in the experts; everything else BF16. 411 GiB |
| Context | 131,072 served (1,048,576 native) |
| Concurrency | `--max-num-seqs 32` |
| Parsers | reasoning `minimax_m3` (patched by the port), tools `minimax_m3_py` |
| Licence | MiniMax Community License: non-commercial grant; commercial use needs "Built with MiniMax M3" and a notice to MiniMax |
| Recipe | [`recipes/amd/MiniMax-M3-MXFP4/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/amd/MiniMax-M3-MXFP4/recipe.yaml) |

## Convert the checkpoint

Download `amd/MiniMax-M3-MXFP4` and convert it once, on any machine with Python and torch (CPU is enough). The
commands, with the validation steps, are in [`ports/minimax-m3/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/minimax-m3/README.md#1-convert-the-checkpoint).
The result goes in `$CHECKPOINT_DIR`.

## Build the image

From a checkout of this repository, on a stock vllm-gaudi v0.26.0 image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/minimax-m3/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <your-minimax-m3-image> .
```

## Launch

Inside the port image (`$COOKBOOK` is `/opt/gaudi-cookbook` there).

```bash
export CHECKPOINT_DIR=/models/MiniMax-M3-FP8
export PT_HPU_LAZY_MODE=0
export LOG_LEVEL_PT_BRIDGE=4
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_SCALE_ADJUSTMENT=false
export VLLM_M3_FP8_MOE=native
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=2048
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000

# Fixes: the v0.26.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 || exit 1

vllm serve "$CHECKPOINT_DIR" \
  --served-model-name amd/MiniMax-M3-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.92 \
  --block-size 128 \
  --enable-prompt-tokens-details \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3_py \
  --trust-remote-code
```

- `VLLM_M3_FP8_MOE=native` runs the experts on `bias_fp8_fused_weights`, Habana's FP8 MoE kernel with SwiGLU-OAI
  inside. `dequant` is the port's reference path (bf16 experts in PyTorch), several times slower at decode.
- `VLLM_SCALE_ADJUSTMENT=false`: on Gaudi 2 the compressed-tensors loader halves FP8 weights by default. The
  converted scales are exact powers of two, so no adjustment is wanted; the checkpoint's config turns it off in the
  process that parses it, this variable in the worker processes.
- `PT_HPU_LAZY_MODE=0` (torch.compile mode) is how this expert path was validated; startup is also faster than in
  lazy mode. `LOG_LEVEL_PT_BRIDGE=4` silences the bridge's per-op log lines in that mode.
- `VLLM_MOE_CHUNK=256` and `VLLM_MOE_TOKEN_BOUNDARY=2048` are the stock FP8 MoE op's chunk hints; the port passes
  them to the fused kernel (1.6x to 2.2x faster per call at 2 to 8 tokens and 1.9x at an 8K prefill chunk, per-call
  one card).
- `--enable-ep-weight-filter` makes each card read only its 16 of the 128 experts: the cold load drops from about 45
  to about 10 minutes from idle shared storage (38 minutes measured on a busy one).
- `--limit-mm-per-prompt '{"image": 0, "video": 0}'` serves text only; the vision tower is not built. Keep the
  served checkpoint the full vision-language one: the weight names and the quantization config expect it.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`: with warmup skipped, the first long prefills compile inline and can run
  past the 300 s default, which stops the engine.
- Graph recipes can be cached with `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,131072`; use one directory per image and
  configuration.

### Buckets

No bucket file and no bucket variables: the image's default bucketing.

## Required fixes

The v0.26.0 profile, applied at container start on top of the port's patches. The fixes this configuration was
checked with are listed in the recipe: the row and logprobs fixes (`prompt-order`, `sampling-rows-refresh`,
`structured-output-rows`, `bad-words-index`, `logitsprocs-rows`, `async-discard-counter`, `resumed-prefill-logits`,
`gather-logprobs-int32`, `prompt-logprobs`, `logprobs-decode-guard`) and `rank-exception-fatal`. M3 routes with
grouped top-k, so `laguna-moe-sigmoid-routing` does not change its routing.

The port adds two patches of its own (in the image): the SwiGLU-OAI FP8 expert path and a fix to vLLM's
`minimax_m3` reasoning parser, which otherwise streams the reasoning as `content` with a raw `<mm:think>` marker
(the chat template quotes both markers in the prompt).

## Verified

Run exactly as written on one 8-card Gaudi 2 node: the port image built by `install.sh` from the stock v0.26.0
image, the v0.26.0 fix profile at start, the Launch block above, then `ports/minimax-m3/tests/hpu/run_gates.sh`:

| Gate | Result |
| --- | --- |
| install | port patches and the v0.26.0 fix profile applied; the server log shows `mode=native` on every card |
| golden | 5/5: 17 x 23 = 391, an `is_prime` function, Canberra, 3 hours 25 minutes, `get_weather({"city": "Phoenix"})` |
| loops | 0/10 greedy answers ran to the token limit |
| needles | 3/3 codes at 16,450, 32,021, 64,007, 100,016 and 125,013 prompt tokens |
| logprobs | `prompt_logprobs` over 11,901 tokens all aligned and in the vocabulary; 16 concurrent `top_logprobs` requests, 0 errors |
| reasoning | streamed and non-streamed: reasoning in `reasoning`, no `<mm:think>` in `content`; streamed tool call with name and arguments |
| speed | 24.4 tok/s at one stream and 18.9 per stream at 8 (256 tokens after a 1.6K prompt, prefill included) |

The KV pool was 1,247,232 tokens (9.5 concurrent full-length requests). The converter was re-run as published: on
one MoE layer it reproduced all 768 tensors of the served checkpoint byte for byte, the shared experts of three
layers 9/9, and `validate.py` passed on the full converted checkpoint (inventory, and 18 sampled experts
element-exact).

Against the fused-MXFP4 configuration of the same model, an earlier build with the same two patches and the same
command (different conditions: the row and logprobs fixes only, not the whole profile) measured:

| Check | FP8 experts vs fused MXFP4 |
| --- | --- |
| Expert weights | equal to the MXFP4 source element by element |
| Next-token agreement, teacher-forced, 6,931 positions | same top token 97.8% of the time; mean loss 0.1754 vs 0.1739 nats |
| Mental arithmetic, 300 synthetic problems | 245 vs 246 correct |
| Greedy golden prompts | same answers 5/5 |

The reasoning-parser fix was also checked on that earlier build: 7 streamed cases (single and multi-turn, tool
calls, thinking enabled and disabled) put the reasoning in `reasoning` with no marker in `content`, 7/7, where the
unpatched parser failed 5.

## Performance

Decode tok/s per stream (first to last streamed token) with all streams started together, temperature 0, second
of two rounds, on the verified configuration above (one node, TP 8 + EP):

| Prompt | 1 stream | 4 streams | 8 streams | 16 streams |
| --- | --- | --- | --- | --- |
| 1,596 tokens | 25.0 | 22.2 | 21.9 | 21.5 |
| 25,595 tokens | 25.4 | 17.7 | 10.9 | 6.5 |

Time to first token at one stream: 0.22 s for 1,596 tokens and 4.2 s for 25,595. With long prompts started together,
each stream's decode overlaps the other streams' chunked prefill, which is what the lower 25K rows show.

The earlier build measured decode with the streams already decoding (different conditions), per stream:

| Context | 1 stream | 4 streams | 8 streams | 16 streams | 32 streams |
| --- | --- | --- | --- | --- | --- |
| 2K | 24.7 | 23.4 | 23.8 | 23.2 | 20.6 |
| 32K | 25.6 | 23.6 | 23.0 | 20.9 | 14.5 |

The same command with the fused MXFP4 kernel instead of FP8 experts ran 22.1 tok/s at one stream and 7.5 per stream
at 8 there (different conditions: a second server on another node in the same session).

## Known issues

- At temperature 0.6 and below, hard prompts can keep reasoning until the token budget runs out (`finish_reason`
  `length` with empty `content`). Serve at the model's default temperature 1.0 (top_p 0.95); penalties do not help.
- With thinking on and a small `max_tokens`, the budget can go entirely to reasoning. Raise the budget.
- Images are not validated. The AMD checkpoint's quantization exclude list does not match vLLM's vision module
  names, so on the MXFP4 path the vision tower is built quantized and fails to load; the converted checkpoint leaves
  the tower in BF16, but image serving was not tested.
- No speculative decoding: no published checkpoint contains M3's MTP weights.

## What not to do

- Do not serve the converted checkpoint with `VLLM_SCALE_ADJUSTMENT` on, or edit its `quantization_config`: the
  FP8 weights would be halved or loaded through a path that runs the wrong activation.
- Do not force the text-only architecture with `--hf-overrides` (`MiniMaxM3SparseForCausalLM`): the weight names and
  the quantization config are written for the vision-language wrapper. Gate images with `--limit-mm-per-prompt`.
- Do not use `--tool-call-parser minimax_m3`; it needs a Rust extension that the image does not have and fails on
  the first request. Use `minimax_m3_py`.
- Do not drop `--enable-ep-weight-filter` on shared storage unless a 45-minute cold load is acceptable.
