# MiniMax-M3 with FP8 experts on Gaudi 2

MiniMax-M3 is a 427B-parameter multimodal MoE (23B active): 60 layers, 128 routed experts with top-4 sigmoid routing
and a score correction bias, one shared expert, and gpt-oss's SwiGLU-OAI activation (clamp to 7, alpha 1.702). The
port serves it text-only on one 8-card Gaudi 2 node at 128K context, from AMD's public MXFP4 checkpoint with the
routed experts re-encoded, exactly, to per-channel FP8.

| | |
| --- | --- |
| Checkpoint | [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4), converted by the port to FP8 experts (411 GiB) |
| Hardware | 8x Gaudi 2, TP 8 with expert parallelism (16 experts per card) |
| Image | stock vllm-gaudi `v0.26.0` plus [`ports/minimax-m3`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/minimax-m3) |
| Context | 131,072 served (1,048,576 native) |
| Recipe | [MiniMax-M3 (FP8 experts)](../recipes/minimax-m3.md) |
| Licence | MiniMax Community License (non-commercial grant; commercial use needs "Built with MiniMax M3" and a notice to MiniMax) |

## Why it needed a port

vllm-gaudi v0.26.0 ships the HPU model for MiniMax-M3, its expert-gather decode path and a pure-Python tool parser.
The only public checkpoints that fit one node are quantized, and none of them runs on that model as released:

- **The BF16 release is 856 GB**, two nodes' worth of HBM.
- **AMD's MXFP4 checkpoint is AMD Quark** (`quant_method: quark`, OCP MXFP4 weights). `quark` is not in the HPU
  platform's list of quantization methods, and vllm-gaudi's native MXFP4 MoE path is written for gpt-oss's
  checkpoint layout.
- **SwiGLU-OAI is not an activation the Habana FP8 MoE op takes.** The op's activation string covers silu and gelu.
  The stock compressed-tensors FP8 MoE method passes the layer's activation through, so an FP8 M3 checkpoint would run
  the wrong activation.

## What was built

### An exact MXFP4 to FP8 re-encoding

An MXFP4 element is an e2m1 value (one mantissa bit) times a power-of-two scale shared by 32 elements. Within one
output channel the scales of M3's experts differ by at most a few binades, so one power-of-two scale per channel puts
every element on an e4m3 normal: the FP8 checkpoint holds exactly the MXFP4 values, in a format the cards' matrix
engine reads directly. Three Gaudi 2 details shape the encoder: the top representable binade is 2^7 (the maximum is
240, not 448), the loader may halve FP8 weights so the lowest exponent is kept one binade above e4m3's smallest
normal, and -0 is written as +0. The shared experts, also MXFP4 in the source, become BF16, again exactly.

The converter re-decodes every chunk in float64 and stops on any difference; `validate.py` re-reads the output with
its own reader and decoders. On the full checkpoint every expert scale is a power of two and every sampled expert
equals its MXFP4 source element by element; everything that is not an expert is byte-identical. The output is a
compressed-tensors checkpoint (FP8 per-channel weights, FP8 dynamic per-token activations) that the stock loader
reads, and `--enable-ep-weight-filter` lets each card read only its own 16 experts.

### SwiGLU-OAI on the fused FP8 kernel

`patches/swigluoai_fp8_moe.py` sends SwiGLU-OAI layers of the compressed-tensors FP8 MoE method to
`torch.ops.hpu.mixture_of_experts.bias_fp8_fused_weights`, the one Habana FP8 MoE overload with gpt-oss's activation
inside the kernel (its alpha and limit defaults are M3's values). Four details matter:

- **Row order.** The kernel reads gate and up rows interleaved (g0, u0, g1, u1, ...). The checkpoint keeps them
  concatenated, so they are interleaved once at load, on the uint8 view. Concatenated rows produce fluent garbage,
  not an error, which is why the CPU test checks that the interleaved reading matches the reference and the
  concatenated reading does not.
- **Activations** are quantized to FP8 per token on the fly.
- **The down projection's input** uses a fixed scale of 56/240. SwiGLU-OAI's clamp bounds that input by
  7 x (7 + 1) = 56, so it never saturates.
- **Biases.** The overload with this activation only exists with biases, so zero biases are registered.

`VLLM_M3_FP8_MOE=dequant` keeps a reference path: the experts decoded to bf16 and run in PyTorch with bf16
activations, which is numerically the MXFP4 checkpoint in bf16.

### Streaming reasoning

M3's chat template injects a system block that describes `<mm:think>` and `</mm:think>` in prose, and both markers
are single tokens. vLLM's `minimax_m3` reasoning parser checks whether the prompt already ended reasoning by scanning
the whole prompt, finds that closer, and answers yes. Streaming then emits the model's reasoning, markers included, as
`content`, and structured outputs believe reasoning is over. Non-streaming requests are unaffected, which hides the
bug from API tests. `patches/reasoning_prompt_scope.py` starts the scan at the last role token, so only the turn
being generated counts. The bug is in vLLM's parser and is present on v0.26.0, v0.30.0 and vLLM main.

## History: three expert paths

The same node shape ran three expert paths before FP8. Each row is the path production ran at the time, measured on
its own configuration, so the rows are **different conditions** (named in the last column); use them for the order
of magnitude, not as an A/B.

| Expert path | 1 stream | 8 streams, per stream | Conditions |
| --- | --- | --- | --- |
| MXFP4 unpacked to bf16 in PyTorch on every step | 11.5 tok/s | 1.6 tok/s | v0.26.0, TP 8, lazy mode, warm, 2 repetitions |
| MXFP4 on Habana's fused MXFP4 kernel (checkpoint re-laid out for it) | 22.1 tok/s | 7.5 tok/s | v0.26.0, TP 8 + EP, compile mode, 2K prompt, same session as the FP8 row |
| FP8 experts (this port) | 24.7 tok/s | 23.8 tok/s | v0.26.0, TP 8 + EP, compile mode, 2K prompt |

The MXFP4 kernel unpacks every expert it touches to bf16 inside each step: about 0.2 ms per expert per card, so the
cost grows with the number of distinct experts a batch hits. That is why per-stream speed fell from 22 to about 7
tok/s at 8 streams. The FP8 kernel reads FP8 directly and is 4 to 7 times cheaper per layer and nearly flat in the
number of tokens (one-card micro-benchmark on real layer-30 weights, 16 experts as under EP 8, per-call one card):

| Tokens in the step | MXFP4 kernel | FP8 kernel | FP8 kernel, `VLLM_MOE_CHUNK=256` |
| --- | --- | --- | --- |
| 1 | 0.21 ms | 0.05 ms | 0.05 ms |
| 8 | 1.47 ms | 0.39 ms | 0.22 ms |
| 64 | 3.89 ms | 0.57 ms | 0.58 ms |
| 8192 (a prefill chunk) | 12.2 ms | 7.9 ms | 4.1 ms |

## Correctness

FP8 experts against the fused MXFP4 path, two servers with the same image and command apart from the expert path
(different conditions: two nodes, same session):

| Check | Result |
| --- | --- |
| Expert weights | equal to the MXFP4 source element by element (whole checkpoint scanned for scales, 452 experts compared) |
| Expert output, one layer | 4.1% relative change from FP8 activations (bf16 rounding alone: 0.4%; AMD's own MXFP4-activation format: 19 to 23%) |
| Next-token agreement, teacher-forced, 6,931 positions | same top token 97.8%; mean loss 0.1754 vs 0.1739 nats |
| Mental arithmetic, 300 synthetic problems | 245 vs 246 correct |
| Greedy golden prompts | same answers 5/5 |
| Greedy loops | 0/10 on both |
| Needles, 3 per prompt | all found at 16K, 32K and 64K on both; FP8 also at 100K and 125K |
| `prompt_logprobs` and `logprobs` | pass on both |

## Performance

The recipe as published (port image from the stock v0.26.0 image, the v0.26.0 fix profile, the recipe command) passes
all serving gates on one node and decodes at 25.0 tok/s at one stream and 21.9 per stream at 8 streams with 1.6K
prompts (streams started together, first to last streamed token); see the [recipe](../recipes/minimax-m3.md#performance).

The longer sweep below is from the earlier build that the FP8 path was developed on (different conditions: the same
patches and command, the row and logprobs fixes instead of the whole profile), per-stream decode while all streams
decode together, temperature 0, every shape warmed first:

| Context | 1 stream | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| 2K | 24.7 | 23.7 | 23.4 | 23.8 | 23.2 | 20.6 |
| 16K | 25.6 | 23.7 | 23.5 | 23.3 | 22.4 | 20.1 |
| 32K | 25.6 | 23.6 | 23.6 | 23.0 | 20.9 | 14.5 |

First token for one cold 2K, 16K and 32K prompt: 0.26 s, 2.0 s and 5.5 s. The KV pool at
`--gpu-memory-utilization 0.92` holds about 1.25M tokens. Cold start is about 10 minutes from idle shared storage with
`--enable-ep-weight-filter` (38 minutes measured on a busy one) and about 45 minutes without it.

## What it taught about Gaudi

- **A lossless format change can be the fastest kernel change.** No new kernel was written: the same numbers in a
  format the matrix engine reads natively removed the per-step unpack.
- **Layout bugs in fused MoE kernels are silent.** The gpt-swiglu overloads read gate and up interleaved; the silu
  overloads read them concatenated. The wrong one produces fluent text.
- **Compressed-tensors on Gaudi 2 halves FP8 weights at load by default.** Power-of-two scales and
  `VLLM_SCALE_ADJUSTMENT=false` keep the re-encoding exact.
- **Weight load scales with what each card reads.** With expert parallelism and per-expert tensor names,
  `--enable-ep-weight-filter` cut a 45-minute cold load to about 10.

## Availability

The port is in [`ports/minimax-m3`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/minimax-m3): the
converter, the two patches, the CPU tests and the serving gates. The recipe is verified as written: the image built
by `install.sh` from the stock v0.26.0 image passes every gate in `run_gates.sh`, and the converter reproduces the
served checkpoint byte for byte. Not covered: images (the HPU vision path builds the tower as MXFP4 on the AMD checkpoint; the FP8
checkpoint leaves it in BF16 but images were not validated) and speculative decoding (no checkpoint publishes M3's
MTP weights).
