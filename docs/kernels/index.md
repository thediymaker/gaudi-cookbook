# Kernels

Custom TPC kernels for Gaudi 2. A kernel is worth maintaining when the stock path has a structural cost that no
flag or batch setting removes. Each one ships with its sources, a build script that runs inside a stock vllm-gaudi
image, a vLLM integration hook that is off by default, and tests that compare it with the stock op.

Kernel sources live in [`kernels/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/kernels) in the
repository. Compiled libraries are never committed: build them inside the image you serve from, because the op
library links against that image's PyTorch build.

## Available kernels

| Kernel | Replaces | Helps when | Models | Gain |
| --- | --- | --- | --- | --- |
| [MXFP4 MoE decode](mxfp4-moe.md) | `mixture_of_experts.bias_mxfp4_fused_weights` for calls of at most 32 tokens | Decode of MXFP4 MoE models, where the routed experts dominate the step | `openai/gpt-oss-120b` (measured), `openai/gpt-oss-20b` (same shapes, not measured) | 3x to 10x per MoE call on one card; 1.1x to 3.2x decode tok/s end to end, largest at 4 to 16 requests in flight |

## When a custom kernel helps

- **The decode step is device-bound in one op family.** Profile a decode step first. With the stock MXFP4 op,
  gpt-oss-120b spends 41% of a batch-1 step and 75 to 82% of a batch 4 to 32 step in the MoE. A kernel cannot help
  a step that is bound by host launches (batch 1 on gpt-oss is mostly host time) or by attention.
- **The stock lowering does extra memory traffic.** The stock MXFP4 path unpacks every touched expert to bf16 in
  memory before a GEMM. A GEMV that reads the packed bytes once avoids a 4x larger write and read.
- **Token counts are small.** At decode sizes the matrix engine gets tiny GEMMs. Prefill-sized calls amortise the
  unpack and stay on the stock op.

## What every kernel here must do

- Run on a released vllm-gaudi image with the Gaudi software of [Getting started](../getting-started.md). Build
  tools come from the Habana base image; no SDK headers are copied into this repository.
- Be off by default behind one environment flag, and fall back to the stock op for any shape, dtype or library
  problem.
- Install through a script that checks every anchor in vllm-gaudi and exits non-zero before writing anything if
  one is missing.
- Ship a parity test against the stock op and an fp32 reference on one card, and a micro benchmark.
