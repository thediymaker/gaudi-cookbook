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
| [MXFP4 MoE decode](mxfp4-moe.md) | The stock MXFP4 MoE op for decode-sized calls (per-variant caps) | Decode of MXFP4 MoE models, where the routed experts dominate the step | Five build variants: `gptoss` (`openai/gpt-oss-120b` measured, `gpt-oss-20b` same shapes), `glm` (GLM-5.3, GLM-5.2, with a prefill unpack), `dsv41` (DeepSeek-V4.1-Flash), `muse` (Muse Spark Open), `kimi` (Kimi-K2.7). Non-gpt-oss models run it through their [port](../ports/index.md) | Per MoE call, one card: gpt-oss 3x to 10x, GLM-5.x 3.4x to 6.5x, DeepSeek-V4.1-Flash 4.0x to 7.2x, Muse Spark Open 4.7x to 5.7x, Kimi 3.3x to 5.7x. End to end: gpt-oss up to 3.2x decode tok/s; GLM-5.3 2.5x single stream (23.3 to 59.1 tok/s) |

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
