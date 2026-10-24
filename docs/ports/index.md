# Ports

Models that released vllm-gaudi images cannot serve, brought up on Intel Gaudi 2. Each port needed model code for an
architecture that vllm-gaudi has no path for, and usually kernels, memory layouts or loaders as well. All of them
serve production traffic except where the page says otherwise.

Each page explains what was hard, how it was solved, which correctness gates it passes, and what it measured. Every
number says how it was measured: a same-pod or same-node A/B, chained A/Bs, or different conditions (named).

| Model | What made it hard | Runs on | Headline |
| --- | --- | --- | --- |
| [Muse Spark Open](muse-spark-open.md) (816B MoE, MXFP4) | Latent MoE with a norm on every expert output, interleaved sliding and global attention, 1M-token windows | 8 cards, one node | Perplexity within 0.4% of the reference; needles pass at 256K, 512K and 1M; custom MXFP4 decode kernels |
| [GLM-5.3-Flash](glm-5-3-flash.md) (320B MoE) | No vLLM backend could run it when the port started: the model was written for vLLM, with KDA and fused mHC kernels on HPU | 8 cards, one node | Decode 22 to 24 tok/s at every context tested, 2K to 517K; decode step 32.5 to 24.5 ms, bit-identical |
| [DeepSeek-V4.1-Flash](deepseek-v4-1-flash.md) | A new attention and expert design with no vllm-gaudi model; lossless MXFP4 re-layout; decoder replay | 8 cards, one node | Decode 23.6 to 35.0 tok/s and prefill 1.35x to 1.55x faster (same-node A/Bs); DSpark drafting on HPU |
| [GLM-5.2 and GLM-5.3 in MXFP4](glm-5-mxfp4.md) (744B MoE) | Loader and router bugs on the native MXFP4 path, sparse DSA attention, a custom MXFP4 encoder | 8 cards, one node | Needles 3/3 from 16K to 128K, where the stock path failed from 32K; MXFP4 kernels 23.25 to 45.6 tok/s at one stream (same-pod A/B) |
| [Beam-1.0-501B-A23B](beam-1-0.md) | NVIDIA-only upstream support, a custom load format, 256K context | 8 cards, one node | Weight load 2177 to 356 s bit-identical; decode 20.6 to 30.5 tok/s (chained A/Bs on one pod) |
| [MiMo-V2.6-Flash](mimo-v2-6-flash.md) (309B MoE) | Sliding-window attention with sinks, asymmetric head sizes, 512K context | 8 cards, one node | 512K served; window-sized KV gives 5.33x the token capacity; needles 3/3 at 500K |
| [Kimi-K2.7-Code](kimi-k2-7.md) (1T MoE) | Too large for one node in FP8: an exact MXFP4 to FP8 expert conversion for two nodes, then a one-node MXFP4 build with the `kimi` kernels | 8 cards, one node (two-node FP8 documented) | One node with the kernels 44.5 tok/s at one stream vs 17.9 on the stock op (same pod) and 36.4 to 38.2 on two nodes in FP8 (different conditions) |
| [MiniMax-M3](minimax-m3.md) (427B MoE) | gpt-oss's clamped SwiGLU in a model the stock FP8 kernel cannot run; an exact MXFP4 to FP8 expert re-encoding | 8 cards, one node | FP8 experts 24.7 tok/s at one stream vs 22.1 on the fused MXFP4 kernel and 11.5 unpacked (different conditions) |
| [Inkling-Small](inkling-small.md) (276B MoE) | No RoPE: a learned relative-position logit per head and short convolutions with state, CUDA-only kernels upstream | 8 cards, one node | 256K tokens per request; 43.9 tok/s at one stream, 21.7 per stream at 16 (single run) |

## What the ports have in common

- **Gaudi software 1.24.1 and HPU graphs.** Most ports run in lazy mode; MiMo-V2.6-Flash runs in torch.compile mode
  with static shapes. Decode speed on large models comes from capturing whole decode steps; see the per-port pages
  for what had to change to make capture safe.
- **The MXFP4 MoE kernels.** Muse Spark Open, the GLM MXFP4 models, DeepSeek-V4.1-Flash and MiMo-V2.6-Pro use the
  cookbook's [MXFP4 MoE decode kernels](../kernels/mxfp4-moe.md) through model-specific integrations.
- **Platform findings.** Several ports ran into the same Gaudi behaviours; they are collected in
  [platform findings](../results/platform-findings.md).
- **Fixes upstream.** Bugs that are not model-specific went into the [fixes pack](../fixes/index.md) and upstream.

## Availability

Every port above is in this repository under [`ports/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports),
with a recipe in [Recipes](../recipes/index.md). A port builds from a stock vllm-gaudi image with its `Dockerfile`
(which runs its `install.sh` and fails closed if the image does not match), and ships CPU tests and device gates
(`tests/hpu/run_gates.sh`). The layout every port follows is in
[`ports/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/README.md). Muse Glimmer 30B has a
small port too ([`ports/muse-glimmer-30b`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/muse-glimmer-30b),
three model patches) without a write-up here; its recipe page explains it.

